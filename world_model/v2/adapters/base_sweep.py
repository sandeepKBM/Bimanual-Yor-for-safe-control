"""adapters/base_sweep.py -- MuJoCo side of the base-route planner (stack/planners/base_route.py).  SIMULATION ONLY.   (table-behind D1, 11-09)

The stack core cannot import physics; base_route.py calls this backend (duck-typed, resolved by base_route.default_backend()).

  BACKEND.world_default(task)            default world of a table-behind task (TaskDef.world_ref / world_kw)
  BACKEND.supports(world)                {"counter": dict(top, x, y), "counter2": dict(top, x, y)}, world frame (nero/base/scene.support_tops_behind when it
                                         exists; else MODEL: counter 1 mirrored about scene.mirror_centre())
  BACKEND.start_pose(world)              (x, y, yaw) of the base rotation centre at the start (MODEL: scene.base_geometry, SwerveBase._yaw0)
  BACKEND.levels()                       {level: (v m/s, a m/s^2, w rad/s, alpha rad/s^2)}   (nero/base/smoke.LEVELS, ASSUMED)
  BACKEND.hold_q(world)                  {side: q (7,)} claw hold pose of the carry (contents/capture_nyu.json, what compile._inputs uses)
  BACKEND.sweep(world, poses, q_hold, obj_off, clearance_min)
                                         swept clearance of a base route: -> dict(min_dist, worst, by_segment, n_poses, clearance_min, ...)

sweep() is a KINEMATIC copy of the two-table swerve scene (a CarryEpisode is built once per world and its qpos copied): for every pose of the base rotation
centre along the route the whole free-joint tree of the base (arms included, arm joints at q_hold) and every other free body (tray, mugs, crate, contents, lid)
is moved rigidly by the SE(2) displacement start -> pose (the object is welded to the claws: ASSUMED A4, arms hold still during base moves); the load starts
at its hold pose + obj_off.  Distance = the convex-hull distance of nero/clearance.py (exact GJK; mj_geomDistance returns 0.0 for separated mesh / box pairs in this repo, clearance._dist docstring, and again
here at the start pose) of
    group "robot": every collidable robot geom (base, torso, arms, grippers, wheels)  vs  every world geom except the floor
    group "load" : every collidable geom of the carried bodies (tray, mugs, crate, ...)  vs  every world geom except the floor
Robot-vs-load pairs are the grasp and are not checked.  Wheel steer angles stay at the start values (the wheels only touch the floor; wheel-vs-counter
is covered by the swept base body).  Not modelled: path-tracking error (no MEASURED margin exists yet: the caller passes the margin it has, 0 otherwise).
"""
from __future__ import annotations

import json

import numpy as np
import mujoco

from world_model.v2.nero import adapters as NA
from world_model.v2.nero import config as C
from world_model.v2.nero.base import scene as BS
from world_model.v2.nero.base import smoke as SM
from world_model.v2.nero.base import swerve as SW
from world_model.v2.nero import clearance as CL
from world_model.v2.nero.clearance import CLEARANCE_MIN          # 5 mm, ASSUMED (clearance.py)
from world_model.v2.nero.contents import envelope as E
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack import tasks as TK

DMAX = 0.10               # m: distance cut-off (pairs farther than this are skipped / reported as DMAX; = clearance._DMAX)
_EP = {}                  # world repr -> CarryEpisode (built once, ~6 s)


def world_default(task):
    from world_model.v2.adapters import contents_specs as CS
    return CS.build_world(task)


def _episode(world):
    k = repr(world)
    if k not in _EP:
        _EP[k] = SM.make_ep(world)
        mujoco.mj_forward(_EP[k].m, _EP[k].d)
    return _EP[k]


def supports(world):
    """MODEL: counter and counter2 (top height, xy extents) in the world frame.  counter2 = counter point-mirrored about scene.mirror_centre() (same height)."""
    f = getattr(BS, "support_tops_behind", None)
    if f is not None:
        return f(world)
    tops = SC.support_tops(world)
    c1 = tops["counter"]
    ctr = BS.mirror_centre()
    (x0, x1), (y0, y1) = c1["x"], c1["y"]
    a, b = BS.mirror(np.array([x0, y0]), ctr), BS.mirror(np.array([x1, y1]), ctr)
    return {"counter": dict(top=float(c1["top"]), x=(float(x0), float(x1)), y=(float(y0), float(y1))),
            "counter2": dict(top=float(c1["top"]), x=(float(min(a[0], b[0])), float(max(a[0], b[0]))), y=(float(min(a[1], b[1])), float(max(a[1], b[1]))))}


def start_pose(world):
    c, _ = BS.base_geometry()
    ep = _episode(world)
    sb = SW.SwerveBase(ep.m, ep.d)
    return (float(c[0]), float(c[1]), float(sb._yaw0))


def levels():
    return {k: tuple(v) for k, v in SM.LEVELS.items()}


def hold_q(world=None):
    cap = json.loads(E.CAP_FILE.read_text())["pot"]
    return {s: np.asarray(cap["q"]["W0_liftoff"][s], float) for s in C.SIDES}


def compose(world):
    """compiled two-table swerve model of the world (the arms' SRS constants are read from it)."""
    return BS.compose(world)


def place_pose_xy(world):
    return [float(x) for x in BS.place_pose_xy(world)]


def _qz(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _dist(m, d, a, b):
    try:
        return float(CL._dist(m, d, a, b))
    except NotImplementedError:                                   # a geom type without a hull (sphere / capsule): the MuJoCo distance (not verified for these)
        return float(mujoco.mj_geomDistance(m, d, a, b, DMAX, np.zeros(6)))


def sweep(world, poses, q_hold=None, obj_off=(0.0, 0.0, 0.0), clearance_min=CLEARANCE_MIN):
    """poses: [(seg_index, x, y, yaw), ...] of the base rotation centre (world frame); the first pose is the start pose.
    -> dict(min_dist, worst=dict(seg, pose, group, pair, dist), by_segment {seg: min}, n_poses, n_pairs, clearance_min, ok)."""
    ep = _episode(world)
    m = ep.m
    d = mujoco.MjData(m)
    d.qpos[:] = ep.d.qpos
    qh = q_hold if q_hold is not None else hold_q(world)
    for s in C.SIDES:
        d.qpos[ep.addr[s]] = np.asarray(qh[s], float)
    base_b = m.body("base").id
    root = m.body_rootid[base_b]
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    base_j = [j for j in free if m.jnt_bodyid[j] == base_b][0]
    load_j = [j for j in free if j != base_j]
    c0 = np.array(BS.base_geometry()[0], float)
    yaw0 = SW.SwerveBase(m, ep.d)._yaw0
    ox = np.array(obj_off, float)
    R0 = np.array([[np.cos(yaw0), -np.sin(yaw0), 0.0], [np.sin(yaw0), np.cos(yaw0), 0.0], [0.0, 0.0, 1.0]])
    q_ref = d.qpos.copy()
    for j in load_j:                                              # load at its hold pose + the offset (start frame -> world through yaw0)
        a = m.jnt_qposadr[j]
        q_ref[a:a + 3] += R0 @ ox
    valid = lambda g: bool(m.geom_contype[g] | m.geom_conaffinity[g])
    world_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and valid(g) and m.geom(g).name != "floor"]
    load_roots = {int(m.jnt_bodyid[j]) for j in load_j}
    robot_g = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root and valid(g)]
    load_g = [g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) in load_roots and valid(g)]
    groups = (("robot", robot_g), ("load", load_g))
    gname = lambda g: m.geom(g).name or (m.body(m.geom_bodyid[g]).name + "/geom")
    rad = {}
    for g in robot_g + load_g + world_g:
        try:
            rad[g] = float(CL._hull_pts(m, g)[1])
        except NotImplementedError:
            pass
    best = dict(dist=DMAX)
    by_seg, n_pairs = {}, 0
    for (seg, x, y, yaw) in poses:
        dl = yaw - yaw0
        R = np.array([[np.cos(dl), -np.sin(dl), 0.0], [np.sin(dl), np.cos(dl), 0.0], [0.0, 0.0, 1.0]])
        t = np.array([x - c0[0], y - c0[1], 0.0])
        d.qpos[:] = q_ref
        for j in [base_j] + load_j:
            a = m.jnt_qposadr[j]
            p = q_ref[a:a + 3].copy()
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, _qz(dl), q_ref[a + 3:a + 7])
            pn = np.array([c0[0], c0[1], 0.0]) + R @ (p - np.array([c0[0], c0[1], 0.0])) + t
            d.qpos[a:a + 3] = pn
            d.qpos[a + 3:a + 7] = q
        mujoco.mj_kinematics(m, d)
        for grp, gl in groups:
            for g1 in gl:
                for g2 in world_g:
                    if g1 in rad and g2 in rad and np.linalg.norm(d.geom_xpos[g1] - d.geom_xpos[g2]) - rad[g1] - rad[g2] > DMAX:
                        continue                                  # all hull points lie within rad of the geom origin: no pair closer than DMAX
                    n_pairs += 1
                    dist = _dist(m, d, g1, g2)
                    if dist < by_seg.get(seg, DMAX):
                        by_seg[seg] = float(dist)
                    if dist < best["dist"]:
                        best = dict(dist=float(dist), seg=int(seg), pose=[float(x), float(y), float(yaw)], group=grp, pair=[gname(g1), gname(g2)])
    md = float(best["dist"])
    return dict(min_dist=md, worst={k: v for k, v in best.items() if k != "dist"} | dict(dist=md), by_segment={int(k): v for k, v in sorted(by_seg.items())},
                n_poses=len(poses), n_pairs=int(n_pairs), clearance_min=float(clearance_min), ok=bool(md >= clearance_min),
                tag="MODEL kinematic sweep; convex-hull GJK distance (nero/clearance._dist) on collision geoms; no path-tracking margin")
