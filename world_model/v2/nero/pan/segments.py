"""pan/segments.py -- the covered-pan push + clamp + lift as a chain of hybrid SEGMENTS (user rule, 10-29).  SIMULATION ONLY.

Rule: each segment is one continuous motion in ONE contact mode, and its boundaries are quasi-static rest points. Contact events
(claw closing, finger under the rim, lift-off) happen inside a segment, never at a seam. Segment k is admitted from boundary k
only if, for every sampled theta in the current box Theta_k:
  (a) the segment never enters the unsafe set U;
  (b) from every point inside it (sampled densely and through contact events) its backup stays out of U and ends in R(Theta).
Task success is measured, not certified. The full plan is only a proposal; segments are chained by their backups.

Segments use the same motions and timing as pan/under_rim.run, which stays unchanged:
  B0     pan resting on the stove, claws open at their pre-grasp poses. The free-space approach to them is NOT simulated (ASSUMED).
  pinch  close the left claw on the handle 0.7 s, settle 0.3 s.            Pan on the stove. Event: left fingers touch the handle.
  push   the left claw slides the pan along -u by `push`, 1.5 s; settle 0.3 s.  Pan sliding on the stove under the left palm.
  clamp  right claw in 1.0 s, close 0.7 s, settle 0.3 s.  Pan on the stove, left holding. Events: right fingers on bottom / lid.
  lift   both claws up LIFT 1.5 s, hold 2.0 s.                              Pan carried. Event: lift-off.
Backups (from any point inside the segment; each starts by holding the measured joint angles for 0.2 s):
  pinch  open the left claw, back the left claw 6 cm out along its approach, settle.
  push   pull the left claw back to where the push started (the pan comes back from the edge), then the pinch backup.
  clamp  open the right claw, return it to its stand-off, then the push backup.
  lift   lower both claws to their pre-lift poses, settle, payload model off, then the clamp backup.

Bound labels follow AGENTS.md "Bounds and assumptions": MEASURED / MODEL / ASSUMED. They are listed in BOUNDS and copied into
every result.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import zlib

os.environ.setdefault("MUJOCO_GL", "osmesa")      # observe_pose renders on the CPU; must precede the first mujoco import

import numpy as np
import mujoco

from .. import config as C
from .. import adapters as A
from .. import gripper as G
from .. import weldcontact as WC
from ..ik import ArmIK
from ..democ import kin as K
from . import covered as CV
from . import scene as S
from .handle_grasp import (DT, LIFT, FINGER_OPEN, LEFT_APPROACH, LEFT_JAW, TwoCtl, find_pregrasp, _set_fingers, handle_point,
                          _arm_contacts)
from .rim_geometry import RIGHT_APPROACH, RIGHT_JAW, APPROACH_IN, EDGE_CLEAR, _perturb_pose

SIDES = C.SIDES
SEGMENTS = ("pinch", "push", "clamp", "lift")

# ---- unsafe set U (ground truth, checked at every 5 ms step) ----
U = dict(
    tilt_deg=15.0,     # ASSUMED: tilt past which the loose lid / contents count as lost
    lid_mm=10.0,       # ASSUMED: lid more than this off its seat = lid off
    drop_mm=15.0,      # ASSUMED: pan centre this far below its resting height = dropped (off the stove or out of the claws)
    joint_tol=0.005,   # ASSUMED tolerance [rad]; the joint ranges are MODEL (MJCF jnt_range)
    effort=1.0,        # MODEL: |tau| / TAU_MAX (config.tau_limits, YOR URDF effort 100 Nm, itself ASSUMED in config.py)
)
#   contents spilled: MODEL geometry: puck centre outside (inner radius - puck radius + 5 mm) or > 6 cm from the bottom
#   off the stove edge: MODEL quasi-static tipping: COM of pan+lid+contents outside the stove top while the pan is not lifted
#   collision: any robot geom touching anything, except gripper (palm/finger) geoms on the pan or lid, and the two fingers of one claw
# ---- task-ready set R(Theta) (ground truth, at the end of a backup) ----
R = dict(
    tilt_deg=3.0,      # ASSUMED: upright
    dz_mm=3.0,         # ASSUMED: resting height within this of the nominal resting height (on the stove, not on the claw)
    lid_mm=5.0,        # ASSUMED: lid on
    support_frac=0.9,  # ASSUMED: the stove carries >= 90 % of the pan + lid + contents weight
    pan_speed=0.005,   # ASSUMED m/s: at rest
    arm_speed=0.05,    # ASSUMED rad/s: arms at rest
    yaw_deg=8.5,       # ASSUMED = the prior-box yaw range (+-8 deg, on which the pinch is admitted) + 0.5 deg settling tolerance
)
R_V1 = R
# R_V2 (user decision 2026-10-29, cert/ v2): a part-pushed pan is task-ready if it is at rest and STABLE, not only when its rim
# circle is on the stove: "rim" is replaced by "stable" = the total COM lies >= STABLE_MARGIN inside the pan-on-stove contact area
# (bottom disc of radius r_out intersected with the stove top; MODEL, cert/model_check.support_polygon).  Other R entries as R_V1.
STABLE_MARGIN = 0.005  # ASSUMED (= cert/model_check.SUPPORT_MARGIN)
# R_V2 reach: a contact-free pre-grasp at the handle with the fingers at the RE-GRASP opening (handle half-thickness + the release
# opening of cert/backups, the way the claw let go), not fully open: a full open puts the lower finger into the stove pad once the
# push has moved the handle over it (smokeB 10-29).  MODEL (kinematic IK + contact check); the re-grasp itself is not yet run.
REGRASP_OPEN = 0.26   # opening fraction (1 = open): ASSUMED design value from the earlier metre form 0.014/2 + 0.006 (handle_t/2 + RELEASE_OPEN) / 0.05
#   contents in: as U. No part of the pan body over the stove edge: MODEL (rim circle inside the stove top; the handle is excluded,
#   it overhangs the stove by design). Handle in the left arm's reach: contact-free IK pre-grasp exists at the TRUE handle pose.
#   Claws released: no robot contact with pan or lid.
# ---- boundary region-of-attraction check (sampled, not a proof) ----
ROA = dict(t_hold=1.0, n_kick=4, kick_pan=0.02, kick_pan_w=0.2, kick_arm=0.05,   # ASSUMED: kicks [m/s, rad/s, rad/s]
           drift_mm=3.0, tilt_deg=1.0, speed=0.005)                              # ASSUMED pass thresholds after t_hold
# ---- right-claw clamp geometry. "v1" = pan/under_rim (10-28): the IK hand point (not the fingertip) stops EDGE_CLEAR short of the
# stove edge, so the fingertip overshoots into the stove side by ~12 mm, and the gripper base clears the table by 0.5 mm; in sim the
# right elbow rests on the table (13-17 N, found 10-29). "v2": fixes below. ----
CLAMP = dict(
    v1=dict(tip=0.0, dz=0.0, open_R=FINGER_OPEN, clear=None, jm=None),
    v2=dict(tip=0.018,      # MODEL: fingertip 18 mm past the IK hand point along the approach (finger mesh at the pre-grasp, 10-29)
            dz=0.010,       # design: claw 10 mm higher -> gripper base ~10 mm above the table (base mesh spans 164 mm along the jaw)
            open_R=0.7,   # design: right opening fraction 0.7 (earlier form: 35 mm per finger of 50 mm) so both still clear the 45 mm pan stack
            clear=0.005,    # ASSUMED: min arm-to-table/stove/counter clearance along the approach + 5 cm lift (kinematic check)
            jm=dict(left=0.05, right=0.01)),   # ASSUMED: min distance [rad] to every joint limit along the push/approach + lift path.
                            # The right arm reaches the far rim near its joint range: best path margin 0.016-0.023 rad in the nominal world
                            # (4 camera-noise seeds), so 0.02 refused half of them; 0.01 = 2x the U tolerance.
)
POT_HANDLE2_LEN = 0.13   # ASSUMED (11-01 pot variant): second handle = the first (symmetric pot); 0.09 refused 6/9 smoke worlds (right claw
                         # hit the stove side: pinch point only ~3.5 cm past the stove edge); pinch at the plan's grasp_frac like the left
STACK_H = 0.045      # MODEL: pan bottom + wall + lid = 2.5 + 40 + 2.5 mm, what the right claw closes on
RETREATS = (0.06, 0.05, 0.04, 0.03, 0.02, 0.01)   # design: pinch-backup retreat along -approach [m]; the longest one whose
                                                  # kinematic path keeps the arm's env clearance is used (0 = stay open)
BRANCH_DT, BRANCH_DT_HOLD = 0.1, 0.5   # design: backup branch spacing in motion / hold phases [s], plus every contact event

BOUNDS = {
    "U": {k: (v, "MODEL" if k == "effort" else "ASSUMED") for k, v in U.items()},
    "R": {k: (v, "ASSUMED") for k, v in R.items()},
    "ROA": {k: (v, "ASSUMED") for k, v in ROA.items()},
    "notes": [
        "pan bottom is 2.5 mm thick: the puck can sink through it in contact; puck contacts count as pan contacts (artefact)",
        "claw forces from joint torques assume perfect joint-friction compensation (contact constraint torques only)",
        "U contents/edge/collision and R edge/reach: MODEL (MJCF geometry, quasi-static COM test, IK)",
        "free-space approach to the pre-grasp poses not simulated (ASSUMED collision-free)",
        "ROA check is sampled: do-nothing hold + n_kick random velocity kicks, thresholds ASSUMED",
        "arm/env clearance: hull vertices -> box (exact) + box-edge samples every 2 mm -> hull planes (lower bound); conservative",
        "pinch-backup retreat: chosen at backup time from the arm's joint angles: longest of RETREATS along the claw's current approach axis "
        "with kinematic env clearance >= CLAMP clear (10-29: a fixed 6 cm, then a B0-checked one, hit the near counter after the claw turned)",
        "backup admission is sampled: branch points every 0.1 s (0.5 s in holds) plus contact events",
    ],
}


ENV_GEOMS = ("pan_table_top", "stove_pad", "counter_near", "floor")


_VERT_CACHE = {}


_BOX_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]   # cube corners differing in one bit
EDGE_STEP = 0.002    # design: spacing of the box-edge samples [m]


def _arm_env_clearance(m, d, side, distmax=0.05):
    """conservative min distance [m] (negative = penetration) from the arm's collision hulls (MuJoCo collides a mesh's convex
    hull) to the environment boxes (table, stove, counter) and the floor plane, capped at distmax:
      - hull vertices -> box: exact signed distance (covers hull vertex / box face);
      - box-edge points every EDGE_STEP near the link -> hull: max over the hull's face planes, a LOWER bound on the point's
        distance (covers a box edge against a hull face, which the vertex test misses);
      - floor: hull vertices (exact for a plane).
    Can only under-read the distance by about EDGE_STEP/2 sampling error, never over-read it for these shapes.
    History 10-29: mj_geomDistance returned 0.0 for separated mesh/box pairs (closest point 0.5 m off the box) and a vertex-only
    test read 3.3 mm at a box-edge contact."""
    key = (id(m), side)
    if key not in _VERT_CACHE:
        from scipy.spatial import ConvexHull
        gs = []
        for g in range(m.ngeom):
            if ((m.body(m.geom_bodyid[g]).name or "").startswith(f"{side}_arm") and (m.geom_contype[g] or m.geom_conaffinity[g])
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH):
                did = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[did]:m.mesh_vertadr[did] + m.mesh_vertnum[did]].astype(float)
                H = ConvexHull(V)
                Vh = V[H.vertices].copy()
                c = Vh.mean(axis=0)
                gs.append((g, Vh, H.equations.copy(), c, float(np.linalg.norm(Vh - c, axis=1).max())))
        boxes = [m.geom(n).id for n in ENV_GEOMS if m.geom_type[m.geom(n).id] == mujoco.mjtGeom.mjGEOM_BOX]
        _VERT_CACHE[key] = (gs, boxes)
    gs, boxes = _VERT_CACHE[key]
    sgn = np.array([[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)]) * 2.0 - 1.0
    best = distmax
    for g, Vh, Eq, c, rad in gs:
        Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        W = Vh @ Rg.T + pg
        best = min(best, float(W[:, 2].min()))                       # floor at z = 0
        cw = c @ Rg.T + pg
        for e in boxes:
            Re, pe, h = d.geom_xmat[e].reshape(3, 3), d.geom_xpos[e], m.geom_size[e]
            q = np.abs((W - pe) @ Re) - h
            best = min(best, float((np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)).min()))
            C = (sgn * h) @ Re.T + pe                                  # box corners, world
            reach = rad + distmax
            pts = []
            for a, b in _BOX_EDGES:
                u = C[b] - C[a]
                L = float(np.linalg.norm(u))
                u = u / L
                t0 = float((cw - C[a]) @ u)
                perp2 = float(np.sum((cw - C[a] - t0 * u) ** 2))
                if perp2 > reach * reach:
                    continue
                half = np.sqrt(reach * reach - perp2)
                lo, hi = max(0.0, t0 - half), min(L, t0 + half)
                if hi < lo:
                    continue
                ts = np.arange(lo, hi + EDGE_STEP, EDGE_STEP)
                pts.append(C[a] + np.minimum(ts, hi)[:, None] * u)
            if pts:
                P = (np.vstack(pts) - pg) @ Rg                           # into the mesh frame
                lb = (P @ Eq[:, :3].T + Eq[:, 3]).max(axis=1)            # >= 0 outside: lower bound on the distance
                best = min(best, float(lb.min()))
    return best


POT_FRAC = (0.0, -0.10, 0.10, -0.20, 0.20)    # design (11-01 pot): pinch-point shift along the handle from grasp_frac (ASSUMED grid,
POT_YAW = (0.0, -15.0, 15.0, -30.0, 30.0)     #   = skills/grasp.py LEFT_FRAC / LEFT_YAW / LEFT_ROLL, raised 10-30)
POT_ROLL = (0.0, -20.0, 20.0)


def _fold_c2(est):
    """11-01 pot: fold a pose estimate's yaw into (-90, 90] deg.  MODEL: the pot (two equal handles, centred lid knob) is the same
    shape turned 180 deg, so the fit cannot tell its handles apart; the hidden contents are not seen either way."""
    y = (float(est["yaw_deg"]) + 90.0) % 180.0 - 90.0
    return dict(est, yaw_deg=y, yaw_folded=bool(abs(y - float(est["yaw_deg"])) > 1e-9))


def _handle_grip(p):
    """opening fraction on a leg where the claw is CLOSED on the handle: jaws posed so the inner gap at the TCP = handle_t + 2 x 2 mm (gripper.opening_for_width)."""
    return G.opening_for_width(p.handle_t)["frac"]


def _pot_pregrasp(m, d, side, p, pos0, R0, aim, Rz, approach0, jaw0, sgn, frac0, cl):
    """pot handle pinch for one claw: the nominal candidate first, then frac x yaw x roll variants in a FIXED order; first one whose
    pre-grasp + 5 cm lift path keeps the clearance / joint-margin bounds (find_pregrasp_clear) wins.  sgn +1 = handle, -1 = handle2.
    Handle points are taken in the NOMINAL pan pose and moved by the robot's aim (estimate), like v1's aim(handle_nom)."""
    from .rim_geometry import _rot
    ax = p.handle_axis()
    length = p.handle_len if sgn > 0 else p.handle2_len
    clear = cl["clear"] or 0.005
    jm = (cl["jm"] or {}).get(side, 0.0)
    last = None
    for k, (df, yaw, roll) in enumerate((a, b, c) for a in POT_FRAC for b in POT_YAW for c in POT_ROLL):
        frac = float(np.clip(frac0 + df, 0.2, 0.9))
        loc = sgn * ax * (p.pan_radius + p.pan_wall_t + frac * length) + np.array([0, 0, p.pan_bottom_t + p.pan_wall_h - p.handle_t / 2])
        tgt = aim(pos0 + R0 @ loc)
        app, jaw = Rz @ approach0, Rz @ jaw0
        app = _rot(jaw, yaw) @ app
        jaw = _rot(app, roll) @ jaw
        budget = dict(n_sol=16, seeds=120) if k == 0 else dict(n_sol=6, seeds=40)
        try:
            r = find_pregrasp_clear(m, d, side, tgt, app, jaw, [(np.array([0, 0, LIFT]), _handle_grip(p))], clear, FINGER_OPEN, jm=jm,
                                    **budget)
            r["cand"] = dict(frac=round(frac, 3), yaw=yaw, roll=roll, k=k)
            return r
        except RuntimeError as e:
            last = e
    raise RuntimeError(f"pot: no {side} handle candidate ({last})")


def find_pregrasp_clear(m, d0, side, target, approach, jaw, deltas, clear, finger_open, jm=0.0, n_sol=16, seeds=120):
    """like handle_grasp.find_pregrasp, but each IK solution is also followed (kinematically, PoseIK, orientation held) through
    the Cartesian waypoints `deltas` from the pre-grasp; keep solutions whose arm stays >= `clear` from the table, stove, counter
    and floor and >= `jm` [rad] from every joint limit along the whole path; best joint margin.  Raises if none."""
    ik = ArmIK(m, side)
    best = None
    worst_seen = -9.0
    for q, e in ik.collect(target, approach, jaw, n_sol=n_sol, seeds=seeds):
        d = mujoco.MjData(m)
        d.qpos[:] = d0.qpos
        d.qpos[ik.qa] = q
        _set_fingers(m, d, side, finger_open)
        mujoco.mj_forward(m, d)
        if _arm_contacts(m, d, side, allow=()) < -1e-3:
            continue
        fwd = (d.xpos[m.body(f"{side}_arm_finger1").id] - d.xpos[m.body(f"{side}_arm_gripper_base").id]) @ approach
        if fwd <= 0:                                  # ArmIK accepts the claw pointing backwards along the approach
            continue
        cl = _arm_env_clearance(m, d, side)
        jmin = float(np.min(np.minimum(q - ik.lo, ik.hi - q)))
        pik = K.PoseIK(m, side)
        p0, R0 = pik.ik.pose(q)
        qq, prev = q.copy(), np.zeros(3)
        for dl in deltas:
            if isinstance(dl, tuple):            # (delta, finger opening on this leg: open on an approach, closed on the object)
                dl, fo = dl
                _set_fingers(m, d, side, fo)
            for a in np.linspace(0, 1, 7)[1:]:
                qq, _ = pik.solve(p0 + prev + a * (np.asarray(dl) - prev), R0, qq, q_post=q, k_post=0.1, iters=60, tol=1e-9)
                d.qpos[ik.qa] = qq
                mujoco.mj_forward(m, d)
                cl = min(cl, _arm_env_clearance(m, d, side))
                jmin = min(jmin, float(np.min(np.minimum(qq - ik.lo, ik.hi - qq))))
            prev = np.asarray(dl, float)
        worst_seen = max(worst_seen, min(cl / max(clear, 1e-9), jmin / max(jm, 1e-9)))
        if cl >= clear and jmin >= jm and (best is None or ik.margin(q) > best[1]):
            best = (q, ik.margin(q), e, cl, jmin)
    if best is None:
        raise RuntimeError(f"no pre-grasp for the {side} claw with >= {clear*1e3:.0f} mm clearance and >= {jm:.2f} rad joint margin "
                           f"along its path (best {worst_seen:.2f} of required)")
    return dict(q=best[0], margin=best[1], ik_err=[float(x) for x in best[2]], target=target, clear_mm=1e3 * best[3],
                path_joint_margin=best[4])


def path_clearance(m, d0, side, q, deltas, finger_open):
    """min env clearance [m] of the arm along Cartesian waypoints `deltas` from joint config q (orientation held, kinematic)."""
    ik = ArmIK(m, side)
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    d.qpos[ik.qa] = q
    _set_fingers(m, d, side, finger_open)
    mujoco.mj_forward(m, d)
    cl = _arm_env_clearance(m, d, side)
    pik = K.PoseIK(m, side)
    p0, R0 = pik.ik.pose(q)
    qq, prev = q.copy(), np.zeros(3)
    for dl in deltas:
        for a in np.linspace(0, 1, 7)[1:]:
            qq, _ = pik.solve(p0 + prev + a * (np.asarray(dl) - prev), R0, qq, q_post=q, k_post=0.1, iters=60, tol=1e-9)
            d.qpos[ik.qa] = qq
            mujoco.mj_forward(m, d)
            cl = min(cl, _arm_env_clearance(m, d, side))
        prev = np.asarray(dl, float)
    return float(cl)


def world_seed(tag):
    return zlib.crc32(str(tag).encode()) & 0x7FFFFFFF


class Episode:
    """One world theta, one plan, with the robot's policy (observe_pose re-aim). Snapshot/restore for branching backups."""

    def __init__(self, world, push=0.07, grasp_frac=0.65, grip_force=10.0, observe="zed", obs_seed=0, retries=3,
                 payload_mass=None, clamp="v2", est=None, obj="pan", claw_N=None, lift_ctl="nominal", grasp=None):
        """est: a GIVEN pose estimate dict(dx, dy, yaw_deg) used instead of running observe_pose (cert/ reconstruction: the
        robot aimed there; this world is one theta it must cover).  None = run observe_pose at noise `observe`."""
        from . import observe_pose as O
        self.obj = obj                                  # 11-01: "pan" (one handle: push + rim clamp) | "pot" (two side handles: pinch both)
        if grasp is None:      # 11-03 USER: "we use the rigid weld for everything"
            grasp = G.default_grasp()
        assert grasp in ("friction", "weld"), grasp
        self.grasp = grasp                              # 11-01 USER scope: "weld" = rigid weld once the claw closes (ASSUMED grasp)
        if obj == "pot":
            push = 0.0
        self.world, self.plan = world, dict(push=push, grasp_frac=grasp_frac, grip_force=grip_force, clamp=clamp, obj=obj, claw_N=claw_N,
                                            lift_ctl=lift_ctl)
        cl = self.clamp = CLAMP[clamp]
        perturb = dict(world.get("perturb") or {})
        placement, hidden_mass = world["placement"], world.get("hidden_mass")
        cp = CV.CoveredParams(placement=placement)
        if hidden_mass is not None:
            cp = dataclasses.replace(cp, hidden_mass=hidden_mass, overload_mass=hidden_mass)
        hl = float(world.get("handle_len") or cp.pan.handle_len)          # 11-01 hard worlds: pot handle length (both handles)
        cp = dataclasses.replace(cp, pan=dataclasses.replace(cp.pan, handle_grasp_frac=grasp_frac,
                                                             handle_len=hl if obj == "pot" else cp.pan.handle_len,
                                                             handle2_len=(hl if world.get("handle_len") else POT_HANDLE2_LEN) if obj == "pot" else 0.0,
                                                             handle2_grasp_frac=grasp_frac))
        self.cp, self.p = cp, cp.pan
        p = self.p
        if grasp == "weld":                             # inactive welds gripper_base <-> pan, switched on at the end of each "close"
            spec = CV.build_spec(cp)
            for s_ in SIDES:
                spec.add_equality(name=f"weld_{s_}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                  name1=f"{s_}_arm_gripper_base", name2="pan", active=False)
            m = self.m = CV.S.apply_sim_defaults(spec.compile())
        else:
            m = self.m = CV.compose(cp)
        m.opt.timestep = DT
        self.weld_id = {s_: m.equality(f"weld_{s_}").id for s_ in SIDES} if grasp == "weld" else {}
        self.wc = WC.WeldContacts(m, "pan", self.weld_id, SIDES, active=(grasp == "weld"))   # 11-03: gripper<->pan contacts off while welded
        for s_ in SIDES:     # nyu weld grasp: the pre-grasp pose has the jaws / palm around (and through) the handle; the jaws only pose, so gripper<->pan contacts are off from the start
            self.wc.set(s_, True)        # (until the first release; weld(s, False) then restores them once the gripper is clear of the pan)
        if perturb.get("mu_scale"):
            m.geom_friction[:, 0] *= perturb["mu_scale"]
        if perturb.get("lid_mu_scale"):        # 11-01 hard worlds: lid friction.  MuJoCo combines a pair by MAX friction unless the
            lid_b = m.body("lid").id           # priorities differ, so the lid geoms get priority 1 (pan 0) and their scaled mu governs
            sel = m.geom_bodyid == lid_b
            m.geom_friction[sel, 0] *= perturb["lid_mu_scale"]
            m.geom_priority[sel] = 1
        self.tau_max = C.tau_limit()      # F3 11-03: one source (config.tau_limit; 16 Nm ASSUMED, BIMANUAL_LEGACY_TAU=1 -> 100)
        d = self.d = mujoco.MjData(m)
        S.settle(m, d, 1.0)
        self.pan, self.lid = m.body("pan").id, m.body("lid").id
        self.hid = m.body("hidden").id if placement != "none" else None
        self.pos0, self.R0 = d.xpos[self.pan].copy(), d.xmat[self.pan].reshape(3, 3).copy()   # nominal settled pose
        self.z_rest = float(self.pos0[2])
        handle_nom = d.site_xpos[m.site("pan_handle_grasp").id].copy()
        handle2_nom = d.site_xpos[m.site("pan_handle2_grasp").id].copy() if obj == "pot" else None
        if observe is not None and est is None:
            rdr = mujoco.Renderer(m, O.CAM["height"], O.CAM["width"])
            tmpl = O.template(m, d, p, renderer=rdr)
        if any(perturb.get(k) for k in ("dx", "dy", "yaw_deg")):
            _perturb_pose(m, d, p.pan_centre_xy, perturb.get("dx", 0.0), perturb.get("dy", 0.0), perturb.get("yaw_deg", 0.0))
            S.settle(m, d, 0.5)
        self.u_true = d.xmat[self.pan].reshape(3, 3) @ self.R0.T @ p.handle_axis()
        u = p.handle_axis()
        left_target, left_app, left_jaw = handle_nom, LEFT_APPROACH, LEFT_JAW
        centre_aim = np.array([p.pan_centre_xy[0], p.pan_centre_xy[1], 0.0])
        self.obs = None
        self.refused = None
        if observe is not None or est is not None:
            rng = np.random.default_rng(obs_seed)
            tries = []
            given = est
            for _ in range(retries if given is None else 1):
                est = dict(given, fit_ok=True) if given is not None else O.estimate(m, d, p, tmpl, observe, rng, renderer=rdr)
                tries.append(est)
                if est["fit_ok"]:
                    break
            if given is None:
                rdr.close()
                self._obs_ctx = dict(tmpl=tmpl, noise=observe, rng=rng, retries=retries)   # 11-01: for reobserve() (same rng stream)
            if obj == "pot" and est.get("fit_ok"):
                est = _fold_c2(est)            # MODEL: two equal handles + centred knob -> the pot looks the same turned 180 deg
            tru = O.true_pose(m, d, p, self.pos0, self.R0)
            self.obs = dict(est=est, true=tru, n_images=len(tries), noise=observe if given is None else "given",
                            residual_mm=1e3 * float(np.hypot(est["dx"] - tru["dx"], est["dy"] - tru["dy"])),
                            residual_yaw_deg=float(est["yaw_deg"] - tru["yaw_deg"]))
            if not est["fit_ok"]:
                self.refused = f"observe_pose: no good fit in {len(tries)} images"
            else:
                aim, Rz = O.aim_transform(est, p.pan_centre_xy)
                left_target, left_app, left_jaw = aim(handle_nom), Rz @ LEFT_APPROACH, Rz @ LEFT_JAW
                u = Rz @ u
                centre_aim = aim(centre_aim)
        self.u_aim, self.left_app = u, left_app
        try:
            if self.refused:
                raise RuntimeError(self.refused)
            if obj == "pot":
                ok_est = self.obs is not None and est.get("fit_ok")
                aim_p = aim if ok_est else (lambda q: np.asarray(q, float))
                Rz_p = Rz if ok_est else np.eye(3)
                preL = _pot_pregrasp(m, d, "left", p, self.pos0, self.R0, aim_p, Rz_p, LEFT_APPROACH, LEFT_JAW, +1, grasp_frac, cl)
            elif cl["jm"] is None:
                preL = find_pregrasp(m, d, "left", left_target, left_app, left_jaw)
            else:
                preL = find_pregrasp_clear(m, d, "left", left_target, left_app, left_jaw,
                                           ([(np.array([0, 0, LIFT]), _handle_grip(p))] if obj == "pot" else
                                            [(-push * u, _handle_grip(p)), (-push * u + np.array([0, 0, LIFT]), _handle_grip(p))]),
                                           cl["clear"], FINGER_OPEN, jm=cl["jm"]["left"])
            d.qpos[ArmIK(m, "left").qa] = preL["q"]
            _set_fingers(m, d, "left", FINGER_OPEN)
            mujoco.mj_forward(m, d)
            stove_edge = p.pan_centre_xy[1] - S.ASSUMED["stove_half"][1]
            if observe is None:
                rim_after = p.pan_centre_xy[1] - push - (p.pan_radius + p.pan_wall_t)
                x_R = p.pan_centre_xy[0]
            else:
                c_after = centre_aim - push * u
                rim_after = c_after[1] - (p.pan_radius + p.pan_wall_t)
                x_R = c_after[0]
            self.overhang_planned = float(stove_edge - rim_after)
            z_mid = p.pan_base_z() + 0.5 * (p.pan_bottom_t + p.pan_wall_h + CV.ASSUMED_COVERED["lid_t"])
            target_R = np.array([x_R, stove_edge - EDGE_CLEAR - cl["tip"], z_mid + cl["dz"]])
            if obj == "pot":                         # right claw pinches handle2 (approach +y towards the pot, jaw z), like the left
                self.overhang_planned = None
                preR = _pot_pregrasp(m, d, "right", p, self.pos0, self.R0, aim_p, Rz_p, RIGHT_APPROACH, RIGHT_JAW, -1, grasp_frac, cl)
            elif cl["clear"] is None:
                preR = find_pregrasp(m, d, "right", target_R - APPROACH_IN * RIGHT_APPROACH, RIGHT_APPROACH, RIGHT_JAW)
            else:
                preR = find_pregrasp_clear(m, d, "right", target_R - APPROACH_IN * RIGHT_APPROACH, RIGHT_APPROACH, RIGHT_JAW,
                                           [(APPROACH_IN * RIGHT_APPROACH, cl["open_R"]),
                                            (APPROACH_IN * RIGHT_APPROACH + np.array([0, 0, LIFT]), 0.5 * STACK_H)],
                                           cl["clear"], cl["open_R"], jm=cl["jm"]["right"])
            d.qpos[ArmIK(m, "right").qa] = preR["q"]
            _set_fingers(m, d, "right", FINGER_OPEN if obj == "pot" else cl["open_R"])
            self.retreat = 0.06
            if cl["clear"] is not None:
                self.retreat = next((r for r in RETREATS
                                     if path_clearance(m, d, "left", preL["q"], [-r * left_app], FINGER_OPEN) >= cl["clear"]), 0.0)
            self.pregrasp_margin = dict(left=preL["margin"], right=preR["margin"], clear_mm=[preL.get("clear_mm"), preR.get("clear_mm")],
                                        path_joint_margin=[preL.get("path_joint_margin"), preR.get("path_joint_margin")],
                                        retreat_m=self.retreat,
                                        cand=[preL.get("cand"), preR.get("cand")])   # 11-01: the pot candidate actually used (frac, yaw, roll)
        except RuntimeError as e:
            self.refused = self.refused or str(e)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        A.disable_arm_servos(m)
        self.addr = {s: A.arm_addrs(m, s)[0] for s in SIDES}
        self.dofs = {s: A.arm_addrs(m, s)[1] for s in SIDES}
        q0 = {s: d.qpos[self.addr[s]].copy() for s in SIDES}
        self.gb = {s: m.body(f"{s}_arm_gripper_base").id for s in SIDES}
        m_nom = float(m.body_mass[self.pan] + m.body_mass[self.lid])
        if payload_mass is not None:
            m_nom = float(payload_mass)
        self.m_nom = m_nom
        self.W = 9.81 * float(m.body_mass[self.pan] + m.body_mass[self.lid] + (m.body_mass[self.hid] if self.hid else 0.0))
        if lift_ctl == "adaptive":       # 11-01: payload-aware (pan/payload_obs.LoadObserver on both arms, robot-side signals)
            from .handle_grasp import AdaptivePanCtl
            self.ctl = AdaptivePanCtl(m, dict(left=q0["left"].tolist(), right=q0["right"].tolist()), self.tau_max, self.pan, self.gb,
                                      m_nom, SIDES, DT, share=0.5)
        else:
            self.ctl = TwoCtl(m, dict(left=q0["left"].tolist(), right=q0["right"].tolist()), self.tau_max, self.pan, self.gb, m_nom,
                              share=0.5)
        self.q0 = q0
        self.NP, self.FDT = self.ctl.fl.NP, self.ctl.fl.DT
        self.act = {s: m.actuator(f"{s}_arm_gripper_pos").id for s in SIDES}
        from .handle_grasp import set_claw_limit
        self.claw_N = set_claw_limit(m, list(self.act.values()), claw_N)   # 11-01: None = compose default (config.claw_default, 40 N)
        self.plan["claw_N"], self.plan["physics_substeps"] = self.claw_N, C.substeps_default()   # record what actually ran
        self.plan["grasp"] = grasp
        self.act_pos = {s: (m.actuator_gainprm[self.act[s], :3].copy(), m.actuator_biasprm[self.act[s], :3].copy(),
                            m.actuator_ctrlrange[self.act[s]].copy()) for s in SIDES}
        self.fing = {g: (s, i) for s in SIDES for i in (1, 2) for g in G.col_geoms(m, s, f"finger{i}")}
        self.grip_geoms = {g: s for s in SIDES for n in ("gripper_base", "finger1", "finger2") for g in G.col_geoms(m, s, n)}
        self.robot_geoms = {g for g in range(m.ngeom) if (m.body(m.geom_bodyid[g]).name or "").startswith(("left_arm", "right_arm"))}
        self.obj_body = {self.pan: "pan", self.lid: "lid"} | ({self.hid: "hidden"} if self.hid else {})
        # the puck sinks through the 2.5 mm pan bottom in contact (thin-shell artefact, 10-29): its contacts with the stove count as
        # support and gripper contacts with it as contact with the pan
        self.carry = {self.pan} | ({self.hid} if self.hid else set())
        self.fjoint = {s: [m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith(f"{s}_arm_gripper_joint")]
                       for s in SIDES}
        self.jrange = {s: m.jnt_range[[m.dof_jntid[a] for a in self.dofs[s]]] for s in SIDES}
        self.pan_qadr = m.jnt_qposadr[m.body(self.pan).jntadr[0]]
        self.pan_dadr = m.jnt_dofadr[m.body(self.pan).jntadr[0]]
        st = self.p.pan_centre_xy
        self.stove = (np.array(st), np.array(S.ASSUMED["stove_half"][:2]))
        self.r_out = p.pan_radius + p.pan_wall_t
        self.lid_seat = np.array([0, 0, p.pan_bottom_t + p.pan_wall_h + CV.ASSUMED_COVERED["lid_t"] / 2])
        # mutable episode state (everything below is in snapshot())
        self.k = 0
        self.paths = {s: [] for s in SIDES}
        self.gmode = {s: "pos" for s in SIDES}
        self.open_cmd = dict(left=FINGER_OPEN, right=FINGER_OPEN if obj == "pot" else cl["open_R"])
        self.gcmd = dict(self.open_cmd)
        self.mem = {}
        self.reset_track()
        self.sense = {}
        self.logging = False
        self.log = []
        self.fv = np.zeros(6)
        self._ev, self._event_now = None, False
        self._sup = []
        for s in SIDES:
            self._apply_grip(s)
        self.Rg0 = d.xmat[self.gb["left"]].reshape(3, 3).copy()          # left claw orientation at its pre-grasp
        self.RgR0 = d.xmat[self.gb["right"]].reshape(3, 3).copy()        # right claw orientation at its pre-grasp
        self.aim_yaw = float(self.obs["est"]["yaw_deg"]) if (self.obs and self.obs["est"].get("fit_ok")) else 0.0

    # ---------------- state ----------------
    _SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def weld(self, s, on):
        """rigid weld of claw s to the object (grasp="weld" only).  On: hold the CURRENT relative pose (body2 = pan in the
        gripper_base frame, anchor 0; checked: zero residual at activation).  Off: release."""
        if s not in self.weld_id:
            return
        i, m, d = self.weld_id[s], self.m, self.d
        if on:
            b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
            R1 = d.xmat[b1].reshape(3, 3)
            qi, q = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(qi, d.xquat[b1])
            mujoco.mju_mulQuat(q, qi, d.xquat[b2])
            m.eq_data[i, 0:3] = 0.0
            m.eq_data[i, 3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])
            m.eq_data[i, 6:10] = q
            m.eq_data[i, 10] = 1.0
        d.eq_active[i] = 1 if on else 0
        self.wc.set(s, bool(on), d)

    def snapshot(self, label=None):
        st = np.zeros(mujoco.mj_stateSize(self.m, self._SPEC))
        mujoco.mj_getState(self.m, self.d, st, self._SPEC)
        weld = {s: self.m.eq_data[i].copy() for s, i in self.weld_id.items()}   # relpose lives in the model, not the state
        return dict(st=st, weld=weld, ctl=copy.deepcopy(self.ctl, {id(self.m): self.m}), k=self.k,
                    paths={s: list(v) for s, v in self.paths.items()}, gmode=dict(self.gmode), gcmd=dict(self.gcmd),
                    mem=copy.deepcopy(self.mem), sense=copy.deepcopy(self.sense), label=label, t=self.k * DT, wc=self.wc.state())

    def restore(self, sn):
        for s, i in self.weld_id.items():
            self.m.eq_data[i] = sn["weld"][s]
        mujoco.mj_setState(self.m, self.d, sn["st"], self._SPEC)   # INTEGRATION state includes eq_active
        self.wc.sync(self.d, sn.get("wc"))
        self.ctl = copy.deepcopy(sn["ctl"], {id(self.m): self.m})
        self.k, self.paths = sn["k"], {s: list(v) for s, v in sn["paths"].items()}
        self.gmode, self.gcmd = dict(sn["gmode"]), dict(sn["gcmd"])
        self.mem, self.sense = copy.deepcopy(sn["mem"]), copy.deepcopy(sn["sense"])
        for s in SIDES:
            self._apply_grip(s)
        self._ev = None
        self._sup = []
        mujoco.mj_forward(self.m, self.d)

    def reset_track(self):
        self.track = dict(viol=None, margin=1.0, worst={}, tau_ratio=0.0, n_sat=0, events=[])
        if getattr(self, "ctl", None) is not None:          # 11-03 F2: FL-MPC fallback counters + true residuals (nero/fl_log.py), same window as the track
            self.ctl.rec.reset()

    # ---------------- low level ----------------
    def tcp(self, s):
        return self.d.xpos[self.gb[s]].copy()

    def gap(self, s):                                    # observable: opening fraction from the nut encoder (1 = open); gap(s) = finger_q(s)
        return G.opening_frac(self.m, self.d, s)

    def finger_q(self, s):
        """opening fraction (1 = open) from the nut encoder."""
        return G.opening_frac(self.m, self.d, s)

    def _apply_grip(self, s):
        a = self.act[s]
        if self.gmode[s] == "force":
            f = self.plan["grip_force"]
            self.m.actuator_gainprm[a, :3] = (1.0, 0.0, 0.0)
            self.m.actuator_biasprm[a, :3] = 0.0
            self.m.actuator_ctrlrange[a] = (-f, f)
        else:
            g, b, r = self.act_pos[s]
            self.m.actuator_gainprm[a, :3], self.m.actuator_biasprm[a, :3], self.m.actuator_ctrlrange[a] = g, b, r
        # gcmd: force mode = closing force [N]; position mode = opening fraction
        self.d.ctrl[a] = G.force_ctrl(self.gcmd[s]) if self.gmode[s] == "force" else G.open_frac_to_ctrl(self.gcmd[s])

    def _add_path(self, s, t0, t1, delta):
        m, d = self.m, self.d
        pik = K.PoseIK(m, s)
        q = d.qpos[self.addr[s]].copy()
        p0, R0 = pik.ik.pose(q)
        wp = [q.copy()]
        for i in range(1, 61):
            a = i / 60
            q, _ = pik.solve(p0 + delta * a * a * (3 - 2 * a), R0, q, q_post=wp[0], k_post=0.1, iters=60, tol=1e-9)
            wp.append(q.copy())
        self.paths[s].append((t0, t1, wp))

    def _cmd(self, s, t):
        q = self.q0[s]
        for t0, t1, wp in self.paths[s]:
            if t < t0:
                break
            a = np.clip((t - t0) / (t1 - t0), 0, 1) * (len(wp) - 1)
            i = min(int(a), len(wp) - 2)
            q = (1 - (a - i)) * wp[i] + (a - i) * wp[i + 1]
        return q

    def _step(self):
        m, d, ctl = self.m, self.d, self.ctl
        t = self.k * DT
        ctl.qe_ref = np.stack([self._cmd("left", t + i * self.FDT) for i in range(self.NP + 1)])
        ctl.qi_ref = np.stack([self._cmd("right", t + i * self.FDT) for i in range(self.NP + 1)])
        te, ti = ctl.tau(d)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        self.wc.tick(d)
        CV.S.step(m, d)                                  # 11-01: 1 ms physics substeps (config.substeps_default)
        self.k += 1
        r = max(float(np.max(np.abs(te) / self.tau_max)), float(np.max(np.abs(ti) / self.tau_max)))
        self.track["tau_ratio"] = max(self.track["tau_ratio"], r)
        self.track["n_sat"] += int(r >= 1.0 - 1e-6)
        self._monitor(t, r)

    # ---------------- ground truth: contacts, U ----------------
    def contacts(self):
        m, d = self.m, self.d
        sup, fz, sq = 0.0, {s: 0.0 for s in SIDES}, {s: 0.0 for s in SIDES}
        touch = {(s, i): set() for s in SIDES for i in (1, 2)}
        grip_obj = {s: 0.0 for s in SIDES}
        coll = None
        for j in range(d.ncon):
            c = d.contact[j]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if not (r1 or r2) and not (b1 in self.carry or b2 in self.carry):
                continue
            mujoco.mj_contactForce(m, d, j, self.fv)
            fn = float(self.fv[0])
            if r1 or r2:
                gr, go = (g1, g2) if r1 else (g2, g1)
                bo = m.geom_bodyid[go]
                if gr in self.grip_geoms and bo in self.obj_body:
                    grip_obj[self.grip_geoms[gr]] += fn
                    if gr in self.fing:
                        touch[self.fing[gr]].add(self.obj_body[bo])
                        sq[self.fing[gr][0]] += fn
                        if bo == self.pan:
                            f_w = c.frame.reshape(3, 3).T @ self.fv[:3]
                            fz[self.fing[gr][0]] += float(f_w[2] if m.geom_bodyid[c.geom2] == self.pan else -f_w[2])
                elif gr in self.grip_geoms and go in self.grip_geoms and self.grip_geoms[gr] == self.grip_geoms[go]:
                    pass                                          # parts of one claw (e.g. fingers meeting on a miss): not a collision
                elif coll is None and c.dist < 0:
                    coll = f"{m.geom(gr).name}-{m.geom(go).name}"
                continue
            o1, o2 = b1 in self.carry, b2 in self.carry
            if o1 == o2:
                continue
            other = b2 if o1 else b1
            if other in self.obj_body:
                continue
            f_w = c.frame.reshape(3, 3).T @ self.fv[:3]
            sup += float(-f_w[2] if o1 else f_w[2])
        return dict(support=sup, fz=fz, squeeze=sq, touch=touch, grip_obj=grip_obj, coll=coll)

    def pan_state(self):
        d = self.d
        Rp = d.xmat[self.pan].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(Rp[2, 2], -1, 1))))
        lid = float(np.linalg.norm(d.xpos[self.lid] - d.xpos[self.pan] - Rp @ self.lid_seat)) * 1e3
        spill = False
        if self.hid is not None:
            h = Rp.T @ (d.xpos[self.hid] - d.xpos[self.pan])
            spill = bool(np.hypot(h[0], h[1]) > self.p.pan_radius - CV.ASSUMED_COVERED["puck_r"] + 0.005 or abs(h[2]) > 0.06)
        mass = [self.pan, self.lid] + ([self.hid] if self.hid else [])
        com = sum(self.m.body_mass[b] * d.xipos[b] for b in mass) / sum(self.m.body_mass[b] for b in mass)
        c, h = self.stove
        dz = float(d.xpos[self.pan][2] - self.z_rest)
        com_in = float(np.min(h - np.abs(com[:2] - c)))              # > 0: COM over the stove top
        rim_in = float(np.min(h - np.abs(d.xpos[self.pan][:2] - c)) - self.r_out)   # > 0: whole rim circle over the stove
        return dict(tilt=tilt, lid_mm=lid, spill=spill, dz_mm=dz * 1e3, com_in=com_in, rim_in=rim_in,
                    v=float(np.linalg.norm(d.qvel[self.pan_dadr:self.pan_dadr + 3])))

    def _monitor(self, t, tau_ratio):
        ps, cs = self.pan_state(), self.contacts()
        jm = min(float(np.min(np.minimum(self.d.qpos[self.addr[s]] - self.jrange[s][:, 0],
                                         self.jrange[s][:, 1] - self.d.qpos[self.addr[s]]))) for s in SIDES)
        lifted = ps["dz_mm"] > 5.0
        marg = dict(tilt=1 - ps["tilt"] / U["tilt_deg"], lid=1 - ps["lid_mm"] / U["lid_mm"],
                    drop=1 + ps["dz_mm"] / U["drop_mm"], spill=-1.0 if ps["spill"] else 1.0,
                    edge=1.0 if lifted else min(1.0, ps["com_in"] / 0.03), joint=min(1.0, (jm - U["joint_tol"]) / 0.1),
                    effort=1.0 if tau_ratio <= U["effort"] + 1e-6 else -1.0, coll=-1.0 if cs["coll"] else 1.0)
        for kk, v in marg.items():
            if v < self.track["worst"].get(kk, (9, 0))[0]:
                self.track["worst"][kk] = (round(float(v), 4), round(t, 3))
        mn = min(marg.values())
        self.track["margin"] = min(self.track["margin"], mn)
        if mn <= 0 and self.track["viol"] is None:
            kk = min(marg, key=marg.get)
            self.track["viol"] = dict(kind=kk, t=round(t, 3), coll=cs["coll"], **{k: round(v, 3) if isinstance(v, float) else v
                                                                              for k, v in ps.items()})
        self._last = (ps, cs)
        self._sup.append(cs["support"])
        if len(self._sup) > 40:
            self._sup.pop(0)
        # contact events (for branch points): left fingers on handle, right fingers on pan/lid, lift-off
        ev = (tuple(sorted(cs["touch"][("left", 1)] | cs["touch"][("left", 2)])),
              tuple(sorted(cs["touch"][("right", 1)] | cs["touch"][("right", 2)])), cs["support"] < 0.5 * self.W)
        if getattr(self, "_ev", None) is not None and ev != self._ev:
            self.track["events"].append((round(t, 3), ev))
            self._event_now = True
        self._ev = ev
        if self.logging and self.k % 10 == 0:
            self.log.append(dict(t=round(t, 3), z=round(float(self.d.xpos[self.pan][2]), 4), tilt=round(ps["tilt"], 2),
                                 sup=round(cs["support"], 3), sq={s: round(v, 2) for s, v in cs["squeeze"].items()},
                                 open_frac={s: round(self.gap(s), 4) for s in SIDES}))

    # ---------------- observable signals ----------------
    def claw_wrench(self, s, sigma_tau=0.0, rng=None):
        """force at the gripper base from joint torques: the arm-joint torque due to CONTACT constraints only (J_c^T f_c), least
        squares through the 6 x 7 body Jacobian.  = joint-torque sensing with PERFECT joint-friction compensation (ASSUMED): the
        MJCF joint frictionloss (0.3 Nm, a constraint in MuJoCo) and joint-limit forces are left out (10-29: including them biased
        the tilt-probe moment ~2.7x).  sigma_tau: ASSUMED sensor noise [Nm]."""
        m, d = self.m, self.d
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, self.gb[s])
        J = np.vstack([jp, jr])[:, self.dofs[s]]
        tc = self.contact_torque()[self.dofs[s]]
        if sigma_tau and rng is not None:
            tc += rng.normal(0, sigma_tau, tc.shape)
        w = np.linalg.lstsq(J.T, tc, rcond=None)[0]
        return w[:3]

    _CONTACT_T = (mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS, mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
                  mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)

    def contact_torque(self):
        d = self.d
        n = d.nefc
        f = d.efc_force[:n].copy()
        f[~np.isin(d.efc_type[:n], self._CONTACT_T)] = 0.0
        res = np.zeros(self.m.nv)
        mujoco.mj_mulJacTVec(self.m, d, res, f)
        return res

    # ---------------- phases ----------------
    def run_phases(self, phases, branch=None, monitor=None):
        """phases: ("hold", dur) | ("stop", dur) | ("move", {side: delta}, dur) | ("move_to", {side: fn()->target}, t_full, d_full)
        | ("close"/"open", side, dur) | ("payload", on) | ("mark", key) .  branch(ep, kind, i, n) -> label|None snapshots;
        monitor(ep) -> reason|None aborts (checked every 10 steps).  Returns (status, info)."""
        snaps = []
        for ph in phases:
            kind = ph[0]
            if kind == "payload":
                self.ctl.on = bool(ph[1])
                continue
            if kind == "mark":
                self.mem[ph[1]] = {s: self.tcp(s) for s in SIDES}
                continue
            t = self.k * DT
            if kind == "move":
                dur = ph[2]
                for s, delta in ph[1].items():
                    self._add_path(s, t, t + dur, np.asarray(delta, float))
            elif kind == "move_to":
                deltas = {s: np.asarray(fn(), float) - self.tcp(s) for s, fn in ph[1].items()}
                dist = max(float(np.linalg.norm(v)) for v in deltas.values())
                dur = max(0.8, ph[2] * dist / ph[3])
                for s, delta in deltas.items():
                    self._add_path(s, t, t + dur, delta)
            elif kind == "retreat":                   # pinch-backup retreat, chosen NOW from the arm's own joint angles: the claw's current
                s = ph[1]                             # approach axis, the longest of RETREATS whose kinematic path keeps the env clearance
                app = self.d.xmat[self.gb[s]].reshape(3, 3) @ self.Rg0.T @ self.left_app
                q = self.d.qpos[self.addr[s]].copy()
                clear = self.clamp["clear"] if self.clamp["clear"] is not None else 0.0
                r = next((r for r in RETREATS if path_clearance(self.m, self.d, s, q, [-r * app], FINGER_OPEN) >= clear), 0.0)
                self.sense.setdefault("retreats_m", []).append(r)
                dur = 0.0
                if r > 0:
                    dur = max(0.8, 1.6 * r / 0.06)
                    self._add_path(s, t, t + dur, -r * app)
            elif kind == "retreat_fixed":             # cert/ v2 stored backups: retreat by a distance FIXED at admission, along the
                s, r = ph[1], float(ph[2])            # claw's current approach axis (arm FK: state feedback, not an online choice)
                app = self.d.xmat[self.gb[s]].reshape(3, 3) @ self.Rg0.T @ self.left_app if s == "left" else \
                    self.d.xmat[self.gb[s]].reshape(3, 3) @ self.RgR0.T @ RIGHT_APPROACH
                dur = max(0.8, 1.6 * r / 0.06) if r > 0 else 0.0
                if r > 0:
                    self._add_path(s, t, t + dur, -r * app)
            elif kind == "stop":                      # decelerate smoothly to rest (cubic Hermite: current velocity -> 0)
                dur = ph[1]
                for s in SIDES:
                    q = self.d.qpos[self.addr[s]].copy()
                    v = self.d.qvel[self.dofs[s]].copy() * dur
                    q1 = q + 0.5 * v
                    wp = [(2 * a ** 3 - 3 * a ** 2 + 1) * q + (a ** 3 - 2 * a ** 2 + a) * v + (-2 * a ** 3 + 3 * a ** 2) * q1
                          for a in np.linspace(0, 1, 31)]
                    self.paths[s].append((t, t + dur, wp))
            elif kind in ("close", "open"):
                s, dur = ph[1], ph[2]
                self.gmode[s] = "force" if kind == "close" else "pos"
                self.gcmd[s] = self.plan["grip_force"] if kind == "close" else self.open_cmd[s]
                if kind == "close" and self.grasp == "weld":   # 11-03: weld grasp under nyu = jaws POSED so the inner gap at the TCP = handle + 2 x 2 mm (no squeeze)
                    self.gmode[s] = "pos"
                    self.gcmd[s] = G.opening_for_width(self.p.handle_t)["frac"]
                if kind == "open" and len(ph) > 3:    # cert/ v2 release: open each finger by a FIXED amount past its encoder reading
                    self.gcmd[s] = min(self.open_cmd[s], self.finger_q(s) + float(ph[3]))
                if kind == "open":
                    self.weld(s, False)
                self._apply_grip(s)
            else:
                dur = ph[1]
            n = int(round(dur / DT))
            for i in range(n):
                self._event_now = False
                self._step()
                if branch is not None:
                    lab = branch(self, kind, i, n)
                    if lab:
                        snaps.append(self.snapshot(lab))
                if monitor is not None and self.k % 10 == 0:
                    why = monitor(self)
                    if why:
                        return "abort", dict(t=round(self.k * DT, 3), why=why, snaps=snaps)
            if kind == "close":
                self.weld(ph[1], True)                # grasp="weld": rigid once the claw has closed (no-op otherwise)
        return "done", dict(snaps=snaps)

    def reobserve(self):
        """11-01: look at the pan again with the same camera model (observe_pose at this episode's noise, same rng stream) -> the
        estimate dict (dx, dy, yaw_deg relative to the nominal pose; fit_ok) or None when the episode was built without a camera.
        Robot-side sensing only; the truth is not read."""
        from . import observe_pose as O
        ctx = getattr(self, "_obs_ctx", None)
        if ctx is None:
            return None
        rdr = mujoco.Renderer(self.m, O.CAM["height"], O.CAM["width"])
        try:
            for _ in range(ctx["retries"]):
                est = O.estimate(self.m, self.d, self.p, ctx["tmpl"], ctx["noise"], ctx["rng"], renderer=rdr)
                if est["fit_ok"]:
                    break
        finally:
            rdr.close()
        return _fold_c2(est) if (self.obj == "pot" and est["fit_ok"]) else est

    def segment_phases(self, seg):
        push, u = self.plan["push"], self.u_aim
        if seg == "start":
            return [("hold", 0.5)]
        if seg == "pinch":
            if self.obj == "pot":
                return [("close", "left", 0.7), ("close", "right", 0.7), ("hold", 0.3)]
            return [("close", "left", 0.7), ("hold", 0.3)]
        if self.obj == "pot" and seg in ("push", "clamp"):
            return [("mark", f"{seg}_start")]        # pot: no push, no rim clamp (both handles are pinched)
        if seg == "push":
            return [("mark", "push_start"), ("move", {"left": -push * u}, 1.5), ("hold", 0.3)]
        if seg == "clamp":
            return [("mark", "clamp_start"), ("move", {"right": APPROACH_IN * RIGHT_APPROACH}, 1.0), ("close", "right", 0.7),
                    ("hold", 0.3)]
        if seg == "lift":
            return [("mark", "lift_start"), ("payload", True), ("move", {s: np.array([0, 0, LIFT]) for s in SIDES}, 1.5),
                    ("hold", 2.0)]
        raise ValueError(seg)

    def backup_phases(self, seg, first=True):
        ph = [("stop", 0.3)] if first else []        # design: backups run at half the forward speed (move_to minimum 0.8 s)
        mem = self.mem
        if seg == "lift":
            ph += [("move_to", {s: (lambda s=s: self.mem["lift_start"][s]) for s in SIDES}, 3.0, LIFT), ("hold", 0.5),
                   ("payload", False)]
            return ph + self.backup_phases("clamp", first=False)
        if seg == "clamp":
            ph += [("open", "right", 0.5), ("move_to", {"right": lambda: self.mem["clamp_start"]["right"]}, 2.0, APPROACH_IN),
                   ("hold", 0.2)]
            return ph + self.backup_phases("push", first=False)
        if seg == "push":
            if "push_start" in mem:
                ph += [("move_to", {"left": lambda: self.mem["push_start"]["left"]}, 3.0, self.plan["push"]), ("hold", 0.3)]
            return ph + self.backup_phases("pinch", first=False)
        if seg == "pinch":
            return ph + [("open", "left", 0.5), ("retreat", "left"), ("hold", 0.5)]
        raise ValueError(seg)

    # ---------------- boundary / end checks ----------------
    def stable_margin(self):
        """MODEL: how far [m] the TRUE total COM lies inside the pan-on-stove contact area (negative = outside)."""
        from .cert import model_check as MC
        m, d = self.m, self.d
        ids = [self.pan, self.lid] + ([self.hid] if self.hid else [])
        Mt = float(sum(m.body_mass[i] for i in ids))
        com = sum(m.body_mass[i] * d.xipos[i] for i in ids)[:2] / Mt
        centre = d.site_xpos[m.site("pan_centre").id][:2]
        P, planes = MC.support_polygon(centre, self.r_out, self.stove[0], self.stove[1])
        if not planes:
            return -1.0
        return float(min(h - n @ com for n, h in planes))

    def in_R(self, reach=True, version="v1"):
        ps, cs = self.pan_state(), self.contacts()
        from . import observe_pose as O
        yaw = O.true_pose(self.m, self.d, self.p, self.pos0, self.R0)["yaw_deg"]
        arm_v = max(float(np.max(np.abs(self.d.qvel[self.dofs[s]]))) for s in SIDES)
        released = all(v < 1e-6 for v in cs["grip_obj"].values())
        mg = dict(tilt=1 - ps["tilt"] / R["tilt_deg"], dz=1 - abs(ps["dz_mm"]) / R["dz_mm"], lid=1 - ps["lid_mm"] / R["lid_mm"],
                  spill=-1.0 if ps["spill"] else 1.0, rim=min(1.0, ps["rim_in"] / 0.01),
                  support=((np.mean(self._sup) if self._sup else cs["support"]) / self.W - R["support_frac"]) / (1 - R["support_frac"]),
                  pan_v=1 - ps["v"] / R["pan_speed"], arm_v=1 - arm_v / R["arm_speed"], yaw=1 - abs(yaw) / R["yaw_deg"],
                  released=1.0 if released else -1.0)
        if version == "v2":
            del mg["rim"], mg["yaw"]                  # R_V2 = at rest, stable, reachable (user 10-29): no yaw proxy, reach decides
            mg["stable"] = min(1.0, (self.stable_margin() - STABLE_MARGIN) / 0.01)
        if reach:
            ht = handle_point(self.m, self.d, self.p, self.plan["grasp_frac"])
            Rz = self.d.xmat[self.pan].reshape(3, 3) @ self.R0.T
            try:
                find_pregrasp(self.m, self.d, "left", ht, Rz @ LEFT_APPROACH, Rz @ LEFT_JAW, n_sol=4, seeds=24,
                              open_w=FINGER_OPEN if version == "v1" else REGRASP_OPEN)
                mg["reach"] = 1.0
            except RuntimeError:
                mg["reach"] = -1.0
        mg = {k: round(float(min(1.0, v)), 4) for k, v in mg.items()}
        la = self.m.jnt_dofadr[self.m.body(self.lid).jntadr[0]]           # diagnostic only (not in R): lid sliding speed; an
        lid_v = 1e3 * float(np.linalg.norm(self.d.qvel[la:la + 3] - self.d.qvel[self.pan_dadr:self.pan_dadr + 3]))   # off-seat lid
        return bool(min(mg.values()) > 0), mg, dict(yaw_deg=round(float(yaw), 2), lid_v_mm_s=round(lid_v, 3),  # creeps in MuJoCo
                                                    **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in ps.items()})

    def roa_check(self, n_kick=None, seed=0):
        """sampled region-of-attraction check at a boundary: do nothing (hold the current references and claw commands) for t_hold,
        then n_kick random velocity kicks to the pan and the arm joints, each followed by t_hold. Pass: pan back within drift_mm /
        tilt_deg of the boundary pose, pan speed < speed, and no entry into U."""
        rng = np.random.default_rng(seed)
        sn = self.snapshot()
        p0, R0 = self.d.xpos[self.pan].copy(), self.d.xmat[self.pan].reshape(3, 3).copy()
        out = []
        for i in range(1 + (ROA["n_kick"] if n_kick is None else n_kick)):
            self.restore(sn)
            if i:
                a = self.pan_dadr
                th = rng.uniform(0, 2 * np.pi)
                self.d.qvel[a:a + 3] += ROA["kick_pan"] * np.array([np.cos(th), np.sin(th), 0.0])
                self.d.qvel[a + 5] += rng.uniform(-1, 1) * ROA["kick_pan_w"]
                for s in SIDES:
                    self.d.qvel[self.dofs[s]] += rng.uniform(-1, 1, len(self.dofs[s])) * ROA["kick_arm"]
            self.reset_track()
            self.run_phases([("hold", ROA["t_hold"])])
            Rr = self.d.xmat[self.pan].reshape(3, 3) @ R0.T
            drift = float(np.linalg.norm(self.d.xpos[self.pan] - p0)) * 1e3
            dth = float(np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))))
            v = self.pan_state()["v"]
            ok = drift < ROA["drift_mm"] and dth < ROA["tilt_deg"] and v < ROA["speed"] and self.track["viol"] is None
            out.append(dict(kick=bool(i), drift_mm=round(drift, 2), dtilt_deg=round(dth, 2), v=round(v, 4), ok=bool(ok),
                            viol=self.track["viol"]))
        self.restore(sn)
        return dict(ok=all(o["ok"] for o in out), n=len(out), trials=out)

    def success(self, seg):
        """ground-truth task-level success of a segment, read at its end boundary (measured, not certified)."""
        ps, cs = self.pan_state(), self.contacts()
        t = cs["touch"]
        if seg == "pinch":
            return bool("pan" in t[("left", 1)] and "pan" in t[("left", 2)])
        if seg == "push":
            moved = float(-(self.d.xpos[self.pan] - self.sense["pan_at_push_start"]) @ self.u_true)
            self.sense["pan_moved_m"] = moved
            return bool(moved >= 0.8 * self.plan["push"] and "pan" in (t[("left", 1)] | t[("left", 2)]))
        if seg == "clamp":
            objs = t[("right", 1)] | t[("right", 2)]
            return bool(len(t[("right", 1)]) and len(t[("right", 2)]) and "pan" in objs)
        if seg == "lift":
            return bool(ps["dz_mm"] >= 0.8 * LIFT * 1e3 and cs["support"] < 0.1 and ps["tilt"] < 6.0 and not ps["spill"]
                        and ps["lid_mm"] < 5.0)
        raise ValueError(seg)

    def observe_boundary(self, seg):
        """robot-observable signals at a boundary (encoders + joint torques)."""
        o = dict(open_frac={s: round(self.gap(s), 4) for s in SIDES})
        if seg in ("pinch", "push"):          # yaw read from the arm: aimed yaw + rotation of the left claw since its pre-grasp (FK)
            Rr = self.d.xmat[self.gb["left"]].reshape(3, 3) @ self.Rg0.T
            o["yaw_arm_deg"] = round(self.aim_yaw + float(np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0]))), 3)
            from . import observe_pose as O
            o["truth_yaw_deg"] = round(float(O.true_pose(self.m, self.d, self.p, self.pos0, self.R0)["yaw_deg"]), 3)   # ground truth
            o["truth_yaw_rel_claw_deg"] = round(o["truth_yaw_deg"] - o["yaw_arm_deg"], 3)
        if seg == "push":
            o["tcp_push_m"] = round(float(-(self.tcp("left") - self.mem["push_start"]["left"]) @ self.u_aim), 4)
        if seg in ("clamp", "lift"):
            o["claw_fz_N"] = {s: round(float(self.claw_wrench(s)[2]), 3) for s in SIDES}
        if seg == "lift":
            o["load"] = load_estimate(self)
        return o


def _branch_rule(dt=BRANCH_DT, dt_hold=BRANCH_DT_HOLD, ev_gap=0.1, ev_max=6):
    st = dict(n=0, k=-10 ** 9)

    def rule(ep, kind, i, n):
        if ep._event_now and st["n"] < ev_max and (ep.k - st["k"]) * DT >= ev_gap:   # contact events, >= 0.1 s apart, <= 6
            st["n"] += 1
            st["k"] = ep.k
            return "event"
        step = int(round((dt_hold if kind in ("hold", "stop") else dt) / DT))
        if (i + 1) % step == 0 or i == n - 1:
            return kind
        return None
    return rule


def check_segment(ep, seg, dense=True, branch_dt=BRANCH_DT, roa=True, reach=True, monitor=None):
    """from ep's current boundary: run segment `seg` (with the runtime monitor, if any) and, from branch points inside it (and its
    end), its backup.  Returns the per-segment certificate record; leaves ep at the segment's end boundary (or the abort point)."""
    pre_k = ep.k
    ep.reset_track()
    ep.sense["pan_at_push_start"] = ep.d.xpos[ep.pan].copy()
    rule = _branch_rule(branch_dt, max(branch_dt, BRANCH_DT_HOLD)) if dense else _branch_rule(branch_dt, branch_dt)
    status, info = ep.run_phases(ep.segment_phases(seg), branch=rule, monitor=monitor)
    nominal = dict(status=status, viol=ep.track["viol"], margin=round(ep.track["margin"], 4), worst=ep.track["worst"],
                   tau_ratio=round(ep.track["tau_ratio"], 4), n_sat=ep.track["n_sat"], events=ep.track["events"], fl=ep.ctl.rec.summary(),
                   abort=info.get("t") and dict(t=info["t"], why=info["why"]))
    if status == "done":
        nominal["success"] = ep.success(seg)
        nominal["observed"] = ep.observe_boundary(seg)
    end = ep.snapshot("end")
    snaps = info["snaps"] + [end]
    backups = []
    for sn in snaps:
        ep.restore(sn)
        lid0 = round(ep.pan_state()["lid_mm"], 2)
        ep.reset_track()
        ep.run_phases(ep.backup_phases(seg))
        ok_R, mg, st = ep.in_R(reach=reach)
        backups.append(dict(t=round(sn["t"], 3), label=sn["label"], lid_mm_at_branch=lid0, safe=ep.track["viol"] is None, viol=ep.track["viol"],
                            margin=round(ep.track["margin"], 4), inR=ok_R, R_margin=mg, state=st,
                            tau_ratio=round(ep.track["tau_ratio"], 4), fl=ep.ctl.rec.summary()))
    ep.restore(end)
    rec = dict(seg=seg, t0=round(pre_k * DT, 3), t1=round(ep.k * DT, 3), nominal=nominal, n_backups=len(backups),
               backups=backups)
    rec["roa"] = ep.roa_check(seed=world_seed(seg)) if (roa and status == "done") else None
    safe = nominal["viol"] is None and all(b["safe"] for b in backups)
    rec["admissible_here"] = bool(safe and all(b["inR"] for b in backups) and (rec["roa"] is None or rec["roa"]["ok"]))
    fails = [b for b in backups if not (b["safe"] and b["inR"])]
    rec["counterexample"] = None
    if nominal["viol"]:
        rec["counterexample"] = dict(where="segment", t=nominal["viol"]["t"], viol=nominal["viol"])
    elif fails:
        rec["counterexample"] = dict(where="backup", t=fails[0]["t"], viol=fails[0]["viol"],
                                     R_fail=[k for k, v in fails[0]["R_margin"].items() if v <= 0])
    elif rec["roa"] is not None and not rec["roa"]["ok"]:
        rec["counterexample"] = dict(where="boundary_roa", t=rec["t1"])
    rec["margin"] = min([nominal["margin"]] + [b["margin"] for b in backups] + [min(b["R_margin"].values()) for b in backups])
    return rec


def boundary_ok(ep, seg, rec):
    """the robot's own (observable) check at the end of a segment: continue to the next segment only if it passes.
    Thresholds ASSUMED (set from the pinch/clamp geometry: handle 14 mm thick, rim stack ~45 mm)."""
    o = rec["nominal"].get("observed")
    if rec["nominal"]["status"] != "done" or o is None:
        return False
    if seg == "pinch":
        return 0.06 <= o["open_frac"]["left"] <= 0.30
    if seg == "push":
        return o["tcp_push_m"] >= 0.8 * ep.plan["push"] and 0.06 <= o["open_frac"]["left"] <= 0.30
    if seg == "clamp":
        return 0.20 <= o["open_frac"]["right"] <= 0.70
    return True


# runtime slip monitor for the lift (robot-observable: finger encoders + joint-torque claw forces), checked every 50 ms.
# ASSUMED thresholds, set from 5 calibration lifts (10-29: empty, 0.4 kg x2, 0.8 kg x2; contact-torque estimate): in the good lifts
# the left gap moved <= 0.6 mm and the right claw carried 0.44-0.72 of the load; in the overload slips the left gap opened 5-7 mm, the
# right share fell to 0.08-0.12 and the right gap collapsed 41 -> 1 mm.
MON = dict(open_L=0.03, open_R=0.04,   # opening-fraction changes (earlier form: 3 / 4 mm of gap sum = fraction x 100 mm)
            share_R=0.2, t_share=0.4)


def default_monitor(seg):
    if seg != "lift":
        return None
    st = {}

    def mon(ep):
        if not st:
            st.update(k0=ep.k, gL=ep.gap("left"), gR=ep.gap("right"))
            return None
        if ep.gap("left") - st["gL"] > MON["open_L"]:
            return "left claw opening (handle twisting in the claw)"
        if abs(ep.gap("right") - st["gR"]) > MON["open_R"]:
            return "right claw gap changed"
        if (ep.k - st["k0"]) * DT > MON["t_share"]:
            fl, fr = ep.claw_wrench("left")[2], ep.claw_wrench("right")[2]
            if fl + fr < -1.0 and fr / (fl + fr) < MON["share_R"]:
                return "right claw load share low (rim slipping)"
        return None
    return mon


def load_estimate(ep):
    """mass probe from joint torques while both claws hold the pan still in the air (contact-torque estimate, no offset).
    MEASURED in the calibration holds (10-29): empty 2.5-3.5 N vs W 2.99 N; 0.4 kg 6.7-6.95 N vs W 6.91 N (n = 2 worlds, not a bound)."""
    f = -(ep.claw_wrench("left")[2] + ep.claw_wrench("right")[2])
    return dict(total_N=round(float(f), 3), contents_kg=round(float(f / 9.81 - ep.m.body_mass[ep.pan] - ep.m.body_mass[ep.lid]), 3))


def certify_world(world, plan, observe="zed", obs_seed=0, dense=True, roa=True, reach=True, branch_dt=BRANCH_DT,
                  monitor=default_monitor, payload_mass=None):
    """library entry for one theta: the robot's policy (observe, re-aim, segment by segment, continue only if the observable
    boundary check passes) with every segment's backup checked from branch points inside it."""
    ep = Episode(world, observe=observe, obs_seed=obs_seed, payload_mass=payload_mass, **plan)
    out = dict(world=world, plan=plan, observe=observe, obs_seed=obs_seed, obs=ep.obs, refused=ep.refused, segments=[],
               bounds=BOUNDS)
    if ep.refused:                          # the robot does not start: the pan is untouched (in R if theta is)
        ok, mg, st = ep.in_R(reach=reach)
        out.update(B0=dict(inR=ok, R_margin=mg, state=st))
        return out
    ep.run_phases(ep.segment_phases("start"))
    out["B0"] = dict(roa=ep.roa_check(seed=1) if roa else None, pregrasp=getattr(ep, "pregrasp_margin", None))
    for seg in SEGMENTS:
        mon = monitor(seg) if monitor else None
        rec = check_segment(ep, seg, dense=dense, branch_dt=branch_dt, roa=roa, reach=reach, monitor=mon)
        rec["boundary_ok"] = boundary_ok(ep, seg, rec)
        out["segments"].append(rec)
        if not rec["boundary_ok"]:
            break
    return out
