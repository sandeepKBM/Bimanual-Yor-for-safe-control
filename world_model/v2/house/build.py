"""build.py -- load a MolmoSpaces house with plain MuJoCo 3.9 and compose YOR + tray + ball into it (Mode D, SIMULATION).

Design (docs/status/molmospaces_integration_plan.md section 3.3):
  * mjSpec only; nothing under YOR/ or scenes/ is edited, everything is built in memory.
  * YOR (scenes/yor_robot_derived.mjcf, the derived copy of the unmodified upstream) hangs under a MOCAP root body
    `yor_root`: moving the base = writing mocap_pos / mocap_quat.  No new degrees of freedom, so mpc_fl indexes
    the same joints by name.  The sim base stays an idealised kinematic base (no swerve model).
  * The tray (+ ball) is built from testbeds/b_tray_ball/tray_env.tray_xml (same geometry/params as the rig) as free
    bodies and is HELD BY THE ARMS WITH TWO mjEQ_WELD constraints (left/right gripper_base <-> tray), relpose
    captured at the hold pose.  NOT pinned kinematically (cf. contents.pin_cup).  Tray geoms keep contype/conaffinity
    1/1 and the arm bits (4|16, 8|32 / 8|16, 4|32) do not overlap them, so arm<->tray contacts are filtered out
    (same trick as sim_common.set_cube_arm_collision for a welded cube).  The weld is a geometric hold; the real
    pinch mechanics are not modelled.
  * "Furniture static": every joint of the house is deleted (free + hinge + slide), so clutter is fixed geometry.
  * Never imports YOR/robot/base.py or YOR/extra/*.  Never imported by tray_rep.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import mujoco

from . import assets as A

REPO = Path(__file__).resolve().parents[3]
YOR_MJCF = REPO / "scenes" / "yor_robot_derived.mjcf"
CANON_XML = REPO / "scenes" / "scene_yor_canonical.xml"


def yor_mjcf_text() -> str:
    """scenes/yor_robot_derived.mjcf as text for MjSpec.from_string.  Its meshdir is relative to scenes/ (so <include> and
    from_file work from any checkout); from_string has no file location, so the meshdir is made absolute here, in memory."""
    return YOR_MJCF.read_text().replace('meshdir="../YOR/', f'meshdir="{REPO.as_posix()}/YOR/', 1)
for _p in (str(REPO), str(REPO / "experiments" / "ab_smooth_vs_bangbang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CANON_GRAVITY = (0.0, 0.0, -9.81)   # the house ships -9.8; the rig uses -9.81 (needed for the 1e-6 tau-equality gate)
ARM_BITS = 0b1100                   # sim_common._LEFT_BIT | _RIGHT_BIT


# ----------------------------------------------------------------------------------------------------- house only
def load_house_spec(idx: int, static: bool = True) -> mujoco.MjSpec:
    A.extract_house(idx)
    sp = mujoco.MjSpec.from_file(str(A.make_stripped(idx)))
    if static:
        for j in list(sp.joints):
            sp.delete(j)
    return sp


def load_house(idx: int, static: bool = True):
    m = load_house_spec(idx, static).compile()
    return m, mujoco.MjData(m)


def model_hash(m: mujoco.MjModel) -> str:
    h = hashlib.sha1()
    for a in (m.geom_pos, m.geom_size, m.body_pos, m.body_mass, m.geom_friction):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.array([m.nbody, m.ngeom, m.njnt, m.nmesh, m.nv], np.int64).tobytes())
    return h.hexdigest()[:12]


def _in_poly(pt, poly):
    x, y = pt
    ins = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-18) + x0:
            ins = not ins
    return ins


def room_polys(idx: int):
    """Room floor polygons in MuJoCo xy (ProcTHOR json x,z -> MuJoCo x,y; verified on door body positions)."""
    out = []
    for r in A.spec_json(idx)["rooms"]:
        out.append((r["id"], r["roomType"], [(p["x"], p["z"]) for p in r["floorPolygon"]]))
    return out


GROUP_PHYS = np.array([0, 0, 0, 1, 1, 0], np.uint8)   # walls (g3) + collision meshes (g4)


def free_radius(m, d, xy, heights=(0.03, 0.15, 0.5, 1.0, 1.6), n_dir=36, cutoff=4.0):
    """Min horizontal free distance around xy over several heights (rays against groups 3+4)."""
    th = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    dirs1 = np.stack([np.cos(th), np.sin(th), np.zeros(n_dir)], -1)
    best = cutoff
    gid = np.zeros(n_dir, np.int32)
    dist = np.zeros(n_dir)
    for z in heights:
        pnt = np.array([xy[0], xy[1], z])
        mujoco.mj_multiRay(m, d, pnt, dirs1.ravel(), GROUP_PHYS, 1, -1, gid, dist, None, n_dir, cutoff)
        dd = np.where(gid >= 0, dist, cutoff)
        best = min(best, float(dd.min()))
    return best


def find_clear_spots(idx: int, m=None, d=None, step=0.25, top=5):
    """Candidate base spots: grid inside every room polygon, ranked by free radius. Returns list of dicts."""
    if m is None:
        m, d = load_house(idx, True)
        mujoco.mj_forward(m, d)
    cand = []
    for rid, rtype, poly in room_polys(idx):
        P = np.array(poly)
        for x in np.arange(P[:, 0].min() + 0.3, P[:, 0].max() - 0.3 + 1e-9, step):
            for y in np.arange(P[:, 1].min() + 0.3, P[:, 1].max() - 0.3 + 1e-9, step):
                if _in_poly((x, y), poly):
                    cand.append((rid, rtype, float(x), float(y)))
    res = []
    for rid, rtype, x, y in cand:
        res.append(dict(room=rid, type=rtype, x=x, y=y, free_r=free_radius(m, d, (x, y))))
    res.sort(key=lambda r: -r["free_r"])
    return res[:top]


# ----------------------------------------------------------------------------------------------------- compose
def _names(spec):
    out = set()
    for attr in ("bodies", "geoms", "joints", "sites", "cameras", "actuators", "meshes", "materials", "lights",
                 "sensors", "tendons", "equalities", "textures", "keys"):
        for o in getattr(spec, attr, []):
            if o.name:
                if not (attr == "bodies" and o.name == "world"):
                    out.add((attr, o.name))
    return out


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Composed:
    """Container for the composed model, with name/address helpers and hold-pose bookkeeping."""

    def __init__(self, model, data, info):
        self.m, self.d, self.info = model, data, info


def compose(idx: int, spawn_xy, yaw: float = 0.0, with_tray: bool = True, static: bool = True,
            dt: float | None = None, gravity=CANON_GRAVITY, ball: bool = True):
    """Build the composed model. `gravity` overrides the house value (pass None to keep the house's -9.8).
    Returns Composed. Raises on any name clash between house and robot/tray."""
    import sim_common as sc            # read-only import (experiments/ab_smooth_vs_bangbang)
    house = load_house_spec(idx, static)
    robot = mujoco.MjSpec.from_string(yor_mjcf_text())   # .mjcf ext has no decoder; meshdir made absolute in memory
    spawn_xy = np.asarray(spawn_xy, float)

    tray = None
    if with_tray:
        from world_model.v2.testbeds.b_tray_ball import tray_env as TE
        tray = mujoco.MjSpec.from_string(TE.tray_xml(TE.BallParams(), lip=False, home=(0.0, 0.0, 5.0)))
    clash = _names(house) & _names(robot)
    if tray is not None:
        clash |= _names(house) & _names(tray)
    if clash:
        raise RuntimeError(f"name clash between house val_{idx} and robot/tray: {sorted(clash)[:10]}")

    root = house.worldbody.add_body(name="yor_root", mocap=True, pos=[spawn_xy[0], spawn_xy[1], 0.0],
                                    quat=yaw_quat(yaw).tolist())
    fr = root.add_frame()
    fr.attach_body(robot.body("base"), "", "")
    if tray is not None:
        f2 = house.worldbody.add_frame(pos=[spawn_xy[0], spawn_xy[1], 5.0])
        f2.attach_body(tray.body("tray"), "", "")
        f3 = house.worldbody.add_frame(pos=[spawn_xy[0], spawn_xy[1], 6.0])
        if ball:
            f3.attach_body(tray.body("ball"), "", "")
        for nm in ("left_arm_gripper_base", "right_arm_gripper_base"):
            house.add_equality(name=f"hold_{nm.split('_')[0]}", type=mujoco.mjtEq.mjEQ_WELD,
                               objtype=mujoco.mjtObj.mjOBJ_BODY, name1=nm, name2="tray", active=0)

    # arm collision bits (sim_common scheme); house-side conaffinity completed after compile (below)
    sc._patch_real_collision_geoms(house)

    m = house.compile()
    # house-side arm bits: only house geoms (everything not in the robot/tray body subtrees)
    own = _robot_tray_bodies(m)
    n_or = 0
    for gi in range(m.ngeom):
        if int(m.geom_bodyid[gi]) in own:
            continue
        if (m.geom_contype[gi] or m.geom_conaffinity[gi]):
            if (m.geom_conaffinity[gi] & ARM_BITS) != ARM_BITS:
                m.geom_conaffinity[gi] |= ARM_BITS
                n_or += 1
    if dt is not None:
        m.opt.timestep = dt
    if gravity is not None:
        m.opt.gravity[:] = gravity
    d = mujoco.MjData(m)
    info = dict(house=idx, spawn=spawn_xy.tolist(), yaw=yaw, n_conaff_or=n_or, dt=float(m.opt.timestep),
                gravity=m.opt.gravity.tolist(), nbody=m.nbody, ngeom=m.ngeom, nv=m.nv, nq=m.nq, nu=m.nu,
                nmocap=m.nmocap, model_hash=model_hash(m), has_tray=with_tray)
    return Composed(m, d, info)


def _robot_tray_bodies(m) -> set:
    """Body ids in the subtree of yor_root + tray + ball."""
    roots = [m.body("yor_root").id]
    for nm in ("tray", "ball"):
        try:
            roots.append(m.body(nm).id)
        except KeyError:
            pass
    out = set()
    for b in range(m.nbody):
        x = b
        while x > 0:
            if x in roots:
                out.add(b)
                break
            x = int(m.body_parentid[x])
    return out


# ----------------------------------------------------------------------------------------------------- hold pose
LEFT_J = [f"left_arm_joint{i}" for i in range(1, 7)]
RIGHT_J = [f"right_arm_joint{i}" for i in range(1, 7)]


def _jaddr(m, names):
    ids = [m.joint(n).id for n in names]
    return np.array([m.jnt_qposadr[i] for i in ids]), np.array([m.jnt_dofadr[i] for i in ids]), ids


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


_IK = {}


def _ik_model():
    if 'm' not in _IK:
        _IK['m'] = mujoco.MjModel.from_xml_path(str(CANON_XML))   # robot-only kinematics (read-only file)
    return _IK['m']


def solve_hold_ik(hand_world, approach_world, jaw_world, side, iters=300, seeds=60, seed=0, collect=0):
    """Damped-least-squares IK for ONE arm in the ROBOT-LOCAL frame (canonical robot-only model; the composed
    model's arm kinematics are the same bodies): ee_grasp site at hand_world, approach (gripper_base z) along
    approach_world, finger-closing axis along +-jaw_world. Returns (q(6), err_dict); with collect=N returns a
    list of up to N distinct converged (q, err) solutions (random restarts) instead."""
    m = _ik_model()
    pre = f"{side}_arm_"
    qadr, dadr, jids = _jaddr(m, [f"{pre}joint{i}" for i in range(1, 7)])
    site = m.site(f"{pre}ee_grasp").id
    gb = m.body(f"{pre}gripper_base").id
    j7 = m.joint(f"{pre}joint7").id
    lo = m.jnt_range[jids, 0].copy()
    hi = m.jnt_range[jids, 1].copy()
    rng = np.random.default_rng(seed)
    dd = mujoco.MjData(m)
    best = (1e9, None, None)
    sols = []
    q_home = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78 if side == "left" else -0.78])
    for s in range(seeds):
        q = q_home.copy() if s == 0 else np.clip(q_home + rng.normal(0, 0.5, 6), lo, hi)
        for it in range(iters):
            dd.qpos[qadr] = q
            mujoco.mj_kinematics(m, dd)
            mujoco.mj_comPos(m, dd)
            p = dd.site_xpos[site]
            Rg = dd.xmat[gb].reshape(3, 3)
            # approach axis = gripper_base z (the ee_grasp site sits 0.1 m along it)
            app = Rg[:, 2]
            jaw = dd.xaxis[j7]
            e_p = hand_world - p
            e_a = np.cross(app, approach_world)
            sgn = 1.0 if jaw @ jaw_world >= 0 else -1.0
            e_j = np.cross(jaw, sgn * jaw_world)
            err = np.concatenate([e_p, -0.5 * e_a, -0.3 * e_j])   # Delta(app x t) must be -(app x t)
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_a) < 1e-3 and np.linalg.norm(e_j) < 2e-2:
                break
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, dd, jp, jr, site)
            # approach / jaw axes rotate with the angular velocity of the gripper body: d axis = w x axis
            Ja = skew(approach_world) @ skew(app) @ jr[:, dadr]          # d(app x t)/dq with app' = w x app
            Jj = skew(sgn * jaw_world) @ skew(jaw) @ jr[:, dadr]
            J = np.vstack([jp[:, dadr], 0.5 * Ja, 0.3 * Jj])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(9), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        dd.qpos[qadr] = q
        mujoco.mj_kinematics(m, dd)
        p = dd.site_xpos[site]
        Rg = dd.xmat[gb].reshape(3, 3)
        jaw = dd.xaxis[j7]
        e = (np.linalg.norm(hand_world - p), np.linalg.norm(np.cross(Rg[:, 2], approach_world)),
             np.linalg.norm(np.cross(jaw, (1 if jaw @ jaw_world >= 0 else -1) * jaw_world)))
        score = e[0] + e[1] + 0.3 * e[2]
        if score < best[0]:
            best = (score, q.copy(), e)
        if e[0] < 2e-4 and e[1] < 2e-3 and e[2] < 3e-2:
            if not collect:
                break
            if all(np.abs(q - o[0]).max() > 0.05 for o in sols):
                sols.append((q.copy(), dict(pos_err=float(e[0]), app_err=float(e[1]), jaw_err=float(e[2]))))
            if len(sols) >= collect:
                break
    if collect:
        return sols
    return best[1], dict(pos_err=best[2][0], app_err=best[2][1], jaw_err=best[2][2])


HOLD_MODES = {   # mode -> (left approach, right approach, jaw axis) in the robot frame
    "side": ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),    # hands reach in from the sides, jaws vertical
    "fwd": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}


def tray_hold_targets(tray_local, mode="side", inset=0.012, hx=0.15, hy=0.10, across=True):
    """Robot-frame hand targets pinching the two opposite rims (left hand = +y).  across=True: tray long axis lies
    across the robot (tray yaw = base yaw + 90 deg), hands on the SHORT rims (|y| = hx - inset): hand separation
    0.276 m.  across=False: long axis forward, hands on the long rims (|y| = hy - inset): 0.176 m, which puts the two
    finger keypoints inside FL-MPC's arm-arm clearance margin (h = 0.002 < H_MARGIN 0.02) -> QP infeasible at hold."""
    c = np.array(tray_local, float)
    al, ar, jw = (np.array(v, float) for v in HOLD_MODES[mode])
    h = (hx if across else hy) - inset
    return c, c + [0.0, h, 0.0], c + [0.0, -h, 0.0], al, ar, jw


def set_hold_state(C: Composed, tray_local=(0.45, -0.21, 0.85), mode="side", finger=0.0175, across=True):
    """Pose the arms by IK, put the tray level between the hands and the ball at rest on it, then capture the weld
    relposes (zero initial constraint violation) and activate both welds. Returns dict of IK errors."""
    m, d = C.m, C.d
    sp = np.array(C.info["spawn"]); yaw = C.info["yaw"]
    R = _rotz(yaw)
    base = np.array([sp[0], sp[1], 0.0])
    d.mocap_pos[0] = base
    d.mocap_quat[0] = yaw_quat(yaw)
    c, hl, hr, al, ar, jw = tray_hold_targets(tray_local, mode, across=across)
    solsL = solve_hold_ik(hl, al, jw, "left", collect=8, seeds=200)
    solsR = solve_hold_ik(hr, ar, jw, "right", collect=8, seeds=200)
    la, _, _ = _jaddr(m, LEFT_J); ra, _, _ = _jaddr(m, RIGHT_J)
    ql = qr = None
    for (q1, e1) in solsL:                # first IK pair with zero arm-arm / self contacts in the composed model
        for (q2, e2) in solsR:
            d.qpos[la] = q1; d.qpos[ra] = q2
            mujoco.mj_forward(m, d)
            if d.ncon == 0:
                ql, el, qr, er = q1, e1, q2, e2
                break
        if ql is not None:
            break
    C.info["ik_pairs_tried"] = (len(solsL), len(solsR))
    if ql is None:
        raise RuntimeError(f"no contact-free IK pair for hold pose (solutions L/R: {len(solsL)}/{len(solsR)})")
    tray_c = base + R @ c
    d.qpos[la] = ql
    d.qpos[ra] = qr
    for side in ("left", "right"):
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_joint7").id]] = finger
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_joint8").id]] = -finger
    if C.info["has_tray"]:
        ta = m.jnt_qposadr[m.joint("tray_free").id]
        d.qpos[ta:ta + 3] = tray_c
        d.qpos[ta + 3:ta + 7] = yaw_quat(yaw + (np.pi / 2 if across else 0.0))
        from world_model.v2.testbeds.b_tray_ball import tray_env as TE
        bp = TE.BallParams()
        ba = m.jnt_qposadr[m.joint("ball_free").id]
        d.qpos[ba:ba + 3] = tray_c + np.array([0.0, 0.0, bp.radius + 0.0005])
        d.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    if C.info["has_tray"]:
        for nm in ("left", "right"):
            capture_weld(m, d, f"hold_{nm}", f"{nm}_arm_gripper_base", "tray")
        mujoco.mj_forward(m, d)
    C.info["hold_q"] = dict(left=ql.tolist(), right=qr.tolist())
    C.info["finger"] = finger
    C.info["hold_mode"] = mode
    C.info["tray_across"] = across
    C.info["tray_local"] = list(tray_local)
    C.info["ik_err"] = {k: {a: float(b) for a, b in v.items()} for k, v in dict(left=el, right=er).items()}
    return C.info["ik_err"]


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def capture_weld(m, d, eq_name, body1, body2):
    """Set the weld target to body2's CURRENT pose in body1's frame and activate it (no snap).
    eq_data[0:3]=anchor in body2 frame (0), [3:6]=relpos in body1 frame, [6:10]=relquat (b1q*relquat==b2q), [10]=torquescale."""
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    b1, b2 = m.body(body1).id, m.body(body2).id
    R1 = d.xmat[b1].reshape(3, 3)
    rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
    q1c = d.xquat[b1] * np.array([1, -1, -1, -1])
    rq = _qmul(q1c, d.xquat[b2])
    m.eq_data[eq, :] = 0.0
    m.eq_data[eq, 3:6] = rel
    m.eq_data[eq, 6:10] = rq / np.linalg.norm(rq)
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


def disable_arm_servos(m):
    """Zero the 12 arm position servos (torque via qfrc_applied), exactly as sim_common.disable_*_position_servos."""
    for side in ("left", "right"):
        for i in range(1, 7):
            a = m.actuator(f"{side}_arm_joint{i}_pos").id
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_biasprm[a, :] = 0.0
