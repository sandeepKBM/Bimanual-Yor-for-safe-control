"""compose.py -- compose YOR (+ tray, + 2-stage lift) into the office scene.  SIMULATION only.

`build.compose` / `lift.compose_lift` are keyed on a ProcTHOR house index (they call `load_house_spec(idx)`).  This module
duplicates their minimal logic against `scene.load_office_spec()` (build.py / lift.py are NOT edited and are reused for everything
else: `Composed`, `_names`, `_robot_tray_bodies`, `yaw_quat`, `model_hash`, `set_hold_state`, `capture_weld`,
`disable_arm_servos`, `lift.patch_robot_spec`, `lift._patch_house_spec`, `lift.set_lift`, `lift.lift_state`).  The returned
`build.Composed` is therefore a drop-in for the house pipeline: `build.set_hold_state(C)`, `lift.set_lift(C, h)`, phase1.Ctl, ...
The hold pose is solved in the robot frame, so it is scene independent.

Never imports YOR/robot/base.py or YOR/extra/*.  The sim base is the same idealised kinematic mocap root as in the houses.
"""
from __future__ import annotations

import math

import numpy as np
import mujoco

from .. import build as B
from .. import lift as L
from . import scene as S

HEAD_POS_LIFT0 = (0.05, -0.209550, 1.15)    # ASSUMED torso/mast head (robot frame, lift 0): perception_bridge.HEAD_NEW_POS_LIFT0 (proposed, confirm on hardware)
HEAD_PITCH_DEG = 36.3                       # pitch down (perception_bridge.HEAD_PITCH_CURRENT)
HEAD_FOVY = 55.0                            # depth_sim head spec


def _head_cam(root, pos=HEAD_POS_LIFT0, pitch_deg=HEAD_PITCH_DEG, fovy=HEAD_FOVY, name="head_cam"):
    """Camera on the mocap root, looking along robot +x, pitched `pitch_deg` down (rides with the base, NOT with the lift)."""
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    up = np.array([math.sin(p), 0.0, math.cos(p)])
    z = -fwd
    x = np.cross(up, z)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, up, z], 1).ravel())
    root.add_camera(name=name, pos=list(pos), quat=q.tolist(), fovy=fovy)


def compose_office(spawn_xy=None, yaw=None, with_tray: bool = True, lift: bool = True, mode: str = "locked", dt: float | None = None,
                   gravity=B.CANON_GRAVITY, ball: bool = True, head_cam: bool = True, head_pitch_deg: float = HEAD_PITCH_DEG,
                   head_fovy: float = HEAD_FOVY, spec: mujoco.MjSpec | None = None) -> B.Composed:
    """YOR under a mocap root `yor_root` in the office, tray (+ball) held by two weld constraints (inactive until set_hold_state).
    lift=True (default; the user's robot) adds the two telescoping slide joints (mode 'locked' | 'actuated', see lift.py), lift=False
    = build.compose equivalent.  spawn_xy/yaw default to waypoint lab_start.  Names are checked for clashes."""
    import sim_common as sc
    meta = S.office_meta()
    if spawn_xy is None:
        spawn_xy = meta["waypoints"]["lab_start"]["xy"]
    if yaw is None:
        yaw = meta["waypoints"]["lab_start"]["heading"]
    spawn_xy = np.asarray(spawn_xy, float)
    house = spec if spec is not None else S.load_office_spec()
    robot = mujoco.MjSpec.from_string(B.yor_mjcf_text())
    if lift:
        L.patch_robot_spec(robot)
    tray = None
    if with_tray:
        from world_model.v2.testbeds.b_tray_ball import tray_env as TE
        tray = mujoco.MjSpec.from_string(TE.tray_xml(TE.BallParams(), lip=False, home=(0.0, 0.0, 5.0)))
    clash = B._names(house) & B._names(robot)
    if tray is not None:
        clash |= B._names(house) & B._names(tray)
    if clash:
        raise RuntimeError(f"name clash between office and robot/tray: {sorted(clash)[:10]}")
    root = house.worldbody.add_body(name="yor_root", mocap=True, pos=[spawn_xy[0], spawn_xy[1], 0.0], quat=B.yaw_quat(yaw).tolist())
    root.add_frame().attach_body(robot.body("base"), "", "")
    if head_cam:
        _head_cam(root, pitch_deg=head_pitch_deg, fovy=head_fovy)
    if tray is not None:
        house.worldbody.add_frame(pos=[spawn_xy[0], spawn_xy[1], 5.0]).attach_body(tray.body("tray"), "", "")
        if ball:
            house.worldbody.add_frame(pos=[spawn_xy[0], spawn_xy[1], 6.0]).attach_body(tray.body("ball"), "", "")
        for nm in ("left_arm_gripper_base", "right_arm_gripper_base"):
            house.add_equality(name=f"hold_{nm.split('_')[0]}", type=mujoco.mjtEq.mjEQ_WELD,
                               objtype=mujoco.mjtObj.mjOBJ_BODY, name1=nm, name2="tray", active=0)
    if lift:
        L._patch_house_spec(house, mode)
    sc._patch_real_collision_geoms(house)
    m = house.compile()
    own = B._robot_tray_bodies(m)
    n_or = 0
    for gi in range(m.ngeom):
        if int(m.geom_bodyid[gi]) in own:
            continue
        if (m.geom_contype[gi] or m.geom_conaffinity[gi]) and (m.geom_conaffinity[gi] & B.ARM_BITS) != B.ARM_BITS:
            m.geom_conaffinity[gi] |= B.ARM_BITS
            n_or += 1
    if dt is not None:
        m.opt.timestep = dt
    if gravity is not None:
        m.opt.gravity[:] = gravity
    d = mujoco.MjData(m)
    info = dict(house="office", spawn=spawn_xy.tolist(), yaw=float(yaw), n_conaff_or=n_or, dt=float(m.opt.timestep),
                gravity=m.opt.gravity.tolist(), nbody=m.nbody, ngeom=m.ngeom, nv=m.nv, nq=m.nq, nu=m.nu, nmocap=m.nmocap,
                model_hash=B.model_hash(m), has_tray=with_tray)
    if lift:
        info.update(lift_mode=mode, lift_h=0.0)
    return B.Composed(m, d, info)


def new_hold_office(spawn_xy=None, yaw=None, dt=0.002, lift_h=0.0, mode="locked", with_tray=True, lift=True, gravity=B.CANON_GRAVITY,
                    servos_off=False, **kw) -> B.Composed:
    """phase1.new_hold / lift.new_hold_lift equivalent: compose, solve the IK hold pose, weld the tray, finger ctrl, set lift height.
    servos_off=False keeps the arm position servos ON with ctrl = hold_q (self-holding: use for smoke/render);
    servos_off=True zeroes them (torque via qfrc_applied, for FL-MPC drivers, as phase1/lift do)."""
    C = compose_office(spawn_xy, yaw, with_tray=with_tray, lift=lift, mode=mode, dt=dt, gravity=gravity, **kw)
    B.set_hold_state(C)
    m, d = C.m, C.d
    fin = C.info["finger"]
    for side in ("left", "right"):
        d.ctrl[m.actuator(f"{side}_arm_joint7_pos").id] = fin
        d.ctrl[m.actuator(f"{side}_arm_joint8_pos").id] = -fin
    if servos_off:
        B.disable_arm_servos(m)
    else:
        for side, key in (("left", "left"), ("right", "right")):
            for i, q in enumerate(C.info["hold_q"][key], 1):
                d.ctrl[m.actuator(f"{side}_arm_joint{i}_pos").id] = q
    if lift:
        L.set_lift(C, lift_h)
    else:
        mujoco.mj_forward(m, d)
    return C


def set_base_pose(C: B.Composed, x: float, y: float, yaw: float, forward: bool = True, carry_tray: bool = True):
    """Move the mocap base (and the welded tray/ball, which are free bodies held only by constraints) to a new pose.
    The tray/ball are teleported rigidly with the base so the weld relpose stays satisfied."""
    m, d = C.m, C.d
    old = d.mocap_pos[0].copy()
    oyaw = 2 * math.atan2(d.mocap_quat[0][3], d.mocap_quat[0][0])
    new = np.array([x, y, 0.0])
    dth = yaw - oyaw
    Rz = B._rotz(dth)
    d.mocap_pos[0] = new
    d.mocap_quat[0] = B.yaw_quat(yaw)
    if carry_tray and C.info.get("has_tray"):
        for jn in ("tray_free", "ball_free"):
            if jn == "ball_free":
                try:
                    m.joint(jn)
                except KeyError:
                    continue
            a = m.jnt_qposadr[m.joint(jn).id]
            p = d.qpos[a:a + 3]
            d.qpos[a:a + 3] = new + Rz @ (p - old)
            q = d.qpos[a + 3:a + 7]
            d.qpos[a + 3:a + 7] = B._qmul(B.yaw_quat(dth), q)
    d.qvel[:] = 0.0
    C.info["spawn"] = [float(x), float(y)]
    C.info["yaw"] = float(yaw)
    if forward:
        mujoco.mj_forward(m, d)
