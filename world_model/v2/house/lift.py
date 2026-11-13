"""lift.py -- lift-ENABLED variant of the composed house model, patched IN MEMORY (MjSpec).  SIMULATION ONLY.

scenes/yor_robot_derived.mjcf has the YOR 2-stage telescoping lift commented out (joints Slider_1 on Lift_Middle, Slider_2 on
Lift_Top, tendon `lift`, equality Slider_1 = Slider_2, actuator `Lift`).  The file is NOT edited.  This module re-creates the
commented definitions with the values of `default class="telescope"`:
    slide joint, axis z, range 0..0.208 per stage, damping 50, stiffness 0.02, armature 0.01, frictionloss 1.5
    actuator: ctrlrange 0..0.416, gainprm 150, biasprm 0 -100 -10, forcerange +-56   (the commented line says
    `<position ... tendon="lift" class="telescope"/>`; the class holds a <general>, so it is rebuilt as a general actuator)
Both arms are children of Lift_Top, so the lift translates the whole arm block along world z (VERIFIED: joint world axis = (0,0,1)).

Modes
  'locked'   (default for studies) both sliders held at height/2 by mjEQ_JOINT equalities; `set_lift(C, h)` changes the height
             between runs (qpos, equality target, tray + ball z shifted so the weld relposes stay satisfied).
  'actuated' tendon `lift` (Slider_1 + Slider_2), equality Slider_1 = Slider_2, actuator `Lift` (the commented-out definition).
`compose_lift` duplicates build.compose (build.py is not edited) with the patched robot spec.  Never imports YOR/robot or YOR/extra.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import build as B

STAGE_RANGE = (0.0, 0.208)
H_MAX = 0.416
TELESCOPE = dict(damping=50.0, stiffness=0.02, armature=0.01, frictionloss=1.5)
LIFT_ACT = dict(ctrlrange=(0.0, H_MAX), gainprm=150.0, biasprm=(0.0, -100.0, -10.0), forcerange=(-56.0, 56.0))


def patch_robot_spec(robot: mujoco.MjSpec):
    """add the two slide joints (telescope class values) to the robot spec."""
    for body, name in (("Lift_Middle", "Slider_1"), ("Lift_Top", "Slider_2")):
        robot.body(body).add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=list(STAGE_RANGE),
                                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, **TELESCOPE)


def _patch_house_spec(house: mujoco.MjSpec, mode: str):
    if mode == "locked":
        for i in (1, 2):
            eq = house.add_equality(name=f"lift_lock_{i}", type=mujoco.mjtEq.mjEQ_JOINT, name1=f"Slider_{i}", active=1)
            eq.data[:5] = 0.0
            eq.solref = [0.005, 1.0]
    elif mode == "actuated":
        t = house.add_tendon(name="lift")
        t.wrap_joint("Slider_1", 1.0)
        t.wrap_joint("Slider_2", 1.0)
        eq = house.add_equality(name="lift_couple", type=mujoco.mjtEq.mjEQ_JOINT, name1="Slider_1", name2="Slider_2", active=1)
        eq.data[:5] = [0.0, 1.0, 0.0, 0.0, 0.0]
        a = house.add_actuator(name="Lift", target="lift", trntype=mujoco.mjtTrn.mjTRN_TENDON,
                               gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                               ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE, forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
                               ctrlrange=list(LIFT_ACT["ctrlrange"]), forcerange=list(LIFT_ACT["forcerange"]))
        a.gainprm[0] = LIFT_ACT["gainprm"]
        a.biasprm[:3] = LIFT_ACT["biasprm"]
    else:
        raise KeyError(mode)


def compose_lift(idx: int, spawn_xy, yaw: float = 0.0, with_tray: bool = True, static: bool = True, dt: float | None = None,
                 gravity=B.CANON_GRAVITY, mode: str = "locked") -> B.Composed:
    """build.compose with the lift joints added (see module doc).  The lift starts at 0 (retracted)."""
    import sim_common as sc
    house = B.load_house_spec(idx, static)
    robot = mujoco.MjSpec.from_string(B.yor_mjcf_text())
    patch_robot_spec(robot)
    spawn_xy = np.asarray(spawn_xy, float)
    tray = None
    if with_tray:
        from world_model.v2.testbeds.b_tray_ball import tray_env as TE
        tray = mujoco.MjSpec.from_string(TE.tray_xml(TE.BallParams(), lip=False, home=(0.0, 0.0, 5.0)))
    clash = B._names(house) & B._names(robot)
    if tray is not None:
        clash |= B._names(house) & B._names(tray)
    if clash:
        raise RuntimeError(f"name clash: {sorted(clash)[:10]}")
    root = house.worldbody.add_body(name="yor_root", mocap=True, pos=[spawn_xy[0], spawn_xy[1], 0.0], quat=B.yaw_quat(yaw).tolist())
    root.add_frame().attach_body(robot.body("base"), "", "")
    if tray is not None:
        house.worldbody.add_frame(pos=[spawn_xy[0], spawn_xy[1], 5.0]).attach_body(tray.body("tray"), "", "")
        house.worldbody.add_frame(pos=[spawn_xy[0], spawn_xy[1], 6.0]).attach_body(tray.body("ball"), "", "")
        for nm in ("left_arm_gripper_base", "right_arm_gripper_base"):
            house.add_equality(name=f"hold_{nm.split('_')[0]}", type=mujoco.mjtEq.mjEQ_WELD,
                               objtype=mujoco.mjtObj.mjOBJ_BODY, name1=nm, name2="tray", active=0)
    _patch_house_spec(house, mode)
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
    info = dict(house=idx, spawn=spawn_xy.tolist(), yaw=yaw, n_conaff_or=n_or, dt=float(m.opt.timestep), gravity=m.opt.gravity.tolist(),
                nbody=m.nbody, ngeom=m.ngeom, nv=m.nv, nq=m.nq, nu=m.nu, nmocap=m.nmocap, model_hash=B.model_hash(m),
                has_tray=with_tray, lift_mode=mode, lift_h=0.0)
    return B.Composed(m, d, info)


def lift_state(C: B.Composed):
    """(Slider_1 + Slider_2) as currently in qpos."""
    m, d = C.m, C.d
    return float(sum(d.qpos[m.jnt_qposadr[m.joint(f"Slider_{i}").id]] for i in (1, 2)))


def set_lift(C: B.Composed, h: float, forward: bool = True):
    """Set the lift to total height h in [0, 0.416] (each stage h/2).  Tray + ball (free joints, welded to the grippers) are shifted
    by the height change so the weld relposes stay satisfied.  Locked mode: also moves the equality targets."""
    m, d = C.m, C.d
    if not (0.0 <= h <= H_MAX + 1e-9):
        raise ValueError(h)
    dz = h - C.info.get("lift_h", 0.0)
    for i in (1, 2):
        d.qpos[m.jnt_qposadr[m.joint(f"Slider_{i}").id]] = h / 2
        if C.info.get("lift_mode") == "locked":
            m.eq_data[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"lift_lock_{i}"), 0] = h / 2
    if C.info.get("has_tray"):
        for jn in ("tray_free", "ball_free"):
            d.qpos[m.jnt_qposadr[m.joint(jn).id] + 2] += dz
    d.qvel[:] = 0.0
    C.info["lift_h"] = float(h)
    if forward:
        mujoco.mj_forward(m, d)


def new_hold_lift(house, spawn, yaw, dt, lift_h=0.0, mode="locked", with_tray=True, gravity=B.CANON_GRAVITY):
    """phase1.new_hold with the lift variant: hold pose solved at lift 0 (the arm joint angles are lift-independent), then raised."""
    C = compose_lift(house, spawn, yaw=yaw, with_tray=with_tray, dt=dt, gravity=gravity, mode=mode)
    B.set_hold_state(C)
    B.disable_arm_servos(C.m)
    fin = C.info["finger"]
    for side in ("left", "right"):
        C.d.ctrl[C.m.actuator(f"{side}_arm_joint7_pos").id] = fin
        C.d.ctrl[C.m.actuator(f"{side}_arm_joint8_pos").id] = -fin
    set_lift(C, lift_h)
    return C
