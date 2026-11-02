import numpy as np
import mujoco

from world_model.v2.nero import config as C
from world_model.v2.nero.pan import covered as CV, handle_grasp as H, scene as S


def _m():
    return CV.compose(CV.CoveredParams(placement="none"))


def test_set_fingers_opens_symmetrically():
    m = _m()
    d = mujoco.MjData(m)
    H._set_fingers(m, d, "left", H.FINGER_OPEN)
    mujoco.mj_forward(m, d)
    q1 = d.qpos[m.jnt_qposadr[m.joint("left_arm_gripper_joint1").id]]
    q2 = d.qpos[m.jnt_qposadr[m.joint("left_arm_gripper_joint2").id]]
    assert np.isclose(q1, H.FINGER_OPEN) and np.isclose(q2, -H.FINGER_OPEN)   # joint2 range is [-0.05, 0]
    f1 = d.geom_xpos[m.geom("left_arm_finger1_col").id]
    f2 = d.geom_xpos[m.geom("left_arm_finger2_col").id]
    assert np.linalg.norm(f1 - f2) > 0.05                                     # fingers apart, not moved together


def test_grip_force_mode_is_constant_force():
    m = _m()
    a = m.actuator("left_arm_gripper_pos").id
    H._grip_force_mode(m, a, 10.0)
    assert np.allclose(m.actuator_gainprm[a, :3], (1, 0, 0)) and np.allclose(m.actuator_biasprm[a, :3], 0)
    assert np.allclose(m.actuator_ctrlrange[a], (-10, 10))
    assert m.actuator_forcerange[a][1] == C.claw_default()                    # compose limit unchanged (40 N since 11-01)


def test_sim_defaults_claw_and_substeps():
    """11-01: pan scenes compose with 40 N claws; scene.step = substeps_default() physics steps of dt/n (torque held), dt restored."""
    m = _m()
    for s in C.SIDES:
        assert m.actuator_forcerange[m.actuator(f"{s}_arm_gripper_pos").id][1] == C.claw_default()
    assert C.claw_default() == (C.GRIPPER_EFFORT if C.LEGACY_CLAW else 40.0)
    n = C.substeps_default()
    d1, d2 = mujoco.MjData(m), mujoco.MjData(m)
    dt = m.opt.timestep
    S.step(m, d1)
    assert m.opt.timestep == dt
    m.opt.timestep = dt / n
    for _ in range(n):
        mujoco.mj_step(m, d2)
    m.opt.timestep = dt
    assert np.allclose(d1.qpos, d2.qpos) and np.isclose(d1.time, d2.time)
