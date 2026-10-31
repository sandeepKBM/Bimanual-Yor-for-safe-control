import numpy as np
import mujoco

from world_model.v2.nero.pan import covered as CV, handle_grasp as H


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
    assert m.actuator_forcerange[a][1] == 10.0                                # MJCF limit unchanged
