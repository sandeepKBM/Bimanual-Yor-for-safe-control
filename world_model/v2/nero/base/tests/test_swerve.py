"""swerve base: kinematics checks without contact dynamics (fast)."""
import numpy as np
import mujoco

from world_model.v2.nero import build_variant as BV
from world_model.v2.nero.base import swerve as SW, scene as BS


def _sb():
    m = mujoco.MjModel.from_xml_string(BV.variant_xml(base="swerve"))
    d = mujoco.MjData(m)
    return m, d, SW.SwerveBase(m, d)


def test_fixed_variant_unchanged_flag():
    assert "floating_base" not in BV.variant_xml()
    assert "floating_base" in BV.variant_xml(base="swerve")


def test_geometry():
    m, d, sb = _sb()
    assert abs(sb.R_wheel - 0.0508) < 1e-3
    assert np.allclose(np.abs(sb.r), [[0.1048, 0.1556]] * 4, atol=1e-3)


def test_ik_pure_rotation_is_tangent_and_straight_is_aligned():
    m, d, sb = _sb()
    th, qd = sb.module_cmd(np.array([0.0, 0.0, 0.4]))
    for i in range(4):                       # steer angle (mod pi) is the tangent direction of the circle about the centre
        tang = np.arctan2(sb.r[i][0], -sb.r[i][1])
        assert abs(np.sin(th[i] - tang)) < 1e-6
    th, qd = sb.module_cmd(np.array([0.2, 0.0, 0.0]))
    assert np.allclose(th, 0.0) and np.allclose(qd, 0.2 / sb.R_wheel)


def test_wheel_reversal_limits_steer_error():
    m, d, sb = _sb()
    th, qd = sb.module_cmd(np.array([-0.2, 0.0, 0.0]))     # backwards: reverse the wheels instead of steering 180 deg
    assert np.allclose(th, 0.0) and np.allclose(qd, -0.2 / sb.R_wheel)


def test_mirror_geometry():
    c = BS.mirror_centre(0.5, 0.5)
    p = np.array([0.38, -0.22])
    q = BS.mirror(p, c)
    assert np.allclose(BS.mirror(q, c), p)


def test_limit_is_on_vector_norm_not_per_axis():
    m, d, sb = _sb()
    sb.lim.a_max, sb.lim.v_max = 0.25, 10.0
    pf = SW.PathFollower(sb)
    dt = 0.005
    new = pf._limit(np.array([5.0, 5.0, 0.0]), dt)               # both axes saturate: |a| must be a_max, not sqrt(2) a_max
    assert abs(np.linalg.norm(new[:2]) - sb.lim.a_max * dt) < 1e-9
    sb._v[:] = 0.0
    for _ in range(2000):
        new = pf._limit(np.array([5.0, 5.0, 0.0]), dt)
    assert np.linalg.norm(new[:2]) <= sb.lim.v_max + 1e-9


def test_drive_torque_clip_and_gate():
    m, d, sb = _sb()
    assert sb.tau_cap > 0 and np.allclose(sb.kv_drive, 1.0)
    d.qvel[sb.drive_v] = 0.0
    sb.apply(np.array([1.0, 0.0, 0.0]), 0.005)                    # huge twist request: |ctrl - w| <= tau_cap / kv
    for i in range(4):
        assert abs(d.ctrl[sb.a_drive[i]] - d.qvel[sb.drive_v[i]]) <= sb.tau_cap / sb.kv_drive[i] + 1e-9
    d.qpos[sb.steer_q] = 0.0
    assert sb.align_gate(np.array([0.2, 0.0, 0.0])) == 1.0        # aligned: no reduction
    assert abs(sb.align_gate(np.array([0.0, 0.2, 0.0])) - sb.lim.gate_min) < 1e-9   # 90 deg off: floor


def test_stop_decays_wheel_command():
    m, d, sb = _sb()
    d.ctrl[sb.a_drive[0]] = 1.0
    sb._dt = 0.005
    sb.stop()
    assert 0.9 < d.ctrl[sb.a_drive[0]] < 1.0
