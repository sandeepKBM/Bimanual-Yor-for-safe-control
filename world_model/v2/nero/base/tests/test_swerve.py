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
