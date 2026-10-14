"""Headless unit smoke (no display, no GPU, no robot).  python -m pytest -q world_model/v2/house/teleop/tests"""
import math
import numpy as np

from world_model.v2.house.teleop import base as BS, operator as OP, scene as SC, shield as SH, geom as GM
from world_model.v2.house.teleop.sim import Cmd, TICK


def test_keyboard_latching_and_events():
    kb = OP.KeyboardOperator()
    for key in (OP.GLFW["UP"], OP.GLFW["UP"], OP.GLFW["LEFT"], OP.GLFW["R"], OP.GLFW["G"]):
        kb.inject(key)
    c = kb.step()
    assert np.allclose(c.twist, np.array([0.5, 0.0, 0.25]) * BS.V_STICK) and c.lift_v > 0 and c.grasp
    assert not kb.step().grasp                      # one-shot
    kb.inject(OP.GLFW["SPACE"])
    assert not np.any(kb.step().twist)


def test_operator_deterministic_per_seed():
    sc = SC.get_scene("h5")
    a = OP.rollout_stream(3, sc, max_s=6.0)
    b = OP.rollout_stream(3, sc, max_s=6.0)
    c = OP.rollout_stream(4, sc, max_s=6.0)
    assert np.array_equal(a.twist, b.twist) and not np.array_equal(a.twist[: len(c.twist)], c.twist[: len(a.twist)])
    assert np.abs(a.twist[:, 0]).max() <= BS.V_STICK[0] + 1e-9           # stick saturation


def test_cup_accel_includes_centripetal_and_tangential():
    r = np.array([0.45, -0.21])
    a = BS.cup_accel_body(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]), r)
    assert np.allclose(a, -1.0 * r)                                      # w^2 r toward the axis
    a = BS.cup_accel_body(np.zeros(3), np.array([0.0, 0.0, 4.0]), r)
    assert np.allclose(a, 4.0 * np.array([-r[1], r[0]]))                 # alpha x r


def test_rect_dist():
    d, q, n = GM.rect_dist_point(np.array([1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert abs(d - (1.0 - GM.FP["x"][1])) < 1e-9 and np.allclose(n, [1, 0])


def test_ir_set_ratio():
    cfg = SH.ShieldCfg()
    assert abs(cfg.ratio - min(0.8 * 0.18, math.tan(0.30))) < 1e-12
