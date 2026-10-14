"""Tests for house/lift.py and the lift-aware parts of perception_bridge.  SIMULATION ONLY.  PYTHONPATH=. python -m pytest -q world_model/v2/house/tests/test_lift.py"""
import numpy as np
import pytest
import mujoco

from world_model.v2.house import lift as L
from world_model.v2.house import perception_bridge as PB


@pytest.fixture(scope="module")
def C():
    return L.new_hold_lift(17, (3.55, 10.05), 0.3, 0.005, lift_h=0.0)


def test_sliders_present_vertical_and_telescope_class_values(C):
    m, d = C.m, C.d
    for i in (1, 2):
        j = m.joint(f"Slider_{i}").id
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
        assert np.allclose(m.jnt_range[j], [0, 0.208]) and np.isclose(m.dof_damping[m.jnt_dofadr[j]], 50.0)
        assert np.isclose(m.dof_armature[m.jnt_dofadr[j]], 0.01) and np.isclose(m.dof_frictionloss[m.jnt_dofadr[j]], 1.5)
        assert abs(d.xaxis[j][2] - 1.0) < 1e-5


def test_set_lift_moves_arms_tray_ball_rigidly_and_keeps_welds(C):
    m, d = C.m, C.d
    z = lambda n: d.xpos[m.body(n).id][2]
    z0 = (z("left_arm_link0"), z("tray"), z("ball"))
    L.set_lift(C, 0.3)
    assert np.allclose([z("left_arm_link0") - z0[0], z("tray") - z0[1], z("ball") - z0[2]], 0.3, atol=2e-4)
    assert abs(L.lift_state(C) - 0.3) < 1e-9
    for k in range(40):
        mujoco.mj_step(m, d)
    assert abs(L.lift_state(C) - 0.3) < 5e-3                      # equality lock holds under the arm / tray load
    L.set_lift(C, 0.0)
    with pytest.raises(ValueError):
        L.set_lift(C, 0.5)


def test_actuated_variant_reaches_commanded_height():
    C2 = L.compose_lift(17, (3.55, 10.05), 0.3, with_tray=False, dt=0.005, mode="actuated")
    m, d = C2.m, C2.d
    d.ctrl[m.actuator("Lift").id] = 0.2
    for k in range(600):
        mujoco.mj_step(m, d)
    assert abs(L.lift_state(C2) - 1.5 * 0.2) < 1.5e-2              # gain 150 / bias 100: L = 1.5 ctrl, within the 1.5 N frictionloss deadband (measured +9.5 mm)


def test_head_mounts_lift_vs_mast(C):
    sc = PB.HouseScene(C)
    out = {}
    for mount in ("lift", "mast"):
        cam = PB.make_rig(sc, "headN", mount, 36.3)[0][0]
        zs = []
        for h in (0.0, 0.416):
            L.set_lift(C, h)
            sc.place_base((3.55, 10.05), 0.3)
            zs.append(cam.pose()[0][2])
        out[mount] = zs
    assert np.isclose(out["lift"][1] - out["lift"][0], 0.416, atol=1e-6)
    assert np.isclose(out["mast"][1] - out["mast"][0], 0.0, atol=1e-9)
    L.set_lift(C, 0.0)
    sc.place_base((3.55, 10.05), 0.3)
