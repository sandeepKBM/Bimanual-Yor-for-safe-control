"""Smoke tests for the pan scene (step 1).  CPU only, no rendering.  Run:
    python -m pytest world_model/v2/nero/pan/tests -q
"""
import numpy as np
import mujoco
import pytest

from world_model.v2.nero.pan import scene as S


@pytest.fixture(scope="module")
def settled():
    m = S.compose()
    d = mujoco.MjData(m)
    drift = S.settle(m, d, 1.5)
    return m, d, drift


def test_pan_mass_is_light_pla(settled):
    m, _, _ = settled
    assert 0.15 <= m.body_mass[m.body("pan").id] <= 0.25          # USER 10-28: light 3D-printed PLA pan


def test_pan_rests_still(settled):
    _, _, drift = settled
    assert drift < 2e-3


def test_food_inside_and_on_bottom(settled):
    m, d, _ = settled
    p = S.PanParams()
    c = d.xpos[m.body("pan").id]
    for i in range(p.food_n):
        x = d.xpos[m.body(f"food{i}").id]
        assert np.linalg.norm((x - c)[:2]) < p.pan_radius - p.food_radius + 1e-3
        assert abs(x[2] - c[2] - p.pan_bottom_t - p.food_radius) < 3e-3


def test_no_arm_contact_at_rest(settled):
    m, d, _ = settled
    for con in d.contact[:d.ncon]:
        names = [m.body(m.geom_bodyid[g]).name for g in (con.geom1, con.geom2)]
        assert not any(n.startswith(("left_arm", "right_arm")) for n in names), names


def test_handle_moment_grows_with_food():
    out = []
    for fm in (0.0, 0.02, 0.04):
        m = S.compose(S.PanParams(food_mass=max(fm, 1e-4)))
        d = mujoco.MjData(m)
        S.settle(m, d, 0.5)
        out.append(S.handle_moment(m, d)["moment_Nm"])
    assert out[0] > 0 and out[0] < out[1] < out[2]
