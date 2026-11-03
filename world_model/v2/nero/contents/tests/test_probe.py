"""probe ladder: belief update rules and rung admission (no long sims).  SIMULATION ONLY."""
import numpy as np
import pytest

from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.nero.contents import probe as PR


def test_interval_spec_tray_one_mug_may_hold_all_fill():
    sp = PR.interval_spec("tray6", 6 * 0.30 + 0.02, 6 * 0.30 + 0.10)       # total fill 0.02..0.10 kg
    assert sp.contents.m_lo == pytest.approx(1.82) and sp.contents.m_hi == pytest.approx(1.90)
    lim_small = sp.open_limits[0].a_max
    lim_full = PR.interval_spec("tray6", 1.8, 1.8 + 1.5).open_limits[0].a_max
    assert lim_small > lim_full                                         # less possible fill per mug -> looser spill limit


def test_rung_torque_row_monotone_in_push():
    us = [PR.rung_torque_u("box", m) for m in (0.5, 1.5, 3.0, 5.0)]
    assert all(b >= a - 1e-9 for a, b in zip(us, us[1:]))


@pytest.fixture(scope="module")
def probe_box():
    w = CC.sample_world("box", (1.0, 1.0001), np.random.default_rng(3))
    ep = PR._episode(w)
    return PR.Probe(ep, "box", eps_lo=0.05, eps_free=0.05, noise=False), ep


def test_first_rung_pushes_no_more_than_lightest_object(probe_box):
    pr, _ = probe_box
    ok, rows = pr.admissible(pr.M_lo)
    assert ok and rows["overshoot"] == pytest.approx(0.0)
    ok2, rows2 = pr.admissible(pr.M_lo + PR.O_MAX + 0.3)
    assert not ok2 and rows2["overshoot"] > PR.O_MAX


def test_ladder_bounds_contain_truth_and_stay_sound(probe_box):
    pr, ep = probe_box
    M = PR.true_mass(ep)
    why = pr.run(M_stop=None)
    assert why == "free"
    for r in pr.rungs:                                       # every intermediate belief keeps the truth
        assert r["Theta"][0] <= M + 1e-9 and r["Theta"][1] >= M - 1e-9
    assert pr.M_hi - pr.M_lo < 0.5                           # narrowed from 4.5 kg wide
    assert ep.track["viol"] is None
