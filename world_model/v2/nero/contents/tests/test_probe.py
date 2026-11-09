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
    return PR.make_probe(ep, "box", eps_lo=0.05, eps_free=0.05, noise=False, method="ladder"), ep


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


def test_poslift_light_pot_goes_free_truth_inside():
    """poslift (default method, user 11-09): a light pot world hangs free within Z_MAX and the set contains the truth."""
    w = CC.sample_world("pot", (0.1, 0.1001), np.random.default_rng(5))
    ep = PR._episode(w)
    pr = PR.make_probe(ep, "pot", eps_lo=0.0766, eps_free=0.0316, noise=False)
    assert PR.method_of(pr) == "poslift" and type(pr) is PR.PosLiftProbe
    M = PR.true_mass(ep)
    pr.M_lo = PR.empty_mass("pot")
    why = pr.run(M_stop=None)
    assert why == "free", (why, pr.rungs[-1])
    assert pr.M_lo - 1e-9 <= M <= pr.M_hi + 1e-9 and pr.M_hi - pr.M_lo < 0.5
    assert max(r["claw_rise_mm"] for r in pr.rungs) <= 1e3 * PR.Z_MAX + 2.0


def test_poslift_heavy_box_lower_bound_only():
    """a heavy box: no free read; the probe stops with a lower bound that stays <= the truth (never 'free', never an upper bound)."""
    w = CC.sample_world("box", (4.9, 4.9001), np.random.default_rng(5))
    ep = PR._episode(w)
    pr = PR.make_probe(ep, "box", eps_lo=0.0766, eps_free=0.0316, noise=False)
    M = PR.true_mass(ep)
    why = pr.run(M_stop=PR.empty_mass("box") + 3.0)
    assert why in ("decided_refuse", "rung_not_admissible", "z_cap", "abort"), why
    assert pr.M_lo <= M + 1e-9 and pr.M_hi >= M - 1e-9


def test_make_probe_methods():
    assert PR.default_calib("poslift").endswith("probe_calib_poslift_2026-11-09/CALIB.json") and PR.POSLIFT_CALIB.exists()
    assert PR.default_calib("ladder").endswith("probe_calib_2026-11-03/CALIB.json")
    with pytest.raises(ValueError):
        PR.make_probe(None, "box", 0, 0, method="nope")
