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
    assert PR.default_calib("poslift").endswith("probe_calib_poslift_adaptive_2026-11-09/CALIB.json") and PR.POSLIFT_CALIB.exists()
    assert PR.poslift_k_step()[1] == "CALIB K_STEP"                     # the adaptive probe has a MEASURED K_STEP (no fallback)
    assert PR.default_calib("ladder").endswith("probe_calib_2026-11-03/CALIB.json")
    with pytest.raises(ValueError):
        PR.make_probe(None, "box", 0, 0, method="nope")


def test_adaptive_dz_shrinks_with_headroom_and_stops_at_dz_min():
    k = 1.0364                                                       # kg/mm
    assert PR.adaptive_dz(10.0, k) == pytest.approx(PR.DZ)           # plenty of headroom: full step
    assert PR.adaptive_dz(0.5 * k, k) == pytest.approx(0.5e-3)       # shrinks proportionally
    assert PR.adaptive_dz(0.5 * k, k) < PR.adaptive_dz(0.8 * k, k) < PR.DZ
    assert PR.adaptive_dz(0.05 * k, k) < PR.DZ_MIN                   # 0.05 mm < DZ_MIN 0.1 mm -> the probe stops
    assert PR.adaptive_dz(-0.1, k) < PR.DZ_MIN
    assert 1e3 * PR.adaptive_dz(0.1 * k, k) == pytest.approx(1e3 * PR.DZ_MIN)   # exactly at DZ_MIN is allowed


def test_l_admit_max_is_the_u_boundary():
    for t in ("box", "tray6", "pot"):
        L = PR.l_admit_max(t)
        assert PR.rung_torque_u(t, L) <= PR.E.U_ADMIT + 1e-9
        assert PR.rung_torque_u(t, L + 0.01) > PR.E.U_ADMIT


def test_rate_free_test_equals_old_test_at_full_step():
    for dL in (-0.2, -0.0499, -0.05, 0.0, 0.0499, 0.05, 0.3):
        assert PR.rate_flat(dL, PR.DZ) == (abs(dL) < PR.FREE_TOL)
    assert not PR.rate_flat(0.03, 0.5 * PR.DZ) and PR.rate_flat(0.02, 0.5 * PR.DZ)   # half step: half the dL allowed


def test_rate_free_test_no_false_free_on_stiff_ramp():
    """synthetic ramp L = k*z, k = K_STEP, plus read noise <= 0.01 kg: at any step size >= DZ_MIN the rate test never reads two flat rungs while on the counter."""
    rng = np.random.default_rng(0)
    k = 1.0364
    for dz in (1e-3, 5e-4, 2.5e-4, 1e-4):
        for _ in range(200):
            L = k * 1e3 * dz * np.arange(10) + rng.uniform(-0.01, 0.01, 10)
            flat = [PR.rate_flat(L[i + 1] - L[i], dz) for i in range(9)]
            assert not any(a and b for a, b in zip(flat, flat[1:]))


def test_free_needs_flat_travel_not_flat_rungs():
    """first mixed calibration, pot i=0: two flat 0.1 mm rungs inside the contact slack must not declare free; at dz = DZ two flat rungs still do (old rule)."""
    f, free = PR.flat_update(0.0, 0.0007, 1e-4)
    f, free = PR.flat_update(f, -0.0005, 1e-4)
    assert not free
    f, free = PR.flat_update(0.0, 0.02, PR.DZ)
    assert not free
    f, free = PR.flat_update(f, 0.03, PR.DZ)
    assert free
    f, free = PR.flat_update(f, 0.2, PR.DZ)                          # a moving rung resets
    assert f == 0.0 and not free
