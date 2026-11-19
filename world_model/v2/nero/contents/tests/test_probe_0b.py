"""Stage 0b (11-12) weigh-probe methods poslift_ff and claw_guard: stop logic on synthetic reads (no sim), helpers, and the per-claw guard vs the static row.
SIMULATION ONLY."""
import types

import numpy as np
import pytest

from world_model.v2.nero.contents import probe as PR


# ------------------------------------------------------------------------------------------------- task keys (B11)
def test_n_cups_parses_behind_keys_and_keeps_old_values():
    assert PR.n_cups("tray6_behind") == 6
    assert [PR.n_cups(t) for t in ("tray6", "tray4", "tray0", "box", "pot", "box_behind")] == [6, 4, 0, 0, 0, 0]


def test_methods_registered_and_poslift_default():
    assert {"poslift", "poslift_ff", "claw_guard"} == set(PR.METHODS)   # ladder: programmatic only (PR._PROGRAMMATIC)
    assert PR.default_calib("poslift") == str(PR.POSLIFT_CALIB)
    for m in ("poslift_ff", "claw_guard"):
        a = PR.method_assumptions(m)
        assert a["probe_method"] == m and any("ASSUMED" in str(v) for v in a.values())
    assert "DM_FINE" in PR.method_assumptions("poslift_ff") and "EPS_CLAW" in PR.method_assumptions("claw_guard")


def test_jump_is_per_task_measured_with_pooled_fallback():
    jb, src = PR.jump_max("box_behind")
    assert jb == pytest.approx(0.7104) and "MEASURED" in src
    assert PR.jump_max("tray6")[0] == pytest.approx(0.5964)
    v, src = PR.jump_max("tray4")
    assert v == pytest.approx(0.7104) and "ASSUMED" in src


# ------------------------------------------------------------------------------------------------- poslift_ff finish phase on synthetic reads
class _Ep:
    def __init__(self):
        self.track = dict(Fz_max=0.0, viol=None)
        self.F_cap = 25.0
        self.obj, self.z0 = 0, 0.0
        self.d = types.SimpleNamespace(xpos=np.zeros((1, 3)))


def _ff(monkeypatch, reads, M_lo=2.2, M_hi=5.25, eps_lo=0.08, eps_free=0.03, nb=0.08):
    """a PosLiftFFProbe whose rung() returns scripted reads; u(L) = L / 3 (synthetic static row: L_admit = 3 kg)."""
    monkeypatch.setattr(PR, "rung_torque_u", lambda task, L, world=None: L / 3.0)
    pr = object.__new__(PR.PosLiftFFProbe)
    pr.ep, pr.task = _Ep(), "box"
    pr.eps_lo, pr.eps_free, pr.M_lo, pr.M_hi, pr.m_ff = eps_lo, eps_free, M_lo, M_hi, M_lo
    pr.rungs, pr.guard_events, pr.sim_abort, pr.u_read_max, pr.peak_u, pr._z_rest = [], [], None, 0.0, 0.0, 0.0
    pr.dm_fine, pr.ff_free_ratio, pr.ff_t_hold, pr.jump, pr.jump_src, pr.ff = 0.1, 0.2, 1.2, 0.71, "test", None
    pr.track = dict(rise_mm=0.0, tilt=0.0, spill=0.0)
    pr.commanded = []
    it = iter(reads)

    def rung(m_to, t_hold=None):
        pr.commanded.append(m_to)
        pr.m_ff = m_to
        return next(it), nb, 0.0
    pr.rung = rung
    pr._claw_z = lambda: 0.0
    pr._ramp_during = lambda m, T: setattr(pr, "m_ff", m)
    return pr


def test_ff_free_after_liftoff_gives_interval_with_truth(monkeypatch):
    M = 2.6
    pr = _ff(monkeypatch, [2.4, 2.5, 2.6, 2.605])
    why = pr._after_height(2.3, 0.08, M_stop=None)
    assert why == "free"
    assert pr.M_lo <= M <= pr.M_hi and pr.M_hi == pytest.approx(2.605 + 0.03 + 0.08)
    assert np.allclose(np.diff([2.2] + pr.commanded), 0.1)                     # each command adds exactly DM_FINE
    assert pr.ff["n"] == 4 and pr.m_ff <= pr.M_lo + 1e-12                      # ff ramped back to M_lo (poslift invariant)


def test_ff_stops_before_a_step_that_could_breach_the_row(monkeypatch):
    reads = [2.3 + 0.1 * k for k in range(1, 30)]                              # load keeps rising: heavy object still on the counter
    pr = _ff(monkeypatch, reads)
    why = pr._after_height(2.3, 0.08, M_stop=None)
    assert why == "rung_not_admissible" and pr.guard_events[-1]["kind"] == "ff_pre_check"
    assert pr.M_hi == 5.25                                                     # lower bound only
    L_prev = [2.3] + [r["L"] for r in pr.rungs[:-1]]
    for lp in L_prev:                                                          # every rung that ran was pre-checked admissible
        assert (lp + 0.1 + 0.08 + 0.08) / 3.0 <= 1.0 + 1e-12
    assert (pr.rungs[-1]["L"] + 0.26) / 3.0 > 1.0                              # and the next one was not


def test_ff_load_guard_on_a_read_above_the_row(monkeypatch):
    pr = _ff(monkeypatch, [2.4, 3.2])
    why = pr._after_height(2.3, 0.08, M_stop=None)
    assert why == "load_guard" and pr.M_hi == 5.25 and pr.M_lo == pytest.approx(3.2 - 0.16)
    assert not pr.rungs[-1]["free"]


def test_ff_decided_refuse(monkeypatch):
    pr = _ff(monkeypatch, [])
    assert pr._after_height(2.3, 0.08, M_stop=2.0) == "decided_refuse" and pr.rungs == []


def test_ff_flat_rule_is_one_sided_on_the_commanded_step():
    assert PR.ff_flat(0.019, 0.1, 0.2) and PR.ff_flat(-0.3, 0.1, 0.2) and not PR.ff_flat(0.021, 0.1, 0.2)
    assert PR.ff_switch(0.70, 0.71) and not PR.ff_switch(0.72, 0.71)


# ------------------------------------------------------------------------------------------------- claw_guard helpers
def test_claw_helpers():
    Ls, nb = dict(left=1.0, right=2.0), dict(left=0.04, right=0.05)
    assert PR.claw_u(Ls, 2.5) == pytest.approx(0.8)
    assert PR.claw_headroom(Ls, nb, 0.1, 2.5) == pytest.approx(2.5 - 2.15)
    sp = PR.split_position(1.5, 1.5, 0.1, 0.1, 0.48)
    assert sp["offset_m"] == pytest.approx(0.0) and sp["offset_lo_m"] < 0 < sp["offset_hi_m"]
    sp = PR.split_position(2.0, 1.0, 0.1, 0.1, 0.48)
    assert sp["offset_lo_m"] <= sp["offset_m"] <= sp["offset_hi_m"] and sp["offset_m"] == pytest.approx((2 / 3 - 0.5) * 0.48, abs=1e-5)
    assert PR.split_position(0.0, 0.0, 0.1, 0.1, 0.48) is None


def test_claw_guard_admissible_and_headroom_on_synthetic_reads(monkeypatch):
    monkeypatch.setattr(PR, "rung_tau_u", lambda task, L, world=None: 0.1)
    monkeypatch.setattr(PR, "l_tau_admit_max", lambda task, world=None: 30.0)
    pr = object.__new__(PR.ClawGuardProbe)
    pr.task, pr.eps_lo, pr.eps_claw, pr.cap_kg, pr.M_lo = "box", 0.08, 0.08, 2.55, 3.4
    pr._last_sides = dict(L=dict(left=1.7, right=1.75), nb=dict(left=0.04, right=0.04), truth=dict(left=0, right=0))
    ok, rows = pr.admissible(3.4)                                              # centred 3.45 kg: today's worst-COM row (l_admit 3.28) would refuse
    assert ok and rows["u_claw"] == pytest.approx((1.75 + 0.12) / 2.55, abs=1e-4)
    assert pr._headroom(3.45, 0.08) == pytest.approx(2.55 - 1.87)
    pr._last_sides = dict(L=dict(left=0.9, right=2.5), nb=dict(left=0.04, right=0.04), truth=dict(left=0, right=0))
    ok, rows = pr.admissible(3.4)                                              # one claw near its limit: refused
    assert not ok and pr._headroom(3.4, 0.08) < 0


# ------------------------------------------------------------------------------------------------- per-claw guard vs the static row (MODEL)
@pytest.mark.parametrize("task", ["box"])
def test_claw_guard_equals_static_row_at_worst_com_and_is_below_it_elsewhere(task):
    """On the model's own lever split, the per-claw payload utilisation at any COM in P is <= the worst-COM row rung_torque_u, with equality at the worst
    COM (payload row binding).  So reading the per-claw loads can only admit what the row would admit at the TRUE COM; it is not weaker as long as the read
    is within EPS_CLAW + noise of the true claw load."""
    env = PR._rest_envelope(task)
    cap = PR.claw_cap_kg(task)
    for L in (1.5, 2.5, 3.2):
        c = L - PR.empty_mass(task)
        u_row = PR.rung_torque_u(task, L)
        vals = []
        for pi in PR.OB.Contents(c, c, PR.interval_spec(task, *PR.prior_contents(task)).contents.P).pi_vertices():
            u, det = env.u("p0", pi, np.zeros(3), "two_lever")
            uc = max(PR.claw_u({s: v / PR.G for s, v in det["Fz"].items()}, cap), PR.rung_tau_u(task, L))
            assert uc <= u_row + 1e-3
            vals.append(uc)
        assert max(vals) == pytest.approx(u_row, abs=1e-3)
