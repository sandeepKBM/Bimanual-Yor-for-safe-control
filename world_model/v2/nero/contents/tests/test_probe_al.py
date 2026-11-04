"""probe_al: likelihood fit, selector admissibility, sound-update equivalence with Probe.run, rungs vs the fixed ladder.  SIMULATION ONLY."""
import json
import pathlib

import numpy as np
import pytest

from world_model.v2.nero.contents import probe as PR
from world_model.v2.nero.contents import probe_al as AL

CALIB_ROWS = pathlib.Path("/data/local/ss5772/contents/probe_calib_2026-11-03/calib.jsonl")
EPS_LO, EPS_FREE, M_UMAX = 0.05, 0.08, 2.5


def synth_lik():
    p = dict(coef=[-0.02, 0.97, -0.02], b=0.01, sd_on=0.0, sd_tr=0.05, sd_free=0.0, nb=0.082)
    return AL.Lik(dict(fam=dict(box=p, tray=p)))


class _Ctl:
    mass = 0.0


class _Ep:
    ctl = _Ctl()

    def run_phases(self, ph):
        pass


class FakeProbe:
    """Probe stand-in: reads are the noise-free synthetic physics; admissibility = torque cap M_UMAX and overshoot row (Probe's rule)."""
    def __init__(self, M, lik, task="box"):
        self.M, self.lik, self.task, self.ep = M, lik, task, _Ep()
        self.M_lo, self.M_hi = 0.75, 5.25
        self.m_ff, self.rungs = 0.0, []
        self.eps_lo, self.eps_free = EPS_LO, EPS_FREE

    def admissible(self, m):
        over = m - self.M_lo
        return (m <= M_UMAX and over <= PR.O_MAX + 1e-9), dict(u=0.0, overshoot=over)

    def rung(self, m):
        self.m_ff = m
        mu, _ = self.lik.mean_sd("box", m, self.M)
        return float(mu), 0.082, 0.0

    def run(self):                                   # Probe.run, verbatim logic (copied via the real class bound to this object)
        return PR.Probe.run(self, M_stop=None)


def make_ctx(lik):
    return AL.Ctx("box", lik, 2.363, EPS_LO, EPS_FREE, M_UMAX)


def test_likelihood_fit_finite():
    if not CALIB_ROWS.exists():
        pytest.skip("calib.jsonl missing")
    lp = AL.fit_likelihood([json.loads(l) for l in open(CALIB_ROWS)])
    assert set(lp["fam"]) == {"box", "tray"}
    for q in lp["fam"].values():
        assert all(np.isfinite(q["coef"])) and np.isfinite(q["b"]) and q["sd_on"] >= 0 and q["n_rungs"] > 10
    mu, sd = AL.Lik(lp).mean_sd("box", np.array([1.0, 2.0]), np.array([3.0, 3.0]))
    assert np.all(np.isfinite(mu)) and np.all(sd > 0) and mu[1] > mu[0]


def test_selector_only_proposes_admissible_increasing_forces():
    lik = synth_lik()
    ctx = make_ctx(lik)
    rng = np.random.default_rng(0)
    for objective in ("decision", "estimate"):
        for lo, last in ((0.75, 0.0), (1.2, 1.5), (2.0, 2.0)):
            cand = AL.candidates(ctx, lo, last, ideas=[0.1, 9.0, lo + 0.3])
            assert all(c <= min(lo + PR.O_MAX, M_UMAX) + 1e-9 and c >= lo - 1e-9 for c in cand)
            assert all(c >= last + (AL.STEP if last > 0 else 0) - 1e-9 for c in cand)
            bel = AL.Belief(0.75, 5.25)
            bel.clip(lo, 5.25)
            m, info = AL.choose(ctx, bel, lo, 5.25, None if last == 0 else 1.3, last, objective, rng)
            assert m is None or (m in cand)


def test_sound_update_equals_probe_run_rule():
    fp = FakeProbe(1.6, synth_lik())
    why = fp.run()
    assert why == "free"
    lo, hi, prev, last = 0.75, 5.25, None, 0.0
    for r in fp.rungs:                                 # replay the recorded reads through sound_update
        lo, hi, fl = AL.sound_update(lo, hi, r["L"], r["noise_bound"], prev, last, r["m_ff"], EPS_LO,
                                     fp.eps_free)
        assert [round(float(lo), 4), round(float(hi), 4)] == r["Theta"] and bool(fl) == r["free"]
        prev, last = r["L"], r["m_ff"]


def _run_fake(M, objective, prior):
    lik = synth_lik()
    fp = FakeProbe(M, lik)
    fp.eps_lo, fp.eps_free = EPS_LO, EPS_FREE
    ctx = make_ctx(lik)
    bel = AL.Belief(fp.M_lo, fp.M_hi, prior, None, M)
    why, free = AL.run_loop(fp, ctx, objective, bel, np.random.default_rng(1))
    return fp, why, free


def test_fixed_objective_schedule_equals_probe_run():
    for M in (1.2, 1.9):
        a = FakeProbe(M, synth_lik())
        a.eps_lo, a.eps_free = EPS_LO, EPS_FREE
        PR.Probe.run(a, M_stop=None)
        b, why, _ = _run_fake(M, "fixed", "flat")
        assert [r["m_ff"] for r in a.rungs] == [r["m_ff"] for r in b.rungs]
        assert [r["Theta"] for r in a.rungs] == [r["Theta"] for r in b.rungs]


def test_decision_objective_not_more_rungs_than_fixed_with_oracle_prior():
    for M in (1.3, 2.0, 2.4):                        # 2.4: the torque cap (M_UMAX) leaves no two rungs above M: neither arm proves free
        fx, _, f1 = _run_fake(M, "fixed", "flat")
        dc, why, f2 = _run_fake(M, "decision", "oracle")
        assert f2 == f1 and len(dc.rungs) <= len(fx.rungs) and f1 == (M < 2.3)
        assert dc.M_lo - 1e-9 <= M <= dc.M_hi + 1e-9


def test_fixed_schedule_equals_probe_run_on_real_sim():
    from world_model.v2.nero.contents import carry_cert as CC
    w = CC.sample_world("box", (1.0, 1.0001), np.random.default_rng(3))
    a = PR.Probe(PR._episode(w), "box", EPS_LO, EPS_FREE, noise=False)
    a.run(M_stop=None)
    b = PR.Probe(PR._episode(w), "box", EPS_LO, EPS_FREE, noise=False)
    ctx = AL.Ctx("box", synth_lik(), 2.363, EPS_LO, EPS_FREE, AL.m_umax("box"))
    AL.run_loop(b, ctx, "fixed", None, np.random.default_rng(0))
    assert [r["m_ff"] for r in a.rungs] == [r["m_ff"] for r in b.rungs]
    assert [r["Theta"] for r in a.rungs] == [r["Theta"] for r in b.rungs]
