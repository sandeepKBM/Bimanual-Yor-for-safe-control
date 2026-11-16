"""Z3 planner gate (owner-written, 11-16): stack/z3plan against an INDEPENDENT oracle (not z3plan/synthetic.py) and brute force.  SIMULATION ONLY (pure).

Owner oracle: each plan p has a random admissible interval [floor(p), cap(p)] of contents mass; p passes on Theta iff floor <= lo and hi <= cap
(monotone: fails on Theta => fails on every superset; passes => passes on every subset).  A fraction of plans is geometrically infeasible
(never passes for any theta or a) and reports block_all_theta_a; in some trials the cap decreases with a, then block_a_ge is reported.

  solve      finds a plan iff brute force has one; the plan passes; with a cost it equals the brute-force optimum
  store      a shared LearnedStore over a random sequence of thetas never changes an answer (vs brute force per theta)
  split      every returned bin's plan passes on its Theta_j; the bins cover Theta_0; adjacent bins overlap by 2 * noise (no world lost)
"""
from __future__ import annotations

import itertools

import numpy as np

from world_model.v2.stack.z3plan.types import Domain, Theta, Learned, OracleResult, ReadingModel
from world_model.v2.stack.z3plan import loop as L
from world_model.v2.stack.z3plan import probe_split as PS

DOMS = (Domain("zc", tuple(np.round(np.arange(0.06, 0.21, 0.02), 3))), Domain("dx", (-0.01, -0.005, 0.0, 0.005, 0.01)),
        Domain("a", (0.05, 0.1, 0.25, 0.5)), Domain("recovery", ("stored_backup", "read_then_retrace")))


def plans():
    for vals in itertools.product(*(d.values for d in DOMS)):
        yield dict(zip((d.name for d in DOMS), vals))


class OwnerOracle:
    def __init__(self, seed, a_mono):
        rng = np.random.default_rng(seed)
        self.geo_bad, self.base = {}, {}
        for p in plans():
            g = tuple(v for k, v in sorted(p.items()) if k != "a")
            if g not in self.geo_bad:
                self.geo_bad[g] = rng.random() < 0.3
                f = float(rng.uniform(0.0, 3.0))
                self.base[g] = (f, f + float(rng.uniform(0.2, 3.0)))
        self.a_mono, self.calls = a_mono, 0

    def interval(self, p):
        g = tuple(v for k, v in sorted(p.items()) if k != "a")
        f, c = self.base[g]
        return g, f, (c - 2.0 * p["a"] if self.a_mono else c + (0.3 if p["a"] == 0.25 else 0.0))

    def __call__(self, p, th):
        self.calls += 1
        g, f, c = self.interval(p)
        if self.geo_bad[g]:
            return OracleResult(False, [dict(name="reach", ok=False)], [Learned("block_all_theta_a", dict(p), None, "geometry")])
        if f <= th.lo and th.hi <= c:
            return OracleResult(True, [dict(name="torque", ok=True)], [])
        lr = [Learned("block_theta_superset", dict(p), th, "monotone in theta")]
        if self.a_mono and th.hi > c:
            lr.append(Learned("block_a_ge", dict(p), th, "cap decreases with a"))
        return OracleResult(False, [dict(name="torque", ok=False)], lr)

    def passes(self, p, th):
        g, f, c = self.interval(p)
        return (not self.geo_bad[g]) and f <= th.lo and th.hi <= c


def test_solve_vs_brute_force_and_cost():
    rng = np.random.default_rng(7)
    for trial in range(8):
        orc = OwnerOracle(100 + trial, a_mono=bool(trial % 2))
        for _ in range(4):
            lo = float(rng.uniform(0.0, 3.0)); th = Theta(lo, lo + float(rng.uniform(0.05, 1.5)))
            feas = [p for p in plans() if orc.passes(p, th)]
            p, log = L.solve(DOMS, orc, th, L.LearnedStore())
            assert (p is None) == (not feas), (trial, th, len(feas))
            if p is not None:
                assert orc.passes(p, th)
            if feas:
                pc, _ = L.solve(DOMS, orc, th, L.LearnedStore(), cost=[("a", "max"), ("zc", "min")])
                best = max(feas, key=lambda q: (q["a"], -q["zc"]))
                assert (pc["a"], pc["zc"]) == (best["a"], best["zc"]), (pc, best)


def test_shared_store_never_changes_an_answer():
    rng = np.random.default_rng(8)
    for trial in range(4):
        orc = OwnerOracle(200 + trial, a_mono=True)
        store = L.LearnedStore()
        for _ in range(10):
            lo = float(rng.uniform(0.0, 3.0)); th = Theta(lo, lo + float(rng.uniform(0.05, 1.5)))
            feas = any(orc.passes(q, th) for q in plans())
            p, _ = L.solve(DOMS, orc, th, store)
            assert (p is not None) == feas, (trial, th)
            if p is not None:
                assert orc.passes(p, th)


def test_probe_split_sound():
    found = 0
    for seed in range(300, 312):
        orc = OwnerOracle(seed, a_mono=True)
        th0, rd = Theta(0.5, 5.0), ReadingModel(offset_kg=2.0, noise_kg=0.17, tag="test")
        bins, log = PS.split(DOMS, orc, th0, rd, max_bins=4)
        if bins is None:
            continue
        found += 1
        ths = [b[0] for b in bins]
        assert abs(ths[0].lo - th0.lo) < 1e-9 and abs(ths[-1].hi - th0.hi) < 1e-9
        for (tj, pj), (tk, _) in zip(bins, bins[1:]):
            assert tk.lo <= tj.hi - 2 * rd.noise_kg + 1e-9, (tj, tk)          # adjacent bins overlap by >= 2 noise: no world lost at an edge
        for tj, pj in bins:
            assert orc.passes(pj, tj), (seed, tj, pj)
    assert found >= 3


# ---------------------------------------------------------------------------------------------------------- v2 (11-16): free knobs, prefilter, status
class KnobScopedOracle(OwnerOracle):
    """failures depend only on some knobs: geometry on (zc, dx) only (recovery irrelevant), the mass cap on (zc, a) only (dx, recovery irrelevant).
    Reports `free` = exactly the knobs the failure does not depend on (TRUE claims), so a correct encoding must still agree with brute force."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.bad_geo = {(z, x): rng.random() < 0.35 for z in DOMS[0].values for x in DOMS[1].values}
        self.cap = {z: float(rng.uniform(0.5, 4.0)) for z in DOMS[0].values}
        self.calls = 0

    def passes(self, p, th):
        return (not self.bad_geo[(p["zc"], p["dx"])]) and th.hi <= self.cap[p["zc"]] - 2.0 * p["a"]

    def __call__(self, p, th):
        self.calls += 1
        if self.bad_geo[(p["zc"], p["dx"])]:
            return OracleResult(False, [dict(name="reach", ok=False)], [Learned("block_all_theta_a", dict(p), None, "geometry", free=("recovery",))])
        if th.hi > self.cap[p["zc"]] - 2.0 * p["a"]:
            return OracleResult(False, [dict(name="torque", ok=False)], [Learned("block_a_ge", dict(p), th, "cap", free=("dx", "recovery")),
                                                                        Learned("block_theta_superset", dict(p), th, "cap", free=("dx", "recovery"))])
        return OracleResult(True, [dict(name="torque", ok=True)], [])


def test_free_knobs_agree_with_brute_force_and_save_calls():
    rng = np.random.default_rng(31)
    n_free = n_plain = 0
    for trial in range(6):
        orc = KnobScopedOracle(400 + trial)
        for _ in range(3):
            lo = float(rng.uniform(0.0, 2.5)); th = Theta(lo, lo + float(rng.uniform(0.05, 1.0)))
            feas = [p for p in plans() if orc.passes(p, th)]
            p, log = L.solve(DOMS, orc, th, L.LearnedStore(), max_iters=2000)
            assert (p is None) == (not feas), (trial, th, len(feas))
            assert log[-1]["status"] == ("unsat" if p is None else "found"), log[-1]
            if p is not None:
                assert orc.passes(p, th)
            n_free += orc.calls; orc.calls = 0
    assert n_free > 0


def test_prefilter_removes_only_rejected_plans():
    orc = OwnerOracle(500, a_mono=True)
    rule = lambda v: (True, "zc too low", ("zc",)) if v["zc"] < 0.1 else None          # noqa: E731
    th = Theta(0.5, 1.0)
    feas = [p for p in plans() if orc.passes(p, th) and p["zc"] >= 0.1]
    p, log = L.solve(DOMS, orc, th, L.LearnedStore(), prefilter=rule, max_iters=2000)
    assert (p is None) == (not feas)
    assert p is None or p["zc"] >= 0.1
    pf = next(e for e in log if e.get("source") == "prefilter")
    n_low = sum(1 for q in plans() if q["zc"] < 0.1)
    assert pf["n_rejected"] == n_low and pf["n_clauses"] == len([z for z in DOMS[0].values if z < 0.1]), pf


def test_cap_is_not_unsat():
    orc = OwnerOracle(600, a_mono=False)
    th = Theta(0.0, 0.1)
    feas = [p for p in plans() if orc.passes(p, th)]
    p, log = L.solve(DOMS, orc, th, L.LearnedStore(), max_iters=1)
    if p is None:
        assert log[-1]["status"] == "max_iters" or (not feas and log[-1]["status"] == "unsat"), log[-1]


# ---------------------------------------------------------------------------------------------------------- review fixes (11-16)
class NonMonotonePass:
    """passes only on EXACTLY the interval it was asked first for a plan with a = 0.5 (like a Rule 1 stage whose path model is built at a_admit(theta)):
    a pass on T says nothing about a subset.  pass_monotone = False must keep the pass cache from answering for Q != T."""
    pass_monotone = False

    def __init__(self, good_theta):
        self.good, self.calls = good_theta, 0

    def __call__(self, p, th):
        self.calls += 1
        ok = p["a"] == 0.5 and p["zc"] == 0.1 and p["dx"] == 0.0 and p["recovery"] == "stored_backup" and (th.lo, th.hi) == (self.good.lo, self.good.hi)
        return OracleResult(ok, [dict(name="r1", ok=ok)], [] if ok else [Learned("block_exact", dict(p), th, "exact only")])


def test_pass_cache_respects_non_monotone_oracle():
    T, Q = Theta(1.0, 2.0), Theta(1.2, 1.8)
    orc, store = NonMonotonePass(T), L.LearnedStore()
    p, _ = L.solve(DOMS, orc, T, store, max_iters=2000)
    assert p is not None
    p2, log2 = L.solve(DOMS, orc, Q, store, max_iters=2000)
    assert p2 is None and log2[-1]["status"] == "unsat", log2[-1]       # the oracle passes nothing on Q; the cache must not answer
    assert not any(e.get("source") == "pass_cache" for e in log2)
    p3, log3 = L.solve(DOMS, orc, T, store, max_iters=2000)              # same T: the exact cache hit is still allowed
    assert p3 is not None and log3[-1]["source"] == "pass_cache"


def test_split_round_cap_is_not_a_negative_answer():
    orc = OwnerOracle(303, a_mono=True)
    th0, rd = Theta(0.5, 5.0), ReadingModel(offset_kg=2.0, noise_kg=0.17, tag="test")
    bins, log = PS.split(DOMS, orc, th0, rd, max_bins=4, max_rounds=1)
    assert bins is None and any(e.get("event") == "cegis_cut" for e in log)          # seed 303: one round is not enough (checked 11-16)
    assert log[-1]["event"] == "capped" and log[-1]["max_rounds"] == 1, log[-1]
    assert not any(e.get("event") == "no_split_up_to" for e in log)
