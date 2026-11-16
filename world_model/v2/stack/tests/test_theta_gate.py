"""K1 gate (owner-written, 11-16): randomized soundness of stack/theta.py.  SIMULATION ONLY (pure math).

Checks the properties the checker relies on, against a Monte-Carlo point oracle (points drawn uniformly, so boundaries have measure 0):
  minus      p in A\\B  <=>  p in A and p not in B
  intersect  p in A&B  <=>  p in A and p in B
  union      p in A|B  <=>  p in A or p in B
  contains   A.contains(B) is never True while a sampled point of B lies outside A (soundness), and is True for B = subsets built from A
  bin_sets   every theta whose noisy reading interval meets Y_j lies in Theta_j (no world lost: soundness of a program's bins)
  covers     a reading not covered by any bin is reported (no false coverage)
"""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.stack.theta import Interval, ReadingSpec, RecoveryProgram, ThetaSet, Cell, bin_sets, covers_readings, reading_range

FIELDS = ("m", "cx", "cy")


def _rand_set(rng, nf, ncells, lo=0.0, hi=1.0):
    cells = []
    for _ in range(ncells):
        b = {}
        for f in FIELDS[:nf]:
            a, c = sorted(rng.uniform(lo, hi, 2))
            b[f] = Interval(a, max(c, a + 1e-3))
        cells.append(Cell(b))
    return ThetaSet(tuple(cells))


def _pts(rng, nf, n=4000, lo=-0.1, hi=1.1):
    return [dict(zip(FIELDS[:nf], rng.uniform(lo, hi, nf))) for _ in range(n)]


def test_set_algebra_vs_point_oracle():
    rng = np.random.default_rng(11)
    for trial in range(60):
        nf = 1 + trial % 3
        A, B = _rand_set(rng, nf, 1 + trial % 3), _rand_set(rng, nf, 1 + (trial // 3) % 3)
        D, I, U = A.minus(B), A.intersect(B), A.union(B)
        for p in _pts(rng, nf, 1500):
            a, b = A.contains_point(p), B.contains_point(p)
            assert D.contains_point(p) == (a and not b), (trial, p)
            assert I.contains_point(p) == (a and b), (trial, p)
            assert U.contains_point(p) == (a or b), (trial, p)


def test_contains_is_sound():
    rng = np.random.default_rng(12)
    n_true = 0
    for trial in range(200):
        nf = 1 + trial % 3
        A, B = _rand_set(rng, nf, 2), _rand_set(rng, nf, 1)
        if A.contains(B):
            n_true += 1
            for p in _pts(rng, nf, 800, 0.0, 1.0):
                if B.contains_point(p):
                    assert A.contains_point(p), (trial, p)
    # a set always contains its own pieces, and the union of a split covers the whole
    whole = ThetaSet.box(m=(0.5, 5.0), cx=(-0.1, 0.1))
    left, right = ThetaSet.box(m=(0.5, 2.0), cx=(-0.1, 0.1)), ThetaSet.box(m=(2.0, 5.0), cx=(-0.1, 0.1))
    assert whole.contains(left) and whole.contains(right)
    assert left.union(right).contains(whole)
    gap = ThetaSet.box(m=(0.5, 1.9), cx=(-0.1, 0.1)).union(right)
    assert not gap.contains(whole)


def test_bin_sets_lose_no_world():
    """1-field (exact) and 2-field (superset) readings: theta with (h(theta) + [-nu, nu]) meeting Y_j must be in Theta_j."""
    rng = np.random.default_rng(13)
    theta = ThetaSet.box(m=(0.5, 5.0), cx=(-0.1, 0.1)).union(ThetaSet.box(m=(5.0, 6.0), cx=(-0.05, 0.05)))
    specs = [ReadingSpec("claw_load", "vertical load [kg]", {"m": 1.0}, 0.3, 0.08, "MEASURED", "probe CALIB"),
             ReadingSpec("wrist_torque", "moment", {"m": 0.4, "cx": 3.0}, 0.0, 0.02, "ASSUMED", "test")]
    for spec in specs:
        R = reading_range(spec, theta)
        edges = np.linspace(R.lo, R.hi, 6)
        prog = RecoveryProgram("p", "prefix", spec, tuple((Interval(a, b), f"plan{j}") for j, (a, b) in enumerate(zip(edges[:-1], edges[1:]))), "counter")
        ok, gaps = covers_readings(prog, theta)
        assert ok and not gaps
        sets = bin_sets(prog, theta)
        exact_expected = sum(1 for c in spec.coef.values() if c) <= 1
        for j, Tj, exact in sets:
            assert exact == exact_expected
            Y = prog.bins[j][0]
            for p in _pts(rng, 2, 3000, -0.2, 6.2):
                p2 = {"m": p["m"], "cx": p["cx"]}
                if not theta.contains_point(p2):
                    continue
                h = spec.offset + sum(c * p2[f] for f, c in spec.coef.items())
                if h + spec.noise >= Y.lo and h - spec.noise <= Y.hi:
                    assert Tj.contains_point(p2), (spec.producer, j, p2)
                elif exact:
                    assert not Tj.contains_point(p2), (spec.producer, j, p2)


def test_covers_reports_gap():
    theta = ThetaSet.box(m=(1.0, 3.0))
    spec = ReadingSpec("claw_load", "kg", {"m": 1.0}, 0.0, 0.1, "MEASURED", "probe")
    prog = RecoveryProgram("p", None, spec, ((Interval(0.0, 1.8), "a"), (Interval(2.0, 4.0), "b")), "counter")
    ok, gaps = covers_readings(prog, theta)
    assert not ok and len(gaps) == 1 and math.isclose(gaps[0].lo, 1.8) and math.isclose(gaps[0].hi, 2.0)
    assert covers_readings(RecoveryProgram.fixed("f", "plan", "counter"), theta) == (True, [])


def test_truth_reading_refused():
    import pytest
    for name in ("truth", "TRUTH", "truth_sim"):
        with pytest.raises(ValueError):
            ReadingSpec(name, "x", {"m": 1.0}, 0.0, 0.0, "MODEL", "t")
