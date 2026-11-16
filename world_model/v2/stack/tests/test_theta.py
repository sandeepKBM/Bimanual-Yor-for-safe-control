"""stack/theta.py: set ops vs a dense-grid oracle, JSON round trips, reading ranges, bins, coverage.  No sim."""
import itertools
import json
import math

import numpy as np
import pytest

from world_model.v2.stack.theta import (Admission, Cell, Interval, ReadingSpec, RecoveryProgram, ThetaSet, bin_sets, covers_readings,
                                        reading_range)
from world_model.v2.stack.types import Bound, ThetaBox

F = ("a", "b", "c")


def rand_set(rng, d, n):
    cells = []
    for _ in range(n):
        lo = rng.uniform(0, 8, d)
        w = rng.uniform(0.5, 4, d)
        cells.append(Cell({F[i]: Interval(lo[i], lo[i] + w[i]) for i in range(d)}))
    return ThetaSet(tuple(cells))


def grid(d, k):
    # points strictly inside (offset by an irrational-ish shift so they avoid every random boundary)
    ax = np.linspace(0.013, 11.987, k)
    return [dict(zip(F[:d], p)) for p in itertools.product(ax, repeat=d)]


def member(s, p):
    return s.contains_point(p)


@pytest.mark.parametrize("d,k", [(1, 400), (2, 60), (3, 20)])
def test_ops_vs_grid(d, k):
    rng = np.random.default_rng(100 + d)
    pts = grid(d, k)
    for _ in range(8):
        A, B = rand_set(rng, d, 3), rand_set(rng, d, 2)
        m, i, u = A.minus(B), A.intersect(B), A.union(B)
        for p in pts:
            a, b = member(A, p), member(B, p)
            assert member(m, p) == (a and not b)
            assert member(i, p) == (a and b)
            assert member(u, p) == (a or b)
        assert A.contains(A)
        assert u.contains(A) and u.contains(B)
        # contains agrees with the oracle on this grid when it says True
        if A.contains(B):
            assert all(member(A, p) for p in pts if member(B, p))


def test_minus_cells_disjoint():
    rng = np.random.default_rng(5)
    A, B = rand_set(rng, 2, 3), rand_set(rng, 2, 3)
    m = A.minus(B)
    for x, y in itertools.combinations(m.cells, 2):
        c = x.intersect(y)
        assert c is None or any(iv.width == 0 for iv in c.bounds.values())


def test_contains_cases():
    whole = ThetaSet.box(a=(0, 2), b=(0, 1))
    left, right = ThetaSet.box(a=(0, 1), b=(0, 1)), ThetaSet.box(a=(1, 2), b=(0, 1))
    halves = left.union(right)
    assert whole.contains(halves) and halves.contains(whole)
    gap = ThetaSet((Cell({"a": Interval(0, 0.9), "b": Interval(0, 1)}), Cell({"a": Interval(1.1, 2), "b": Interval(0, 1)})))
    assert not gap.contains(whole)
    assert whole.contains(gap)
    assert not whole.contains(ThetaSet.box(a=(0, 2.5), b=(0, 1)))
    assert ThetaSet().is_empty() and whole.contains(ThetaSet())


def test_field_mismatch_raises():
    with pytest.raises(ValueError):
        ThetaSet((Cell({"a": Interval(0, 1)}), Cell({"b": Interval(0, 1)})))
    with pytest.raises(ValueError):
        ThetaSet.box(a=(0, 1)).minus(ThetaSet.box(b=(0, 1)))
    with pytest.raises(ValueError):
        Interval(2, 1)


def test_corners_and_thetabox():
    s = ThetaSet.box(b=(0, 1), a=(2, 3))
    c = s.corners()
    assert s.fields == ("a", "b") and c.shape == (4, 2)
    assert {tuple(r) for r in c} == {(2, 0), (2, 1), (3, 0), (3, 1)}
    tb = ThetaBox({"m": Bound(0.1, 0.5, "ASSUMED", "x", "kg"), "mu": Bound(0.3, 0.9, "MODEL", "y")})
    t = ThetaSet.from_thetabox(tb)
    assert t.fields == ("m", "mu") and t.cells[0].bounds["m"] == Interval(0.1, 0.5)


SPEC = ReadingSpec("joint_torque", "vertical load", {"m": 1.0}, 0.3, 0.05, "ASSUMED", "test")
SPEC2 = ReadingSpec("joint_torque", "load", {"m": 1.0, "mu": -2.0}, 0.0, 0.1, "MODEL", "test")


def prog(spec, bins):
    return RecoveryProgram("p", "pre", spec, tuple((Interval(a, b), f"plan{i}") for i, (a, b) in enumerate(bins)), "table")


def test_json_roundtrips():
    th = ThetaSet.box(source="probe", provenance={"id": "p1", "n": 3}, m=(0.1, 0.5), mu=(0.2, 0.8)).union(ThetaSet.box(m=(1, 2), mu=(0, 1)))
    assert ThetaSet.from_json(json.loads(json.dumps(th.to_json()))) == th
    assert Cell.from_json(th.cells[0].to_json()) == th.cells[0]
    assert ReadingSpec.from_json(json.loads(json.dumps(SPEC2.to_json()))) == SPEC2
    p = prog(SPEC, [(0, 0.5), (0.5, 1.0)])
    assert RecoveryProgram.from_json(json.loads(json.dumps(p.to_json()))) == p
    fx = RecoveryProgram.fixed("f", "planX", "hold")
    assert RecoveryProgram.from_json(json.loads(json.dumps(fx.to_json()))).bins[0][0] == Interval(-math.inf, math.inf)
    ad = Admission(2, "p", th, {0.25: [("p", th), ("f", th)], 0.5: []}, th, [{"row": "tau", "ok": True}], ["ASSUMED noise"])
    ad2 = Admission.from_json(json.loads(json.dumps(ad.to_json())))
    assert ad2.theta_cert == th and ad2.table[0.25][1] == ("f", th) and ad2.rows == ad.rows and ad2.step == 2
    assert Admission(0, "x", th).to_json()["fallback_ok_on"] is None


def test_truth_producer_raises():
    for name in ("truth", "TRUTH", "truth_producer"):
        with pytest.raises(ValueError):
            ReadingSpec(name, "s", {"m": 1.0}, 0.0, 0.0, "ASSUMED", "t")
    with pytest.raises(ValueError):
        ReadingSpec("probe", "s", {"m": 1.0}, 0.0, 0.0, "GUESS", "t")


def test_reading_range_exact():
    th = ThetaSet.box(m=(0.1, 0.5), mu=(0.2, 0.8)).union(ThetaSet.box(m=(1, 2), mu=(0, 1)))
    r = reading_range(SPEC2, th)
    ys = [SPEC2.offset + SPEC2.coef["m"] * m + SPEC2.coef["mu"] * mu for m, mu in th.corners()]
    assert r.lo == pytest.approx(min(ys) - 0.1) and r.hi == pytest.approx(max(ys) + 0.1)


def test_bin_sets_one_field_exact():
    th = ThetaSet.box(m=(0.0, 1.0))
    p = prog(SPEC, [(0.3, 0.6), (0.6, 1.0), (1.0, 1.4)])
    res = bin_sets(p, th)
    assert [r[0] for r in res] == [0, 1, 2] and all(r[2] for r in res)
    for (j, tj, _), (iv, _) in zip(res, p.bins):
        for m in np.linspace(0.0007, 0.9993, 997):
            y = m + 0.3
            meets = (y - 0.05 <= iv.hi) and (y + 0.05 >= iv.lo)
            assert tj.contains_point({"m": m}) == meets, (j, m)


def test_bin_sets_two_fields_superset():
    th = ThetaSet.box(m=(0.0, 1.0), mu=(0.0, 1.0))
    p = prog(SPEC2, [(-2.0, -0.5), (-0.5, 0.5), (0.5, 1.0)])
    res = bin_sets(p, th)
    assert all(not r[2] for r in res)
    rng = np.random.default_rng(1)
    for (j, tj, _), (iv, _) in zip(res, p.bins):
        for m, mu in rng.uniform(0.001, 0.999, (2000, 2)):
            y = m - 2 * mu
            if y - 0.1 <= iv.hi and y + 0.1 >= iv.lo:
                assert tj.contains_point({"m": m, "mu": mu}), (j, m, mu)


def test_bin_sets_fixed_and_constant():
    th = ThetaSet.box(m=(0, 1))
    assert bin_sets(RecoveryProgram.fixed("f", "x", "hold"), th) == [(0, th, True)]
    const = ReadingSpec("probe", "c", {"m": 0.0}, 1.0, 0.0, "ASSUMED", "t")
    r = bin_sets(prog(const, [(0.0, 0.5), (0.9, 1.1)]), th)
    assert r[0][1].is_empty() and r[1][1] == th


def test_covers_readings():
    th = ThetaSet.box(m=(0.0, 1.0))          # readings in [0.25, 1.35]
    ok, gaps = covers_readings(prog(SPEC, [(0.0, 0.8), (0.8, 2.0)]), th)
    assert ok and gaps == []
    ok, gaps = covers_readings(prog(SPEC, [(0.0, 0.6), (0.9, 2.0)]), th)
    assert not ok and gaps == [Interval(0.6, 0.9)]
    ok, gaps = covers_readings(prog(SPEC, [(0.5, 2.0)]), th)
    assert not ok and gaps == [Interval(0.25, 0.5)]
    ok, gaps = covers_readings(prog(SPEC, [(0.0, 1.0)]), th)
    assert not ok and gaps[-1].hi == pytest.approx(1.35)
    assert covers_readings(RecoveryProgram.fixed("f", "x", "hold"), th) == (True, [])
