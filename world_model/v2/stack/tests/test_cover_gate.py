"""K8 gate (owner-written, 11-16): Rule 3 runtime cover lookup (stack/checks/cover.py).  SIMULATION ONLY (pure).

  narrowing  any subset of the admitted recovery's verified set keeps it ("continue"), at every branch time it is listed
  switch     a set outside it but inside another listed recovery's set -> "switch" to that one
  fallback   outside every listed set -> "fallback"; recovery named only if fallback_ok_on contains the set (D1 recorded, not required)
  time       the lookup uses the latest table time <= t, never a later one
"""
from __future__ import annotations

import numpy as np

from world_model.v2.stack import theta as TH
from world_model.v2.stack.checks import cover as COV


def box(lo, hi):
    return TH.ThetaSet.box(m=(lo, hi))


def _adm():
    A, B = box(1.0, 2.0), box(1.5, 3.0)
    table = {0.0: [("A", A), ("B", B)], 0.25: [("B", B)], 0.5: [("A", A), ("B", B)]}
    return TH.Admission(0, "A", A, table, box(0.0, 1.2), [], [])


def test_narrowing_keeps_current_recovery():
    adm, rng = _adm(), np.random.default_rng(41)
    for _ in range(300):
        a, b = np.sort(rng.uniform(1.0, 2.0, 2))
        for t in (0.0, 0.1, 0.5, 0.9):
            d = COV.cover(box(float(a), float(b)), adm, t)
            assert d.action == "continue" and d.recovery == "A", (a, b, t, d)


def test_switch_fallback_and_time():
    adm = _adm()
    d = COV.cover(box(2.2, 2.8), adm, 0.6)
    assert d.action == "switch" and d.recovery == "B"
    d = COV.cover(box(1.6, 1.9), adm, 0.3)                    # in A and B; at 0.25 only B is listed: B's set contains it -> switch, not continue
    assert d.action == "switch" and d.recovery == "B"
    d = COV.cover(box(0.5, 0.9), adm, 0.6)                    # outside A and B, inside fallback_ok_on
    assert d.action == "fallback" and d.recovery is not None
    d = COV.cover(box(3.5, 4.0), adm, 0.6)                    # outside everything, fallback not certified there
    assert d.action == "fallback" and d.recovery is None and "not certified" in d.why
    assert COV.cover(box(1.6, 1.9), adm, 0.2499).action == "continue"   # latest time <= t is 0.0, not 0.25


def test_timing_fields():
    adm, rng = _adm(), np.random.default_rng(42)
    th = [box(*sorted(map(float, rng.uniform(0, 4, 2)))) for _ in range(200)]
    tm = COV.cover_timing(adm, th)
    assert tm["n"] == 200 and tm["p50"] <= tm["p95"] <= tm["max"]
    assert COV.cover(box(1.2, 1.8), adm, 0.3).action == "fallback"     # at 0.25 only B (from 1.5) is listed: not covered
