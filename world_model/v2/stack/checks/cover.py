"""Rule 3 runtime cover lookup (checker design 4.4).  PURE: no physics, no I/O.

cover(theta_new, admission, t): given the belief set theta_new that arrived at time t during a step, answer from the precomputed
admission table whether the admitted recovery still covers it (continue), another precomputed recovery does (switch), or neither (fallback).
D1 (user 11-16): the fallback's set F is the prior box Theta_0; whether admission REQUIRES the fallback on all of Theta_0 is OPEN, so the
decision records `why` when the fallback is not certified on theta_new, and does not refuse."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

FALLBACK_NAME = "fallback"      # Admission stores no fallback name (only fallback_ok_on); label chosen here (guess)


@dataclass(frozen=True)
class CoverDecision:
    action: str                 # "continue" | "switch" | "fallback"
    recovery: "str | None"
    why: str
    lookup_s: float
    certified: bool = True      # 11-17: False = nothing is certified on theta_new (recovery None: empty / field-mismatched / uncovered set)


def branch_time(admission, t: float):
    """latest table time <= t (None if the table is empty or t is before its first time)."""
    ts = [float(k) for k in admission.table if float(k) <= float(t) + 1e-9]
    return max(ts) if ts else None


def cover(theta_new, admission, t: float) -> CoverDecision:
    t0 = time.perf_counter()

    def out(action, rec, why, cert=True):
        return CoverDecision(action, rec, why, time.perf_counter() - t0, cert)

    # 11-17 (fail closed): an EMPTY theta_new (contradictory evidence) is contained in every set (minus(...).is_empty()), which used to answer "continue"; a set over other fields raises.
    # Both answer fallback with recovery None = nothing certified.
    if theta_new is None or theta_new.is_empty():
        return out("fallback", None, "contradiction: theta_new is empty, nothing is certified", False)
    for _t, _lst in admission.table.items():
        for _n, _s in _lst:
            if not _s.is_empty() and _s.fields != theta_new.fields:
                return out("fallback", None, f"field mismatch: theta_new over {theta_new.fields}, table over {_s.fields}: nothing certified", False)
    fb_ = admission.fallback_ok_on
    if fb_ is not None and not fb_.is_empty() and fb_.fields != theta_new.fields:
        return out("fallback", None, "field mismatch with fallback_ok_on: nothing certified", False)

    tb = branch_time(admission, t)
    lst = []
    if tb is not None:
        for k, v in admission.table.items():
            if abs(float(k) - tb) < 1e-9:
                lst = list(v)
                break
    for name, s in lst:
        if name == admission.recovery and s.contains(theta_new):
            return out("continue", name, f"admitted recovery {name!r} verified on theta_new at branch t={tb}")
    for name, s in lst:
        if name != admission.recovery and s.contains(theta_new):
            return out("switch", name, f"{name!r} verified on theta_new at branch t={tb}; admitted {admission.recovery!r} is not")
    fb = admission.fallback_ok_on
    if fb is not None and fb.contains(theta_new):
        return out("fallback", FALLBACK_NAME, f"no listed recovery covers theta_new at branch t={tb}; fallback certified on it")
    return out("fallback", None, "fallback not certified on this set (D1: F = prior box; requirement OPEN, recorded not enforced)"
               + ("" if tb is not None else "; no branch time <= t in the table"), False)


def cover_timing(admission, thetas, t: "float | None" = None) -> dict:
    """wall time of cover() over a list of ThetaSets (t default: the last table time)."""
    if t is None:
        t = max((float(k) for k in admission.table), default=0.0)
    xs = np.array([cover(th, admission, t).lookup_s for th in thetas], float)
    if xs.size == 0:
        return dict(p50=None, p95=None, max=None, n=0)
    return dict(p50=float(np.percentile(xs, 50)), p95=float(np.percentile(xs, 95)), max=float(xs.max()), n=int(xs.size))
