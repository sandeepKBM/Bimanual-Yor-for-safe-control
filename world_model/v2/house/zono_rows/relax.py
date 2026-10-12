"""relax.py -- minimal-uniform-slack search for the zonotope HOCBF rows (SIMULATION ONLY, pure python).

When the hard QP is infeasible (FL-MPC falls back: slack on its own arm-arm rows or a = 0), the zonotope rows can be loosened by one scalar s
(m/s^2 of keypoint-normal acceleration: b_zono += s, every zonotope row of both arms, nothing else).  Feasibility of {rows with b + s} is
MONOTONE in s, so the smallest certified s is found by a geometric bracket search followed by a few bisections:

    start s = max(s0, s_prev/2)
      certified      -> halve while still certified (down to s0) to find a failing lower bracket   [only when started above s0]
      not certified  -> double (clamped to s_max) until certified or s_max has failed (-> unrecovered)
    then `n_bisect` bisections between the last failure and the first success; return the smallest certified s.
    Every `solve(s)` call is counted; the total is capped at `max_solves` (the budget may cut the bisection short).

The bracket ratio hi/lo is <= 2 before bisection, so the returned s is within 2^-n_bisect (12.5% for n_bisect=3) of the true minimum
(absolute error <= s0 / 2^n_bisect = 0.0063 m/s^2 when the minimum lies below s0).
"""
from __future__ import annotations

import numpy as np


def relax_search(solve, s_prev=0.0, s0=0.05, s_max=50.0, n_bisect=3, max_solves=12):
    """solve(s) -> bool (True = certified at slack s).  s = 0 is assumed to have FAILED already (not counted).
    Returns dict(s=smallest certified s or None, n_solves, certified, history=[(s, ok)])."""
    hist = []

    def call(s):
        ok = bool(solve(float(s)))
        hist.append((float(s), ok))
        return ok

    lo, hi = 0.0, None
    s = max(s0, 0.5 * s_prev)
    if call(s):
        hi = s
        while hi > s0 and len(hist) < max_solves:                 # started above s0: look for the failing lower bracket
            t = max(0.5 * hi, s0)
            if call(t):
                hi = t
            else:
                lo = t
                break
    else:
        lo = s
        while len(hist) < max_solves and lo < s_max:
            t = min(2.0 * lo, s_max)
            if call(t):
                hi = t
                break
            lo = t
    if hi is None:
        return dict(s=None, n_solves=len(hist), certified=False, history=hist)
    for _ in range(n_bisect):
        if len(hist) >= max_solves:
            break
        mid = 0.5 * (lo + hi)
        if call(mid):
            hi = mid
        else:
            lo = mid
    return dict(s=float(hi), n_solves=len(hist), certified=True, history=hist)


# --------------------------------------------------------------------------------------------------------------- selective (per-row) slack
def screen_hopeless(A, b, tau_lim, frac=0.9):
    """per-row slack vector s (>= 0) for the rows  a @ tau <= b  that are (nearly) impossible even at full torque, |tau_j| <= tau_lim.
    The most negative a @ tau reachable inside the box is  -tau_lim * sum|a|  (=: -t).  A row with b < -frac * t demands more than `frac` of that
    best effort -> its right-hand side is lifted to -frac * t  (s = -frac * t - b), every other row gets s = 0.  Pure, vectorised."""
    b = np.asarray(b, float)
    if not len(b):
        return np.zeros(0)
    A = np.asarray(A, float)
    t = tau_lim * np.abs(A).sum(1)
    return np.maximum(0.0, -frac * t - b)


def residual_slack(A, b, tau, tol=1e-6, bump=1e-3, floor=1e-4):
    """per-row slack that EXACTLY legitimises `tau` (the torque of a solve that certified under some larger slack): s_i = max(0, a_i tau - b_i),
    bumped by a hair so the row is not on its boundary, and 0 for every row tau already satisfies.  Since tau satisfies  a_i tau <= b_i + s_i
    for all i, it stays feasible for the QP with those rows (feasible set only grew), and s <= any uniform slack that tau certified."""
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if not len(b):
        return np.zeros(0)
    r = A @ np.asarray(tau, float) - b
    return np.where(r > tol, r * (1.0 + bump) + floor, 0.0)
