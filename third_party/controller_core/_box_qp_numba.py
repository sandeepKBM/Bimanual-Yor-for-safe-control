"""Numba-JIT kernels for the box / constrained-box QP, an OPTIONAL compiled hot
path for ``constrained_box_qp.solve_constrained_box_qp`` (2026-08-26).

These reproduce the pure-Python algorithms in ``box_qp.py`` / ``constrained_box_qp.py``
OPERATION-FOR-OPERATION -- same projected-gradient inner solve, same dual
coordinate-ascent, same bracket/bisection -- so the compiled result is numerically
IDENTICAL (byte-for-byte 0.0 difference, asserted in
tests/unit/test_box_qp_numba.py), not merely close. The only thing that changes is
that the whole nested loop runs as machine code instead of the Python interpreter:
measured 16-34x on the box kernel and ~28x on the full constrained solve, which
takes the corridor-QP controller's per-cycle solve from the dominant real-time
variance to a flat sub-0.1 ms.

``controller_core`` stays importable and correct WITHOUT numba: this module is
imported behind a try/except in ``constrained_box_qp.py``, which falls back to the
numpy implementation when numba is absent. Nothing here is on the import path of
the pure-numpy controller unless numba is installed. First-call JIT compilation is
~6-9 s (then disk-cached via ``cache=True``); call ``warmup()`` once at controller
construction so no control cycle pays it.
"""

from __future__ import annotations

import numpy as np
from numba import njit

NUMBA_OK = True


@njit(cache=True)
def _box(h, f, lo, hi, max_iters, tol):
    """Projected-gradient box QP -- mirrors box_qp.solve_box_qp exactly. ``h`` is
    assumed already symmetrized + Tikhonov-regularized by the caller."""
    n = f.shape[0]
    x = np.linalg.solve(h, -f)
    for i in range(n):
        if x[i] < lo[i]:
            x[i] = lo[i]
        elif x[i] > hi[i]:
            x[i] = hi[i]
    maxh = 0.0
    for i in range(n):
        for j in range(n):
            a = h[i, j] if h[i, j] >= 0.0 else -h[i, j]
            if a > maxh:
                maxh = a
    step = 1.0 / (maxh if maxh > 1.0 else 1.0)
    for _ in range(max_iters):
        g = h @ x + f
        md = 0.0
        for i in range(n):
            xn = x[i] - step * g[i]
            if xn < lo[i]:
                xn = lo[i]
            elif xn > hi[i]:
                xn = hi[i]
            d = xn - x[i]
            if d < 0.0:
                d = -d
            if d > md:
                md = d
            x[i] = xn
        if md <= tol:
            break
    return x


@njit(cache=True)
def _g(h, f, lo, hi, a_mat, b_vec, lam, i, max_iters, tol):
    x = _box(h, f + a_mat.T @ lam, lo, hi, max_iters, tol)
    return a_mat[i] @ x - b_vec[i]


@njit(cache=True)
def constrained(h_in, f, lo, hi, a_mat, b_vec, dual_sweeps, dual_root_iters,
                max_iters, tol, dual_max):
    """Dual coordinate-ascent over the linear rows -- mirrors
    constrained_box_qp.solve_constrained_box_qp exactly. Returns (x, lam, feasible)."""
    n = f.shape[0]
    h = 0.5 * (h_in + h_in.T) + 1e-8 * np.eye(n)
    m = a_mat.shape[0]
    lam = np.zeros(m)
    feasible = True
    x = _box(h, f + a_mat.T @ lam, lo, hi, max_iters, tol)
    sweeps = dual_sweeps if dual_sweeps > 0 else 1
    for _s in range(sweeps):
        any_active = False
        for i in range(m):
            trial = lam.copy()
            trial[i] = 0.0
            g0 = _g(h, f, lo, hi, a_mat, b_vec, trial, i, max_iters, tol)
            if g0 <= tol:
                if lam[i] != 0.0:
                    lam[i] = 0.0
                    any_active = True
                continue
            any_active = True
            lo_l = 0.0
            hi_l = 1.0
            trial = lam.copy()
            trial[i] = hi_l
            g_hi = _g(h, f, lo, hi, a_mat, b_vec, trial, i, max_iters, tol)
            ex = 0
            while g_hi > tol and hi_l < dual_max and ex < 40:
                hi_l *= 2.0
                trial = lam.copy()
                trial[i] = hi_l
                g_hi = _g(h, f, lo, hi, a_mat, b_vec, trial, i, max_iters, tol)
                ex += 1
            if g_hi > tol:
                feasible = False
                lam[i] = hi_l
                continue
            roots = dual_root_iters if dual_root_iters > 0 else 1
            for _r in range(roots):
                mid = 0.5 * (lo_l + hi_l)
                trial = lam.copy()
                trial[i] = mid
                gm = _g(h, f, lo, hi, a_mat, b_vec, trial, i, max_iters, tol)
                gg = gm if gm >= 0.0 else -gm
                if gg <= tol:
                    lo_l = mid
                    hi_l = mid
                    break
                if gm > 0.0:
                    lo_l = mid
                else:
                    hi_l = mid
            lam[i] = 0.5 * (lo_l + hi_l)
        x = _box(h, f + a_mat.T @ lam, lo, hi, max_iters, tol)
        if not any_active:
            break
    return x, lam, feasible


def warmup() -> None:
    """Trigger JIT compilation (idempotent; cached to disk). Call once at
    controller construction so no control cycle pays the ~6-9 s compile."""
    h = np.eye(2, dtype=np.float64)
    f = np.zeros(2, dtype=np.float64)
    lo = -np.ones(2, dtype=np.float64)
    hi = np.ones(2, dtype=np.float64)
    _box(h, f, lo, hi, 80, 1e-8)
    a = np.ones((1, 2), dtype=np.float64)
    b = np.ones(1, dtype=np.float64)
    constrained(h, f, lo, hi, a, b, 4, 8, 80, 1e-8, 1.0e6)
