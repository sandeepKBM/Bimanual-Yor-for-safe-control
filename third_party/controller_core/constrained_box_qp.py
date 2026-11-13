"""Box QP extended with a small number of general linear inequality
constraints, via dual ascent on scalar Lagrange multipliers -- reusing
box_qp.solve_box_qp as the inner solver rather than writing a new active-set
QP solver from scratch.

Problem: min 0.5 x'Hx + f'x  s.t. lo <= x <= hi,  A_ineq @ x <= b_ineq

For a FIXED set of multipliers lambda >= 0 (one per linear constraint),
minimizing the Lagrangian 0.5x'Hx + f'x + lambda' @ (A_ineq @ x - b_ineq)
over the box alone is exactly solve_box_qp(H, f + A_ineq.T @ lambda, lo, hi)
-- the SAME already-tested box solver, just with a shifted linear term.

For each constraint i, g_i(lambda_i) = (A_ineq @ x(lambda))[i] - b_ineq[i]
is a continuous, monotonically non-increasing function of lambda_i (holding
the other multipliers fixed) for lambda_i >= 0 -- increasing lambda_i pushes
x harder away from violating constraint i. Complementary slackness requires
either lambda_i = 0 (constraint inactive, g_i(0) <= 0 already) or
g_i(lambda_i) = 0 (constraint active, tight). This is a classic dual
coordinate-ascent structure: root-find each lambda_i (bisection + secant,
not a fixed large iteration count -- see solve_constrained_box_qp's own
iteration budget reasoning) with the other multipliers held fixed, sweep
over constraints a bounded number of times.

Deliberately scoped to a SMALL number of constraints (real-time budget: each
root-find iteration is one solve_box_qp call, ~0.3-1ms measured on this
hardware -- see docs/status or the 2026-08-03 session notes for the actual
timing numbers this scoping is based on). Not a general-purpose QP solver.
"""

from __future__ import annotations

import numpy as np

from .box_qp import solve_box_qp

# OPTIONAL compiled hot path (2026-08-26). The Numba kernels reproduce the
# pure-Python dual-ascent below OPERATION-FOR-OPERATION (byte-identical results,
# asserted in tests/unit/test_box_qp_numba.py) but run as machine code -- ~28x
# faster on the per-cycle solve, removing the corridor-QP controller's dominant
# real-time variance. controller_core stays pure-numpy and correct if numba is
# absent: this import is guarded and the numpy path below is the fallback. Set
# USE_NUMBA = False to force the numpy path (used by the equivalence tests).
try:  # pragma: no cover - exercised only where numba is installed
    from . import _box_qp_numba as _nb

    USE_NUMBA = True
except Exception:  # numba not installed / failed to import
    _nb = None
    USE_NUMBA = False


def numba_warmup() -> bool:
    """Compile the Numba kernels now (idempotent, disk-cached) so no control
    cycle pays the one-time JIT cost. Returns True if the compiled path is
    active. No-op and False when numba is unavailable."""
    if USE_NUMBA and _nb is not None:
        _nb.warmup()
        return True
    return False


def solve_constrained_box_qp(
    hessian: np.ndarray,
    linear: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    a_ineq: np.ndarray | None = None,
    b_ineq: np.ndarray | None = None,
    *,
    max_iters: int = 80,
    tol: float = 1e-8,
    dual_sweeps: int = 4,
    dual_root_iters: int = 8,
    dual_max: float = 1.0e6,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve ``min 0.5 x'Hx + f'x`` s.t. ``lo<=x<=hi`` and ``A_ineq @ x <= b_ineq``.

    Returns (x, active_lambda, feasible). ``active_lambda`` are the final
    dual multipliers (0 for inactive constraints) -- useful for diagnostics
    (which constraint, if any, is binding). ``feasible`` is False if the box
    constraints alone make some row of A_ineq @ x <= b_ineq unreachable even
    at dual_max (reported explicitly, not silently ignored -- this is the
    real diagnostic benefit of a genuine constraint over a soft penalty: an
    infeasible problem is detected, not silently mis-answered).

    With a_ineq/b_ineq None or empty, this is exactly solve_box_qp (kept as
    a single entry point so callers don't need two code paths).
    """
    h = 0.5 * (np.asarray(hessian, dtype=np.float64) + np.asarray(hessian, dtype=np.float64).T)
    f = np.asarray(linear, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    n = int(f.shape[0])

    if a_ineq is None or b_ineq is None or np.asarray(a_ineq).size == 0:
        x = solve_box_qp(h, f, lo, hi, max_iters=max_iters, tol=tol)
        return x, np.zeros(0, dtype=np.float64), True

    a_mat = np.asarray(a_ineq, dtype=np.float64).reshape(-1, n)
    b_vec = np.asarray(b_ineq, dtype=np.float64).reshape(-1)
    m = a_mat.shape[0]

    if USE_NUMBA and _nb is not None:
        # Compiled hot path -- identical algorithm/result to the loop below.
        # `h` here is already symmetrized; the kernel re-symmetrizes and adds
        # the 1e-8 Tikhonov term itself, exactly as this function does, so pass
        # the raw `h`.
        x_nb, lam_nb, feasible_nb = _nb.constrained(
            h, f, lo, hi, a_mat, b_vec,
            int(max(1, dual_sweeps)), int(max(1, dual_root_iters)),
            int(max(1, max_iters)), float(tol), float(dual_max),
        )
        return (np.asarray(x_nb, dtype=np.float64),
                np.asarray(lam_nb, dtype=np.float64), bool(feasible_nb))

    lam = np.zeros(m, dtype=np.float64)

    def solve_for(lam_vec: np.ndarray) -> np.ndarray:
        f_shifted = f + a_mat.T @ lam_vec
        return solve_box_qp(h, f_shifted, lo, hi, max_iters=max_iters, tol=tol)

    x = solve_for(lam)
    feasible = True

    for _sweep in range(max(1, int(dual_sweeps))):
        any_active = False
        for i in range(m):

            def g(lam_i: float, _i: int = i, _lam: np.ndarray = lam) -> float:
                trial = _lam.copy()
                trial[_i] = lam_i
                x_trial = solve_for(trial)
                return float(a_mat[_i] @ x_trial - b_vec[_i])

            g0 = g(0.0)
            if g0 <= tol:
                # Constraint already satisfied with this multiplier at 0 --
                # complementary slackness holds with lambda_i = 0.
                if lam[i] != 0.0:
                    lam[i] = 0.0
                    any_active = True
                continue

            any_active = True
            # g is monotonically non-increasing in lambda_i >= 0. Bracket a
            # root via doubling, then bisect -- robust even if g is only
            # piecewise-linear (kinks from the inner box QP's own clipping),
            # where a pure secant method could overshoot.
            lo_l, hi_l = 0.0, 1.0
            g_hi = g(hi_l)
            expand_iters = 0
            while g_hi > tol and hi_l < dual_max and expand_iters < 40:
                hi_l *= 2.0
                g_hi = g(hi_l)
                expand_iters += 1
            if g_hi > tol:
                # Even at dual_max the constraint can't be satisfied within
                # the box -- genuinely infeasible (e.g. torque limits don't
                # allow holding Y this tight while also doing the rest of
                # the task). Report it, don't silently clamp.
                feasible = False
                lam[i] = hi_l
                continue
            for _ in range(max(1, int(dual_root_iters))):
                mid = 0.5 * (lo_l + hi_l)
                g_mid = g(mid)
                if abs(g_mid) <= tol:
                    lo_l = hi_l = mid
                    break
                if g_mid > 0:
                    lo_l = mid
                else:
                    hi_l = mid
            lam[i] = 0.5 * (lo_l + hi_l)

        x = solve_for(lam)
        if not any_active:
            break

    return x, lam, feasible
