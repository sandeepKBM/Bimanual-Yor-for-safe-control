"""Small dense box-constrained quadratic program solver (no external deps)."""

from __future__ import annotations

import numpy as np


def build_weighted_least_squares_qp(
    terms: list[tuple[np.ndarray, np.ndarray, float]],
    *,
    reg: float = 0.0,
    n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds (hessian, linear) for ``solve_box_qp`` from a sum of weighted
    least-squares terms plus an optional Tikhonov regularizer:

        min  reg*||x||^2 + sum_i weight_i * ||A_i @ x - b_i||^2

    Each entry in ``terms`` is (A_i, b_i, weight_i). This is the single
    shared interface for building this class of QP objective in this
    package (added 2026-08-06) -- extracted from
    cartesian_velocity_controller/modes.py's compute_ik_seeded, which
    previously built its own (task-only) hessian/linear inline, and which
    now uses this same helper for BOTH its task term and its posture term
    in one call. Centralizing this math in one place means a future QP
    controller in this package reuses proven, tested code instead of
    re-deriving the same Tikhonov-regularized weighted-least-squares
    algebra with its own inline hessian/linear construction (a real
    redundancy risk once more than one QP-based mechanism exists in the
    same file, which is exactly what motivated pulling this out here).

    Unconstrained minimizer is x = -H^-1 @ f (what solve_box_qp reduces to
    when the box bounds are permissive); with a single term and reg=0 this
    is the ordinary weighted normal equations, A^T@A @ x = A^T@b.

    Does NOT itself solve anything -- pass the returned (hessian, linear)
    straight to solve_box_qp for the actual (optionally box-constrained)
    solve, keeping "build the objective" and "solve it subject to bounds"
    as two separate, single-purpose steps."""
    if not terms:
        raise ValueError("terms must be non-empty")
    dim = int(n) if n is not None else int(np.asarray(terms[0][0]).shape[1])
    hessian = 2.0 * max(float(reg), 0.0) * np.eye(dim, dtype=np.float64)
    linear = np.zeros(dim, dtype=np.float64)
    for a, b, weight in terms:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        w = float(weight)
        hessian += 2.0 * w * (a.T @ a)
        linear += -2.0 * w * (a.T @ b)
    return hessian, linear


def solve_box_qp(
    hessian: np.ndarray,
    linear: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_iters: int = 80,
    tol: float = 1e-8,
) -> np.ndarray:
    """Solve ``min 0.5 x' H x + f' x`` subject to ``lower <= x <= upper``."""
    h = 0.5 * (np.asarray(hessian, dtype=np.float64) + np.asarray(hessian, dtype=np.float64).T)
    f = np.asarray(linear, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    n = int(f.shape[0])
    if h.shape != (n, n):
        raise ValueError(f"Hessian shape mismatch: {h.shape} vs n={n}")
    if lo.shape != (n,) or hi.shape != (n,):
        raise ValueError("Bounds must match decision vector length")
    reg = 1.0e-8
    h = h + reg * np.eye(n, dtype=np.float64)
    try:
        x = -np.linalg.solve(h, f)
    except np.linalg.LinAlgError:
        x = -np.linalg.lstsq(h, f, rcond=None)[0]
    x = np.clip(x, lo, hi)
    step = 1.0 / max(float(np.max(np.abs(h))), 1.0)
    for _ in range(max(1, int(max_iters))):
        grad = h @ x + f
        x_new = np.clip(x - step * grad, lo, hi)
        if float(np.max(np.abs(x_new - x))) <= float(tol):
            x = x_new
            break
        x = x_new
    return x
