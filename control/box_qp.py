"""Small dense box-constrained QP solver — no external deps.

Ported verbatim (behaviour-preserving) from
``/common/users/ss5772/real_Cartpole/controller_core/box_qp.py`` so the
``bimanual/control`` package is self-contained and does not depend on
``controller_core`` being on the path (it is not, in this repo's env).

Solves ``min 0.5 x' H x + f' x   s.t.  lower <= x <= upper`` by a projected
(clipped) gradient descent seeded at the clipped unconstrained minimiser.
"""

from __future__ import annotations

import numpy as np


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
