"""sets/qp.py -- a small dedicated QP solver for the safety filter.   SIMULATION / SAVED DATA ONLY.

Problem:   min_u  1/2 (u - u_nom)' D^2 (u - u_nom)    s.t.   A u <= b        (D diagonal: the normalised-deviation weights)
i.e. the projection of u_nom onto a polytope in the metric D. With y = D (u - u_nom) this is the minimum-norm point of
{ y : A D^-1 y <= b - A u_nom }, which Hildreth's dual coordinate ascent solves (numba, no allocation in the loop):
    y = -A~' lam,   lam >= 0,   sweep i:  lam_i <- max(0, lam_i + (A~_i y - b~_i) / |A~_i|^2)
Why not the repo's Goldfarb-Idnani (experiments/realtime/rt_qp.py)? That one is exact but is wired to the FL-MPC workspace layout and to a numba
cache guard on its own files; pulling it into this package would couple two unrelated modules. Hildreth is 40 lines, converges for
feasible problems, and its failure mode (no convergence) is the infeasibility signal we want. Every solution is re-checked: the filter
accepts it only if  max(A u - b) <= tol  AND the full nonlinear predicted margin is non-negative (filter.py). The test compares it
with OSQP on random and on recorded instances.
"""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
def hildreth(At, bt, max_sweeps, tol):
    """min |y|^2/2  s.t.  At y <= bt.  Returns y (n,), n_sweeps, max_violation. Infeasible / not converged: max_violation > tol."""
    m, n = At.shape
    q = np.zeros(m)
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += At[i, j] * At[i, j]
        q[i] = s
    lam = np.zeros(m)
    y = np.zeros(n)
    sweeps = 0
    for sw in range(max_sweeps):
        sweeps = sw + 1
        chg = 0.0
        for i in range(m):
            if q[i] < 1e-18:
                continue
            r = -bt[i]
            for j in range(n):
                r += At[i, j] * y[j]
            nl = lam[i] + r / q[i]
            if nl < 0.0:
                nl = 0.0
            d = nl - lam[i]
            if d != 0.0:
                for j in range(n):
                    y[j] -= At[i, j] * d
                lam[i] = nl
                if abs(d) > chg:
                    chg = abs(d)
        # primal residual
        viol = -1e30
        for i in range(m):
            r = -bt[i]
            for j in range(n):
                r += At[i, j] * y[j]
            if q[i] < 1e-18:
                if -bt[i] > viol:
                    viol = -bt[i]
            elif r > viol:
                viol = r
        if viol <= tol and chg < 1e-9:
            return y, sweeps, max(viol, 0.0)
    viol = -1e30
    for i in range(m):
        r = -bt[i]
        for j in range(n):
            r += At[i, j] * y[j]
        if q[i] < 1e-18:
            if -bt[i] > viol:
                viol = -bt[i]
        elif r > viol:
            viol = r
    return y, sweeps, max(viol, 0.0)


def project(A, b, u_nom, w, max_sweeps=1000, tol=1e-7):
    """minimise 1/2 sum w_i^2 (u_i - u_nom_i)^2 s.t. A u <= b.  w (n,) = D diagonal.
    Rows are normalised to unit length in the scaled space so that `tol` is a distance. Returns (u, ok, n_sweeps, max_violation)."""
    A = np.ascontiguousarray(A, np.float64)
    if len(A) == 0:
        return u_nom.copy(), True, 0, 0.0
    Ad = A / w[None, :]
    nr = np.linalg.norm(Ad, axis=1)
    keep = nr > 1e-12
    if not keep.all():
        if np.any(b[~keep] < -1e-9):
            return u_nom.copy(), False, 0, float(-(b[~keep]).min())
        Ad, A, b = Ad[keep], A[keep], b[keep]
        nr = nr[keep]
    At = Ad / nr[:, None]
    bt = (b - A @ u_nom) / nr
    y, sw, viol = hildreth(np.ascontiguousarray(At), np.ascontiguousarray(bt), max_sweeps, tol)
    u = u_nom + y / w
    ok = bool(viol <= 5e-4)
    return u, ok, int(sw), float(viol)
