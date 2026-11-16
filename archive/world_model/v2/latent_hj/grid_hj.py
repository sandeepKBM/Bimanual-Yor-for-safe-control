"""grid_hj.py -- EXACT (grid) planar HJ safety value for the liquid proxy.

Planar (x-z plane) restriction of slosh_pend: state x = (theta, theta_dot, alpha)
(bob angle from down, its rate, cup tilt), controls u = (a_x, w_y) from
slosh_pend.control_set(planar=True). Dynamics = the SAME 3-D vector semi-implicit
update as slosh_pend.step_np (evaluated on planar vectors and mapped back), so the
grid and the neural learner see bit-compatible dynamics.

Discounted safety Bellman (Fisac et al. 2019), semi-Lagrangian value iteration:
    V(x) = (1-g) l(x) + g * min( l(x), max_u V(f(x,u)) )          (g = gamma)
gamma = 1 gives the undiscounted (true) viability value (run to a fixed horizon).
Trilinear interpolation, clamped at the grid edges. numba-parallel.

    python -m world_model.v2.latent_hj.grid_hj --gamma 0.99 --out $SCR/grid_g099.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.latent_hj import slosh_pend as sp  # noqa: E402


@njit(cache=True, fastmath=False)
def _planar_step(th, thd, al, ax, wy, L, c2, h, sub, tilt_lim, G):
    # planar vectors (x, z) of the 3-D update (y components are identically 0)
    nx, nz = np.sin(th), -np.cos(th)
    vx, vz = thd * np.cos(th), thd * np.sin(th)
    cx, cz = np.sin(al), np.cos(al)
    for _ in range(sub):
        fx, fz = -ax, -G                      # g_vec - a  (a_z = 0)
        dp = nx * fx + nz * fz
        fx -= nx * dp; fz -= nz * dp
        # w x n with w = (0, wy, 0): (wy*nz, 0, -wy*nx)
        wnx, wnz = wy * nz, -wy * nx
        vv = vx * vx + vz * vz
        ax_ = fx / L - vv * nx - c2 * (vx - wnx)
        az_ = fz / L - vv * nz - c2 * (vz - wnz)
        vx += h * ax_; vz += h * az_
        nx += h * vx; nz += h * vz
        nn = np.sqrt(nx * nx + nz * nz); nx /= nn; nz /= nn
        dp = nx * vx + nz * vz
        vx -= nx * dp; vz -= nz * dp
        # c += h * (w x c) = h * (wy*cz, 0, -wy*cx)
        cx2 = cx + h * wy * cz; cz2 = cz - h * wy * cx
        cn = np.sqrt(cx2 * cx2 + cz2 * cz2); cx = cx2 / cn; cz = cz2 / cn
        a_ = np.arctan2(cx, cz)
        if a_ > tilt_lim:
            cx, cz = np.sin(tilt_lim), np.cos(tilt_lim)
        elif a_ < -tilt_lim:
            cx, cz = -np.sin(tilt_lim), np.cos(tilt_lim)
    th2 = np.arctan2(nx, -nz)
    thd2 = vx * np.cos(th2) + vz * np.sin(th2)
    al2 = np.arctan2(cx, cz)
    return th2, thd2, al2


@njit(cache=True)
def _interp3(V, g0, g1, g2, x0, x1, x2):
    n0, n1, n2 = V.shape
    f0 = (x0 - g0[0]) / (g0[1] - g0[0])
    f1 = (x1 - g1[0]) / (g1[1] - g1[0])
    f2 = (x2 - g2[0]) / (g2[1] - g2[0])
    f0 = min(max(f0, 0.0), n0 - 1.000001)
    f1 = min(max(f1, 0.0), n1 - 1.000001)
    f2 = min(max(f2, 0.0), n2 - 1.000001)
    i0 = int(f0); i1 = int(f1); i2 = int(f2)
    t0 = f0 - i0; t1 = f1 - i1; t2 = f2 - i2
    c00 = V[i0, i1, i2] * (1 - t0) + V[i0 + 1, i1, i2] * t0
    c01 = V[i0, i1, i2 + 1] * (1 - t0) + V[i0 + 1, i1, i2 + 1] * t0
    c10 = V[i0, i1 + 1, i2] * (1 - t0) + V[i0 + 1, i1 + 1, i2] * t0
    c11 = V[i0, i1 + 1, i2 + 1] * (1 - t0) + V[i0 + 1, i1 + 1, i2 + 1] * t0
    c0 = c00 * (1 - t1) + c10 * t1
    c1 = c01 * (1 - t1) + c11 * t1
    return c0 * (1 - t2) + c1 * t2


@njit(parallel=True, cache=True)
def _precompute_next(g0, g1, g2, U, L, c2, h, sub, tilt_lim, G):
    n0, n1, n2 = len(g0), len(g1), len(g2)
    nu = U.shape[0]
    NX = np.empty((n0, n1, n2, nu, 3))
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                for u in range(nu):
                    a, b, c = _planar_step(g0[i], g1[j], g2[k], U[u, 0], U[u, 1], L, c2, h, sub, tilt_lim, G)
                    NX[i, j, k, u, 0] = a; NX[i, j, k, u, 1] = b; NX[i, j, k, u, 2] = c
    return NX


@njit(parallel=True, cache=True)
def _bellman(V, l, NX, g0, g1, g2, gamma):
    n0, n1, n2, nu, _ = NX.shape
    out = np.empty_like(V)
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                best = -1e9
                for u in range(nu):
                    v = _interp3(V, g0, g1, g2, NX[i, j, k, u, 0], NX[i, j, k, u, 1], NX[i, j, k, u, 2])
                    if v > best:
                        best = v
                lk = l[i, j, k]
                out[i, j, k] = (1 - gamma) * lk + gamma * min(lk, best)
    return out


@njit(parallel=True, cache=True)
def _greedy(V, NX, g0, g1, g2):
    n0, n1, n2, nu, _ = NX.shape
    out = np.empty((n0, n1, n2), np.int64)
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                best = -1e9; bu = 0
                for u in range(nu):
                    v = _interp3(V, g0, g1, g2, NX[i, j, k, u, 0], NX[i, j, k, u, 1], NX[i, j, k, u, 2])
                    if v > best:
                        best = v; bu = u
                out[i, j, k] = bu
    return out


def make_grid(P: sp.PendParams, n_th=137, n_thd=121, n_al=49, th_max=1.9, thd_max=30.0):
    g0 = np.linspace(-th_max, th_max, n_th)
    g1 = np.linspace(-thd_max, thd_max, n_thd)
    g2 = np.linspace(-P.tilt_lim, P.tilt_lim, n_al)
    return g0, g1, g2


def solve(P: sp.PendParams, gamma=0.99, iters=None, tol=1e-5, grid=None, verbose=True, horizon_s=None):
    """Returns dict(V, l, grid, U, greedy, iters, resid). gamma=1 => run `horizon_s`
    worth of backups (finite-horizon viability value)."""
    g0, g1, g2 = grid if grid is not None else make_grid(P)
    U = sp.control_set(P, planar=True)[:, [0, 4]].copy()   # (a_x, w_y)
    TH, THD, AL = np.meshgrid(g0, g1, g2, indexing="ij")
    l = np.clip((P.phi_star - np.abs(TH + AL)) / P.phi_star, P.m_lo, 1.0)
    t0 = time.time()
    NX = _precompute_next(g0, g1, g2, U, P.L, 2 * P.zeta * P.w1, P.dt / P.sub, P.sub, P.tilt_lim, sp.G)
    if verbose:
        print(f"[grid] {l.shape} x {len(U)} controls, next-state table {time.time()-t0:.1f}s", flush=True)
    V = l.copy()
    if iters is None:
        iters = int(round(horizon_s / P.dt)) if gamma >= 1.0 else int(np.ceil(np.log(tol) / np.log(gamma))) + 50
    resid = np.nan
    for it in range(iters):
        Vn = _bellman(V, l, NX, g0, g1, g2, gamma)
        resid = float(np.abs(Vn - V).max())
        V = Vn
        if verbose and (it % 100 == 0 or it == iters - 1):
            print(f"[grid] it {it} resid {resid:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if gamma < 1.0 and resid < tol:
            break
    greedy = _greedy(V, NX, g0, g1, g2)
    return dict(V=V, l=l, g0=g0, g1=g1, g2=g2, U=U, greedy=greedy, iters=it + 1, resid=resid,
                gamma=gamma, P=P.as_dict())


def interp(res, th, thd, al):
    """Vectorized trilinear lookup of a solved grid value (python, for evaluation)."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((res["g0"], res["g1"], res["g2"]), res["V"], bounds_error=False, fill_value=None)
    pts = np.stack([np.clip(th, res["g0"][0], res["g0"][-1]), np.clip(thd, res["g1"][0], res["g1"][-1]),
                    np.clip(al, res["g2"][0], res["g2"][-1])], -1)
    return f(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--horizon-s", type=float, default=3.0, help="gamma=1 only")
    ap.add_argument("--res", type=float, default=1.0, help="grid resolution multiplier")
    ap.add_argument("--h", type=float, default=0.04)
    ap.add_argument("--A-max", type=float, default=5.0)
    ap.add_argument("--W-max", type=float, default=2.0)
    ap.add_argument("--m-lo", type=float, default=-1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = sp.PendParams(h=a.h, A_max=a.A_max, W_max=a.W_max, m_lo=a.m_lo)
    grid = make_grid(P, n_th=int(136 * a.res) + 1, n_thd=int(120 * a.res) + 1, n_al=int(48 * a.res) + 1)
    r = solve(P, gamma=a.gamma, grid=grid, horizon_s=a.horizon_s)
    np.savez_compressed(a.out, **{k: v for k, v in r.items() if k != "P"}, P=str(r["P"]))
    V, l = r["V"], r["l"]
    safe_l = (l > 0).mean(); safe_v = (V > 0).mean()
    print(f"[grid] done iters={r['iters']} resid={r['resid']:.1e}  frac l>0 {safe_l:.3f}  frac V>0 {safe_v:.3f}  "
          f"frac (l>0 & V<=0) {((l > 0) & (V <= 0)).mean():.3f}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_NUM_THREADS", "8")
    main()
