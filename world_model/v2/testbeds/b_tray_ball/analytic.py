"""analytic.py -- reduced analytic models of a ball on a kinematically driven tray (numba, CPU).

NE  (full Newton-Euler rolling model, "9x9"): world-frame ball state (p, v, omega) = 9 numbers; tray = kinematic pose from
    bead_env's kin_step. Per 5 ms step solve  [m A - F = m g ; k m r^2 alpha + r n x F = tau_rr ; A + r n x alpha = A_C(+...)]
    for (A, alpha, F). Rolling regime by default; switches to sliding when |F_t| > mu N; airborne if N < 0.
    Includes the tray's rotation (centrifugal, Euler, Coriolis) exactly. Rolling resistance tau_rr = -mu_roll N tanh(|w_rel|/eps) w_rel_hat.
QS  (quasi-static 2-D model, "5/7"): rho'' = ((g - a)_tangential - (mu_roll/r) (N/m) dir) / (1 + k)  in the tray frame; drops all
    tray-rotation terms. State (x, y, vx, vy) = 4. This is the textbook a = 5/7 g sin(theta) with the tray-acceleration pseudo force.
Parameters: r, m, k (= I/(m r^2)), mu, mu_roll. Mass cancels in rolling; r only enters the margin and mu_roll / r.
"""
from __future__ import annotations

import numpy as np
import numba as nb

from world_model.v2.testbeds.b_tray_ball.tray_env import _kin_step_nb, DT, G, HX, HY, sdf_rect

GVEC = np.array([0.0, 0.0, -G])


@nb.njit(cache=True)
def _cross_mat(n):
    return np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])


@nb.njit(cache=True)
def _solve_ne(vb, wb, pb, n, pO, vO, aO, Om, Omd, r, m, k, tau_rr):
    nn = _cross_mat(n)
    ndot = np.cross(Om, n)
    s = pb - pO - r * n
    rhs3 = aO + np.cross(Omd, s) + np.cross(Om, vb - r * ndot - vO) + r * np.cross(wb, ndot)
    M = np.zeros((9, 9))
    b = np.zeros(9)
    for i in range(3):
        M[i, i] = m
        M[i, 6 + i] = -1.0
        b[i] = m * GVEC[i]
        M[3 + i, 3 + i] = k * m * r * r
        M[6 + i, i] = 1.0
        b[3 + i] = tau_rr[i]
        b[6 + i] = rhs3[i]
    for i in range(3):
        for j in range(3):
            M[3 + i, 6 + j] = r * nn[i, j]
            M[6 + i, 3 + j] = r * nn[i, j]
    x = np.linalg.solve(M, b)
    return x[0:3], x[3:6], x[6:9], rhs3


@nb.njit(cache=True)
def ne_step(pb, vb, wb, pO, vO, n, Om, Omd, aO, r, m, k, mu, mu_roll, dt):
    """advance the ball one step given the tray axis n, tray angular velocity Om and its rate Omd.
    Returns pb, vb, wb, N, mode (0 roll, 1 slide, 2 air)."""
    z3 = np.zeros(3)
    A, al, F, rhs3 = _solve_ne(vb, wb, pb, n, pO, vO, aO, Om, Omd, r, m, k, z3)
    N = F @ n
    mode = 0
    if N < 0.0:
        mode = 2
        A = GVEC.copy()
        al = z3
    else:
        # rolling resistance (smooth Coulomb on the relative tangential spin)
        wrel = wb - Om
        wt = wrel - (wrel @ n) * n
        nw = np.sqrt(wt @ wt)
        if mu_roll > 0.0:
            tau = -mu_roll * N * np.tanh(nw / 0.02) * wt / (nw + 1e-12)
            A, al, F, rhs3 = _solve_ne(vb, wb, pb, n, pO, vO, aO, Om, Omd, r, m, k, tau)
            N = F @ n
        Ft = F - N * n
        if N > 0.0 and np.sqrt(Ft @ Ft) > mu * N:
            mode = 1
            vC = vO + np.cross(Om, pb - pO - r * n)
            vs = vb - r * np.cross(wb, n) - vC
            vs = vs - (vs @ n) * n
            nvs = np.sqrt(vs @ vs)
            if nvs > 1e-4:
                dirn = -vs / nvs
            else:
                dirn = Ft / np.sqrt(Ft @ Ft)
            N = m * (n @ (rhs3 - GVEC))
            if N < 0.0:
                N = 0.0
            F = N * n + mu * N * dirn
            A = F / m + GVEC
            al = -r * np.cross(n, F) / (k * m * r * r)
    wb2 = wb + al * dt
    vb2 = vb + A * dt
    pb2 = pb + vb2 * dt
    return pb2, vb2, wb2, N, mode


def ne_rollout(pb, vb, wb, pos, v, quat, U, p, steps_per_tick=4, tilt_lim=1.5, Om_prev=None, lag=0):
    # lag=0: use the tray rotation of the CURRENT interval; lag=1: the previous interval (MuJoCo pins K_k and carries the
    # finite-difference velocity of K_{k-1}->K_k, i.e. one step late).
    """roll the NE model under the held action sequence U (K,6) (one row per 50 Hz tick). Returns loc (K,3) tray-frame ball
    centre at the END of each tick, d (K,) edge distance (sdf), Om_last."""
    Om_prev = np.zeros(3) if Om_prev is None else np.array(Om_prev, float)
    Om_pp = Om_prev.copy()
    K = len(U)
    locs = np.zeros((K, 3))
    ds = np.zeros(K)
    pos, v, quat = np.array(pos, float), np.array(v, float), np.array(quat, float)
    pb, vb, wb = np.array(pb, float), np.array(vb, float), np.array(wb, float)
    for t in range(K):
        a, w = np.ascontiguousarray(U[t, :3], np.float64), np.ascontiguousarray(U[t, 3:], np.float64)
        for _ in range(steps_per_tick):
            c0 = _axis(quat)
            pos2, v2, quat2 = _kin_step_nb(pos, v, quat, a, w, tilt_lim, DT)
            c1 = _axis(quat2)
            Om_new = np.cross(c0, c1) / DT
            if lag:
                Om_eq, Omd = Om_prev, (Om_prev - Om_pp) / DT
            else:
                Om_eq, Omd = Om_new, (Om_new - Om_prev) / DT
            pb, vb, wb, N, mode = ne_step(pb, vb, wb, pos, v2, c0, Om_eq, Omd, a, p.radius, p.mass,
                                          p.inertia_k, p.mu, p.mu_roll, DT)
            Om_pp, Om_prev = Om_prev, Om_new
            pos, v, quat = pos2, v2, quat2
        R = _rot(quat)
        loc = (pb - pos) @ R
        locs[t] = loc
        ds[t] = sdf_rect(loc[0], loc[1], p_lip(p))
    return locs, ds, Om_prev


_LIP = {}


def p_lip(p):
    return _LIP.get("lip", False)


def set_lip(flag):
    _LIP["lip"] = bool(flag)


@nb.njit(cache=True)
def _axis(q):
    return np.array([2.0 * (q[1] * q[3] + q[0] * q[2]), 2.0 * (q[2] * q[3] - q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])])


@nb.njit(cache=True)
def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------------------------------------ QS (5/7) model
@nb.njit(cache=True)
def qs_rollout(x0, y0, vx0, vy0, pos, v, quat, U, r, k, mu_roll, steps_per_tick, tilt_lim, dt):
    """tray-frame 2-D rolling model. (vx, vy) = ball velocity relative to the tray surface frame (tray rotation ignored).
    Returns xs, ys (K,) at the end of each tick."""
    K = U.shape[0]
    xs = np.zeros(K)
    ys = np.zeros(K)
    x, y, vx, vy = x0, y0, vx0, vy0
    for t in range(K):
        a = U[t, :3].copy()
        w = U[t, 3:].copy()
        for _ in range(steps_per_tick):
            c = _axis(quat)
            pos2, v2, quat2 = _kin_step_nb(pos, v, quat, a, w, tilt_lim, dt)
            R = _rot(quat)
            ge = GVEC - a                            # effective gravity in the world
            gl = R.T @ ge                            # tray-frame components
            Nm = -gl[2]                              # N/m
            if Nm < 0.0:
                Nm = 0.0
            sp = np.sqrt(vx * vx + vy * vy)
            ax_ = gl[0]
            ay_ = gl[1]
            fr = (mu_roll / r) * Nm * np.tanh(sp / 0.01)
            if sp > 1e-9:
                ax_ -= fr * vx / sp
                ay_ -= fr * vy / sp
            else:
                gt = np.sqrt(ax_ * ax_ + ay_ * ay_)
                lim = (mu_roll / r) * Nm
                if gt <= lim:
                    ax_, ay_ = 0.0, 0.0
                else:
                    ax_ -= lim * ax_ / gt
                    ay_ -= lim * ay_ / gt
            vx += ax_ / (1.0 + k) * dt
            vy += ay_ / (1.0 + k) * dt
            x += vx * dt
            y += vy * dt
            pos, v, quat = pos2, v2, quat2
        xs[t] = x
        ys[t] = y
    return xs, ys
