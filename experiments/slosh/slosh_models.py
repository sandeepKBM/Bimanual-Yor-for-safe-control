#!/usr/bin/env python
"""slosh_models.py -- features + compact spill predictors for the learned-slosh-
constraint study (SIM DATA ONLY; no hardware). All functions are BATCHED over
episodes: arrays are (E, T, 3); loops run over time only.

Inputs are ONLY cup kinematics a robot knows from its own FK / plan: cup origin
position p(t) and cup axis n(t) (the cup is axisymmetric, so twist about n is
irrelevant). A twist-free "swing" frame R_c(t) (minimal rotation taking world z
to n) is used for every cup-frame quantity.

Models (each returns a per-step CAUSAL score, higher = more spill risk):
  (a0) static tilt            theta0 = angle(n, z)
  (a1) effective gravity      theta_eff = angle(n, f),  f = a - g  (specific force)
  (b1) WORLD-frame nonlinear spherical pendulum (FLUID picture: the free surface
       stays level while the cup rotates), bob direction u on S^2, pivot accel a:
         u'' = w^2 P_perp(u)(g_eff/|g|) - 2 zeta w u' - |u'|^2 u,   g_eff = -f
       phi = angle(-u, n)   (surface normal vs cup axis)
  (b2) CUP-frame 3-state oscillator (GRANULAR picture: the pile co-rotates with
       the cup, then relaxes toward effective gravity):
         s'' = w^2 (e_c - s) - 2 zeta w s',   e_c = R_c^T f/|f|
       phi = atan2(|s_xy|, s_z)
       -> LTI in s given e_c; e_c is a smooth function of cup accel & tilt, so
          over an MPC horizon phi is cheaply differentiable (analytic Jacobian).
  risk = phi + kappa * max(dphi/dt, 0)   (kappa: "approaching the rim" lookahead)
  (c) (b2) latent + learned residual (torch, train_eval.py)
  (d) tiny GRU on raw kinematics (world-model-like reference, train_eval.py)
"""
from __future__ import annotations

import numpy as np

G = 9.81
GVEC = np.array([0.0, 0.0, -G])
DT = 0.005


def quat_axis(quat):
    """cup +z axis in world from (...,4) wxyz quats."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def central_diff(x, dt=DT):
    """d/dt along axis -2 (time), central, edge-clamped."""
    y = np.empty_like(x)
    y[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2 * dt)
    y[..., 0, :] = y[..., 1, :]; y[..., -1, :] = y[..., -2, :]
    return y


def swing_frames(n):
    """(...,3,3) minimal rotations taking world z to n (columns = cup-frame axes).
    Rodrigues with axis z x n: R = I + [v]x + [v]x^2/(1+c)."""
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    c = np.clip(n[..., 2], -0.999, None)
    vx_, vy_ = -n[..., 1], n[..., 0]                       # v = z x n = (-ny, nx, 0)
    K = np.zeros(n.shape[:-1] + (3, 3))
    K[..., 0, 2] = vy_; K[..., 1, 2] = -vx_; K[..., 2, 0] = -vy_; K[..., 2, 1] = vx_
    R = np.eye(3) + K + (K @ K) / (1 + c)[..., None, None]
    return R


def kinematics(pos, n, dt=DT):
    """Per-step features from cup origin pos (E,T,3) + axis n (E,T,3)."""
    v = central_diff(pos, dt)
    a = central_diff(v, dt)
    j = central_diff(a, dt)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    ndot = central_diff(n, dt)
    f = a - GVEC
    R = swing_frames(n)
    rot = lambda x: np.einsum("...ji,...j->...i", R, x)   # R^T x
    fn = np.linalg.norm(f, axis=-1) + 1e-9
    return dict(v=v, a=a, j=j, n=n, ndot=ndot, f=f, R=R, f_c=rot(f), j_c=rot(j), v_c=rot(v),
                ndot_c=rot(ndot),
                theta0=np.arccos(np.clip(n[..., 2], -1, 1)),
                theta_eff=np.arccos(np.clip(np.sum(n * f, -1) / fn, -1, 1)), fnorm=fn)


def rate(x, dt=DT):
    r = np.zeros_like(x); r[..., 1:] = np.diff(x, axis=-1) / dt
    return r


def risk(phi, kappa, dt=DT):
    return phi + kappa * np.clip(rate(phi, dt), 0, None)


def pendulum_world(K, w, zeta, dt=DT):
    """(b1) batched nonlinear spherical pendulum (world frame)."""
    f = K["f"]; n = K["n"]; E, T, _ = f.shape
    u = -f[:, 0] / np.linalg.norm(f[:, 0], axis=-1, keepdims=True); ud = np.zeros((E, 3))
    phi = np.empty((E, T)); c = 2 * zeta * w
    for k in range(T):
        ge = -f[:, k] / G
        acc = w * w * (ge - np.sum(ge * u, -1, keepdims=True) * u) - c * ud \
            - np.sum(ud * ud, -1, keepdims=True) * u
        ud = ud + dt * acc
        ud -= np.sum(ud * u, -1, keepdims=True) * u
        u = u + dt * ud; u /= np.linalg.norm(u, axis=-1, keepdims=True)
        phi[:, k] = np.arccos(np.clip(-np.sum(u * n[:, k], -1), -1, 1))
    return phi


def pendulum_cup(K, w, zeta, dt=DT, return_state=False):
    """(b2) batched cup-frame oscillator, semi-implicit Euler (== the discrete
    LTI model an MPC would embed)."""
    e = K["f_c"] / K["fnorm"][..., None]
    E, T, _ = e.shape
    s = e[:, 0].copy(); sd = np.zeros((E, 3))
    S = np.empty((E, T, 3)); SD = np.empty((E, T, 3)); c = 2 * zeta * w
    for k in range(T):
        sd = sd + dt * (w * w * (e[:, k] - s) - c * sd)
        s = s + dt * sd
        S[:, k] = s; SD[:, k] = sd
    phi = np.arctan2(np.linalg.norm(S[..., :2], axis=-1), S[..., 2])
    return (phi, S, SD) if return_state else phi


def lti_world(K, w, zeta, dt=DT, return_state=False):
    """(b3) WORLD-frame LTI 2nd-order filter of the effective-gravity DIRECTION
    (linear surrogate of (b1): cup rotation acts instantly through n, cup
    accelerations are low-pass filtered):
        x'' = w^2 (e - x) - 2 zeta w x',   e = f/|f|  (world)
        phi = angle(x, n)
    Semi-implicit Euler == z_{k+1} = A z_k + B e_k with z = (x, x') in R^6."""
    e = K["f"] / K["fnorm"][..., None]; n = K["n"]
    E, T, _ = e.shape
    x = e[:, 0].copy(); xd = np.zeros((E, 3)); X = np.empty((E, T, 3)); c = 2 * zeta * w
    for k in range(T):
        xd = xd + dt * (w * w * (e[:, k] - x) - c * xd)
        x = x + dt * xd
        X[:, k] = x
    cosang = np.sum(X * n, -1) / (np.linalg.norm(X, axis=-1) + 1e-9)
    phi = np.arccos(np.clip(cosang, -1, 1))
    return (phi, X) if return_state else phi


# ------------------------------------------------------------------ metrics
def auroc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score, y):
    y = np.asarray(y, bool)
    o = np.argsort(-score, kind="stable"); ys = y[o]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1)
    return float(np.sum(prec[ys]) / max(ys.sum(), 1))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def platt_fit(score, y, iters=200):
    """1-D logistic regression p = sigmoid(A*s + B) (Newton)."""
    s = np.asarray(score, float); y = np.asarray(y, float)
    mu, sd = s.mean(), s.std() + 1e-9; z = (s - mu) / sd
    A, B = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A * z + B)))
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        wgt = p * (1 - p) + 1e-9
        H = np.array([[np.sum(wgt * z * z), np.sum(wgt * z)], [np.sum(wgt * z), np.sum(wgt)]])
        step = np.linalg.solve(H + 1e-6 * np.eye(2), g)
        A, B = A - step[0], B - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return lambda x: 1 / (1 + np.exp(-(A * (np.asarray(x) - mu) / sd + B)))
