"""slosh_pend.py -- analytic liquid proxy used as the Stage-0 GROUND-TRUTH latent.

Same model as world_model/v2/data/contents_study.pendulum_run (which produced the
spill_v2 `pend_*` labels): equivalent-mechanical first slosh mode of a cylindrical
tank (Abramson/Ibrahim) = spherical pendulum of length L = g/w1^2,
w1^2 = (g*xi/R) tanh(xi*h/R), xi = 1.841, pivot on the cup axis at the free-surface
height h. Bob direction n (unit, from pivot to bob; hangs along effective gravity):

    f   = (g_vec - a_pivot) projected orthogonal to n
    n'' = f/L - |n'|^2 n - 2 zeta w1 (n' - omega x n)        (omega = cup angular vel)
    phi = angle(-n, c)                                        (c = cup axis)
    spill  <=>  phi >= phi* = atan((H - h)/R)

Latent state z = (n, n', c) (9 numbers, 6 DOF). For reachability the FALLBACK
controls are u = (a_pivot (3), omega (3)); the fallback set used for training
keeps a_z = 0 and omega_z = 0 (horizontal accel + tilt rate), and the cup tilt is
kept inside TILT_LIM (projected), consistently in 3-D and in the planar grid.

Everything is written twice: numpy (observer / sim / grid) and torch (batched,
differentiable: learner + controller linearization). SIMULATION ONLY.
"""
from __future__ import annotations

import numpy as np

G = 9.81
GVEC = np.array([0.0, 0.0, -G])
CUP_INNER_R = 0.0285            # == contents.CUP_INNER_R
CUP_HEIGHT = 0.080              # == contents.CUP_HEIGHT
XI = 1.841


class PendParams:
    """Liquid-proxy constants for fill height h [m]."""

    def __init__(self, h=0.04, zeta=0.02, A_max=5.0, W_max=2.0, tilt_lim=1.5, dt=0.01, sub=4, m_lo=-1.0):
        self.h, self.zeta = float(h), float(zeta)
        self.w1 = float(np.sqrt(G * XI / CUP_INNER_R * np.tanh(XI * h / CUP_INNER_R)))
        self.L = G / self.w1 ** 2
        self.phi_star = float(np.arctan((CUP_HEIGHT - h) / CUP_INNER_R))
        self.A_max, self.W_max, self.tilt_lim = float(A_max), float(W_max), float(tilt_lim)
        self.dt, self.sub = float(dt), int(sub)       # Bellman step (control held) and sub-steps
        self.m_lo = float(m_lo)                        # lower clip of the normalized margin (v0: -1)

    def as_dict(self):
        return dict(h=self.h, zeta=self.zeta, w1=self.w1, L=self.L, phi_star=self.phi_star,
                    A_max=self.A_max, W_max=self.W_max, tilt_lim=self.tilt_lim, dt=self.dt, sub=self.sub,
                    m_lo=self.m_lo)

    @staticmethod
    def from_meta(d):
        keys = ("h", "zeta", "A_max", "W_max", "tilt_lim", "dt", "sub")
        return PendParams(**{k: d[k] for k in keys}, m_lo=d.get("m_lo", -1.0))


# =============================================================================== numpy
def margin_np(n, c, P: PendParams):
    """normalized spill margin l = (phi* - phi)/phi*, clipped to [-1, 1]; (...,3) inputs."""
    cs = np.clip(-np.sum(n * c, -1), -1.0, 1.0)
    phi = np.arccos(cs)
    return np.clip((P.phi_star - phi) / P.phi_star, P.m_lo, 1.0)


def phi_np(n, c):
    return np.arccos(np.clip(-np.sum(n * c, -1), -1.0, 1.0))


def _project_tilt_np(c, tilt_lim):
    cz = np.clip(c[..., 2], -1, 1)
    tilt = np.arccos(cz)
    over = tilt > tilt_lim
    if np.any(over):
        hxy = c[..., :2] / (np.linalg.norm(c[..., :2], axis=-1, keepdims=True) + 1e-12)
        cc = np.concatenate([hxy * np.sin(tilt_lim), np.full(c.shape[:-1] + (1,), np.cos(tilt_lim))], -1)
        c = np.where(over[..., None], cc, c)
    return c


def step_np(n, nd, c, a, w, P: PendParams, dt=None, sub=None, project=True):
    """Advance (n, n', c) by dt under constant pivot accel a (...,3) and cup angular
    velocity w (...,3). Semi-implicit Euler with `sub` sub-steps (== pendulum_run)."""
    dt = P.dt if dt is None else dt
    sub = P.sub if sub is None else sub
    h = dt / sub
    c2 = 2.0 * P.zeta * P.w1
    n, nd, c = n.copy(), nd.copy(), c.copy()
    for _ in range(sub):
        f = GVEC - a
        f = f - n * np.sum(n * f, -1, keepdims=True)
        ndd = f / P.L - np.sum(nd * nd, -1, keepdims=True) * n - c2 * (nd - np.cross(w, n))
        nd = nd + h * ndd
        n = n + h * nd
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        nd = nd - n * np.sum(n * nd, -1, keepdims=True)
        c = c + h * np.cross(w, c)
        c = c / np.linalg.norm(c, axis=-1, keepdims=True)
        if project:
            c = _project_tilt_np(c, P.tilt_lim)
    return n, nd, c


class PendulumObserver:
    """Causal online liquid-proxy simulator driven by the REALIZED cup motion (the
    ground-truth 'contents' in Stage 0/2). Call update(pivot_pos, cup_axis) once per
    sim step (dt_sim); it finite-differences the pivot acceleration (backward) and
    the cup angular velocity, and integrates with sub-steps (no tilt projection:
    this is the physics, not the fallback abstraction)."""

    def __init__(self, P: PendParams, dt_sim=0.005, sub=4):
        self.P, self.dt, self.sub = P, float(dt_sim), int(sub)
        self.reset()

    def reset(self, c0=None):
        self.p_hist = []
        self.c_prev = None
        c0 = np.array([0.0, 0.0, 1.0]) if c0 is None else np.asarray(c0, float)
        self.n = -c0 / np.linalg.norm(c0)
        self.nd = np.zeros(3)
        self.c = c0.copy()
        self.phi = 0.0
        self.a_last = np.zeros(3)
        self.w_last = np.zeros(3)

    def state(self):
        return self.n.copy(), self.nd.copy(), self.c.copy()

    def update(self, pivot, c):
        pivot = np.asarray(pivot, float); c = np.asarray(c, float) / np.linalg.norm(c)
        self.p_hist.append(pivot.copy())
        if len(self.p_hist) > 3:
            self.p_hist.pop(0)
        if self.c_prev is None:
            self.c_prev = c.copy(); self.c = c.copy(); self.n = -c.copy(); self.nd[:] = 0.0
            self.phi = 0.0
            return self.phi
        if len(self.p_hist) >= 3:
            a = (self.p_hist[-1] - 2 * self.p_hist[-2] + self.p_hist[-3]) / self.dt ** 2
        else:
            a = np.zeros(3)
        # angular velocity of the axis (component orthogonal to c; twist is irrelevant)
        w = np.cross(self.c_prev, c) / self.dt
        self.a_last, self.w_last = a, w
        h = self.dt / self.sub
        c2 = 2.0 * self.P.zeta * self.P.w1
        n, nd = self.n, self.nd
        for _ in range(self.sub):
            f = GVEC - a
            f = f - n * (n @ f)
            ndd = f / self.P.L - (nd @ nd) * n - c2 * (nd - np.cross(w, n))
            nd = nd + h * ndd
            n = n + h * nd
            n = n / np.linalg.norm(n)
            nd = nd - n * (n @ nd)
        self.n, self.nd, self.c, self.c_prev = n, nd, c.copy(), c.copy()
        self.phi = float(np.arccos(np.clip(-n @ c, -1, 1)))
        return self.phi


def replay_np(pivot, axis, P: PendParams, dt_sim=0.005, sub=4):
    """Offline replay of an episode's cup pivot/axis (T,3) -> n, nd, c, phi (T,...)."""
    ob = PendulumObserver(P, dt_sim, sub)
    T = len(pivot)
    N = np.zeros((T, 3)); ND = np.zeros((T, 3)); Cc = np.zeros((T, 3)); PHI = np.zeros(T)
    for k in range(T):
        PHI[k] = ob.update(pivot[k], axis[k])
        N[k], ND[k], Cc[k] = ob.state()
    return N, ND, Cc, PHI


# =============================================================================== planar embedding
def planar_to_3d(th, thd, al):
    """planar (x-z plane) state -> (n, nd, c). th: bob angle from down (toward +x),
    thd its rate, al: cup axis tilt toward +x. phi = |th + al|."""
    th, thd, al = np.broadcast_arrays(np.asarray(th, float), np.asarray(thd, float), np.asarray(al, float))
    z = np.zeros_like(th)
    n = np.stack([np.sin(th), z, -np.cos(th)], -1)
    nd = thd[..., None] * np.stack([np.cos(th), z, np.sin(th)], -1)
    c = np.stack([np.sin(al), z, np.cos(al)], -1)
    return n, nd, c


def planar_from_3d(n, nd, c):
    th = np.arctan2(n[..., 0], -n[..., 2])
    thd = np.sum(nd * np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], -1), -1)
    al = np.arctan2(c[..., 0], c[..., 2])
    return th, thd, al


# =============================================================================== controls
def control_set(P: PendParams, n_dir=8, mags=(0.5, 1.0), planar=False):
    """Discrete fallback control set U = {(a (3), w (3))}. 3-D: 1 + n_dir*len(mags) horizontal
    accels x {0, +-W x, +-W y} tilt rates. planar: a_x in {-A,-A/2,0,A/2,A} x w_y in {-W,0,W}."""
    if planar:
        A = [np.array([s * P.A_max, 0, 0]) for s in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        W = [np.array([0, s * P.W_max, 0]) for s in (-1.0, 0.0, 1.0)]
    else:
        A = [np.zeros(3)]
        for m in mags:
            for k in range(n_dir):
                ang = 2 * np.pi * k / n_dir
                A.append(np.array([m * P.A_max * np.cos(ang), m * P.A_max * np.sin(ang), 0.0]))
        W = [np.zeros(3), np.array([P.W_max, 0, 0]), np.array([-P.W_max, 0, 0]),
             np.array([0, P.W_max, 0]), np.array([0, -P.W_max, 0])]
    U = np.array([np.concatenate([a, w]) for a in A for w in W])
    return U


# =============================================================================== torch
def torch_fns():
    """Lazily import torch and return the batched differentiable functions."""
    import torch

    gvec = None

    def _g(ref):
        nonlocal gvec
        if gvec is None or gvec.device != ref.device or gvec.dtype != ref.dtype:
            gvec = torch.tensor([0.0, 0.0, -G], device=ref.device, dtype=ref.dtype)
        return gvec

    def margin_t(n, c, P: PendParams):
        cs = torch.clamp(-(n * c).sum(-1), -1 + 1e-6, 1 - 1e-6)
        phi = torch.arccos(cs)
        return torch.clamp((P.phi_star - phi) / P.phi_star, P.m_lo, 1.0)

    def phi_t(n, c):
        return torch.arccos(torch.clamp(-(n * c).sum(-1), -1 + 1e-6, 1 - 1e-6))

    def project_tilt_t(c, tilt_lim):
        cz = torch.clamp(c[..., 2], -1 + 1e-7, 1 - 1e-7)
        over = torch.arccos(cz) > tilt_lim
        hxy = c[..., :2] / (c[..., :2].norm(dim=-1, keepdim=True) + 1e-12)
        cc = torch.cat([hxy * np.sin(tilt_lim), torch.full_like(c[..., 2:], np.cos(tilt_lim))], -1)
        return torch.where(over[..., None], cc, c)

    def step_t(n, nd, c, a, w, P: PendParams, dt=None, sub=None, project=True):
        dt = P.dt if dt is None else dt
        sub = P.sub if sub is None else sub
        h = dt / sub
        c2 = 2.0 * P.zeta * P.w1
        g = _g(n)
        for _ in range(sub):
            f = g - a
            f = f - n * (n * f).sum(-1, keepdim=True)
            ndd = f / P.L - (nd * nd).sum(-1, keepdim=True) * n - c2 * (nd - torch.cross(w.expand_as(n), n, dim=-1))
            nd = nd + h * ndd
            n = n + h * nd
            n = n / n.norm(dim=-1, keepdim=True)
            nd = nd - n * (n * nd).sum(-1, keepdim=True)
            c = c + h * torch.cross(w.expand_as(c), c, dim=-1)
            c = c / c.norm(dim=-1, keepdim=True)
            if project:
                c = project_tilt_t(c, P.tilt_lim)
        return n, nd, c

    def invariant_features(n, nd, c, nd_scale=0.04):
        """Smooth yaw-invariant features (the value is invariant to rotation about
        world z): z-components + planar dot/cross products of the horizontal parts.
        nd is scaled by ~1/w1 so every feature is O(1)."""
        nd = nd * nd_scale
        nh, ndh, ch = n[..., :2], nd[..., :2], c[..., :2]
        dot = lambda a, b: (a * b).sum(-1)
        crs = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return torch.stack([n[..., 2], c[..., 2], nd[..., 2],
                            dot(nh, ch), crs(nh, ch),
                            dot(ndh, nh), crs(nh, ndh),
                            dot(ndh, ch), crs(ch, ndh),
                            dot(ndh, ndh)], -1)

    return dict(margin=margin_t, phi=phi_t, step=step_t, feats=invariant_features)
