"""vdirect/pess.py -- baselines that fix the max WITHOUT new data. SIMULATION / SAVED DATA ONLY.

(M) in-data max (FISOR / IQL-style behaviour-support restriction): the Fisac max and the greedy fallback range only
    over the fallback-set elements whose nearest-u occupancy in the TRAIN data is >= 0.1% of ticks
    (diagnose.py coverage quantization: d = |da|^2/5^2 + |dw|^2/2^2).
(P) uncertainty-pessimistic value (UNISafe / Cao & Bansal-style): an E-member bootstrap ensemble of one-step
    next-observation heads g_e(z, u) -> o_n' on the frozen WM latent gives sigma(z, u) = mean_dim std_e; a
    transition whose sigma exceeds tau (= 99th percentile of sigma on held-out VAL transitions under the logged
    actions) is treated as a failure: the successor's margin is min(l^(z'), 1 - sigma/tau), clipped to [-1, 1].
"""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.wm_audit import common as C


def in_data_actions(thresh=1e-3):
    tr = C.load_split("train")
    A = tr["act"].astype(np.float64)
    U = C.fallback_U()
    cnt = np.zeros(len(U))
    for i in range(0, len(A), 200000):
        a = A[i:i + 200000]
        d = (np.linalg.norm((a[:, None, :3] - U[None, :, :3]) / 5.0, axis=2) ** 2
             + np.linalg.norm((a[:, None, 3:] - U[None, :, 3:]) / 2.0, axis=2) ** 2)
        cnt += np.bincount(d.argmin(1), minlength=len(U))
    frac = cnt / cnt.sum()
    return np.flatnonzero(frac >= thresh), frac


class Heads:
    def __init__(self, dz, E=5, dev="cuda", hidden=256):
        import torch.nn as nn
        self.E, self.dev = E, dev
        self.nets = [nn.Sequential(nn.Linear(dz + 6, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 9)).to(dev) for _ in range(E)]
        self.tau = None

    def _x(self, wm, z, u):
        return __import__("torch").cat([z, wm.na(u)], -1)

    def fit(self, wm, Z, A, O1, steps=3000, batch=4096, seed=0):
        import torch
        g = torch.Generator(device=self.dev).manual_seed(seed)
        N = len(Z)
        for e, net in enumerate(self.nets):
            torch.manual_seed(seed * 100 + e)
            boot = torch.randint(N, (N,), device=self.dev, generator=g)
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=5e-5)
            for it in range(steps):
                j = boot[torch.randint(N, (batch,), device=self.dev, generator=g)]
                loss = ((net(self._x(wm, Z[j], A[j])) - O1[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
        return self

    def sigma(self, wm, z, u):
        import torch
        x = self._x(wm, z, u)
        return torch.stack([n(x) for n in self.nets], 0).std(0).mean(-1)


def transitions(wm, D, dev, stride=1, max_ticks=None, seed=0):
    """(z_t, a_t, o_n,t+1) from filtered latents of split D (t < L-1)."""
    import torch
    Zs = C.filtered_latents(wm, D, dev)
    Z, A, O1 = [], [], []
    for e, z in enumerate(Zs):
        s, L = D["start"][e], D["lens"][e]
        t = np.arange(0, L - 1, stride)
        Z.append(z[t]); A.append(D["act"][s + t]); O1.append(D["obs"][s + t + 1])
    Z, A, O1 = map(lambda x: np.concatenate(x, 0), (Z, A, O1))
    if max_ticks is not None and len(Z) > max_ticks:
        j = np.random.default_rng(seed).choice(len(Z), max_ticks, replace=False)
        Z, A, O1 = Z[j], A[j], O1[j]
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    with torch.no_grad():
        O1n = wm.no(T(O1))
    return T(Z), T(A), O1n


def fit_heads(wm, dev, E=5, steps=3000, seed=0, q=0.99):
    """train the ensemble on nominal TRAIN transitions; tau = q-quantile of sigma on VAL transitions."""
    import torch
    tr, va = C.load_split("train"), C.load_split("val")
    Z, A, O1 = transitions(wm, tr, dev, stride=2)            # ~530k nominal transitions (GPU-memory cap)
    H = Heads(Z.shape[1], E, dev).fit(wm, Z, A, O1, steps=steps, seed=seed)
    Zv, Av, _ = transitions(wm, va, dev, max_ticks=200000, seed=seed)
    with torch.no_grad():
        s = torch.cat([H.sigma(wm, Zv[i:i + 65536], Av[i:i + 65536]) for i in range(0, len(Zv), 65536)])
    H.tau = float(torch.quantile(s[torch.randperm(len(s), device=dev)[:100000]], q))
    U = torch.tensor(C.fallback_U(), dtype=torch.float32, device=dev)
    with torch.no_grad():
        j = torch.randperm(len(Zv), device=dev)[:4000]
        zz = Zv[j][:, None, :].expand(-1, len(U), -1).reshape(-1, Zv.shape[1])
        su = H.sigma(wm, zz, U[None].expand(len(j), -1, -1).reshape(-1, 6)).view(len(j), len(U))
    info = dict(E=E, steps=steps, tau=H.tau, sigma_val_median=float(s.median()),
                frac_U_sigma_gt_tau=float((su > H.tau).float().mean()),
                per_u_frac_gt_tau=(su > H.tau).float().mean(0).cpu().numpy().round(3).tolist())
    return H, info


class PessDyn:
    """reach_value Dyn over z_aug = [z (dz), sigma of the transition that produced z]."""

    def __init__(self, base, heads, bank=None):
        import torch
        self.base, self.heads, self.wm = base, heads, base.wm
        self.U, self.P, self.dev, self.dh = base.U, base.P, base.dev, base.dh
        self.bank = None if bank is None else torch.cat([bank, torch.zeros(len(bank), 1, device=bank.device)], -1)

    def sample_states(self, B):
        import torch
        return self.bank[torch.randint(len(self.bank), (B,), device=self.dev)]

    def _proj_u(self, z, u):
        import torch
        o_n = z[:, self.dh:]
        c = o_n[:, :3] * self.wm.o_sd[:3] + self.wm.o_mu[:3]
        c = c / (c.norm(dim=-1, keepdim=True) + 1e-9)
        tilt = torch.arccos(c[:, 2].clamp(-1, 1))
        dcz = u[:, 3] * c[:, 1] - u[:, 4] * c[:, 0]
        block = (tilt >= self.P.tilt_lim - 0.02) & (dcz < 0)
        u = u.clone(); u[block, 3:] = 0.0
        return u

    def step(self, za, u):
        import torch
        z = za[..., :-1]
        shp = torch.broadcast_shapes(z.shape[:-1], u.shape[:-1])
        zf = z.expand(*shp, z.shape[-1]).reshape(-1, z.shape[-1])
        uf = u.expand(*shp, u.shape[-1]).reshape(-1, u.shape[-1])
        z2 = self.base.step(zf, uf)
        sig = self.heads.sigma(self.wm, zf, self._proj_u(zf, uf))
        return torch.cat([z2, sig[:, None]], -1).view(*shp, -1)

    def margin(self, za):
        import torch
        l = self.base.margin(za[..., :-1])
        return torch.clamp(torch.minimum(l, 1.0 - za[..., -1] / self.heads.tau), -1.0, 1.0)

    def features(self, za):
        return za[..., :-1]


class SubsetDyn:
    """ProjWMDyn with the max restricted to a subset of the fallback set (arm M)."""

    def __init__(self, base, idx):
        import torch
        self.base = base
        self.U = base.U[torch.as_tensor(idx, device=base.dev)]
        self.P, self.dev, self.bank = base.P, base.dev, base.bank

    def sample_states(self, B):
        return self.base.sample_states(B)

    def step(self, z, u):
        return self.base.step(z, u)

    def margin(self, z):
        return self.base.margin(z)

    def features(self, z):
        return z
