"""reach_value.py -- neural HJ reachability value in a (latent) state space.

This is the REACHABILITY learner (LatentSafe / Fisac 2019), NOT a sign(l) classifier:

    V(z) = (1-g) l(z) + g * min( l(z), max_u V(f(z,u)) )                     (1)

learned by fitted value iteration with a Polyak target net, where
  * f is a (latent) dynamics model -- analytic pendulum (Stage 0) or a learned WM
    latent step (Stage 1) -- behind the tiny `Dyn` interface below;
  * max_u is over a DISCRETE fallback control set (all candidates evaluated in
    imagination every backup);
  * ROLLOUT-BASED BOOTSTRAPPING: in addition to the 1-step target, the batch is
    rolled K steps forward under the greedy (argmax_u V_targ) fallback and the
    Fisac backup is taken along that imagined rollout, bootstrapping from V_targ
    only at step K (target = mix of 1-step and K-step backups). The K-step part
    is achievable by an actual policy, which damps max-operator overestimation
    (the unsafe direction);
  * structural V <= l: V = l - softplus(g_theta(features)) (true for (1) for any g).
  * gamma annealed toward gamma_final (Fisac et al.): gamma -> 1 recovers the
    undiscounted viability value.

Dyn interface (all torch, batched over leading dims):
    U                  (nu, du) control candidates
    sample_states(B)   -> z (B, dz)
    step(z, u)         -> z'            (z (...,dz), u (...,du), broadcastable)
    margin(z)          -> l (...)       in [-1, 1], > 0 safe
    features(z)        -> x (..., dx)   network input
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    """V(z) = l(z) - softplus(g(x)),  g an MLP with SiLU (smooth => usable gradients)."""

    def __init__(self, dx, hidden=256, layers=3, v_lo=-1.0):
        super().__init__()
        self.v_lo = float(v_lo)
        mods, d = [], dx + 1
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.g = nn.Sequential(*mods)
        nn.init.constant_(self.g[-1].bias, -3.0)

    def forward(self, x, l):
        # true V in [-1, l]; the clamp keeps off-distribution states (e.g. phi >> phi*)
        # from producing unbounded values/gradients in the controller rows
        return torch.clamp(l - F.softplus(self.g(torch.cat([x, l[..., None]], -1))[..., 0]), min=self.v_lo)


def value_of(net, dyn, z):
    return net(dyn.features(z), dyn.margin(z))


@torch.no_grad()
def greedy_next(net, dyn, z, chunk=None):
    """For each z (B,dz): evaluate V at f(z,u) for every candidate u; returns
    (z_next_best (B,dz), v_best (B,), u_idx (B,))."""
    U = dyn.U
    zz = z[:, None, :].expand(-1, U.shape[0], -1)
    nxt = dyn.step(zz, U[None].expand(z.shape[0], -1, -1))
    v = value_of(net, dyn, nxt)                       # (B, nu)
    vb, ib = v.max(1)
    return nxt[torch.arange(z.shape[0], device=z.device), ib], vb, ib


def train(dyn, out: Path, iters=30000, batch=2048, lr=3e-4, gamma0=0.9, gamma_final=0.99,
          anneal_frac=0.5, K=4, lam=0.5, tau=0.005, hidden=256, layers=3, log_every=500,
          seed=0, device="cuda", eval_fn=None, replay_frac=0.3, replay_size=200000):
    torch.manual_seed(seed); np.random.seed(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    z0 = dyn.sample_states(4)
    dx = dyn.features(z0).shape[-1]
    v_lo = getattr(getattr(dyn, 'P', None), 'm_lo', -1.0)
    net = ValueNet(dx, hidden, layers, v_lo).to(device)
    targ = ValueNet(dx, hidden, layers, v_lo).to(device)
    targ.load_state_dict(net.state_dict())
    for p in targ.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    replay = None
    curve = []
    t0 = time.time()
    for it in range(1, iters + 1):
        frac = min(1.0, it / max(1, anneal_frac * iters))
        # anneal in log(1-gamma) (Fisac et al. style), gamma0 -> gamma_final
        gamma = 1.0 - math.exp((1 - frac) * math.log(1 - gamma0) + frac * math.log(1 - gamma_final))
        nb = batch if replay is None else int(batch * (1 - replay_frac))
        z = dyn.sample_states(nb)
        if replay is not None and batch - nb > 0:
            z = torch.cat([z, replay[torch.randint(len(replay), (batch - nb,), device=replay.device)]], 0)
        with torch.no_grad():
            l0 = dyn.margin(z)
            z1, v1, _ = greedy_next(targ, dyn, z)
            y1 = (1 - gamma) * l0 + gamma * torch.minimum(l0, v1)            # 1-step target (1)
            if K > 1:
                zs, ls = [z1], [dyn.margin(z1)]
                zk = z1
                for _ in range(K - 1):
                    zk, _, _ = greedy_next(targ, dyn, zk)
                    zs.append(zk); ls.append(dyn.margin(zk))
                y = value_of(targ, dyn, zs[-1])
                for k in range(len(zs) - 1, -1, -1):
                    y = (1 - gamma) * ls[k] + gamma * torch.minimum(ls[k], y) if k < len(zs) - 1 else y
                yK = (1 - gamma) * l0 + gamma * torch.minimum(l0, y)           # K-step greedy-rollout target
                tgt = lam * yK + (1 - lam) * y1
                newz = torch.cat(zs, 0)
            else:
                tgt = y1
                newz = z1
            newz = newz[torch.randperm(len(newz), device=newz.device)[:batch // 2]]
            replay = newz if replay is None else torch.cat([replay, newz], 0)[-replay_size:]
        v = value_of(net, dyn, z)
        loss = F.mse_loss(v, tgt)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step(); sched.step()
        with torch.no_grad():
            for p, q in zip(net.parameters(), targ.parameters()):
                q.mul_(1 - tau).add_(tau * p)
        if it % log_every == 0 or it == 1:
            rec = dict(it=it, loss=float(loss), gamma=gamma, sec=time.time() - t0,
                       frac_l_pos_v_neg=float(((l0 > 0) & (v <= 0)).float().mean()))
            if eval_fn is not None:
                rec.update(eval_fn(net))
            curve.append(rec)
            print("[reach] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
                  flush=True)
    return net, curve


def save(net, path, meta: dict):
    torch.save(dict(state=net.state_dict(), meta=meta), path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ck["meta"]
    net = ValueNet(m["dx"], m.get("hidden", 256), m.get("layers", 3),
                   v_lo=m.get("P", {}).get("m_lo", -1.0)).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, m


# =============================================================================== analytic dyn
class PendDyn:
    """Dyn for the analytic liquid proxy. z = [n(3), nd(3), c(3)]."""

    def __init__(self, P, device="cuda", planar_frac=0.3, thd_max=30.0):
        from world_model.v2.latent_hj import slosh_pend as sp
        self.P, self.dev = P, device
        self.fn = sp.torch_fns()
        self.U = torch.tensor(sp.control_set(P), dtype=torch.float32, device=device)
        self.planar_frac, self.thd_max = planar_frac, thd_max

    def _unit_from(self, tilt, az, up):
        s = torch.sin(tilt)
        v = torch.stack([s * torch.cos(az), s * torch.sin(az), torch.cos(tilt) * (1 if up else -1)], -1)
        return v

    def sample_states(self, B):
        d = self.dev
        az_n = torch.rand(B, device=d) * 2 * math.pi
        tilt_n = torch.rand(B, device=d) * 1.9
        n = self._unit_from(tilt_n, az_n, up=False)
        az_c = az_n + (torch.rand(B, device=d) - 0.5) * 2 * math.pi * torch.rand(B, device=d)
        tilt_c = torch.rand(B, device=d) * self.P.tilt_lim
        c = self._unit_from(tilt_c, az_c, up=True)
        # tangent velocity
        r = torch.randn(B, 3, device=d)
        r = r - n * (r * n).sum(-1, keepdim=True)
        r = r / (r.norm(dim=-1, keepdim=True) + 1e-9)
        mag = self.thd_max * torch.rand(B, device=d) ** 1.5
        nd = r * mag[:, None]
        # planar subset (x-z plane): same marginal ranges
        m = torch.rand(B, device=d) < self.planar_frac
        if m.any():
            k = int(m.sum())
            th = (torch.rand(k, device=d) * 2 - 1) * 1.9
            al = (torch.rand(k, device=d) * 2 - 1) * self.P.tilt_lim
            thd = (torch.rand(k, device=d) * 2 - 1) * self.thd_max
            z0 = torch.zeros_like(th)
            n[m] = torch.stack([torch.sin(th), z0, -torch.cos(th)], -1)
            nd[m] = thd[:, None] * torch.stack([torch.cos(th), z0, torch.sin(th)], -1)
            c[m] = torch.stack([torch.sin(al), z0, torch.cos(al)], -1)
        return torch.cat([n, nd, c], -1)

    def step(self, z, u):
        n, nd, c = z[..., 0:3], z[..., 3:6], z[..., 6:9]
        a, w = u[..., 0:3], u[..., 3:6]
        shp = torch.broadcast_shapes(n.shape, a.shape)
        n2, nd2, c2 = self.fn["step"](n.expand(shp), nd.expand(shp), c.expand(shp), a.expand(shp), w.expand(shp), self.P)
        return torch.cat([n2, nd2, c2], -1)

    def margin(self, z):
        return self.fn["margin"](z[..., 0:3], z[..., 6:9], self.P)

    def features(self, z):
        return self.fn["feats"](z[..., 0:3], z[..., 3:6], z[..., 6:9], nd_scale=1.0 / self.P.w1)
