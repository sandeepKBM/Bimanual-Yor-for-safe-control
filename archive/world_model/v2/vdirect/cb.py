"""vdirect/cb.py -- Cao & Bansal, "When World Models Lie: Adaptive Safety Analysis Under Wrong Imaginations"
(arXiv 2609.34300), adapted to the kin_wm latent. Deployment-time only: NO training, NO new data.
SIMULATION / SAVED DATA ONLY. Adaptations are listed in $VD_ROOT/cb/IMPL_CB.md.

Paper -> here
  latent z = [h, z_stoch] (DreamerV3)    -> z = [h, o_n] (kin_wm; h is shared by prior and posterior in both)
  e_t = ||z_t - zhat_t||_2 (eq. 9)       -> ||o_n,t - dec(h_t)||_2 (the h blocks cancel exactly)
  ACI (eq. 6): a_{t+1} = a_t + lam (alpha - err_t), err_t = 1[e_t > q_t]; q_{t+1} = ceil((t+1)(1-a_{t+1}))-th order
               statistic of e_1..e_t (inf if > t, -inf if <= 0); q_1 = inf        -> identical, alpha 0.2, lam 0.05
  U_{t+1} = {z : ||z - zhat_{t+1}|| <= min(q_{t+1}, q_max)} (eq. 10)              -> identical (full latent, L2)
  q_max: trajectory-level CP, alpha_traj 0.1, alpha_cal 0.05, 100 calibration trajectories -> 100 VAL episodes
  Vbar(zhat_{t+1}) = min_{z in U_{t+1}} V(z) by PGD (eq. 8; PGD settings not given) -> 10 L2 steps of 2.5 r/10,
               start at the centre, min over the iterates (Madry et al. 2018 L2 defaults)
  filter (eq. 4): pi_nom if Vbar(zhat_{t+1}) > delta else pi_safe = argmax_a V (nominal actor)
               -> the trigger/flag is Vbar <= 0 (threshold 0, PREREG addendum); the fallback action is the NOMINAL
                  greedy argmax_u V(f(z,u)) (the paper's pi_safe). Secondary, NOT in the paper: 'robust' fallback
                  argmax_u Vbar(f(z,u)) over the same set.
"""
from __future__ import annotations

import bisect
import math

import numpy as np

from world_model.v2.wm_audit import common as C

ALPHA, LAM = 0.2, 0.05                 # paper (Franka sim + hardware)
ALPHA_TRAJ, ALPHA_CAL, N_CAL = 0.1, 0.05, 100     # paper (Franka sim)
PGD_STEPS, PGD_FRAC = 10, 2.5


# ------------------------------------------------------------------ errors along logged episodes
def episode_errors(wm, D, eps, dev, chunk=64):
    """-> list of (L,) arrays e with e[0] = nan (no prediction for the first tick), e[t] = ||o_n,t - dec(h_t)||."""
    import torch
    out = []
    with torch.no_grad():
        for i0 in range(0, len(eps), chunk):
            ii = np.asarray(eps[i0:i0 + chunk])
            obs, act, lens = C.pad_episodes(D, ii)
            H, on = C.filter_fast(wm, torch.tensor(obs, device=dev), torch.tensor(act, device=dev))
            e = (on - wm.dec(H)).norm(dim=-1).cpu().numpy()
            for j, ep in enumerate(ii):
                L = D["lens"][ep]
                x = e[j, :L].copy(); x[0] = np.nan
                out.append(x)
    return out


def calibrate_qmax(wm, dev, seed=0):
    """trajectory-level conformal trust radius on N_CAL random VAL episodes (never TEST)."""
    D = C.load_split("val")
    rng = np.random.default_rng(seed)
    eps = np.sort(rng.choice(len(D["lens"]), min(N_CAL, len(D["lens"])), replace=False))
    E = episode_errors(wm, D, eps, dev)
    Q = np.array([np.quantile(e[1:], 1 - ALPHA_TRAJ, method="higher") for e in E])
    n = len(Q)
    k = math.ceil((n + 1) * (1 - ALPHA_CAL))
    qmax = float(np.sort(Q)[min(k, n) - 1])
    return qmax, dict(n_cal=int(n), k=int(k), qmax=qmax, Qtraj_median=float(np.median(Q)),
                      e_median=float(np.median(np.concatenate([e[1:] for e in E]))),
                      e_p90=float(np.quantile(np.concatenate([e[1:] for e in E]), 0.9)))


class ACI:
    """Adaptive Conformal Inference on the latent prediction error (paper eqs. 6, 9, 10)."""

    def __init__(self, qmax, alpha=ALPHA, lam=LAM):
        self.alpha, self.lam, self.qmax = alpha, lam, qmax
        self.at, self.q, self.buf = alpha, math.inf, []
        self.n_err, self.n_cov_bar, self.n = 0, 0, 0
        self.r_sum = 0.0

    def radius(self):
        return max(0.0, min(self.q, self.qmax))

    def update(self, e):
        """observe e_t (scored against the q_t / radius in force), then form q_{t+1}."""
        r = self.radius()
        err = e > self.q
        self.n += 1; self.n_err += int(err); self.n_cov_bar += int(e <= r); self.r_sum += r
        self.at = self.at + self.lam * (self.alpha - float(err))
        bisect.insort(self.buf, e)
        t = len(self.buf)
        k = math.ceil((t + 1) * (1 - self.at))
        self.q = math.inf if k > t else (-math.inf if k <= 0 else self.buf[k - 1])
        return err


# ------------------------------------------------------------------ pessimistic value (PGD over the L2 ball)
def V_of(net, wm, z):
    return net(z, wm.margin(z[..., :wm.d_h], z[..., wm.d_h:]))


def pgd_min(net, wm, zc, r, block=None, steps=PGD_STEPS, frac=PGD_FRAC, chunk=32768):
    """min of V over {z : ||z - zc||_2 <= r} (per-row radius r >= 0). block: None = full latent (paper);
    'o' = only the o_n block (secondary sensitivity). -> (B,) torch."""
    import torch
    out = []
    for i in range(0, len(zc), chunk):
        c = zc[i:i + chunk].detach(); rr = torch.as_tensor(r[i:i + chunk], dtype=c.dtype, device=c.device)
        with torch.no_grad():
            best = V_of(net, wm, c)
        if steps > 0 and bool((rr > 0).any()):
            mask = torch.ones(c.shape[1], device=c.device)
            if block == "o":
                mask[:wm.d_h] = 0.0
            eta = (frac * rr / steps)[:, None]
            z = c.clone()
            for _ in range(steps):
                z = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    g = torch.autograd.grad(V_of(net, wm, z).sum(), z)[0] * mask
                gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                z = z.detach() - eta * g / gn
                d = z - c
                dn = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
                z = c + d * torch.clamp(rr[:, None] / dn, max=1.0)
                with torch.no_grad():
                    best = torch.minimum(best, V_of(net, wm, z))
        out.append(best)
    return torch.cat(out)
