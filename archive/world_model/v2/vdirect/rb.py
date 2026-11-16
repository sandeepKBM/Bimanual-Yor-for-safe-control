"""vdirect/rb.py -- root-restricted random branches (PREREG_ADDENDUM_RB). SIMULATION / SAVED DATA ONLY.

Every arm keeps R's branch-ACTION law (train_kinwm.make_branches) and R's training recipe; only the ROOTS differ.
  Rc: roots drawn uniformly from the eligible candidate set (branches.candidates: t0 in [60, L-1), logged l > 0,
      stride 2) -- the root-law control for Rb.
  Rb: roots drawn uniformly from the LOWER HALF of the nominal value's accepted set:
      0 < V_nom(z) <= median of V_nom over the accepted candidates.
      V_nom = the round-0 probe value on the NOM m0 WM (the same value as F's round 0). Computing it costs no
      true-system rollout, so the budget counts branches only, as for R.
"""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.wm_audit import common as C
from world_model.v2.wm_audit.train_kinwm import feats_t


def make_branches_at(D, e, t0, Lb, W, dev, rng, p_fallback=0.5):
    """train_kinwm.make_branches with the roots (e, t0) given and the action rng passed in. The code after the root
    draw is copied verbatim; rb_selftest.py checks bitwise equality with make_branches on its own roots."""
    import torch
    n_br = len(e)
    s0 = D["start"][e] + t0
    pre = (W - Lb)
    ix = s0[:, None] + np.arange(-pre, 0)[None]
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    obs_pre, act_pre, l_pre = T(D["obs"][ix]), T(D["act"][ix]), T(D["l"][ix])
    n_pre, nd_pre = T(D["n"][ix]), T(D["nd"][ix])
    c_pre = obs_pre[..., :3] / obs_pre[..., :3].norm(dim=-1, keepdim=True)
    f_pre = feats_t(n_pre, nd_pre, c_pre)
    U = C.fallback_U()
    A = np.zeros((n_br, Lb, 6), np.float32)
    for b in range(n_br):
        k = 0
        gain = 1.0 if rng.random() < 0.4 else rng.uniform(0.15, 1.0)
        while k < Lb:
            L = int(rng.integers(3, 16))
            r_ = rng.random()
            if r_ < 0.2:
                u = np.zeros(6)
            elif r_ < 0.2 + 0.8 * p_fallback:
                u = U[rng.integers(len(U))].copy()
            else:
                r = 5.0 * math.sqrt(rng.random()); th = rng.uniform(0, 2 * math.pi)
                u = np.array([r * math.cos(th), r * math.sin(th), rng.normal(0, 0.5),
                              rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), 0.0])
            A[b, k:k + L] = gain * u
            k += L
    A = T(A)
    n = T(D["n"][s0]); nd = T(D["nd"][s0]); o = T(D["obs"][s0])
    c = o[:, :3] / o[:, :3].norm(dim=-1, keepdim=True); v = o[:, 3:6]
    obs_b, act_b, l_b, f_b = [], [], [], []
    obs_b.append(o); l_b.append(T(D["l"][s0])); f_b.append(feats_t(n, nd, c))
    with torch.no_grad():
        for k in range(Lb):
            a, w = A[:, k, :3] - 2.0 * v, A[:, k, 3:]
            ls, weff, n, nd, c = C.pend_rollout_torch(n, nd, c, a[:, None], w[:, None], project=True, return_state=True)
            v = v + C.DT_WM * a
            w_obs = weff[:, 0]
            act_b.append(torch.cat([a, w_obs], -1))
            if k < Lb - 1:
                obs_b.append(torch.cat([c, v, w_obs], -1)); l_b.append(ls[:, 0]); f_b.append(feats_t(n, nd, c))
    obs = torch.cat([obs_pre, torch.stack(obs_b, 1)], 1)
    act = torch.cat([act_pre, torch.stack(act_b, 1)], 1)
    l = torch.cat([l_pre, torch.stack(l_b, 1)], 1)
    feat = torch.cat([f_pre, torch.stack(f_b, 1)], 1)
    obs[..., 3:6] = obs[..., 3:6].clamp(-3.0, 3.0)
    return dict(obs=obs, act=act, l=l, feat=feat)


def nominal_V(D, cand, wm, net, dev):
    from world_model.v2.vdirect import branches as BRS
    return BRS.ValueScorer(wm, net, dev).state_values(BRS.latents_at(wm, D, cand, dev))[0].cpu().numpy()


def pick_roots(D, n_br, root_seed, n_cand, V=None, band_q=0.5):
    """-> (roots (n_br, 2) [e, t0], meta). V=None: Rc (all candidates); V given: Rb (lower band of the accepted)."""
    from world_model.v2.vdirect import branches as BRS
    cand = BRS.candidates(D, n_cand, root_seed)
    meta = dict(n_cand=int(len(cand)), band_q=None if V is None else band_q)
    if V is None:
        pool = np.arange(len(cand))
    else:
        assert len(V) == len(cand)
        acc = V > 0
        cut = float(np.quantile(V[acc], band_q))
        pool = np.flatnonzero(acc & (V <= cut))
        meta.update(frac_accepted=float(acc.mean()), V_cut=cut)
    rr = np.random.default_rng(root_seed + 7)
    j = rr.choice(pool, n_br, replace=n_br > len(pool))
    meta.update(pool_size=int(len(pool)), with_replacement=bool(n_br > len(pool)),
                root_l_mean=float(D["l"][D["start"][cand[j, 0]] + cand[j, 1]].mean()))
    if V is not None:
        meta.update(root_V_mean=float(V[j].mean()), root_V_max=float(V[j].max()))
    return cand[j], cand, meta


def band_branches(D, roots, dev, seed):
    """R's action law from the given roots; action rng = default_rng(seed) (R uses seed = run seed + 1)."""
    from world_model.v2.vdirect import branches as BRS
    BR = make_branches_at(D, roots[:, 0], roots[:, 1], BRS.LB, BRS.W_WIN, dev, np.random.default_rng(seed))
    lb = BR["l"][:, BRS.PRE:]
    return BR, dict(n=int(len(roots)), branch_spill_frac=float((lb < 0).any(1).float().mean()),
                    tick_spill_frac=float((lb < 0).float().mean()))
