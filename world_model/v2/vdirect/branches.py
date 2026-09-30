"""vdirect/branches.py -- DIRECTED counterfactual branches (value- or uncertainty-directed).
SIMULATION / SAVED DATA ONLY. Ground-truth labels come from the SAME pendulum integrator that
train_kinwm.make_branches uses (common.pend_rollout_torch, 200 Hz, tilt projection).

A branch = 160-tick window: the 60-tick REAL logged prefix + a 100-tick (2 s) counterfactual continuation
from a logged TRAIN state (e, t0), exactly the tensor format of make_branches (obs, act, l, feat).

Branch points (states) -- scored on a fixed-size uniform random candidate set of eligible TRAIN states
(t0 in [60, L-1), logged margin l > 0, stride 2):
  V (value-directed): half from the pool of states with the smallest |V(z)| (value boundary), half from the pool
     with the largest l^(z) - V(z) (WM-value vs margin-head disagreement; V <= l^ by construction).
  U (uncertainty-directed): the pool with the largest max_u D(z, u), D = std over an M-member WM ensemble of the
     predicted min margin over H ticks under the held fallback action u (disagreement on counterfactual actions).
  Pools are the top `pool_mult` x (#to draw) candidates of each score; draws are uniform without replacement.

Branch actions -- closed loop on the TRUE pendulum, the WM(s) filtering the true observation stream every tick
(closed_loop.py semantics: u in the 85-element fallback set applied directly, no velocity feedback, gain 1):
  piecewise segments of 3-15 ticks (same law as make_branches); each segment is
    'greedy'  (p_greedy): every tick u = argmax_u score(z_t, u)
    'runner'  (1-p_greedy): u = the rank-r action (r uniform in {2,3,4}) at the segment start, held
  with score = Q(z,u) = V(f_WM(z,u)) for V, = D(z,u) for U.
"""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.wm_audit import common as C
from world_model.v2.wm_audit.train_kinwm import feats_t
from world_model.v2.wm_audit.value_gate import ProjWMDyn

W_WIN, LB, PRE = 160, 100, 60


def candidates(D, n_cand, seed, stride=2):
    """uniform random sample of eligible TRAIN states (e, t0): t0 in [60, L-1), stride 2, logged l > 0."""
    rng = np.random.default_rng(seed)
    E, T = [], []
    for e in range(len(D["lens"])):
        L = D["lens"][e]
        if L <= W_WIN - LB + 5:
            continue
        t = np.arange(PRE, L - 1, stride)
        t = t[D["l"][D["start"][e] + t] > 0]
        E.append(np.full(len(t), e)); T.append(t)
    E, T = np.concatenate(E), np.concatenate(T)
    j = rng.choice(len(E), min(n_cand, len(E)), replace=False)
    return np.stack([E[j], T[j]], 1)


def latents_at(wm, D, pick, dev):
    """filtered latents z = [h_t, o_n,t] at (e, t) pairs (closed_loop semantics), (N, dh+9) torch."""
    import torch
    eps = np.unique(pick[:, 0])
    Z = dict(zip(eps.tolist(), C.filtered_latents(wm, D, dev, eps)))
    return torch.tensor(np.stack([Z[e][t] for e, t in pick]), dtype=torch.float32, device=dev)


# ============================================================================ scores
class ValueScorer:
    """Q(z,u) = V(f_WM(z,u)) over the fallback set (exactly the closed-loop wm_greedy objective)."""

    def __init__(self, wm, net, dev):
        self.wm, self.net, self.dev = wm, net, dev
        self.dyn = ProjWMDyn(wm, None, dev, project=True)
        self.U = self.dyn.U
        self.models = [wm]

    def state_values(self, z, chunk=65536):
        import torch
        out_v, out_l = [], []
        with torch.no_grad():
            for i in range(0, len(z), chunk):
                zz = z[i:i + chunk]
                lh = self.wm.margin(zz[:, :self.wm.d_h], zz[:, self.wm.d_h:])
                out_v.append(self.net(zz, lh)); out_l.append(lh)
        return torch.cat(out_v), torch.cat(out_l)

    def action_scores(self, zs, chunk=2048):
        """zs: list (one per model) of (B, dz) latents -> (B, nu)."""
        import torch
        z = zs[0]
        B, nu = len(z), len(self.U)
        out = []
        with torch.no_grad():
            for i in range(0, B, chunk):
                zz = z[i:i + chunk]
                nxt = self.dyn.step(zz[:, None, :].expand(-1, nu, -1), self.U[None].expand(len(zz), -1, -1))
                out.append(self.net(nxt, self.dyn.margin(nxt)))
        return torch.cat(out)


class EnsembleScorer:
    """D(z,u) = std over M WMs of min_{k<=H} l^_m(f_m^k(z_m, u)) (held u, tilt-projected like the value's Dyn)."""

    def __init__(self, wms, dev, H=25):
        self.models, self.dev, self.H = wms, dev, H
        self.dyns = [ProjWMDyn(w, None, dev, project=True) for w in wms]
        self.U = self.dyns[0].U

    def action_scores(self, zs, chunk=1024):
        import torch
        B, nu = len(zs[0]), len(self.U)
        out = []
        with torch.no_grad():
            for i in range(0, B, chunk):
                mins = []
                for dyn, z in zip(self.dyns, zs):
                    zz = z[i:i + chunk]
                    b = len(zz)
                    x = zz[:, None, :].expand(-1, nu, -1).reshape(b * nu, -1)
                    u = self.U[None].expand(b, -1, -1).reshape(b * nu, -1)
                    m = torch.full((b * nu,), 1.0, device=self.dev)
                    for _ in range(self.H):
                        x = dyn.step(x, u)
                        m = torch.minimum(m, dyn.margin(x))
                    mins.append(m.view(b, nu))
                out.append(torch.stack(mins, 0).std(0))
        return torch.cat(out)


# ============================================================================ state selection
def select_states(cand, scores_list, n_list, pool_mult, rng):
    """for each (score (higher = more relied-on), n): draw n uniformly w/o replacement from the top pool_mult*n
    candidates of that score (excluding ones already drawn). -> (picks (N,2), which-criterion (N,))"""
    taken = np.zeros(len(cand), bool)
    picks, crit = [], []
    for ci, (s, n) in enumerate(zip(scores_list, n_list)):
        if n <= 0:
            continue
        s = np.where(taken, -np.inf, s)
        pool = np.argsort(-s, kind="stable")[:max(n, pool_mult * n)]
        pool = pool[np.isfinite(s[pool])]
        j = rng.choice(pool, n, replace=False)
        taken[j] = True
        picks.append(cand[j]); crit.append(np.full(n, ci))
    return np.concatenate(picks, 0), np.concatenate(crit)


# ============================================================================ closed-loop branch rollouts
def rollout_branches(D, pick, scorer, dev, seed, p_greedy=0.6, ranks=(1, 2, 3), chunk=4096):
    """closed-loop counterfactual branches from logged TRAIN states `pick` (N,2). -> dict(obs, act, l, feat) of
    (N, 160, .) float tensors on `dev` (make_branches format) + per-branch meta."""
    import torch
    rng = np.random.default_rng(seed)
    N = len(pick)
    s0 = D["start"][pick[:, 0]] + pick[:, 1]
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    # segment plan (same length law as make_branches): per tick, mode (1 greedy / 0 runner) and seg-start flag
    mode = np.zeros((N, LB), bool); start = np.zeros((N, LB), bool); rank = np.zeros((N, LB), np.int64)
    for b in range(N):
        k = 0
        while k < LB:
            L = int(rng.integers(3, 16))
            g = rng.random() < p_greedy
            mode[b, k:k + L] = g; start[b, k] = True
            rank[b, k:k + L] = 0 if g else ranks[int(rng.integers(len(ranks)))]
            k += L
    ix = s0[:, None] + np.arange(-PRE, 0)[None]
    obs_pre, act_pre, l_pre = T(D["obs"][ix]), T(D["act"][ix]), T(D["l"][ix])
    n_pre, nd_pre = T(D["n"][ix]), T(D["nd"][ix])
    c_pre = obs_pre[..., :3] / obs_pre[..., :3].norm(dim=-1, keepdim=True)
    f_pre = feats_t(n_pre, nd_pre, c_pre)
    # initial filtered latents per model
    Z0 = [latents_at(m, D, pick, dev) for m in scorer.models]
    U = scorer.U
    n = T(D["n"][s0]); nd = T(D["nd"][s0]); o = T(D["obs"][s0])
    c = o[:, :3] / o[:, :3].norm(dim=-1, keepdim=True); v = o[:, 3:6]
    dh = [m.d_h for m in scorer.models]
    hs = [z[:, :d].clone() for z, d in zip(Z0, dh)]
    o_n = Z0[0][:, dh[0]:].clone()                 # identical fixed normalization for every model
    obs_b, act_b, l_b, f_b = [o], [], [T(D["l"][s0])], [feats_t(n, nd, c)]
    u_hold = torch.zeros(N, 6, device=dev)
    mode_t, start_t, rank_t = (torch.tensor(x, device=dev) for x in (mode, start, rank))
    n_greedy_ticks = 0
    with torch.no_grad():
        for k in range(LB):
            zs = [torch.cat([h, o_n], -1) for h in hs]
            sc = scorer.action_scores(zs, chunk=chunk)                       # (N, nu)
            order = sc.argsort(1, descending=True)
            u_greedy = U[order[:, 0]]
            u_rank = U[order.gather(1, rank_t[:, k:k + 1])[:, 0]]
            u_hold = torch.where((start_t[:, k] & ~mode_t[:, k])[:, None], u_rank, u_hold)
            u = torch.where(mode_t[:, k][:, None], u_greedy, u_hold)
            n_greedy_ticks += int(mode_t[:, k].sum())
            ls, weff, n, nd, c = C.pend_rollout_torch(n, nd, c, u[:, None, :3], u[:, None, 3:], project=True,
                                                      return_state=True)
            w_obs = weff[:, 0]
            act = torch.cat([u[:, :3], w_obs], -1)
            act_b.append(act)
            v = v + C.DT_WM * u[:, :3]
            o_next = torch.cat([c, v.clamp(-3.0, 3.0), w_obs], -1)
            for i, m in enumerate(scorer.models):
                hs[i], _ = m.step(hs[i], o_n, act)
            o_n = scorer.models[0].no(o_next)
            if k < LB - 1:
                obs_b.append(o_next); l_b.append(ls[:, 0]); f_b.append(feats_t(n, nd, c))
    obs = torch.cat([obs_pre, torch.stack(obs_b, 1)], 1)
    act = torch.cat([act_pre, torch.stack(act_b, 1)], 1)
    l = torch.cat([l_pre, torch.stack(l_b, 1)], 1)
    feat = torch.cat([f_pre, torch.stack(f_b, 1)], 1)
    obs[..., 3:6] = obs[..., 3:6].clamp(-3.0, 3.0)
    lb = l[:, PRE:]
    meta = dict(n=int(N), branch_spill_frac=float((lb < 0).any(1).float().mean()),
                tick_spill_frac=float((lb < 0).float().mean()), greedy_tick_frac=n_greedy_ticks / (N * LB),
                l0_mean=float(l[:, PRE].mean()))
    return dict(obs=obs, act=act, l=l, feat=feat), meta


def random_branches(D, n_br, dev, seed):
    """the existing recipe (train_kinwm.make_branches, seed = run seed + 1 as in train_kinwm)."""
    from world_model.v2.wm_audit.train_kinwm import make_branches
    BR = make_branches(D, n_br, LB, W_WIN, dev, seed=seed)
    lb = BR["l"][:, PRE:]
    return BR, dict(n=int(n_br), branch_spill_frac=float((lb < 0).any(1).float().mean()),
                    tick_spill_frac=float((lb < 0).float().mean()))


def value_directed(D, n_br, wm, net, dev, seed, n_cand=60000, pool_mult=4, p_greedy=0.6):
    scorer = ValueScorer(wm, net, dev)
    rng = np.random.default_rng(seed)
    cand = candidates(D, n_cand, seed)
    z = latents_at(wm, D, cand, dev)
    V, lh = scorer.state_values(z)
    V, lh = V.cpu().numpy(), lh.cpu().numpy()
    nb = n_br // 2
    picks, crit = select_states(cand, [-np.abs(V), lh - V], [nb, n_br - nb], pool_mult, rng)
    BR, meta = rollout_branches(D, picks, scorer, dev, seed + 7, p_greedy=p_greedy)
    sel = {tuple(p): i for i, p in enumerate(cand.tolist())}
    ii = np.array([sel[tuple(p)] for p in picks.tolist()])
    meta.update(kind="V", n_cand=int(len(cand)),
                cand_V_mean=float(V.mean()), cand_absV_lt_0p1=float((np.abs(V) < 0.1).mean()),
                cand_disagree_mean=float((lh - V).mean()), cand_disagree_gt_0p1=float(((lh - V) > 0.1).mean()),
                pick_V_mean=float(V[ii].mean()), pick_disagree_mean=float((lh - V)[ii].mean()),
                pick_l_mean=float(D["l"][D["start"][picks[:, 0]] + picks[:, 1]].mean()))
    return BR, meta, picks


def uncertainty_directed(D, n_br, wms, dev, seed, n_cand=60000, pool_mult=4, p_greedy=0.6, H=25):
    scorer = EnsembleScorer(wms, dev, H=H)
    rng = np.random.default_rng(seed)
    cand = candidates(D, n_cand, seed)
    zs = [latents_at(m, D, cand, dev) for m in wms]
    Dz = scorer.action_scores(zs).cpu().numpy()                      # (N, nu)
    s = Dz.max(1)
    picks, crit = select_states(cand, [s], [n_br], pool_mult, rng)
    BR, meta = rollout_branches(D, picks, scorer, dev, seed + 7, p_greedy=p_greedy)
    sel = {tuple(p): i for i, p in enumerate(cand.tolist())}
    ii = np.array([sel[tuple(p)] for p in picks.tolist()])
    meta.update(kind="U", n_cand=int(len(cand)), H=H, cand_Dmax_mean=float(s.mean()),
                pick_Dmax_mean=float(s[ii].mean()),
                pick_l_mean=float(D["l"][D["start"][picks[:, 0]] + picks[:, 1]].mean()))
    return BR, meta, picks
