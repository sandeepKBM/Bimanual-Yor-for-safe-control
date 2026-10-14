"""wm_audit/train_kinwm.py -- fast trainer for the kin_wm latent WM (same architecture and
state-dict as world_model/v2/latent_hj/kin_wm.py, so kin_wm.load_wm / WMDyn / reach_value work
unchanged) with the audit's switches. SIMULATION / SAVED DATA ONLY.

  --frac F        train on a random F of the TRAIN episodes (data-scaling pilot)
  --aux-w W       PRIVILEGED supervision (Kim et al. 2025 'estimation gap' fix): an aux head
                  predicts the pendulum slosh state (10 yaw-invariant feats of n, n', c) from the
                  latent, on filtered AND imagined (overshooting) states. Never a WM input.
  --branch-frac B fraction of each batch drawn from COUNTERFACTUAL BRANCHES: from a train state,
                  the cup follows random piecewise-constant evasive/fallback actions for 2 s and
                  the liquid proxy is integrated exactly (the same pendulum as the labels). This is
                  the 'collect with randomized / evasive actions' fix, synthesized for the proxy.
  --gp G          gradient penalty on the margin head (LatentCBF, Nakamura et al. 2025)

Speed: filtering uses one cuDNN GRU call with the GRUCell's weights (identical math).
  CUDA_VISIBLE_DEVICES=3 python -m world_model.v2.wm_audit.train_kinwm --out $AUD/runs/x --steps 15000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C


def feats_t(n, nd, c):
    fn = C.sp.torch_fns()
    return fn["feats"](n, nd, c, nd_scale=1.0 / C.P0.w1).clamp(-4.0, 4.0)


# ----------------------------------------------------------------------------- counterfactual branches
def make_branches(D, n_br, Lb, W, dev, seed=0, p_fallback=0.5):
    """-> dict of (n_br, W, .) tensors: obs, act, l, feat, is_branch mask; windows END at the
    branch end, so the first W-Lb ticks are the real logged prefix (burn-in)."""
    import torch
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(D["lens"] > W - Lb + 5)
    e = rng.choice(ok, n_br)
    # branch point t0 >= W-Lb so the window [t0-(W-Lb), t0+Lb) has a full real prefix
    t0 = np.array([rng.integers(W - Lb, D["lens"][i] - 1) for i in e])
    s0 = D["start"][e] + t0
    pre = (W - Lb)
    ix = s0[:, None] + np.arange(-pre, 0)[None]
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    obs_pre, act_pre, l_pre = T(D["obs"][ix]), T(D["act"][ix]), T(D["l"][ix])
    n_pre, nd_pre = T(D["n"][ix]), T(D["nd"][ix])
    c_pre = obs_pre[..., :3] / obs_pre[..., :3].norm(dim=-1, keepdim=True)
    f_pre = feats_t(n_pre, nd_pre, c_pre)
    # branch actions: piecewise constant, 3..15-tick segments (60-300 ms): 20% hold-still,
    # 40% a fallback-set element, 40% uniform in the box |a_h|<=5, |w_xy|<=2
    U = C.fallback_U()
    A = np.zeros((n_br, Lb, 6), np.float32)
    for b in range(n_br):
        k = 0
        gain = 1.0 if rng.random() < 0.4 else rng.uniform(0.15, 1.0)     # per-branch intensity
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
    # state at the branch point (tick t0): logged pendulum state + cup kinematics
    n = T(D["n"][s0]); nd = T(D["nd"][s0]); o = T(D["obs"][s0])
    c = o[:, :3] / o[:, :3].norm(dim=-1, keepdim=True); v = o[:, 3:6]
    obs_b, act_b, l_b, f_b = [], [], [], []
    obs_b.append(o); l_b.append(T(D["l"][s0])); f_b.append(feats_t(n, nd, c))
    with torch.no_grad():
        for k in range(Lb):
            a, w = A[:, k, :3] - 2.0 * v, A[:, k, 3:]             # mild velocity feedback (|v| stays ~<2 m/s)
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
    # velocity drift guard: cup-origin velocity in branches is kept to the data range
    obs[..., 3:6] = obs[..., 3:6].clamp(-3.0, 3.0)
    return dict(obs=obs, act=act, l=l, feat=feat)


# ----------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=160)
    ap.add_argument("--burn", type=int, default=60)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n-os", type=int, default=1024)
    ap.add_argument("--w-mar", type=float, default=2.0)
    ap.add_argument("--aux-w", type=float, default=0.0)
    ap.add_argument("--branch-frac", type=float, default=0.0)
    ap.add_argument("--n-branch", type=int, default=30000)
    ap.add_argument("--branch-len", type=int, default=100)
    ap.add_argument("--gp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--eval-eps", type=int, default=300)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from world_model.v2.latent_hj import kin_wm
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    trD, va = C.load_split("train"), C.load_split("val")
    n_tr = len(trD["lens"])
    keep = np.sort(rng.choice(n_tr, max(8, int(round(a.frac * n_tr))), replace=False))
    tr = C.subset_eps(trD, keep)
    del trD
    wm = kin_wm.build_model().to(dev)
    aux = nn.Sequential(nn.Linear(wm.d_h + 9, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 10)).to(dev)
    params = list(wm.parameters()) + (list(aux.parameters()) if a.aux_w > 0 else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    W, burn, K = a.window, a.burn, a.K

    # privileged features for real episodes (aux targets only)
    def add_feats(D):
        n = torch.tensor(D["n"]); nd = torch.tensor(D["nd"]); c = torch.tensor(D["obs"][:, :3])
        c = c / c.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            D["feat"] = feats_t(n, nd, c).numpy().astype(np.float32)
    add_feats(tr); add_feats(va)
    gpu = {k: torch.tensor(tr[k], device=dev) for k in ("obs", "act", "l", "feat")}
    gva = {k: torch.tensor(va[k], device=dev) for k in ("obs", "act", "l", "feat")}

    BR = None
    if a.branch_frac > 0:
        t0 = time.time()
        BR = make_branches(tr, a.n_branch, a.branch_len, W, dev, seed=a.seed + 1)
        print(f"[branch] {a.n_branch} branches x {a.branch_len} ticks in {time.time()-t0:.0f}s; "
              f"branch-spill frac {(BR['l'][:, -a.branch_len:] < 0).any(1).float().mean():.2f} "
              f"tick-spill frac {(BR['l'][:, -a.branch_len:] < 0).float().mean():.2f}", flush=True)

    # normalization stats over the data the WM will see (+ branch windows, sampled at the batch ratio)
    O, Aa = tr["obs"], tr["act"]
    if BR is not None:
        mix = a.branch_frac / max(1e-6, 1 - a.branch_frac)
        nbt = int(min(len(O) * mix, BR["obs"].shape[0] * BR["obs"].shape[1]))
        bo = BR["obs"].reshape(-1, 9).cpu().numpy(); ba = BR["act"].reshape(-1, 6).cpu().numpy()
        j = rng.choice(len(bo), nbt, replace=False)
        O = np.concatenate([O, bo[j]], 0); Aa = np.concatenate([Aa, ba[j]], 0)
    wm.o_mu.copy_(torch.tensor(O.mean(0))); wm.o_sd.copy_(torch.tensor(O.std(0) + 1e-3))
    wm.a_mu.copy_(torch.tensor(Aa.mean(0))); wm.a_sd.copy_(torch.tensor(Aa.std(0) + 1e-3))
    del O, Aa

    def batch(D, G, B, from_branch=0):
        ok = np.flatnonzero(D["lens"] > W + 1)
        wts = np.where(D["onset"][ok] >= 0, 2.0, 1.0); wts /= wts.sum()
        e = rng.choice(ok, B, p=wts)
        s0 = np.array([0 if rng.random() < 0.3 else rng.integers(0, D["lens"][i] - W) for i in e])
        ix = torch.tensor((D["start"][e] + s0)[:, None] + np.arange(W)[None], device=dev)
        out = [G[k][ix] for k in ("obs", "act", "l", "feat")] + [torch.tensor(s0 == 0, device=dev)]
        if from_branch and BR is not None:
            j = torch.randint(len(BR["obs"]), (from_branch,), device=dev)
            add = [BR[k][j] for k in ("obs", "act", "l", "feat")] + [torch.zeros(from_branch, dtype=torch.bool, device=dev)]
            out = [torch.cat([x, y], 0) for x, y in zip(out, add)]
        return out

    def loss_fn(obs, act, l, feat, from_start, train=True):
        Hs, on = C.filter_fast(wm, obs, act)
        B, T, _ = obs.shape
        m = torch.ones(B, T, device=dev); m[~from_start, :burn] = 0.0
        L_obs = (((wm.dec(Hs[:, 1:]) - on[:, 1:]) ** 2).mean(-1) * m[:, 1:]).sum() / m[:, 1:].sum()
        if a.gp > 0 and train:
            xin = torch.cat([Hs, on], -1)
            xs = xin[m > 0][torch.randint(int((m > 0).sum()), (2048,), device=dev)].detach().requires_grad_(True)
            lg = torch.tanh(wm.mhead(xs)[..., 0])
            gr = torch.autograd.grad(lg.sum(), xs, create_graph=True)[0]
            L_gp = (gr ** 2).sum(-1).mean()
        else:
            L_gp = torch.zeros((), device=dev)
        lh = wm.margin(Hs, on)
        L_mar = (((lh - l) ** 2) * m).sum() / m.sum()
        L_aux = torch.zeros((), device=dev)
        if a.aux_w > 0:
            fa = aux(torch.cat([Hs, on], -1))
            L_aux = (((fa - feat) ** 2).mean(-1) * m).sum() / m.sum()
        ts = torch.randint(burn, T - K - 1, (a.n_os,), device=dev)
        bs = torch.randint(0, B, (a.n_os,), device=dev)
        h = Hs[bs, ts]; o_n = on[bs, ts]
        L_os_o, L_os_m, L_os_a = 0.0, 0.0, 0.0
        for k in range(K):
            h, o_hat = wm.step(h, o_n, act[bs, ts + k])
            L_os_o = L_os_o + ((o_hat - on[bs, ts + k + 1]) ** 2).mean()
            L_os_m = L_os_m + ((wm.margin(h, o_hat) - l[bs, ts + k + 1]) ** 2).mean()
            if a.aux_w > 0:
                L_os_a = L_os_a + ((aux(torch.cat([h, o_hat], -1)) - feat[bs, ts + k + 1]) ** 2).mean()
            o_n = o_hat
        tot = (L_obs + L_os_o / K + a.w_mar * (L_mar + L_os_m / K) + a.aux_w * (L_aux + L_os_a / K)
               + a.gp * L_gp)
        return tot, dict(obs=float(L_obs), mar=float(L_mar), os_o=float(L_os_o / K), os_m=float(L_os_m / K),
                         aux=float(L_aux), gp=float(L_gp))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    curve = []; t0 = time.time()
    nb = int(round(a.batch * a.branch_frac)) if BR is not None else 0
    for it in range(1, a.steps + 1):
        obs, act, l, feat, fs = batch(tr, gpu, a.batch - nb, from_branch=nb)
        loss, lg = loss_fn(obs, act, l, feat, fs)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step(); sched.step()
        if it % a.log_every == 0 or it == 1:
            with torch.no_grad():
                vl = loss_fn(*batch(va, gva, 256), train=False)[1]
            curve.append(dict(it=it, train=lg, val=vl, sec=time.time() - t0))
            print(f"[wm] it {it} ({time.time()-t0:.0f}s) train " + " ".join(f"{k}={v:.4f}" for k, v in lg.items())
                  + " | val " + " ".join(f"{k}={v:.4f}" for k, v in vl.items()), flush=True)
    wm.eval()
    torch.save(dict(state={k: v.cpu() for k, v in wm.state_dict().items()}, args=vars(a), n_train_eps=int(len(keep)),
                    aux_state={k: v.cpu() for k, v in aux.state_dict().items()} if a.aux_w > 0 else None),
               out / "wm.pt")
    (out / "curve.json").write_text(json.dumps(curve))
    # ---------------------------------------------------------------- held-out evaluation
    ev = evaluate(wm, dev, n_eps=a.eval_eps)
    ev.update(n_train_eps=int(len(keep)), n_train_ticks=int(tr["lens"].sum()), train_sec=time.time() - t0, args=vars(a))
    (out / "eval.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1), flush=True)


def evaluate(wm, dev, n_eps=300, Ks=(1, 10, 25, 50), seed=0):
    """held-out TEST split: filtered margin, open-loop multi-step prediction under DATASET actions,
    margin AUROCs, and the counterfactual (fallback-action) test."""
    import torch
    from world_model.v2.wm_audit import diagnose as dg
    D = C.load_split("test")
    eps = np.arange(min(n_eps, len(D["lens"])))
    Z = C.filtered_latents(wm, D, dev, eps)
    dh = wm.d_h
    ev = {}
    # filtered margin
    lh_all, l_all, y05 = [], [], []
    for z, e in zip(Z, eps):
        s, L = D["start"][e], D["lens"][e]
        with torch.no_grad():
            zt = torch.tensor(z, device=dev)
            lh = wm.margin(zt[:, :dh], zt[:, dh:]).cpu().numpy()
        l = D["l"][s:s + L]
        fut = np.array([l[t + 1:t + 26].min() if t + 1 < L else l[-1] for t in range(L)])
        lh_all.append(lh); l_all.append(l); y05.append(fut)
    lh_all, l_all, y05 = map(np.concatenate, (lh_all, l_all, y05))
    pos = l_all > 0
    ev["filtered_margin"] = dict(mae=float(np.abs(lh_all - l_all).mean()), corr=float(np.corrcoef(lh_all, l_all)[0, 1]),
                                 auroc_now=C.auroc(-lh_all, l_all < 0),
                                 auroc_spill_within_0p5s_given_safe_now=C.auroc(-lh_all[pos], y05[pos] < 0))
    # open-loop multi-step prediction (dataset actions) from filtered states
    rng = np.random.default_rng(seed)
    Kmax = max(Ks)
    starts = []
    for z, e in zip(Z, eps):
        L = D["lens"][e]
        for t in range(20, L - Kmax - 1, 7):
            starts.append((e, t))
    starts = np.array(starts)
    starts = starts[rng.choice(len(starts), min(6000, len(starts)), replace=False)]
    zmap = dict(zip(eps.tolist(), Z))
    z0 = np.stack([zmap[e][t] for e, t in starts])
    s0 = D["start"][starts[:, 0]] + starts[:, 1]
    idx = s0[:, None] + np.arange(Kmax)[None]
    with torch.no_grad():
        act = torch.tensor(D["act"][idx], device=dev)
        h = torch.tensor(z0[:, :dh], device=dev); o = torch.tensor(z0[:, dh:], device=dev)
        LM, OE = [], []
        for k in range(Kmax):
            h, o = wm.step(h, o, act[:, k])
            LM.append(wm.margin(h, o).cpu().numpy())
            o_true = wm.no(torch.tensor(D["obs"][idx[:, k] + 1], device=dev))
            OE.append(((o - o_true) ** 2).mean(-1).cpu().numpy())
    LM, OE = np.stack(LM, 1), np.stack(OE, 1)
    LT = D["l"][idx + 1]
    ev["openloop_dataset_actions"] = {f"k{k}": dict(obs_mse_norm=float(OE[:, k - 1].mean()),
                                                    margin_mae=float(np.abs(LM[:, k - 1] - LT[:, k - 1]).mean()),
                                                    margin_auroc=C.auroc(-LM[:, k - 1], LT[:, k - 1] < 0))
                                      for k in Ks}
    # counterfactual fallback-action test
    pick, l0, R = dg.counterfactual(wm, D, dev, n_states=1500, K=50, seed=seed)
    ev["counterfactual"] = dg.cf_metrics(l0, R, 50)
    return ev


if __name__ == "__main__":
    main()
