"""wm_audit/diagnose.py -- action coverage, latent probes and counterfactual action tests
for a kin_wm-style latent WM (SIMULATION / SAVED DATA ONLY; no hardware, no network).

  coverage : dataset (cup accel, axis ang-vel) distribution vs the fallback set U
  probe    : can the privileged slosh state / physics value be read out of the WM latent?
             (Kim, Nakamura, Bajcsy 2025: estimation gap vs prediction gap)
  cf       : counterfactual action test: WM-predicted margins vs the ground-truth liquid
             proxy under (i) logged dataset actions and (ii) every held fallback action

  python -m world_model.v2.wm_audit.diagnose coverage --out $AUD/results/coverage.json
  CUDA_VISIBLE_DEVICES=1 python -m world_model.v2.wm_audit.diagnose probe --wm <wm.pt> --out ...
  CUDA_VISIBLE_DEVICES=1 python -m world_model.v2.wm_audit.diagnose cf --wm <wm.pt> --out ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C


# ============================================================================ coverage
def cmd_coverage(a):
    tr = C.load_split("train")
    A = tr["act"].astype(np.float64)
    acc, w = A[:, :3], A[:, 3:]
    ah = np.linalg.norm(acc[:, :2], axis=1); az = np.abs(acc[:, 2])
    wh = np.linalg.norm(w[:, :2], axis=1); wz = np.abs(w[:, 2])
    pct = lambda x: {f"p{q}": float(np.percentile(x, q)) for q in (50, 90, 99, 99.9)} | {"max": float(x.max())}
    out = dict(n_ticks=int(len(A)), n_eps=int(len(tr["lens"])),
               acc_h=pct(ah), acc_z=pct(az), w_h=pct(wh), w_z=pct(wz))
    U = C.fallback_U()
    # moving vs idle ticks
    moving = (ah > 0.2) | (wh > 0.05)
    out["frac_ticks_moving"] = float(moving.mean())
    tol_a, tol_w = a.tol_a, a.tol_w
    # per-u single-tick and sustained coverage (same u within tolerance for k consecutive ticks)
    lens, st = tr["lens"], tr["start"]
    per_u = []
    for i, u in enumerate(U):
        near = (np.linalg.norm(acc - u[:3], axis=1) < tol_a) & (np.linalg.norm(w - u[3:], axis=1) < tol_w)
        rec = dict(u=[round(float(x), 2) for x in u], frac_ticks=float(near.mean()))
        for k in (10, 25):
            cnt, tot = 0, 0
            for s, L in zip(st, lens):
                x = near[s:s + L].astype(np.int32)
                if L < k:
                    continue
                cs = np.concatenate([[0], np.cumsum(x)])
                win = cs[k:] - cs[:-k]
                cnt += int((win == k).sum()); tot += len(win)
            rec[f"frac_windows_held_{k}"] = cnt / max(tot, 1)
        per_u.append(rec)
    out["per_u"] = per_u
    nz = [r for r in per_u if (abs(r["u"][0]) + abs(r["u"][1]) > 0 or abs(r["u"][3]) + abs(r["u"][4]) > 0)]
    out["summary"] = dict(
        n_u=len(U),
        n_u_with_any_tick=int(sum(r["frac_ticks"] > 0 for r in per_u)),
        n_u_ge_1e3_ticks=int(sum(r["frac_ticks"] >= 1e-3 for r in per_u)),
        n_u_held_0p2s_ge_1e4=int(sum(r["frac_windows_held_10"] >= 1e-4 for r in per_u)),
        n_u_held_0p5s_any=int(sum(r["frac_windows_held_25"] > 0 for r in per_u)),
        median_frac_ticks_nonzero_u=float(np.median([r["frac_ticks"] for r in nz])),
        tol_a=tol_a, tol_w=tol_w)
    # nearest-u quantization: which U cells does the data occupy, overall vs in danger states
    sub = np.random.default_rng(0).choice(len(A), min(len(A), 250000), replace=False)
    d = (np.linalg.norm((acc[sub, None, :] - U[None, :, :3]) / 5.0, axis=2) ** 2
         + np.linalg.norm((w[sub, None, :] - U[None, :, 3:]) / 2.0, axis=2) ** 2)
    q = d.argmin(1)
    l = tr["l"][sub]
    danger = (l > 0) & (l < 0.5)
    hist_all = np.bincount(q, minlength=len(U)) / len(q)
    hist_d = np.bincount(q[danger], minlength=len(U)) / max(danger.sum(), 1)
    out["nearest_u_hist_all"] = hist_all.round(5).tolist()
    out["nearest_u_hist_danger"] = hist_d.round(5).tolist()
    out["frac_ticks_nearest_u0"] = float(hist_all[0]); out["frac_danger_nearest_u0"] = float(hist_d[0])
    out["n_danger_ticks"] = int(danger.sum())
    # action entropy (nats) of the nearest-u histogram
    ent = lambda h: float(-(h[h > 0] * np.log(h[h > 0])).sum())
    out["entropy_all"], out["entropy_danger"], out["entropy_uniform"] = ent(hist_all), ent(hist_d), float(np.log(len(U)))
    # tilt range seen
    tilt = np.degrees(np.arccos(np.clip(tr["obs"][:, 2], -1, 1)))
    out["tilt_deg"] = pct(tilt)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_u", "nearest_u_hist_all", "nearest_u_hist_danger")},
                     indent=1), flush=True)


# ============================================================================ probe
def _ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = Ytr.mean(0)
    Xa = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    W = np.linalg.solve(Xa.T @ Xa + lam * len(Xa) * np.eye(Xa.shape[1]), Xa.T @ (Ytr - ym))
    P = np.concatenate([Xte, np.ones((len(Xte), 1))], 1) @ W + ym
    return 1 - ((P - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0), P


def _mlp_fit(Xtr, Ytr, Xte, Yte, dev, steps=3000, cls=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ymu, ysd = (0.0, 1.0) if cls else (Ytr.mean(0), Ytr.std(0) + 1e-6)
    X = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    Y = torch.tensor((Ytr - ymu) / ysd, dtype=torch.float32, device=dev)
    Xt = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for it in range(steps):
        i = torch.randint(len(X), (1024,), device=dev)
        p = net(X[i])
        loss = nn.functional.binary_cross_entropy_with_logits(p, Y[i]) if cls else ((p - Y[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = net(Xt).cpu().numpy()
    if cls:
        return P
    P = P * ysd + ymu
    return 1 - ((P - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)


def probe_targets(D, eps, stride, vnet, vdyn):
    import torch
    fn = C.sp.torch_fns()
    rows = []
    for e in eps:
        s, L = D["start"][e], D["lens"][e]
        ix = np.arange(s, s + L, stride)
        rows.append(ix)
    ix = np.concatenate(rows)
    n = torch.tensor(D["n"][ix], dtype=torch.float32); nd = torch.tensor(D["nd"][ix], dtype=torch.float32)
    c = torch.tensor(D["obs"][ix, :3], dtype=torch.float32); c = c / c.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        feats = fn["feats"](n, nd, c, nd_scale=1.0 / C.P0.w1).clamp(-4, 4).numpy()
        V = C.phys_V(vnet, vdyn, n, nd, c).numpy()
    l = D["l"][ix]
    phid, ndn = [], []
    for e in eps:
        s_, L_ = D["start"][e], D["lens"][e]
        cc = D["obs"][s_:s_ + L_, :3] / np.linalg.norm(D["obs"][s_:s_ + L_, :3], axis=1, keepdims=True)
        ph = np.arccos(np.clip(-(D["n"][s_:s_ + L_] * cc).sum(1), -1, 1))
        phid.append(np.gradient(ph, C.DT_WM)[::stride] / C.P0.w1)
        ndn.append(np.linalg.norm(D["nd"][s_:s_ + L_], axis=1)[::stride] / C.P0.w1)
    phid = np.clip(np.concatenate(phid), -4, 4); ndn = np.clip(np.concatenate(ndn), 0, 4)
    phi = np.arccos(np.clip(-(D["n"][ix] * D["obs"][ix, :3]).sum(1) / np.linalg.norm(D["obs"][ix, :3], axis=1), -1, 1))
    Y = np.concatenate([feats, phi[:, None], l[:, None], V[:, None], (V - l)[:, None], phid[:, None], ndn[:, None]], 1)
    names = ["n_z", "c_z", "nd_z", "n.c_h", "nxc_h", "nd.n_h", "nxnd_h", "nd.c_h", "cxnd_h", "|nd_h|^2", "phi", "l",
             "V_phys", "V_phys-l", "phi_dot/w1", "|nd|/w1"]
    keep = l > -0.2                                 # not deep in spill (whirling bob is irrelevant to HJ)
    return ix[keep], Y[keep], names, keep


def cmd_probe(a):
    import torch
    from world_model.v2.latent_hj import kin_wm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wm = kin_wm.load_wm(a.wm, dev)
    vnet, vdyn, _ = C.load_phys_value()
    out = dict(wm=a.wm)
    sets = {}
    for name, split, neps in (("fit", "val", a.n_eps), ("eval", "test", a.n_eps)):
        D = C.load_split(split)
        eps = np.arange(min(neps, len(D["lens"])))
        Z = C.filtered_latents(wm, D, dev, eps)
        ix, Y, names, keep = probe_targets(D, eps, a.stride, vnet, vdyn)
        Zc = np.concatenate([z[::a.stride] for z in Z], 0)[keep]
        # 1-step-lagged obs history (o_{t-1}, o_{t-2}) as a no-memory baseline with finite differences
        O = D["obs"][ix]
        sets[name] = dict(Z=Zc, Y=Y, O=O, l=D["l"][ix])
    dh = wm.d_h
    feats = dict(h=lambda s: s["Z"][:, :dh], o=lambda s: s["O"], h_o=lambda s: s["Z"])
    for fname, f in feats.items():
        r2, _ = _ridge_r2(f(sets["fit"]), sets["fit"]["Y"], f(sets["eval"]), sets["eval"]["Y"])
        out[f"ridge_R2[{fname}]"] = dict(zip(names, np.round(r2, 4).tolist()))
        print(fname, "ridge", out[f"ridge_R2[{fname}]"], flush=True)
        r2m = _mlp_fit(f(sets["fit"]), sets["fit"]["Y"], f(sets["eval"]), sets["eval"]["Y"], dev)
        out[f"mlp_R2[{fname}]"] = dict(zip(names, np.round(r2m, 4).tolist()))
        print(fname, "mlp", out[f"mlp_R2[{fname}]"], flush=True)
    # doomed-state classification probe: y = (l > 0) & (V_phys <= 0), on l>0 states
    for fname, f in feats.items():
        m_fit = sets["fit"]["Y"][:, 11] > 0; m_ev = sets["eval"]["Y"][:, 11] > 0
        yf = (sets["fit"]["Y"][m_fit, 12] <= 0).astype(np.float32)[:, None]
        ye = (sets["eval"]["Y"][m_ev, 12] <= 0)
        p = _mlp_fit(f(sets["fit"])[m_fit], yf, f(sets["eval"])[m_ev], ye[:, None].astype(np.float32), dev, cls=True)
        out[f"doomed_probe_auroc[{fname}]"] = C.auroc(p[:, 0], ye)
        out["doomed_frac_among_lpos_eval"] = float(ye.mean())
        print(fname, "doomed-probe AUROC", out[f"doomed_probe_auroc[{fname}]"], flush=True)
    # the WM's own margin head vs l, and the value-relevant question: does l^ separate doomed states?
    with torch.no_grad():
        Ze = torch.tensor(sets["eval"]["Z"], device=dev)
        lh = wm.margin(Ze[:, :dh], Ze[:, dh:]).cpu().numpy()
    le, Ve = sets["eval"]["Y"][:, 11], sets["eval"]["Y"][:, 12]
    m = le > 0
    out["margin_head_eval"] = dict(mae=float(np.abs(lh - le).mean()), corr=float(np.corrcoef(lh, le)[0, 1]),
                                   auroc_lhat_for_doomed=C.auroc(-lh[m], Ve[m] <= 0),
                                   auroc_ltrue_for_doomed=C.auroc(-le[m], Ve[m] <= 0))
    print(out["margin_head_eval"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


# ============================================================================ counterfactual
def sample_states(D, n, rng, K, danger_frac=0.67):
    """(episode, tick) pairs with t+K inside the episode: 1/3 from the 0.6 s before a spill onset
    (where doomed states live), 1/3 from 0<l<0.6 (near danger), 1/3 uniform."""
    cand, dang, pre = [], [], []
    for e in range(len(D["lens"])):
        s, L = D["start"][e], D["lens"][e]
        on = D["onset"][e]
        for t in range(20, L - K - 1, 2):
            cand.append((e, t))
            if 0 < D["l"][s + t] < 0.6:
                dang.append((e, t))
            if on >= 0 and on - 30 <= t < on:
                pre.append((e, t))
    cand, dang, pre = np.array(cand), np.array(dang), np.array(pre)
    n3 = n // 3
    npre = min(n3, len(pre)); nd_ = min(n3, len(dang))
    pick = np.concatenate([pre[rng.choice(len(pre), npre, replace=False)],
                           dang[rng.choice(len(dang), nd_, replace=False)],
                           cand[rng.choice(len(cand), n - npre - nd_, replace=False)]], 0)
    return pick


def spearman_rows(A, B):
    ra = np.argsort(np.argsort(A, 1), 1).astype(float); rb = np.argsort(np.argsort(B, 1), 1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    ok = den > 0
    return np.where(ok, (ra * rb).sum(1) / np.where(ok, den, 1), np.nan)


def counterfactual(wm, D, dev, n_states=3000, K=50, seed=0, chunk=500):
    import torch
    rng = np.random.default_rng(seed)
    pick = sample_states(D, n_states, rng, K)
    eps = np.unique(pick[:, 0])
    Zep = dict(zip(eps.tolist(), C.filtered_latents(wm, D, dev, eps)))
    z0 = np.stack([Zep[e][t] for e, t in pick]); s0 = D["start"][pick[:, 0]] + pick[:, 1]
    n0, nd0 = D["n"][s0], D["nd"][s0]
    c0 = D["obs"][s0, :3] / np.linalg.norm(D["obs"][s0, :3], axis=1, keepdims=True)
    l0 = D["l"][s0]
    U = torch.tensor(C.fallback_U(), dtype=torch.float32, device=dev)
    nu = len(U)
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    res = dict(l_gt=[], l_wm=[], lg_data=[], lw_data=[], lgt_replay_data=[])
    with torch.no_grad():
        for i0 in range(0, len(pick), chunk):
            sl = slice(i0, i0 + chunk)
            B = len(pick[sl])
            # ---- logged dataset actions (K ticks)
            idx = s0[sl][:, None] + np.arange(K)[None]
            act = T(D["act"][idx])
            l_true = D["l"][idx + 1]
            lw, _ = C.wm_rollout(wm, T(z0[sl]), act[..., :3], act[..., 3:])
            lg_rep, _ = C.pend_rollout_torch(T(n0[sl]), T(nd0[sl]), T(c0[sl]), act[..., :3], act[..., 3:], project=False)
            res["lg_data"].append(l_true); res["lw_data"].append(lw.cpu().numpy())
            res["lgt_replay_data"].append(lg_rep.cpu().numpy())
            # ---- every held fallback action
            rep = lambda x: T(x).repeat_interleave(nu, 0)
            a_seq = U[:, :3].repeat(B, 1)[:, None].expand(-1, K, -1).contiguous()
            w_seq = U[:, 3:].repeat(B, 1)[:, None].expand(-1, K, -1).contiguous()
            lg, weff = C.pend_rollout_torch(rep(n0[sl]), rep(nd0[sl]), rep(c0[sl]), a_seq, w_seq, project=True)
            lwu, _ = C.wm_rollout(wm, rep(z0[sl]), a_seq, weff)          # WM gets the SAME realized motion
            res["l_gt"].append(lg.view(B, nu, K).cpu().numpy()); res["l_wm"].append(lwu.view(B, nu, K).cpu().numpy())
    R = {k: np.concatenate(v, 0) for k, v in res.items()}
    return pick, l0, R


def cf_metrics(l0, R, K):
    out = {}
    ks = [k for k in (1, 5, 10, 25, 50) if k <= K]
    out["replay_check_mae_data"] = {k: float(np.abs(R["lgt_replay_data"][:, k - 1] - R["lg_data"][:, k - 1]).mean()) for k in ks}
    out["wm_mae_dataset_actions"] = {k: float(np.abs(R["lw_data"][:, k - 1] - R["lg_data"][:, k - 1]).mean()) for k in ks}
    out["wm_mae_fallback_actions"] = {k: float(np.abs(R["l_wm"][:, :, k - 1] - R["l_gt"][:, :, k - 1]).mean()) for k in ks}
    for H in (10, 25, 50):
        if H > K:
            continue
        mg = R["l_gt"][:, :, :H].min(2); mw = R["l_wm"][:, :, :H].min(2)       # (N, nu) min margin over horizon
        rho = spearman_rows(mw, mg)
        pos = l0 > 0
        gt_escape = (mg > 0).any(1); wm_escape = (mw > 0).any(1)
        doomed = pos & ~gt_escape
        safe = pos & gt_escape
        gstd, wstd = mg.std(1), mw.std(1)
        out[f"H{H}"] = dict(
            spearman_across_U_median=float(np.nanmedian(rho)),
            spearman_across_U_frac_gt_0p5=float(np.nanmean(rho > 0.5)),
            sign_agree_minmargin=float(((mg > 0) == (mw > 0)).mean()),
            action_sensitivity_ratio_median=float(np.median(wstd[gstd > 0.02] / gstd[gstd > 0.02])),
            frac_lpos_doomed_openloop_gt=float(doomed.sum() / max(pos.sum(), 1)),
            wm_recall_openloop_doomed=float((~wm_escape[doomed]).mean()) if doomed.any() else float("nan"),
            wm_false_doomed_rate=float((~wm_escape[safe]).mean()) if safe.any() else float("nan"),
            best_u_match=float((mg.argmax(1) == mw.argmax(1)).mean()),
            gt_value_of_wm_best_u_regret=float(np.mean(mg.max(1) - mg[np.arange(len(mg)), mw.argmax(1)])),
            wm_optimism_best_u=float(np.mean(mw.max(1) - mg.max(1))),
        )
    return out


def cmd_cf(a):
    import torch
    from world_model.v2.latent_hj import kin_wm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wm = kin_wm.load_wm(a.wm, dev)
    D = C.load_split("test")
    t0 = time.time()
    pick, l0, R = counterfactual(wm, D, dev, a.n_states, a.K, a.seed)
    out = dict(wm=a.wm, n_states=len(pick), K=a.K, sec=time.time() - t0) | cf_metrics(l0, R, a.K)
    print(json.dumps(out, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    if a.save_npz:
        np.savez_compressed(Path(a.out).with_suffix(".npz"), pick=pick, l0=l0, **R)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage"); c.add_argument("--out", required=True)
    c.add_argument("--tol-a", type=float, default=1.0); c.add_argument("--tol-w", type=float, default=0.5)
    p = sub.add_parser("probe"); p.add_argument("--wm", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n-eps", type=int, default=400); p.add_argument("--stride", type=int, default=3)
    f = sub.add_parser("cf"); f.add_argument("--wm", required=True); f.add_argument("--out", required=True)
    f.add_argument("--n-states", type=int, default=3000); f.add_argument("--K", type=int, default=50)
    f.add_argument("--seed", type=int, default=0); f.add_argument("--save-npz", action="store_true")
    a = ap.parse_args()
    dict(coverage=cmd_coverage, probe=cmd_probe, cf=cmd_cf)[a.cmd](a)


if __name__ == "__main__":
    main()
