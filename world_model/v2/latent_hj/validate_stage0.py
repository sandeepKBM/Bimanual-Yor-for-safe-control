"""validate_stage0.py -- Stage 0: reachability learner on the ANALYTIC liquid-proxy
latent, validated against the exact grid HJ and gated on held-out episodes.

  train   : fit V (reach_value.train, PendDyn) on GPU
  compare : learned V vs exact planar grid value (grid_hj) on planar states:
            MAE, safe-set IoU / sign agreement, greedy-escape agreement, and the
            cosine of d/du V(f(z,u))|_{u=0} (learned autograd vs grid finite diff)
  gate    : VALUE-VALIDATION GATE on held-out spill_v2 episodes (dense 200 Hz cup
            motion, liquid proxy replayed): alarm = first step with score < eps,
            eps chosen on VAL shards for an episode-level false-alarm budget on
            non-spill episodes, reported on TEST shards: lead time before spill
            onset for (i) learned V, (ii) instantaneous margin l (= the F2
            classifier mistake), (iii) any extra scorer passed in.

  CUDA_VISIBLE_DEVICES=6 python -m world_model.v2.latent_hj.validate_stage0 train --out $S/v0
  python -m world_model.v2.latent_hj.validate_stage0 compare --ckpt $S/v0/value.pt --grid $S/grid_g099.npz
  python -m world_model.v2.latent_hj.validate_stage0 gate --ckpt $S/v0/value.pt --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.latent_hj import slosh_pend as sp  # noqa: E402

SCR = Path(os.environ.get("LATENT_HJ_SCR", "/common/home/ss5772/.claude/jobs/f1580deb/tmp/latent_hj"))


# ------------------------------------------------------------------ train
def cmd_train(a):
    import torch
    from world_model.v2.latent_hj import reach_value as rv
    P = sp.PendParams(h=a.h, A_max=a.A_max, W_max=a.W_max, m_lo=a.m_lo)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dyn = rv.PendDyn(P, device=dev)
    grid = None
    if a.grid and Path(a.grid).exists():
        grid = dict(np.load(a.grid, allow_pickle=True))

    def eval_fn(net):
        if grid is None:
            return {}
        r = compare_grid(net, dyn, grid, n=20000, seed=1, grads=False)
        return dict(mae=r["mae_all"], mae_rel=r["mae_relevant"], iou=r["iou_safe"])

    net, curve = rv.train(dyn, a.out, iters=a.iters, batch=a.batch, lr=a.lr, gamma0=a.gamma0,
                          gamma_final=a.gamma, K=a.K, lam=a.lam, device=dev, eval_fn=eval_fn,
                          log_every=a.log_every, seed=a.seed)
    dx = dyn.features(dyn.sample_states(2)).shape[-1]
    meta = dict(dx=dx, hidden=256, layers=3, gamma=a.gamma, K=a.K, lam=a.lam, iters=a.iters, P=P.as_dict(),
                kind="pend_analytic")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rv.save(net.cpu(), out / "value.pt", meta)
    (out / "curve.json").write_text(json.dumps(curve))
    print(f"[train] saved {out/'value.pt'}", flush=True)


# ------------------------------------------------------------------ grid compare
def compare_grid(net, dyn, grid, n=50000, seed=0, grads=True, eps_rel=0.02):
    """Learned (3-D net on planar-embedded states) vs exact planar grid."""
    import torch
    from world_model.v2.latent_hj import grid_hj as gh
    rng = np.random.default_rng(seed)
    g0, g1, g2 = grid["g0"], grid["g1"], grid["g2"]
    th = rng.uniform(g0[0] * 0.95, g0[-1] * 0.95, n)
    thd = rng.uniform(g1[0] * 0.8, g1[-1] * 0.8, n)
    al = rng.uniform(g2[0] * 0.95, g2[-1] * 0.95, n)
    Vg = gh.interp(grid, th, thd, al)
    lg = np.clip((dyn.P.phi_star - np.abs(th + al)) / dyn.P.phi_star, dyn.P.m_lo, 1)
    nn_, nd, c = sp.planar_to_3d(th, thd, al)
    dev = next(net.parameters()).device
    z = torch.tensor(np.concatenate([nn_, nd, c], -1), dtype=torch.float32, device=dev)
    with torch.no_grad():
        Vl = net(dyn.features(z), dyn.margin(z)).cpu().numpy()
    rel = (lg > -0.2)                        # "relevant": not already deep in spill
    out = dict(n=n, mae_all=float(np.abs(Vl - Vg).mean()), mae_relevant=float(np.abs(Vl - Vg)[rel].mean()),
               p95_abs_err=float(np.percentile(np.abs(Vl - Vg), 95)),
               corr=float(np.corrcoef(Vl, Vg)[0, 1]))
    sl, sg = Vl > 0, Vg > 0
    out["iou_safe"] = float((sl & sg).sum() / max((sl | sg).sum(), 1))
    out["sign_agree"] = float((sl == sg).mean())
    out["false_safe_frac"] = float((sl & ~sg).mean())      # learned says safe, exact says doomed (dangerous)
    out["false_unsafe_frac"] = float((~sl & sg).mean())
    # the states where reachability matters: l>0 but exact V<=0 (doomed though currently fine)
    doomed = (lg > 0) & (Vg <= 0)
    out["frac_doomed_but_l_pos"] = float(doomed.mean())
    out["learned_detects_doomed"] = float((Vl[doomed] <= 0).mean()) if doomed.any() else float("nan")
    out["l_detects_doomed"] = 0.0     # by definition l>0 there
    out["mae_l_as_value"] = float(np.abs(lg - Vg).mean())
    if not grads:
        return out
    # ---- escape direction: d/du V(f(z,u)) at u=0 over planar controls (a_x, w_y)
    m = (lg > 0.0) & (Vg > -0.5) & (np.abs(thd) < 20)
    idx = np.flatnonzero(m)[:4000]
    P = dyn.P
    U = torch.zeros(len(idx), 6, device=dev, requires_grad=True)
    zz = z[idx]
    nxt = dyn.step(zz, U)
    v = net(dyn.features(nxt), dyn.margin(nxt))
    gl = torch.autograd.grad(v.sum(), U)[0][:, [0, 4]].cpu().numpy()
    # grid finite difference of V_grid(f(x,u)) wrt (a_x, w_y) at u=0 (central, one control step)
    L, c2, h = P.L, 2 * P.zeta * P.w1, P.dt / P.sub
    def vg_next(ax, wy):
        o = np.array([gh._planar_step(th[i], thd[i], al[i], ax, wy, L, c2, h, P.sub, P.tilt_lim, sp.G) for i in idx])
        return gh.interp(grid, o[:, 0], o[:, 1], o[:, 2])
    da, dw = 1.0, 0.4
    gg = np.stack([(vg_next(da, 0) - vg_next(-da, 0)) / (2 * da), (vg_next(0, dw) - vg_next(0, -dw)) / (2 * dw)], 1)
    # scale-normalize the two control axes by their bounds before the cosine
    s = np.array([P.A_max, P.W_max])
    gl_s, gg_s = gl * s, gg * s
    ok = (np.linalg.norm(gg_s, axis=1) > 1e-4) & (np.linalg.norm(gl_s, axis=1) > 1e-6)
    cos = np.sum(gl_s * gg_s, 1)[ok] / (np.linalg.norm(gl_s, axis=1)[ok] * np.linalg.norm(gg_s, axis=1)[ok])
    out["grad_cos_median"] = float(np.median(cos)) if ok.any() else float("nan")
    out["grad_cos_frac_pos"] = float((cos > 0).mean()) if ok.any() else float("nan")
    out["grad_cos_frac_gt_0.7"] = float((cos > 0.7).mean()) if ok.any() else float("nan")
    out["grad_n"] = int(ok.sum())
    # greedy-escape quality: exact value achieved by the learned greedy action vs the exact optimum
    Uc = torch.tensor(sp.control_set(P, planar=True), dtype=torch.float32, device=dev)
    with torch.no_grad():
        nx = dyn.step(zz[:, None, :].expand(-1, len(Uc), -1), Uc[None].expand(len(idx), -1, -1))
        vv = net(dyn.features(nx), dyn.margin(nx)).cpu().numpy()
    ul = vv.argmax(1)
    Upl = sp.control_set(P, planar=True)[:, [0, 4]]
    vgrid_all = np.stack([vg_next(u[0], u[1]) for u in Upl], 1)
    out["greedy_regret_mean"] = float(np.mean(vgrid_all.max(1) - vgrid_all[np.arange(len(idx)), ul]))
    out["greedy_exact_match"] = float(np.mean(vgrid_all.argmax(1) == ul))
    return out


def cmd_compare(a):
    import torch
    from world_model.v2.latent_hj import reach_value as rv
    net, meta = rv.load(a.ckpt)
    P = sp.PendParams.from_meta(meta["P"])
    dyn = rv.PendDyn(P, device="cpu")
    res = {}
    for gp in a.grid:
        grid = dict(np.load(gp, allow_pickle=True))
        t0 = time.time()
        r = compare_grid(net, dyn, grid, n=a.n, seed=0, grads=True)
        r["grid"] = gp; r["sec"] = time.time() - t0
        res[Path(gp).name] = r
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if len(a.grid) >= 2:
        # grid-vs-grid (resolution / discount) sanity: how exact is "exact"?
        from world_model.v2.latent_hj import grid_hj as gh
        A = dict(np.load(a.grid[0], allow_pickle=True)); B = dict(np.load(a.grid[1], allow_pickle=True))
        rng = np.random.default_rng(3)
        th = rng.uniform(-1.7, 1.7, 50000); thd = rng.uniform(-24, 24, 50000); al = rng.uniform(-1.4, 1.4, 50000)
        va, vb = gh.interp(A, th, thd, al), gh.interp(B, th, thd, al)
        res["grid_vs_grid"] = dict(a=a.grid[0], b=a.grid[1], mae=float(np.abs(va - vb).mean()),
                                   sign_agree=float(((va > 0) == (vb > 0)).mean()))
        print(json.dumps(res["grid_vs_grid"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ gate
def _ep_scores(args):
    """worker: one shard -> per-episode score traces (subsampled) + onset."""
    path, ckpt, stride = args
    import torch
    torch.set_num_threads(1)
    from world_model.v2.latent_hj import reach_value as rv, spill_data as sd
    net, meta = rv.load(ckpt)
    P = sp.PendParams.from_meta(meta["P"])
    dyn = rv.PendDyn(P, device="cpu")
    rows = []
    for ep in sd.load_dense(path):
        r = sd.replay_episode(ep, P)
        z = torch.tensor(np.concatenate([r["n"], r["nd"], r["c"]], 1)[::stride], dtype=torch.float32)
        with torch.no_grad():
            V = net(dyn.features(z), dyn.margin(z)).numpy()
        tilt = np.degrees(np.arccos(np.clip(r["c"][:, 2], -1, 1)))
        rows.append(dict(shard=Path(path).name, seed=ep["seed"], onset=r["onset"], stride=stride,
                         V=V.astype(np.float32), l=r["l"][::stride].astype(np.float32),
                         tilt_max=float(tilt.max()), T=len(r["phi"])))
    return rows


def first_cross(x, thr):
    i = np.flatnonzero(x < thr)
    return int(i[0]) if len(i) else -1


def gate_eval(rows_val, rows_test, key, fa_budget=0.05, dt=0.005):
    """eps from VAL non-spill episodes (episode-level FA <= budget), report TEST lead times."""
    neg_min = np.array([r[key].min() for r in rows_val if r["onset"] < 0])
    # alarm iff min score < eps -> FA rate = mean(neg_min < eps); pick the largest eps with FA <= budget
    eps = float(np.quantile(neg_min, fa_budget)) if len(neg_min) else 0.0
    leads, fa = [], []
    for r in rows_test:
        k = first_cross(r[key], eps)
        if r["onset"] < 0:
            fa.append(k >= 0)
        else:
            on_s = r["onset"] // r["stride"]
            leads.append((on_s - k) * r["stride"] * dt if (0 <= k) else -np.inf)
    leads = np.array(leads)
    fin = leads[np.isfinite(leads)]
    return dict(key=key, eps=eps, fa_budget=fa_budget, test_fa=float(np.mean(fa)) if fa else float("nan"),
                n_spill=int(len(leads)), n_neg=int(len(fa)),
                det_before_onset=float(np.mean(leads > 0)), det_lead_ge_0p1s=float(np.mean(leads >= 0.1)),
                det_lead_ge_0p25s=float(np.mean(leads >= 0.25)),
                lead_median_s=float(np.median(fin)) if len(fin) else float("nan"),
                lead_p10_s=float(np.percentile(fin, 10)) if len(fin) else float("nan"),
                lead_p90_s=float(np.percentile(fin, 90)) if len(fin) else float("nan"),
                missed=float(np.mean(~np.isfinite(leads))))


def cmd_gate(a):
    import multiprocessing as mp
    from world_model.v2.latent_hj import spill_data as sd
    sp_ = sd.shard_split(sd.list_shards())
    val = sp_["val"][: a.max_shards]; test = sp_["test"][: a.max_shards]
    jobs = [(p, a.ckpt, a.stride) for p in val + test]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=3) as pool:
        res = pool.map(_ep_scores, jobs, chunksize=1)
    rows_val = [r for rs, (p, *_ ) in zip(res, jobs) if p in val for r in rs]
    rows_test = [r for rs, (p, *_ ) in zip(res, jobs) if p in test for r in rs]
    print(f"[gate] val eps {len(rows_val)} test eps {len(rows_test)} ({time.time()-t0:.0f}s)", flush=True)
    out = dict(ckpt=a.ckpt, n_val=len(rows_val), n_test=len(rows_test))
    for subset, filt in (("all", lambda r: True), ("carry_tilt_le_60deg", lambda r: r["tilt_max"] <= 60)):
        rv_, rt_ = [r for r in rows_val if filt(r)], [r for r in rows_test if filt(r)]
        for fa in (0.02, 0.05, 0.10):
            for key in ("V", "l"):
                g = gate_eval(rv_, rt_, key, fa)
                g["subset"] = subset
                out[f"{subset}/{key}/fa{fa}"] = g
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    np.savez_compressed(Path(a.out).with_suffix(".npz"),
                        test_onset=np.array([r["onset"] for r in rows_test]),
                        test_Vmin=np.array([r["V"].min() for r in rows_test]),
                        test_lmin=np.array([r["l"].min() for r in rows_test]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", default=str(SCR / "v0"))
    t.add_argument("--iters", type=int, default=30000)
    t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--gamma0", type=float, default=0.9)
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--K", type=int, default=4)
    t.add_argument("--lam", type=float, default=0.5)
    t.add_argument("--h", type=float, default=0.04)
    t.add_argument("--A-max", type=float, default=5.0)
    t.add_argument("--W-max", type=float, default=2.0)
    t.add_argument("--m-lo", type=float, default=-1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=1000)
    t.add_argument("--grid", default=str(SCR / "grid_g099.npz"))
    c = sub.add_parser("compare")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--grid", nargs="+", required=True)
    c.add_argument("--n", type=int, default=50000)
    c.add_argument("--out", default=str(SCR / "stage0_compare.json"))
    g = sub.add_parser("gate")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--stride", type=int, default=2)
    g.add_argument("--max-shards", type=int, default=1000)
    g.add_argument("--out", default=str(SCR / "stage0_gate.json"))
    a = ap.parse_args()
    dict(train=cmd_train, compare=cmd_compare, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    main()
