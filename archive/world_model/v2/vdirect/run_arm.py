"""vdirect/run_arm.py -- one cell of the value-directed-branch kill test. SIMULATION / SAVED DATA ONLY
(no hardware, no network). Every output goes under $VD_ROOT (default /data/local/ss5772/bimanual_runs/vdirect).

  --arm SETUP          shared reference normalization + closed-loop state pick + phys/hold references
  --arm NOM  --seed s --member m   nominal-only WM pre-train (S_NOM steps); member 0 also trains the round-0 probe value
  --arm R    --budget b --seed s   random branches (train_kinwm.make_branches), all at step 0, S_TOTAL steps
  --arm V    --budget b --seed s   value-directed, ROUNDS DAgger rounds (from NOM m0)
  --arm Vn   --budget b --seed s   value-directed, ONE shot from the round-0 model (no iteration)
  --arm U    --budget b --seed s   ensemble-disagreement-directed (M_ENS WMs: NOM m0..m2), ROUNDS rounds, final = member 0
  --arm D    --budget b --seed s   DAgger on-policy fallback rollouts: uniform branch points, pure greedy actions, ROUNDS rounds
  --arm B0   --seed s              no branches: nominal WM to S_TOTAL, then values N0 (full max), M (in-data max),
                                   P (uncertainty-pessimistic)
  (PREREG_ADDENDUM_F; same rounds / probes / budget split as V)
  --arm F    --budget b --seed s   confirming falsifier: half of each round's budget = greedy-fallback probes on the
                                   TRUE pendulum from accepted near-boundary states (kept as data), half = random
                                   fallback-set branches from the CONFIRMED counterexample states
  --arm F0   --budget b --seed s   falsify only: the whole round budget on probes (literal predator-prey)
  --arm Rr   --budget b --seed s   random branches (make_branches) under the SAME NOM warm start + 3-round schedule
VD_SRC_ROOT (default VD_ROOT): where shared/ and the NOM pre-trains are read from (the F grid reuses the ilab grid's).
Every final model: wm.pt, eval.json (train_kinwm.evaluate: counterfactual Spearman etc.), value (VAL_IT iters, fresh),
gate.json (value_gate.gate), cl.json + cl_states.npz (closed loop + winner's curse), result.json, _DONE.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("VD_ROOT", "/data/local/ss5772/bimanual_runs/vdirect"))
SRC = Path(os.environ.get("VD_SRC_ROOT", str(ROOT)))
CFG = dict(S_NOM=3000, S_ROUND=4000, ROUNDS=3, S_TOTAL=15000, VAL_IT=10000, PROBE_IT=4000, N_CAND=60000,
           POOL_MULT=4, P_GREEDY=0.6, H_U=25, M_ENS=3, EVAL_EPS=300, HEAD_E=5, HEAD_STEPS=3000, VAL_BATCH=2048)
SMOKE = dict(S_NOM=30, S_ROUND=20, ROUNDS=3, S_TOTAL=90, VAL_IT=6, PROBE_IT=4, N_CAND=2000, EVAL_EPS=20, VAL_BATCH=64,
             HEAD_STEPS=30, CL_MAX=100)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def shared():
    return SRC / "shared"


def get_norm(dev):
    from world_model.v2.vdirect.trainer import compute_norm, dump_json
    p = shared() / "norm.json"
    if not p.exists():
        dump_json(compute_norm(dev), p)
    return json.loads(p.read_text())


def train_value(wm, dev, out, iters, seed, kind="std", U_idx=None, heads=None, batch=2048):
    import torch
    from world_model.v2.latent_hj import reach_value as rv
    from world_model.v2.wm_audit import common as C
    from world_model.v2.wm_audit import value_gate as VG
    from world_model.v2.vdirect.pess import PessDyn, SubsetDyn
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bank = VG.build_bank(wm, dev, project=True, seed=seed)
    base = VG.ProjWMDyn(wm, bank, dev, project=True)
    dyn = base if kind == "std" else (SubsetDyn(base, U_idx) if kind == "M" else PessDyn(base, heads, bank))
    net, curve = rv.train(dyn, out, iters=iters, batch=batch, gamma0=0.8, gamma_final=0.98, K=4, lam=0.5, device=dev,
                          log_every=max(iters // 5, 1), seed=seed)
    meta = dict(dx=int(bank.shape[1]), hidden=256, layers=3, gamma=0.98, K=4, kind=f"vdirect_{kind}",
                P=C.P0.as_dict() | dict(dt=C.DT_WM), project=True, iters=iters, seed=seed,
                U_idx=None if U_idx is None else [int(i) for i in U_idx])
    rv.save(net.cpu(), out / "value.pt", meta)
    (out / "curve.json").write_text(json.dumps(curve))
    net = net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    log(f"[value {kind}] {iters} iters in {time.time()-t0:.0f}s")
    return net


def finalize(wm, dev, out, seed, cfg, values=(("std", None),), rounds_meta=None, t_start=None):
    """eval.json, then per value kind: value, gate.json, closed loop -> result.json."""
    import torch
    from world_model.v2.wm_audit import train_kinwm as TK
    from world_model.v2.wm_audit import value_gate as VG
    from world_model.v2.vdirect import cl_eval as CE
    from world_model.v2.vdirect import pess as PS
    out = Path(out)
    t0 = time.time()
    ev = TK.evaluate(wm, dev, n_eps=cfg["EVAL_EPS"])
    (out / "eval.json").write_text(json.dumps(ev, indent=1))
    log(f"[eval] {time.time()-t0:.0f}s")
    result = dict(eval_cf=ev["counterfactual"], eval_filtered=ev["filtered_margin"], rounds=rounds_meta, values={})
    for kind, extra in values:
        vdir = out / f"value_{kind}"
        U_idx, heads, hinfo = None, None, None
        if kind == "M":
            U_idx, frac = PS.in_data_actions()
            hinfo = dict(U_idx=U_idx.tolist(), occupancy=frac[U_idx].round(5).tolist())
        if kind == "P":
            t1 = time.time()
            heads, hinfo = PS.fit_heads(wm, dev, E=cfg["HEAD_E"], steps=cfg["HEAD_STEPS"], seed=seed)
            log(f"[heads] tau={hinfo['tau']:.4f} frac_U>tau={hinfo['frac_U_sigma_gt_tau']:.3f} ({time.time()-t1:.0f}s)")
        net = train_value(wm, dev, vdir, cfg["VAL_IT"], seed, kind=kind, U_idx=U_idx, heads=heads,
                          batch=cfg["VAL_BATCH"])
        t1 = time.time()
        g = VG.gate(wm, net, dev, vdir)
        log(f"[gate {kind}] {time.time()-t1:.0f}s")
        t1 = time.time()
        pess = None
        if kind == "P":
            pess = PS.PessDyn(VG.ProjWMDyn(wm, None, dev, project=True), heads)
        pol = CE.Policy(wm, net, dev, U_idx=U_idx, pess=pess)
        cl, per = CE.evaluate(wm, pol, dev, shared() / "cl_states.npz", shared() / "cl_ref.npz",
                              max_states=cfg.get("CL_MAX"))
        np.savez_compressed(vdir / "cl_states.npz", **per)
        (vdir / "cl.json").write_text(json.dumps(cl, indent=1))
        log(f"[closed-loop {kind}] wm_greedy spill(savable)={cl['wm_greedy']['spill_rate_phys_savable']:.4f} "
            f"({time.time()-t1:.0f}s)")
        s = g["states"]
        result["values"][kind] = dict(
            cl_spill_savable=cl["wm_greedy"]["spill_rate_phys_savable"], cl=cl,
            doomed_recall=s["doomed_recall_Vwm"], doomed_recall_at_5pct_fu=s["doomed_recall_Vwm_at_5pct_false_unsafe"],
            false_safe=s["false_safe_rate_Vwm"], iou=s["iou_safe_Vwm"], auroc_doomed=s["auroc_doomed_given_lpos_Vwm"],
            grad_cos_K10_median=g["grad_cos_K10"]["Vwm_vs_Vphys"]["median"],
            grad_cos_K10_frac_gt_0p7=g["grad_cos_K10"]["Vwm_vs_Vphys"]["frac_gt_0p7"], heads=hinfo)
    result["spearman_H25"] = ev["counterfactual"]["H25"]["spearman_across_U_median"]
    result["spearman_H50"] = ev["counterfactual"]["H50"]["spearman_across_U_median"]
    result["wall_sec"] = None if t_start is None else time.time() - t_start
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def split_budget(b, R):
    q = b // R
    return [q] * (R - 1) + [b - q * (R - 1)]


def cand_seed(seed, r):
    return 100000 + 1000 * seed + r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["SETUP", "NOM", "R", "V", "Vn", "U", "D", "B0", "F", "F0", "Rr"])
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    from world_model.v2.vdirect import branches as BRS
    from world_model.v2.vdirect.trainer import Trainer, train_data, dump_json
    from world_model.v2.vdirect import cl_eval as CE
    cfg = dict(CFG) | (SMOKE if a.smoke else {})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    log(f"[run] arm={a.arm} b={a.budget} s={a.seed} m={a.member} smoke={a.smoke} dev={dev} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} cfg={cfg}")
    if a.arm == "SETUP":
        get_norm(dev)
        CE.cl_states(dev, shared() / "cl_states.npz")
        ref = CE.reference_outcomes(dev, shared() / "cl_states.npz", shared() / "cl_ref.npz")
        st = CE.cl_states(dev, shared() / "cl_states.npz")
        sav = st["Vp0"] > 0
        log(f"[setup] phys_greedy spill(savable)={(ref['phys_greedy'][sav] < 0).mean():.4f} "
            f"hold={(ref['hold'][sav] < 0).mean():.4f} n_savable={sav.sum()}")
        (shared() / "_DONE_SETUP").write_text(time.strftime("%FT%TZ", time.gmtime()))
        return
    norm = get_norm(dev)
    tr = train_data(dev)[0]
    sm = "smoke_" if a.smoke else ""
    nomdir = lambda s, m: (SRC / "runs" / f"{sm}nom_s{s}_m{m}")
    if a.arm == "NOM":
        out = nomdir(a.seed, a.member); out.mkdir(parents=True, exist_ok=True)
        T = Trainer(a.seed + 1000 * a.member, norm, total_steps=cfg["S_TOTAL"], dev=dev)
        T.train(cfg["S_NOM"], tag=f" nom s{a.seed} m{a.member}")
        T.save_ckpt(out / "ckpt.pt")
        if a.member == 0:
            train_value(T.frozen_wm(), dev, out / "probe", cfg["PROBE_IT"], seed=a.seed, batch=cfg["VAL_BATCH"])
        (out / "_DONE").write_text(time.strftime("%FT%TZ", time.gmtime()))
        return
    out = Path(a.out) if a.out else ROOT / "runs" / f"{sm}{a.arm}_b{a.budget}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rounds = []

    def load_probe0():
        from world_model.v2.latent_hj import reach_value as rv
        net, _ = rv.load(str(nomdir(a.seed, 0) / "probe" / "value.pt"), dev)
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    def warm(member=0):
        T = Trainer(a.seed + 1000 * member, norm, total_steps=cfg["S_TOTAL"], dev=dev)
        return T.load_ckpt(nomdir(a.seed, member) / "ckpt.pt")

    if a.arm == "R":
        T = Trainer(a.seed, norm, total_steps=cfg["S_TOTAL"], dev=dev)
        t0 = time.time()
        BR, meta = BRS.random_branches(tr, a.budget, dev, seed=a.seed + 1)
        meta.update(round=0, gen_sec=time.time() - t0, wm_step=0); rounds.append(meta)
        log(f"[branches R] {meta}")
        T.add_branches(BR)
        T.train(cfg["S_TOTAL"], tag=f" R b{a.budget} s{a.seed}")
        wm = T.frozen_wm()
    elif a.arm in ("V", "Vn", "D", "F", "F0", "Rr"):
        T = warm(0)
        R = 1 if a.arm == "Vn" else cfg["ROUNDS"]
        nb = split_budget(a.budget, R)
        steps = [cfg["S_TOTAL"] - cfg["S_NOM"]] if R == 1 else [cfg["S_ROUND"]] * R
        for r in range(R):
            wm_r = T.frozen_wm()
            if a.arm == "Rr":                      # random branches need no probe value
                net = None
            else:
                net = load_probe0() if r == 0 else train_value(wm_r, dev, out / f"probe_r{r}", cfg["PROBE_IT"],
                                                                seed=a.seed * 10 + r, batch=cfg["VAL_BATCH"])
            t0 = time.time()
            if a.arm == "Rr":
                BR, meta = BRS.random_branches(tr, nb[r], dev, seed=cand_seed(a.seed, r) + 1)
                picks = np.zeros((0, 2), np.int64)
                meta.update(kind="Rr")
            elif a.arm in ("F", "F0"):
                BR, meta, picks = BRS.falsify_directed(tr, nb[r], wm_r, net, dev, cand_seed(a.seed, r),
                                                       n_cand=cfg["N_CAND"], pool_mult=cfg["POOL_MULT"],
                                                       branch=(a.arm == "F"))
            elif a.arm == "D":
                rng = np.random.default_rng(cand_seed(a.seed, r))
                cand = BRS.candidates(tr, cfg["N_CAND"], cand_seed(a.seed, r))
                picks = cand[rng.choice(len(cand), nb[r], replace=False)]
                BR, meta = BRS.rollout_branches(tr, picks, BRS.ValueScorer(wm_r, net, dev), dev,
                                                cand_seed(a.seed, r) + 7, p_greedy=1.0)
                meta.update(kind="D", pick_l_mean=float(tr["l"][tr["start"][picks[:, 0]] + picks[:, 1]].mean()))
            else:
                BR, meta, picks = BRS.value_directed(tr, nb[r], wm_r, net, dev, cand_seed(a.seed, r),
                                                     n_cand=cfg["N_CAND"], pool_mult=cfg["POOL_MULT"],
                                                     p_greedy=cfg["P_GREEDY"])
            meta.update(round=r, gen_sec=time.time() - t0, wm_step=T.step); rounds.append(meta)
            np.save(out / f"picks_r{r}.npy", picks)
            log(f"[branches {a.arm} r{r}] {meta}")
            T.add_branches(BR)
            T.train(steps[r], tag=f" {a.arm} b{a.budget} s{a.seed} r{r}")
        wm = T.frozen_wm()
    elif a.arm == "U":
        Ts = [warm(m) for m in range(cfg["M_ENS"])]
        nb = split_budget(a.budget, cfg["ROUNDS"])
        for r in range(cfg["ROUNDS"]):
            wms = [T.frozen_wm() for T in Ts]
            t0 = time.time()
            BR, meta, picks = BRS.uncertainty_directed(tr, nb[r], wms, dev, cand_seed(a.seed, r), n_cand=cfg["N_CAND"],
                                                       pool_mult=cfg["POOL_MULT"], p_greedy=cfg["P_GREEDY"],
                                                       H=cfg["H_U"])
            meta.update(round=r, gen_sec=time.time() - t0, wm_step=Ts[0].step); rounds.append(meta)
            np.save(out / f"picks_r{r}.npy", picks)
            log(f"[branches U r{r}] {meta}")
            for m, T in enumerate(Ts):
                T.add_branches(BR)
                T.train(cfg["S_ROUND"], tag=f" U b{a.budget} s{a.seed} r{r} m{m}")
            del wms
        T = Ts[0]
        wm = T.frozen_wm()
    elif a.arm == "B0":
        T = warm(0)
        T.train(cfg["S_TOTAL"] - cfg["S_NOM"], tag=f" B0 s{a.seed}")
        wm = T.frozen_wm()
    T.save_wm(out / "wm.pt")
    dump_json(T.curve, out / "wm_curve.json")
    dump_json(rounds, out / "rounds.json")
    vals = (("std", None), ("M", None), ("P", None)) if a.arm == "B0" else (("std", None),)
    res = finalize(wm, dev, out, a.seed, cfg, values=vals, rounds_meta=rounds, t_start=t_start)
    log("[result] " + json.dumps({k: v["cl_spill_savable"] for k, v in res["values"].items()}))
    (out / "_DONE").write_text(time.strftime("%FT%TZ", time.gmtime()))


if __name__ == "__main__":
    main()
