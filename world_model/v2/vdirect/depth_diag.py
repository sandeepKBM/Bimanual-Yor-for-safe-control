"""vdirect/depth_diag.py -- EXPLORATORY (not pre-registered) diagnostic of addendum F's sampling prior.
SIMULATION / SAVED DATA ONLY (no hardware, no network). Reads the ilab grid read-only; writes only under $VD_DIAG_OUT.

Question: where are the nominal value's errors? F probes only the thinnest shell of the accepted set (smallest
positive V). Here, for F's exact round-0 inputs (NOM m0 WM + round-0 probe value, seed s, F's 60k candidate set):
  - accepted states (V > 0) are split into V-depth strata (quantiles of V among the accepted: 0-1%, 1-10%, then
    deciles), plus the rejected states (V <= 0);
  - `per` states per stratum get F's exact probe (V's greedy fallback on the TRUE pendulum, 100 ticks;
    fail = true margin < 0 within T_conf = 50 ticks);
  - each probed state is labelled with the Stage-0 physics value (savable iff phys_V > 0).
Equivalence check: F_b1000's round-0 picks are re-derived and compared with the saved picks_r0.npy (+ their hit rate
vs rounds.json), so the value / candidates / probe are shown to be F's own.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("VD_SRC_ROOT", "/common/users/ss5772/bimanual_data/vdirect_ilab"))
FDIR = Path(os.environ.get("VD_F_ROOT", "/common/users/ss5772/bimanual_data/vdirect_f"))
OUT = Path(os.environ.get("VD_DIAG_OUT", "/common/users/ss5772/bimanual_data/vdirect_depth_diag"))
EDGES = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cp(k, n, a=0.05):
    """two-sided Clopper-Pearson interval."""
    from scipy.stats import beta
    if n == 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def probe(BRS, D, picks, scorer, dev, seed, T_conf=50):
    BR, _ = BRS.rollout_branches(D, picks, scorer, dev, seed, p_greedy=1.0)
    lmin = BR["l"][:, BRS.PRE + 1:BRS.PRE + 1 + T_conf].min(1).values.cpu().numpy()
    return lmin


def phys_values(C, vnet, vdyn, D, picks, dev):
    import torch
    s0 = D["start"][picks[:, 0]] + picks[:, 1]
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    c = T(D["obs"][s0, :3]); c = c / c.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        return C.phys_V(vnet, vdyn, T(D["n"][s0]), T(D["nd"][s0]), c).cpu().numpy()


def run_seed(s, per, dev, norm, tr, vnet, vdyn):
    import torch
    from world_model.v2.vdirect import branches as BRS
    from world_model.v2.vdirect.run_arm import CFG, cand_seed, split_budget
    from world_model.v2.vdirect.trainer import Trainer
    from world_model.v2.latent_hj import reach_value as rv
    from world_model.v2.wm_audit import common as C
    t0 = time.time()
    wm = Trainer(s, norm, total_steps=CFG["S_TOTAL"], dev=dev).load_ckpt(SRC / "runs" / f"nom_s{s}_m0" / "ckpt.pt") \
        .frozen_wm()
    net, _ = rv.load(str(SRC / "runs" / f"nom_s{s}_m0" / "probe" / "value.pt"), dev)
    for p in net.parameters():
        p.requires_grad_(False)
    scorer = BRS.ValueScorer(wm, net, dev)
    seed = cand_seed(s, 0)
    cand = BRS.candidates(tr, CFG["N_CAND"], seed)
    V = scorer.state_values(BRS.latents_at(wm, tr, cand, dev))[0].cpu().numpy()
    Vp_all = phys_values(C, vnet, vdyn, tr, cand, dev)
    acc = V > 0
    res = dict(seed=s, n_cand=int(len(cand)), frac_accepted=float(acc.mean()),
               frac_doomed_all=float((Vp_all <= 0).mean()),
               frac_doomed_accepted=float((Vp_all[acc] <= 0).mean()))

    # ---- equivalence: re-derive F_b1000 round-0 picks exactly as falsify_directed does
    n_probe = int(round(0.5 * split_budget(1000, CFG["ROUNDS"])[0]))
    rng = np.random.default_rng(seed)
    picksF, _ = BRS.select_states(cand, [np.where(acc, -V, -np.inf)], [n_probe], CFG["POOL_MULT"], rng)
    fdir = FDIR / "runs" / f"F_b1000_s{s}"
    saved = np.load(fdir / "picks_r0.npy")[:n_probe]
    rs = json.loads((fdir / "rounds.json").read_text())
    rs = rs.get("rounds", rs) if isinstance(rs, dict) else rs
    same = set(map(tuple, picksF.tolist())) & set(map(tuple, saved.tolist()))
    # uniform draws from a pool whose ORDER differs by float noise differ even when the pool SET is the same,
    # so the decisive check is pool membership of F's saved picks
    sv = np.where(acc, -V, -np.inf)
    pool = set(map(tuple, cand[np.argsort(-sv, kind="stable")[:CFG["POOL_MULT"] * n_probe]].tolist()))
    lminF = probe(BRS, tr, picksF, scorer, dev, seed + 7)
    lminS = probe(BRS, tr, saved, scorer, dev, seed + 7)
    res["equiv"] = dict(n_probe=n_probe, picks_identical=bool(np.array_equal(picksF, saved)),
                        pick_overlap=len(same) / n_probe,
                        saved_picks_in_rederived_pool=float(np.mean([tuple(p) in pool for p in saved.tolist()])),
                        hit_rate_saved_picks_here=float((lminS < 0).mean()),
                        hit_rate_rederived=float((lminF < 0).mean()),
                        hit_rate_logged=float(rs[0]["hit_rate"]), frac_acc_logged=float(rs[0]["cand_frac_accepted"]))
    log(f"[s{s}] equiv {res['equiv']}  frac_acc={res['frac_accepted']:.4f}")

    # ---- stratified probes
    rng = np.random.default_rng(10_000 + s)
    ia = np.flatnonzero(acc)
    order = ia[np.argsort(V[ia], kind="stable")]           # accepted, smallest positive V first
    strata = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        strata.append((f"acc q{100*lo:g}-{100*hi:g}%", order[int(lo * len(order)):int(hi * len(order))]))
    strata.append(("rejected (V<=0)", np.flatnonzero(~acc)))
    idx, lab = [], []
    for k, (_, members) in enumerate(strata):
        j = rng.choice(members, min(per, len(members)), replace=False)
        idx.append(j); lab.append(np.full(len(j), k))
    idx, lab = np.concatenate(idx), np.concatenate(lab)
    lmin = probe(BRS, tr, cand[idx], scorer, dev, seed + 7)
    fail = lmin < 0
    Vp = Vp_all[idx]
    n_err_tot = 0.0
    rows = []
    for k, (name, members) in enumerate(strata):
        m = lab == k
        f, sav = fail[m], Vp[m] > 0
        row = dict(stratum=name, n_pop=int(len(members)),
                   V_range=[float(V[members].min()), float(V[members].max())] if len(members) else None,
                   n=int(m.sum()), fail=float(f.mean()) if m.any() else float("nan"), fail_ci=cp(int(f.sum()), int(m.sum())),
                   frac_doomed_pop=float((Vp_all[members] <= 0).mean()) if len(members) else float("nan"),
                   fail_given_savable=float(f[sav].mean()) if sav.any() else float("nan"), n_savable=int(sav.sum()),
                   fail_given_doomed=float(f[~sav].mean()) if (~sav).any() else float("nan"),
                   n_doomed=int((~sav).sum()),
                   share_fail_on_savable=float((f & sav).sum() / max(f.sum(), 1)))
        if k < len(strata) - 1:                             # accepted strata only: estimated #errors in the stratum
            row["est_errors"] = row["n_pop"] * row["fail"]
            n_err_tot += row["est_errors"]
        rows.append(row)
    for row in rows[:-1]:
        row["share_of_accepted_errors"] = row["est_errors"] / max(n_err_tot, 1e-9)
    res["strata"] = rows
    res["F_round0_pool_frac_of_accepted"] = {str(b): 4 * int(round(0.5 * split_budget(b, CFG["ROUNDS"])[0]))
                                             / int(acc.sum()) for b in (300, 1000, 3000, 10000)}
    res["wall_sec"] = time.time() - t0
    np.savez_compressed(OUT / f"probes_s{s}.npz", cand=cand[idx], stratum=lab, V=V[idx], Vp=Vp, lmin=lmin)
    log(f"[s{s}] done in {res['wall_sec']:.0f}s")
    return res


def table(all_res):
    L = ["| stratum (of accepted, by V) | V range (s0) | pop (s0) | fail rate s0 / s1 / s2 [95% CI pooled] "
         "| doomed (pop) | fail given savable | share of accepted errors |", "|---|---|---|---|---|---|---|"]
    for k in range(len(all_res[0]["strata"])):
        r = [a["strata"][k] for a in all_res]
        kf = sum(round(x["fail"] * x["n"]) for x in r); n = sum(x["n"] for x in r)
        ci = cp(kf, n)
        vr = r[0]["V_range"]
        share = " / ".join(f"{100*x['share_of_accepted_errors']:.1f}%" for x in r) if "share_of_accepted_errors" in r[0] \
            else "n/a"
        L.append(f"| {r[0]['stratum']} | {vr[0]:.3f}..{vr[1]:.3f} | {r[0]['n_pop']} | "
                 + " / ".join(f"{x['fail']:.2f}" for x in r) + f" [{ci[0]:.2f}, {ci[1]:.2f}] | "
                 + " / ".join(f"{100*x['frac_doomed_pop']:.0f}%" for x in r) + " | "
                 + " / ".join(f"{x['fail_given_savable']:.2f}" for x in r) + f" | {share} |")
    return "\n".join(L)


# ============================================================================ --final mode (final deployed values)
# Same probe, on the held-out VAL split (disjoint from every arm's training data and branch roots), one candidate set
# per seed shared by all models:
#   own view:    each model's accepted set stratified by its OWN V (where its remaining errors sit vs its boundary);
#   common view: one probe set stratified by the NOMINAL V; every model rolls its own greedy fallback from the same
#                states; false-safe = model V > 0 AND its fallback fails (which part of the original band was fixed).
def load_nom(s, dev, norm):
    from world_model.v2.vdirect.run_arm import CFG
    from world_model.v2.vdirect.trainer import Trainer
    from world_model.v2.latent_hj import reach_value as rv
    wm = Trainer(s, norm, total_steps=CFG["S_TOTAL"], dev=dev).load_ckpt(SRC / "runs" / f"nom_s{s}_m0" / "ckpt.pt") \
        .frozen_wm()
    net, _ = rv.load(str(SRC / "runs" / f"nom_s{s}_m0" / "probe" / "value.pt"), dev)
    for p in net.parameters():
        p.requires_grad_(False)
    return wm, net, dict(run=f"nom_s{s}_m0 (+ round-0 probe value)", cl_spill_savable=None)


def load_final(spec, s, dev):
    from world_model.v2.latent_hj import kin_wm
    from world_model.v2.latent_hj import reach_value as rv
    runs = [r / "runs" / f"{spec}_s{s}" for r in (FDIR, SRC)]
    run = next((r for r in runs if (r / "_DONE").exists()), None)
    if run is None:
        raise FileNotFoundError(f"no finished run among {runs}")
    wm = kin_wm.load_wm(str(run / "wm.pt"), dev)
    net, _ = rv.load(str(run / "value_std" / "value.pt"), dev)
    for p in net.parameters():
        p.requires_grad_(False)
    cl = json.loads((run / "result.json").read_text())["values"]["std"]["cl_spill_savable"]
    return wm, net, dict(run=str(run), cl_spill_savable=float(cl))


def own_strata(V):
    acc = V > 0
    ia = np.flatnonzero(acc)
    order = ia[np.argsort(V[ia], kind="stable")]
    st = [(f"acc q{100*lo:g}-{100*hi:g}%", order[int(lo * len(order)):int(hi * len(order))])
          for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    return st + [("rejected (V<=0)", np.flatnonzero(~acc))]


def sample_strata(strata, per, rng):
    idx, lab = [], []
    for k, (_, members) in enumerate(strata):
        j = rng.choice(members, min(per, len(members)), replace=False)
        idx.append(j); lab.append(np.full(len(j), k))
    return np.concatenate(idx), np.concatenate(lab)


def run_seed_final(s, specs, per, dev, norm, va, vnet, vdyn):
    from world_model.v2.vdirect import branches as BRS
    from world_model.v2.vdirect.run_arm import CFG, cand_seed
    from world_model.v2.wm_audit import common as C
    t0 = time.time()
    seed = cand_seed(s, 0) + 500
    cand = BRS.candidates(va, CFG["N_CAND"], seed)
    Vp_all = phys_values(C, vnet, vdyn, va, cand, dev)
    N = len(cand)
    models, Vs = {}, {}
    for spec in ["nom"] + list(specs):
        wm, net, info = load_nom(s, dev, norm) if spec == "nom" else load_final(spec, s, dev)
        sc = BRS.ValueScorer(wm, net, dev)
        Vs[spec] = sc.state_values(BRS.latents_at(wm, va, cand, dev))[0].cpu().numpy()
        models[spec] = (sc, info)
    res = dict(seed=s, split="val", n_cand=int(N), frac_doomed_all=float((Vp_all <= 0).mean()), own={}, common={})
    store = dict(cand=cand, Vp=Vp_all)
    # ---- own view
    for spec, (sc, info) in models.items():
        V = Vs[spec]
        strata = own_strata(V)
        idx, lab = sample_strata(strata, per, np.random.default_rng(20_000 + s))
        fail = probe(BRS, va, cand[idx], sc, dev, seed + 7) < 0
        rows, tot = [], 0.0
        for k, (name, members) in enumerate(strata):
            m = lab == k
            f = fail[m]
            row = dict(stratum=name, n_pop=int(len(members)), n=int(m.sum()),
                       V_range=[float(V[members].min()), float(V[members].max())] if len(members) else None,
                       fail=float(f.mean()) if m.any() else float("nan"), fail_ci=cp(int(f.sum()), int(m.sum())),
                       frac_doomed_pop=float((Vp_all[members] <= 0).mean()) if len(members) else float("nan"))
            if not name.startswith("rejected"):
                row["est_errors"] = row["n_pop"] * (row["fail"] if m.any() else 0.0); tot += row["est_errors"]
            rows.append(row)
        for row in rows[:-1]:
            row["share_of_accepted_errors"] = row["est_errors"] / max(tot, 1e-9)
        res["own"][spec] = dict(info, frac_accepted=float((V > 0).mean()), est_false_safe_over_all=tot / N,
                                est_fail_over_accepted=tot / max(int((V > 0).sum()), 1), strata=rows)
        log(f"[s{s} own {spec}] acc={100*(V>0).mean():.1f}% false-safe(all)={100*tot/N:.2f}% cl={info['cl_spill_savable']}")
    # ---- common view: strata of the NOMINAL value, same states for every model
    strata = own_strata(Vs["nom"])
    idx, lab = sample_strata(strata, per, np.random.default_rng(30_000 + s))
    store.update(common_idx=idx, common_stratum=lab)
    for spec, (sc, info) in models.items():
        fail = probe(BRS, va, cand[idx], sc, dev, seed + 7) < 0
        fs = (Vs[spec][idx] > 0) & fail
        rows, tot = [], 0.0
        for k, (name, members) in enumerate(strata):
            m = lab == k
            rate = float(fs[m].mean()) if m.any() else 0.0
            tot += len(members) * rate
            rows.append(dict(stratum=name, n_pop=int(len(members)), n=int(m.sum()), false_safe=rate,
                             false_safe_ci=cp(int(fs[m].sum()), int(m.sum())),
                             fail_any=float(fail[m].mean()) if m.any() else float("nan"),
                             accepted=float((Vs[spec][idx][m] > 0).mean()) if m.any() else float("nan")))
        res["common"][spec] = dict(est_false_safe_over_all=tot / N, strata=rows)
        store[f"V_{spec}"] = Vs[spec][idx]; store[f"fail_{spec}"] = fail
    np.savez_compressed(OUT / f"final_val_s{s}.npz", **store)
    res["wall_sec"] = time.time() - t0
    log(f"[s{s}] final done in {res['wall_sec']:.0f}s (N_val_cand={N})")
    return res


def tables_final(all_res, specs):
    sp = ["nom"] + list(specs)
    L = ["### Own view: each value's accepted set stratified by its OWN V; cell = fail rate s0/s1/s2 "
         "(share of that value's errors)", "",
         "| stratum | " + " | ".join(sp) + " |", "|---|" + "---|" * len(sp)]
    for k in range(len(EDGES)):
        cells = []
        for spec in sp:
            r = [a["own"][spec]["strata"][k] for a in all_res]
            f = "/".join(f"{x['fail']:.2f}" for x in r)
            sh = "/".join(f"{100*x['share_of_accepted_errors']:.0f}" for x in r) if "share_of_accepted_errors" in r[0] else ""
            cells.append(f"{f}" + (f" ({sh}%)" if sh else ""))
        L.append(f"| {all_res[0]['own'][sp[0]]['strata'][k]['stratum']} | " + " | ".join(cells) + " |")
    L.append("| **accepted %** | " + " | ".join("/".join(f"{100*a['own'][s_]['frac_accepted']:.1f}" for a in all_res)
                                          for s_ in sp) + " |")
    L.append("| **est. false-safe over all VAL cand %** | " + " | ".join(
        "/".join(f"{100*a['own'][s_]['est_false_safe_over_all']:.1f}" for a in all_res) for s_ in sp) + " |")
    L.append("| closed-loop spill (savable TEST) % | " + " | ".join(
        "/".join("-" if a["own"][s_]["cl_spill_savable"] is None else f"{100*a['own'][s_]['cl_spill_savable']:.1f}"
                 for a in all_res) for s_ in sp) + " |")
    L += ["", "### Common view: same states, stratified by the NOMINAL V; cell = false-safe rate s0/s1/s2 "
          "(model V > 0 AND its own fallback fails)", "",
          "| nominal stratum | " + " | ".join(sp) + " |", "|---|" + "---|" * len(sp)]
    for k in range(len(EDGES)):
        L.append(f"| {all_res[0]['common']['nom']['strata'][k]['stratum']} | " + " | ".join(
            "/".join(f"{a['common'][s_]['strata'][k]['false_safe']:.2f}" for a in all_res) for s_ in sp) + " |")
    L.append("| **est. false-safe over all %** | " + " | ".join(
        "/".join(f"{100*a['common'][s_]['est_false_safe_over_all']:.1f}" for a in all_res) for s_ in sp) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per", type=int, default=200)
    ap.add_argument("--final", nargs="*", default=None,
                    help="final-value mode on VAL, e.g. --final F_b1000 R_b1000 Rr_b1000 (nominal always included)")
    a = ap.parse_args()
    if a.final is not None:
        import torch
        from world_model.v2.vdirect.trainer import train_data
        from world_model.v2.wm_audit import common as C
        OUT.mkdir(parents=True, exist_ok=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"[depth_diag final] specs={a.final} seeds={a.seeds} per={a.per} dev={dev} "
            f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        norm = json.loads((SRC / "shared" / "norm.json").read_text())
        va = train_data(dev)[1]
        vnet, vdyn, _ = C.load_phys_value(device=dev)
        all_res = [run_seed_final(s, a.final, a.per, dev, norm, va, vnet, vdyn) for s in a.seeds]
        tag = "_".join(a.final)
        (OUT / f"final_val_{tag}_per{a.per}.json").write_text(json.dumps(all_res, indent=1))
        md = tables_final(all_res, a.final)
        (OUT / f"final_val_{tag}_per{a.per}.md").write_text(md + "\n")
        print(md, flush=True)
        return
    import torch
    from world_model.v2.vdirect.trainer import train_data
    from world_model.v2.wm_audit import common as C
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[depth_diag] seeds={a.seeds} per={a.per} dev={dev} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    norm = json.loads((SRC / "shared" / "norm.json").read_text())
    tr = train_data(dev)[0]
    vnet, vdyn, _ = C.load_phys_value(device=dev)
    all_res = [run_seed(s, a.per, dev, norm, tr, vnet, vdyn) for s in a.seeds]
    (OUT / f"depth_diag_per{a.per}.json").write_text(json.dumps(all_res, indent=1))
    md = table(all_res)
    (OUT / f"depth_diag_per{a.per}.md").write_text(md + "\n")
    print(md, flush=True)


if __name__ == "__main__":
    main()
