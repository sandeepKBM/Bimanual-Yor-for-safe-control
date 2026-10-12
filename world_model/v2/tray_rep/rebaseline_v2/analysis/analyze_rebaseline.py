"""rebaseline_v2/analysis/analyze_rebaseline.py -- observations only (no verdicts). SAVED DATA ONLY. CPU.

Re-analysis of Experiments A, B, C-F5 (open loop) and C-F5 closed loop on the FIXED pin_cup physics (v2 data, v2 mu_eff map, eps_a = 0),
next to the old-physics numbers (read from the old summary JSONs; old analysis scripts are frozen and only imported for their helpers).
Bootstrap: sets.expA_analyze.boot_diff (hierarchical paired, carries x seeds, B = 10000); time ratio: boed.fxi_analyze.boot_ratio.

  python -m world_model.v2.tray_rep.rebaseline_v2.analysis.analyze_rebaseline
Outputs: <RB>/analysis/RESULT_REBASELINE.md, <RB>/analysis/summary_rebaseline.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.boed import fxi_analyze as FX
from world_model.v2.tray_rep.boed import report as RP
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.bridge import metrics as M
from world_model.v2.tray_rep.sets import expA_analyze as A

DATA = Path("/common/users/ss5772/bimanual_data/tray_rep")
RB = DATA / "rebaseline_v2"
EXPA, EXPB, CF5, CF5CL = RB / "sets_expA", RB / "vlmprior_expB", RB / "boed_f5", RB / "boed_f5_cl"
OLD_A = json.load(open(DATA / "sets/expA/analysis/summary_A.json"))
OLD_B = json.load(open(DATA / "vlmprior/expB/analysis/summary_B.json"))
OLD_C = json.load(open(DATA / "boed_f5/analysis/summary_CF5.json"))
N, SEEDS3 = 600, (0, 1, 2)
rng = np.random.default_rng(20261011)
RES = dict(A={}, B={}, C={})
L = []


def pp(x, d=1):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100*x:.{d}f}%"


def f3(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def aug(recs):
    for s in recs:
        for r in recs[s]:
            r["interv_c"] = r["n_interv"] / max(r["n_ticks"], 1)
            n = max(r.get("x_f5_n", 0), 1)
            r["backup_c"] = r.get("x_f5_backup", 0) / n
            r["infeas_c"] = r.get("x_f5_infeas", 0) / n
    return recs


def ci(t, scale=1.0, d=3, unit=""):
    p, lo, hi = t
    return f"{scale*p:+.{d}f}{unit} [{scale*lo:+.{d}f}, {scale*hi:+.{d}f}]"


# ======================================================================================= A
A_SPECS = {
    "piper_trk": {"F0": ("F0", (0,), N), "F1": ("F1:l_align=4,l_prox=0", SEEDS3, N), "F2": ("F2:theta_max=0.05,a_cap=0.4", (0,), N),
                  "F4": ("F4:check_every=5", (0,), 100), "F5 (v2 map, eps_a 0)": ("F5:calib_file=v2,eps_a=0", SEEDS3, N),
                  "F5 (v2 map, eps_a 0.08)": ("F5:calib_file=v2,eps_a=0.08", SEEDS3, N), "F5 raw map (eps_a 0)": ("F5:calib=0,eps_a=0", SEEDS3, N)},
    "piper": {"F0": ("F0", (0,), N), "F1": ("F1:l_align=4,l_prox=10", (0,), N), "F2": ("F2:theta_max=0.05,a_cap=0.4", (0,), N),
              "F4": ("F4:check_every=5", (0,), 100), "F5 (v2 map, eps_a 0)": ("F5:calib_file=v2,eps_a=0", (0,), N),
              "F5 (v2 map, eps_a 0.08)": ("F5:calib_file=v2,eps_a=0.08", (0,), N), "F5 raw map (eps_a 0)": ("F5:calib=0,eps_a=0", (0,), N)},
}
OLD_NAME = {"F0": "F0", "F1": "F1", "F2": "F2", "F4": "F4", "F5 (v2 map, eps_a 0)": None, "F5 (v2 map, eps_a 0.08)": "F5", "F5 raw map (eps_a 0)": "F5_raw"}
MAINF5, F5E8, F5RAW = "F5 (v2 map, eps_a 0)", "F5 (v2 map, eps_a 0.08)", "F5 raw map (eps_a 0)"

L += ["# Rebaseline on fixed pin_cup physics (observations; old-physics numbers next to new)", "",
      "Conventions: OLD = old physics, old data, old mu_eff map, F5 eps_a 0.08 (RESULT_A/B/CF5 summaries). NEW = fixed pin_cup, data_*_v2, "
      "calib_v2 mu_eff map, eps_a 0 unless stated. 600 TEST carries per run; seeds 0-2 where the arm has a noisy input or model seed. "
      "Bootstrap = hierarchical paired (carries x seeds), B = 10000, 95% percentile interval. Old F5 row = old map + eps_a 0.08, so its "
      "nearest NEW counterpart is 'F5 v2 map eps_a 0.08'.", ""]

for setup in ("piper_trk", "piper"):
    L += [f"## A: {setup} ({'tracker noise' if setup == 'piper_trk' else 'noise-free'})", "",
          "| arm | seeds | OLD drop | NEW drop | OLD interv | NEW interv | OLD dev | NEW dev | OLD progress | NEW progress | NEW goal | NEW ms/tick mean |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    data, rs = {}, {}
    for arm, (spec, seeds, n) in A_SPECS[setup].items():
        recs = A.load(EXPA, setup, spec, seeds, n)
        if recs is None:
            L.append(f"| {arm} | MISSING |")
            continue
        aug(recs)
        data[arm] = recs
        flat = [r for s in seeds for r in recs[s]]
        sm = M.summarize(flat)
        ms = [r["x_f5_ms_mean"] for r in flat if "x_f5_ms_mean" in r]
        rs[arm] = dict(spec=spec, n=n, seeds=list(seeds), drop=sm["drop"]["rate"], drop_k=sm["drop"]["k"], drop_n=sm["drop"]["n"], interv=sm["interv_rate"],
                       dev=sm["mean_dev_norm"], progress=sm["progress"], goal=sm["goal"]["rate"], n_tunnel=sm["n_tunnel"],
                       ms_mean=float(np.mean(ms)) if ms else None, mu_bins=A.mu_bins(recs),
                       backup=float(np.mean([r["backup_c"] for r in flat])) if ms else None, infeas=float(np.mean([r["infeas_c"] for r in flat])) if ms else None)
        o = OLD_A[setup]["arms"].get(OLD_NAME[arm]) if OLD_NAME[arm] else None
        rs[arm]["old"] = None if o is None else {k: o[k] for k in ("drop", "interv", "dev", "progress", "goal")}
        L.append(f"| {arm} | {list(seeds)} | {pp(o['drop']) if o else '-'} | {pp(sm['drop']['rate'])} ({sm['drop']['k']}/{sm['drop']['n']}) | "
                 f"{pp(o['interv']) if o else '-'} | {pp(sm['interv_rate'])} | {f3(o['dev']) if o else '-'} | {f3(sm['mean_dev_norm'])} | "
                 f"{f3(o['progress']) if o else '-'} | {f3(sm['progress'])} | {f3(sm['goal']['rate'])} | "
                 f"{f3(float(np.mean(ms)), 1) if ms else '-'} |")
    o = OLD_A[setup]["arms"]
    L.append(f"| (old only) F3 latent HJ Rc3k | | {pp(o['F3_Rc3k']['drop'])} | | {pp(o['F3_Rc3k']['interv'])} | | {f3(o['F3_Rc3k']['dev'])} | | {f3(o['F3_Rc3k']['progress'])} | | | |")
    L.append(f"| (old only) F5 raw map, eps_a 0.08 | | {pp(o['F5_raw']['drop'])} | | {pp(o['F5_raw']['interv'])} | | {f3(o['F5_raw']['dev'])} | | {f3(o['F5_raw']['progress'])} | | | |")
    L.append("")
    if MAINF5 in rs and rs[MAINF5]["backup"] is not None:
        L.append("F5 backup-law share of ticks / QP-infeasible share of ticks (NEW): " + "; ".join(
            f"{a}: {pp(rs[a]['backup'])} / {pp(rs[a]['infeas'])}" for a in (MAINF5, F5E8, F5RAW) if a in rs) + ". (OLD tracker F5: ~9% / ~8%, RESULT_A.)")
        L.append("")
    con = {}
    pairs = [(MAINF5, "F0", "dropped"), (MAINF5, "F2", "dropped"), (MAINF5, "F2", "progress"), (MAINF5, "F1", "dropped"),
             (F5E8, MAINF5, "dropped"), (F5E8, MAINF5, "progress"), (F5E8, MAINF5, "interv_c"), (F5E8, MAINF5, "dev_norm_sum"),
             (F5RAW, MAINF5, "dropped"), (F5RAW, MAINF5, "progress"), (F5RAW, MAINF5, "interv_c"), (F5RAW, F5E8, "dropped"), (F5RAW, F5E8, "progress")]
    for x, y, k in pairs:
        if x in data and y in data:
            con[f"{x} - {y} : {k}"] = A.boot_diff(A.mat(data[x], k), A.mat(data[y], k), rng)
    if MAINF5 in data and "F4" in data:
        f5 = {s: v[:100] for s, v in data[MAINF5].items()}
        con[f"{MAINF5} - F4 : dropped (first 100)"] = A.boot_diff(A.mat(f5, "dropped"), A.mat(data["F4"], "dropped"), rng)
    L += ["| contrast (NEW) | point | 95% CI | OLD counterpart |", "|---|---|---|---|"]
    for k, t in con.items():
        sc, u = (100, " pp") if k.endswith("dropped") or "dropped (" in k else (1, "")
        oldk = None
        if "raw map (eps_a 0) - F5 (v2 map, eps_a 0.08)" in k:
            oldk = "F5_raw-F5:" + k.split(":")[-1].strip()
        old = ci(OLD_A[setup]["contrasts"][oldk], sc, 3, u) if oldk in OLD_A[setup]["contrasts"] else "-"
        L.append(f"| {k} | {sc*t[0]:+.3f}{u} | [{sc*t[1]:+.3f}, {sc*t[2]:+.3f}] | {old} |")
    L += ["", "Drop rate by mu bin, NEW (n carries per bin; bins edges " + ", ".join(f"{x:.2g}" for x in A.MU_EDGES) + ") | OLD in brackets:", "",
          "| arm | bin 0 | bin 1 | bin 2 | bin 3 | bin 4 |", "|---|---|---|---|---|---|"]
    for arm, r in rs.items():
        ob = OLD_A[setup]["arms"].get(OLD_NAME[arm])["mu_bins"] if OLD_NAME[arm] else None
        L.append(f"| {arm} | " + " | ".join(f"{100*d:.1f}% ({n}) [{('%.1f%%' % (100*ob[k][0])) if ob else '-'}]" for k, (d, n) in enumerate(r["mu_bins"])) + " |")
    L.append("")
    RES["A"][setup] = dict(arms=rs, contrasts=con)

# val F5 vs raw (calibration/selection split)
L += ["### A: VAL carries (600), F5 v2 map vs raw map (eps_a 0), NEW only", "", "| setup | arm | seeds | drop | interv | dev | progress |", "|---|---|---|---|---|---|---|"]
RES["A"]["val"] = {}
for setup in ("piper_trk", "piper"):
    for arm, spec in ((MAINF5, "F5:calib_file=v2,eps_a=0"), (F5RAW, "F5:calib=0,eps_a=0")):
        seeds = SEEDS3 if setup == "piper_trk" else (0,)
        recs = A.load(EXPA, setup, spec, seeds, N, split="val")
        if recs is None:
            continue
        sm = M.summarize([r for s in seeds for r in recs[s]])
        RES["A"]["val"][f"{setup}|{arm}"] = dict(drop=sm["drop"]["rate"], interv=sm["interv_rate"], dev=sm["mean_dev_norm"], progress=sm["progress"])
        L.append(f"| {setup} | {arm} | {list(seeds)} | {pp(sm['drop']['rate'])} ({sm['drop']['k']}/{sm['drop']['n']}) | {pp(sm['interv_rate'])} | "
                 f"{f3(sm['mean_dev_norm'])} | {f3(sm['progress'])} |")
L.append("")

# ======================================================================================= B
RHOS = (0.0, 0.1, 0.2, 0.4)
SETUP_CAL = {"piper_trk": "calib_file=v2,eps_a=0,lam=0.1,sa=0.04", "piper": "calib_file=v2,eps_a=0,lam=0.03,sa=0.02"}


def bspec(setup, arm, rho=None):
    base = f"F5P:arm={arm}" if rho is None else f"F5P:arm={arm},rho={rho:g},pseed=0"
    return base + "," + SETUP_CAL[setup]


CK = ("t0", "t1s", "t2s", "end")
for setup in ("piper_trk", "piper"):
    seeds = SEEDS3 if setup == "piper_trk" else (0,)
    RES["B"][setup] = {}
    for rho in RHOS:
        data, rs = {}, {}
        for arm in ("point", "band", "conf", "unif", "oracle"):
            sp = bspec(setup, arm, rho if arm in ("point", "band", "conf") else None)
            recs = A.load(EXPB, setup, sp, seeds, N)
            if recs is None:
                L.append(f"MISSING B {setup} {sp}")
                continue
            aug(recs)
            data[arm] = recs
            flat = [r for s in seeds for r in recs[s]]
            sm = M.summarize(flat)
            rs[arm] = dict(spec=sp, drop=sm["drop"]["rate"], drop_k=sm["drop"]["k"], drop_n=sm["drop"]["n"], interv=sm["interv_rate"],
                           dev=sm["mean_dev_norm"], progress=sm["progress"], goal=sm["goal"]["rate"], n_tunnel=sm["n_tunnel"],
                           cov={})
            rs[arm]["cov"] = {c: float(np.mean([r[f"x_p_cov_{c}"] for r in flat if r.get(f"x_p_cov_{c}") is not None] or [np.nan])) for c in CK}
            rs[arm]["logw"] = float(np.mean([r.get("x_p_logw_mean", np.nan) for r in flat]))
        con = {}
        for x, y, k in [("point", "conf", "dropped"), ("band", "conf", "dropped"), ("point", "oracle", "dropped"), ("conf", "unif", "dropped"),
                        ("point", "conf", "progress"), ("conf", "unif", "progress"), ("oracle", "unif", "progress"),
                        ("conf", "unif", "n_interv"), ("point", "conf", "interv_c"), ("oracle", "unif", "interv_c")]:
            if x in data and y in data:
                con[f"{x}-{y}:{k}"] = A.boot_diff(A.mat(data[x], k), A.mat(data[y], k), rng)
        head = None
        if all(a in rs for a in ("conf", "unif", "oracle")):
            head = (rs["oracle"]["progress"] - rs["unif"]["progress"]) / rs["unif"]["progress"]
            rs["_kb2_rel"] = (rs["conf"]["progress"] - rs["unif"]["progress"]) / rs["unif"]["progress"]
        RES["B"][setup][f"rho={rho:g}"] = dict(arms=rs, contrasts=con, oracle_headroom=head)
        ob = OLD_B.get(setup, {}).get(f"rho={rho:g}", {})
        oa = ob.get("arms", {})
        tag = "main" if (setup == "piper_trk") else "secondary (seed 0)"
        L += [f"## B: {setup}, rho = {rho:g} ({tag})", "",
              "| arm | OLD drop | NEW drop (k/n) | OLD interv | NEW interv | OLD dev | NEW dev | OLD progress | NEW progress | NEW coverage t0/1s/2s/end | OLD coverage t0/end |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm, r in rs.items():
            if arm.startswith("_"):
                continue
            o = oa.get(arm)
            L.append(f"| {arm} | {pp(o['drop'], 2) if o else '-'} | {pp(r['drop'], 2)} ({r['drop_k']}/{r['drop_n']}) | {pp(o['interv']) if o else '-'} | {pp(r['interv'])} | "
                     f"{f3(o['dev']) if o else '-'} | {f3(r['dev'])} | {f3(o['progress'], 4) if o else '-'} | {f3(r['progress'], 4)} | "
                     + "/".join(f"{r['cov'][c]:.2f}" for c in CK) + f" | {('%.2f / %.2f' % (o['cov']['t0'][0], o['cov']['end'][0])) if o else '-'} |")
        L += ["", "| contrast | NEW point | NEW 95% CI | OLD point | OLD 95% CI |", "|---|---|---|---|---|"]
        for k, t in con.items():
            sc, u = (100, " pp") if k.endswith("dropped") else (1, "")
            ot = ob.get("contrasts", {}).get(k)
            L.append(f"| {k} | {sc*t[0]:+.4f}{u} | [{sc*t[1]:+.4f}, {sc*t[2]:+.4f}] | " + (f"{sc*ot[0]:+.4f}{u} | [{sc*ot[1]:+.4f}, {sc*ot[2]:+.4f}] |" if ot else "- | - |"))
        L += ["", f"Oracle headroom (progress oracle vs unif): NEW {f3(head, 4) if head is not None else '-'} (OLD {f3(ob.get('headroom'), 4)}); "
                  f"conf vs unif relative progress: NEW {f3(rs.get('_kb2_rel'), 4)} (OLD {f3(ob.get('kb2_rel'), 4)})", ""]
    # unif vs plain F5 (A) -- plain F5 for the same carries/seeds, matched model (v2 map, eps_a 0)
    u = A.load(EXPB, setup, bspec(setup, "unif"), seeds, N)
    f5 = A.load(EXPA, setup, "F5:calib_file=v2,eps_a=0", seeds, N)
    if u is not None and f5 is not None:
        aug(u); aug(f5)
        c = {k: A.boot_diff(A.mat(u, k), A.mat(f5, k), rng) for k in ("dropped", "progress", "n_interv", "interv_c")}
        RES["B"][setup]["unif_vs_F5"] = c
        oc = OLD_B.get(setup, {}).get("unif_vs_F5", {})
        L += [f"Online shrink alone ({setup}): NEW unif - plain F5: " + "; ".join(f"{k} {p:+.4f} [{lo:+.4f}, {hi:+.4f}]" for k, (p, lo, hi) in c.items())
              + ". OLD: " + "; ".join(f"{k} {p:+.4f} [{lo:+.4f}, {hi:+.4f}]" for k, (p, lo, hi) in oc.items()), ""]

# ======================================================================================= C
CFGS = [("F5", "F5:calib_file=v2,eps_a=0", 0.99, "F5@0.99"), ("oracle", "F5P:arm=oracle,calib_file=v2,eps_a=0,lam=0.1,sa=0.04", 0.99, "F5P:arm=oracle@0.99"),
        ("unif", "F5P:arm=unif,calib_file=v2,eps_a=0,lam=0.1,sa=0.04", 0.99, "F5P:arm=unif@0.99"),
        ("conf0.2", "F5P:arm=conf,rho=0.2,pseed=0,calib_file=v2,eps_a=0,lam=0.1,sa=0.04", 0.99, "F5P:arm=conf,rho=0.2,pseed=0@0.99"),
        ("point0.2", "F5P:arm=point,rho=0.2,pseed=0,calib_file=v2,eps_a=0,lam=0.1,sa=0.04", 0.99, "F5P:arm=point,rho=0.2,pseed=0@0.99"),
        ("conf0.4", "F5P:arm=conf,rho=0.4,pseed=0,calib_file=v2,eps_a=0,lam=0.1,sa=0.04", 0.99, "F5P:arm=conf,rho=0.4,pseed=0@0.99"),
        ("point0.4", "F5P:arm=point,rho=0.4,pseed=0,calib_file=v2,eps_a=0,lam=0.1,sa=0.04", 0.99, "F5P:arm=point,rho=0.4,pseed=0@0.99"),
        ("F5@0.4", "F5:calib_file=v2,eps_a=0", 0.4, "F5@0.4")]


def load_cfg(root, pre, spec, I):
    d = root / F.spec_tag(spec) / "test" / "piper_trk"
    out = {}
    for s in SEEDS3:
        recs = {}
        for p in glob.glob(str(d / f"{pre}__I-{I:g}__s{s}__p*of4.jsonl")):
            recs.update({r["ep"]: r for r in RP.read_jsonl(p)})
        if sorted(recs) != list(range(N)):
            return None
        out[s] = [recs[e] for e in range(N)]
    return out


# per-carry extras
t_arr = lambda r: float(r["cl_t_arrive_s"]) if (r.get("cl_arrived") and not r["dropped"]) else np.nan
arrived = lambda r: float(bool(r.get("cl_arrived")) and not bool(r["dropped"]))
goal = lambda r: float(bool(r["goal_reached"]))
tpen = lambda r: float(r["path_len_done"] / max(r["path_len_nom"], 1e-9))

OL = {k: load_cfg(CF5, "FIX", sp, I) for k, sp, I, _ in CFGS}
CL = {k: load_cfg(CF5CL, "CL", sp, I) for k, sp, I, _ in CFGS}
P6 = FX.p6()
P6_ok = P6 is not None


def row(v):
    D, Sx, G, T, I = FX.arr(v, FX.drop), FX.arr(v, FX.succ), FX.arr(v, goal), FX.arr(v, FX.tgoal), FX.arr(v, FX.interv)
    r = dict(drop=float(D.mean()), drop_k=int(D.sum()), success=float(Sx.mean()), goal=float(G.mean()), t_goal=float(np.nanmean(T)), n_t=int(np.isfinite(T).sum()),
             interv=float(I.mean()), path_ratio=float(FX.arr(v, tpen).mean()), mean_n_ticks=float(FX.arr(v, lambda r: r["n_ticks"]).mean()))
    if "cl_arrived" in v[0][0]:
        TA = FX.arr(v, t_arr)
        r.update(arrived=float(FX.arr(v, arrived).mean()), t_arrive=float(np.nanmean(TA)), n_ta=int(np.isfinite(TA).sum()))
    return r


L += ["## C-F5: BOED fixed-intensity carries, piper_trk, 600 TEST x seeds 0-2, I_rel = 0.99 unless stated", "",
      "OL = open-loop BOED plan (waypoint mode); CL = closed-loop nominal planner (plan_cl.CLPlan). success = goal reached and not dropped; time-to-goal = "
      "t_complete_s of successes (mean over each config's own successes; the paired ratios below use both-complete carries). "
      "OLD column = old-physics open loop (RESULT_CF5). Old physics closed loop exists only on VAL (plan_cl/val), not on TEST.", "",
      "| config | OLD OL drop | NEW OL drop (k/1800) | NEW CL drop (k/1800) | OLD OL success | NEW OL success | NEW CL success | OLD OL goal-time s | NEW OL | NEW CL | "
      "OLD OL interv | NEW OL interv | NEW CL interv |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
ROWS = {"OL": {}, "CL": {}}
for k, sp, I, oldk in CFGS:
    o = OLD_C["arms"].get(oldk)
    cells = []
    for nm, D in (("OL", OL), ("CL", CL)):
        if D[k] is not None:
            ROWS[nm][k] = row(D[k])
    ro, rc = ROWS["OL"].get(k), ROWS["CL"].get(k)
    L.append(f"| {k} | {pp(o['drop'], 2) if o else '-'} | {pp(ro['drop'], 2)} ({ro['drop_k']}) | {pp(rc['drop'], 2)} ({rc['drop_k']}) | "
             f"{pp(o['success']) if o else '-'} | {pp(ro['success'])} | {pp(rc['success'])} | {f3(o['t_goal'], 2) if o else '-'} | {f3(ro['t_goal'], 2)} | {f3(rc['t_goal'], 2)} | "
             f"{pp(o['interv']) if o else '-'} | {pp(ro['interv'])} | {pp(rc['interv'])} |")
if P6_ok:
    ROWS["P6"] = row(P6)
    p = ROWS["P6"]
    L.append(f"| P6 = frozen F3 @ 0.4 (OLD physics, stored) | {pp(OLD_C['arms']['P6 (frozen F3@0.4)']['drop'], 2)} | {pp(p['drop'], 2)} ({p['drop_k']}) | | "
             f"{pp(p['success'])} | | | {f3(p['t_goal'], 2)} | | | {pp(p['interv'])} | | |")
L += ["", "Other CL / OL descriptors (NEW; mean over 1800):", "", "| config | OL goal-reached | CL goal-reached | CL arrived (stricter, not dropped) | CL mean arrive time s (n) | OL path done/nominal | CL path done/nominal | OL mean ticks | CL mean ticks |",
      "|---|---|---|---|---|---|---|---|---|"]
for k, *_ in CFGS:
    ro, rc = ROWS["OL"].get(k), ROWS["CL"].get(k)
    L.append(f"| {k} | {pp(ro['goal'])} | {pp(rc['goal'])} | {pp(rc['arrived'])} | {f3(rc['t_arrive'], 2)} ({rc['n_ta']}) | {f3(ro['path_ratio'], 2)} | {f3(rc['path_ratio'], 2)} | "
             f"{f3(ro['mean_n_ticks'], 0)} | {f3(rc['mean_n_ticks'], 0)} |")
L.append("")
RES["C"]["rows"] = ROWS

CON = {"OL": {}, "CL": {}, "CL-OL": {}}
TR = {"OL": {}, "CL": {}, "CL-OL": {}}


def cc(D, nm, a, b, key, f, scale=1.0):
    x, y = (D[a] if a != "P6" else P6), (D[b] if b != "P6" else P6)
    if x is None or y is None:
        return
    CON[nm][f"{a} - {b} : {key}"] = A.boot_diff(FX.arr(x, f), FX.arr(y, f), rng)


def rr(D, nm, a, b):
    x, y = (D[a] if a != "P6" else P6), (D[b] if b != "P6" else P6)
    if x is None or y is None:
        return
    TR[nm][f"{a} / {b}"] = FX.boot_ratio(FX.arr(x, FX.tgoal), FX.arr(y, FX.tgoal), rng)


for nm, D in (("OL", OL), ("CL", CL)):
    for rho in ("0.2", "0.4"):
        for key, f in (("drop", FX.drop), ("success", FX.succ), ("interv", FX.interv)):
            cc(D, nm, f"point{rho}", f"conf{rho}", key, f)
        rr(D, nm, f"point{rho}", f"conf{rho}")
    for a, b in (("oracle", "F5"), ("unif", "F5"), ("conf0.2", "F5"), ("conf0.2", "unif"), ("oracle", "conf0.2")):
        for key, f in (("drop", FX.drop), ("success", FX.succ), ("interv", FX.interv)):
            cc(D, nm, a, b, key, f)
        rr(D, nm, a, b)
    if P6_ok:
        for a in ("F5", "oracle", "F5@0.4"):
            for key, f in (("drop", FX.drop), ("success", FX.succ)):
                cc(D, nm, a, "P6", key, f)
            rr(D, nm, a, "P6")
    cc(D, nm, "F5", "F5@0.4", "drop", FX.drop); cc(D, nm, "F5", "F5@0.4", "interv", FX.interv)
for k, *_ in CFGS:
    if OL[k] is None or CL[k] is None:
        continue
    for key, f in (("drop", FX.drop), ("success", FX.succ), ("interv", FX.interv)):
        x, y = OL[k], CL[k]
        CON["CL-OL"][f"CL:{k} - OL:{k} : {key}"] = A.boot_diff(FX.arr(y, f), FX.arr(x, f), rng)
    TR["CL-OL"][f"CL:{k} / OL:{k}"] = FX.boot_ratio(FX.arr(CL[k], FX.tgoal), FX.arr(OL[k], FX.tgoal), rng)

OLDK = {"point0.2": "F5P:arm=point,rho=0.2,pseed=0@0.99", "conf0.2": "F5P:arm=conf,rho=0.2,pseed=0@0.99", "point0.4": "F5P:arm=point,rho=0.4,pseed=0@0.99",
        "conf0.4": "F5P:arm=conf,rho=0.4,pseed=0@0.99", "F5": "F5@0.99", "oracle": "F5P:arm=oracle@0.99", "unif": "F5P:arm=unif@0.99", "F5@0.4": "F5@0.4", "P6": "P6 (frozen F3@0.4)"}


def oldkey(k, key):
    a, rest = k.split(" - ")
    b = rest.split(" : ")[0]
    if a in OLDK and b in OLDK:
        return f"{OLDK[a]} - {OLDK[b]} : {key}"
    return None


for nm in ("OL", "CL"):
    L += [f"### C-F5 contrasts, {nm} (NEW){'  | OLD = old-physics open loop' if nm == 'OL' else ''}", "", "| contrast | point | 95% CI | OLD point | OLD 95% CI |", "|---|---|---|---|---|"]
    for k, t in CON[nm].items():
        key = k.split(" : ")[1]
        sc, u = (100, " pp") if key in ("drop", "success", "interv") else (1, "")
        ok = oldkey(k, key) if nm == "OL" else None
        ot = OLD_C["contrasts"].get(ok) if ok else None
        L.append(f"| {k} | {sc*t[0]:+.2f}{u} | [{sc*t[1]:+.2f}, {sc*t[2]:+.2f}] | " + (f"{sc*ot[0]:+.2f}{u} | [{sc*ot[1]:+.2f}, {sc*ot[2]:+.2f}] |" if ot else "- | - |"))
    L += ["", f"Time-to-goal ratios, {nm} (both-complete carries; n = pairs)", "", "| ratio | value | 95% CI | n | OLD value (OL) |", "|---|---|---|---|---|"]
    for k, (p, lo, hi, n) in TR[nm].items():
        a, b = k.split(" / ")
        ok = f"{OLDK.get(a)} / {OLDK.get(b)}" if nm == "OL" else None
        ot = OLD_C["time_ratios"].get(ok) if ok else None
        L.append(f"| {k} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {n} | {('%.3f [%.3f, %.3f]' % tuple(ot[:3])) if ot else '-'} |")
    L.append("")
L += ["### C-F5 closed loop minus open loop, same carries and seeds (NEW)", "", "| contrast | point | 95% CI |", "|---|---|---|"]
for k, t in CON["CL-OL"].items():
    sc, u = (100, " pp")
    L.append(f"| {k} | {sc*t[0]:+.2f}{u} | [{sc*t[1]:+.2f}, {sc*t[2]:+.2f}] |")
L += ["", "| time ratio CL / OL (both complete) | ratio | 95% CI | n |", "|---|---|---|---|"]
for k, (p, lo, hi, n) in TR["CL-OL"].items():
    L.append(f"| {k} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {n} |")
L.append("")

# P6 sanity: are the stored P6 carries the same carries (mu) as the new v2 data?
if P6_ok and OL["F5"] is not None:
    mism = sum(abs(a["mu"] - b["mu"]) > 1e-12 * max(abs(a["mu"]), 1) for s in SEEDS3 for a, b in zip(P6[s], OL["F5"][s]))
    mism_seed = sum(a["seed"] != b["seed"] for s in SEEDS3 for a, b in zip(P6[s], OL["F5"][s]))
    L += [f"Frozen P6 (old physics) vs NEW carries: records whose mu differs {mism}/{N*3}; whose episode seed differs {mism_seed}/{N*3}.", ""]
    RES["C"]["p6_mu_mismatch"] = int(mism)
RES["C"]["contrasts"] = CON
RES["C"]["time_ratios"] = TR

(RB / "analysis").mkdir(parents=True, exist_ok=True)
md = "\n".join(L) + "\n"
(RB / "analysis" / "RESULT_REBASELINE.md").write_text(md)
(RB / "analysis" / "summary_rebaseline.json").write_text(json.dumps(RES, indent=1, default=float))
print(md, flush=True)
