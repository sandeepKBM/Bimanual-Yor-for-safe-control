"""boed/fxi_analyze.py -- Experiment C-F5 (exploratory extension; bimanual_data/tray_rep/boed_f5/PREREG_CF5_FROZEN.md). SAVED DATA ONLY.

Fixed-intensity carries of the BOED harness (boed/run_fxi.py), piper_trk, 600 TEST carries x seeds 0-2, with different carry filters.
  PRIMARY  KS1 "sets beat points at speed" (I = 0.99, rho = 0.2): PASS iff the 95% CI of drop(F5P point) - drop(F5P conf) lies above 0.
  Descriptive: the same at rho = 0.4; F5 / oracle / unif / F3 at I = 0.99 and F5 at I = 0.4 against the frozen P6 (F3 at I = 0.4);
  success = goal reached and not dropped; time to goal on carries both arms complete; intervention rate; the reproduction of P6 by
  F3 at I = 0.4 (same carries, seeds, filter, intensity) on dropped / n_ticks / t_complete_s.
Bootstrap: sets.expA_analyze.boot_diff (carries x seeds, B = 10000); time ratio: the same resampling, ratio of means on both-complete carries.
rng 20261010.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.boed import report as RP
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.sets import expA_analyze as A

OUT = Path("/common/users/ss5772/bimanual_data/tray_rep/boed_f5")
FROZEN = Path("/common/users/ss5772/bimanual_data/tray_rep/boed")
SEEDS, N = (0, 1, 2), 600
CONFIGS = [("F5P:arm=point,rho=0.2,pseed=0", 0.99), ("F5P:arm=conf,rho=0.2,pseed=0", 0.99), ("F5", 0.99), ("F5P:arm=oracle", 0.99),
           ("F5P:arm=unif", 0.99), ("F3", 0.99), ("F5P:arm=point,rho=0.4,pseed=0", 0.99), ("F5P:arm=conf,rho=0.4,pseed=0", 0.99),
           ("F5", 0.4), ("F3", 0.4)]


def load_cfg(filt, I):
    d = OUT / F.spec_tag(filt) / "test" / "piper_trk"
    out = {}
    for s in SEEDS:
        recs = {}
        for p in glob.glob(str(d / f"FIX__I-{I:g}__s{s}__p*of4.jsonl")):
            recs.update({r["ep"]: r for r in RP.read_jsonl(p)})
        if not all(e in recs for e in range(N)):
            return None
        out[s] = [recs[e] for e in range(N)]
    return out


def p6():
    d = RP.load(FROZEN, "main", "test")["piper_trk"]["P6"]
    return {s: [d[s][e] for e in range(N)] for s in SEEDS} if all(s in d and len(d[s]) >= N for s in SEEDS) else None


def arr(recs, f):
    return np.stack([np.array([f(r) for r in recs[s]], float) for s in SEEDS], 1)


succ = lambda r: float(bool(r["goal_reached"]) and not bool(r["dropped"]))
drop = lambda r: float(bool(r["dropped"]))
interv = lambda r: float(r["interv_rate"])
tgoal = lambda r: float(r["t_complete_s"]) if (r["goal_reached"] and not r["dropped"]) else np.nan


def boot_ratio(TA, TB, rng, B=10000):
    ok = np.isfinite(TA) & np.isfinite(TB)
    C, S = TA.shape
    pt = float(np.nanmean(np.where(ok, TA, np.nan)) / np.nanmean(np.where(ok, TB, np.nan)))
    v = np.empty(B)
    for b in range(B):
        c, s = rng.integers(0, C, C), rng.integers(0, S, S)
        o = ok[c][:, s]
        v[b] = TA[c][:, s][o].mean() / TB[c][:, s][o].mean() if o.any() else np.nan
    return pt, float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), int(ok.sum())


def main():
    rng = np.random.default_rng(20261010)
    data = {f"{f}@{I:g}": load_cfg(f, I) for f, I in CONFIGS}
    data["P6 (frozen F3@0.4)"] = p6()
    L = ["## C-F5 (piper_trk, 600 TEST carries x seeds 0-2; exploratory)", "",
         "| config | drop | success (goal, no drop) | time to goal s (successes) | interv rate |", "|---|---|---|---|---|"]
    res = {"arms": {}}
    for k, v in data.items():
        if v is None:
            L.append(f"| {k} | MISSING | | | |")
            continue
        D, Sx, T, I = arr(v, drop), arr(v, succ), arr(v, tgoal), arr(v, interv)
        res["arms"][k] = dict(drop=float(D.mean()), success=float(Sx.mean()), t_goal=float(np.nanmean(T)), interv=float(I.mean()))
        L.append(f"| {k} | {100*D.mean():.2f}% | {100*Sx.mean():.1f}% | {np.nanmean(T):.2f} | {100*I.mean():.1f}% |")
    con = {}

    def c(a, b, key, f):
        if data.get(a) is not None and data.get(b) is not None:
            con[f"{a} - {b} : {key}"] = A.boot_diff(arr(data[a], f), arr(data[b], f), rng)

    pt2, cf2 = "F5P:arm=point,rho=0.2,pseed=0@0.99", "F5P:arm=conf,rho=0.2,pseed=0@0.99"
    pt4, cf4 = "F5P:arm=point,rho=0.4,pseed=0@0.99", "F5P:arm=conf,rho=0.4,pseed=0@0.99"
    c(pt2, cf2, "drop", drop); c(pt4, cf4, "drop", drop); c(pt2, cf2, "success", succ); c(pt4, cf4, "success", succ)
    for a in ("F5@0.99", "F5P:arm=oracle@0.99", "F3@0.99", "F5@0.4"):
        c(a, "P6 (frozen F3@0.4)", "drop", drop); c(a, "P6 (frozen F3@0.4)", "success", succ)
    c("F5P:arm=oracle@0.99", "F5@0.99", "interv", interv); c("F5P:arm=oracle@0.99", "F5@0.99", "success", succ)
    c("F5P:arm=unif@0.99", "F5@0.99", "interv", interv); c(cf2, "F5@0.99", "interv", interv)
    c("F3@0.99", "F5@0.99", "drop", drop)
    res["contrasts"] = con
    L += ["", "| contrast | point | 95% CI |", "|---|---|---|"]
    for k, (p, lo, hi) in con.items():
        L.append(f"| {k} | {p:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    rat = {}
    for a, b in (("F5@0.99", "P6 (frozen F3@0.4)"), ("F5P:arm=oracle@0.99", "F5@0.99"), ("F5@0.4", "P6 (frozen F3@0.4)"),
                 ("F3@0.99", "F5@0.99"), (cf2, "F5@0.99"), (pt2, cf2)):
        if data.get(a) is not None and data.get(b) is not None:
            rat[f"{a} / {b}"] = boot_ratio(arr(data[a], tgoal), arr(data[b], tgoal), rng)
    res["time_ratios"] = rat
    L += ["", "| time-to-goal ratio (carries both complete) | ratio | 95% CI | n |", "|---|---|---|---|"]
    for k, (p, lo, hi, n) in rat.items():
        L.append(f"| {k} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {n} |")
    ks1 = None
    k = f"{pt2} - {cf2} : drop"
    if k in con:
        ks1 = "PASS" if con[k][1] > 0 else "KILL"
    res["KS1"] = ks1
    # reproduction of the frozen P6 by F3 at I = 0.4
    if data.get("F3@0.4") is not None and data.get("P6 (frozen F3@0.4)") is not None:
        a, b = data["F3@0.4"], data["P6 (frozen F3@0.4)"]
        bad = {fld: sum(ra[fld] != rb[fld] and not (isinstance(ra[fld], float) and np.isnan(ra[fld]) and np.isnan(rb[fld]))
                        for s in SEEDS for ra, rb in zip(a[s], b[s])) for fld in ("dropped", "n_ticks", "t_complete_s", "n_interv")}
        res["reproduction_P6"] = bad
        L += ["", f"Reproduction of frozen P6 by F3@0.4 (records differing, of {N*len(SEEDS)}): {bad}"]
    L += ["", f"**KS1 (sets beat points at speed, rho = 0.2): {ks1}**", ""]
    md = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "tables_CF5.md").write_text(md)
    (OUT / "analysis" / "summary_CF5.json").write_text(json.dumps(res, indent=1, default=float))
    print(md, flush=True)


if __name__ == "__main__":
    main()
