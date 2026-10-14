"""G3 comparison: comprehensive_eval results for --controller-impl py vs rt (vs rt_pipe ...).
SIMULATION DATA ONLY. Usage:
  python rt_gate_g3.py --res py=<results_py.json> rt=<results_rt.json> [rt_pipe=...] [--out g3.json]
Prints per-seed and aggregate collision / spill / h_min_ref / fling / placed_strict / tilt /
jerk / SPARC, paired differences vs the first entry, and flags every SAFETY-metric regression
(collision or spill count up, h_min_ref worst down by > 2 mm, true cup tilt worst up by > 1 deg,
fling worst up by > 0.05 m/s).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

KEYS = [("collision", "collision"), ("spill", "spilled_max"), ("h_min_ref", "h_min_ref"),
        ("fling", "right_ee_speed_max_ms"), ("placed_strict", "placed_strict_final"),
        ("placed", "placed_final"), ("tilt_deg", "true_cup_tilt_max_deg"), ("jerk", "jerk_rms_right_ee"),
        ("sparc", "sparc_right_ee"), ("relaxed", "fl_relaxed_steps"), ("zero_fb", "fl_zero_fallback_steps"),
        ("qp_infeas", "qp_infeasible_frac_max")]


def load(fn):
    d = json.load(open(fn))
    rows = {r["seed"]: r for r in d["rows"] if r.get("ok") and r["controller"] == "flmpc"}
    bad = [r for r in d["rows"] if not r.get("ok")]
    return rows, bad, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sets = []
    for spec in args.res:
        name, fn = spec.split("=", 1)
        rows, bad, d = load(fn)
        sets.append((name, rows, bad))
    seeds = sorted(set.intersection(*[set(r) for _, r, _ in sets]))
    out = dict(seeds=seeds, per_seed={}, aggregate={}, flags=[], failed={n: [(b["seed"], b.get("err")) for b in bad]
                                                                         for n, _, bad in sets})
    print(f"seeds (common, ok): {seeds}")
    hdr = "seed " + " ".join(f"{n:>30s}" for n, _, _ in sets)
    for s in seeds:
        out["per_seed"][s] = {n: {k: rows[s].get(src) for k, src in KEYS} for n, rows, _ in sets}
    for n, rows, _ in sets:
        g = [rows[s] for s in seeds]
        agg = {}
        for k, src in KEYS:
            v = np.array([float(r.get(src, np.nan)) if r.get(src) is not None else np.nan for r in g])
            if k in ("collision",):
                agg[k] = int(np.nansum(v))
            elif k == "spill":
                agg[k] = int(np.sum(v > 0))
            else:
                agg[k] = dict(mean=float(np.nanmean(v)), worst=float(np.nanmin(v) if k in ("h_min_ref", "sparc", "placed_strict", "placed") else np.nanmax(v)))
        out["aggregate"][n] = agg
    base = sets[0][0]
    A0 = out["aggregate"][base]
    for n, _, _ in sets[1:]:
        A = out["aggregate"][n]
        if A["collision"] > A0["collision"]:
            out["flags"].append(f"{n}: collision {A['collision']} > {base} {A0['collision']}")
        if A["spill"] > A0["spill"]:
            out["flags"].append(f"{n}: spill episodes {A['spill']} > {base} {A0['spill']}")
        if A["h_min_ref"]["worst"] < A0["h_min_ref"]["worst"] - 0.002:
            out["flags"].append(f"{n}: h_min_ref worst {A['h_min_ref']['worst']:.4f} < {base} {A0['h_min_ref']['worst']:.4f} - 2mm")
        if A["tilt_deg"]["worst"] > A0["tilt_deg"]["worst"] + 1.0:
            out["flags"].append(f"{n}: cup tilt worst {A['tilt_deg']['worst']:.2f} > {base} {A0['tilt_deg']['worst']:.2f} + 1deg")
        if A["fling"]["worst"] > A0["fling"]["worst"] + 0.05:
            out["flags"].append(f"{n}: fling worst {A['fling']['worst']:.3f} > {base} {A0['fling']['worst']:.3f} + 0.05")
        # paired differences
        pd = {}
        for k, src in KEYS:
            if k in ("collision", "spill"):
                continue
            nan_if_none = (lambda v: np.nan if v is None else float(v))
            a = np.array([nan_if_none(sets[0][1][s].get(src)) for s in seeds])
            b = np.array([nan_if_none(sets[[x[0] for x in sets].index(n)][1][s].get(src)) for s in seeds])
            dd = b - a
            pd[k] = dict(mean=float(np.nanmean(dd)), std=float(np.nanstd(dd)), min=float(np.nanmin(dd)),
                         max=float(np.nanmax(dd)))
        out["paired_diff_vs_" + base + "_" + n] = pd
    print(json.dumps(out["aggregate"], indent=1))
    for k in out:
        if k.startswith("paired_diff"):
            print(k, json.dumps(out[k], indent=1))
    print("FLAGS:", out["flags"] or "none")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
