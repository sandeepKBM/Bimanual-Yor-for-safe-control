"""people_track/ps_report.py -- table of a ps_sweep directory: per (walker, shield, cond) and pooled over walkers.  SIMULATION ONLY.

    python -m world_model.v2.house.people_track.ps_report DIR [--shield new] [--seeds 0-8]
Columns: eps, contact eps (true_contact_ticks > 0), finished, mean task time over finished eps, min TRUE clearance, beads out (sum / eps with beads),
walker in FOV % (ticks with the walker within 4 m), glances (mean per ep), unseen-cap ticks (mean).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def rows(d, shield=None, seeds=None):
    out = []
    for f in sorted(glob.glob(str(Path(d) / "*_s*.json"))):
        if "summary" in f:
            continue
        j = json.loads(Path(f).read_text())
        if "walker" not in j or (shield and j.get("shield_variant") != shield):
            continue
        if seeds is not None and j["seed"] not in seeds:
            continue
        out.append(j)
    return out


def fmt(label, rs):
    n = len(rs)
    fin = [r for r in rs if r["task_done"]]
    fov = [r["walker_fov_frac_near"] for r in rs if r.get("walker_fov_frac_near") is not None]
    gl = [r["glance_stats"]["n_glances"] for r in rs if r.get("glance_stats")]
    gs = [r["glance_stats"]["glance_s"] for r in rs if r.get("glance_stats")]
    uc = [r.get("n_unseen_cap_ticks", 0) * 0.02 for r in rs]
    mc = [r["min_true_clear_m"] for r in rs if r.get("min_true_clear_m") is not None]
    return (f"{label:34s} n={n:2d} contact={sum(r['true_contact_ticks'] > 0 for r in rs):2d} fin={len(fin):2d}/{n:<2d} "
            f"t_fin={np.mean([r['task_time_s'] for r in fin]) if fin else float('nan'):5.1f} minTrueClr={min(mc) if mc else float('nan'):6.3f} "
            f"beads={sum(r['spills_beads_out'] for r in rs):2d}({sum(r['spills_beads_out'] > 0 for r in rs)}ep) "
            f"fov%near={100 * np.mean(fov) if fov else float('nan'):5.1f} glances={np.mean(gl) if gl else 0:4.1f} glance_s={np.mean(gs) if gs else 0:4.1f} cap_s={np.mean(uc):5.1f} "
            f"stuck={sum(bool(r.get('aborted_stuck')) for r in rs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--shield", default="new")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--walkers", default=None, help="comma list (default: all found)")
    a = ap.parse_args()
    seeds = None
    if a.seeds:
        s = a.seeds
        seeds = set(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1)) if "-" in s else {int(x) for x in s.split(",")}
    R = rows(a.dir, a.shield, seeds)
    if a.walkers:
        R = [r for r in R if r["walker"] in a.walkers.split(",")]
    conds = sorted({r["cond"] for r in R}, key=lambda c: (c != "truth", len(c), c))
    walkers = sorted({r["walker"] for r in R})
    for w in walkers + ["ALL"]:
        print(f"--- walker: {w}")
        for c in conds:
            rs = [r for r in R if r["cond"] == c and (w == "ALL" or r["walker"] == w)]
            if rs:
                print(fmt(c, rs))


if __name__ == "__main__":
    main()
