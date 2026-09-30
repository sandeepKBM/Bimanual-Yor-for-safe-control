"""G2 closed-loop gate for rt_flmpc (seed 0, gate_equivalence's library-default config
unless --profile says otherwise). SIMULATION ONLY.

Runs the flmpc episode with --controller-impl py (fast path installed; bit-identical to
the original, see gate_equivalence.py) and with --controller-impl <impl>, then reports
collision / placed_final / h_min for both, whether py matches the known baseline, and the
max |diff| of every logged per-step series (NaN-aware), plus the rt counters.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import rt_harness as H
import fast_barriers as fb
import fast_qp as fq
import _demo_env as env

demo = H.demo


def run(profile, seed, impl, max_steps, fast=True):
    if fast:
        fb.install(demo)
        fq.install()
    try:
        info = H.build(profile, seed)
        t0 = time.perf_counter()
        log, summ = H.run(info, profile, max_steps=max_steps, controller_impl=impl)
        wall = time.perf_counter() - t0
    finally:
        if fast:
            fb.uninstall()
            fq.uninstall()
    return log, summ, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", default="rt")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=H.RUN_DIR + "/g2")
    ap.add_argument("--py-log", default=None, help="reuse a saved py log npz instead of re-running py")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.profile}_s{args.seed}"
    if args.py_log and os.path.exists(args.py_log):
        z = np.load(args.py_log, allow_pickle=True)
        lp = {k: z[k] for k in z.files}
        sp, wp = demo.summarize(lp), float("nan")
    else:
        lp, sp, wp = run(args.profile, args.seed, "py", args.max_steps)
        np.savez_compressed(f"{args.out}/log_py_{tag}.npz", **{k: np.asarray(v) for k, v in lp.items()})
    lr, sr, wr = run(args.profile, args.seed, args.impl, args.max_steps)
    np.savez_compressed(f"{args.out}/log_{args.impl}_{tag}.npz", **{k: np.asarray(v) for k, v in lr.items()})
    diffs = {}
    for k in lp:
        if k not in lr or k == "solve_time":
            continue
        a, b = np.asarray(lp[k], float), np.asarray(lr[k], float)
        if a.shape != b.shape:
            n = min(len(a), len(b)) if a.ndim and b.ndim else 0
            diffs[k] = f"shape {a.shape} vs {b.shape}" + (f"; max|diff| first {n}: {float(np.nanmax(np.abs(a[:n]-b[:n]))):.3e}" if n else "")
            continue
        if not np.array_equal(a, b, equal_nan=True):
            diffs[k] = float(np.nanmax(np.abs(a - b)))
    base = env.BASELINE_SEED0_FLMPC
    res = dict(profile=args.profile, seed=args.seed, impl=args.impl,
               py=dict(collision=sp["collision"], placed_final=sp["placed_final"], h_min=sp["h_min"],
                       matches_known_baseline=bool(args.profile == "gate" and all(sp[k] == v for k, v in base.items())),
                       wall_s=wp),
               rt=dict(collision=sr["collision"], placed_final=sr["placed_final"], h_min=sr["h_min"],
                       placed_strict_final=sr.get("placed_strict_final"), spilled_max=sr.get("spilled_max"),
                       right_ee_speed_max_ms=sr.get("right_ee_speed_max_ms"), wall_s=wr,
                       **{k: (np.asarray(v).tolist()) for k, v in lr.items() if k.startswith("fl_rt_")}),
               n_series_differing=len(diffs), series_max_abs_diff=diffs,
               summary_diffs={k: [sp[k], sr[k]] for k in sp if k in sr and sp[k] != sr[k]
                              and not (isinstance(sp[k], float) and np.isnan(sp[k]) and np.isnan(sr[k]))})
    json.dump(res, open(f"{args.out}/g2_{args.impl}_{tag}.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
