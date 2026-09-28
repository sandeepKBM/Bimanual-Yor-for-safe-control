#!/usr/bin/env python
"""experiments/imitation/eval_policy.py -- closed-loop eval (SIM ONLY, CPU).

The BC policy REPLACES the scripted right-arm task reference (and gripper/
weld events); FL-MPC (mpc_fl.FLMPC via run_episode(controller="flmpc"),
unmodified, all hard constraints on) executes it. The scripted expert is run
on the SAME held-out seeds with the same episode length as the baseline.

Metrics per episode: placed (selected cubes actually moved into the keep
zone and resting), arm-arm collision, cup tilt max, balls spilled, h_min,
and the SAFETY-OVERRIDE meter (il_common): ||a_con - a_unc|| of FL-MPC's
first-block accel (rad/s^2) = how far the safety rows bent the motion away
from pure tracking of the proposed reference; + QP-infeasible fraction +
reference-vs-executed EE deviation.

  OMP_NUM_THREADS=1 ... python experiments/imitation/eval_policy.py \\
      --ckpt <run>/bc.pt --seeds 1000:1006 --workers 12 --out <run>/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

KEYS = ("seed", "mode", "placed_sel_final", "n_cubes_sel", "collision", "cup_tilt_max_deg", "spilled_max",
        "h_min", "right_cup_clearance_min_m", "ov_int_mean", "ov_int_p95", "ov_int_frac_gt1",
        "ov_int_frac_gt5", "ov_ego_mean", "ov_ego_frac_gt1", "ov_infeasible_frac",
        "ref_ee_dev_mean_m", "ref_ee_dev_p95_m", "n_pin_on", "pin_gate_dists", "n_steps",
        "right_ee_speed_max_ms", "wall_s", "error")


def _one(args):
    seed, mode, ckpt, n_cubes, time_scale, cup_jitter = args
    import il_common as C
    t0 = time.time()
    pol = None
    if mode == "policy":
        from bc_model import BCPolicy
        pol = BCPolicy(ckpt)
    try:
        s, ep, log, info = C.run(seed, mode=mode, policy=pol, n_cubes=n_cubes, time_scale=time_scale,
                                 cup_jitter_m=cup_jitter, override_meter=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        return dict(seed=seed, mode=mode, error=repr(e) + traceback.format_exc()[-800:])
    s["wall_s"] = time.time() - t0
    return {k: s[k] for k in KEYS if k in s}


def agg(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        R = [r for r in rows if r["mode"] == mode and "error" not in r]
        if not R:
            continue
        f = lambda k: float(np.mean([r[k] for r in R if k in r]))
        out[mode] = dict(n=len(R), placed_mean=f("placed_sel_final"),
                         full_success=float(np.mean([r["placed_sel_final"] == r["n_cubes_sel"] for r in R])),
                         collision_rate=float(np.mean([r["collision"] for r in R])),
                         cup_tilt_max_deg_mean=f("cup_tilt_max_deg"),
                         cup_tilt_max_deg_worst=float(np.max([r["cup_tilt_max_deg"] for r in R])),
                         spilled_max_mean=f("spilled_max"), h_min_mean=f("h_min"),
                         ov_int_mean=f("ov_int_mean"), ov_int_frac_gt1=f("ov_int_frac_gt1"),
                         ov_int_frac_gt5=f("ov_int_frac_gt5"), ov_ego_frac_gt1=f("ov_ego_frac_gt1"),
                         ov_infeasible_frac=f("ov_infeasible_frac"), ref_ee_dev_mean_m=f("ref_ee_dev_mean_m"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", default="1000:1004")
    ap.add_argument("--modes", default="policy,expert")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-cubes", type=int, default=3)
    ap.add_argument("--time-scale", type=float, default=1.25)
    ap.add_argument("--cup-jitter", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    jobs = [(s, m, a.ckpt, a.n_cubes, a.time_scale, a.cup_jitter)
            for s in range(lo, hi) for m in a.modes.split(",")]
    rows = []
    with Pool(min(a.workers, len(jobs)), maxtasksperchild=2) as pool:
        for r in pool.imap_unordered(_one, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)
    res = dict(rows=sorted(rows, key=lambda r: (r["mode"], r["seed"])), summary=agg(rows), args=vars(a))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
