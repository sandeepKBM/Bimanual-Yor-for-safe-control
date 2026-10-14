"""grasp/sweep.py -- capture-range sweep of the friction grasp: tray placement error (offset 0-4 cm) x (yaw 0-8 deg), 3 seeds per cell.  SIMULATION ONLY.

  cd /common/users/ss5772/bimanual ; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 BIMANUAL_TELEOP_FAST=1 PYTHONPATH=$PWD:$PWD/experiments/ab_smooth_vs_bangbang:$PWD/experiments/realtime
  PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
  $PY -m world_model.v2.house.grasp.sweep --look 1 --workers 28 --out /data/local/ss5772/molmospaces/grasp/v2/sweep_look1.json
  $PY -m world_model.v2.house.grasp.sweep --look 0 --workers 28 --out /data/local/ss5772/molmospaces/grasp/v2/sweep_look0.json
  $PY -m world_model.v2.house.grasp.sweep --report /data/local/ss5772/molmospaces/grasp/v2/sweep_look1.json /data/local/ss5772/molmospaces/grasp/v2/sweep_look0.json

Cell (offset, yaw): seed k places the tray EXACTLY `offset` away along tray direction (20 + 120 k) deg and yawed by (-1)^k * yaw (so a cell covers the approach
axis, the along-edge axis and a diagonal).  --look 0 switches off the lateral look (the hover-pose estimate is then the only camera information); fusion
and all the other options stay as configured.  Outcome per episode: verified grasp (HOLD) on the FIRST attempt, truly held (tray rose with the hands in the tug).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np


def _one(args):
    off, deg, k, look = args
    from .smoke import run_episode
    seed = 100 + int(round(off * 100)) * 10 + int(deg) + 1000 * k
    try:
        r = run_episode(seed, grasp_only=True, offset_m=off, offset_deg=deg, ring=(20 + 120 * k, (-1) ** k), cfg_kw=dict(look=bool(look)), verbose=False)
        mt = r.get("metrics", {})
        fe = (mt.get("final_err") or [None])[-1]
        row = dict(offset=off, deg=deg, k=k, seed=seed, look=look, state=r.get("grasp_state"), attempts=mt.get("attempts"), held=bool(r.get("truly_held")),
                   first=bool(r.get("grasp_state") == "HOLD" and mt.get("attempts") == 1), fail=mt.get("fail_log"),
                   t=None if not mt.get("t_done") else float(mt["t_done"] - mt["t_trigger"]),
                   final_err=fe, gap=mt.get("gap"), tug_mm=r.get("tug_rise_true_mm"))
    except Exception as e:                                           # a crashed episode is a failed cell, not a lost sweep
        row = dict(offset=off, deg=deg, k=k, look=look, state="CRASH", first=False, held=False, fail=repr(e))
    print(json.dumps({k_: row.get(k_) for k_ in ("offset", "deg", "k", "state", "first", "held", "t")}, default=float), flush=True)
    return row


def report(paths):
    for p in paths:
        R = json.load(open(p))
        offs = sorted({r["offset"] for r in R})
        degs = sorted({r["deg"] for r in R})
        print(f"\n{p}  (look={R[0]['look']}, n={len(R)})   cell = first-attempt verified grasps / seeds  (truly held)")
        print("offset\\yaw " + " ".join(f"{d:>7.0f}" for d in degs))
        for o in offs:
            cells = []
            for d in degs:
                c = [r for r in R if r["offset"] == o and r["deg"] == d]
                cells.append(f"{sum(r['first'] for r in c)}/{len(c)}({sum(r['held'] for r in c)})".rjust(7))
            print(f"{o*100:>6.0f} cm " + " ".join(cells))
        ok = [r for r in R if r["first"]]
        if ok:
            print("median time (first-attempt successes): %.2f s" % np.median([r["t"] for r in ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", type=int, default=1)
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.03, 0.04])
    ap.add_argument("--degs", type=float, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", nargs="+", default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report)
        return
    jobs = [(o, d, k, a.look) for o in a.offsets for d in a.degs for k in range(a.seeds)]
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=2) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rows, open(a.out, "w"), default=float)
    tmp = a.out or "/dev/null"
    if a.out:
        report([tmp])


if __name__ == "__main__":
    main()
