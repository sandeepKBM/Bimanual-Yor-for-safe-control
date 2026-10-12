"""robust/launch_local.py -- local (no Slurm) launcher and Slurm task-file writer for the perception-robustness tests.
CPU ONLY. SIMULATION / SAVED DATA ONLY. NOT run at full scale by the agent that wrote it: the owner launches it.

Order (each stage is resumable; rerun = resume):
  oracle_val   certified-savable labels on VAL (needed by test1 --split val and by the conformal threshold)
  test1        false-safe map. TEST states, and VAL states for the cond `trk`/`trkB10f` (conformal tau)
  budget       oracle tolerable delay + turnaround (Test 3 budget)
  tune         VAL: rho*, profile, match the 5 filters to the same intervention rate (one task per cell)
  test2        TEST: the 5 filters (+ no filter) at the matched thresholds (needs `tune`)
  test3        latency sweep (needs `tune` for the rob threshold; falls back to thr = tau)
  report       tables + plots (md, json, png) under <OUT>/report/<tag>/
  smoke        a few states / carries per condition, both setups, plus determinism checks -> <OUT>/smoke/

  python -m world_model.v2.tray_rep.robust.launch_local --stage test1 --dry-run
  python -m world_model.v2.tray_rep.robust.launch_local --stage test1 --jobs 2 --workers 2          # 4 cores
  python -m world_model.v2.tray_rep.robust.launch_local --stage tune --list-tasks                   # writes slurm/robust_tasks_tune.txt
Slurm: sbatch --array=0-<N-1>%12 world_model/v2/tray_rep/robust/slurm/robust_array.sbatch tune   (owner submits; this launcher does not)
Cores used = --jobs x --workers. Westeros is shared: stay at or under the cores you may use.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path("/common/users/ss5772/bimanual")
ROOT = Path("/common/users/ss5772/bimanual_data/tray_rep")
OUT = ROOT / "robust"
PY = "/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python"
M = "world_model.v2.tray_rep.robust"
SLURM = REPO / "world_model/v2/tray_rep/robust/slurm"
SETUPS = ("piper", "piper_trk")
ARMS = ("N0", "Rc3k")
SEEDS = (0, 1, 2)
COND1 = ("off", "trk", "trkn6", "trklat5", "trkB10s", "trkB10f", "trkB25s")       # Test 1 conditions
COND2 = ("trk", "trkB10f", "trkB25f")                                             # Test 2 conditions
VAL_CONDS = ("trk", "trkB10s", "trkB10f")                                         # Test 1 VAL conditions (conformal tau)


def log(*a):
    print(time.strftime("%FT%T"), *a, flush=True)


def plan(stage, a):
    """-> list of (name, module-args string) without --workers."""
    t = []
    setups = a.setups.split(",")
    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    if stage == "oracle_val":
        n = a.oracle_shards
        t += [(f"oracle_val_{i}of{n}", f"{M}.oracle_val --n {a.oracle_n} --shard {i}/{n}") for i in range(n)]
    elif stage == "test1":
        for su, ar, sd, c in itertools.product(setups, arms, seeds, a.cond1.split(",")):
            t.append((f"t1_{su}_{ar}_s{sd}_{c}", f"{M}.test1 run --setup {su} --arm {ar} --seed {sd} --cond {c} --split test --tag {a.tag}"))
        for su, ar, sd, c in itertools.product(setups, arms, seeds, VAL_CONDS):
            t.append((f"t1v_{su}_{ar}_s{sd}_{c}", f"{M}.test1 run --setup {su} --arm {ar} --seed {sd} --cond {c} --split val --tag {a.tag}"))
    elif stage == "budget":
        t += [(f"budget_{su}", f"{M}.budget run --setup {su} --n-states {a.budget_n} --tag {a.tag}") for su in ("piper",)]   # oracle labels are identical in both setups
    elif stage in ("tune", "test2"):
        verb = "tune" if stage == "tune" else "test"
        for su, ar, sd, c in itertools.product(setups, arms, seeds, a.cond2.split(",")):
            extra = f" --n-val {a.n_val}" if verb == "tune" else ""
            if verb == "tune" and c in VAL_CONDS:
                extra += f" --test1-val {OUT}/test1/{a.tag}/val/{su}/{ar}_s{sd}__{c}.jsonl"
            t.append((f"{stage}_{su}_{ar}_s{sd}_{c}", f"{M}.tune {verb} --setup {su} --arm {ar} --seed {sd} --cond {c} --tag {a.tag}{extra}"))
    elif stage == "test3":
        for su, ar, sd in itertools.product(setups, arms, seeds):
            if ar != "Rc3k" and not a.test3_all_arms:
                continue
            t.append((f"test3_{su}_{ar}_s{sd}", f"{M}.test3 run --setup {su} --arm {ar} --seed {sd} --tag {a.tag}"))
    elif stage == "report":
        t.append(("report", f"{M}.report all --tag {a.tag}"))
    else:
        raise SystemExit(f"unknown stage {stage}")
    return t


def run_tasks(tasks, jobs, workers, dry):
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    q = Queue()
    for x in tasks:
        q.put(x)
    logd = OUT / "logs"
    logd.mkdir(parents=True, exist_ok=True)
    fails = []

    def worker():
        while True:
            try:
                name, args = q.get_nowait()
            except Exception:
                return
            cmd = [PY, "-m"] + args.split() + (["--workers", str(workers)] if " report" not in args else [])
            t0 = time.time()
            with open(logd / f"{name}.log", "a") as f:
                rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT)
            log(f"[{'ok' if rc == 0 else 'FAIL'}] {name} rc={rc} {time.time() - t0:.0f}s")
            if rc:
                fails.append(name)

    if dry:
        for name, args in tasks:
            print(f"{name}: {PY} -m {args} --workers {workers}")
        return 0
    th = [threading.Thread(target=worker) for _ in range(jobs)]
    [x.start() for x in th]
    [x.join() for x in th]
    log(f"done, {len(fails)} failed {fails}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["oracle_val", "test1", "budget", "tune", "test2", "test3", "report", "smoke"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--setups", default=",".join(SETUPS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--cond1", default=",".join(COND1))
    ap.add_argument("--cond2", default=",".join(COND2))
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--oracle-n", type=int, default=1500)
    ap.add_argument("--oracle-shards", type=int, default=10)
    ap.add_argument("--budget-n", type=int, default=100)
    ap.add_argument("--test3-all-arms", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-tasks", action="store_true", help="write slurm/robust_tasks_<stage>.txt and stop")
    a = ap.parse_args()
    if a.stage == "smoke":
        sys.exit(subprocess.call([PY, "-m", f"{M}.smoke", "all", "--workers", str(a.workers)], cwd=str(REPO),
                                 env=dict(os.environ, PYTHONPATH=str(REPO))))
    tasks = plan(a.stage, a)
    if a.list_tasks:
        SLURM.mkdir(parents=True, exist_ok=True)
        p = SLURM / f"robust_tasks_{a.stage}.txt"
        p.write_text("\n".join(args for _, args in tasks) + "\n")
        log(f"wrote {p} ({len(tasks)} tasks): sbatch --array=0-{len(tasks) - 1}%12 {SLURM}/robust_array.sbatch {a.stage}")
        return
    log(f"stage {a.stage}: {len(tasks)} tasks, jobs={a.jobs} x workers={a.workers} = {a.jobs * a.workers} cores")
    sys.exit(run_tasks(tasks, a.jobs, a.workers, a.dry_run))


if __name__ == "__main__":
    main()
