"""wm/launch_local.py -- full-scale tray_rep phase-1b pipeline on westeros (GPUs 4-7, ONE job per GPU), no Slurm.
SIMULATION / SAVED DATA ONLY (no hardware, no network). NOT launched by the agent that wrote it: the owner runs it.

  python -m world_model.v2.tray_rep.wm.launch_local --stage data      # 2 x 3000 nominal episodes, merge
  python -m world_model.v2.tray_rep.wm.launch_local --stage oracle    # 2000 TEST states (full mode), copied to tray_only
  python -m world_model.v2.tray_rep.wm.launch_local --stage prep      # norm + branches (R, Rc x 300/1000/3000 x seeds 0-2) + CL refs
  python -m world_model.v2.tray_rep.wm.launch_local --stage arms --only N0     # 6 N0 cells first, then decide with G1
  python -m world_model.v2.tray_rep.wm.launch_local --stage g1
  python -m world_model.v2.tray_rep.wm.launch_local --stage arms                # all 42 cells (done cells are skipped)
  add --dry-run to print the commands only. Run detached:
  setsid nohup $PY -m world_model.v2.tray_rep.wm.launch_local --stage arms > $R/logs/arms.log 2>&1 &
Arms queue = slurm/tr_tasks.txt (the same file the Slurm array uses). A cell with _DONE is skipped; a failed cell is
retried once. CPU stages use --workers 16. Wait for a stage to end before starting the next.
Before the arms: check nvidia-smi for GPUs 4-7 (another job may hold one). GPUs 0-3 belong to another project.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path("/common/users/ss5772/bimanual")
ROOT = Path("/common/users/ss5772/bimanual_data/tray_rep")
PY = "/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python"
GPUS = [4, 5, 6, 7]
SETUP = "piper"
MODES = ("full", "tray_only")
TASKS = REPO / "world_model/v2/tray_rep/wm/slurm/tr_tasks.txt"
M = "world_model.v2.tray_rep"


def log(*a):
    print(time.strftime("%FT%T"), *a, flush=True)


def env_for(tag, gpu):
    tmp = ROOT / "tmp" / tag
    tmp.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, TMPDIR=str(tmp), MPLCONFIGDIR=str(tmp / "mpl"), PYTHONPYCACHEPREFIX=str(tmp / "pyc"),
             XDG_CACHE_HOME=str(tmp / "cache"), CUDA_CACHE_PATH=str(tmp / "nv"), NUMBA_CACHE_DIR=str(tmp / "numba"),
             OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
             PYTHONUNBUFFERED="1", PYTHONPATH=str(REPO))
    e["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    return e


def run(tag, cmd, gpu=None, dry=False):
    log(("DRY " if dry else "RUN ") + tag + ": " + " ".join(cmd))
    if dry:
        return 0
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "logs" / f"{tag}.log", "a") as f:
        return subprocess.call(cmd, cwd=REPO, env=env_for(tag, gpu), stdout=f, stderr=subprocess.STDOUT)


def data_dir(mode):
    return ROOT / f"data_{SETUP}_{mode}"


def stage_data(a):
    for m in MODES:
        base = [PY, "-m", f"{M}.gen_nominal", "--config", SETUP, "--obs-mode", m, "--n-eps", "3000", "--workers", str(a.workers)]
        run(f"gen_{m}", base, dry=a.dry_run)                      # resumable: finished shards are skipped
        run(f"merge_{m}", base + ["--merge"], dry=a.dry_run)


def stage_oracle(a):
    d = data_dir("full")
    run("oracle", [PY, "-m", f"{M}.oracle", "run", "--data", str(d), "--split", "test", "--n", "2000", "--workers",
                   str(a.workers), "--tag", "test_v0"], dry=a.dry_run)
    if not a.dry_run:                                              # same episodes + snapshots in both modes -> same oracle
        (data_dir("tray_only") / "oracle").mkdir(parents=True, exist_ok=True)
        for ext in ("npz", "json"):
            shutil.copy(d / "oracle" / f"test_v0.{ext}", data_dir("tray_only") / "oracle" / f"test_v0.{ext}")


def stage_prep(a):
    for m in MODES:
        d = str(data_dir(m))
        run(f"prep_norm_{m}", [PY, "-m", f"{M}.wm.prep", "norm", "--data", d, "--workers", str(a.workers)], dry=a.dry_run)
        run(f"prep_branches_{m}", [PY, "-m", f"{M}.wm.prep", "branches", "--data", d, "--laws", "R,Rc", "--budgets",
                                   "300,1000,3000", "--seeds", "0,1,2", "--workers", str(a.workers)], dry=a.dry_run)
        run(f"prep_refs_{m}", [PY, "-m", f"{M}.wm.prep", "refs", "--data", d], dry=a.dry_run)


def stage_g1(a):
    for m in MODES:
        run(f"g1_{m}", [PY, "-m", f"{M}.wm.g1", "--data", str(data_dir(m)), "--setup", SETUP, "--obs-mode", m], dry=a.dry_run)


def stage_arms(a):
    lines = [l.split() for l in TASKS.read_text().splitlines() if l.strip()]
    if a.only:
        lines = [l for l in lines if l[1] in a.only.split(",")]
    q = Queue()
    for l in lines:
        q.put(l)

    def worker(gpu):
        while True:
            try:
                args = q.get_nowait()
            except Exception:
                return
            d = dict(zip(args[::2], args[1::2]))
            tid = f"{d['--arm']}_{d['--obs-mode']}_b{d['--budget']}_s{d['--seed']}"
            if (ROOT / "runs" / SETUP / tid / "_DONE").exists():
                log(f"skip {tid} (done)")
                continue
            for attempt in (1, 2):
                t0 = time.time()
                rc = run(f"{tid}.a{attempt}", [PY, "-m", f"{M}.wm.run_arm", *args, "--setup", SETUP, "--workers", "2"],
                         gpu, a.dry_run)
                log(f"EXIT {tid} gpu={gpu} attempt={attempt} rc={rc} {time.time()-t0:.0f}s")
                if a.dry_run or (rc == 0 and (ROOT / "runs" / SETUP / tid / "_DONE").exists()):
                    break

    ths = [threading.Thread(target=worker, args=(g,)) for g in GPUS]       # ONE job per GPU (11 GB cards OOM with two)
    for t in ths:
        t.start()
    for t in ths:
        t.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["data", "oracle", "prep", "arms", "g1"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default="", help="arms stage: comma list of arms (e.g. N0 or R,Rc)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"data": stage_data, "oracle": stage_oracle, "prep": stage_prep, "arms": stage_arms, "g1": stage_g1}[a.stage](a)
    log(f"stage {a.stage} finished")


if __name__ == "__main__":
    main()
