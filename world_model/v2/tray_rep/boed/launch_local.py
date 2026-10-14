"""boed/launch_local.py -- local (no Slurm) launcher for the online-BOED kill test. CPU ONLY. SIMULATION / SAVED DATA ONLY.
NOT run at full scale by the agent that wrote it: the owner runs it.

Stages (in this order):
  check     probe / belief checks: analytic-vs-MuJoCo gap, posterior coverage on random mu_roll (both setups)        [minutes]
  table     VAL cells "FIX:I,j": F3-filtered carry at intensity I, mu forced to node j  -> predicted-drop table      [the big VAL job]
  assemble  cells -> <out>/<tag>/table_<setup>.json (+ .md), pooled over the value seeds found
  test      P1..P6 (or --policies P7,P7s: --out <new dir> --table-dir <frozen table dir>) on the TEST carries, both setups, value seeds 0-2
  report    tables, belief calibration, paired-bootstrap criteria
  smoke     everything tiny (20 VAL carries per policy, 4 mu nodes, 5 carries per cell), tag "smoke"

  python -m world_model.v2.tray_rep.boed.launch_local --stage table --dry-run
  python -m world_model.v2.tray_rep.boed.launch_local --stage table --workers 4 --jobs 1
  python -m world_model.v2.tray_rep.boed.launch_local --stage test  --list-tasks --shards 4      # Slurm task file
Total cores = jobs x workers. Finished jobs are skipped (summary json exists with the right n); a killed job resumes.
Run detached:  setsid nohup $PY -m world_model.v2.tray_rep.boed.launch_local --stage test --workers 4 > $R/boed/logs/test.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np

REPO = Path("/common/users/ss5772/bimanual")
ROOT = Path("/common/users/ss5772/bimanual_data/tray_rep")
OUT = ROOT / "boed"
PY = "/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python"
M = "world_model.v2.tray_rep.boed"
SLURM = REPO / "world_model/v2/tray_rep/boed/slurm"
SETUPS = ("piper", "piper_trk")
POLICIES = ("P1", "P2", "P6", "P3", "P4", "P5")


def log(*a):
    print(time.strftime("%FT%T"), *a, flush=True)


def igrid_arg(a):
    return tuple(float(x) for x in a.igrid.split(","))


def plan(stage, a):
    jobs = []
    if stage == "table":
        for setup in a.setups:
            for vs in a.seeds:
                for I in igrid_arg(a):
                    for j in range(a.n_nodes):
                        jobs.append(dict(policy=f"FIX:I={I:g},j={j}", setup=setup, seed=vs, split="val", n=a.table_n, shard="0/1"))
    elif stage == "test":
        for setup in a.setups:
            for vs in a.seeds:
                for p in a.policies:
                    for s in range(a.shards):
                        jobs.append(dict(policy=p, setup=setup, seed=vs, split=a.split, n=a.test_n or None, shard=f"{s}/{a.shards}"))
    return jobs


def args_of(j, a):
    c = ["--policy", j["policy"], "--setup", j["setup"], "--seed", str(j["seed"]), "--split", j["split"], "--tag", a.tag, "--out", str(a.out),
         "--n-nodes", str(a.n_nodes), "--eps", str(a.eps), "--shard", j["shard"]]
    if j["n"]:
        c += ["--n-eps", str(j["n"])]
    return c + list(a.extra)


def cmd_of(j, workers, a):
    return [PY, "-m", f"{M}.run"] + args_of(j, a) + ["--workers", str(workers)]


def env_for(tag, out=None):
    tmp = Path(out or OUT) / "tmp" / tag
    tmp.mkdir(parents=True, exist_ok=True)
    return dict(os.environ, TMPDIR=str(tmp), MPLCONFIGDIR=str(tmp / "mpl"), PYTHONPYCACHEPREFIX=str(tmp / "pyc"), XDG_CACHE_HOME=str(tmp / "cache"),
                NUMBA_CACHE_DIR=str(tmp / "numba"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                PYTHONUNBUFFERED="1", PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="")


def is_done(j, a):
    sys.path.insert(0, str(REPO))
    from world_model.v2.tray_rep.boed import run as R
    _, _, ps = R.run_paths(a.out, a.tag, j["split"], j["setup"], j["policy"], j["seed"], j["shard"])
    if not ps.exists():
        return False
    s = json.loads(ps.read_text())
    from world_model.v2.tray_rep.store import Store
    si, sn = (int(x) for x in j["shard"].split("/"))
    n_all = j["n"] or Store(ROOT / f"data_{j['setup']}_full", j["split"]).n_eps
    want = len(range(n_all)[si::sn])
    return s.get("n", 0) >= want


def run_jobs(jobs, a):
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    q = Queue()
    for j in jobs:
        q.put(j)
    fails, lock = [], threading.Lock()

    def worker():
        while True:
            try:
                j = q.get_nowait()
            except Exception:
                return
            c = cmd_of(j, a.workers, a)
            if a.dry_run:
                log("DRY", " ".join(c[2:]))
                continue
            if is_done(j, a):
                log("skip (done)", j["policy"], j["setup"], j["seed"], j["shard"])
                continue
            with lock:
                log("RUN ", " ".join(c[2:]))
            with open(a.out / "logs" / f"boed_{a.tag}.log", "a") as f:
                rc = subprocess.call(c, cwd=REPO, env=env_for(a.tag, a.out), stdout=f, stderr=subprocess.STDOUT)
            if rc != 0:
                with lock:
                    fails.append((j, rc))
                    log("FAIL", rc, j["policy"], j["setup"], j["seed"])

    ths = [threading.Thread(target=worker) for _ in range(a.jobs)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    return fails


def sub(mod_cmd, a, extra=()):
    c = [PY, "-m", f"{M}.{mod_cmd[0]}"] + list(mod_cmd[1:]) + list(extra)
    log("RUN ", " ".join(c[2:]))
    return subprocess.call(c, cwd=REPO, env=env_for(a.tag, a.out))


def assemble(a):
    rc = 0
    for setup in a.setups:
        rc |= sub(["table", "assemble", "--setup", setup, "--tag", a.tag, "--out", str(a.out), "--igrid", a.igrid, "--n-nodes", str(a.n_nodes)], a)
    return rc


def report(a, split):
    return sub(["report", "all", "--tag", a.tag, "--split", split, "--out", str(a.out)], a)


def check(a):
    B = OUT / a.tag
    B.mkdir(parents=True, exist_ok=True)
    rc = sub(["check_model", "gap", "--n", "4", "--out", str(B / "check_gap.json")], a)
    for setup in a.setups:
        rc |= sub(["check_model", "posterior", "--setup", setup, "--n", str(a.check_n), "--out", str(B / f"check_posterior_{setup}.json")], a)
        rc |= sub(["check_model", "battery", "--setup", setup, "--n", str(max(a.check_n, 100)), "--out", str(B / f"check_battery_{setup}.json")], a)
    return rc


def smoke(a):
    """~20 VAL carries per policy, both setups, tiny table. Numbers are for plumbing, not evidence."""
    a.tag, a.n_nodes, a.table_n, a.test_n, a.split, a.seeds, a.shards, a.check_n = "smoke", 4, 5, 20, "val", [0], 1, 40
    fails = []
    if check(a):
        fails.append("check")
    fails += run_jobs(plan("table", a), a)
    if assemble(a):
        fails.append("assemble")
    fails += run_jobs(plan("test", a), a)
    report(a, "val")
    return fails


def write_tasks(jobs, name, a):
    SLURM.mkdir(parents=True, exist_ok=True)
    p = SLURM / f"boed_tasks_{name}.txt"
    lines = [" ".join(f'"{x}"' if " " in x else x for x in args_of(j, a)) for j in jobs]
    p.write_text("\n".join(lines) + "\n")
    log(f"wrote {p} ({len(lines)} tasks). sbatch --array=0-{len(lines) - 1}%12 {SLURM}/boed_array.sbatch {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["smoke", "check", "table", "assemble", "test", "report"])
    ap.add_argument("--workers", type=int, default=4, help="processes per job")
    ap.add_argument("--jobs", type=int, default=1, help="jobs in parallel (cores = jobs x workers)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--setups", default=",".join(SETUPS))
    ap.add_argument("--seeds", default="0,1,2", help="value seeds")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--igrid", default="1,0.75,0.55,0.4")
    ap.add_argument("--n-nodes", type=int, default=8)
    ap.add_argument("--table-n", type=int, default=40, help="VAL carries per (setup, seed, I, mu node) cell")
    ap.add_argument("--test-n", type=int, default=0, help="TEST carries (0 = all 600)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--shards", type=int, default=1, help="split each test run into this many pieces (Slurm array)")
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--check-n", type=int, default=60)
    ap.add_argument("--table-dir", default="", help="folder holding table_<setup>.json (e.g. the frozen .../boed/main); passed to run.py")
    ap.add_argument("--p7-opts", default="", help="P7 certificate settings k=v,... (see run.py)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-tasks", action="store_true")
    a = ap.parse_args()
    a.out, a.setups, a.policies = Path(a.out), a.setups.split(","), a.policies.split(",")
    a.seeds = [int(s) for s in str(a.seeds).split(",")] if not isinstance(a.seeds, list) else a.seeds
    a.extra = (["--table-dir", a.table_dir] if a.table_dir else []) + (["--p7-opts", a.p7_opts] if a.p7_opts else [])
    if a.stage == "smoke":
        fails = smoke(a)
    elif a.stage == "check":
        sys.exit(check(a))
    elif a.stage == "assemble":
        sys.exit(assemble(a))
    elif a.stage == "report":
        sys.exit(report(a, a.split))
    else:
        jobs = plan(a.stage, a)
        log(f"stage {a.stage}: {len(jobs)} jobs, {a.jobs} x {a.workers} cores")
        if a.list_tasks:
            write_tasks(jobs, a.stage, a)
            return
        fails = run_jobs(jobs, a)
    log(f"done, {len(fails)} failed jobs")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
