"""bridge/launch_local.py -- local (no Slurm) launcher for the bridge experiment. CPU ONLY. SIMULATION / SAVED DATA ONLY.
NOT launched at full scale by the agent that wrote it: the owner runs it once the trained values exist.

  python -m world_model.v2.tray_rep.bridge.launch_local --stage smoke --workers 8          # ~20 carries, smoke1b models, then report
  python -m world_model.v2.tray_rep.bridge.launch_local --stage tune   --dry-run            # print the VAL jobs
  python -m world_model.v2.tray_rep.bridge.launch_local --stage tune   --workers 8 --jobs 4
  python -m world_model.v2.tray_rep.bridge.launch_local --stage select                      # VAL -> chosen_params.json
  python -m world_model.v2.tray_rep.bridge.launch_local --stage test   --workers 8 --jobs 4
  python -m world_model.v2.tray_rep.bridge.launch_local --stage report                      # tables + PNGs
  python -m world_model.v2.tray_rep.bridge.launch_local --stage tune --list-tasks           # write slurm/bridge_tasks_tune.txt
  Run detached:  setsid nohup $PY -m world_model.v2.tray_rep.bridge.launch_local --stage test --workers 8 > $R/bridge/logs/test.log 2>&1 &
Order: tune -> select -> test -> report. Finished jobs are skipped (a job's summary json exists with the right n); a killed job resumes.
Total cores used = --jobs x --workers. Stay at or under the cores you may use (westeros is shared).

Grids (override with flags):
  F1  l_align {0,1,4,16} x l_prox {0,10,40,160,640} (minus 0/0), grid_frac 0.25, d0 0.05 m, theta_ref 0.2 rad
  F2  theta_max {0.05,0.10,0.15,0.20,0.30} rad x a_cap {0.4,0.7,1.0}
  F3  arm {N0, Rc3k} x seed {0,1,2} x tau {-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4}
Seeds in the test stage: F3 0-2 (model seeds); F1 0-2 in trk (tracker noise stream), 0 in noise-free; F0 and F2 use no noisy input: seed 0.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path("/common/users/ss5772/bimanual")
ROOT = Path("/common/users/ss5772/bimanual_data/tray_rep")
OUT = ROOT / "bridge"
PY = "/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python"
M = "world_model.v2.tray_rep.bridge"
SLURM = REPO / "world_model/v2/tray_rep/bridge/slurm"
SETUPS = ("piper", "piper_trk")

F1_LA, F1_LP = (0, 1, 4, 16), (0, 10, 40, 160, 640)
F2_TH, F2_AC = (0.05, 0.10, 0.15, 0.20, 0.30), (0.4, 0.7, 1.0)
TAUS = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
ARMS = ("N0", "Rc3k")


def log(*a):
    print(time.strftime("%FT%T"), *a, flush=True)


def f1_specs(la=F1_LA, lp=F1_LP):
    return [f"F1:l_align={a:g},l_prox={p:g}" for a, p in itertools.product(la, lp) if a or p]


def f2_specs(th=F2_TH, ac=F2_AC):
    return [f"F2:theta_max={t:g},a_cap={c:g}" for t, c in itertools.product(th, ac)]


def f3_specs(arms, taus):
    return [f"F3:arm={a},tau={t:g}" for a, t in itertools.product(arms, taus)]


def job(spec, setup, seed, split, n, tag, extra=None):
    return dict(spec=spec, setup=setup, seed=seed, split=split, n=n, tag=tag, extra=extra or [])


def plan(stage, a):
    """list of jobs for a stage."""
    jobs = []
    seeds_f3 = [int(s) for s in a.seeds.split(",")]
    taus = [float(t) for t in a.taus.split(",")]
    arms = a.arms.split(",")
    if stage in ("tune", "smoke_tune"):
        for setup in a.setups:
            jobs.append(job("F0", setup, 0, "val", a.val_n, a.tag, a.extra))
            for s in f1_specs(a.f1_la, a.f1_lp):
                for sd in ([0, 1, 2] if setup == "piper_trk" and a.f1_seeds_trk else [0]):
                    jobs.append(job(s, setup, sd, "val", a.val_n, a.tag, a.extra))
            for s in f2_specs(a.f2_th, a.f2_ac):
                jobs.append(job(s, setup, 0, "val", a.val_n, a.tag, a.extra))
            for s in f3_specs(arms, taus):
                for sd in seeds_f3:
                    jobs.append(job(s, setup, sd, "val", a.val_n, a.tag, a.extra))
    elif stage in ("test", "smoke_test"):
        for setup in a.setups:
            jobs.append(job("F0", setup, 0, "test", a.test_n, a.tag, a.extra))
            for arm in arms:
                jobs.append(job(f"@F1@{arm}", setup, 0, "test", a.test_n, a.tag, a.extra))
                if setup == "piper_trk":
                    jobs += [job(f"@F1@{arm}", setup, sd, "test", a.test_n, a.tag, a.extra) for sd in (1, 2)]
            jobs.append(job("@F2", setup, 0, "test", a.test_n, a.tag, a.extra))
            for arm in arms:
                for sd in seeds_f3:
                    jobs.append(job(f"@F3@{arm}", setup, sd, "test", a.test_n, a.tag, a.extra))
            if a.f4:
                jobs.append(job(f"F4:check_every={a.f4_every}", setup, 0, "test", a.f4_n, a.tag, a.extra))
    return jobs


def args_of(j, a):
    c = ["--filter", j["spec"], "--setup", j["setup"], "--seed", str(j["seed"]), "--split", j["split"], "--tag", j["tag"],
         "--out", str(a.out)]
    if j["n"]:
        c += ["--n-eps", str(j["n"])]
    return c + list(j["extra"])


def cmd_of(j, workers, a):
    return [PY, "-m", f"{M}.run"] + args_of(j, a) + ["--workers", str(workers)]


def env_for(tag):
    tmp = OUT / "tmp" / tag
    tmp.mkdir(parents=True, exist_ok=True)
    return dict(os.environ, TMPDIR=str(tmp), MPLCONFIGDIR=str(tmp / "mpl"), PYTHONPYCACHEPREFIX=str(tmp / "pyc"),
                XDG_CACHE_HOME=str(tmp / "cache"), NUMBA_CACHE_DIR=str(tmp / "numba"), OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="")           # CPU only, always


def is_done(j, a):
    """summary exists and covers the requested n."""
    sys.path.insert(0, str(REPO))
    from world_model.v2.tray_rep.bridge import run as R
    from world_model.v2.tray_rep.store import Store
    try:
        spec = R.resolve_spec(j["spec"], a.out, j["tag"], j["setup"])
    except (FileNotFoundError, KeyError):
        return False
    _, _, ps = R.paths(a.out, j["tag"], j["split"], j["setup"], spec, j["seed"])
    if not ps.exists():
        return False
    s = json.loads(ps.read_text())
    want = j["n"] or Store(ROOT / f"data_{j['setup']}_full", j["split"]).n_eps
    return s.get("n", 0) >= want


def run_jobs(jobs, a):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    q = Queue()
    for j in jobs:
        q.put(j)
    fails = []
    lock = threading.Lock()

    def worker(wid):
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
                log("skip (done)", j["spec"], j["setup"], j["seed"], j["split"])
                continue
            lf = OUT / "logs" / f"bridge_{j['tag']}.log"
            with lock:
                log("RUN ", " ".join(c[2:]))
            with open(lf, "a") as f:
                rc = subprocess.call(c, cwd=REPO, env=env_for(j["tag"]), stdout=f, stderr=subprocess.STDOUT)
            if rc != 0:
                with lock:
                    fails.append((j, rc))
                    log("FAIL", rc, j["spec"], j["setup"], j["seed"])

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(a.jobs)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    return fails


def report(a, split="test", tag=None):
    for sub in ("table", "plots"):
        c = [PY, "-m", f"{M}.report", sub, "--tag", tag or a.tag, "--split", split, "--out", str(a.out)]
        log("RUN ", " ".join(c[2:]))
        subprocess.call(c, cwd=REPO, env=env_for(tag or a.tag))


def smoke(a):
    """~20 TEST carries per filter, both setups, smoke1b models (tiny, untrained: numbers are meaningless), then the checks."""
    S = ROOT / "smoke1b" / "runs"
    a.tag, a.runs = "smoke", str(S)
    a.extra = ["--runs-root", str(S), "--model-setup", "piper"]
    a.arms, a.seeds, a.taus = "N0,Rc50", "0", "-0.1,0.0,0.2"
    a.f1_la, a.f1_lp, a.f2_th, a.f2_ac = (1, 4), (0, 40), (0.1, 0.2), (0.7, 1.0)
    a.f1_seeds_trk = False
    a.val_n, a.test_n, a.f4, a.f4_every, a.f4_n = 10, 20, True, 5, 20
    (OUT / "smoke").mkdir(parents=True, exist_ok=True)
    fails = run_jobs(plan("tune", a), a)
    c = [PY, "-m", f"{M}.report", "select", "--tag", "smoke", "--out", str(a.out)]
    log("RUN ", " ".join(c[2:]))
    subprocess.call(c, cwd=REPO, env=env_for("smoke"))
    fails += run_jobs(plan("test", a), a)
    report(a, "test", "smoke")
    report(a, "val", "smoke")
    return fails


def write_tasks(jobs, name, a):
    SLURM.mkdir(parents=True, exist_ok=True)
    p = SLURM / f"bridge_tasks_{name}.txt"
    lines = []
    for j in jobs:
        lines.append(" ".join(args_of(j, a)))            # no spaces inside any argument
    p.write_text("\n".join(lines) + "\n")
    log(f"wrote {p} ({len(lines)} tasks). sbatch --array=0-{len(lines) - 1}%12 {SLURM}/bridge_array.sbatch {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["smoke", "tune", "select", "test", "report"])
    ap.add_argument("--workers", type=int, default=4, help="processes per job (cores per job)")
    ap.add_argument("--jobs", type=int, default=1, help="jobs in parallel (total cores = jobs x workers)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--setups", default=",".join(SETUPS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2", help="model seeds for F3")
    ap.add_argument("--taus", default=",".join(f"{t:g}" for t in TAUS))
    ap.add_argument("--val-n", type=int, default=200, help="VAL carries per tuning run (the VAL split has 600)")
    ap.add_argument("--test-n", type=int, default=0, help="TEST carries (0 = all 600)")
    ap.add_argument("--f4", action="store_true", help="also run F4 (oracle look-ahead) in the test stage: about 45 CPU-s per carry")
    ap.add_argument("--f4-n", type=int, default=100)
    ap.add_argument("--f4-every", type=int, default=5)
    ap.add_argument("--no-f1-seeds-trk", dest="f1_seeds_trk", action="store_false", help="tune F1 on tracker seed 0 only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-tasks", action="store_true", help="write the Slurm task file instead of running")
    a = ap.parse_args()
    a.out = Path(a.out)
    a.setups = a.setups.split(",")
    a.f1_la, a.f1_lp, a.f2_th, a.f2_ac, a.extra = F1_LA, F1_LP, F2_TH, F2_AC, []
    if a.stage == "smoke":
        fails = smoke(a)
    elif a.stage == "select":
        c = [PY, "-m", f"{M}.report", "select", "--tag", a.tag, "--out", str(a.out)]
        log("RUN ", " ".join(c[2:]))
        sys.exit(subprocess.call(c, cwd=REPO, env=env_for(a.tag)))
    elif a.stage == "report":
        report(a, "val")
        report(a, "test")
        return
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
