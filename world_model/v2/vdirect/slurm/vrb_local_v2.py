"""vrb_local_v2.py -- (DEVIATION D1, see vdirect_rb/DEVIATIONS_RB.md: ONE job per GPU + expandable_segments, after
2-per-GPU co-location caused CUDA OOM in finalize) run the PREREG_ADDENDUM_RB grid on westeros GPUs 4-7 (the bimanual share), no Slurm.
SIMULATION / SAVED DATA ONLY: no hardware, no network.

Queue: NOM s6-8 (run_arm.py --arm NOM, unchanged), then the 27 cells of vrb_tasks.txt (run_rb.py); an Rb cell waits
for its seed's NOM. Two workers per GPU. A cell with _DONE is skipped; a failed cell is retried once. When the grid
is finished: conservatism arrays (cb_eval cons, CPU, 16 parallel), then analyze_rb, then the sentinel analysis/_DONE.
Run detached:  setsid nohup $PY world_model/v2/vdirect/slurm/vrb_local.py > $VD_ROOT/logs/local.log 2>&1 &
"""
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/common/users/ss5772/bimanual")
ROOT = Path("/common/users/ss5772/bimanual_data/vdirect_rb")
PY = "/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python"
GPUS = [4, 5, 6, 7]


def log(*a):
    print(time.strftime("%FT%T"), *a, flush=True)


def env_for(tid, gpu):
    tmp = ROOT / "tmp" / tid
    tmp.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, VD_ROOT=str(ROOT), VD_SRC_ROOT=str(ROOT), TMPDIR=str(tmp), MPLCONFIGDIR=str(tmp / "mpl"),
             PYTHONPYCACHEPREFIX=str(tmp / "pyc"), XDG_CACHE_HOME=str(tmp / "cache"), CUDA_CACHE_PATH=str(tmp / "nv"),
             OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
             PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    e["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    return e


def run(tid, cmd, gpu, done):
    if done.exists():
        log(f"skip {tid} (done)"); return 0
    for attempt in (1, 2):
        t0 = time.time()
        with open(ROOT / "logs" / f"{tid}.a{attempt}.log", "w") as f:
            rc = subprocess.call(cmd, cwd=REPO, env=env_for(tid, gpu), stdout=f, stderr=subprocess.STDOUT)
        log(f"EXIT {tid} gpu={gpu} attempt={attempt} rc={rc} {time.time()-t0:.0f}s")
        if rc == 0 and done.exists():
            return 0
    return rc


def main():
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    tasks = [(f"nom_s{s}_m0", [PY, "-m", "world_model.v2.vdirect.run_arm", "--arm", "NOM", "--seed", str(s),
                               "--member", "0"], ROOT / "runs" / f"nom_s{s}_m0" / "_DONE", None) for s in (6, 7, 8)]
    for line in (REPO / "world_model/v2/vdirect/slurm/vrb_tasks.txt").read_text().split("\n"):
        if not line.strip():
            continue
        w = line.split()
        tid = f"{w[1]}_b{w[3]}_s{w[5]}"
        dep = ROOT / "runs" / f"nom_s{w[5]}_m0" / "_DONE" if w[1] == "Rb" else None
        tasks.append((tid, [PY, "-m", "world_model.v2.vdirect.run_rb"] + w, ROOT / "runs" / tid / "_DONE", dep))
    lock = threading.Lock()
    slots = list(GPUS)

    def work(t):
        tid, cmd, done, dep = t
        while dep is not None and not dep.exists():
            time.sleep(30)
        with lock:
            gpu = slots.pop(0)
        try:
            return tid, run(tid, cmd, gpu, done)
        finally:
            with lock:
                slots.append(gpu)

    import sys
    wait = [int(p) for p in sys.argv[1:]]
    while any(os.path.exists(f"/proc/{p}") for p in wait):
        time.sleep(30)
    log(f"[vrb_local_v2] in-flight PIDs {wait} gone; {len(tasks)} tasks on GPUs {GPUS} x1")
    # NOM first, then R/Rc, then Rb (the queue order of the pool = submission order)
    with ThreadPoolExecutor(len(GPUS)) as ex:
        res = list(ex.map(work, tasks))
    bad = [tid for tid, rc in res if rc != 0]
    log(f"[vrb_local] grid finished; failed: {bad}")
    cells = [t for t in tasks if not t[0].startswith("nom")]

    def cons(t):
        tid = t[0]
        out = ROOT / "cb" / "cons" / f"{tid}.npz"
        if not (ROOT / "runs" / tid / "_DONE").exists():
            return tid, 4
        cmd = [PY, "-m", "world_model.v2.vdirect.cb_eval", "cons", "--run", str(ROOT / "runs" / tid), "--out", str(out)]
        return tid, run(f"cons_{tid}", cmd, None, Path(str(out) + "._DONE"))

    (ROOT / "cb" / "cons").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        cres = list(ex.map(cons, cells))
    log(f"[vrb_local] cons finished; failed: {[t for t, rc in cres if rc != 0]}")
    rc = subprocess.call([PY, "-m", "world_model.v2.vdirect.analyze_rb"], cwd=REPO, env=env_for("analyze", None),
                         stdout=open(ROOT / "logs" / "analyze.log", "w"), stderr=subprocess.STDOUT)
    (ROOT / "analysis" / "_DONE").write_text(f"{time.strftime('%FT%TZ', time.gmtime())} analyze_rc={rc} "
                                             f"grid_failed={bad}\n")
    log(f"[vrb_local] analysis rc={rc}; done")


if __name__ == "__main__":
    main()
