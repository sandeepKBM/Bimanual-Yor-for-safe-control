"""vdirect/driver.py -- detached, resumable driver for the kill-test grid. SIMULATION ONLY (local GPUs; no network,
no SSH, no hardware).

  setsid nohup nice -n 10 $PY -m world_model.v2.vdirect.driver > $VD_ROOT/logs/driver.log 2>&1 < /dev/null &

* every cell = one detached process (start_new_session, nice 10, CUDA_VISIBLE_DEVICES=<one GPU>), log in
  $VD_ROOT/logs/<task>.log ending with an 'EXIT <code>' line; the run itself writes <run dir>/_DONE on success;
* cells with _DONE are skipped, so the driver can be killed and restarted at any time (live cells from a previous
  driver are adopted via their pid file);
* GPUs from VD_GPUS (default 4,5,6), at most VD_PER_GPU (default 3,3,2) concurrent cells per GPU (<= 8 CPU workers);
  before each launch the GPU must have >= VD_MIN_FREE_MB free memory (nvidia-smi), else the launch waits;
* priority: SETUP -> NOM -> budgets <= 3k (all arms, seed-major) -> B0 -> 10k -> 30k;
* one automatic retry per failed cell; status in $VD_ROOT/status.json and the AUTO block of PROGRESS.md.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get("VD_ROOT", "/data/local/ss5772/bimanual_runs/vdirect"))
REPO = Path("/common/users/ss5772/bimanual")
ENV_SH = REPO / "world_model/v2/vdirect/env.sh"
GPUS = [int(g) for g in os.environ.get("VD_GPUS", "4,5,6").split(",")]
PER_GPU = [int(x) for x in os.environ.get("VD_PER_GPU", "3,3,2").split(",")]
MIN_FREE = int(os.environ.get("VD_MIN_FREE_MB", "2500"))
SEEDS = [0, 1, 2]
SMOKE = os.environ.get("VD_SMOKE", "0") == "1"


def grid():
    """-> ordered list of (task_id, run_dir, args, deps)."""
    sm = "smoke_" if SMOKE else ""
    T = [("SETUP", ROOT / "shared", ["--arm", "SETUP"], [])]
    setup_done = ROOT / "shared" / "_DONE_SETUP"
    for s in SEEDS:
        for m in range(3):
            T.append((f"NOM_s{s}_m{m}", ROOT / "runs" / f"{sm}nom_s{s}_m{m}",
                      ["--arm", "NOM", "--seed", str(s), "--member", str(m)], ["SETUP"]))
    def cell(arm, b, s):
        deps = ["SETUP"]
        if arm in ("V", "Vn", "D", "B0"):
            deps.append(f"NOM_s{s}_m0")
        if arm == "U":
            deps += [f"NOM_s{s}_m{m}" for m in range(3)]
        rid = f"{arm}_b{b}_s{s}"
        return (rid, ROOT / "runs" / f"{sm}{rid}", ["--arm", arm, "--budget", str(b), "--seed", str(s)], deps)
    low = [1000, 3000] if not SMOKE else [30]
    for b in low:
        for s in SEEDS:
            for arm in ("R", "V", "U", "Vn", "D"):
                T.append(cell(arm, b, s))
    for s in SEEDS:
        T.append(cell("B0", 0, s))
    if not SMOKE:
        for s in SEEDS:
            for arm in ("R", "V", "U", "Vn", "D"):
                T.append(cell(arm, 10000, s))
        for s in SEEDS:
            T.append(cell("R", 30000, s))
    return T, setup_done


def done_path(tid, rdir, setup_done):
    return setup_done if tid == "SETUP" else rdir / "_DONE"


def gpu_free_mb(g):
    if os.environ.get("VD_DRYRUN") == "1":
        return 1e9
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        tot, used = [float(x) for x in out.strip().split(",")]
        return tot - used
    except Exception:
        return -1.0


def alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        st = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return st != "Z"
    except Exception:
        return False


def exit_code(logf):
    try:
        for line in reversed(Path(logf).read_text(errors="ignore").splitlines()[-5:]):
            if line.startswith("EXIT "):
                return int(line.split()[1])
    except Exception:
        pass
    return None


def launch(tid, args, gpu):
    logf = ROOT / "logs" / f"{tid}.log"
    pyargs = " ".join(args)
    cmd = (f"source {ENV_SH}; export VD_ROOT={ROOT}; export CUDA_VISIBLE_DEVICES={gpu}; "
           f"echo \"[driver] start $(date -u +%FT%TZ) gpu={gpu} host=$(hostname)\"; "
           f"$PY -m world_model.v2.vdirect.run_arm {pyargs}{' --smoke' if SMOKE else ''}; rc=$?; "
           f"echo \"[driver] end $(date -u +%FT%TZ)\"; echo \"EXIT $rc\"")
    with open(logf, "a") as f:
        p = subprocess.Popen(["nice", "-n", "10", "bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(REPO))
    return p.pid, logf


def write_progress(state, t0):
    rows = []
    for tid, st in state.items():
        rows.append(f"| {tid} | {st['status']} | {st.get('gpu', '')} | {st.get('pid', '')} | {st.get('tries', 0)} | "
                    f"{st.get('log', '')} |")
    cnt = {}
    for st in state.values():
        cnt[st["status"]] = cnt.get(st["status"], 0) + 1
    block = (f"<!-- AUTO:BEGIN (driver.py, {time.strftime('%FT%TZ', time.gmtime())}) -->\n"
             f"Driver pid {os.getpid()}, up {int((time.time() - t0) / 60)} min. Counts: {cnt}\n\n"
             "| task | status | gpu | pid | tries | log |\n|---|---|---|---|---|---|\n" + "\n".join(rows) +
             "\n<!-- AUTO:END -->")
    p = ROOT / "PROGRESS.md"
    txt = p.read_text() if p.exists() else "# vdirect PROGRESS\n\n"
    if "<!-- AUTO:BEGIN" in txt and "<!-- AUTO:END -->" in txt:
        a = txt.index("<!-- AUTO:BEGIN"); b = txt.index("<!-- AUTO:END -->") + len("<!-- AUTO:END -->")
        txt = txt[:a] + block + txt[b:]
    else:
        txt = txt.rstrip() + "\n\n## Grid status (auto)\n\n" + block + "\n"
    tmp = p.with_suffix(".tmp"); tmp.write_text(txt); tmp.replace(p)
    (ROOT / "status.json").write_text(json.dumps(state, indent=1))


def main():
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    T, setup_done = grid()
    t0 = time.time()
    prev = {}
    if (ROOT / "status.json").exists():
        try:
            prev = json.loads((ROOT / "status.json").read_text())
        except Exception:
            prev = {}
    state = {}
    for tid, rdir, args, deps in T:
        st = dict(status="pending", tries=0, run_dir=str(rdir))
        p = prev.get(tid, {})
        if done_path(tid, rdir, setup_done).exists():
            st["status"] = "done"
        elif p.get("status") == "running" and p.get("pid") and alive(int(p["pid"])):
            st.update(status="running", pid=p["pid"], gpu=p.get("gpu"), log=p.get("log"), tries=p.get("tries", 1))
        elif p.get("status") == "failed":
            st.update(status="failed", tries=p.get("tries", 2), log=p.get("log"))
        state[tid] = st
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("flag", True))
    print(f"[driver] {len(T)} tasks; gpus={GPUS} per_gpu={PER_GPU} root={ROOT}", flush=True)
    while not stop["flag"]:
        # poll running
        for tid, rdir, args, deps in T:
            st = state[tid]
            if st["status"] != "running":
                continue
            if alive(int(st["pid"])):
                continue
            rc = exit_code(st["log"])
            if done_path(tid, rdir, setup_done).exists() and rc in (0, None):
                st["status"] = "done"
                print(f"[driver] {time.strftime('%H:%M:%S')} DONE {tid} rc={rc}", flush=True)
            else:
                st["status"] = "failed" if st["tries"] >= 2 else "pending"
                print(f"[driver] {time.strftime('%H:%M:%S')} FAIL {tid} rc={rc} tries={st['tries']}", flush=True)
        # launch
        used = {g: sum(1 for s in state.values() if s["status"] == "running" and s.get("gpu") == g) for g in GPUS}
        for tid, rdir, args, deps in T:
            st = state[tid]
            if st["status"] != "pending":
                continue
            if any(state[d]["status"] != "done" for d in deps):
                continue
            cand = [g for g, cap in zip(GPUS, PER_GPU) if used[g] < cap]
            cand = sorted(cand, key=lambda g: used[g])
            gsel = None
            for g in cand:
                if gpu_free_mb(g) >= MIN_FREE:
                    gsel = g
                    break
            if gsel is None:
                break
            if os.environ.get("VD_DRYRUN") == "1":
                print(f"[dryrun] would launch {tid} gpu={gsel} args={' '.join(args)} deps={deps}", flush=True)
                st["status"] = "done"
                continue
            pid, logf = launch(tid, args, gsel)
            st.update(status="running", pid=pid, gpu=gsel, log=str(logf), tries=st["tries"] + 1,
                      started=time.strftime("%FT%TZ", time.gmtime()))
            used[gsel] += 1
            print(f"[driver] {time.strftime('%H:%M:%S')} LAUNCH {tid} gpu={gsel} pid={pid}", flush=True)
            time.sleep(20)                     # stagger (data loading, GPU memory ramp)
        if os.environ.get("VD_DRYRUN") != "1":
            write_progress(state, t0)
        if all(s["status"] in ("done", "failed") for s in state.values()):
            break
        time.sleep(30)
    if os.environ.get("VD_DRYRUN") == "1":
        return
    write_progress(state, t0)
    (ROOT / "_DONE_DRIVER").write_text(time.strftime("%FT%TZ", time.gmtime()))
    print("[driver] finished", {k: v["status"] for k, v in state.items() if v["status"] != "done"}, flush=True)


if __name__ == "__main__":
    main()
