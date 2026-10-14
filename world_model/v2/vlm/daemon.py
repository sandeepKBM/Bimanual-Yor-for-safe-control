"""vlm/daemon.py -- long-lived VLM service (runs INSIDE a Slurm job on ilab3, `vlm` env). Model loaded ONCE; requests via an NFS file
queue. No sockets, no network at inference (HF_HUB_OFFLINE=1, node-local weights).

    <SVC>/req/<id>.json     request, same schema as runner.py (id, images, text, system, max_new_tokens, temperature, n, top_p).
                            Clients write <id>.json.tmp then os.replace -> only complete files are picked up.
    <SVC>/resp/<id>.json    runner.run_one output (+ "model"), or {"id", "error"} on a per-request failure (the daemon keeps running)
    <SVC>/heartbeat.json    {model, host, job, pid, t, uptime_s, n_done, busy, last_seconds}  rewritten every HEARTBEAT_S
    <SVC>/STOP              sentinel: finish the current request and exit cleanly
Client side: world_model/v2/vlm/service.py.  Launch: world_model/v2/vlm/slurm/daemon.sbatch.
"""
import argparse, json, os, socket, sys, time, traceback
from pathlib import Path

from world_model.v2.vlm.runner import load, run_one

SVC = Path("/common/users/ss5772/bimanual_data/vlm/svc")
HEARTBEAT_S = 5.0
POLL_S = 0.2


def _atomic_write(p: Path, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--svc", default=str(SVC))
    a = ap.parse_args()
    svc = Path(a.svc)
    for d in ("req", "resp", "img"):
        (svc / d).mkdir(parents=True, exist_ok=True)
    stop = svc / "STOP"
    if stop.exists():
        stop.unlink()
    t0 = time.time()
    proc, model, mid = load(a.model)
    print(f"[daemon] loaded {mid} in {time.time() - t0:.1f}s on {socket.gethostname()}", flush=True)
    st = dict(model=a.model, model_id=mid, host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"), pid=os.getpid(),
              t_start=time.time(), n_done=0, n_err=0, busy=None, last_seconds=None)
    last_hb = 0.0

    def beat(force=False):
        nonlocal last_hb
        now = time.time()
        if force or now - last_hb >= HEARTBEAT_S:
            _atomic_write(svc / "heartbeat.json", dict(st, t=now, uptime_s=round(now - st["t_start"], 1)))
            last_hb = now

    beat(force=True)
    while not stop.exists():
        reqs = sorted((p for p in (svc / "req").glob("*.json")), key=lambda p: p.stat().st_mtime)
        if not reqs:
            beat()
            time.sleep(POLL_S)
            continue
        p = reqs[0]
        try:
            r = json.loads(p.read_text())
        except Exception:                       # half-visible on NFS: retry next poll
            time.sleep(POLL_S)
            continue
        rid = str(r.get("id", p.stem))
        st["busy"] = rid
        beat(force=True)
        try:
            o = run_one(proc, model, r, a.model)
            o["model"] = a.model
            st["n_done"] += 1
            st["last_seconds"] = o["seconds"]
            print(f"[daemon] {rid} {o['seconds']}s new={o['n_new_tokens']} vram={o['peak_vram_gb']}GB", flush=True)
        except Exception:
            o = dict(id=rid, error=traceback.format_exc()[-4000:], model=a.model)
            st["n_err"] += 1
            print(f"[daemon] {rid} ERROR", flush=True)
        _atomic_write(svc / "resp" / f"{rid}.json", o)
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        st["busy"] = None
        beat(force=True)
    st["stopped"] = True
    beat(force=True)
    print("[daemon] STOP sentinel: exiting", flush=True)


if __name__ == "__main__":
    sys.exit(main())
