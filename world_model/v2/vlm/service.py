"""vlm/service.py -- workstation-side client of the long-lived VLM daemon (vlm/daemon.py on a cluster GPU node). stdlib only; file queue on NFS.

    from world_model.v2.vlm import service as VS
    VS.status()                      # heartbeat dict + 'age_s' + 'up'
    out = VS.ask("Describe the scene.", images=["/path/a.png"], max_new_tokens=256)   # -> runner output dict ('texts', 'seconds', ...)

Images are copied into <SVC>/img/ (NFS, readable on the daemon node). Start / stop the daemon with:
    ssh <slurm-login> "cd <repo> && sbatch world_model/v2/vlm/slurm/daemon.sbatch"     # MODEL=... via --export
    python -m world_model.v2.vlm.service stop
APPROVAL SCOPE (user, 2026-10-12): 1 cluster GPU via Slurm, offline weights, no hosted APIs, no robot.
"""
import json, os, shutil, sys, time, uuid
from pathlib import Path

SVC = Path(os.environ.get("BIMANUAL_VLM_SVC", "data/bimanual_data/vlm/svc"))
UP_AGE_S = 30.0


def status():
    hb = SVC / "heartbeat.json"
    if not hb.exists():
        return dict(up=False, reason="no heartbeat")
    try:
        d = json.loads(hb.read_text())
    except Exception as e:
        return dict(up=False, reason=f"unreadable heartbeat: {e}")
    d["age_s"] = round(time.time() - d.get("t", 0), 1)
    d["up"] = d["age_s"] < UP_AGE_S and not d.get("stopped")
    return d


def up():
    return status()["up"]


def ask(text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1, timeout_s=180.0, poll_s=0.1, rid=None):
    """Submit one request and block until the response file appears. Raises TimeoutError / RuntimeError."""
    rid = rid or f"r{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    (SVC / "img").mkdir(parents=True, exist_ok=True)
    imgs = []
    for k, p in enumerate(images):
        dst = SVC / "img" / f"{rid}_{k}{Path(p).suffix or '.png'}"
        shutil.copyfile(p, dst)
        imgs.append(str(dst))
    req = dict(id=rid, images=imgs, text=text, system=system, max_new_tokens=max_new_tokens, temperature=temperature, n=n)
    q = SVC / "req" / f"{rid}.json"
    q.parent.mkdir(parents=True, exist_ok=True)
    tmp = q.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(req))
    os.replace(tmp, q)
    resp = SVC / "resp" / f"{rid}.json"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        # list the directory instead of stat-ing the expected name: NFS caches the negative lookup of resp/<id>.json until the parent's
        # attribute cache expires (acdirmin, ~30 s here), so a bare exists() saw responses ~30 s late; opendir revalidates the directory.
        if f"{rid}.json" in os.listdir(resp.parent):
            try:
                o = json.loads(resp.read_text())
            except Exception:
                time.sleep(poll_s)
                continue
            resp.unlink(missing_ok=True)
            for p in imgs:
                Path(p).unlink(missing_ok=True)
            if "error" in o:
                raise RuntimeError(o["error"])
            o["wall_s"] = round(time.time() - t0, 3)
            return o
        time.sleep(poll_s)
    raise TimeoutError(f"VLM request {rid} not answered in {timeout_s}s (status: {status()})")


def stop():
    (SVC / "STOP").write_text(str(time.time()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps(status(), indent=1))
    elif cmd == "stop":
        stop(); print("STOP sentinel written")
    elif cmd == "ask":
        print(json.dumps(ask(" ".join(sys.argv[2:]) or "Say OK."), indent=1))
