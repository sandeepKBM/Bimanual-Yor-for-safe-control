"""vlm/client.py -- westeros-side client (stdlib only). Writes request JSONL on NFS, submits ONE Slurm batch via `ssh rlab1 sbatch`,
waits for the sentinel file (<out>.DONE / <out>.FAIL) with a sleep loop (no pgrep), returns parsed outputs.

    from world_model.v2.vlm.client import run_batch
    outs = run_batch("myname", [{"id": "a", "images": [...], "text": "..."}, ...])   # dict id -> output record

CLI: python -m world_model.v2.vlm.client submit <req.jsonl> [--model qwen3.5-9b] [--no-wait]
APPROVAL SCOPE: ssh rlab1 / sbatch to ilab3 (1 GPU) only. No hosted APIs.
"""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = "/common/users/ss5772/bimanual"
DATA = Path("/common/users/ss5772/bimanual_data/vlm")
SBATCH = f"{REPO}/world_model/v2/vlm/slurm/run_batch.sbatch"


def submit(req_path, out_path, model="qwen3.5-9b", time_limit="02:00:00"):
    for suf in (".DONE", ".FAIL"):
        p = Path(str(out_path) + suf)
        if p.exists():
            p.unlink()
    cmd = (f"cd {REPO} && sbatch --parsable --time={time_limit} --export=ALL,REQ={req_path},OUT={out_path},MODEL={model} {SBATCH}")
    r = subprocess.run(["ssh", "rlab1", cmd], capture_output=True, text=True, timeout=120)
    jid = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if r.returncode or not jid.split(";")[0].isdigit():
        raise RuntimeError(f"sbatch failed: {r.stdout} {r.stderr}")
    return jid.split(";")[0]


def wait(out_path, timeout_s=7200, poll_s=10):
    done, fail = Path(str(out_path) + ".DONE"), Path(str(out_path) + ".FAIL")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if done.exists():
            return
        if fail.exists():
            raise RuntimeError("runner failed:\n" + fail.read_text())
        time.sleep(poll_s)
    raise TimeoutError(str(out_path))


def read_out(out_path):
    return {o["id"]: o for o in (json.loads(l) for l in open(out_path) if l.strip())}


def run_batch(name, requests, model="qwen3.5-9b", wait_s=7200, time_limit="02:00:00"):
    (DATA / "req").mkdir(parents=True, exist_ok=True)
    (DATA / "out").mkdir(parents=True, exist_ok=True)
    req, out = DATA / "req" / f"{name}.jsonl", DATA / "out" / f"{name}.jsonl"
    req.write_text("".join(json.dumps(r) + "\n" for r in requests))
    jid = submit(req, out, model, time_limit)
    print(f"submitted job {jid} ({len(requests)} requests); waiting for {out}.DONE", flush=True)
    wait(out, wait_s)
    return read_out(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit"])
    ap.add_argument("req")
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    out = DATA / "out" / (Path(a.req).stem + ".jsonl")
    jid = submit(Path(a.req).resolve(), out, a.model)
    print("job", jid, "->", out)
    if not a.no_wait:
        wait(out)
        print("done")
