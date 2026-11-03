"""pan/cert/run_trials.py -- run admit.world_trials on a list of worlds in parallel (one JSON per world).  SIMULATION ONLY, CPU.

    python -m world_model.v2.nero.pan.cert.run_trials --out DIR --set full --tags I03,V10 --family 0.06,0.03,0.01 \
        [--perturb-lid 1e-6] --procs P
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

PLAN = dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")


def _job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    tag, world, est, kw = job
    t0 = time.time()
    try:
        from .. import segments as S
        from . import admit as AD
        ep = S.Episode(world, observe="zed", obs_seed=S.world_seed(tag), est=est, **PLAN)
        ep.reset_track()
        r = AD.world_trials(ep, **kw)
        r.update(tag=tag, world=world, est=est, obs=ep.obs)
    except Exception:
        r = dict(tag=tag, world=world, est=est, error=traceback.format_exc()[-1500:])
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def run(jobs, out, procs):
    out = Path(out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in (out / "raw").glob("*.json")}
    jobs = [j for j in jobs if j[0] not in done]
    with get_context("spawn").Pool(procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, jobs):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            segs = [(s["seg"], s["nominal"]["status"], {k: v["n_bad"] for k, v in s["family"].items()}) for s in r.get("segments", [])]
            print(r["tag"], r.get("refused") and r["refused"][:50], segs, r.get("error", "")[-300:], r["wall_s"], flush=True)


def main(argv=None):
    from ..segcert import worlds
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="full", choices=["full", "iid", "overload"])
    ap.add_argument("--tags", default=None)
    ap.add_argument("--family", default="0.06,0.04,0.03,0.02,0.01")
    ap.add_argument("--perturb-lid", type=float, default=0.0)
    ap.add_argument("--procs", type=int, required=True)
    a = ap.parse_args(argv)
    ws = worlds(a.set)
    if a.tags:
        keep = a.tags.split(",")
        ws = [(t, w) for t, w in ws if t in keep]
    kw = dict(family=tuple(float(x) for x in a.family.split(",")), perturb_lid=a.perturb_lid)
    run([(t, w, None, kw) for t, w in ws], a.out, a.procs)
    (Path(a.out) / f"_DONE_{a.set}").write_text("ok")


if __name__ == "__main__":
    main()
