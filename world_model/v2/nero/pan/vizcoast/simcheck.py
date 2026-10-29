"""Stage B: falsification search for an admitted plan -- run it in sim (pan/under_rim.py) at the 2^5 vertices of its belief box
(pose dx, dy, yaw; friction; hidden mass at the far centre).  SIMULATION ONLY, CPU only (no rendering).

A vertex search is not a proof (interior failures can exist); it finds counterexamples, which join the evidence store.
Stops early (pool terminated) once `stop_after` failures are seen.  Results are appended to a JSONL evidence file.
"""
from __future__ import annotations

import json
import zlib
import os
import traceback
from multiprocessing import get_context
from pathlib import Path

from . import domain as D
from . import evidence as E


def _kwargs(plan, th):
    sig = plan.signature()
    return dict(placement=th["placement"], hidden_mass=th["hidden_mass"] if th["hidden_mass"] > 0 else None, push=sig[2],
                grasp_frac=sig[1], grip_force=sig[3],
                perturb=dict(dx=th["dx"], dy=th["dy"], yaw_deg=th["yaw_deg"], mu_scale=th["mu_scale"]),
                payload_mass="true" if sig[4] == "known" else None,
                observe=dict(noise=E.obs_tag(sig)[4:], retries=3, seed=zlib.crc32(json.dumps(th, sort_keys=True).encode()))
                if E.obs_tag(sig) else None)


def _one(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("MUJOCO_GL", "osmesa")             # built observe_pose renders depth on the CPU
    from .. import under_rim as U
    tag, kw = job
    try:
        r = U.run(**kw)
        r.pop("log", None)
    except Exception:
        r = dict(kw, error=traceback.format_exc()[-600:])
    r["tag"] = tag
    return r


def _order(vs):
    """hardest-looking vertices first (large yaw/dy toward the claw, high friction, heavy) so early stop bites sooner."""
    return sorted(vs, key=lambda t: -(t["dy"] > 0) - (t["mu_scale"] > 1.0) - (t["hidden_mass"] > 0) - 0.5 * (t["yaw_deg"] > 0))


def make_sim(out_jsonl, procs=8, stop_after=2, raw_dir=None):
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def sim(plan):
        box = plan.aim_belief()
        vs = _order(box.vertices())
        rid = f"r{sum(1 for _ in open(out_jsonl)) if out_jsonl.exists() else 0:04d}"
        jobs = [(f"{rid}_v{i:02d}", _kwargs(plan, th)) for i, th in enumerate(vs)]
        sig = plan.signature()
        new, nfail = [], 0
        ctx = get_context("spawn")
        with ctx.Pool(procs, maxtasksperchild=2) as pool:
            for r in pool.imap_unordered(_one, jobs):
                th = E._theta(r)
                e = dict(sig=sig, theta=th, held=bool(r.get("held", False)), cat=E.category(r),
                         src=f"{out_jsonl.name}:{r['tag']}", plan=str(plan))
                new.append(e)
                with open(out_jsonl, "a") as f:
                    f.write(json.dumps(e, default=float) + "\n")
                if raw_dir:
                    Path(raw_dir).mkdir(parents=True, exist_ok=True)
                    (Path(raw_dir) / f"{r['tag']}.json").write_text(json.dumps(r, indent=1, default=float))
                nfail += not e["held"]
                if stop_after and nfail >= stop_after:
                    pool.terminate()
                    break
        return new

    return sim
