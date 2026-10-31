"""Run the faithful VIZ-COAST baseline on the covered pan, then EXECUTE its grounded plans in sim.  SIMULATION ONLY, CPU only.

    python -m world_model.v2.nero.pan.vizcoast_faithful.run_baseline --out DIR [--profiles lid two_arm none buggy | --llm SPEC]
           [--seeds 0 1 2 3 4] [--plans-from planning.json] [--execute N --worlds setA|obsbox|full [--observe NOISE] --procs P]

Planning: one VIZ-COAST run per (VRM profile, stream seed).  Execution: each distinct grounded plan is run open loop (plans
from the nominal scene, as VIZ-COAST assumes a known world) on the first N theta of the pre-registered robustness set
(results/pan_handle_grasp_2026-10-28/robust_L70/cases.json, Set A: pose +-1.5 cm, yaw +-8 deg, mu x0.6-1.2, 0-0.4 kg).
Executors: push+clamp+lift-two -> pan/under_rim.run; left one-arm -> pan/handle_grasp.run (nominal only, no pose error support);
right one-arm -> no executor (reported).
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")   # before ANY mujoco import: spawned workers re-import this module first (10-29 bug)

import argparse
import json
import traceback
from multiprocessing import get_context
from pathlib import Path

from ...config import REPO
from . import coast as CO
from . import streams as ST
from . import vrm as V

CASES = Path(REPO) / "world_model/v2/nero/results/pan_handle_grasp_2026-10-28/robust_L70/cases.json"


def skeleton(plan):
    names = [a.split()[0].strip("(") for a in plan]
    if "clamp" in names and "lift-two" in names:
        return "push_clamp_lift_two"
    if names[-1] == "lift-one":
        return "one_arm_" + plan[-1].split()[1]
    if names[-1] == "lift-two":
        return "two_claw_handle"
    return "other"


def _exec(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("MUJOCO_GL", "osmesa")             # observe_pose renders depth on the CPU
    tag, kind, kw = job
    try:
        if kind == "under_rim":
            from .. import under_rim as U
            r = U.run(**kw)
        else:
            from .. import handle_grasp as H
            r = H.run(**kw)
        r.pop("log", None)
    except Exception:
        r = dict(kw, error=traceback.format_exc()[-600:])
    r["tag"] = tag
    return r


def worlds(kind, n):
    """execution worlds.  setA: first n of the robust_L70 Set A (full +-1.5 cm / 8 deg box; the 10-28 run).
    obsbox: the box OUR layer certified after observe_pose (+-5 mm / 2 deg, same mu/mass): its 32 vertices + the 30 interior
    worlds of the 10-28 interior check (same RNG sequence).  full: the full prereg box: 32 vertices + 30 interior (observe_eval)."""
    if kind == "setA":
        return [(c["tag"], dict(placement=c["placement"], hidden_mass=c["hidden_mass"], perturb=c["perturb"]))
                for c in json.load(open(CASES)) if c["tag"].startswith("A")][:n]
    import numpy as np
    from ..vizcoast import domain as D
    out = []
    if kind == "obsbox":
        box = D.Belief(dxy=D.OBS_DXY, yaw=D.OBS_YAW)
        rng = np.random.default_rng(1)                         # = scratch vc_interior.py (10-28 interior check of our plan)
        PL = ["none", "near_handle", "far_centre", "far_left", "far_right"]
        inter = []
        for i in range(30):
            pl = PL[rng.integers(5)]
            hm = float(rng.uniform(0, 0.4)) if pl != "none" else None
            inter.append(dict(placement=pl, hidden_mass=hm, perturb=dict(dx=float(rng.uniform(-.005, .005)), dy=float(rng.uniform(-.005, .005)),
                         yaw_deg=float(rng.uniform(-2, 2)), mu_scale=float(rng.uniform(0.6, 1.2)))))
    elif kind == "full":
        from ..observe_eval import _random_thetas
        box = D.PREREG
        inter = [dict(placement=t["placement"], hidden_mass=t["hidden_mass"] or None,
                      perturb={k: t[k] for k in ("dx", "dy", "yaw_deg", "mu_scale")}) for t in _random_thetas(30, 1)]
    else:
        raise ValueError(kind)
    for i, t in enumerate(box.vertices()):
        out.append((f"V{i:02d}", dict(placement=t["placement"], hidden_mass=t["hidden_mass"] or None,
                                      perturb={k: t[k] for k in ("dx", "dy", "yaw_deg", "mu_scale")})))
    out += [(f"I{i:02d}", w) for i, w in enumerate(inter)]
    return out[:n] if n else out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiles", nargs="+", default=["lid", "two_arm", "none", "buggy"])
    ap.add_argument("--llm", default=None, help="pan/llm spec (e.g. codex:gpt-5.6-sol:high): replaces the scripted profiles")
    ap.add_argument("--image", default=None, help="scene image given to the VRM (the paper's VLM sees one)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--plans-from", default=None, help="reuse planning.json from an earlier run (execute only)")
    ap.add_argument("--execute", type=int, default=0, help="number of worlds (0 = none; with --worlds obsbox/full: 0 = none, -1 = all)")
    ap.add_argument("--worlds", default="setA", choices=["setA", "obsbox", "full"])
    ap.add_argument("--observe", default=None, help="observe_pose noise level: execute with the BUILT observe_pose (re-aim)")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.plans_from:
        runs = json.load(open(a.plans_from))
    else:
        runs = []
        from .. import llm as L
        specs = [a.llm] if a.llm else [f"mock:{p}" for p in a.profiles]
        for spec in specs:
            for seed in a.seeds:
                be = L.make(spec, out / "llm_cache" / f"seed{seed}", mock=V.mock_backend)   # one fresh model sample per run
                r = CO.run(be, ST.Scene(seed), image=a.image)
                prof = spec.split(":")[1] if spec.startswith("mock:") else spec
                r.update(profile=prof, stream_seed=seed, model=L.info(spec), skeleton=skeleton(r["plan"]) if r["plan"] else None)
                runs.append(r)
                print(prof, seed, r["status"], r["skeleton"], [{k: v for k, v in g.items() if k in ("frac", "push", "overhang", "grip")}
                                                              for g in (r["grounded"] or [])], flush=True)
        (out / "planning.json").write_text(json.dumps(runs, indent=1, default=float))
    if not a.execute:
        return
    ws = worlds(a.worlds, a.execute if a.execute > 0 else 0)
    obs = dict(noise=a.observe, retries=3) if a.observe else None
    jobs, seen = [], set()
    for r in runs:
        if not r["plan"]:
            continue
        g = {k: v for gg in r["grounded"] for k, v in gg.items() if k in ("frac", "push")}
        key = (r["skeleton"], round(g.get("frac", 0), 4), round(g.get("push", 0), 4))
        if key in seen:
            continue
        seen.add(key)
        pid = f"{r['skeleton']}_f{g.get('frac', 0):.3f}_p{g.get('push', 0):.3f}"
        if r["skeleton"] == "push_clamp_lift_two":
            for j, (tag, w) in enumerate(ws):
                kw = dict(w, push=g["push"], grasp_frac=g["frac"])
                if obs:
                    kw["observe"] = dict(obs, seed=3000 + j)
                jobs.append((f"{pid}__{tag}", "under_rim", kw))
        elif r["skeleton"] == "one_arm_l" and a.worlds == "setA":
            for pl in ("none", "near_handle", "far_centre", "far_left", "far_right"):
                jobs.append((f"{pid}__{pl}", "handle_grasp", dict(placement=pl, grip="force", grip_force=10.0, grasp_frac=g["frac"])))
        else:
            print("no executor for", pid, "on worlds", a.worlds, flush=True)
    raw = out / "exec_raw"
    raw.mkdir(exist_ok=True)
    done = {p.stem for p in raw.glob("*.json")}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"executing {len(jobs)} runs (worlds={a.worlds}, observe={a.observe})", flush=True)
    with get_context("spawn").Pool(a.procs, maxtasksperchild=2) as pool:
        for res in pool.imap_unordered(_exec, jobs):
            (raw / f"{res['tag']}.json").write_text(json.dumps(res, indent=1, default=float))
            print(res["tag"], res.get("held"), flush=True)
    (out / "_DONE_exec").write_text("ok")


if __name__ == "__main__":
    main()
