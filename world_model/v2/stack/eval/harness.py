"""stack/eval/harness.py -- enumerate a universe's worlds, run the per-world job in a spawn Pool, write one schema record per world.  SIMULATION ONLY.

    python -m world_model.v2.stack.eval.harness --universe contents --set cert --tasks tray2 --n 1 --out DIR --procs N [--max-worlds K]

out_dir/SPEC.json, raw/<world_id-safe>.json (atomic, resumable: existing files are skipped), results.jsonl (all raw, at the end).
Contents job (legacy path until the planner/executor migration lands): carry.world_trial with the certificate's kwargs
(a_plan = carry_cert.static_cell admitted a for the cell, m_hat = cell centre, fric_comp / weld_solref from config.carry_physics()).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

PIN = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


@dataclass
class EvalSpec:
    universe: str = "contents"
    world_set: str = "cert"
    n: int | None = 1
    seed: int = 0
    planner: str = "scripted"
    proposer: dict = field(default_factory=lambda: dict(spec=None, model=None, reasoning=None))
    producers: list = field(default_factory=lambda: ["truth"])
    run_kwargs: dict = field(default_factory=dict)            # per-universe job kwargs (contents: branch_backups, ctl, a_plan override)
    universe_kwargs: dict = field(default_factory=dict)       # e.g. contents: tasks=["tray2"]
    max_worlds: int | None = None                             # first K worlds only (smoke)


def safe_name(world_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", world_id).strip("_") + "-" + hashlib.sha1(world_id.encode()).hexdigest()[:6]


def _universe(spec):
    from .. import registry
    return registry.get("universes", spec.universe)(**spec.universe_kwargs)


# ----------------------------------------------------------------------------------------------------------------------- contents job
def _contents_job(uni, world, spec):
    from ...nero import config as C
    from ...nero.contents import carry as CR
    from ...nero.contents import carry_cert as CC
    from ...nero.contents import scenes as SC
    from . import schema
    t0 = time.time()
    task, cell = world.theta["task"], tuple(world.theta["cell"])
    rk = dict(spec.run_kwargs)
    world0 = SC.TrayWorld(n_cups=int(task[4:])) if task.startswith("tray") else SC.BoxWorld()
    st = CC.static_cell(task, cell, world0)
    a = rk.pop("a_plan", None) or st["a_admit"]
    cp = C.carry_physics()
    kw = dict(branch_backups=rk.pop("branch_backups", True), ctl=rk.pop("ctl", "lever"), fric_comp=rk.pop("fric_comp", cp["fric_comp"]),
              weld_solref=rk.pop("weld_solref", cp["weld_solref"]))
    assum = dict(uni.assumptions(), static_action=st["action"], a_admit=a, u_nom=st["u_nom"], run_kwargs=dict(spec.run_kwargs),
                 route="legacy carry.world_trial (ScriptedPlanner not wired yet)", proposer="none (scripted plan; no LLM)")
    base = dict(universe=uni.name, world_id=world.world_id, seed=world.seed, theta=world.theta, task=task, planner=spec.planner,
                proposer=spec.proposer, producers=spec.producers, assumptions=assum)
    if a is None:
        return schema.build_record(**base, outcome="refused", guarantee="proof",
                                   metrics=dict(static=dict(action=st["action"], u_nom=st["u_nom"])), wall_s=time.time() - t0,
                                   verdicts=[dict(checker="static_cell", admit=False, guarantee="proof", margins={}, why=st["action"])])
    from ...adapters import outcome_carry as OCC
    ep = CR.CarryEpisode(world.native, a_plan=a, ctl=kw["ctl"], m_hat=0.5 * sum(st["contents_m"]), fric_comp=kw["fric_comp"], weld_solref=kw["weld_solref"])
    ep.run_phases([("hold", 0.4)])                            # = carry.world_trial, with the episode kept to measure the end state
    r = CR.run_segments(ep, dict(tag=world.world_id, task=ep.task, theta=CR._theta(world.native), truth=CR._jsonable(ep.truth0), a_plan=a, m_nom=ep.m_nom, ctl=kw["ctl"],
                                 strict_claw=False, fric_comp=ep.fc, coop=ep.coop, weld_solref=ep.weld_solref, physics=ep.physics_flags(), segments=[]),
                        kw["branch_backups"], time.time())
    verdicts, viol = [], None
    for s in r["segments"]:
        nom, fam = s["nominal"], s["family"][0.0]
        admit = (not nom["viol"]) and fam["n_bad"] == 0          # end_in_R is only meaningful at rest (reported in why)
        if nom["viol"] and viol is None:
            viol = dict(seg=s["seg"], **{k: v for k, v in nom["viol"].items() if isinstance(v, (str, int, float))})
        verdicts.append(dict(checker=f"segment:{s['seg']}", admit=bool(admit), guarantee="sampled",
                             margins={k: nom.get(k) for k in ("obj_tilt_max", "cup_tilt_max", "spill_ratio", "tau_ratio") if k in nom},
                             why=f"viol={nom['viol']} end_in_R={nom['end_in_R']} backups bad {fam['n_bad']}/{fam['n']}"))
    okR, whyR = r["end_in_R"]
    complete = len(r["segments"]) == len(CR.SEGS)
    unsafe = bool(viol or r.get("pre_hold_viol"))
    m = OCC.measure(ep, task, complete, unsafe, r.get("pre_hold_viol"), end_in_R=(okR, whyR))     # the ONE goal + success rule (stack.outcome)
    outcome = "violation" if m["result"]["unsafe"] else ("ok" if m["result"]["safe_success"] else "fail")
    metrics = dict(result=m["result"], goal=m["goal"], measured=m["measured"], end_in_R=bool(okR), R_fail=list(whyR), pre_hold_viol=r.get("pre_hold_viol"), viol=viol, n_segments=len(r["segments"]),
                   truth_end=r.get("truth_end"), physics=r.get("physics"), sim_wall_s=r.get("wall_s"))
    return schema.build_record(**base, outcome=outcome, guarantee="sampled", metrics=metrics, verdicts=verdicts, wall_s=time.time() - t0)


JOBS = {"contents": _contents_job}


def _job(args):
    spec_d, world, out_dir = args
    for k in PIN:
        os.environ[k] = "1"
    spec = EvalSpec(**spec_d)
    path = Path(out_dir) / "raw" / (safe_name(world.world_id) + ".json")
    if path.exists():
        return str(path), "skipped"
    from . import schema
    try:
        uni = _universe(spec)
        rec = JOBS[spec.universe](uni, world, spec)
    except Exception as e:                                   # reported, never dropped
        import traceback
        rec = schema.build_record(universe=spec.universe, world_id=world.world_id, seed=world.seed, theta=world.theta,
                                  task=world.theta.get("task", ""), planner=spec.planner, proposer=spec.proposer, producers=spec.producers,
                                  outcome="error", metrics=dict(error=repr(e), tb=traceback.format_exc()[-1500:]))
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, path)
    return str(path), rec["outcome"]


def run(spec: EvalSpec, out_dir, procs: int = 1) -> list[dict]:
    import multiprocessing as mp
    for k in PIN:
        os.environ[k] = "1"                                  # inherited by the spawned workers
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    uni = _universe(spec)
    worlds = uni.worlds(spec.world_set, spec.n, spec.seed)
    if spec.max_worlds is not None:
        worlds = worlds[:spec.max_worlds]
    from . import schema
    (out / "SPEC.json").write_text(json.dumps(dict(spec=dataclasses.asdict(spec), n_worlds=len(worlds), code_rev=schema.code_rev(),
                                                   date=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1, default=str))
    jobs = [(dataclasses.asdict(spec), w, str(out)) for w in worlds]
    todo = [j for j in jobs if not (out / "raw" / (safe_name(j[1].world_id) + ".json")).exists()]
    print(f"{len(worlds)} worlds, {len(worlds) - len(todo)} already done, running {len(todo)} on {procs} procs", flush=True)
    if todo:
        with mp.get_context("spawn").Pool(max(1, procs), maxtasksperchild=3) as pool:
            for p, o in pool.imap_unordered(_job, todo):
                print(f"  {Path(p).name}: {o}", flush=True)
    recs = [json.loads((out / "raw" / (safe_name(w.world_id) + ".json")).read_text()) for w in worlds
            if (out / "raw" / (safe_name(w.world_id) + ".json")).exists()]
    tmp = out / "results.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in recs))
    os.replace(tmp, out / "results.jsonl")
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="contents")
    ap.add_argument("--set", default="cert", dest="world_set")
    ap.add_argument("--tasks", default="tray2,tray4,tray6,box", help="contents only")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner", default="scripted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--max-worlds", type=int, default=None)
    ap.add_argument("--no-branch", action="store_true", help="contents: skip the stored-backup branches (fast, no sampled backup rows)")
    a = ap.parse_args(argv)
    rk, uk = {}, {}
    if a.no_branch:
        rk["branch_backups"] = False
    if a.universe == "contents":
        uk["tasks"] = a.tasks.split(",")
    spec = EvalSpec(universe=a.universe, world_set=a.world_set, n=a.n, seed=a.seed, planner=a.planner, run_kwargs=rk, universe_kwargs=uk,
                    max_worlds=a.max_worlds)
    t0 = time.time()
    recs = run(spec, a.out, a.procs)
    from collections import Counter
    print(dict(Counter(r["outcome"] for r in recs)), f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
