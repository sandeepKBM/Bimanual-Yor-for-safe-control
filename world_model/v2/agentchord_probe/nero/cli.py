"""agentchord_probe/nero/cli.py -- AgentChord's planner on our hidden-contents carry tasks.  SIMULATION ONLY (no hardware; LLM only via the
owner-run `plan --backend codex`).

  plan --task tray4 --backend codex --model gpt-5.6-terra [--reasoning high] --out DIR     plan ONCE per task variant (cached in DIR)
  plan --task tray4 --backend replay --replay-log OLD/llm_log.jsonl --out DIR              answers only from a logged run (no model call)
  plan --task tray4 --backend canned --canned replies.json --out DIR                        [task_graph JSON, recovery_spec JSON] (tests)
  exec --plan DIR/plan.json --task tray4 --n 20 --procs 2 --out OUT                         run the cached plan on the carry_cert worlds

Env: PYTHONPATH=<worktree> MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1; python = /data/local/ss5772/agentchord/venv_nero/bin/python
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from world_model.v2.nero import gripper as GR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--task", required=True)
    p.add_argument("--backend", required=True, choices=["codex", "replay", "canned"])
    p.add_argument("--model")
    p.add_argument("--reasoning")
    p.add_argument("--replay-log")
    p.add_argument("--canned")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--no-recovery", action="store_true")
    p.add_argument("--out", required=True)
    e = sp.add_parser("exec")
    e.add_argument("--plan", required=True)
    e.add_argument("--task", required=True)
    e.add_argument("--n", type=int, default=5, help="worlds per cell")
    e.add_argument("--cells", default="all", help="'all' or comma list of cell indices")
    e.add_argument("--procs", type=int, default=1)
    e.add_argument("--seeds", choices=["cert", "baseline"], default="cert")
    e.add_argument("--m-hat", choices=["prior", "cell"], default="prior")
    e.add_argument("--mon-period", type=float, default=0.1)
    e.add_argument("--edge-settle", type=float, default=0.3, help="dwell after each motion edge [s] (0.3 = carry.py segment dwell)")
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        from .plan import make_plan
        meta = make_plan(a.task, a.out, backend=a.backend, model=a.model, reasoning=a.reasoning, replay_log=a.replay_log, canned=a.canned,
                         regenerate=a.regenerate, recovery=not a.no_recovery)
        print(json.dumps({k: meta[k] for k in ("task", "plan_id", "backend", "models", "n_llm_calls", "n_errors", "plan_wall_s")}, default=str))
        return
    import multiprocessing as mp
    from . import tasks as T
    from .execute import ASSUMPTIONS, run_world
    plan = json.loads(pathlib.Path(a.plan).read_text())
    cells = T.cell_list(a.task)
    if a.cells != "all":
        cells = [cells[int(k)] for k in a.cells.split(",")]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(date=time.strftime("%Y-%m-%d"), task=a.task, plan=a.plan, plan_id=plan["plan_id"], backend=plan["backend"], models=plan["models"],
                reasoning=plan.get("reasoning"), caveats=plan["caveats"], codex_note=plan.get("codex_note"), n_per_cell=a.n, cells=cells,
                seeds=a.seeds, m_hat=a.m_hat, mon_period_s=a.mon_period, edge_settle_s=a.edge_settle, assumptions=ASSUMPTIONS, **GR.provenance("rigid weld"), setup_deviations=plan["setup_deviations"],
                prompt_hash=plan["prompt_hash"])
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    jobs = [dict(plan=a.plan, task=a.task, cell=list(c), i=i, seeds=a.seeds, m_hat=a.m_hat, mon_period=a.mon_period, edge_settle=a.edge_settle) for c in cells for i in range(a.n)]
    print(f"{len(jobs)} worlds", flush=True)
    recs = []
    t0 = time.time()
    with open(out / "trials.jsonl", "a") as f:
        if a.procs <= 1:
            it = map(run_world, jobs)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(min(a.procs, 2), maxtasksperchild=3)       # AGENTS: at most 2 processes on the loaded machine
            it = pool.imap_unordered(run_world, jobs)
        for r in it:
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            recs.append(r)
            print(f"{r['task']} {r['cell']} i={r['i']} {r['outcome']} {r.get('reason') or ''} ac={r.get('ac_success')} ours={r.get('success_ours')} {r['wall_s']}s", flush=True)
    summ = summarize(recs)
    summ["wall_s"] = round(time.time() - t0)
    (out / "RESULT.json").write_text(json.dumps(dict(metadata=meta, summary=summ), indent=1, default=float))
    print(json.dumps(summ, default=float))


def summarize(recs):
    from collections import Counter
    return dict(n=len(recs), outcomes=dict(Counter(r["outcome"] for r in recs)),
                rejections=dict(Counter(x["status"] for r in recs for x in r.get("rejections", []))),
                ac_success=sum(bool(r.get("ac_success")) for r in recs), success_ours=sum(bool(r.get("success_ours")) for r in recs),
                any_U=sum(bool(r.get("any_U")) for r in recs), monitors_fired=sum(len(r.get("monitors_fired", [])) for r in recs),
                recoveries_run=sum(r.get("n_recoveries_run", 0) for r in recs))


if __name__ == "__main__":
    main()
