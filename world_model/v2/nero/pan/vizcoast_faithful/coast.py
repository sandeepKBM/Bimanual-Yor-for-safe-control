"""The VIZ-COAST loop (paper Sec. III): VRM constraints -> Z3 SATPlan -> COAST stream refinement -> on failure: feedback cache +
temporary state-action block, replan; after K failures drop the blocks and ask the VRM to revise; no plan -> relax; program
error -> fix.  Fails after 100 refinement failures or 600 s.  SIMULATION ONLY.
"""
from __future__ import annotations

import time
import traceback

from . import pddl_lite as PL
from . import satplan as SP
from . import vrm as V

MAX_FAIL, TIMEOUT_S = 100, 600.0      # from the paper (100 refinement failures or 600 s)


def run(backend, scene, K=5, h_max=8, image=None, max_vrm_calls=10):
    task = PL.load(V.PDDL)
    vrm = V.VRM(backend, scene, image)
    t0 = time.time()
    code = vrm.initial()
    blocks, feedback, pending, events = [], [], [], []
    n_fail = 0
    t_plan = 0.0
    while True:
        if time.time() - t0 > TIMEOUT_S or n_fail > MAX_FAIL or len(vrm.log) > max_vrm_calls:
            return _out("fail_limit", None, None, task, vrm, events, n_fail, t_plan, t0, scene, code)
        if code is None:
            events.append(dict(event="no_code_in_reply"))
            code = vrm.fix("(none)", "no python code block in the reply")
            continue
        try:
            tp = time.time()
            plan, H, n_vlm = SP.plan(task, code, blocks, h_max)
            t_plan += time.time() - tp
        except Exception:
            err = traceback.format_exc(limit=2)[-800:]
            events.append(dict(event="program_error", error=err.splitlines()[-1]))
            code = vrm.fix(code, err)
            continue
        if plan is None:
            events.append(dict(event="no_plan", horizon=h_max, n_blocks=len(blocks)))
            if blocks:
                blocks = []
            new = vrm.relax(code, h_max)
            if new == code:
                return _out("no_plan", None, None, task, vrm, events, n_fail, t_plan, t0, scene, code)
            code = new
            continue
        ok, grounded, i, reason = scene.refine(plan)
        events.append(dict(event="refine", plan=[str(a) for a in plan], ok=ok, failed_at=None if ok else str(plan[i]), reason=reason,
                           horizon=H, n_vlm_clauses=n_vlm))
        if ok:
            return _out("ok", plan, grounded, task, vrm, events, n_fail, t_plan, t0, scene, code)
        n_fail += 1
        st = SP.states_along(task, plan)[i]
        blocks.append((st, plan[i]))
        pending.append(dict(plan=" ".join(map(str, plan)), action=str(plan[i]), reason=reason))
        if len(pending) >= K:
            blocks = []
            feedback += pending
            code = vrm.revise(code, pending)
            pending = []


def _out(status, plan, grounded, task, vrm, events, n_fail, t_plan, t0, scene, code):
    return dict(status=status, plan=[str(a) for a in plan] if plan else None, grounded=grounded, refinement_failures=n_fail,
                task_planning_s=round(t_plan, 3), wall_s=round(time.time() - t0, 2), stream_calls=scene.n_stream_calls,
                vrm_calls=len(vrm.log), vrm_log=vrm.log, events=events, final_program=code)
