"""nero/contents/runtime_rules.py -- the glue between the checkers (adapters/) and the executor (stack/executor.py) for Rules 3 and 4 at run time.  SIMULATION ONLY, offline.
stack/executor.py may import no adapters and no new nero.contents module, so what it needs from them is handed in as data / callables built here:

  certs_from_rule1(recs)          {step index: Rule 1 step record} for run_segments(certs=...) (adapters/rule1_contents.run_cell records)
  rule4_hook(task, wp, a, tilt=)  handles["rule4"](seg, t_branch, theta_new, rest_at=None) -> dict(accepted, record, target, wall_s, n_rounds): the Rule 4 propose-check loop with the
                                  OFFLINE mock target proposer (rule4_contents.MockTargetProposer; no VLM / LLM / network is ever called) on a CarryState at that branch time
  theta_new_from_read(...)        Theta_0 ∩ {theta : h(theta) in [y - nu, y + nu]} (D1) = what a contradicting reading leaves

Wiring (probe_select / behind_job; BIMANUAL_LEGACY_NO_RUNTIME_RULES=1 = off):
    r1 = run_cell(..., fallback=True)                         # the wide table needs Theta_0
    certs = RR.certs_from_rule1(r1); handles = dict(rule4=RR.rule4_hook(task, wp, a, tilt=<the plan's TiltSpec from planned_path_rows>))
    CR.run_segments(ep, rec, branch_backups=True, certs=certs, handles=handles)
"""
from __future__ import annotations

import time

from world_model.v2.adapters import rule1_contents as RC
from world_model.v2.adapters import rule4_contents as R4C
from world_model.v2.stack import theta as TH
from world_model.v2.stack.checks import rule4 as R4


def certs_from_rule1(recs) -> dict:
    """run_cell step records -> {step index: record}.  A step without an admitted record is kept (the executor refuses the run: fail closed)."""
    return {int(r["step"]): r for r in recs}


def theta_new_from_read(y: float, nb: float, spec: TH.ReadingSpec, theta_t: TH.ThetaSet, theta_0: TH.ThetaSet) -> TH.ThetaSet:
    """D1: a reading CONSISTENT with theta_t (y inside reading_range(spec, theta_t)) gives theta_t ∩ slab; a contradicting one gives Theta_0 ∩ {theta : h(theta) in [y - nu, y + nu]},
    nu = spec.noise (= EPS_FREE + the MAX nb over the task's path samples, so >= the measured nb: sound).  nb is accepted for the record only."""
    return TH.theta_from_reading(spec, theta_0, float(y), theta_t=theta_t)


def rule4_hook(task: str, wp=None, a: "float | None" = None, planner: str = "closed_chain", allow_set_down: bool = True, tilt=None):
    """the Rule 4 callable for stack/executor.py.  Offline mock target proposer only.  The sim is paused while this computes (latency 0 in sim time, ASSUMED; wall_s recorded).
    tilt (11-17): the per-plan load tilt measurement (collision_rows.TiltSpec) of the plan being run; it is set on the Rule 1 context EXPLICITLY at every call (the cached Ctx is
    shared).  None (and not BIMANUAL_LEGACY_UNTILTED_COLLISION=1) = Rule 4 refuses: MISSING, no target is checked."""
    from world_model.v2.adapters import collision_rows as RCOL

    def hook(seg, t_branch, theta_new, rest_at=None):
        t0 = time.time()
        if tilt is None and not RCOL.legacy_untilted():
            return dict(accepted=False, record=None, target=None, wall_s=round(time.time() - t0, 3), n_rounds=0, missing=True,
                        why="MISSING: no per-plan load tilt measurement for the Rule 4 collision rows (rule4_hook(tilt=))")
        ctx = RC.get_ctx(task, planner, wp)
        ctx.tilt = tilt
        model = ctx.model(float(a if a is not None else RC.AD.A_NOM))
        st = R4C.CarryState(ctx, model, seg, float(t_branch), rest_at)
        check0, _tr = R4C.make_check(st, theta_new)
        last = {}

        def check(tg):
            ok, rows, adm = check0(tg)
            if adm is not None:
                last[tg.name] = adm
            return ok, rows, adm

        acc, log = R4.loop(st, theta_new, R4C.MockTargetProposer(task, allow_set_down), check, failing_rows=())
        rec = None if acc is None else last.get(acc.name)
        return dict(accepted=bool(acc is not None and rec is not None), record=rec, target=None if acc is None else acc.name, wall_s=round(time.time() - t0, 3),
                    n_rounds=len(log), why=None if acc is not None else "no accepted target in MAX_ROUNDS")
    return hook
