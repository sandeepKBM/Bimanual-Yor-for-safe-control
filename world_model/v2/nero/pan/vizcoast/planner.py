"""Constraint-pruned task search + CEGIS loop.  SIMULATION ONLY.

enumerate_plans: breadth-first over skill sequences (shortest first, fixed prior order inside a length), proposer constraints
prune actions (VIZ-COAST), duplicate configurations are merged.  solve: certify each candidate (certify.check); a rejection's
counterexample theta* is cached with its signature and blocks every later candidate whose belief box contains theta*
(generalisation beyond the failed instance; a no-clean-pre-grasp failure blocks the left pinch position for ANY push/payload);
an admitted candidate optionally goes to Stage B (simcheck) whose failures join the evidence and trigger a replan.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from . import certify as CE
from . import domain as D
from . import evidence as E
from . import facts as F


@dataclass
class Stats:
    nodes: int = 0
    pruned_by_proposer: int = 0
    candidates: int = 0
    pruned_by_counterexample: int = 0
    certificate_calls: int = 0
    sim_rounds: int = 0
    sim_runs: int = 0
    rejected_by_rule: dict = field(default_factory=dict)


def enumerate_plans(belief, cons, stats, max_depth=6):
    q = deque([D.Plan(acts=[], beliefs=[], final=belief, state=D.INIT)])
    seen = set()
    while q:
        p = q.popleft()
        for a in D.actions(p.state):
            if not F.allowed(a, p.state, cons):
                stats.pruned_by_proposer += 1
                continue
            s2, b2 = D.apply(p.state, p.final, a)
            c = D.Plan(acts=p.acts + [a], beliefs=p.beliefs + [p.final], final=b2, state=s2)
            ab = c.aim_belief()
            key = (s2, b2, tuple(sorted(map(str, c.acts))), ab.dxy, ab.yaw)
            if key in seen:
                continue
            seen.add(key)
            stats.nodes += 1
            if D.GOAL in s2:
                yield c
            elif len(c.acts) < max_depth:
                q.append(c)


def _cex_scope(sig, cat):
    """generalisation key: a pre-grasp failure depends only on (holders, left pinch position) and the aim pose."""
    return ("pregrasp", sig[0], sig[1], E.obs_tag(sig)) if cat == "no_clean_pregrasp" else ("sig", sig)


def _blocked(cex, plan):
    sig, box = plan.signature(), plan.aim_belief()
    for c in cex:
        if c["scope"] in (("sig", sig), ("pregrasp", sig[0], sig[1], E.obs_tag(sig))) and box.contains(c["theta"]):
            return c
    return None


def solve(belief, cons, db, sim=None, max_depth=6, max_rounds=6, trace_cap=400):
    """-> dict(plan, verdict, stage, stats, trace, counterexamples).  sim(plan) -> list of new evidence entries (or None)."""
    st = Stats()
    cex, trace = [], []
    t0 = time.time()
    for rnd in range(max_rounds):
        chosen = None
        for plan in enumerate_plans(belief, cons, st, max_depth):
            st.candidates += 1
            hit = _blocked(cex, plan)
            if hit:
                st.pruned_by_counterexample += 1
                continue
            v = CE.check(plan, db)
            st.certificate_calls += 1
            if len(trace) < trace_cap:
                trace.append(dict(round=rnd, plan=str(plan), ok=v.ok, rule=v.rule, evidence=v.evidence, detail=v.detail,
                                  counterexample=v.counterexample, support=v.support))
            if not v.ok:
                st.rejected_by_rule[v.rule] = st.rejected_by_rule.get(v.rule, 0) + 1
                if v.counterexample is not None:
                    cex.append(dict(scope=_cex_scope(plan.signature(), v.counterexample.get("cat")), theta=v.counterexample,
                                    rule=v.rule, plan=str(plan)))
                continue
            chosen = (plan, v)
            break
        if chosen is None:
            return _out(None, None, "no_plan", st, trace, cex, t0)
        plan, v = chosen
        if sim is None:
            return _out(plan, v, "A", st, trace, cex, t0)
        new = sim(plan)
        st.sim_rounds += 1
        st.sim_runs += len(new)
        db.extend(new)
        fails = [e for e in new if not e["held"]]
        trace.append(dict(round=rnd, plan=str(plan), stageB=dict(runs=len(new), failed=len(fails),
                                                                 cats=sorted({e["cat"] for e in fails}))))
        if not fails:
            return _out(plan, v, "B", st, trace, cex, t0)
    return _out(None, None, "round_limit", st, trace, cex, t0)


def _out(plan, v, stage, st, trace, cex, t0):
    return dict(plan=str(plan) if plan else None, plan_obj=plan, stage=stage,
                verdict=None if v is None else dict(rule=v.rule, evidence=v.evidence, detail=v.detail, support=v.support),
                assumptions=list(plan.final.assumptions) if plan else [],
                aim_belief=plan.aim_belief().describe() if plan else None,
                stats=st.__dict__, trace=trace,
                counterexamples=[dict(c, scope=list(c["scope"]) if not isinstance(c["scope"], str) else c["scope"]) for c in cex],
                wall_s=round(time.time() - t0, 2))
