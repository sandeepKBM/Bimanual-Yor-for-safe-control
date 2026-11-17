"""nero/contents/probe_select.py -- OUR per-world run on schema v2: the LLM's probes -> coverage -> Rule 1 -> Rule 2 re-ask -> carry.  SIMULATION ONLY.   (11-09, probe-selection build step 6)

Called by probe._e2e_job (fixed base) and behind_job.behind_job (table behind) when the proposer plan is schema v2 (plan["plan_dict"] has "phenomena"); the v1 plans (numeric belief,
BIMANUAL_LEGACY_PROPOSER_V1=1) and the plan-less default job keep their own code unchanged.

One world, one episode (the carry continues in the episode the probe ran in).  Attempt loop (NO cap on re-asks, ruling 11-09; every attempt is recorded):
  1 PROBES    stack.probing.run_probes on Theta_t with the probes THIS plan lists that have not run yet (N2: only the LLM's selection runs; an omitted skill's phenomenon sits on its
              fallback).  Before each probe the checker evaluates the skill's own limit rows (Weigh.limit_rows = checks/probe_rung); static verdict "carry" on all of Theta_t ->
              "no_probe_needed".  Readings are intersected into Theta_t on their own fields only (the world is never shrunk to model sensing).
  2 T3        a reading disjoint from the earlier sound set (or an internally inconsistent one) STOPS the run: counted event, no carry, the object is let go in place.
  3 COVERAGE  checks/coverage on the checker's own list PHI(task): measured, approved fallback, else uncovered -> refuse (enforce mode, ruling 11-09 Q1).
  4 RULE 1    the static admission (carry_cert.static_cell on the plan's compiled segments) on Theta_t's contents interval; table behind: + the base level (adapters/base_envelope
              .pick_level) and the plan recompiled at that level.  Admitted -> carry (stored backups); not admitted -> 5.
  5 RULE 2    T1: a reading showed a phenomenon in PHI the plan did not name (Reading.detail["shows"]) -> re-ask.  T2: no plan recovery is viable on Theta_t, i.e. the admission
              refuses (a lower-bound-only weigh ends here too) -> re-ask with the readings, the probes the checker skipped, the failing rows and the worst vertex.
              The re-ask (propose.reask_rule2_until_accepted) returns a plan that is validated + compiled (planner build; table behind at L1, the per-world level at step 4); then 1-4 again.
              FIXED POINT: a re-asked plan whose id was already tried cannot change the (deterministic) verdict: the loop stops there ("reask_fixed_point"), counted.  Not a cap.
    T4 (11-16, USER-approved; env BIMANUAL_LEGACY_NO_T4=1 = off, byte-identical to before): the static admission admits -> Rule 1 (adapters/rule1_contents.run_cell: per step, ONE recovery viable for
              every theta from every branch point; n_sampled = 0, closed_chain, the plan's wp and a_admit).  A refused step's failing rows are tagged by stack/checks/fixability.  Any FIXABLE /
              FIXABLE_GEOMETRY row -> re-ask (trigger "T4", evidence rule1_refusals + fixed_constraints); only NOT_FIXABLE rows -> refuse (refused_by "rule1", no re-ask).  The same failing-FIXABLE
              signature on MAX_SAME_T4_SIGNATURE (3, ASSUMED, USER 11-16) consecutive T4 attempts -> stop ("t4_repeat") and refuse.  A T4 loop that ends without a plan Rule 1 admits refuses.
Not run here: a re-ask needs plan["proposer"] = dict(spec, cache, offline[, reask_limit]); without it a refusal ends the run ("reask_unavailable").
Out of scope (ruling 11-09 Q1 option c): PH4 / PH5 / PH6 (catalog.OUT_OF_SCOPE_NOTE) in every record.
Bounds: nothing numeric is new here; the probe numbers are probe.py's (EPS from the poslift CALIB, MEASURED), the admission thresholds are the checkers' (U_ADMIT 1.0 MODEL).
"""
from __future__ import annotations

import dataclasses
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from world_model.v2.adapters import outcome_carry as OCC, skills_contents as SK
from world_model.v2.stack import u_groups as _UG
from world_model.v2.nero.contents import carry as CR, probe as P
from world_model.v2.stack import probing as PB
from world_model.v2.stack.checks import fixability as FX
from world_model.v2.stack.proposer import compile as PC, propose as PP, schema as S
from world_model.v2.stack.skills import catalog as CAT

FIELD = SK.FIELD
THETA_MODES = (None, "no_probe", "oracle_mass", "oracle_theta", "estimate")
REFUSED_BY = ("static", "coverage", "base_level", "compile", "T3", "probe", "rule1")
MAX_SAME_T4_SIGNATURE = PP.MAX_SAME_T4_SIGNATURE      # ASSUMED, USER 11-16 approved N = 3 (one place: stack/proposer/propose.py)
RULE1_TASKS = ("box", "tray2", "tray4", "tray6", "box_behind", "tray6_behind")     # adapters/rule1_contents supports these (not pot)
T4_REFUSE_ACTION = "refuse (Rule 1, not fixable by the plan)"
T4_NO_PLAN_ACTION = "refuse (Rule 1: no re-asked plan admitted)"


@dataclass
class Ctx:
    task: str                         # the task name of the plan / records (box | tray6 | pot | tray6_behind | box_behind)
    pt: str                           # the probe's task key (the base task for a table-behind task)
    i: int
    seed: int
    ep: object
    world: object
    plan: dict                        # job plan: wp, fallback, plan_dict (accepted v2 plan), plan_id, proposer
    eps_lo: float
    eps_free: float
    static: Callable                  # (c_lo, c_hi, wp) -> carry_cert.static_cell dict
    cmax: Callable                    # (wp) -> heaviest admissible contents mass or None
    check: Callable                   # (plan dict) -> CompiledPlan | raises PlanRejected   (validation compile of a re-asked plan)
    method: str | None = None
    adaptive: bool = True
    behind: bool = False
    level: Callable | None = None     # behind: (c_lo, c_hi) -> (level | None, rows)
    at_level: Callable | None = None  # behind: (plan dict, level) -> WaypointPlan  (raises PlanRejected)
    rows_json: Callable | None = None
    coverage_mode: str = "enforce"
    make_skills: Callable | None = None   # tests: (weigh) -> {skill id: ProbeSkill}
    rule1: Callable | None = None     # T4 (11-16): (task, (c_lo, c_hi), wp, a) -> [run_cell step records]; None = adapters/rule1_contents.run_cell (closed_chain, static rows only); tests inject a stub
    # --- demo_runner arms (11-12, Stage 0d).  All default to off: the default path is bit-identical.
    theta_mode: str | None = None     # None = probe -> Theta_t (default) | "no_probe" (admit on Theta_0, no probe) | "oracle_mass" / "oracle_theta" (Theta = the TRUE contents point [c, c], no probe;
                                      # oracle_theta also needs `static` / `level` closed over the true COM point) | "estimate" (probe, then admit on the point [mid, mid] of its interval)
    on_phase: Callable | None = None  # (name) -> None, called at the phase boundaries: "probe" | "ramp_hold" | "nominal" | "refused_fallback"
    dry: bool = False                 # return right after the admission (no ramp / carry / fallback): smoke and unit tests


def clean(x):
    """JSON-friendly: inf -> None, tuples -> lists, numpy scalars -> float."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if math.isinf(x) or math.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def reading_json(rd):
    return dict(skill=rd.skill, output=rd.output, fields={f: clean(v) for f, v in rd.fields.items()}, why=rd.detail.get("why"))


def failing_rows(st):
    """the failing admission rows of a carry_cert.static_cell dict, as short dicts (the re-ask evidence)."""
    out = []
    for seg, rr in (st or {}).get("rows", {}).items():
        for name, r in rr.items():
            if isinstance(r, dict) and r.get("ok") is False:
                d = dict(row=f"{name}[{seg}]")
                for k in ("u_max", "a", "a_plan", "a_open", "margin_m", "worst"):
                    if k in r:
                        d[k] = r[k]
                out.append(d)
    return clean(out)


def legacy_no_t4():
    return os.environ.get("BIMANUAL_LEGACY_NO_T4") == "1"


def run_rule1(c, lo, hi, wp, a):
    """Rule 1 per step on the admitted plan (rule1_contents.run_cell, n_sampled = 0, closed_chain, fallback not evaluated) -> the step records."""
    if c.rule1 is not None:
        return c.rule1(c.task, (lo, hi), wp, a)
    from world_model.v2.adapters import rule1_contents as R1
    with tempfile.TemporaryDirectory(prefix="t4_rule1_") as td:
        import pathlib
        return R1.run_cell(c.task, (lo, hi), pathlib.Path(td), n_sampled=0, planner="closed_chain", fallback=False, wp=wp, a=a)


def rule1_failures(recs):
    """the failing rows of every refused step of run_cell's records (all programs), tagged -> (fixable [dict], fixed [dict], untagged [dict]).  Row dicts carry step_index, step,
    recovery_program, branch_t, row, pair, distance_m, d_min_m, kind, knobs, key, why."""
    fixable, fixed, untagged, seen = [], [], [], set()
    for r in recs:
        if r.get("admitted"):
            continue
        rows = [(pn, x) for pn, pp in (r.get("per_program") or {}).items() for x in pp.get("failing", [])] or [(None, x) for x in (r.get("rows") or [])]
        if not rows:                                           # run_cell refusal without rows (the plan's a above a_admit)
            rows = [(None, dict(name="rule1_a_exceeds", guarantee="missing", why=r.get("refusal") or ""))]
        for pn, x in rows:
            pi = FX.parent_info(x.get("name", ""))
            for lf in FX.leaves(x):
                tg = FX.tag_leaf(lf["name"], lf["guarantee"], lf["why"])
                d = dict(step_index=r.get("step"), step=r.get("seg"), recovery_program=pn or pi["program"], branch_t=pi["branch_t"], row=lf["name"], pair=tg.pair or None,
                         distance_m=tg.distance_m, d_min_m=tg.d_min_m, swept_bound_m=tg.swept_bound_m, kind=tg.kind, knobs=list(tg.knobs), key=tg.key, why=(lf["why"] or tg.why)[:240])
                dk = (d["step_index"], d["recovery_program"], d["branch_t"], d["key"], d["row"], d["pair"])
                if dk in seen:
                    continue
                seen.add(dk)
                (fixable if tg.kind in (FX.FIXABLE, FX.FIXABLE_GEOMETRY) else untagged if tg.kind == FX.UNTAGGED else fixed).append(d)
    return fixable, fixed, untagged


def t4_signature(fixable):
    """sorted tuple of (step, row-family key, pair) of the failing FIXABLE rows (the repeat guard)."""
    return tuple(sorted({(d["step_index"], d["key"], d["pair"] or "") for d in fixable}, key=lambda x: (x[0] if x[0] is not None else -1, x[1], x[2])))


def evidence(trigger, readings, events, st, cov, shown=None, extra_rows=None, action=None, rule1_refusals=None, fixed_constraints=None):
    """the Rule 2 evidence dict (prompt.reask_evidence_block): only measurements and checker outputs (no truth)."""
    skipped = [dict(probe=e["skill"], why=e["kind"], failed_rows=e.get("failed_rows", [])) for e in events if e["kind"] in ("skipped", "no_probe_needed")]
    ev = dict(trigger=trigger, readings=[reading_json(r) for r in readings], probes_skipped=clean(skipped), failing_rows=failing_rows(st) + clean(extra_rows or []),
              coverage=None if cov is None else [dict(phi=r.phi, status=r.status, by=r.by) for r in cov.rows], action=action)
    if st and st.get("worst_nom"):
        ev["worst"] = clean(dict(st["worst_nom"], contents_mass_set_kg=st.get("contents_m")))
    if shown:
        ev["phenomena_shown"] = list(shown)
    if rule1_refusals:
        ev["rule1_refusals"] = clean([{k: d[k] for k in ("step_index", "step", "recovery_program", "branch_t", "row", "pair", "distance_m", "d_min_m", "kind", "knobs")} for d in rule1_refusals])
    if fixed_constraints:
        ev["fixed_constraints"] = clean([dict(step_index=d["step_index"], row=d["row"], why=d["why"]) for d in fixed_constraints])
    return {k: v for k, v in ev.items() if v not in (None, [], {})} | dict(trigger=trigger)


def _why(weigh, events):
    if any(e["kind"] == "ran" for e in events):
        return weigh.why
    if any(e["kind"] == "T3" for e in events):
        return "T3"
    if any(e["kind"] == "skipped" for e in events):
        return "probe_skipped"
    if any(e["kind"] == "no_probe_needed" for e in events):
        return "no_probe_needed"
    return "no_probe_selected"


def run_world(c: Ctx):
    """-> rec (the record of the fixed-base job, plus rec["v2"] and the table-behind fields the caller adds)."""
    t0 = time.time()
    task, pt, ep = c.task, c.pt, c.ep
    e = P.empty_mass(pt)
    c0 = tuple(float(x) for x in P.prior_contents(pt))
    phi = CAT.phi(task)
    cfg = c.plan.get("proposer") or {}
    M = P.true_mass(ep)
    weigh = SK.Weigh(ep, pt, c.eps_lo, c.eps_free, rng=np.random.default_rng(c.seed), method=c.method, adaptive=c.adaptive)
    skills = {"weigh": weigh} if c.make_skills is None else c.make_skills(weigh)
    cur_pd, cur_wp, cur_cp = c.plan["plan_dict"], c.plan["wp"], None
    first_wp = cur_wp
    seen = {cur_pd.get("plan_id")}
    ran, events, readings = set(), [], []
    cum = PB.ProbeRun(theta_t={FIELD: c0})
    mode = c.theta_mode
    if mode not in THETA_MODES:
        raise ValueError(f"theta_mode {mode!r} not in {THETA_MODES}")
    if mode in ("oracle_mass", "oracle_theta"):
        cum.theta_t = {FIELD: (M - e, M - e)}                              # ORACLE: the true contents mass (truth, not a measurement); the world is the point
    estimate = None
    if c.on_phase:
        c.on_phase("probe")
    attempts, reasks, t_events = [], [], []
    stopped, end_reason, decision, st, cov, wp_run, lvl_rows = None, None, None, None, None, None, None
    t4_sigs, t4_pending = [], None                                          # T4 (11-16): the failing-FIXABLE signatures so far; the Rule 1 refusal of the current attempt
    while True:
        k = len(attempts)
        t4_pending, r1_fixable, r1_fixed, t4_ev = None, [], [], {}
        named = [p["id"] for p in cur_pd["phenomena"]]
        new_names = [p.get("name") for p in cur_pd["phenomena"] if p["id"] == CAT.NEW] + [p["id"] for p in cur_pd["phenomena"] if p["id"] != CAT.NEW and p["id"] not in phi]
        att = dict(attempt=k, plan_id=cur_pd.get("plan_id"), phenomena=cur_pd["phenomena"], probes=list(cur_pd["probes"]))
        cm = c.cmax(cur_wp)
        weigh.m_stop = None if cm is None else e + cm
        todo = [] if mode in ("no_probe", "oracle_mass", "oracle_theta") else [s for s in cur_pd["probes"] if s not in ran]
        cat = {sid: dataclasses.replace(s, preconditions=tuple(p for p in s.preconditions if p not in ran)) for sid, s in CAT.SKILLS.items()}
        run = PB.run_probes(todo, skills, cum.theta_t, static_same=lambda th, w_=cur_wp: not c.static(*th[FIELD], w_)["action"].startswith("refuse"), skill_catalog=cat)
        att["probe_events"] = clean(run.events)
        events += [dict(ev, attempt=k) for ev in run.events]
        readings += run.readings
        ran |= {ev["skill"] for ev in run.events if ev["kind"] in ("ran", "T3")}
        cum.theta_t = run.theta_t
        cum.measured = list(dict.fromkeys(cum.measured + run.measured))
        if mode == "estimate":                                              # admit on the POINT [mid, mid] of the probe interval (the estimate); the interval is kept in the record
            lo_e, hi_e = cum.theta_t[FIELD]
            if estimate is None:
                estimate = dict(interval=[lo_e, hi_e], point=0.5 * (lo_e + hi_e), lower_bound_only=any(r.output == "lower_bound" for r in readings))
            cum.theta_t = {FIELD: (estimate["point"], estimate["point"])}
        for ev in run.events:
            if ev["kind"] == "T3":
                t_events.append(dict(kind="T3", attempt=k, skill=ev["skill"], contradictions=clean(ev["contradictions"])))
        for rd in run.readings:
            if rd.detail.get("inconsistent"):
                t_events.append(dict(kind="T3", attempt=k, skill=rd.skill, contradictions=[dict(field=FIELD, why="reading lower bound above its upper bound", reading=clean(rd.detail["inconsistent"]))]))
                run.stopped = "T3"
        if run.stopped == "T3":
            stopped = "T3"
            att["stopped"] = "T3"
            attempts.append(att)
            break
        shown = sorted({s for rd in run.readings for s in rd.detail.get("shows", []) if s in phi and s not in named})
        cov = CAT.coverage(task, cum, mode=c.coverage_mode, named=named, new_names=new_names)
        c_lo, c_hi = cum.theta_t[FIELD]
        st, lvl_rows, wp_run = None, None, None
        if cov.refuses:
            decision = dict(action="refuse (uncovered)", a_admit=None, u_nom=None, contents=[c_lo, c_hi], refused_by="coverage", uncovered=cov.uncovered)
        else:
            st = c.static(c_lo, c_hi, cur_wp)
            decision = dict(action=st["action"], a_admit=st["a_admit"], u_nom=st["u_nom"], contents=[c_lo, c_hi], refused_by=None if st["a_admit"] is not None else "static")
            if c.behind and decision["a_admit"] is not None:                  # base level: highest of L1..L3 with every base row passing on Theta; then the plan at that level
                lvl, lvl_rows = c.level(c_lo, c_hi)
                decision["base_level"] = lvl
                if lvl is None:
                    decision = dict(decision, action="refuse (base level)", a_admit=None, refused_by="base_level",
                                    why_base={L: [kk + ": " + v.why for kk, v in rr.items() if not v.ok] for L, rr in lvl_rows.items()})
                else:
                    try:
                        wp_run = c.at_level(cur_pd, lvl)
                    except S.PlanRejected as ex:
                        decision = dict(decision, action="refuse (compile at level)", a_admit=None, refused_by="compile", why_compile=[ex.stage, list(ex.reasons)])
        att.update(coverage=cov.to_dict(), decision=clean(decision), t1_shown=shown)
        attempts.append(att)
        if shown:
            trigger = "T1"
        elif decision["a_admit"] is None:
            trigger = "T2"
        elif legacy_no_t4() or c.task not in RULE1_TASKS:
            break
        else:                                                               # T4: the static admission admits; Rule 1 (a recovery viable from every stop point) on the plan it admits
            wp_r1 = wp_run if c.behind else cur_wp
            try:
                r1 = run_rule1(c, round(c_lo, 4), round(c_hi, 4), wp_r1, decision["a_admit"])
            except Exception as ex:                                         # owner 11-16: fail CLOSED (a crashed safety check never lets the carry run); recorded
                att["rule1"] = dict(error=repr(ex)[:300])
                decision, end_reason = dict(decision, action="refuse (Rule 1 error)", a_admit=None, refused_by="rule1"), "rule1_error"
                att["decision"] = clean(decision)
                t_events.append(dict(kind="T4", attempt=k, error=att["rule1"]["error"], reasked=False))
                break
            att["rule1"] = dict(a=decision["a_admit"], steps=clean([dict(step=x.get("step"), seg=x.get("seg"), admitted=bool(x.get("admitted")), program=x.get("program"), refusal=x.get("refusal"),
                                                                          programs=x.get("programs")) for x in r1]))
            if all(x.get("admitted") for x in r1):
                break
            r1_fixable, r1_fixed, r1_untag = rule1_failures(r1)
            att["rule1"]["fixable"], att["rule1"]["fixed"], att["rule1"]["untagged"] = clean(r1_fixable), clean(r1_fixed), clean(r1_untag)
            sig = t4_signature(r1_fixable)
            t4_pending = dict(decision, action=T4_NO_PLAN_ACTION, a_admit=None, refused_by="rule1", rule1_rows=clean(r1_fixable + r1_fixed + r1_untag))
            if not r1_fixable:                                              # nothing the plan can fix: refuse here, no re-ask
                t4_pending["action"] = T4_REFUSE_ACTION
                att["decision"] = clean(t4_pending)
                decision, end_reason = t4_pending, "rule1_not_fixable"
                t_events.append(dict(kind="T4", attempt=k, refused_steps=sorted({d["step_index"] for d in r1_fixed}), signature=[], fixable=False, reasked=False))
                t4_pending = None
                break
            t4_sigs.append(sig)
            att["rule1"]["signature"] = clean(sig)
            trigger = "T4"
            t4_ev = dict(refused_steps=sorted({d["step_index"] for d in r1_fixable + r1_fixed}), signature=clean(sig), fixable=True, rows=[f"{d['step_index']}:{d['key']}:{d['pair'] or ''}" for d in r1_fixable])
            if len(t4_sigs) >= MAX_SAME_T4_SIGNATURE and len(set(t4_sigs[-MAX_SAME_T4_SIGNATURE:])) == 1:     # the same failure on N consecutive T4 attempts: stop
                end_reason = "t4_repeat"
                t_events.append(dict(kind="T4", attempt=k, refused_by="rule1", **t4_ev, repeat_stop=True))
                break
        lower_only = any(r.output == "lower_bound" for r in readings)
        t_events.append(dict(kind=trigger, attempt=k, refused_by=decision["refused_by"], lower_bound_only=bool(lower_only), phenomena_shown=shown,
                             failing_rows=[r["row"] for r in failing_rows(st)], **t4_ev))
        if not cfg.get("spec"):
            end_reason = "reask_unavailable"
            break
        if cfg.get("reask_limit") and len(reasks) >= int(cfg["reask_limit"]):
            end_reason = "reask_limit"
            break
        ev = evidence(trigger, readings, events, st, cov, shown, action=decision["action"], rule1_refusals=r1_fixable, fixed_constraints=r1_fixed)
        if decision["refused_by"] == "base_level":
            ev["failing_rows"] = ev.get("failing_rows", []) + [dict(row="base_level", why=clean(decision.get("why_base")))]
        if cov.refuses:
            ev["failing_rows"] = ev.get("failing_rows", []) + [dict(row="coverage", why=f"uncovered phenomena {cov.uncovered}")]
        new_pd, new_cp, rec_q, info = PP.reask_rule2_until_accepted(task, cur_pd, ev, c.check, backend_spec=cfg["spec"], cache_dir=cfg.get("cache"), offline=bool(cfg.get("offline")),
                                                                     images=cfg.get("images"), image_note=cfg.get("image_note"))
        small = {kk: rec_q.get(kk) for kk in ("kind", "spec", "model", "reasoning", "backend", "request_hash", "raw_reply_path", "status", "stage", "reasons", "plan_id", "cached", "trigger")}
        reasks.append(dict(attempt=k, trigger=trigger, from_plan=cur_pd.get("plan_id"), to_plan=new_pd.get("plan_id") if new_cp is not None else None, n_asks=info["n_asks"],
                           rejected_attempts=clean(info["attempts"]), proposer_error=info["proposer_error"], record=small, wall_s=info["wall_s"]))
        if new_cp is None:
            end_reason = "reask_error"
            break
        if new_pd.get("plan_id") in seen:
            end_reason = "reask_fixed_point"
            break
        seen.add(new_pd.get("plan_id"))
        cur_pd, cur_cp, cur_wp = new_pd, new_cp, new_cp.wp
    if t4_pending is not None and not stopped and (decision or {}).get("a_admit") is not None and end_reason is not None:
        att_last = attempts[-1]                                             # T4 ended without a plan that Rule 1 admits (re-ask unavailable / limit / error / fixed point / repeat): refuse
        decision = t4_pending
        att_last["decision"] = clean(decision)
    # ------------------------------------------------------------------------------------------------------------------ execute
    pr = weigh.probe
    if c.dry:
        return clean(dict(dry=True, task=task, i=c.i, seed=c.seed, M_true=M, theta_mode=mode, estimate=estimate, decision=None if stopped else decision, stopped=stopped,
                          theta_t={f: list(v) for f, v in cum.theta_t.items()}, coverage=None if cov is None else cov.to_dict(), reask_end=end_reason, n_reasks=len(reasks)))
    ep.mark_contents("after_probe")
    probe_viol = ep.track["viol"]
    w_ok = "PH1" in cum.measured
    theta_total = [pr.M_lo, pr.M_hi] if w_ok else [e + cum.theta_t[FIELD][0], e + cum.theta_t[FIELD][1]]
    sound = bool(theta_total[0] - 1e-9 <= M <= theta_total[1] + 1e-9)
    why = _why(weigh, events)
    rec = dict(task=task, i=c.i, seed=c.seed, probe_method=P.method_of(pr), probe_assumptions=P.method_assumptions(P.method_of(pr)), M_true=M, why=why, rungs=pr.rungs,
               Theta=theta_total, theta_sound=sound, peak_u=round(getattr(pr, "peak_u", 0.0), 4), probe_time_s=getattr(pr, "probe_time_s", None), probe_viol=probe_viol,
               probe_track=dict(pr.track), theta_world=CR._theta(c.world), u_read_max=round(float(getattr(pr, "u_read_max", 0.0)), 4), guard_events=list(getattr(pr, "guard_events", [])),
               probe_summary=getattr(pr, "summary", None), sim_abort=getattr(pr, "sim_abort", None))
    if c.behind:
        rec.update(base_level=None, base_rows=None, base_source=None)
        if lvl_rows is not None:
            rec.update(base_level=(decision or {}).get("base_level"), base_rows=c.rows_json(lvl_rows))
    admitted = bool(decision and decision["a_admit"] is not None and not stopped)
    rec["decision"] = None if stopped else decision
    if c.on_phase:
        c.on_phase("ramp_hold" if admitted else "refused_fallback")
    if admitted:
        if c.behind:
            ep.wp, ep.cc = wp_run, None                  # the carry runs the plan at the picked level (the geometry key ignores the level)
        elif cur_wp is not first_wp:
            ep.wp, ep.cc = cur_wp, None                  # a re-asked plan
        ep.set_belief(0.5 * (decision["contents"][0] + decision["contents"][1]), a_plan=decision["a_admit"])
        m_nom = ep.m_nom
        ep.ctl.mass = pr.m_ff
        pr._ramp(m_nom, 0.3)
        ep.run_phases([("hold", 0.4)])
        ep.ctl.mass = m_nom
        if c.on_phase:
            c.on_phase("nominal")
        carry = CR.run_segments(ep, dict(segments=[]), branch_backups=True)
        rec["carry"] = dict(segments=carry["segments"], end_in_R=carry["end_in_R"], pre_hold_viol=carry["pre_hold_viol"])
        rec["outcome"] = "carried" if (carry["end_in_R"][0] and not any(s["nominal"]["viol"] for s in carry["segments"])) else "carry_failed"
    else:
        pr._ramp(0.0, 0.5)
        fb = tuple(cur_cp.fallback) if cur_cp is not None else tuple(c.plan.get("fallback") or ())
        if fb:
            ep.run_phases(PC.fallback_phases(fb, ep))
            rec["fallback"] = list(fb)
        else:
            ep.run_phases([("hold", 0.3), ("weld", False), ep.retreat_phase(), ("hold", 0.5)])
        okR, whyR = ep.in_R()
        rec["refuse_end_in_R"] = [okR, whyR]
        rec["outcome"] = "refused_in_R" if okR and not ep.track["viol"] else "refused_bad"
    rec["any_U"] = bool(probe_viol or ep.track["viol"] or any(s["nominal"]["viol"] for s in rec.get("carry", {}).get("segments", []))
                        or rec.get("carry", {}).get("pre_hold_viol"))            # 11-12 B12: the ramp / hold between probe and carry
    rec["recovery"] = _UG.recovery_of(rec.get("carry", {}).get("segments"))                 # 11-12 B4 step 1: recorded, not in the outcome
    rec["recovery_bad"] = _UG.recovery_bad(rec["recovery"])
    carried = "carry" in rec
    rec.update(OCC.measure(ep, task, carried, rec["any_U"], rec["carry"]["pre_hold_viol"] if carried else None, refused=not carried, reason=rec["outcome"],
                           end_in_R=tuple(rec["carry"]["end_in_R"] if carried else rec["refuse_end_in_R"])))
    rec["refused_by"] = None if carried else ("T3" if stopped else (decision or {}).get("refused_by") or "probe")
    if end_reason == "reask_error" and not carried and not stopped:      # 11-12 B7: a proposer error ended the Rule 2 loop (not a static refusal)
        rec["refused_by"] = "proposer_error"
    counts = Counter(t["kind"] for t in t_events)
    final_pd = cur_pd
    if mode is not None:
        rec["theta_mode"] = mode
        rec["estimate"] = clean(estimate)
    rec["v2"] = clean(dict(
        plan_id=final_pd.get("plan_id"), first_plan_id=c.plan["plan_dict"].get("plan_id"), phenomena=final_pd["phenomena"], handles={p["id"] if p["id"] != CAT.NEW else f"NEW:{p.get('name')}": p["handle"] for p in final_pd["phenomena"]},
        probes_selected=list(final_pd["probes"]), probes_run=[ev["skill"] for ev in events if ev["kind"] == "ran"],
        probes_skipped=[dict(skill=ev["skill"], kind=ev["kind"], failed_rows=ev.get("failed_rows")) for ev in events if ev["kind"] in ("skipped", "no_probe_needed", "not_run")],
        readings=[reading_json(r) for r in readings], theta_t={f: list(v) for f, v in cum.theta_t.items()}, coverage=None if cov is None else cov.to_dict(),
        n_reasks=len(reasks), reasks=reasks, reask_end=end_reason, t_events=t_events, t_counts=dict(counts), stopped=stopped, attempts=attempts,
        out_of_scope=CAT.OUT_OF_SCOPE_NOTE, guarantee="proof for the admission rows under MODEL; Theta_t from the weigh = sampled (calibration n = 20 per task)"))
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# ------------------------------------------------------------------------------------------------------------------ jobs
def is_v2(plan):
    pd = (plan or {}).get("plan_dict")
    return bool(pd) and "phenomena" in pd and "probes" in pd


def fixed_job(args, method=None, adaptive=True, make_skills=None, rule1=None):
    """args = (task, i, eps_lo, eps_free, plan): the schema-v2 counterpart of probe._e2e_job (fixed base).  Never raises."""
    task, i, eps_lo, eps_free, plan = args
    seed = P._seed("e2e", task, i)
    t0 = time.time()
    try:
        w = P._world(task, seed)
        ep = P._episode(w, plan["wp"])
        ctx = Ctx(task=task, pt=task, i=i, seed=seed, ep=ep, world=w, plan=plan, eps_lo=eps_lo, eps_free=eps_free,
                  static=lambda lo, hi, wp, w_=w: P.static_interval(task, lo, hi, w_, wp=wp), cmax=lambda wp: P.c_admit_max(task, wp=wp),
                  check=lambda pl: PC.compile_plan(pl, task), method=method, adaptive=adaptive, make_skills=make_skills, rule1=rule1)
        return run_world(ctx)
    except Exception as e:
        import traceback
        return dict(task=task, i=i, seed=seed, error=repr(e), tb=traceback.format_exc()[-800:], wall_s=round(time.time() - t0, 1))
