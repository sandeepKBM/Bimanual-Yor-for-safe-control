"""nero/contents/behind_job.py -- OUR per-world job for the table-behind tasks (tray6_behind, box_behind): probe -> Theta -> static admission -> base level -> carry.  SIMULATION ONLY.
Sibling of probe._e2e_job (called from same_worlds._job for tasks whose TaskDef.base is set); the fixed-base job is NOT touched.

  world      nero/base/smoke.behind_world(task, i): the ONE world list of both methods (AgentChord's behind.run_world uses the same function).
  episode    CarryEpisode(world, base="swerve", goal_support="counter2", wp=...): the probe runs in the SAME swerve scene the carry uses (base standing still; the base driver
             is idle, no base_move runs during the probe).  The probe code itself is unchanged.
  Theta      the probe's contents interval [c_lo, c_hi] (as probe._e2e_job).
  static     carry_cert.static_cell(task=*_behind, spec = interval_spec(base task, c_lo, c_hi), wp): arm rows on the planned path + the support row on counter2
             (carry_cert.support_row, base-task branch).  Refused -> let go in place (the plan's fallback).
  base level adapters/base_envelope.pick_level: the highest of L1..L3 at which base_tip, base_torque and (tray) base_open_content ALL pass on Theta with the "measured"
             acceleration box (guarantee: sampled, n of the base card); none -> refused (reason recorded).  Every row of every level is recorded.
  carry      the proposer's plan compiled with that level (stack/proposer/compile.compile_plan(level=)) -> CarryEpisode.wp -> carry.run_segments (the closed-chain planner;
             its base_move runs through the executor phase ("base_move", ref)).
  outcome    adapters/outcome_carry.measure (stack.outcome: success = finished AND goal AND never in U AND at rest in R).  U includes the base flags (nero/base/monitors.py)
             and the per-claw payload row (same_worlds._init_worker wraps CarryEpisode with strict_claw=True).
"""
from __future__ import annotations

import dataclasses
import functools
import time

import numpy as np

from world_model.v2.adapters import base_envelope as BE, outcome_carry as OCC
from world_model.v2.nero.base import smoke as SM
from world_model.v2.nero.contents import carry as CR, carry_cert as CC, probe as P
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.proposer import compile as PC, schema as S

BASE_FLAGS = ("wheel_liftoff", "base_world_contact", "load_world_contact", "route_unfinished")      # nero/base/monitors.py + drive.py (as agentchord_probe.nero.behind)
LEVELS = ("L1", "L2", "L3")
SOURCE = "measured"          # acceleration box of the base rows that ADMITS: the base card (MEASURED, sampled); "command" never admits (base_tip.py header)


def base_task(task):
    return task[:-len("_behind")]


def _static(task, c_lo, c_hi, w, wp):
    return CC.static_cell(task, (round(c_lo, 4), round(c_hi, 4)), w, spec=P.interval_spec(base_task(task), c_lo, c_hi), wp=wp)


@functools.lru_cache(maxsize=None)
def c_admit_max(task, wp, tol=0.01):
    """as probe.c_admit_max (bisection on the point interval [c, c]) with the behind task's static rows on wp; the default world (the verdict does not depend on the hidden world)."""
    bt = base_task(task)
    w = SM.task_world(bt)
    lo, hi = P.prior_contents(bt)
    refuse = lambda c: _static(task, c, c, w, wp)["action"].startswith("refuse")
    if refuse(lo):
        return None
    if not refuse(hi):
        return hi
    a, b = lo, hi
    while b - a > tol:
        c = 0.5 * (a + b)
        a, b = (a, c) if refuse(c) else (c, b)
    return a


def _row(r):
    return dict(ok=bool(r.ok), margin=None if r.margin is None else float(r.margin), unit=r.unit, guarantee=r.guarantee, n=r.n, why=r.why,
                worst=CR._jsonable(r.worst))


def _rows_json(rows):
    return {L: {k: _row(v) for k, v in rr.items()} for L, rr in rows.items()}


def _first_viol(ep, probe_viol):
    v = probe_viol or ep.track["viol"]
    return None if not v else dict(kind=v.get("kind"), t=v.get("t"))


def behind_job(args, method=None, adaptive=True):
    """args = (task, i, eps_lo, eps_free, plan).  task = tray6_behind | box_behind.  plan: dict(wp = the compiled WaypointPlan (any level), plan_dict = the proposer's accepted
    plan (compile_plan recompiles it at the world's level), fallback, belief, outside_b, plan_id).  -> record dict (never raises)."""
    task, i, eps_lo, eps_free, plan = args
    bt = base_task(task)
    t0 = time.time()
    seed = None
    try:
        seed, w = SM.behind_world(task, i)
        from world_model.v2.nero.contents import probe_select as PS
        if PS.is_v2(plan):                                   # schema-v2 plan (11-09): the LLM's probes -> coverage -> Rule 1 + base level -> Rule 2 re-ask -> carry
            return v2_job(task, i, seed, w, eps_lo, eps_free, plan, method, adaptive)
        wp0 = plan["wp"]
        ep = CR.CarryEpisode(w, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF, wp=wp0, base="swerve", goal_support=TK.task(task).goal_support)
        M = P.true_mass(ep)
        cmax = c_admit_max(task, wp0)
        M_stop = None if cmax is None else P.empty_mass(bt) + cmax
        pr = P.make_probe(ep, bt, eps_lo, eps_free, rng=np.random.default_rng(seed), noise=True, method=method, adaptive=adaptive)
        c0 = P.prior_contents(bt)
        if not _static(task, *c0, w, wp0)["action"].startswith("refuse"):
            why = "no_probe_needed"                          # verdict "carry" on all of Theta_0 (static rows only; the base level is picked below on Theta)
            pr.M_lo, pr.M_hi = P.empty_mass(bt) + c0[0], P.empty_mass(bt) + c0[1]
        else:
            why = pr.run(M_stop=M_stop)
        probe_viol = ep.track["viol"]
        sound = bool(pr.M_lo - 1e-9 <= M <= pr.M_hi + 1e-9)
        rec = dict(task=task, i=i, seed=seed, probe_method=P.method_of(pr), probe_assumptions=P.method_assumptions(P.method_of(pr)), M_true=M, why=why,
                   rungs=pr.rungs, Theta=[pr.M_lo, pr.M_hi], theta_sound=sound, peak_u=round(getattr(pr, "peak_u", 0.0), 4), probe_time_s=getattr(pr, "probe_time_s", None),
                   probe_viol=probe_viol, probe_track=dict(pr.track), theta_world=CR._theta(w), base_level=None, base_rows=None,
                   u_read_max=round(float(getattr(pr, "u_read_max", 0.0)), 4), guard_events=list(getattr(pr, "guard_events", [])), probe_summary=getattr(pr, "summary", None),
                   sim_abort=getattr(pr, "sim_abort", None), base_source=SOURCE)
        decision, wp = None, None
        if why in ("free", "no_probe_needed"):
            c_lo, c_hi = pr.M_lo - P.empty_mass(bt), pr.M_hi - P.empty_mass(bt)
            st = _static(task, c_lo, c_hi, w, wp0)
            decision = dict(action=st["action"], a_admit=st["a_admit"], u_nom=st["u_nom"], contents=[c_lo, c_hi], refused_by=None)
            if decision["a_admit"] is None:
                decision["refused_by"] = "static"
            if plan.get("belief"):                           # Rule 2 (as probe._e2e_job): the probe set inside the proposer's belief B?
                rec["probe_in_B"] = PC.probe_in_belief(bt, plan["belief"], c_lo, c_hi)
                if not rec["probe_in_B"]["inside"] and plan.get("outside_b", "refuse") == "refuse" and decision["a_admit"] is not None:
                    decision = dict(decision, action="refuse (probe set outside B)", a_admit=None, why_B=rec["probe_in_B"], refused_by="outside_B")
            if decision["a_admit"] is not None:              # base level: highest of L1..L3 with every base row passing on Theta
                lvl, rows = BE.pick_level(task, P.interval_spec(bt, c_lo, c_hi), LEVELS, SOURCE)
                rec.update(base_level=lvl, base_rows=_rows_json(rows))
                if lvl is None:
                    decision = dict(decision, action="refuse (base level)", a_admit=None, refused_by="base_level",
                                    why_base={L: [k + ": " + v.why for k, v in rr.items() if not v.ok] for L, rr in rows.items()})
                else:
                    try:
                        wp = (PC.compile_plan(plan["plan_dict"], task, build=False, level=lvl).wp if plan.get("plan_dict")
                              else dataclasses.replace(wp0, base=tuple((a, r, lvl) for a, r, _ in wp0.base)))
                    except S.PlanRejected as e:
                        decision = dict(decision, action="refuse (compile at level)", a_admit=None, refused_by="compile", why_compile=[e.stage, list(e.reasons)])
        rec["decision"] = decision
        if decision and decision["a_admit"] is not None:
            ep.wp, ep.cc = wp, None                          # the carry runs the plan at the picked level (the geometry key ignores the level)
            ep.set_belief(0.5 * (decision["contents"][0] + decision["contents"][1]), a_plan=decision["a_admit"])
            m_nom = ep.m_nom
            ep.ctl.mass = pr.m_ff
            pr._ramp(m_nom, 0.3)
            ep.run_phases([("hold", 0.4)])
            ep.ctl.mass = m_nom
            carry = CR.run_segments(ep, dict(segments=[]), branch_backups=True)
            rec["carry"] = dict(segments=carry["segments"], end_in_R=carry["end_in_R"], pre_hold_viol=carry["pre_hold_viol"])
            rec["outcome"] = "carried" if (carry["end_in_R"][0] and not any(s["nominal"]["viol"] for s in carry["segments"])) else "carry_failed"
        else:
            pr._ramp(0.0, 0.5)
            if plan.get("fallback"):                          # the proposer's fallback: the object is still on its start support (counter 1)
                ep.run_phases(PC.fallback_phases(plan["fallback"], ep))
                rec["fallback"] = list(plan["fallback"])
            else:
                ep.run_phases([("hold", 0.3), ("weld", False), ep.retreat_phase(), ("hold", 0.5)])
            okR, whyR = ep.in_R()
            rec["refuse_end_in_R"] = [okR, whyR]
            rec["outcome"] = "refused_in_R" if okR and not ep.track["viol"] else "refused_bad"
        rec["any_U"] = bool(probe_viol or ep.track["viol"] or any(s["nominal"]["viol"] for s in rec.get("carry", {}).get("segments", [])))
        carried = "carry" in rec
        rec.update(OCC.measure(ep, task, carried, rec["any_U"], rec["carry"]["pre_hold_viol"] if carried else None, refused=not carried, reason=rec["outcome"],
                               end_in_R=tuple(rec["carry"]["end_in_R"] if carried else rec["refuse_end_in_R"])))
        bl = getattr(ep, "base_log", [])
        fl = set(ep.bmon.flagged)
        rec["base"] = dict(level=rec["base_level"], n_base_edges=len(bl), n_aborted=sum(1 for b in bl if b.get("aborted")),
                           route_end_err_mm=[b.get("err_mm") for b in bl], route_end_err_deg=[b.get("err_deg") for b in bl],
                           U_flags={k: (k in fl) for k in BASE_FLAGS}, first_viol=CR._jsonable(ep.track["viol"]), base_end_pose=[float(v) for v in ep.sb.pose()])
        rec["first_U"] = _first_viol(ep, probe_viol)
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    except Exception as e:
        import traceback
        return dict(task=task, i=i, seed=seed, error=repr(e), tb=traceback.format_exc()[-1200:])


def v2_job(task, i, seed, w, eps_lo, eps_free, plan, method=None, adaptive=True, make_skills=None):
    """the schema-v2 per-world run for a table-behind task (probe_select.run_world with this module's static rows, base level pick and compile-at-level) + the base summary."""
    from world_model.v2.nero.contents import probe_select as PS
    bt = base_task(task)
    t0 = time.time()
    try:
        wp0 = plan["wp"]
        ep = CR.CarryEpisode(w, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF, wp=wp0, base="swerve", goal_support=TK.task(task).goal_support)
        ctx = PS.Ctx(task=task, pt=bt, i=i, seed=seed, ep=ep, world=w, plan=plan, eps_lo=eps_lo, eps_free=eps_free,
                     static=lambda lo, hi, wp: _static(task, lo, hi, w, wp), cmax=lambda wp: c_admit_max(task, wp),
                     check=lambda pl: PC.compile_plan(pl, task, level="L1"), method=method, adaptive=adaptive, behind=True,
                     level=lambda lo, hi: BE.pick_level(task, P.interval_spec(bt, lo, hi), LEVELS, SOURCE),
                     at_level=lambda pd, lvl: PC.compile_plan(pd, task, build=False, level=lvl).wp, rows_json=_rows_json, make_skills=make_skills)
        rec = PS.run_world(ctx)
        rec["base_source"] = SOURCE
        bl = getattr(ep, "base_log", [])
        fl = set(ep.bmon.flagged)
        rec["base"] = dict(level=rec["base_level"], n_base_edges=len(bl), n_aborted=sum(1 for b in bl if b.get("aborted")),
                           route_end_err_mm=[b.get("err_mm") for b in bl], route_end_err_deg=[b.get("err_deg") for b in bl],
                           U_flags={k: (k in fl) for k in BASE_FLAGS}, first_viol=CR._jsonable(ep.track["viol"]), base_end_pose=[float(v) for v in ep.sb.pose()])
        rec["first_U"] = _first_viol(ep, rec["probe_viol"])
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    except Exception as e:
        import traceback
        return dict(task=task, i=i, seed=seed, error=repr(e), tb=traceback.format_exc()[-1200:])
