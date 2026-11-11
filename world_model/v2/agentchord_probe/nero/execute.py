"""agentchord_probe/nero/execute.py -- execute a cached AgentChord plan on the worlds of carry_cert / baseline_ungated.  SIMULATION ONLY, no LLM.

Per world: CarryEpisode (controller = carry.LeverCtl, a_plan = envelope.A_NOM, payload model m_hat = prior centre [or the cell centre with
m_hat='cell', as baseline_ungated], the planner never sees the hidden fill), 0.4 s load-transfer hold (as carry.world_trial), then THEIR compiled
graph is run with THEIR AgentTaskGraph.run via CompileAgent.act, our atoms.drive and THEIR monitor_functions.

Outcome: completed | plan_rejected (an edge was rejected: unmappable atom / one-arm / arm mismatch / unknown object / bad call / far regrasp; the episode stops
there, state as is) | graph_error (their runtime raised, e.g. a monitor fired with no branch) | budget | error.
Success (USER 11-09, ONE rule for both methods, stack.outcome): rec['result'] = Outcome.record(): success = finished AND the task's goal terms
(AgentChord's ac_success: object tilt < 45, every mug tilt < 45, xy_err <= 0.05, z_err <= 0.02; stack/tasks.py); unsafe = any U entry (incl. pre-hold);
in_R = end at rest in R; safe_success = success AND not unsafe.  rec['goal'] = the goal Row, rec['measured'] = the raw xy_err / z_err / tilt_deg / mug_tilt_deg.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from world_model.v2.stack import tasks as TK

from . import tasks as T

def _tau_note():
    """the torque box this process actually uses (config.tau_limit(), one source).  11-07 fix (clarity report 11-06 (f)9): the records said
    "16 Nm box" while the 11-05 runs used the 100 Nm MJCF default."""
    from world_model.v2.nero import config as NC
    src = "BIMANUAL_TAU_MIT16=1: MIT command range, ASSUMED" if NC.TAU_MIT16 else "MJCF forcerange, MODEL, USER 11-05"
    return f"{float(NC.tau_limit()[0]):g} Nm torque box ({src})"


ASSUMPTIONS = dict(
    goal="stack/tasks.py goal terms = AgentChord ac_success (ASSUMED tolerances, adopted USER 11-09): xy <= 0.05 m, z <= 0.02 m, tilt < 45 deg",
    carry_physics="config.carry_physics(): fric_comp=True + weld solref WELD_STIFF (11-03 item 6; before: MuJoCo default weld, no friction comp); records carry rec['physics']",
    mon_period_s="ASSUMED 0.1 s: one monitor sample = 20 control steps (their sample = one trajectory waypoint)",
    edge_settle_s="MODEL 0.3 s hold after every motion edge (= carry.segment_phases dwell); --edge-settle 0 = back-to-back",
    arm_tol_m="ASSUMED 0.03 m (max disagreement of the two arms' requested TCP deltas)", regrasp_tol_m="ASSUMED 0.05 m",
    budgets="ASSUMED 120 s sim time and 40 edges per episode",
    ik_reach="IK miss of the straight-line path is recorded per edge (ik_miss_m), never a rejection", plan_geometry="PLANS destinations ASSUMED (user-confirmed 11-02)",
    grasp="ASSUMED rigid weld (user 11-01), on from t = 0 at the scripted grasp pose", controller=f"carry.LeverCtl, {_tau_note()}, a_plan = A_NOM",
    worlds="theta ~ uniform per cell (carry_cert.sample_world); prior box = union of the cells", perception="none (planner sees a render + facts)")


def _wrap_edges(graph, env):
    """tag each compiled edge's actions so drive() knows the edge id / is_recovery (wrappers only; their graph objects are not modified)."""
    for eid, e in graph.edges.items():
        for slot in ("left_arm_action", "right_arm_action"):
            a = getattr(e, slot)
            if a is None:
                continue

            def f(*args, _a=a, _id=eid, _rec=e.is_recovery, **kw):
                env.cur_edge = (_id, bool(_rec))
                return _a(*args, **kw)
            f.func, f.keywords = _a_func(a), getattr(a, "keywords", {})
            setattr(e, slot, f)


def _a_func(a):
    return getattr(a, "func", a)


def run_world(args):
    """args: dict(plan, task, cell, i, seed, m_hat('prior'|'cell'), mon_period, seeds).  -> record dict (never raises)."""
    from . import shim
    shim.install(scratch_dir=pathlib.Path(args["plan"]).parent / "_unused_db")
    from .sim_imports import CR, E, SC
    from .atoms import BudgetExceeded, PlanRejected
    from .env import NeroEnv
    t0 = time.time()
    task, cell, i = args["task"], tuple(args["cell"]), args["i"]
    plan = json.loads(pathlib.Path(args["plan"]).read_text())
    rec = dict(plan_id=plan["plan_id"], task=task, cell=list(cell), i=i, seed=None, outcome=None, reason=None,
               proposer=plan.get("proposer"), plan_backend=plan.get("backend"), plan_models=plan.get("models"))
    try:
        if args.get("world") is not None:                          # same_world_trial: (seed, world) given
            seed, w = args["world"]
        else:
            seed, w = T.world_for(task, cell, i, args.get("seeds", "cert"))
        rec["seed"] = seed
        m_hat = args.get("m_hat_kg")                               # same_world_trial: prior mid-contents (as baseline_ungated._trial); else None = prior centre (CarryEpisode default; the planner knows no contents)
        if args.get("m_hat") == "cell":                            # as baseline_ungated: the contents mass the cell centre implies
            from world_model.v2.nero.contents import objects as OB
            n = int(task[4:]) if task.startswith("tray") else 0
            m_hat = n * (OB.CUP["empty"] + 0.5 * (cell[0] + cell[1])) if n else 0.5 * (cell[0] + cell[1])
        ep = CR.CarryEpisode(w, a_plan=E.A_NOM, m_hat=m_hat, planner="scripted")       # physics defaults = config.carry_physics() (item 6); AgentChord runs its own plans (11-05: immune to BIMANUAL_PLANNER)
        ep.run_phases([("hold", 0.4)])
        rec["pre_hold_viol"] = ep.track["viol"]                    # 11-03 item 12c: a U entry during the load-transfer hold counts
        rec["physics"] = ep.physics_flags()
        ep.reset_track()
        env = NeroEnv(ep, mon_period=args.get("mon_period", 0.1), edge_settle=args.get("edge_settle", 0.3))
        rec["theta"] = CR._theta(w)
        rec["truth_total_kg"] = round(float(SC.object_truth(ep.m, ep.d)["total_mass"]), 3)
        from embodichain.agents.hierarchy.compile_agent import CompileAgent
        from embodichain.agents.agentchord import graph_spec as GS
        from embodichain.agents.agentchord import monitor_functions as MF
        from embodichain.agents.agentchord.agent_graph import AgentTaskGraph
        from .atoms import ATOMS_MODULE
        bundle = GS.load_agent_graph_bundle(plan["graph"])
        graph = GS.compile_agent_graph_spec(bundle["task_graph"], bundle.get("recovery_graph"), env=env, graph_cls=AgentTaskGraph,
                                            action_module=ATOMS_MODULE, monitor_module=MF)
        _wrap_edges(graph, env)
        try:
            graph.run(env=env)                                     # THEIR AgentTaskGraph.run -> our drive
            rec["outcome"] = "completed"
            ep.run_phases([("hold", 0.5)])                         # settle (as carry.py's release tail)
        except PlanRejected as r:
            rec.update(outcome="plan_rejected", reason=r.reason)
        except BudgetExceeded as b:
            rec.update(outcome="budget", reason=str(b))
        except RuntimeError as e:                                  # their runtime (no recovery branch, max_transitions, ...)
            rec.update(outcome="graph_error", reason=repr(e))
        okR, whyR = ep.in_R()
        rec.update(edges=env.log, monitor_errors=env.monitor_errors, end_in_R=okR, R_fail=whyR, track={k: (v if k == "viol" else round(float(v), 4)) for k, v in ep.track.items()},
                   sim_s=round(ep.k * CR.DT, 2))
        rec["path_log"] = [dict(p, err_last=round(p["err_last"], 5), err_max=round(p["err_max"], 5), margin_min=round(p["margin_min"], 4)) for p in ep.path_log]   # 11-03: every _add_path (IK miss over all waypoints, joint margin); track also carries path_ik_err_max / path_q_margin_min / path_unreachable
        ex = [e for e in env.log if e["status"] in ("executed", "aborted_by_monitor")]
        rec.update(n_edges_executed=len(ex), n_rejected=sum(1 for e in env.log if str(e["status"]).startswith(("rejected", "unmappable"))),
                   rejections=[dict(edge=e["edge_id"], status=e["status"], detail=e.get("detail")) for e in env.log
                               if str(e["status"]).startswith(("rejected", "unmappable"))],
                   monitors_fired=[dict(edge=e["edge_id"], **e["monitor"]) for e in env.log if e.get("monitor")],
                   n_recoveries_run=sum(1 for e in ex if e["is_recovery"]))
        done = rec["outcome"] == "completed"
        from world_model.v2.adapters import outcome_carry as OCC
        viol, pre = ep.track["viol"], rec["pre_hold_viol"]
        rec.update(OCC.measure(ep, task, done, viol is not None or pre is not None, pre, refused=rec["outcome"] == "plan_rejected", reason=str(rec.get("reason") or ""),
                               end_in_R=(okR, whyR)))
        rec["any_U"] = viol is not None or pre is not None
    except Exception as e:                                         # a crash is reported, never silently dropped
        import traceback
        rec.update(outcome="error", reason=repr(e), tb=traceback.format_exc()[-1500:])
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def same_world_trial(task, i, plan_dir, mon_period=0.1, edge_settle=0.3, level=None):
    """THEIR cached compiled graph on exactly the world probe._world(task, probe._seed("e2e", task, i)) (the worlds of nero/contents/same_worlds.py).
    U includes the payload rating: same_worlds._init_worker (strict_claw=True, T_cap=None) is applied to CarryEpisode first.  Payload feedforward
    mass m_hat = prior mid-contents (ASSUMED; as baseline_ungated._trial called by same_worlds 'theirs': the planner has no estimate).
    -> dict with method='theirs', any_U, first_U {seg, kind} | None, outcome, result {success, unsafe, in_R, safe_success}, plan_id, model/reasoning (proposer), seed_e2e.
    `outcome` is run_world's (completed | plan_rejected | graph_error | budget | error); same_worlds.summary's 'carried' column is for 'ours' only.
    plan_dir: directory holding plan.json (or the plan.json path)."""
    from world_model.v2.nero.contents import same_worlds as SW, probe as P
    SW._init_worker()
    if task.endswith("_behind"):                                   # table-behind tasks (swerve base): behind.run_world on world i of smoke.behind_world; level = the base speed cap
        from . import behind as BH
        return BH.same_world_trial(task, i, plan_dir, level, mon_period, edge_settle)
    pj = pathlib.Path(plan_dir)
    pj = pj if pj.suffix == ".json" else pj / "plan.json"
    seed = P._seed("e2e", task, i)
    w = P._world(task, seed)
    lo, hi = TK.prior_box(task)
    m_hat = 0.5 * (lo + hi)                                      # ASSUMED, identical to same_worlds 'theirs' (BU._trial(..., m_hat=0.5*(lo+hi)))
    r = run_world(dict(plan=str(pj), task=task, cell=[lo, hi], i=i, world=(seed, w), m_hat_kg=m_hat, mon_period=mon_period, edge_settle=edge_settle))
    fu = None
    if r.get("any_U"):                                           # first U entry: the pre-hold phase, else the first edge whose cumulative track has viol
        if r.get("pre_hold_viol"):
            fu = dict(seg="pre_hold", kind=r["pre_hold_viol"].get("kind"))
        else:
            e = next((e for e in r.get("edges", []) if (e.get("track") or {}).get("viol")), None)
            v = (e["track"]["viol"] if e else None) or (r.get("track") or {}).get("viol") or {}
            fu = dict(seg=e["edge_id"] if e else None, kind=v.get("kind"))
    pr = r.get("proposer") or {}
    r.update(method="theirs", seed_e2e=seed, first_U=fu, plan_model=pr.get("model"), plan_reasoning=pr.get("reasoning"))
    return r
