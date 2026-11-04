"""agentchord_probe/nero/execute.py -- execute a cached AgentChord plan on the worlds of carry_cert / baseline_ungated.  SIMULATION ONLY, no LLM.

Per world: CarryEpisode (controller = carry.LeverCtl, a_plan = envelope.A_NOM, payload model m_hat = prior centre [or the cell centre with
m_hat='cell', as baseline_ungated], the planner never sees the hidden fill), 0.4 s load-transfer hold (as carry.world_trial), then THEIR compiled
graph is run with THEIR AgentTaskGraph.run via CompileAgent.act, our atoms.drive and THEIR monitor_functions.

Outcome: completed | plan_rejected (an edge was rejected: unmappable atom / one-arm / arm mismatch / unknown object / bad call / far regrasp; the episode stops
there, state as is) | graph_error (their runtime raised, e.g. a monitor fired with no branch) | budget | error.
Success, two definitions (both recorded):
  ac_success   THEIR style (coarse, as their pour check 'bottle and cup not tipped >= 45 deg'): outcome completed, object tilt < 45 deg, no mug tilted
               >= 45 deg, object resting within AC_XY_TOL of the goal xy and AC_Z_TOL of the goal support height.     # ASSUMED tolerances
  ours         outcome completed AND carry.in_R() (rest set R) AND no entry into U during the whole episode (ep.track['viol'] is None)
               AND no U entry in the pre-hold phase (11-03 item 12c: pre_hold_viol) AND the object within GOAL_TOL_M of the task goal (12b).
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from . import tasks as T

AC_XY_TOL, AC_Z_TOL = 0.05, 0.02          # m, ASSUMED
GOAL_TOL_M = 0.03                         # m, USER 11-03 (was ASSUMED; item 12b): success_ours needs the object within this of the task goal (xy and support-relative z)
ASSUMPTIONS = dict(
    goal_tol=f"USER 11-03 {GOAL_TOL_M} m (xy and z, success_ours, 11-03)",
    carry_physics="config.carry_physics(): fric_comp=True + weld solref WELD_STIFF (11-03 item 6; before: MuJoCo default weld, no friction comp); records carry rec['physics']",
    mon_period_s="ASSUMED 0.1 s: one monitor sample = 20 control steps (their sample = one trajectory waypoint)",
    edge_settle_s="MODEL 0.3 s hold after every motion edge (= carry.segment_phases dwell); --edge-settle 0 = back-to-back",
    arm_tol_m="ASSUMED 0.03 m (max disagreement of the two arms' requested TCP deltas)", regrasp_tol_m="ASSUMED 0.05 m",
    budgets="ASSUMED 120 s sim time and 40 edges per episode", ac_success_tol=f"ASSUMED xy {AC_XY_TOL} m, z {AC_Z_TOL} m, tilt < 45 deg",
    ik_reach="IK miss of the straight-line path is recorded per edge (ik_miss_m), never a rejection", plan_geometry="PLANS destinations ASSUMED (user-confirmed 11-02)",
    grasp="ASSUMED rigid weld (user 11-01), on from t = 0 at the scripted grasp pose", controller="carry.LeverCtl, 16 Nm box, a_plan = A_NOM",
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


def goal_check(ep, task):
    w = ep.world
    gx, gy = T.goal_xy(task, w)
    sup = ("shelf", w.shelf_top) if task == "box" else ("counter", w.counter_top)
    z_goal = sup[1] + (ep.z0 - w.counter_top)
    p = ep.d.xpos[ep.obj]
    tilt = CR_tilt(ep)
    mug_tilt = max([CR_tilt_b(ep, b) for b in ep.cups] or [0.0])
    return dict(xy_err=round(float(np.hypot(p[0] - gx, p[1] - gy)), 4), z_err=round(float(abs(p[2] - z_goal)), 4), tilt_deg=round(tilt, 2),
                mug_tilt_deg=round(mug_tilt, 2), goal_support=sup[0])


def success_ours(done, okR, viol, pre_hold_viol, gc):
    """-> (goal_ok, success_ours, any_U).  11-03 items 12b, 12c: the goal position is checked (xy and z within GOAL_TOL_M) and a U entry in the
    pre-hold phase counts."""
    goal_ok = bool(gc["xy_err"] <= GOAL_TOL_M and gc["z_err"] <= GOAL_TOL_M)
    return goal_ok, bool(done and okR and viol is None and pre_hold_viol is None and goal_ok), (viol is not None or pre_hold_viol is not None)


def CR_tilt(ep):
    from .sim_imports import CR
    return CR._tilt_deg(ep.d.xmat[ep.obj].reshape(3, 3))


def CR_tilt_b(ep, b):
    from .sim_imports import CR
    return CR._tilt_deg(ep.d.xmat[b].reshape(3, 3))


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
    rec = dict(plan_id=plan["plan_id"], task=task, cell=list(cell), i=i, seed=None, outcome=None, reason=None)
    try:
        seed, w = T.world_for(task, cell, i, args.get("seeds", "cert"))
        rec["seed"] = seed
        m_hat = None                                               # prior centre (CarryEpisode default; the planner knows no contents)
        if args.get("m_hat") == "cell":                            # as baseline_ungated: the contents mass the cell centre implies
            from world_model.v2.nero.contents import objects as OB
            n = int(task[4:]) if task.startswith("tray") else 0
            m_hat = n * (OB.CUP["empty"] + 0.5 * (cell[0] + cell[1])) if n else 0.5 * (cell[0] + cell[1])
        ep = CR.CarryEpisode(w, a_plan=E.A_NOM, m_hat=m_hat)       # physics defaults = config.carry_physics() (item 6)
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
        gc = goal_check(ep, task)
        rec.update(edges=env.log, monitor_errors=env.monitor_errors, end_in_R=okR, R_fail=whyR, goal=gc, track={k: (v if k == "viol" else round(float(v), 4)) for k, v in ep.track.items()},
                   sim_s=round(ep.k * CR.DT, 2))
        rec["path_log"] = [dict(p, err_last=round(p["err_last"], 5), err_max=round(p["err_max"], 5), margin_min=round(p["margin_min"], 4)) for p in ep.path_log]   # 11-03: every _add_path (IK miss over all waypoints, joint margin); track also carries path_ik_err_max / path_q_margin_min / path_unreachable
        ex = [e for e in env.log if e["status"] in ("executed", "aborted_by_monitor")]
        rec.update(n_edges_executed=len(ex), n_rejected=sum(1 for e in env.log if str(e["status"]).startswith(("rejected", "unmappable"))),
                   rejections=[dict(edge=e["edge_id"], status=e["status"], detail=e.get("detail")) for e in env.log
                               if str(e["status"]).startswith(("rejected", "unmappable"))],
                   monitors_fired=[dict(edge=e["edge_id"], **e["monitor"]) for e in env.log if e.get("monitor")],
                   n_recoveries_run=sum(1 for e in ex if e["is_recovery"]))
        done = rec["outcome"] == "completed"
        rec["ac_success"] = bool(done and gc["tilt_deg"] < 45 and gc["mug_tilt_deg"] < 45 and gc["xy_err"] <= AC_XY_TOL and gc["z_err"] <= AC_Z_TOL)
        rec["goal_ok"], rec["success_ours"], rec["any_U"] = success_ours(done, okR, ep.track["viol"], rec["pre_hold_viol"], gc)
    except Exception as e:                                         # a crash is reported, never silently dropped
        import traceback
        rec.update(outcome="error", reason=repr(e), tb=traceback.format_exc()[-1500:])
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec
