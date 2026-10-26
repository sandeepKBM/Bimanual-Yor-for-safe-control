"""tamp/run_tamp.py -- plan -> execute -> REPLAN from the current sim state when an action fails / stalls.  SIMULATION ONLY, CPU only.

    python -m world_model.v2.house.tamp.run_tamp --filter none --no-people --seed 0
    python -m world_model.v2.house.tamp.run_tamp --filter gatekeeper --people --seed 3 --max-replans 3

Loop: world_from_sim(sim) -> planner.solve_task(goal) -> Executor.run_action per plan step.  On a non-ok ActionResult the world is rebuilt from the
CURRENT sim state (held=(item, Grasp) if the sim still holds the item; mid-route start confs are served by partial routes) and the same goal is re-planned,
at most --max-replans times, within a total sim budget --max-s.  One JSON per episode in --out.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..gatekeeper.belief import GKConfig
from ..teleop import geom as GM
from ..teleop.sim import Cmd, TICK, TeleopSim
from . import executor as EX
from . import hop as HOP
from . import segments as SEG
from . import people_pred as PP
from . import planner as TP
from . import world as W
from .exec_smoke import make_filter
from .streams import Streams
from .types import Cert, Conf, Route, plan_to_json

OUT = Path("/data/local/ss5772/molmospaces/tamp/run_tamp")
_GK0 = GKConfig()                   # defaults shown / used by the --pred-ev, --pred-ap CLI flags


def _reject_counts(stream_log):
    c = Counter()
    for e in stream_log:
        if not e["ok"]:
            c[f"{e['stream']}: {str(e['reason'])[:60]}"] += 1
    return dict(c)


def _solve(sim, goal, held, plan_time, legacy=True, cert="rest", people_pred="frozen", t_offset=0.0, seg_cache=None, ev_scale=1.0,
           robot_pose=None, start_v=None, info=None):
    """legacy=True (--recovery none): the original people-free certification, no shifted routes (bit-identical to the pre-recovery code).
    cert="segments": routes carry timed gatekeeper-format segments at a planner-chosen speed (tamp/segments.py); "rest" = the original from-rest samples.
    people_pred / t_offset / seg_cache: see Streams (time-indexed people prediction of the immediate route; defaults = frozen = unchanged).
    robot_pose / start_v (plan-prediction monitor replan): the robot's start conf and body twist = the state the committed trajectory predicts at t_flag + L; None = the sim's.
    info: optional dict that receives the planner World (info["world"]) so the caller can read the people the plan was certified for."""
    world = W.world_from_sim(sim, held=held, robot_pose=robot_pose)
    if info is not None:
        info["world"] = world
    if people_pred != "frozen" or t_offset or seg_cache is not None:
        assert cert == "segments" and not legacy
        return TP.solve_task(world, goal, max_time=plan_time, streams=Streams(world, cert_mode=cert, people_pred=people_pred, t_offset=t_offset, seg_cache=seg_cache, people_ev_scale=ev_scale, start_v=start_v))
    if legacy:
        res = TP.solve_task(world, goal, max_time=plan_time, streams=Streams(world, near_m=0.0, lateral_offsets=(), cert_mode=cert))
    elif cert != "rest":
        res = TP.solve_task(world, goal, max_time=plan_time, streams=Streams(world, cert_mode=cert))
    else:
        res = TP.solve_task(world, goal, max_time=plan_time)
    return res


def _seg_summary(results, plans, cert):
    """episode-level segment bookkeeping (machine-written): planned vs admitted-at-execution vs refused, speeds per executed route.  None for cert="rest"."""
    if cert != "segments":
        return None
    routes = []
    for r in results:
        sg = (r.get("info") or {}).get("segments")
        if sg:
            routes.append(dict(action=r["name"], t_start=r["t_start"], t_end=r["t_end"], action_status=r["status"], **sg))
    return dict(cert=cert, n_routes_executed=len(routes), n_segments_planned=sum(x["n_segments_planned"] for x in routes),
                n_segments_admitted=sum(x["n_segments_admitted"] for x in routes), n_refused=sum(x["n_refused"] for x in routes),
                n_segment_refused_actions=sum(1 for r in results if r["status"] == "segment_refused"),
                speeds_per_route=[x["speeds_used"] for x in routes], routes=routes,
                planned_speeds_per_plan=[[a.get("v_max") for e in (p["plan"] or []) for a in e["args"] if isinstance(a, dict) and a.get("_type") == "Route"] for p in plans])


REST_V = 0.03                       # m/s: "at rest" (same threshold the executor uses for its stopped timer)
RETREAT_MIN_M = 0.30                # m: a rest pose closer than this to the current pose is not a retreat target


class _WaitOp:
    phase = "WAIT"


def _hold_at_rest(sim, filt, ex, wait_s):
    """zero command through the SAME filter for wait_s (rest is the T1 backup).  Rows are logged in the executor so episode metrics include it."""
    t_end = sim.t + wait_s
    while sim.t < t_end:
        cmd = Cmd()
        safe, info = filt.filter(sim, cmd)
        o = sim.tick(safe)
        ex._log(o, info, cmd, safe, _WaitOp())


def _coast_ticks(sim, filt, ex, n_ticks, until_rest=False, max_extra_s=20.0):
    """advance n_ticks with a ZERO operator command through the filter, which is in coast mode (it follows only the committed trajectory).  until_rest: keep going (at most
    max_extra_s more) until the base speed is <= REST_V.  Rows are logged in the executor.  -> ticks run."""
    n = 0
    for _ in range(max(0, int(n_ticks))):
        cmd = Cmd()
        safe, info = filt.filter(sim, cmd)
        ex._log(sim.tick(safe), info, cmd, safe, _WaitOp())
        n += 1
    if until_rest:
        for _ in range(int(max_extra_s / TICK)):
            if float(np.abs(sim.base.v).max()) <= REST_V:
                break
            cmd = Cmd()
            safe, info = filt.filter(sim, cmd)
            ex._log(sim.tick(safe), info, cmd, safe, _WaitOp())
            n += 1
    return n


def _world_people(world):
    """[(c, v_hat)] the plan was certified for (world.bel_people: the belief at plan time)."""
    b = world.bel_people
    if b is None or not len(b.people):
        return []
    vel = b.people_vel if len(b.people_vel) == len(b.people) else tuple(np.zeros(2) for _ in b.people)
    return [(np.asarray(c, float)[:2], np.asarray(v, float)[:2]) for c, v in zip(b.people, vel)]


def _plan_sources(plan):
    return [getattr(a, "source", None) for _, args in (plan or []) for a in args if isinstance(a, Route)]


def _retreat_candidates(st, q1, q2):
    """candidate routes q1 -> q2: the stream's candidates, else the sub-polyline of a scene route between the projections of q1 and q2 (travelled either way)."""
    c = list(st._candidate_routes(q1, q2))
    if c:
        return c
    a, b = np.array([q1.x, q1.y]), np.array([q2.x, q2.y])
    out = []
    for name, pts in st.w.routes:
        P = np.asarray(pts, float)
        pa, pb = st._project(P, a), st._project(P, b)
        if pa is None or pb is None or pa[1] > 0.6 or pb[1] > 0.6:
            continue
        (ia, _, xa), (ib, _, xb) = pa, pb
        if (ia, np.hypot(*(xa - P[ia]))) <= (ib, np.hypot(*(xb - P[ib]))):          # q2 is ahead of q1 along the route
            seq = [tuple(a)] + [tuple(map(float, v)) for v in P[ia + 1:ib + 1]] + [tuple(b)]
            rev = False
        else:                                                                       # q2 is behind: travel back along the route
            seq = [tuple(a)] + [tuple(map(float, v)) for v in P[ib + 1:ia + 1][::-1]] + [tuple(b)]
            rev = st.w.back_mode == "reverse"
        seq = [p for k, p in enumerate(seq) if k == 0 or math.hypot(p[0] - seq[k - 1][0], p[1] - seq[k - 1][1]) > 1e-6]
        if len(seq) >= 2:
            out.append((tuple(seq), rev, f"retreat:{name}"))
    return out


CORRIDOR_X = 2.13449            # h5 out_route corridor centreline (door2 -> cafe), x = const for y in [2.5, 4.49]


def _h5_with_stand(offset_m):
    """h5 SceneSpec factory with p_stand moved to `offset_m` from the corridor centreline (same y = 3.5, same side, -x).
    Scenario 2 of docs/paper/experiment_plan_2026-10-16.md (livelock curve vs offset).  Default h5 = offset 1.03 m."""
    import numpy as np
    from ..teleop import scene as SC
    base = SC._h5

    def make():
        sc = base()
        for i, p in enumerate(sc.people):
            if p.name == "p_stand":
                pts = p.pts.copy()
                pts[:, 1] = CORRIDOR_X - offset_m
                sc.people[i] = SC.PersonTrack(p.name, pts, p.dynamic)
        return sc
    return make


WALKER_Y = 3.8                  # h5 p_walker crosses the corridor at y = 3.8 (x from 3.6 to 0.9)
WALKER_X_END = 0.9


def _h5_with_walker(delay_s=2.0, speed=None, x0=3.6, stand_offset=None):
    """h5 SceneSpec factory with a tunable p_walker.  Polyline: stands at (x0, 3.8) until t = delay_s, then walks at constant `speed` [m/s] in -x along y = 3.8 (across
    the corridor out_route at x = 2.134, between door2 y=4.49 and the counter y=2.0) to (0.9, 3.8), where it stops.  Defaults reproduce the scene's h5 walker
    (delay 2 s, 8 s for 2.7 m = 0.3375 m/s, from x = 3.6).  The walker crosses the route centreline at t = delay_s + (x0 - 2.134) / speed (undisturbed).  delay_s = 0 and
    a large x0 make it already walking at t = 0 (so a constant-velocity prediction has a velocity to use).  p_stand: the scene's, or moved by `stand_offset`."""
    from ..teleop import scene as SC
    base = _h5_with_stand(stand_offset) if stand_offset is not None else SC._h5
    v = 2.7 / 8.0 if speed is None else float(speed)

    def make():
        sc = base()
        for i, p in enumerate(sc.people):
            if p.name == "p_walker":
                t_end = float(delay_s) + (x0 - WALKER_X_END) / v
                if delay_s > 0:
                    rows = [[0.0, x0, WALKER_Y], [float(delay_s), x0, WALKER_Y], [t_end, WALKER_X_END, WALKER_Y], [1e4, WALKER_X_END, WALKER_Y]]
                else:     # already walking at t = 0 (PersonTrack.at reports zero velocity AT the first row): start the track earlier on the same line
                    rows = [[-100.0, x0 + 100.0 * v, WALKER_Y], [t_end, WALKER_X_END, WALKER_Y], [1e4, WALKER_X_END, WALKER_Y]]
                pts = np.array(rows)
                sc.people[i] = SC.PersonTrack(p.name, pts, p.dynamic)
        return sc
    return make


def _ladder(sim, filt, ex, filter_kind, people, seed, goal, max_s, stall_s, plan_time, verbose, stand_offset, wait_s, max_waits, max_recoveries, t_wall, cert="rest",
            people_pred="frozen", wait_mode="fixed", replan_each_move=False, people_clock="legacy", ev_scale=1.0,
            plan_monitor="off", plan_latency=2.0, max_plan_flags=6, hop="off", hop_cfg=None):
    """recovery ladder.  No task plan certifies -> WAIT (hold at rest, replan) up to max_waits in a row -> RETREAT (certified move back to the last rest pose
    on the executed path, replan).  A non-ok action -> rebuild + replan (kind 'replan').  Every event is logged with its outcome."""
    plans, results, events = [], [], []
    held = None
    rest_poses = [[float(x) for x in sim.pose()]]          # the start pose (base at rest at t=0) qualifies
    n_wait_row, completed, aborted = 0, False, ""
    pending = None                                          # event awaiting the outcome of the next task action
    reason_next = "initial plan"
    retreat_failed_at = None                                # pose at which a retreat was uncertifiable (avoid re-trying it from the same place)
    stall_poses = []                                        # end poses of consecutive failed actions (replan livelock detector)
    timed_waits, move_replans = [], []                      # timed-wait log (tau chosen, probes, replan outcome) ; replans forced before a move action
    assert people_pred == "frozen" or cert == "segments", "--people-pred time needs --cert segments"
    assert wait_mode == "fixed" or people_pred == "time", "--wait timed needs --people-pred time"
    pm_on = plan_monitor == "on"
    if pm_on:
        filt.set_plan_latency(2.0 if plan_latency == "measured" else float(plan_latency))
    plan_flags, pre, flag_open, pm_disabled, last_L = [], None, None, False, None       # plan-prediction monitor bookkeeping (opt-in)
    if pm_on:
        assert cert == "segments" and people_pred == "time" and getattr(filt, "pm", None) is not None, \
            "--plan-monitor on needs --cert segments --people-pred time --filter gatekeeper (and W.RUNTIME_GK['plan_monitor']='on' before the filter is built)"

    hop_on = hop != "off"
    hc = dict(durations=HOP.HOP_DURATIONS, min_m=HOP.HOP_MIN_M, dirs=HOP.HOP_DIRS, max_hops=20, compass=False, clear_stop=True)
    hc.update(hop_cfg or {})
    last_route = None                                       # hop "prog" direction: the route of the first move of the most recent solved plan (segments.Path), opt-in bookkeeping
    hop_chains = []                                         # one summary per hop chain (opt-in)
    if hop_on:
        assert hop in ("before_wait", "after_wait") and cert == "segments" and hasattr(filt, "hop_commit"), \
            "--hop needs --cert segments --filter gatekeeper (the hop is committed through the SegmentGatekeeper)"

    def event(kind, reason, **kw):
        nonlocal pending
        ev = dict(kind=kind, t=sim.t, pose=[float(x) for x in sim.pose()], reason=reason, next_action=None, next_ok=None, **kw)
        events.append(ev)
        pending = ev
        if verbose:
            print(f"  [recovery] {kind} t={sim.t:.1f} pose=({ev['pose'][0]:.2f},{ev['pose'][1]:.2f}) {reason}", flush=True)
        return ev

    def hop_chain(where):
        """opt-in HOP CHAIN (tamp/hop.py): repeat { first admissible hop direction -> execute it through the SegmentGatekeeper (re-certified from the ACTUAL state, committed, followed in
        coast mode to rest) -> re-observe at rest -> replan; if the plan solves, hand it to the main loop } up to hc["max_hops"] hops or until no hop is admissible.  The whole chain
        is ONE recovery event of kind "hop" (created at the first executed hop) and never counts as a failed action.  -> pre dict (res, info, t_obs) if a plan solved, else None."""
        nonlocal pending
        ev, hops, sol, n_solve, wall_solve = None, [], None, 0, 0.0
        stop = "max_hops"
        while len(hops) < hc["max_hops"]:
            if sim.t >= max_s - 1.0:
                stop = "sim budget"
                break
            if float(np.abs(sim.base.v).max()) > REST_V:
                stop = "not at rest"
                break
            cur = sim.pose()
            tgt = next((p for p in reversed(rest_poses) if math.hypot(p[0] - cur[0], p[1] - cur[1]) >= RETREAT_MIN_M), None)
            if tgt is None:
                tgt = next((p for p in reversed(rest_poses) if math.hypot(p[0] - cur[0], p[1] - cur[1]) >= 0.05), None)
            ch = filt.hop_plan(sim, hc["dirs"], hc["durations"], hc["min_m"], route_path=last_route, back_target=tgt, compass=hc["compass"])
            if ch is None:
                stop = "no admissible hop"
                break
            if ch["kind"] == "clear" and hc["clear_stop"] and HOP.out_of_tubes(ch["bel"], cur):
                stop = "out of every tracked tube"
                break
            if ev is None:
                ev = event("hop", f"no certified task plan ({where})", where=where)
            t_b = sim.t
            rec = filt.hop_commit(sim, ch["d"], ch["t_h"], kind=ch["kind"], min_m=hc["min_m"])
            if not rec["ok"]:
                stop = "re-check refused: " + rec["why"]
                break
            n_ticks, dmin = int(round(rec["t_h"] / TICK)), math.inf
            for k_ in range(n_ticks + int(20.0 / TICK)):
                if k_ >= n_ticks and float(np.abs(sim.base.v).max()) <= REST_V:
                    break
                cmd = Cmd()
                safe, info = filt.filter(sim, cmd)
                ex._log(sim.tick(safe), info, cmd, safe, _WaitOp())
                for xy, _v in (sim.people_state or []):
                    dmin = min(dmin, float(GM.rect_dist_point(np.asarray(xy, float), sim.pose(), getattr(filt, "fp_now", GM.FP))[0]))
            filt.coast(False)
            pa = sim.pose()
            rec.update(t_start=float(t_b), t_end=float(sim.t), pose_after=[float(x) for x in pa], displacement=round(float(math.hypot(pa[0] - rec["pose_before"][0], pa[1] - rec["pose_before"][1])), 4),
                       min_person_dist=None if dmin == math.inf else round(dmin, 3), tried=ch["tried"])
            hops.append(rec)
            if verbose:
                print(f"    HOP {rec['kind']:5s} t_h={rec['t_h']:.2f} a_r={rec['a_r']:.2f} disp={rec['displacement']:.3f} m min_person={rec['min_person_dist']} margin={rec['margin']:.3f}", flush=True)
            wi, t_obs = {}, sim.t                           # re-observe at rest -> replan (people sets reset to the new observation)
            tw = time.perf_counter()
            res2 = _solve(sim, goal, held, plan_time, legacy=False, cert=cert, people_pred=people_pred, ev_scale=ev_scale, **(dict(info=wi) if pm_on else {}))
            wall_solve += time.perf_counter() - tw
            n_solve += 1
            cur = sim.pose()
            lock = len(stall_poses) >= 2 and all(math.hypot(p[0] - cur[0], p[1] - cur[1]) < 0.15 for p in stall_poses[-2:])
            if res2["solved"] and not lock:
                sol = dict(res=res2, info=wi, t_obs=t_obs)
                stop = "plan solved"
                break
        if ev is not None:
            ev.update(n_hops=len(hops), directions=[h_["kind"] for h_ in hops], durations=[h_["t_h"] for h_ in hops], displacements=[h_["displacement"] for h_ in hops],
                      margins=[round(h_["margin"], 4) for h_ in hops], min_person_dist=[h_["min_person_dist"] for h_ in hops], stop=stop, solved=sol is not None,
                      n_replan_calls=n_solve, replan_wall_s=round(wall_solve, 2))
            hop_chains.append(dict(t=ev["t"], where=where, n_hops=len(hops), stop=stop, solved=sol is not None))
            pending = ev
            if verbose:
                print(f"  [hop-chain] {where}: {len(hops)} hops, stop={stop!r}, solved={sol is not None}", flush=True)
        else:                                               # zero-hop attempt: no recovery event, but keep the reason (logging only)
            hop_chains.append(dict(t=float(sim.t), where=where, n_hops=0, stop=stop, solved=False))
            if verbose:
                print(f"  [hop-chain] {where}: 0 hops, stop={stop!r}", flush=True)
        return sol

    def handle_plan_flag(r):
        """the executor returned "plan_falsified" (committed trajectory still being followed in coast mode).  Judgement latency L: plan ONCE now (people observed now, robot start =
        the state the committed trajectory predicts at now + L, people clock offset by L), advance L in coast mode, then the caller executes the precomputed plan."""
        nonlocal pm_disabled, flag_open, last_L
        flag_live = filt.plan_flag_log[-1]
        flag = dict(flag_live)
        cont = filt.plan_continue                                  # "continue" decision: the old route is still being driven (route alive in the gatekeeper)
        t_ret = sim.t

        def solve_at(L):
            st = filt.predict_old_route(sim, L) if cont else filt.predict_state(sim, L)
            moving = float(np.abs(st["v"]).max()) > 1e-3
            wi = {}
            tw = time.perf_counter()
            res_ = _solve(sim, goal, held, plan_time, legacy=False, cert=cert, people_pred="time", t_offset=L, ev_scale=ev_scale,
                          robot_pose=st["pose"], start_v=(st["v"] if moving else None), info=wi)
            return res_, st, time.perf_counter() - tw, wi

        measured = plan_latency == "measured"
        L = (last_L if last_L is not None else 2.0) if measured else float(plan_latency)
        res_n, st, wall, wi = solve_at(L)
        wall_first = wall
        if measured:
            lm = math.ceil(wall / TICK - 1e-9) * TICK
            if abs(lm - L) > 1e-9:
                L = lm
                res_n, st, wall, wi = solve_at(L)
            last_L = L
            filt.set_plan_latency(L)
        rec = dict(kind="plan_falsified", t=flag["t"], t_return=t_ret, monitor=flag, L=L, L_mode="measured" if measured else "fixed", wall_s=round(wall, 3),
                   wall_first_s=round(wall_first, 3), latency_violated=bool(wall > L), start_kind=st["kind"], start_pose=list(st["pose"]),
                   start_v=[float(x) for x in st["v"]], plan_existed=bool(res_n["solved"]), first_segment_certified=None if res_n["solved"] else False,
                   first_action=None, route_status=r.status, start_err_m=None)
        n_budget = max(0, int((max_s - sim.t) / TICK))
        if cont:
            filt.set_t_ready(filt.n_calls * TICK + L)
        _coast_ticks(sim, filt, ex, min(int(round(L / TICK)), n_budget))              # brake decision: coast mode (committed trajectory only); continue: normal old-route boundaries
        if filt.sr is not None:
            rec["old_route"] = filt.end_route("plan_falsified_handover")                # continue / switched_to_brake: hand over at t_ready
        if not res_n["solved"]:
            _coast_ticks(sim, filt, ex, 0, until_rest=True)
        for k_ in ("decision", "decision_initial", "t_conf", "t_ready", "t_stop", "margin", "t_switch", "t_conf_switch", "t_stop_switch"):
            if k_ in flag_live:
                rec[k_] = flag_live[k_]
        pa = sim.pose()
        rec["start_err_m"] = round(float(math.hypot(pa[0] - st["pose"][0], pa[1] - st["pose"][1])), 4)
        rec["t_ready"] = sim.t
        filt.coast(False)
        plan_flags.append(rec)
        flag_open = rec if res_n["solved"] else None
        if len(plan_flags) >= max_plan_flags:
            pm_disabled = True
        if verbose:
            print(f"  [plan-flag] t={flag['t']:.2f} person={flag['person']} kind={flag['kind']} pos_err={flag['pos_err']:.2f} vel_err={flag['vel_err']:.2f} L={L:.2f}s wall={wall:.2f}s "
                  f"violated={rec['latency_violated']} start={st['kind']} plan={res_n['solved']}" + (" -> monitor disabled (cap)" if pm_disabled else ""), flush=True)
        return dict(res=res_n, info=wi, t_obs=t_ret)

    while True:
        if sim.t >= max_s:
            aborted = "sim budget exhausted"
            break
        if pre is not None:
            res, wi, t_obs = pre["res"], pre["info"], pre["t_obs"]
            pre = None
        else:
            wi, t_obs = {}, sim.t
            res = _solve(sim, goal, held, plan_time, legacy=False, cert=cert, people_pred=people_pred, ev_scale=ev_scale, **(dict(info=wi) if pm_on else {}))
        srcs = _plan_sources(res["plan"])
        plans.append(dict(sim_t=sim.t, held=None if held is None else repr(held[1]), solved=res["solved"], solve_s=res["time_s"], cost=res["cost"],
                          plan=plan_to_json(res["plan"]), route_sources=srcs, reject_counts=_reject_counts(res["stream_log"]), checker_stats=res["checker_stats"]))
        if hop_on and res["solved"]:
            rt = next((a_ for n_, args_ in res["plan"] if n_.startswith("move") for a_ in args_ if isinstance(a_, Route)), None)
            if rt is not None:
                last_route = SEG.Path(rt.pts)
        if pm_on and res["solved"] and not pm_disabled:
            filt.set_plan_prediction(_world_people(wi["world"]), t_obs)         # arm the monitor with the people prediction this plan was certified under
        if pending is not None:
            pending["plan_route_sources"] = srcs
            pending["replan_solved"] = bool(res["solved"])
        if timed_waits and timed_waits[-1]["replan_solved"] is None and timed_waits[-1].get("event") is not None:
            timed_waits[-1]["replan_solved"] = bool(res["solved"])
        if verbose:
            print(f"  plan #{len(plans)-1} ({reason_next}): solved={res['solved']} {[a for a, _ in (res['plan'] or [])]} routes={srcs} solve={res['time_s']}s", flush=True)
        # replan livelock: the last 2 failed actions ended within 0.15 m of each other -> the "certified" plan does not move us; treat as no plan
        livelock = len(stall_poses) >= 2 and all(math.hypot(p[0] - stall_poses[-1][0], p[1] - stall_poses[-1][1]) < 0.15 for p in stall_poses[-2:])
        if not res["solved"] or livelock:
            if res["solved"]:
                plans[-1]["discarded_livelock"] = True
                if verbose:
                    print("  plan discarded: replan livelock (2 failed actions ended at the same place)", flush=True)
            if len(events) >= max_recoveries:
                aborted = "max recoveries reached"
                break
            if hop == "before_wait" and not res["solved"] and sim.t < max_s:
                pre = hop_chain("before_wait")
                if pre is not None:
                    reason_next = "after hop chain"
                    continue
            if wait_mode == "timed" and not res["solved"] and n_wait_row < max_waits and sim.t < max_s:
                # TIMED WAIT: probe start offsets tau (robot holding at rest, people predicted tau + t_start ahead); the planner is re-run with a world time
                # offset (same world, shared segment cache -> only the immediate route is re-certified per tau).  If some tau works, hold tau and REPLAN fresh.
                cache = {}
                tp0 = time.time()
                tau, probes = PP.probe_wait(lambda tau: _solve(sim, goal, held, plan_time, legacy=False, cert=cert, people_pred="time", t_offset=tau, seg_cache=cache, ev_scale=ev_scale)["solved"])
                rec = dict(t=sim.t, tau=tau, probes=probes, probe_wall_s=round(time.time() - tp0, 1), replan_solved=None)
                timed_waits.append(rec)
                if verbose:
                    print(f"  [timed-wait] t={sim.t:.1f} probes={probes} -> tau={tau}", flush=True)
                if tau is not None:
                    n_wait_row += 1
                    event("wait", "no certified task plan (timed)", wait_s=tau, timed=True, tau=tau)
                    rec["event"] = len(events) - 1
                    _hold_at_rest(sim, filt, ex, min(tau, max(0.0, max_s - sim.t)))
                    reason_next = f"after timed wait {tau}s"
                    continue
            if n_wait_row < max_waits or (retreat_failed_at is not None):
                n_wait_row += 1
                event("wait", "replan livelock" if res["solved"] else "no certified task plan", wait_s=wait_s)
                _hold_at_rest(sim, filt, ex, min(wait_s, max(0.0, max_s - sim.t)))
                reason_next = "after wait"
                continue
            if hop == "after_wait" and not res["solved"] and sim.t < max_s:
                pre = hop_chain("after_wait")
                if pre is not None:
                    reason_next = "after hop chain"
                    continue
            # RETREAT: certified move back to the last rest pose (>= RETREAT_MIN_M away) on the executed path
            cur = sim.pose()
            tgt = next((p for p in reversed(rest_poses) if math.hypot(p[0] - cur[0], p[1] - cur[1]) >= RETREAT_MIN_M), None)
            world = W.world_from_sim(sim, held=held)
            st = Streams(world, cert_mode=cert)
            ev_reason = ("replan livelock" if res["solved"] else "no certified task plan") + " after waits"
            route, why = None, "no rest pose on the executed path far enough from here"
            if tgt is not None:
                q1 = Conf("q_here", float(cur[0]), float(cur[1]), float(cur[2]))
                q2 = Conf("q_rest", tgt[0], tgt[1], tgt[2])
                holding = held is not None
                item = held[0] if holding else None
                why = "no route candidate connects the current pose to the rest pose"
                for pts, rev, src in _retreat_candidates(st, q1, q2):
                    route, w_ = st.build_route(q1, q2, pts, rev, holding, item, src)
                    if route is None:
                        why = f"{src}: {w_}"
                        continue
                    break
            if route is None:
                event("retreat_uncertifiable", f"{ev_reason}: {why}")
                retreat_failed_at = [float(x) for x in cur]
                continue                                    # keep WAITing until the budget ends it
            event("retreat", ev_reason, target=list(tgt), route_source=route.source)
            args = (q1, q2, route) if held is None else (held[0], held[1], q1, q2, route)
            name = "move_free" if held is None else "move_holding"
            r = ex.run_action(name, args, timeout_s=min(max_s - sim.t, EX.DEFAULT_TIMEOUT.get(name, 60.0)), stall_s=stall_s)
            results.append(dict(plan_index=-1, recovery="retreat", **r.to_json()))
            events[-1]["retreat_status"] = r.status
            if verbose:
                print(f"    RETREAT {name:13s} {r.status:8s} t={r.t_start:6.1f}->{r.t_end:6.1f}  {r.reason}", flush=True)
            if r.ok:
                rest_poses.append([float(x) for x in sim.pose()])
            n_wait_row, retreat_failed_at = 0, None
            stall_poses = []
            reason_next = "after retreat"
            continue
        n_wait_row = 0
        retreat_failed_at = None
        failed = None
        falsified = None
        for k_act, (name, args) in enumerate(res["plan"]):
            budget = max_s - sim.t
            if budget <= 0:
                aborted = "sim budget exhausted"
                break
            if replan_each_move and people_pred == "time" and k_act > 0 and name.startswith("move"):
                # the plan's later moves were certified with people-free H0; re-solve from the CURRENT state so this move is the immediate (people-predicted) route
                move_replans.append(dict(t=sim.t, before=name, plan_index=len(plans) - 1))
                reason_next = f"replan before {name}"
                break
            if np.linalg.norm(sim.observe()["v"]) <= REST_V:
                rest_poses.append([float(x) for x in sim.pose()])          # action start pose at rest
            r = ex.run_action(name, args, timeout_s=min(budget, EX.DEFAULT_TIMEOUT.get(name, 60.0)), stall_s=stall_s)
            results.append(dict(plan_index=len(plans) - 1, **r.to_json()))
            if flag_open is not None and name.startswith("move"):
                sg_ = (r.info or {}).get("segments") or {}
                flag_open["first_action"], flag_open["first_action_status"] = name, r.status
                flag_open["first_segment_certified"] = bool(sg_.get("n_segments_admitted", 0) >= 1)
                flag_open = None
            if pending is not None and pending["next_action"] is None:
                pending["next_action"], pending["next_ok"] = name, bool(r.ok)
            if verbose:
                print(f"    {name:13s} {r.status:8s} t={r.t_start:6.1f}->{r.t_end:6.1f}  {r.reason}", flush=True)
            if name == "pick":
                held = (args[0], args[2])
            if name == "place" and r.ok:
                held = None
            if held is not None and not sim.observe()["grasped"]:
                held = None
            if r.ok and np.linalg.norm(sim.observe()["v"]) <= REST_V:
                rest_poses.append([float(x) for x in sim.pose()])
            if r.status == "plan_falsified":                # not a failed action (no stall pose, no recovery count): judge, replan from the predicted state
                falsified = r
                break
            if not r.ok:
                failed = r
                break
        if aborted:
            break
        if falsified is not None:
            pre = handle_plan_flag(falsified)
            reason_next = f"plan falsified ({plan_flags[-1]['monitor']['kind']})"
            continue
        if failed is None and move_replans and move_replans[-1]["plan_index"] == len(plans) - 1 and reason_next.startswith("replan before"):
            continue                                        # not a failure: the loop re-solves from the current state
        if failed is None:
            completed = True
            break
        if len(events) >= max_recoveries:
            aborted = "max recoveries reached"
            break
        reason_next = f"{failed.name} {failed.status}: {failed.reason}"
        stall_poses.append([float(x) for x in sim.pose()])
        event("replan", reason_next, failed_action=failed.name)
    kinds = Counter(e["kind"] for e in events)
    summ = dict(filter=filter_kind, people=people, seed=seed, stand_offset=stand_offset, goal=goal, recovery="ladder", max_s=max_s, stall_s=stall_s,
                wait_s=wait_s, max_waits=max_waits, people_pred=people_pred, wait_mode=wait_mode, replan_each_move=replan_each_move, people_clock=people_clock, people_ev_scale=ev_scale,
                timed_waits=timed_waits, move_replans=move_replans, n_move_replans=len(move_replans), n_timed_waits=sum(1 for w_ in timed_waits if w_["tau"] is not None), max_recoveries=max_recoveries,
                completed=completed, aborted=aborted, n_replans=kinds.get("replan", 0), n_recoveries=len(events), recoveries_by_kind=dict(kinds),
                sim_s=sim.t, wall_s=time.time() - t_wall, recoveries=events, rest_poses=rest_poses,
                final_route_sources=plans[-1]["route_sources"] if plans and completed else [],
                plans=plans, results=results, episode=ex.episode_metrics(),
                filter_counters=EX._jsonable(filt.counters()) if hasattr(filt, "counters") else {}, cert=cert, segments=_seg_summary(results, plans, cert),
                plan_monitor=plan_monitor, plan_latency=plan_latency, max_plan_flags=max_plan_flags, plan_flags=plan_flags, n_plan_flags=len(plan_flags),
                plan_monitor_disabled=pm_disabled)
    if hop_on:
        summ.update(hop=hop, hop_cfg={k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in hc.items()}, hop_chains=hop_chains, n_hops=sum(c_["n_hops"] for c_ in hop_chains))
    return summ


def run_episode(filter_kind="none", people=False, seed=0, goal="(On tray desk)", max_replans=3, max_s=200.0, stall_s=10.0, plan_time=60.0, verbose=True,
                stand_offset=None, recovery="none", wait_s=3.0, max_waits=2, max_recoveries=8, cert="rest",
                people_pred="frozen", wait_mode="fixed", replan_each_move=False, people_clock="legacy", walker=None, ev_scale=1.0,
                plan_monitor="off", plan_latency=2.0, max_plan_flags=6, hop="off", hop_cfg=None):
    """walker: None (scene's) or dict(delay_s, speed, x0) -> _h5_with_walker (this process only)."""
    t_wall = time.time()
    if walker is not None:
        from ..teleop import scene as SC
        SC.SCENES["h5"] = _h5_with_walker(stand_offset=stand_offset, **walker)
    elif stand_offset is not None:         # this process only: every get_scene("h5") (sim, rig, world) sees the moved person
        from ..teleop import scene as SC
        SC.SCENES["h5"] = _h5_with_stand(stand_offset)
    sim = TeleopSim("h5", grasped=False, seed=seed, with_people=people, counter=True, desk=True, people_clock=people_clock)
    if cert == "segments":
        assert filter_kind == "gatekeeper", "--cert segments needs --filter gatekeeper (the segments are the gatekeeper's Traj format)"
        from .seg_gate import make_seg_filter
        filt = make_seg_filter()
    else:
        filt = make_filter(filter_kind, sim)
    ex = EX.Executor(sim, filt, seed=seed)
    if recovery == "ladder":
        return _ladder(sim, filt, ex, filter_kind, people, seed, goal, max_s, stall_s, plan_time, verbose, stand_offset, wait_s, max_waits, max_recoveries, t_wall, cert=cert,
                       people_pred=people_pred, wait_mode=wait_mode, replan_each_move=replan_each_move, people_clock=people_clock, ev_scale=ev_scale,
                       plan_monitor=plan_monitor, plan_latency=plan_latency, max_plan_flags=max_plan_flags, hop=hop, hop_cfg=hop_cfg)
    plans, results, replans = [], [], []
    held = None                                   # (item, Grasp) of the last executed pick that the sim still holds
    reason_next = "initial plan"
    n_replans, completed, aborted = 0, False, ""
    while True:
        res = _solve(sim, goal, held, plan_time, cert=cert)
        plans.append(dict(sim_t=sim.t, held=None if held is None else repr(held[1]), solved=res["solved"], solve_s=res["time_s"], cost=res["cost"],
                          plan=plan_to_json(res["plan"]), reject_counts=_reject_counts(res["stream_log"]), checker_stats=res["checker_stats"]))
        replans.append(dict(reason=reason_next, solve_s=res["time_s"], reject_counts=plans[-1]["reject_counts"], solved=res["solved"]))
        if verbose:
            print(f"  plan #{len(plans)-1} ({reason_next}): solved={res['solved']} {[a for a, _ in (res['plan'] or [])]} solve={res['time_s']}s", flush=True)
        if not res["solved"]:
            aborted = "no plan from the current state"
            break
        failed = None
        for name, args in res["plan"]:
            budget = max_s - sim.t
            if budget <= 0:
                aborted = "sim budget exhausted"
                break
            r = ex.run_action(name, args, timeout_s=min(budget, EX.DEFAULT_TIMEOUT.get(name, 60.0)), stall_s=stall_s)
            results.append(dict(plan_index=len(plans) - 1, **r.to_json()))
            if verbose:
                print(f"    {name:13s} {r.status:8s} t={r.t_start:6.1f}->{r.t_end:6.1f}  {r.reason}", flush=True)
            if name == "pick":
                held = (args[0], args[2])         # tentative; confirmed below from the sim
            if name == "place" and r.ok:
                held = None
            if held is not None and not sim.observe()["grasped"]:
                held = None
            if not r.ok:
                failed = r
                break
        if aborted:
            break
        if failed is None:
            completed = True
            break
        if n_replans >= max_replans:
            aborted = "max replans reached"
            break
        if sim.t >= max_s:
            aborted = "sim budget exhausted"
            break
        n_replans += 1
        reason_next = f"{failed.name} {failed.status}: {failed.reason}"
    # "completed" = the planner's goal plan ran to the end (all actions ok); the physical goal is not re-verified here beyond the executor's post-conditions
    summ = dict(filter=filter_kind, people=people, seed=seed, stand_offset=stand_offset, goal=goal, max_replans=max_replans, max_s=max_s, stall_s=stall_s,
                completed=completed, aborted=aborted, n_replans=n_replans, sim_s=sim.t, wall_s=time.time() - t_wall,
                plans=plans, replans=replans, results=results, episode=ex.episode_metrics(),
                filter_counters=EX._jsonable(filt.counters()) if hasattr(filt, "counters") else {}, cert=cert, segments=_seg_summary(results, plans, cert))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", choices=["shield", "gatekeeper", "none"], default="none")
    ap.add_argument("--people", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goal", default="(On tray desk)")
    ap.add_argument("--max-replans", type=int, default=3)
    ap.add_argument("--recovery", choices=["none", "ladder"], default="ladder")
    ap.add_argument("--cert", choices=["rest", "segments"], default="rest", help="route certificate: rest = legacy from-rest samples; segments = timed gatekeeper-format segments (needs --filter gatekeeper)")
    ap.add_argument("--wait-s", type=float, default=3.0)
    ap.add_argument("--max-waits", type=int, default=2)
    ap.add_argument("--max-recoveries", type=int, default=8)
    ap.add_argument("--max-s", type=float, default=200.0, help="total sim-time budget [s]")
    ap.add_argument("--stall-s", type=float, default=10.0)
    ap.add_argument("--plan-time", type=float, default=60.0)
    ap.add_argument("--stand-offset", type=float, default=None, help="move p_stand to this distance [m] from the corridor centreline (default: scene's 1.03)")
    ap.add_argument("--cup-row", choices=["attached", "always"], default="attached",
                    help="gatekeeper cup row: attached = only while the tray is on the robot (10-22 fix); always = pre-10-22 behaviour (cup on the counter counted)")
    ap.add_argument("--c2-ready", choices=["off", "rank"], default="off", help="segment mode: rank admitted candidates by C2-ready (a certified next move from the rest pose under walk/stop people); never admits/rejects")
    ap.add_argument("--c2-ready-scope", choices=["both", "gate"], default="both", help="where --c2-ready rank runs: both = planner route streams AND executor boundaries (today) | gate = executor segment boundaries only (planner solves stay fast)")
    ap.add_argument("--c2-ready-tube-h", type=float, default=0.0, help="segment mode, executor boundaries: tracked-tube ranking term horizon [s]; candidates whose REST pose stays off every tracked tube for [t_rest, t_rest + h] are preferred (ranking only, never admits/rejects); 0 = off")
    ap.add_argument("--c2-ready-tube-clear", type=float, default=0.0, help="tube term: extra clearance [m] beyond d_person the rest pose must keep to the tube (0 = not touching; hop-usable ~0.5, see tamp/hop_e3.py)")
    ap.add_argument("--seg-offsets", default="", help="segment mode: comma-separated lateral route offsets [m, left +] tried at segment boundaries (e.g. 0.15,-0.15,0.3,-0.3); empty = off")
    ap.add_argument("--people-clock", choices=["legacy", "time"], default="legacy", help="TeleopSim people clock (legacy: the h5 walker never moves; time: it walks its track)")
    ap.add_argument("--people-pred", choices=["frozen", "time"], default="frozen", help="segment planner people model: frozen (plan-time position, s<=near_m) | time (immediate route vs people at c + v_hat t_start)")
    ap.add_argument("--wait", choices=["fixed", "timed"], default="fixed", help="no-plan wait: fixed = --wait-s; timed = probe tau in (0.5..8 s) with people predicted ahead (needs --people-pred time)")
    ap.add_argument("--replan-each-move", action=argparse.BooleanOptionalAction, default=False, help="re-solve from the current state before each later move action (needs --people-pred time)")
    ap.add_argument("--people-ev-scale", type=float, default=1.0, help="scale of the velocity-error growth e_v*t added to a predicted person's radius (1 = spec; the sim truth velocity has no error, 0.1 keeps a 2 m-away standing person certifiable over a 13 s route)")
    ap.add_argument("--walker-delay", type=float, default=None, help="custom p_walker: start delay [s] (scene's = 2)")
    ap.add_argument("--walker-speed", type=float, default=None, help="custom p_walker: speed [m/s] (scene's = 0.3375)")
    ap.add_argument("--walker-x0", type=float, default=None, help="custom p_walker: start x [m] on y = 3.8 (scene's = 3.6)")
    ap.add_argument("--plan-monitor", choices=["off", "on"], default="off", help="segment mode: flag a route whose people prediction is falsified (tamp/plan_monitor.py) and replan from the state the committed trajectory predicts after the judgement latency; needs --people-pred time")
    ap.add_argument("--plan-latency", default="2.0", help="judgement latency L [s] (float) or 'measured' (= the wall-clock solve time of that planner call, rounded up to the 0.02 s tick)")
    ap.add_argument("--plan-eps-p", type=float, default=0.30, help="monitor: position deviation [m] from the plan's prediction")
    ap.add_argument("--plan-eps-v", type=float, default=0.20, help="monitor: velocity deviation [m/s] from the plan's prediction")
    ap.add_argument("--plan-r-rel", type=float, default=2.0, help="monitor: a violation counts only within this distance [m] of the remaining planned route")
    ap.add_argument("--plan-confirm", type=int, default=2, help="monitor: consecutive violating gatekeeper steps (0.1 s) before the flag")
    ap.add_argument("--plan-decision", choices=["timing", "brake"], default="timing", help="after a monitor flag: timing = keep driving the old route while t_ready + t_stop + margin < t_conf (else brake); brake = always coast on the committed trajectory")
    ap.add_argument("--plan-conf-margin", type=float, default=0.5, help="t_margin [s] of the timing test")
    ap.add_argument("--max-plan-flags", type=int, default=6, help="monitor flags per episode before the monitor is disabled (not counted in --max-recoveries)")
    ap.add_argument("--hop", choices=["off", "before_wait", "after_wait"], default="off", help="opt-in hop chain (tamp/hop.py, T4(c)): when a replan fails, run rest-to-rest hops (re-observe + replan after each) before the timed wait / before the retreat; needs --recovery ladder --cert segments --filter gatekeeper")
    ap.add_argument("--hop-durations", default=",".join(str(x) for x in HOP.HOP_DURATIONS), help="hop durations [s] tried longest first (each rounded down to a multiple of gk.h)")
    ap.add_argument("--hop-min-m", type=float, default=HOP.HOP_MIN_M, help="a hop that moves less than this [m] is not taken")
    ap.add_argument("--hop-dirs", default=",".join(HOP.HOP_DIRS), help="hop direction kinds in priority order (prog, clear, back)")
    ap.add_argument("--hop-max", type=int, default=20, help="max hops per chain")
    ap.add_argument("--hop-compass", action=argparse.BooleanOptionalAction, default=False, help="last resort: also try the 8 compass directions")
    ap.add_argument("--hop-clear-stop", action=argparse.BooleanOptionalAction, default=True, help="stop a chain when the next hop would be a 'clear' hop but the footprint is already out of every tracked tube for 4 s")
    ap.add_argument("--runtime-rows", choices=["all", "cup"], default="all", help="runtime gatekeeper rows: all = cup + hw + walls + people (default) | cup = ONLY cup + hw rows, people part of the falsification monitor off, never hands off to the Shield (a pose falsification commits the cup-safe brake and refuses the route; the planner owns walls and people)")
    ap.add_argument("--plan-cert", choices=["off", "pred"], default="off", help="plan-time route certifier (needs --cert segments): off = unchanged | pred = tamp/plan_cert.py: cup + hw + walls + ALL people over the WHOLE route, constant-velocity prediction disc only (+ separating-motion exit rule)")
    ap.add_argument("--pred-ev", type=float, default=_GK0.e_v, help="person velocity-estimate error e_v [m/s] of the prediction disc r0 + e_v t + a_p t^2/2 (default: GKConfig.e_v)")
    ap.add_argument("--pred-ap", type=float, default=_GK0.a_p, help="person acceleration bound a_p [m/s^2] of the prediction disc (default: GKConfig.a_p)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    W.RUNTIME_GK["cup_row"] = a.cup_row                    # read by world_from_sim, make_filter and make_seg_filter
    W.RUNTIME_GK["c2_ready"] = a.c2_ready
    if a.c2_ready_scope != "both":                         # only non-default values are written: RUNTIME_GK (and so every default run) is untouched
        W.RUNTIME_GK["c2_ready_scope"] = a.c2_ready_scope
    if a.c2_ready_tube_h > 0.0:
        W.RUNTIME_GK["c2_ready_tube_h"], W.RUNTIME_GK["c2_ready_tube_clear"] = a.c2_ready_tube_h, a.c2_ready_tube_clear
    W.RUNTIME_GK["seg_offsets"] = tuple(float(x) for x in a.seg_offsets.split(",") if x.strip())
    W.RUNTIME_GK["plan_monitor"] = a.plan_monitor
    W.RUNTIME_GK["plan_eps_p"], W.RUNTIME_GK["plan_eps_v"], W.RUNTIME_GK["plan_r_rel"], W.RUNTIME_GK["plan_confirm"] = a.plan_eps_p, a.plan_eps_v, a.plan_r_rel, a.plan_confirm
    W.RUNTIME_GK["plan_decision"], W.RUNTIME_GK["plan_conf_margin"] = a.plan_decision, a.plan_conf_margin
    if a.runtime_rows != "all":                            # opt-in planner-owns-safety knobs: only non-default values are written (RUNTIME_GK of a default run is untouched)
        W.RUNTIME_GK["runtime_rows"] = a.runtime_rows
    if a.plan_cert != "off":
        assert a.cert == "segments", "--plan-cert pred needs --cert segments"
        W.RUNTIME_GK["plan_cert"] = a.plan_cert
    if a.pred_ev != _GK0.e_v:
        W.RUNTIME_GK["e_v"] = a.pred_ev
    if a.pred_ap != _GK0.a_p:
        W.RUNTIME_GK["a_p"] = a.pred_ap
    plan_latency = "measured" if a.plan_latency == "measured" else float(a.plan_latency)
    name = a.name or f"{a.filter}_{'people' if a.people else 'nopeople'}_s{a.seed}"
    print(f"[run_tamp] {name}", flush=True)
    walker = None
    if a.walker_delay is not None or a.walker_speed is not None or a.walker_x0 is not None:
        walker = dict(delay_s=2.0 if a.walker_delay is None else a.walker_delay, speed=a.walker_speed, x0=3.6 if a.walker_x0 is None else a.walker_x0)
    s = run_episode(a.filter, a.people, a.seed, a.goal, a.max_replans, a.max_s, a.stall_s, a.plan_time, stand_offset=a.stand_offset,
                    recovery=a.recovery, wait_s=a.wait_s, max_waits=a.max_waits, max_recoveries=a.max_recoveries, cert=a.cert,
                    people_pred=a.people_pred, wait_mode=a.wait, replan_each_move=a.replan_each_move, people_clock=a.people_clock, walker=walker, ev_scale=a.people_ev_scale,
                    plan_monitor=a.plan_monitor, plan_latency=plan_latency, max_plan_flags=a.max_plan_flags,
                    hop=a.hop, hop_cfg=dict(durations=tuple(float(x) for x in a.hop_durations.split(",") if x.strip()), min_m=a.hop_min_m,
                                            dirs=tuple(x.strip() for x in a.hop_dirs.split(",") if x.strip()), max_hops=a.hop_max, compass=a.hop_compass, clear_stop=a.hop_clear_stop))
    s["walker"] = walker
    s["cup_row"] = a.cup_row
    s["runtime_rows"], s["plan_cert"], s["pred_ev"], s["pred_ap"] = a.runtime_rows, a.plan_cert, a.pred_ev, a.pred_ap
    s["c2_ready"] = a.c2_ready
    if a.c2_ready_scope != "both" or a.c2_ready_tube_h > 0.0:
        s["c2_ready_scope"], s["c2_ready_tube_h"], s["c2_ready_tube_clear"] = a.c2_ready_scope, a.c2_ready_tube_h, a.c2_ready_tube_clear
    s["seg_offsets"] = list(W.RUNTIME_GK["seg_offsets"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(s, indent=1, default=str))
    e = s["episode"]
    print(f"[run_tamp] {name}: completed={s['completed']} n_replans={s['n_replans']} n_recoveries={s.get('n_recoveries')} by_kind={s.get('recoveries_by_kind')} final_routes={s.get('final_route_sources')} aborted={s['aborted']!r} sim_s={s['sim_s']:.1f} wall_s={s['wall_s']:.0f} "
          f"seg={None if s.get('segments') is None else {k: s['segments'][k] for k in ('n_segments_planned', 'n_segments_admitted', 'n_refused', 'speeds_per_route')}} min_person={e.get('min_person_distance_m')} spilled={e.get('spilled')} person_contacts={e.get('person_contact_ticks')} "
          f"wall_coll={e.get('wall_collision_ticks')}", flush=True)
    print(f"[run_tamp] wrote {out / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
