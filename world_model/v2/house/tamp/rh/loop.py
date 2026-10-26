"""tamp/rh/loop.py -- the SHARED receding-horizon episode loop (lattice, MPPI, ...) in the h5 doorway scenario.  SIMULATION ONLY (MuJoCo, offline; no robot, no network).

    python -m world_model.v2.house.tamp.rh.loop --planner lattice --seed 1 [--horizon 1.5] [--out DIR]

Task ("task": "base_drive_to_counter"): BASE DRIVE ONLY, no pick / place.  The robot (carrying the tray + cup, TeleopSim grasped=True) starts at lab_start and has to reach the
cafe_counter_pick xy within 5 cm AT REST.  Scenario = run_tamp's: _h5_with_walker(delay, speed, x0, stand_offset) + people_clock="time" (the walker yields to the robot, as in the
sweeps).  Walker / stand parameters come from the sweep scenarios.json (sweep_1022c2) by seed.

Control path (NO Shield anywhere): tamp/seg_gate.SegmentGatekeeper in COAST mode with GKConfig(runtime_rows="cup") -- it follows the committed Traj, runs only the cup + hw
rows and the pose part of the falsification monitor, and on a pose falsification commits the cup-safe brake (counted, n_falsified).  The loop commits the Trajs the planner
admitted: Gatekeeper._commit(traj, T_S, 0, belief, t).

Loop (api.py contract):
  * at rest: propose(state) -> admit -> commit the first admitted MOTION candidate.  Nothing admitted (the lattice's "stop now" candidate is not motion): WAIT wait_s, propose again.
  * driving: lead_s before the committed nominal part ends, the state at the nominal end is PREDICTED (SegmentGatekeeper.predict_state: replicates filter_base tick by tick),
    the proposer runs from it with the people observed NOW (truth), and the admitted Traj is committed exactly at that boundary (nearest sim tick).  Nothing admitted: the
    committed Traj is kept (its backup brakes to rest by construction); at rest the at-rest rule above applies.
  * planning compute is not charged to simulated time (planner_ms is reported); people observations are the sim truth at decision time (people_source="truth").
  * For a replan from the PREDICTED state the people were observed lead s earlier, so the loop calls api.admit(t0 = delay) (via admit_at) -- the people are advanced by
    v_hat * delay and their error has already grown for `delay`.
Stop: goal reached, total waiting >= max_wait_s, or sim time >= max_s.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ...gatekeeper.belief import GKConfig
from ...gatekeeper.traj import backup_from
from ...teleop import scene as SC
from ...teleop.sim import Cmd, TICK, TeleopSim
from .. import run_tamp as RT
from .. import world as W
from ..seg_gate import SegmentGatekeeper
from .api import Candidate, RHConfig, State, admit, plan_gk

SCENARIOS = "/data/local/ss5772/molmospaces/tamp/run_tamp/sweep_1022c2/scenarios.json"
OUT_ROOT = "/data/local/ss5772/molmospaces/tamp/rh"
GOAL_TOL = 0.05              # m, goal xy tolerance (robot at rest)
REST_V = 1e-3                # m/s (and rad/s): "at rest"
PERSON_TORSO_R = 0.22        # teleop.sim.PERSON_TORSO_R (pdist = footprint-to-centre distance - torso radius)


def runtime_gk(**over) -> GKConfig:
    """the runtime gatekeeper config of this experiment: world.RUNTIME_GK + runtime_rows="cup" (user decision 10-26)."""
    return dataclasses.replace(GKConfig(), **dict(W.RUNTIME_GK, runtime_rows="cup", **over))


def is_motion(c: Candidate) -> bool:
    return c.traj.t_sw > 1e-9 and float(np.abs(c.traj.V).max()) > 1e-9


def admit_at(cands, state: State, world, gk, cfg: RHConfig, holding: bool = True, delay_s: float = 0.0):
    """api.admit with a delay between the people observation and the START of the trajectory (see the module docstring).  -> (Candidate or None, log).
    Thin alias kept for callers/tests: api.admit now takes t0 itself."""
    return admit(cands, state, world, gk, cfg, holding, t0=max(0.0, float(delay_s)))


def people_obs(sim):
    return [(np.asarray(xy, float)[:2].copy(), np.asarray(v, float)[:2].copy()) for xy, v in (sim.people_state or [])]


def load_scenario(seed: int, path: str = SCENARIOS) -> dict:
    for s in json.loads(Path(path).read_text()):
        if int(s["seed"]) == int(seed):
            return s
    raise KeyError(f"seed {seed} not in {path}")


def build(seed, stand_offset, walker, gk_plan: GKConfig, carry: bool = True):
    """-> (sim, world).  The planner world comes from a coffee-world sim (counter + desk, tray resting at the counter dock: that is where cup_rel = the cup in the carried-tray
    frame is read from); the episode sim carries the tray (grasped=True, no counter slab) when `carry`, else it is the coffee-world sim itself (empty hands)."""
    sim_w = TeleopSim("h5", grasped=False, seed=seed, with_people=carry is False, counter=True, desk=True, people_clock="time")
    world = W.world_from_sim(sim_w, gk=gk_plan)
    if not carry:
        return sim_w, world
    del sim_w
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, counter=False, desk=False, people_clock="time")
    return sim, world


def _ms(x):
    a = np.array(x) if len(x) else np.zeros(1)
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(a.max()), n=len(x))


def rh_episode(proposer, seed, stand_offset, walker, cfg: RHConfig = RHConfig(), gk_runtime: GKConfig | None = None, max_s: float = 200.0, people_source: str = "truth",
               out=None, carry: bool = True, verbose: bool = True, name: str | None = None, observe: str = "truth") -> dict:
    assert people_source == "truth", "only the sim-truth people observation is implemented"
    assert observe in ("truth", "shadow"), "observe: 'truth' (default) | 'shadow' (stereo pipeline in parallel, never drives; tamp/rh/shadow.py)"
    t_wall = time.time()
    gk_rt = gk_runtime or runtime_gk()
    assert gk_rt.runtime_rows == "cup", "this loop runs the 'planner owns safety' runtime (runtime_rows='cup')"
    gk_plan = dataclasses.replace(gk_rt, runtime_rows="all")
    orig = SC.SCENES["h5"]
    try:
        if walker is not None:
            SC.SCENES["h5"] = RT._h5_with_walker(stand_offset=stand_offset, **walker)
        elif stand_offset is not None:
            SC.SCENES["h5"] = RT._h5_with_stand(stand_offset)
        sim, world = build(seed, stand_offset, walker, gk_plan, carry)
        if not hasattr(proposer, "propose"):                  # a factory world -> Proposer (MPPI needs the world's occupancy grid)
            proposer = proposer(world)
        res = _run(sim, world, proposer, cfg, gk_rt, gk_plan, max_s, verbose, observe=observe, seed=seed)
    finally:
        SC.SCENES["h5"] = orig
    res.update(task="base_drive_to_counter", planner=getattr(proposer, "name", type(proposer).__name__), planner_params=getattr(proposer, "params", lambda: {})(),
               seed=seed, stand_offset=stand_offset, walker=walker, carry=carry, people_source=people_source, cfg=dataclasses.asdict(cfg),
               runtime_gk=dict(runtime_rows=gk_rt.runtime_rows, person_model=gk_rt.person_model, cup_row=gk_rt.cup_row, keep_rule=gk_rt.keep_rule), wall_s=time.time() - t_wall,
               max_s=max_s, goal_tol_m=GOAL_TOL)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        nm = name or f"{res['planner']}_s{seed}"
        (out / f"{nm}.json").write_text(json.dumps(res, indent=1, default=_json))
        if verbose:
            print(f"[rh] wrote {out / (nm + '.json')}", flush=True)
    return res


def _json(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _run(sim, world, proposer, cfg: RHConfig, gk_rt: GKConfig, gk_plan: GKConfig, max_s: float, verbose: bool, observe: str = "truth", seed: int = 0) -> dict:
    wp = sim.scene.waypoints
    goal = np.array(wp["cafe_counter_pick"][:2], float)
    route = np.array(sim.scene.out_route, float)
    filt = SegmentGatekeeper(gk=gk_rt)
    filt.docks = [wp["cafe_counter_pick"][:2], wp["lab_desk_place"][:2]]
    filt.coast(True)
    h = gk_rt.h
    st = dict(committed=None, pending=None, rest_next=0.0, waited=0.0, n_fals_seen=0, done=False, reason="", t_done=None)
    planner_ms, admit_ms, admit_log, top_fail = [], [], [], Counter()
    cnt = Counter()
    metrics = dict(min_person_centre=math.inf, min_person_surf=math.inf, contact_ticks=0, min_wall=math.inf, peak_acc=0.0, peak_tray_acc=0.0, max_cup_tilt=0.0, beads_out=0,
                   commit_pose_err=0.0, commit_v_err=0.0)

    shadow = None
    if observe == "shadow":
        from .shadow import Shadow
        shadow = Shadow(sim, world, gk_plan, cfg, seed=seed)

    def commit(tr, t, bel_t=None):
        bel = filt._bel_now(sim, t)
        tr.t0 = t
        filt._commit(tr, tr.T_S, 0.0, bel, t)
        t_b = t + max(tr.t_sw, 0.0)
        tr_rest = tr.t_rest if tr.t_rest is not None else tr.N * tr.h
        motion = tr.t_sw > 1e-9 and float(np.abs(tr.V).max()) > 1e-9
        st["committed"] = dict(tr=tr, t_commit=t, t_b=t_b, t_dec=max(t, t_b - cfg.lead_s), t_end=t + tr_rest, motion=motion, decided=not motion)

    def log_decision(kind, t, state, cands, adm, log, p_ms, a_ms, delay):
        cnt["n_replans"] += 1
        planner_ms.append(p_ms)
        admit_ms.append(a_ms)
        if log:
            top_fail[log[0][3] if not log[0][1] else "ok"] += 1
        admit_log.append([round(t, 2), kind, [round(float(x), 3) for x in state.pose], round(delay, 2), len(cands), len(log), None if adm is None else adm.source,
                          None if adm is None or not log else round(log[-1][2], 4), None if not log else log[0][3]])

    def decide(kind, t, state, delay):
        t0 = time.perf_counter()
        cands = list(proposer.propose(state, world, goal, route, cfg))
        t1 = time.perf_counter()
        adm, log = admit_at(cands, state, world, gk_plan, cfg, True, delay)
        t2 = time.perf_counter()
        log_decision(kind, t, state, cands, adm, log, (t1 - t0) * 1e3, (t2 - t1) * 1e3, delay)
        if shadow is not None:                                                          # observation only: never changes what the loop does
            shadow.on_decision(kind, t, state, delay, cands, adm, tuple(float(x) for x in sim.pose()))
        return adm

    while True:
        t = filt.n_calls * TICK
        pose, v = sim.pose(), np.asarray(sim.base.v, float)
        at_rest = float(np.abs(v).max()) <= REST_V
        cm = st["committed"]
        idle = cm is None or t >= cm["t_end"] - 1e-9
        if at_rest and idle and float(np.hypot(*(pose[:2] - goal))) <= GOAL_TOL:
            st["done"], st["t_done"] = True, sim.t
            break
        if sim.t >= max_s:
            st["reason"] = "max_s"
            break
        if st["waited"] >= cfg.max_wait_s:
            st["reason"] = "max_wait"
            break
        if filt.n_falsified != st["n_fals_seen"]:                                       # the runtime monitor fired: the cup-safe brake is the new committed Traj
            st["n_fals_seen"] = filt.n_falsified
            tr = filt.traj
            st["pending"] = None
            st["committed"] = cm = dict(tr=tr, t_commit=t, t_b=t, t_dec=t, t_end=t + (tr.t_rest if tr.t_rest is not None else tr.N * tr.h), motion=False, decided=True)
            idle = False
            cnt["n_brake_events"] += 1
        pend = st["pending"]
        if pend is not None and t >= pend["t_c"] - 1e-9:                                # commit exactly at the boundary (nearest sim tick)
            tr = pend["cand"].traj
            metrics["commit_pose_err"] = max(metrics["commit_pose_err"], float(np.hypot(*(pose[:2] - np.array(pend["pose"][:2])))))
            metrics["commit_v_err"] = max(metrics["commit_v_err"], float(np.abs(v - np.array(pend["v"])).max()))
            commit(tr, t)
            st["pending"] = None
            cnt["n_admitted"] += 1
            cm = st["committed"]
            idle = False
        if cm is not None and cm["motion"] and not cm["decided"] and st["pending"] is None and t >= cm["t_dec"] - 1e-9:
            cm["decided"] = True                                                        # one decision per committed Traj
            m = max(1, int(round((cm["t_b"] - t) / TICK)))
            pr = filt.predict_state(sim, m * TICK)
            state = State(tuple(float(x) for x in pr["pose"]), tuple(float(x) for x in pr["v"]), t + m * TICK, people_obs(sim))
            adm = decide("drive", t, state, m * TICK)
            if adm is not None and is_motion(adm):
                st["pending"] = dict(cand=adm, t_c=t + m * TICK, pose=pr["pose"], v=pr["v"])
            else:
                cnt["n_no_admit"] += 1                                                  # keep the committed Traj: it brakes to rest
        elif at_rest and idle and st["pending"] is None and t >= st["rest_next"] - 1e-9:
            state = State(tuple(float(x) for x in pose), tuple(float(x) for x in v), t, people_obs(sim))
            adm = decide("rest", t, state, 0.0)
            if adm is not None and is_motion(adm):
                commit(adm.traj, t)
                cnt["n_admitted"] += 1
            else:
                cnt["n_no_admit"] += 1
                cnt["n_waits"] += 1
                st["waited"] += cfg.wait_s
                st["rest_next"] = t + cfg.wait_s
                if st["committed"] is None:
                    commit(backup_from(np.zeros(3), filt._bel_now(sim, t), gk_rt, t0=t), t)
        cm = st["committed"]
        safe, info = filt.filter(sim, Cmd(twist=np.zeros(3)))
        o = sim.tick(safe)
        if shadow is not None:
            shadow.step(sim)
        pd = float(o["pdist"])
        metrics["min_person_surf"] = min(metrics["min_person_surf"], pd)
        metrics["min_person_centre"] = min(metrics["min_person_centre"], pd + PERSON_TORSO_R)
        metrics["contact_ticks"] += int(pd <= 0.0)
        metrics["min_wall"] = min(metrics["min_wall"], float(o["wall"]))
        metrics["peak_acc"] = max(metrics["peak_acc"], float(np.hypot(*o["acc_body"][:2])))
        metrics["peak_tray_acc"] = max(metrics["peak_tray_acc"], float(np.hypot(*o["tray_acc"][:2])))
        metrics["max_cup_tilt"] = max(metrics["max_cup_tilt"], float(o["cup_tilt"]))
        metrics["beads_out"] = int(o["beads_out"])
        if verbose and filt.n_calls % 500 == 0:
            print(f"[rh] t={sim.t:6.1f} pose=({o['pose'][0]:.2f},{o['pose'][1]:.2f},{o['pose'][2]:.2f}) v={np.round(o['v'], 2)} d_goal={np.hypot(*(o['pose'][:2] - goal)):.2f} "
                  f"person={pd:.2f} replans={cnt['n_replans']} adm={cnt['n_admitted']} noadm={cnt['n_no_admit']} waits={cnt['n_waits']}", flush=True)
    fin = sim.pose()
    n_fals = filt.n_falsified
    sh = None
    if shadow is not None:
        sh = shadow.result()
        shadow.close()
    res = dict(done=bool(st["done"]), reason=st["reason"] or ("goal" if st["done"] else ""), t_done=st["t_done"], sim_s=float(sim.t), n_replans=cnt["n_replans"], n_admitted=cnt["n_admitted"],
                n_no_admit=cnt["n_no_admit"], wait_s_total=float(st["waited"]), n_waits=cnt["n_waits"], n_brake_events=cnt["n_brake_events"], n_falsified=int(n_fals),
                falsify_reasons=dict(filt.falsify_reasons),
                min_true_person_dist_m=None if not math.isfinite(metrics["min_person_centre"]) else metrics["min_person_centre"],
                min_person_surface_dist_m=None if not math.isfinite(metrics["min_person_surf"]) else metrics["min_person_surf"],
                person_contact_ticks=metrics["contact_ticks"], min_wall_clearance_m=metrics["min_wall"], cup_lost=sim.cup_lost_t is not None, cup_lost_at_s=sim.cup_lost_t,
                spill_beads_out=metrics["beads_out"], peak_base_accel_ms2=metrics["peak_acc"], peak_tray_acc_ms2=metrics["peak_tray_acc"], max_cup_tilt_deg=metrics["max_cup_tilt"],
                final_pose=[float(x) for x in fin], final_goal_dist_m=float(np.hypot(*(fin[:2] - goal))), final_speed=float(np.abs(sim.base.v).max()),
                commit_pose_err_m=metrics["commit_pose_err"], commit_v_err=metrics["commit_v_err"],
                planner_ms=_ms(planner_ms), admit_ms=_ms(admit_ms), top_candidate_fail_reasons=dict(top_fail),
                admit_log_columns=["t", "kind", "state_pose", "delay_s", "n_cands", "n_tried", "admitted_source", "admitted_margin", "top_candidate_reason"], admit_log=admit_log)
    if sh is not None:                                                                  # keys added ONLY in shadow mode (the truth result is unchanged)
        res["observe"] = "shadow"
        res["shadow"] = sh
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="lattice", choices=["lattice", "mppi"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", default=SCENARIOS)
    ap.add_argument("--horizon", type=float, default=RHConfig.horizon_s)
    ap.add_argument("--lead", type=float, default=RHConfig.lead_s)
    ap.add_argument("--ev", type=float, default=RHConfig.e_v)
    ap.add_argument("--ap", type=float, default=RHConfig.a_p)
    ap.add_argument("--wait-s", type=float, default=RHConfig.wait_s)
    ap.add_argument("--max-wait-s", type=float, default=RHConfig.max_wait_s)
    ap.add_argument("--max-s", type=float, default=200.0)
    ap.add_argument("--no-carry", action="store_true", help="empty hands (the coffee-world sim itself) instead of carrying the tray + cup")
    ap.add_argument("--observe", default="truth", choices=["truth", "shadow"], help="shadow: run the stereo pipeline in parallel and log what the ESTIMATE path would admit (the robot is still driven on truth)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    sc = load_scenario(a.seed, a.scenarios)
    walker = dict(delay_s=sc["walker_delay"], speed=sc["walker_speed"], x0=sc["walker_x0"])
    cfg = RHConfig(horizon_s=a.horizon, lead_s=a.lead, e_v=a.ev, a_p=a.ap, wait_s=a.wait_s, max_wait_s=a.max_wait_s)
    if a.planner == "lattice":
        from .lattice import LatticeProposer
        prop = LatticeProposer()
    else:
        from .mppi_prop import MPPIProposer                                               # written in another worktree
        prop = lambda world: MPPIProposer(world, cfg=cfg)                                 # built after the scene (needs world.bel0.grid)
    out = a.out or f"{OUT_ROOT}/{a.planner}_h{a.horizon:g}"
    r = rh_episode(prop, a.seed, sc["stand_offset"], walker, cfg, max_s=a.max_s, out=out, carry=not a.no_carry, name=a.name, observe=a.observe)
    print(f"[rh] {r['planner']} seed {a.seed}: done={r['done']} ({r['reason']}) t_done={r['t_done']} sim_s={r['sim_s']:.1f} wall_s={r['wall_s']:.0f} replans={r['n_replans']} "
          f"admitted={r['n_admitted']} no_admit={r['n_no_admit']} waits={r['n_waits']} wait_s={r['wait_s_total']:.1f} min_person_centre={r['min_true_person_dist_m']} "
          f"surf={r['min_person_surface_dist_m']} contacts={r['person_contact_ticks']} wall={r['min_wall_clearance_m']:.3f} cup_lost={r['cup_lost']} brake_events={r['n_brake_events']} "
          f"planner_ms={r['planner_ms']['median']:.0f}/{r['planner_ms']['p95']:.0f}", flush=True)


if __name__ == "__main__":
    main()
