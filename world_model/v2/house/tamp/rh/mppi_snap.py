"""tamp/rh/mppi_snap.py -- STATE-SNAPSHOT harness for the MPPI proposer: build the h5 world, take a snapshot (pose, v, people), propose -> api.admit, print the admit
log with the certifier's row margins.  SIMULATION ONLY (no physics stepping; the GPU is used by the MPPI planner only).

    CUDA_VISIBLE_DEVICES=4 python -m world_model.v2.house.tamp.rh.mppi_snap --snapshot standoff_seed1 [--chain 20] [--K 4096] [--horizon 1.5]

Presets (--snapshot):
  standoff_seed1  robot (2.14, 3.56, -1.578) at rest, walker (2.6, 3.8) v = 0, p_stand (2.13449 - 1.32, 3.5) v = 0, goal cafe_counter_pick (2.135, 2.0)
                  (the sweep_1022c2 seed-1 doorway standoff)
  start_seed0     robot at lab_start (3.1, 6.09, -2.33121118) at rest; people = the seed-0 scenario of sweep_1022c2/scenarios.json (tamp.run_tamp._h5_with_walker)
                  sampled from their tracks at --t-obs (default 0.0: position and track velocity at t = 0); goal cafe_counter_pick

--chain N [--static] : receding-horizon chain of N calls from the snapshot: after each call the state advances (open loop, people at constant velocity) to
            (nominal end - lead_s) of the admitted candidate (or by wait_s at rest if none was admitted) and the planner is called again (warm start).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import numpy as np

from ...gatekeeper.checks import world_poses
from .. import plan_cert as PC
from .api import RHConfig, State, admit, plan_gk

SCEN = "/data/local/ss5772/molmospaces/tamp/run_tamp/sweep_1022c2/scenarios.json"
ROWS = ("person", "person_exit", "rest_person", "wall", "cup")


def build_world(seed: int = 0):
    from ...teleop.sim import TeleopSim
    from .. import world as W
    sim = TeleopSim("h5", grasped=False, seed=seed, with_people=False, counter=True, desk=True)
    return sim, W.world_from_sim(sim)


def snapshot(name: str, sim, t_obs: float = 0.0):
    wp = sim.scene.waypoints
    goal = np.array(wp["cafe_counter_pick"], float)
    if name == "standoff_seed1":
        pose = (2.14, 3.56, -1.578)
        people = [(np.array([2.6, 3.8]), np.zeros(2)), (np.array([2.13449 - 1.32, 3.5]), np.zeros(2))]
    elif name == "start_seed0":
        from .. import run_tamp as RT
        from ...teleop import scene as SC
        sc = next(s for s in json.load(open(SCEN)) if s["seed"] == 0)
        spec = RT._h5_with_walker(delay_s=sc["walker_delay"], speed=sc["walker_speed"], x0=sc["walker_x0"], stand_offset=sc["stand_offset"])()
        pose = tuple(wp["lab_start"])
        people = [(np.asarray(p.at(t_obs)[0], float), np.asarray(p.at(t_obs)[1], float)) for p in spec.people if p.dynamic]
    else:
        raise SystemExit(f"unknown snapshot {name!r}")
    return State(pose=pose, v=(0.0, 0.0, 0.0), t=0.0, people=people), goal


def evaluate(cands, state, world, gk, cfg, holding=True):
    """per candidate: the certify_plan verdict + margins, exactly as api.admit runs it (without stopping at the first pass)."""
    g = plan_gk(gk, cfg)
    out = []
    for c in cands:
        ok, m, r, mg = PC.certify_plan(c.traj, world, g, people=state.people, t0=0.0, pose0=state.pose, holding=holding)
        out.append((c, bool(ok), float(m), str(r), mg))
    return out


def fmt(mg):
    return " ".join(f"{k}={mg[k]:+.3f}" for k in ROWS if k in mg)


def advance(state: State, cand, cfg: RHConfig):
    """open-loop next state: along the admitted candidate to (nominal end - lead_s); people at constant velocity.  cand None: wait wait_s at rest."""
    if cand is None:
        dt, pose, v = cfg.wait_s, state.pose, (0.0, 0.0, 0.0)
    else:
        tr = cand.traj
        dt = max(0.3, tr.t_sw - cfg.lead_s)
        n = int(round(dt / tr.h))
        P = tr.poses()[min(n, tr.N):min(n, tr.N) + 1]
        pose = tuple(float(x) for x in world_poses(P, np.asarray(state.pose, float))[0])
        v = tuple(float(x) for x in tr.V[min(n, tr.N)])
    ppl = [(np.asarray(c, float) + np.asarray(vv, float) * dt, np.asarray(vv, float)) for c, vv in state.people]
    return State(pose=pose, v=v, t=state.t + dt, people=ppl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="standoff_seed1", choices=["standoff_seed1", "start_seed0"])
    ap.add_argument("--t-obs", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--horizon", type=float, default=1.5)
    ap.add_argument("--e-v", type=float, default=0.15)
    ap.add_argument("--n-extra", type=int, default=10)
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--static", action="store_true", help="with --chain: keep the snapshot state, only the clock advances by horizon - lead_s (latency under an unchanged scene)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-graph", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    sim, world = build_world()
    print(f"[snap] world built in {time.perf_counter() - t0:.1f} s", flush=True)
    cfg = RHConfig(horizon_s=a.horizon, e_v=a.e_v)
    state, goal = snapshot(a.snapshot, sim, a.t_obs)
    print(f"[snap] {a.snapshot}: pose={tuple(round(float(x), 3) for x in state.pose)} v={state.v} goal={tuple(goal.round(3))} people={[(c.round(2).tolist(), v.round(2).tolist()) for c, v in state.people]}", flush=True)
    from .mppi_prop import MPPIProposer
    t0 = time.perf_counter()
    prop = MPPIProposer(world, cfg, K=a.K, n_extra=a.n_extra, goal_xy=goal[:2], use_graph=not a.no_graph)
    print(f"[snap] proposer ready in {time.perf_counter() - t0:.1f} s (planner warm-up {prop.warmup_ms:.0f} ms)", flush=True)
    res = []
    n = max(1, a.chain)
    for it in range(n):
        cands = prop.propose(state, world, goal, None, cfg)
        ev = evaluate(cands, state, world, world.gk, cfg)
        chosen, log = admit(cands, state, world, world.gk, cfg)
        st = prop.last_stats
        verbose = it == 0 or a.chain == 0
        if verbose:
            print(f"[call {it}] t={state.t:.2f} pose={tuple(round(float(x), 3) for x in state.pose)} v={tuple(round(float(x), 3) for x in state.v)}  plan {st['plan_ms']:.1f} ms  convert {st['convert_ms']:.1f} ms  "
                  f"{len(cands)} candidates", flush=True)
            for k, (c, ok, m, r, mg) in enumerate(ev):
                print(f"   {k:2d} {c.source:12s} progress={c.progress:+.3f} m  ok={int(ok)} min={m:+.3f} ({r:11s}) {fmt(mg)}  t_sw={c.traj.t_sw:.2f}", flush=True)
        src = chosen.source if chosen is not None else None
        res.append(dict(call=it, t=state.t, pose=[float(x) for x in state.pose], admitted=src, progress=None if chosen is None else float(chosen.progress),
                        margins=None if chosen is None else {k: float(v) for k, v in next(e for e in ev if e[0] is chosen)[4].items()},
                        n_ok=int(sum(e[1] for e in ev)), n=len(ev), **{k: float(v) for k, v in st.items() if k.endswith("_ms")}))
        print(f"[call {it}] admit -> {src}" + ("" if chosen is None else f"  (progress {chosen.progress:+.3f} m, {fmt(res[-1]['margins'])})") + f"   [{sum(e[1] for e in ev)}/{len(ev)} certify]", flush=True)
        if it + 1 < n:
            if a.static:
                state = dataclasses.replace(state, t=state.t + cfg.horizon_s - cfg.lead_s)
            else:
                state = advance(state, chosen, cfg)
    if a.chain:
        pm = np.array([r["plan_ms"] for r in res]); cm = np.array([r["convert_ms"] for r in res]); tm = np.array([r["total_ms"] for r in res])
        print(f"[chain {n}] plan_ms median {np.median(pm):.1f} p95 {np.percentile(pm, 95):.1f} max {pm.max():.1f} | convert_ms median {np.median(cm):.1f} p95 {np.percentile(cm, 95):.1f} | "
              f"total_ms median {np.median(tm):.1f} p95 {np.percentile(tm, 95):.1f} | admitted {sum(r['admitted'] is not None for r in res)}/{n}", flush=True)
        print(f"[chain {n}] final pose {tuple(round(float(x), 3) for x in state.pose)}  progress sum {sum(r['progress'] or 0 for r in res):.3f} m", flush=True)
    if a.json:
        open(a.json, "w").write(json.dumps(dict(snapshot=a.snapshot, K=a.K, horizon=a.horizon, warmup_ms=prop.warmup_ms, calls=res), indent=1))


if __name__ == "__main__":
    main()
