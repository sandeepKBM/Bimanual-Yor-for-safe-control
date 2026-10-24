"""tamp/exec_smoke.py -- run a HAND-WRITTEN h5 coffee-fetch plan through the TAMP Executor (no planner).  SIMULATION ONLY, CPU only.

    python -m world_model.v2.house.tamp.exec_smoke --filter none --no-people --seed 0
    python -m world_model.v2.house.tamp.exec_smoke --filter shield --people
    python -m world_model.v2.house.tamp.exec_smoke --filter gatekeeper --people      # GKConfig(person_model="accel", candidates="set", keep_rule="monitor", cup_chord=True)

Plan: move_free(lab_start -> cafe dock, out_route forward) ; pick(tray) ; move_holding(cafe -> lab desk, out_route reversed, base drives BACKWARDS) ; place(tray on desk).
Writes /data/local/ss5772/molmospaces/tamp/exec_smoke/<name>.json and prints a compact summary.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from ..teleop import scene as SC
from ..teleop import shield as SH
from ..teleop.sim import TeleopSim
from . import executor as EX
from .types import Conf, Grasp, Placement, Route, plan_to_json

OUT = Path("/data/local/ss5772/molmospaces/tamp/exec_smoke")


def hold_xy(pose):
    """world xy of the tray centre when the base is at `pose` holding the tray (as TeleopSim places the tray / slab)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (pose[0] + c * SC.TRAY_LOCAL[0] - s * SC.TRAY_LOCAL[1], pose[1] + s * SC.TRAY_LOCAL[0] + c * SC.TRAY_LOCAL[1])


def build_h5_plan(scene):
    """hand-written plan from the SceneSpec waypoints / out_route."""
    wp = scene.waypoints
    q_start = Conf("lab_start", *wp["lab_start"], surface=None)
    q_cafe = Conf("cafe_dock", *wp["cafe_counter_pick"], surface="counter")
    q_desk = Conf("desk_dock", *wp["lab_desk_place"], surface="desk")
    out = tuple((float(x), float(y)) for x, y in scene.out_route)
    r_out = Route(q_start.name, q_cafe.name, out, reverse=False, kind="free")
    r_back = Route(q_cafe.name, q_desk.name, tuple(reversed(out)), reverse=True, kind="holding")        # back_mode "reverse": back along out_route
    g = Grasp("tray", "bimanual_squeeze", 2, mass_range=(0.3, 0.5), mu_range=(0.2, 0.3))            # not certified here (hand-written)
    cx, cy = hold_xy(wp["cafe_counter_pick"])
    dx, dy = hold_xy(wp["lab_desk_place"])
    p_cafe = Placement("tray", "counter", cx, cy, SC.TRAY_HOLD_Z, wp["cafe_counter_pick"][2] + math.pi / 2)
    p_desk = Placement("tray", "desk", dx, dy, SC.COUNTER_TOP_Z + 0.006, wp["lab_desk_place"][2] + math.pi / 2)
    return [("move_free", (q_start, q_cafe, r_out)),
            ("pick", ("tray", p_cafe, g, q_cafe)),
            ("move_holding", ("tray", g, q_cafe, q_desk, r_back)),
            ("place", ("tray", p_desk, g, q_desk, "desk"))]


def make_filter(kind, sim):
    if kind == "none":
        return SH.Passthrough()
    if kind == "shield":
        from .world import RUNTIME_GK
        return SH.Shield(SH.ShieldCfg(cup_row=RUNTIME_GK.get("cup_row", "always")), occ=sim.occ_ctrl, a_hw=sim.a_hw)
    if kind == "gatekeeper":
        import dataclasses
        from ..gatekeeper.belief import GKConfig
        from ..gatekeeper.gate import Gatekeeper
        from .world import RUNTIME_GK
        return Gatekeeper(gk=dataclasses.replace(GKConfig(), **RUNTIME_GK))
    raise ValueError(kind)


def run_smoke(filter_kind="none", people=False, seed=0, max_s=200.0, stall_s=10.0, verbose=True):
    t_wall = time.time()
    sim = TeleopSim("h5", grasped=False, seed=seed, with_people=people, counter=True, desk=True)
    plan = build_h5_plan(sim.scene)
    filt = make_filter(filter_kind, sim)
    ex = EX.Executor(sim, filt, seed=seed)
    results = []
    for name, args in plan:                      # = ex.run_plan(plan, max_s=max_s), unrolled for progress printing
        r = ex.run_plan([(name, args)], stall_s=stall_s, max_s=max_s - sim.t)[0]
        results.append(r)
        if verbose:
            print(f"  {name:13s} {r.status:8s} t={r.t_start:6.1f}->{r.t_end:6.1f}  {r.reason}", flush=True)
        if not r.ok:
            break
    summ = dict(filter=filter_kind, people=people, seed=seed, max_s=max_s, stall_s=stall_s, plan_completed=len(results) == len(plan) and all(r.ok for r in results),
                n_actions=len(plan), n_ok=sum(r.ok for r in results), sim_s=sim.t, wall_s=time.time() - t_wall,
                operator_params={k: getattr(ex.params, k) for k in ("ou_sigma", "bump_rate", "jog_bump_rate", "panic_rate", "lat_bias_sigma", "reaction_s", "sw_radius")},
                plan=plan_to_json(plan), results=[r.to_json() for r in results], episode=ex.episode_metrics(),
                filter_counters=EX._jsonable(filt.counters()) if hasattr(filt, "counters") else {})
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", choices=["shield", "gatekeeper", "none"], default="none")
    ap.add_argument("--people", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-s", type=float, default=200.0, help="total sim-time budget for the plan [s]")
    ap.add_argument("--stall-s", type=float, default=10.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    name = a.name or f"{a.filter}_{'people' if a.people else 'nopeople'}_s{a.seed}"
    print(f"[exec_smoke] {name}", flush=True)
    s = run_smoke(a.filter, a.people, a.seed, a.max_s, a.stall_s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(s, indent=1, default=str))
    e = s["episode"]
    print(f"[exec_smoke] {name}: completed={s['plan_completed']} ({s['n_ok']}/{s['n_actions']}) sim_s={s['sim_s']:.1f} wall_s={s['wall_s']:.0f} "
          f"min_person={e.get('min_person_distance_m')} spilled={e.get('spilled')} person_contacts={e.get('person_contact_ticks')} "
          f"wall_coll={e.get('wall_collision_ticks')} peak_base_acc={e.get('peak_base_accel_ms2'):.2f} peak_tray_acc={e.get('peak_tray_acc_ms2'):.2f} "
          f"interv_frac={e.get('intervention_fraction_gt2pct'):.3f}", flush=True)
    for r in s["results"]:
        print(f"  {r['name']:13s} {r['status']:8s} {r['t_start']:6.1f}->{r['t_end']:6.1f} checks={ {k: v.get('ok') for k, v in r['checks'].items()} } {r['reason']}", flush=True)
    print(f"[exec_smoke] wrote {out / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
