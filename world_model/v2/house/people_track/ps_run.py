"""people_track/ps_run.py -- moving-walker person-safety harness (TRUE geometry metrics).  SIMULATION ONLY.

One h5 coffee-run `carry` episode with scripted walkers that REALLY walk (TeleopSim(people_clock="time")), a per-person reaction policy
(yield distance / give-up timeout) and a shield variant.  Metrics use ground-truth geometry, independent of what the shield sees:
  * footprint clearance   = rect_dist(footprint rectangle, person axis) - torso radius          (the shield's own geometry)
  * TRUE clearance        = min mujoco.mj_geomDistance over ALL robot+tray geoms (base, tray, both arms, grippers) vs the person capsule
                            (the carried tray / arms stick out beyond the footprint AABB when the arms move)
  * contact ticks, split by robot MOVING (|v_xy| > 0.03 m/s or |wz| > 0.05) vs STATIONARY at the contact tick
  * deadlock = ticks where the person is yielding AND the robot is (almost) stationary AND the task is not done
Usage:
    python -m world_model.v2.house.people_track.ps_run --seed 0 --walker yield --shield old --out DIR
    --walker {yield,oblivious,patient,giveup,detour}    --shield {old,new}   --cond {truth,depth,seg,param:..}   --trace (per-tick npz)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import mujoco

from world_model.v2.house.teleop import base as BS, operator as OP, shield as SH, geom as GM
from world_model.v2.house.teleop import run as RUN
from world_model.v2.house.teleop.sim import TeleopSim, TICK, PersonPolicy, PERSON_TORSO_R

OUT = Path(os.environ.get("PEOPLE_SHIELD_OUT", "/data/local/ss5772/molmospaces/people_shield"))

WALKERS = {
    "yield": PersonPolicy(),                                                          # stops 0.15 m from the robot, waits forever
    "oblivious": PersonPolicy(yield_m=None),                                           # follows the script regardless of the robot
    "giveup": PersonPolicy(giveup_s=3.0, giveup_mode="continue"),                     # waits 3 s then walks on
    "detour": PersonPolicy(giveup_s=2.0, giveup_mode="detour"),                       # waits 2 s then side-steps around the robot
}


class TrueGeom:
    """min distance between every robot/tray geom and the person capsule + head (mj_geomDistance, ground truth)."""

    def __init__(self, sim):
        m = sim.m
        self.sim = sim
        self.rg = [g for g in range(m.ngeom) if sim.robot_geom[g] and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
        self.pg = {p.name: [m.geom(f"{p.name}_torso").id, m.geom(f"{p.name}_head").id] for p in sim.scene.people if p.dynamic and sim.with_people}
        self.fromto = np.zeros(6)

    def min_dist(self, rmax=1.5):
        sim = self.sim
        m, d = sim.m, sim.d
        best, who = np.inf, None
        for nm, gl in self.pg.items():
            for pgid in gl:
                for rg in self.rg:
                    if np.linalg.norm(d.geom_xpos[rg][:2] - d.geom_xpos[pgid][:2]) > 2.5:
                        continue
                    dist = mujoco.mj_geomDistance(m, d, rg, pgid, rmax, self.fromto)
                    if dist < best:
                        best, who = dist, (nm, m.geom(rg).name or m.body(m.geom_bodyid[rg]).name)
        return float(best), who


def build_shield(kind, sim, cond, carry, seed, dock_wp, cfg_over=None):
    cfg = None
    if kind == "new":
        cfg = replace(SH.ShieldCfg(), **{**SH.PEOPLE_V2, **(cfg_over or {})})
    elif cfg_over:
        cfg = replace(SH.ShieldCfg(), **cfg_over)
    if cond == "truth":
        sh = SH.Shield(cfg, occ=sim.occ, a_hw=sim.a_hw)
        perc = None
    else:
        from . import perception as PC, shield_pt as SPT
        from .run_pt import parse_cond
        pcfg, ucfg = parse_cond(cond)
        perc = PC.PerceptionStack(sim, replace(pcfg, seed=seed))
        sh = SPT.PerceivedShield(perc, cfg, occ=sim.occ, a_hw=sim.a_hw, ucfg=ucfg)
    sh.docks = dock_wp
    return sh, perc


def run(seed=0, walker="yield", shield="old", cond="truth", max_s=60.0, scene="h5", trace=None, stuck_s=25.0, tail_s=1.0, verbose=False, cfg_over=None):
    pol = WALKERS[walker]
    sim = TeleopSim(scene, grasped=True, seed=seed, with_people=True, people_clock="time", people_policy=pol)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    sh, perc = build_shield(shield, sim, cond, True, seed, [wp["cafe_counter_pick"][:2]], cfg_over)
    tg = TrueGeom(sim)
    rows, tr, ctl_ms = [], [], []
    done_t = None
    t_wall = time.time()
    n_stuck = int(stuck_s / TICK)
    aborted = False
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        safe, info = sh.filter(sim, cmd)
        pre_v = sim.base.v.copy()
        o = sim.tick(safe)
        o.update(info)
        o["interv_vs_raw"] = 0.0
        o["u_op"] = cmd.twist.copy(); o["u_safe"] = safe.twist.copy(); o["phase"] = op.phase
        rows.append(o)
        near = min((np.hypot(*(xy - sim.pose()[:2])) for xy, _ in sim.people_state), default=9.0) < 3.0
        td, who = tg.min_dist() if near else (np.inf, None)
        spd = float(np.hypot(*sim.base.v[:2])); wz = abs(float(sim.base.v[2]))
        moving = spd > 0.03 or wz > 0.05
        yl = getattr(sim, "person_yielding", {})
        tr.append(dict(t=o["t"], pose=o["pose"].copy(), v=sim.base.v.copy(), u_op=cmd.twist.copy(), u_safe=safe.twist.copy(), label=info["label"],
                       qp=info["qp"], pd=o["pdist"], true=td, who=who, moving=moving, phase=op.phase,
                       yielding=any(yl.values()), ppl=[(xy.copy(), v.copy()) for xy, v in sim.people_state]))
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
        if not op.done and k > n_stuck and k * TICK > stuck_s:
            if np.hypot(*(rows[-1]["pose"][:2] - rows[-n_stuck]["pose"][:2])) < 0.05:
                aborted = True
                break
    summ = RUN.summarize(sim, op, sh, rows, shield != "raw", seed, "carry", done_t, [0.0], time.time() - t_wall)
    T = tr
    pdv = np.array([x["pd"] for x in T]); tdv = np.array([x["true"] for x in T]); mv = np.array([x["moving"] for x in T])
    yl = np.array([x["yielding"] for x in T])
    fp_c = pdv <= 0
    tr_c = tdv <= 0
    stat = ~mv
    # deadlock: yielding walker + stationary robot while the task is unfinished
    dead = yl & stat & np.array([x["phase"] in ("OUT", "BACK") for x in T])
    summ.update(
        n_deadlock_yield=int(getattr(sh, 'n_deadlock', 0)), n_squeeze_yield=int(getattr(sh, 'n_squeeze', 0)), n_pred_rows_used=int(getattr(sh, 'n_pred', 0)),
        walker=walker, shield_variant=shield, cond=cond, aborted_stuck=bool(aborted),
        fp_contact_ticks=int(fp_c.sum()), fp_contact_moving=int((fp_c & mv).sum()), fp_contact_stationary=int((fp_c & stat).sum()),
        true_contact_ticks=int(tr_c.sum()), true_contact_moving=int((tr_c & mv).sum()), true_contact_stationary=int((tr_c & stat).sum()),
        min_fp_clear_m=float(pdv.min()), min_true_clear_m=float(tdv.min()) if np.isfinite(tdv.min()) else None,
        min_true_clear_who=(T[int(np.argmin(tdv))]["who"] if np.isfinite(tdv.min()) else None),
        deadlock_s=float(dead.sum() * TICK), yielding_s=float(yl.sum() * TICK), stationary_s=float(stat.sum() * TICK),
        walker_final_xy=[float(x) for x in sim.people_state[0][0]],
    )
    if trace:
        np.savez_compressed(trace, t=[x["t"] for x in T], pose=[x["pose"] for x in T], v=[x["v"] for x in T], u_op=[x["u_op"] for x in T],
                            u_safe=[x["u_safe"] for x in T], pd=pdv, true=tdv, label=[x["label"] for x in T], moving=mv, yielding=yl,
                            walker=[x["ppl"][0][0] for x in T], phase=[x["phase"] for x in T])
    return summ, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--walker", default="yield", choices=list(WALKERS))
    ap.add_argument("--shield", default="old", choices=["old", "new"])
    ap.add_argument("--cond", default="truth")
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--cfg", default="", help="ShieldCfg overrides, e.g. squeeze_s=None,pred_taus=() (applied on top of PEOPLE_V2 for --shield new)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import ast
    cfg_over = {k.strip(): ast.literal_eval(v.strip()) for k, v in (kv.split("=", 1) for kv in a.cfg.split(";") if kv)} if a.cfg else None
    tag = f"{a.walker}_{a.shield}{a.tag}_{a.cond.replace(':', '-')}_s{a.seed}"
    summ, T = run(a.seed, a.walker, a.shield, a.cond, a.max_s, trace=(out / f"{tag}.npz") if a.trace else None, cfg_over=cfg_over)
    (out / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
    keys = ["task_done", "task_time_s", "spills_beads_out", "person_contact_ticks", "true_contact_ticks", "true_contact_moving",
            "true_contact_stationary", "min_fp_clear_m", "min_true_clear_m", "min_true_clear_who", "deadlock_s", "n_deadlock_yield", "n_squeeze_yield", "aborted_stuck", "walker_final_xy"]
    print(tag, {k: summ[k] for k in keys}, flush=True)


if __name__ == "__main__":
    main()
