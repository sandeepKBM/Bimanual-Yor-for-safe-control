"""loc/smoke.py -- localization / mapping smokes for the teleoperated shield.  SIMULATION ONLY.

    python -m world_model.v2.house.loc.smoke carry  --seed 0 --cond est_perc --out DIR     # normal h5 carry (shield on)
    python -m world_model.v2.house.loc.smoke stress --seed 0 --cond est_perc --out DIR     # operator pushes the base into each wall in turn (no people)
conditions:  truth        truth pose + true grid                 (the original shield; bit-identical)
             est_true     estimated pose + true grid
             est_perc     estimated pose + perceived grid (head-ZED mapping pass)
             est_perc_m   est_perc + a constant localization margin (--margin, default 0.05 m)
             est_perc_k   est_perc + an adaptive margin = --margin x the estimator 1-sigma proxy (use --margin 1.5)
All conditions use the SAME operator noise seed.  Truth wall clearance (sim.occ + true pose) is always the metric.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import estimator as ES

CONDS = dict(truth=dict(), est_true=dict(pose_source="est"), est_perc=dict(pose_source="est", occ_source="perceived"),
             est_perc_m=dict(pose_source="est", occ_source="perceived"), est_perc_k=dict(pose_source="est", occ_source="perceived"))


def make_kw(cond, seed, loc_scale=1.0, occ_map=None):
    kw = dict(CONDS[cond])
    if kw:
        kw.update(loc_seed=seed, loc_cfg=ES.scaled(ES.LocCfg(), loc_scale))
        if kw.get("occ_source") == "perceived":
            kw["occ_map"] = occ_map
    return kw


def shield_cfg(cond, margin):
    from ..teleop import shield as SH
    if cond == "est_perc_k":                    # adaptive: margin = k * the estimator's own 1-sigma proxy (1 cm floor + 2 cm/sqrt(m since the last relocalization))
        return SH.ShieldCfg(loc_margin_k=margin)
    return SH.ShieldCfg(loc_margin=margin) if cond == "est_perc_m" else None


def run_carry(seed, cond, loc_scale=1.0, margin=0.05, max_s=30.0, occ_map=None, scene="h5"):
    from ..teleop import run as RU
    kw = make_kw(cond, seed, loc_scale, occ_map)
    t0 = time.time()
    s, rows = RU.run_episode(True, seed, "carry", scene, max_s, "closed", None, cfg=shield_cfg(cond, margin), verbose=False, loc_kw=kw or None)
    keep = ["task_done", "task_time_s", "sim_s", "spills_beads_out", "min_wall_clearance_m", "wall_collision_ticks", "physical_house_contacts",
            "min_person_distance_m", "person_contact_ticks", "intervention_fraction_gt2pct", "intervention_labels_ticks", "final_dist_to_goal_m",
            "shield_qp_fallbacks"]
    out = {k: s[k] for k in keep}
    out.update({k: v for k, v in s.items() if k.startswith("loc_")})
    out.update(cond=cond, seed=seed, loc_scale=loc_scale, margin=margin if cond in ("est_perc_m", "est_perc_k") else 0.0, wall_s=time.time() - t0)
    wall_ticks = int(sum(1 for r in rows if r.get("label") == "wall"))
    out["wall_label_ticks"] = wall_ticks
    return out


def run_stress(seed, cond, loc_scale=1.0, margin=0.05, push_s=7.0, occ_map=None, scene="h5"):
    """operator pushes the base (shield on, no people) forward / backward / left / right for push_s each; the shield must stop it at its wall buffer.
    Reports the TRUE footprint-to-wall clearance when stopped, the clearance the shield believed (EDT of its grid at the ESTIMATED pose), and contacts."""
    from ..teleop import sim as SM, shield as SH, geom as GM
    kw = make_kw(cond, seed, loc_scale, occ_map)
    sim = SM.TeleopSim(scene, grasped=True, seed=seed, with_people=False, **kw)
    sh = SH.Shield(shield_cfg(cond, margin), occ=sim.occ_ctrl, a_hw=sim.a_hw)
    wp = sim.scene.waypoints
    sh.docks = []
    pushes = [("fwd", np.array([0.45, 0.0, 0.0])), ("back", np.array([-0.45, 0.0, 0.0])), ("left", np.array([0.0, 0.3, 0.0])),
              ("right", np.array([0.0, -0.3, 0.0]))]
    n_push = int(push_s / SM.TICK)
    res, trace = [], []
    t0 = time.time()
    for name, u in pushes:
        cl_stop, bel_stop, wl = [], [], []
        for k in range(n_push):
            cmd = SM.Cmd(twist=u.copy())
            safe, info = sh.filter(sim, cmd)
            o = sim.tick(safe)
            pe = sim.ctrl_pose()
            bel = float(GM.wall_clearance(sim.occ_ctrl, pe, sim._fp_pts))
            wl.append((o["wall"], bel, float(np.hypot(*sim.base.v[:2]))))
            trace.append((o["t"], o["wall"], bel))
        wl = np.array(wl)
        stopped = wl[:, 2] < 0.02
        tail = wl[-int(1.0 / SM.TICK):]
        res.append(dict(push=name, true_clr_end_m=float(tail[:, 0].mean()), believed_clr_end_m=float(tail[:, 1].mean()),
                        err_end_m=float(tail[:, 1].mean() - tail[:, 0].mean()), true_clr_min_m=float(wl[:, 0].min()),
                        stopped=bool(stopped[-int(1.0 / SM.TICK):].all()), stop_clr_when_stopped_m=float(wl[stopped, 0].min()) if stopped.any() else None))
    tr = np.array(trace)
    out = dict(cond=cond, seed=seed, loc_scale=loc_scale, margin=margin if cond in ("est_perc_m", "est_perc_k") else 0.0, pushes=res,
               true_clr_min_m=float(tr[:, 1].min()), believed_minus_true_max_m=float((tr[:, 2] - tr[:, 1]).max()),
               wall_collision_ticks=int((tr[:, 1] <= 0).sum()), physical_house_contacts=int(sim.bump_contacts),
               wall_s=time.time() - t0)
    if sim.loc is not None:
        out.update({"loc_" + k: v for k, v in sim.loc.errors().items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["carry", "stress"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", choices=list(CONDS), default="truth")
    ap.add_argument("--loc-scale", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--push-s", type=float, default=7.0)
    ap.add_argument("--occ-map", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.kind == "carry":
        r = run_carry(a.seed, a.cond, a.loc_scale, a.margin, a.max_s, a.occ_map)
    else:
        r = run_stress(a.seed, a.cond, a.loc_scale, a.margin, a.push_s, a.occ_map)
    f = out / f"{a.kind}_{a.cond}{a.tag}_s{a.seed}.json"
    f.write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
