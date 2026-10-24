"""tamp/hop_e3.py -- E3: the T4(c) hop lemma versus the gatekeeper's checker.  SIMULATION ONLY (checker calls only, no stepping).

    python -m world_model.v2.house.tamp.hop_e3 [--cup] [--json /data/local/ss5772/...json]

One STATIC person (truth belief, e_p = 0, v_hat = 0, accel person model = the runtime gatekeeper config) at footprint distance d straight ahead of the robot, empty hands (or
--cup: tray held, cup row active), no walls (grid removed, so only the person rows decide).  Hop direction = straight at the person (the lemma's worst case).
Columns:
    d          m      distance from the person centre to the front edge of the footprint
    t_bar      s      lemma t_bar:  largest t with  rho_i0 + d_person + g(t) + sw(t) + a_r(t) t^2/4 <= d  (accel/speed discs, either one; sw = the checker's sweep allowance at the peak speed)
    t_bar_raw  s      the lemma as written in the draft (no sweep allowance, no d_person)
    t_h        s      LONGEST admissible hop found by check1 + check2 over EVERY multiple of h (0.05 s)
    t_h_lib    s      same over the library grid (--hop-durations default) and the 0.02 m minimum displacement
    disp       m      displacement of the t_h hop
    a_r        m/s^2  acceleration used
Expected relation: t_h <= t_bar + h (the checker evaluates the last MOVING sample, t - h; the lemma uses t); t_h much smaller than t_bar would be a refusal the lemma does not predict.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math

import numpy as np

from ..teleop import geom as GM
from ..teleop.sim import TeleopSim
from . import hop as HOP
from . import world as W

DISTS = (0.36, 0.40, 0.45, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5)
ROBOT_POSE = (0.0, 0.0, 0.0)


def belief_with_person(world, d, holding=False, pose=ROBOT_POSE):
    """truth belief with one static person at footprint distance d straight ahead (+x of the robot at `pose`, yaw 0), no walls."""
    fx = GM.FP["x"][1]
    c = np.array([pose[0] + fx + d, pose[1]])
    b = world.bel0.with_(people=(c,), people_vel=(np.zeros(2),), people_e_p=(), people_e_v=(), people_stale=(), grid=None)
    w2 = dataclasses.replace(world, bel_people=b)
    return W.runtime_belief_at(w2, pose, holding, "tray" if holding else None, people=True).with_(grid=None)


def e3_rows(world, gk, dists=DISTS, holding=False):
    rows = []
    for d in dists:
        bel = belief_with_person(world, d, holding)
        full = HOP.longest_admissible(bel, gk, (1.0, 0.0), durations=[gk.h * n for n in range(40, 1, -1)], min_m=0.0)
        lib = HOP.longest_admissible(bel, gk, (1.0, 0.0), HOP.HOP_DURATIONS, HOP.HOP_MIN_M)
        rows.append(dict(d=d, t_bar=HOP.lemma_tbar(bel, gk, d), t_bar_raw=HOP.lemma_tbar(bel, gk, d, sweep=False, d_person=False),
                         t_h=0.0 if full is None else full["t_h"], t_h_lib=0.0 if lib is None else lib["t_h"],
                         disp=0.0 if full is None else full["disp"], a_r=None if full is None else full["a_r"],
                         a_r_lib=None if lib is None else lib["a_r"], disp_lib=0.0 if lib is None else lib["disp"]))
    return rows


def frozen_radius(gk, bel):
    """analytic frozen radius from the implemented rows (T4(b)): r0 + d_person + sweep at sample 0, per person model; the checker's accel/speed max-margin rule -> the smaller."""
    r0 = GM.PERSON_CLEAR_R + bel.e_p
    h = gk.h
    return dict(speed=r0 + bel.d_person + 0.5 * h * bel.v_p_max, accel=r0 + bel.d_person + 0.5 * h * (bel.e_v + bel.a_p * h))


def print_table(rows, title):
    print(title)
    print(f"{'d[m]':>6} {'t_bar':>7} {'t_bar_raw':>9} {'t_h':>6} {'t_h_lib':>7} {'disp[m]':>8} {'a_r':>6}  note")
    for r in rows:
        note = ""
        if r["t_h"] == 0.0 and r["t_bar"] == 0.0:
            note = "frozen (no hop; lemma agrees)"
        elif r["t_h"] > r["t_bar"] + 0.05 + 1e-9:
            note = "CHECKER ADMITS LONGER THAN LEMMA (mismatch)"
        elif r["t_h"] < r["t_bar"] - 0.10 - 1e-9:
            note = "checker refuses > 2 grid steps below lemma"
        print(f"{r['d']:6.2f} {r['t_bar']:7.3f} {r['t_bar_raw']:9.3f} {r['t_h']:6.2f} {r['t_h_lib']:7.2f} {r['disp']:8.4f} {(r['a_r'] or 0):6.2f}  {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cup", action="store_true", help="also print the table with the tray held (cup row active)")
    ap.add_argument("--json", default=None, help="optional: write the rows here (use /data/local/ss5772/...)")
    a = ap.parse_args()
    sim = TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)
    world = W.world_from_sim(sim)
    gk = world.gk
    out = {}
    for holding in ((False, True) if a.cup else (False,)):
        rows = e3_rows(world, gk, holding=holding)
        b0 = belief_with_person(world, 1.0, holding)
        print(f"[E3] hands {'with tray (cup row)' if holding else 'empty'}: h={gk.h} a_cup_max={b0.a_cup_max:.3f} plan_frac*a_cup_max={gk.plan_frac * b0.a_cup_max:.3f} a_hw={list(b0.a_hw)} "
              f"v_cap={list(b0.v_cap)} person_model={b0.person_model} v_p={b0.v_p_max} e_v={b0.e_v} a_p={b0.a_p} d_person={b0.d_person} r0={GM.PERSON_CLEAR_R + b0.e_p}")
        print(f"[E3] frozen radius (r0 + d_person + sweep_0): {frozen_radius(gk, b0)}")
        print_table(rows, "[E3] table")
        out["tray" if holding else "empty"] = rows
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
