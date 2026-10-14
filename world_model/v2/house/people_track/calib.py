"""people_track/calib.py -- collect tracker residuals, fit / evaluate the conformal person margin.  SIMULATION ONLY, CPU (ray casting, no GL).

collect : h5 `carry` episodes, walkers {yield, oblivious, giveup, detour}, the TRUTH-shield drives the robot (so every episode runs the full route and the
          walker is seen the way it would be) while a PerceptionStack (depth detector) runs PASSIVELY on the same sim.  Per tick and per dynamic person
          with a track within 1.5 m, one row:   [seed, walker_id, k, person, since_upd_s, age_s, e_T, |dp|, |dv|, fov, visible, range, r_ksigma]
          e_T = |dp + T dv| (T = 0.5 s) ; r_ksigma = the old isotropic k-sigma inflation (2 sqrt(lambda_max(F P F')), capped 0.5) for comparison.
          Also counts person-ticks with NO track within 1.5 m (misses) split by in-FOV+visible.
fit     : ConformalMargin.fit on the collected rows -> json.
eval    : coverage of a fitted margin (and of the k-sigma radius) on HELD-OUT seeds.

    python -m world_model.v2.house.people_track.calib collect --seeds 100-107 --out DIR/calib --workers 16
    python -m world_model.v2.house.people_track.calib collect --seeds 200-203 --out DIR/heldout --workers 16
    python -m world_model.v2.house.people_track.calib fit --in DIR/calib --alpha 0.05 --out DIR/conformal.json
    python -m world_model.v2.house.people_track.calib eval --in DIR/heldout --conformal DIR/conformal.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from world_model.v2.house.teleop import operator as OP, shield as SH
from world_model.v2.house.teleop.sim import TeleopSim, TICK
from . import perception as PC
from .conformal import ConformalMargin
from .ps_run import WALKERS

WALKER_ID = {k: i for i, k in enumerate(WALKERS)}
T_H = 0.5
MATCH = 1.5
COLS = ["seed", "walker", "k", "person", "since_upd_s", "age_s", "e_T", "dp", "dv", "fov", "visible", "range", "r_ksigma", "trk_range"]


def collect_one(seed, walker, max_s=45.0, detector="depth"):
    pol = WALKERS[walker]
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time", people_policy=pol)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    cfg = replace(SH.ShieldCfg(), **SH.PEOPLE_V2)
    sh = SH.Shield(cfg, occ=sim.occ, a_hw=sim.a_hw)
    sh.docks = [wp["cafe_counter_pick"][:2]]
    perc = PC.PerceptionStack(sim, PC.PerceptionCfg(detector=detector, seed=seed))
    dyn = list(range(len(sc.people)))          # ALL people (static mannequins too): the runtime shield inflates every track
    rows = []
    miss = dict(n_seen_noTrack=0, n_seen=0, n_all_noTrack=0, n_all=0)
    Fh = np.hstack([np.eye(2), T_H * np.eye(2)])
    done_t = None
    cam = perc.cam
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        perc.step(sim)
        tr = perc.tracks
        for i in dyn:
            xy, v = sim.people_state[i]
            fov, vis, rng = perc.vis_now[i]
            best = None
            for t in tr:
                dd = float(np.hypot(*(t.x[:2] - xy)))
                if dd < MATCH and (best is None or dd < best[0]):
                    best = (dd, t)
            miss["n_all"] += 1; miss["n_seen"] += int(vis)
            if best is None:
                miss["n_all_noTrack"] += 1; miss["n_seen_noTrack"] += int(vis)
                continue
            t = best[1]
            dp = t.x[:2] - xy
            dv = t.x[2:] - v
            eT = float(np.hypot(*(dp + T_H * dv)))
            su, ag = PC.track_state(t, sim.k)
            S = Fh @ t.P @ Fh.T
            rk = min(2.0 * math.sqrt(max(float(np.linalg.eigvalsh(S)[-1]), 0.0)), 0.5)
            rows.append([seed, WALKER_ID[walker], k, i, su, ag, eT, float(np.hypot(*dp)), float(np.hypot(*dv)), float(fov), float(vis), float(rng), rk,
                         PC.track_range(t, cam)])
        safe, info = sh.filter(sim, cmd)
        o = sim.tick(safe)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= 1.0:
            break
    return np.array(rows, float).reshape(-1, len(COLS)), miss, dict(done=bool(op.done), sim_s=float(sim.t))


def _job(a):
    seed, walker, out, max_s = a
    f = Path(out) / f"res_{walker}_s{seed}.npz"
    if f.exists():
        return str(f)
    X, miss, info = collect_one(seed, walker, max_s)
    np.savez_compressed(f, X=X, miss=json.dumps(miss), info=json.dumps(info), cols=json.dumps(COLS))
    print(f"collected {walker} s{seed}: {len(X)} rows, miss={miss}, {info}", flush=True)
    return str(f)


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def load(d, walkers=None):
    Xs, miss = [], dict(n_seen_noTrack=0, n_seen=0, n_all_noTrack=0, n_all=0)
    for f in sorted(glob.glob(str(Path(d) / "res_*.npz"))):
        z = np.load(f)
        w = Path(f).stem.split("_")[1]
        if walkers and w not in walkers:
            continue
        Xs.append(z["X"])
        for k, v in json.loads(str(z["miss"])).items():
            miss[k] += v
    return (np.concatenate(Xs) if Xs else np.zeros((0, len(COLS)))), miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "fit", "eval"])
    ap.add_argument("--seeds", default="100-107")
    ap.add_argument("--walkers", default="yield,oblivious,giveup,detour")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-s", type=float, default=45.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--conformal", default=None)
    ap.add_argument("--cap", type=float, default=0.6)
    a = ap.parse_args()
    if a.cmd == "collect":
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        jobs = [(s, w, str(out), a.max_s) for s in parse_seeds(a.seeds) for w in a.walkers.split(",")]
        with mp.Pool(a.workers, maxtasksperchild=3) as pool:
            list(pool.imap_unordered(_job, jobs))
        return
    X, miss = load(a.inp)
    S = X[:, [4, 5, 6, 13]]
    if a.cmd == "fit":
        m = ConformalMargin.fit(S, alpha=a.alpha, t_horizon=T_H, cap=a.cap)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        m.save(a.out)
        print(json.dumps(dict(n=len(S), glob=m.glob, vis=m.vis, cell=m.cell, miss=miss), indent=1))
        print("in-sample coverage:", json.dumps(m.coverage(S))[:400])
        return
    m = ConformalMargin.load(a.conformal)
    cov = m.coverage(S)
    rk = X[:, 12]
    cov["ksigma"] = dict(coverage=float((X[:, 6] <= rk + 1e-12).mean()), mean_radius=float(rk.mean()))
    seen = X[:, 10] > 0
    cov["seen_only"] = dict(n=int(seen.sum()), cov=float((S[seen, 2] <= np.array([m.radius(s[0], s[1], s[3]) for s in S[seen]]) + 1e-12).mean()) if seen.any() else None)
    cov["miss"] = miss
    cov["e_T_quantiles"] = {str(q): float(np.quantile(X[:, 6], q)) for q in (0.5, 0.9, 0.95, 0.99)} if len(X) else None
    print(json.dumps(cov, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()
