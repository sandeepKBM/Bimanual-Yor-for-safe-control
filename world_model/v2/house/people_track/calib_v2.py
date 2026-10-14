"""people_track/calib_v2.py -- conformal-margin calibration for the estimated-pose detectors (neural / fusion / depth_perc) on the human-mesh scene.  SIMULATION ONLY.

Same residual table and fit as people_track.calib (COLS, ConformalMargin), but
  * the sim runs pose_source='est' (+ optional visual human meshes) and one PerceptionStack PER DETECTOR runs passively on the same episode (the neural / fusion
    stacks share one RGB render + one inference per frame);
  * tracks live in the ESTIMATED frame, so the truth they are compared with is mapped into that frame (p_est = Rz(dyaw)(p - b_true) + b_est): the residual is
    the RELATIVE (robot - person) error the shield actually sees, not the common-mode localization error;
  * one output dir per detector:  <out>/<det>/res_<walker>_s<seed>.npz  -> `fit` writes conformal_<det>[_aNN].json (picked up by ps_run via PEOPLE_CONFORMAL_DIR).

    python -m world_model.v2.house.people_track.calib_v2 collect --detectors neural,fusion@both --human-mesh articulated --seeds 100-107 --workers 8 \
        --sock /path/detect.sock --out DIR/calib_hm
    python -m world_model.v2.house.people_track.calib_v2 fit --in DIR/calib_hm --detectors neural,fusion --alphas 0.05,0.10,0.20 --out DIR/conformal_hm
    # then:  PEOPLE_CONFORMAL_DIR=DIR/conformal_hm python -m ...ps_sweep --conds 'neural+look+conf20,fusion@both+look+conf20' --human-mesh articulated ...
"""
from __future__ import annotations

import argparse
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
from .calib import COLS, MATCH, T_H, WALKER_ID, load, parse_seeds
from .conformal import ConformalMargin
from .detect_v2 import est_correction
from .ps_run import WALKERS


def _tag(det):
    """output tag of a detector cond ('neural', 'fusion@both', 'fusion@both+surf3'): '@' dropped, '+' -> '_'  (ps_run looks up conformal_<tag>[_aNN].json)."""
    return det.replace("@", "").replace("+", "_")


def collect_one(seed, walker, dets, human_mesh=None, sock="", max_s=45.0):
    pol = WALKERS[walker]
    SimCls = TeleopSim
    kw = {}
    if human_mesh:
        from .human_vis import human_sim_class
        SimCls = human_sim_class(TeleopSim)
        kw["human_variant"] = human_mesh
    sim = SimCls("h5", grasped=True, seed=seed, with_people=True, people_clock="time", people_policy=pol, pose_source="est", occ_source="perceived", **kw)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    cfg = replace(SH.ShieldCfg(), **SH.PEOPLE_V2)
    sh = SH.Shield(cfg, occ=sim.occ_ctrl, a_hw=sim.a_hw)             # the TRUTH shield drives (every episode runs the full route); perception is passive
    sh.docks = [wp["cafe_counter_pick"][:2]]
    stacks = {}
    for d in dets:
        from .run_pt import parse_cond
        pcfg, _ = parse_cond(d)                                      # same cond grammar as ps_run ('fusion@both', 'neural+surf3', ...)
        stacks[d] = PC.PerceptionStack(sim, replace(pcfg, seed=seed, neural_sock=sock))
    rows = {d: [] for d in dets}
    miss = {d: dict(n_seen_noTrack=0, n_seen=0, n_all_noTrack=0, n_all=0) for d in dets}
    Fh = np.hstack([np.eye(2), T_H * np.eye(2)])
    done_t = None
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        bt, be, dy = est_correction(sim)
        c, s = math.cos(dy), math.sin(dy)
        Rz = np.array([[c, -s], [s, c]])
        for d, perc in stacks.items():
            perc.step(sim)
            for i in range(len(sc.people)):
                xy, v = sim.people_state[i]
                xy_e, v_e = Rz @ (np.asarray(xy) - bt) + be, Rz @ np.asarray(v)
                fov, vis, rng = perc.vis_now[i]
                best = None
                for t in perc.tracks:
                    dd = float(np.hypot(*(t.x[:2] - xy_e)))
                    if dd < MATCH and (best is None or dd < best[0]):
                        best = (dd, t)
                miss[d]["n_all"] += 1; miss[d]["n_seen"] += int(vis)
                if best is None:
                    miss[d]["n_all_noTrack"] += 1; miss[d]["n_seen_noTrack"] += int(vis)
                    continue
                t = best[1]
                dp, dv = t.x[:2] - xy_e, t.x[2:] - v_e
                eT = float(np.hypot(*(dp + T_H * dv)))
                su, ag = PC.track_state(t, sim.k)
                S = Fh @ t.P @ Fh.T
                rk = min(2.0 * math.sqrt(max(float(np.linalg.eigvalsh(S)[-1]), 0.0)), 0.5)
                rows[d].append([seed, WALKER_ID[walker], k, i, su, ag, eT, float(np.hypot(*dp)), float(np.hypot(*dv)), float(fov), float(vis), float(rng), rk,
                                PC.track_range(t, perc.cam)])
        safe, info = sh.filter(sim, cmd)
        o = sim.tick(safe)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= 1.0:
            break
    return {d: (np.array(r, float).reshape(-1, len(COLS)), miss[d]) for d, r in rows.items()}, dict(done=bool(op.done), sim_s=float(sim.t))


def _job(a):
    seed, walker, dets, hm, sock, out, max_s = a
    todo = [d for d in dets if not (Path(out) / _tag(d) / f"res_{walker}_s{seed}.npz").exists()]
    if not todo:
        return 0
    res, info = collect_one(seed, walker, todo, hm, sock, max_s)
    for d, (X, miss) in res.items():
        f = Path(out) / _tag(d) / f"res_{walker}_s{seed}.npz"
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, X=X, miss=json.dumps(miss), info=json.dumps(info), cols=json.dumps(COLS))
        print(f"collected {d} {walker} s{seed}: {len(X)} rows, miss={miss}, {info}", flush=True)
    return len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "fit"])
    ap.add_argument("--detectors", default="neural,fusion@both")
    ap.add_argument("--human-mesh", default=None, choices=[None, "articulated"])
    ap.add_argument("--sock", default=os.environ.get("DETECT_SOCK", ""))
    ap.add_argument("--seeds", default="100-107")
    ap.add_argument("--walkers", default="yield,oblivious,giveup,detour")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-s", type=float, default=45.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--cap", type=float, default=0.6)
    a = ap.parse_args()
    dets = a.detectors.split(",")
    if a.cmd == "collect":
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        jobs = [(s, w, dets, a.human_mesh, a.sock, str(out), a.max_s) for s in parse_seeds(a.seeds) for w in a.walkers.split(",")]
        with mp.Pool(a.workers, maxtasksperchild=2) as pool:
            list(pool.imap_unordered(_job, jobs, chunksize=1))
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for d in dets:
        dd = Path(a.inp) / _tag(d)
        if not dd.exists():                                            # first calibration used tags 'neural' / 'fusion'
            dd = Path(a.inp) / d.split("@")[0].split("+")[0]
        X, miss = load(dd)
        S = X[:, [4, 5, 6, 13]]
        for al in [float(v) for v in a.alphas.split(",")]:
            m = ConformalMargin.fit(S, alpha=al, t_horizon=T_H, cap=a.cap)
            nm = f"conformal_{_tag(d)}.json" if abs(al - 0.05) < 1e-9 else f"conformal_{_tag(d)}_a{int(round(al * 100)):02d}.json"
            m.save(str(out / nm))
            cov = m.coverage(S)
            print(d, al, nm, "n=", len(S), "glob=", getattr(m, "glob", None), "in-sample coverage:", json.dumps(cov)[:200], "miss:", miss, flush=True)


if __name__ == "__main__":
    main()
