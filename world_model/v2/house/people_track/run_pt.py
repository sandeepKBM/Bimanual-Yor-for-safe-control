"""people_track/run_pt.py -- the h5 coffee-run `carry` episode with the shield fed by a camera tracker instead of ground truth.  SIMULATION ONLY.

Conditions (`--cond`):
    truth                    original shield on sim.people_state (exactly teleop/run.py's shielded run; + visibility bookkeeping)
    seg                      PRIVILEGED seg-mask person points  -> KF -> uncertainty-inflated keep-out
    depth                    depth-only geometry               -> KF -> uncertainty-inflated keep-out
    param:SIGMA:LAT:DROP     truth + N(0, SIGMA^2) + LAT ticks latency + DROP frame-dropout, FOV+LOS gated -> KF -> keep-out
    any of the above with suffix '+noinfl' (k_sigma = 0, original 0.30+0.10 radius, no unseen cap), '+nocap' (inflation, no unseen cap), '+buf' (inflation + the original 0.10 buffer)
    depth@occ / depth@both : static rejection with the shield's sparse sim.occ only / background-subtraction + sim.occ (default depth = background subtraction)

Single run:  python -m world_model.v2.house.people_track.run_pt --cond depth --seed 0 --max-s 40
Sweep:       python -m world_model.v2.house.people_track.run_pt --sweep --seeds 0-8 --conds truth,seg,depth,param:0.05:3:0.0,param:0.15:3:0.1 --workers 40
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

from world_model.v2.house.teleop import base as BS, operator as OP, shield as SH
from world_model.v2.house.teleop import run as RUN
from world_model.v2.house.teleop.sim import TeleopSim, TICK
from . import perception as PC
from . import shield_pt as SPT
from .tracker import TrackerCfg
from .sim_fix import TeleopSimWalking

OUT = Path(os.environ.get("PEOPLE_TRACK_OUT", "data/molmospaces/people_track"))
MATCH_M = 0.75


def parse_cond(cond):
    """-> (PerceptionCfg or None for truth, UncertaintyCfg)"""
    base, *flags = cond.split("+")
    smap = "raycast"
    if "@" in base:
        base, smap = base.split("@")
    u = SPT.UncertaintyCfg()
    if "noinfl" in flags:
        u = replace(u, k_sigma=0.0, d_buffer=0.10, unseen_cap=False)     # original radius (0.30 + 0.10) on tracker output: isolates perception error from the margin policy
    if "soft" in flags:
        u = replace(u, soft_floor=0.05)
    if "buf" in flags:
        u = replace(u, d_buffer=0.10)
    if "nocap" in flags:
        u = replace(u, unseen_cap=False)
    if base == "truth":
        return None, u
    parts = base.split(":")
    if parts[0] == "param":
        sig, lat, drop = float(parts[1]), int(parts[2]), float(parts[3])
        return PC.PerceptionCfg(detector="param", sigma=sig, latency_ticks=lat, p_drop=drop), u
    pc = PC.PerceptionCfg(detector=parts[0], static_map=smap)
    sf = [f for f in flags if f.startswith("surf")]               # neural / fusion: '+surf' / '+surf5' = visible-surface centroid + 3 / 5 cm (no r=0.22 prior); default: the prior
    if sf:
        pc = replace(pc, neural_centroid="surf", neural_r_off=(float(sf[0][4:]) / 100.0 if sf[0][4:] else 0.03))
    return pc, u


def run_episode_pt(cond="depth", seed=0, task="carry", scene="h5", max_s=40.0, tail_s=1.0, video=None, log_every=0, with_people=True, verbose=False, walk=True, yield_m=None, stuck_s=10.0, human_mesh=None):
    pcfg, ucfg = parse_cond(cond)
    carry = task == "carry"
    SimCls = TeleopSimWalking if walk else TeleopSim
    if walk and yield_m is not None:
        SimCls = type("TeleopSimWalkingY", (TeleopSimWalking,), dict(yield_m=float(yield_m)))
    est_kw = dict(pose_source="est", occ_source="perceived") if (pcfg is not None and pcfg.detector in ("depth_perc", "neural", "fusion", "full")) else {}     # estimated-pose detectors
    if human_mesh:                                           # visual Rocketbox humans (people_track.human_vis), default off
        from .human_vis import human_sim_class
        SimCls = human_sim_class(SimCls)
        est_kw = dict(est_kw, human_variant=human_mesh)
    sim = SimCls(scene, grasped=carry, seed=seed, with_people=with_people, counter=not carry, desk=not carry, **est_kw)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, task, sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    if pcfg is None:
        perc = PC.TruthPerception(sim)
        shield = SH.Shield(None, occ=sim.occ, a_hw=sim.a_hw)
        _orig_filter = shield.filter

        class _Wrap:                                     # truth: step the bookkeeping, then the unmodified Shield
            n_fallback = property(lambda s: shield.n_fallback)
            n_veto = property(lambda s: shield.n_veto)

            def filter(s, sim_, cmd):
                perc.step(sim_)
                return _orig_filter(sim_, cmd)
        shield_obj = _Wrap()
        shield.docks = [wp["cafe_counter_pick"][:2], wp["lab_desk_place"][:2]] if not carry else [wp["cafe_counter_pick"][:2]]
    else:
        perc = PC.PerceptionStack(sim, replace(pcfg, seed=seed))
        shield = SPT.PerceivedShield(perc, None, occ=sim.occ, a_hw=sim.a_hw, ucfg=ucfg)
        shield.docks = [wp["cafe_counter_pick"][:2], wp["lab_desk_place"][:2]] if not carry else [wp["cafe_counter_pick"][:2]]
        shield_obj = shield
    rec = RUN.Recorder(sim, str(video)) if video else None
    rows, ctl_ms, log = [], [], []
    stuck = False
    n_stuck = int(stuck_s / TICK) if stuck_s else 0
    done_t = None
    t_wall = time.time()
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        truth = [(xy.copy(), v.copy()) for xy, v in sim.people_state]       # state at the start of this tick (what perception sees)
        safe, info = shield_obj.filter(sim, cmd)
        log.append(dict(t=sim.t, k=sim.k, truth=truth, vis=list(perc.vis_now),
                        tracks=[(t.x.copy(), t.P.copy()) for t in perc.tracks]))
        t0 = time.perf_counter()
        o = sim.tick(safe)
        ctl_ms.append((time.perf_counter() - t0) * 1000 / 4)
        o.update(info)
        u_hw = np.clip(cmd.twist, obs["v"] - sim.a_hw * TICK, obs["v"] + sim.a_hw * TICK)
        o["interv_vs_raw"] = float(np.linalg.norm((u_hw - safe.twist) / BS.V_STICK))
        o["u_op"] = cmd.twist.copy(); o["u_safe"] = safe.twist.copy(); o["phase"] = op.phase
        rows.append(o)
        if rec:
            rec.frame(sim, k, o, cond)
        if verbose and log_every and k % log_every == 0:
            print(f"[{cond}] t={o['t']:5.1f} ph={op.phase} pose=({o['pose'][0]:.2f},{o['pose'][1]:.2f}) pd={o['pdist']:.2f} "
                  f"ntr={len(perc.tracks)} lab={o['label']}", flush=True)
        if op.done and done_t is None:
            done_t = o["t"]
        if stuck_s and not op.done and k * TICK > stuck_s and k > n_stuck:      # deadlock guard: robot has not moved for stuck_s of sim time
            if np.hypot(*(rows[-1]["pose"][:2] - rows[-n_stuck]["pose"][:2])) < 0.05:
                stuck = True
                break
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
    if rec:
        rec.close()
    summ = RUN.summarize(sim, op, shield_obj, rows, True, seed, task, done_t, ctl_ms, time.time() - t_wall)
    summ.update(cond=cond, aborted_stuck=bool(stuck), person_contact=bool(summ["person_contact_ticks"] > 0), **perception_metrics(sim, perc, log, rows))
    return summ, rows


def perception_metrics(sim, perc, log, rows):
    n = len(log)
    npers = len(log[0]["truth"]) if n else 0
    names = [p.name for p in sim.scene.people][:npers]
    out = dict(persons={}, n_frames=perc.n_frames, render_ms_mean=float(np.mean(perc.render_ms)) if perc.render_ms else 0.0)
    all_pos_err, all_vel_err = [], []
    n_false = 0
    for i, nm in enumerate(names):
        fov = np.array([l["vis"][i][0] for l in log]); vis = np.array([l["vis"][i][1] for l in log])
        det_t = np.full(n, False); pe = np.full(n, np.nan); ve = np.full(n, np.nan)
        for j, l in enumerate(log):
            xy, v = l["truth"][i]
            best = None
            for x, _P in l["tracks"]:
                dd = float(np.hypot(*(x[:2] - xy)))
                if best is None or dd < best[0]:
                    best = (dd, x)
            if best is not None and best[0] < MATCH_M:
                det_t[j] = True
                pe[j] = best[0]
                ve[j] = float(np.hypot(*(best[1][2:] - v)))
        fv = np.flatnonzero(vis)
        t_first_vis = float(log[fv[0]]["t"]) if len(fv) else None
        dd_ = np.flatnonzero(det_t)
        t_first_det = float(log[dd_[0]]["t"]) if len(dd_) else None
        ttfd = None
        if len(fv):
            after = dd_[dd_ >= fv[0]]
            ttfd = float(log[after[0]]["t"] - log[fv[0]]["t"]) if len(after) else None
        # frame-level recall of the detector (detections delivered for frames captured while the person was visible)
        rec_n = rec_h = 0
        for kc, _k, dets in perc.frame_log:
            if kc < n and log[kc]["vis"][i][1]:
                rec_n += 1
                xy = log[kc]["truth"][i][0]
                rec_h += int(any(np.hypot(*(d - xy)) < MATCH_M for d, _t in dets))
        out["persons"][nm] = dict(
            frac_in_fov=float(fov.mean()), frac_visible=float(vis.mean()), t_first_visible_s=t_first_vis, t_first_confirmed_s=t_first_det,
            time_to_first_detection_s=ttfd, missed_tick_rate=float(1 - det_t[vis].mean()) if vis.any() else None,
            frame_recall=float(rec_h / rec_n) if rec_n else None, n_visible_frames=rec_n,
            rmse_pos_m=float(np.sqrt(np.nanmean(pe ** 2))) if np.isfinite(pe).any() else None,
            rmse_vel_ms=float(np.sqrt(np.nanmean(ve ** 2))) if np.isfinite(ve).any() else None,
            p95_pos_err_m=float(np.nanpercentile(pe, 95)) if np.isfinite(pe).any() else None)
        all_pos_err.append(pe[vis]); all_vel_err.append(ve[vis])
    for l in log:                                           # tracks > 1.5 m from every person = false tracks
        for x, _P in l["tracks"]:
            if not any(np.hypot(*(x[:2] - xy)) < 1.5 for xy, _v in l["truth"]):
                n_false += 1
    out["false_track_ticks"] = int(n_false)
    pe = np.concatenate(all_pos_err) if all_pos_err else np.zeros(0)
    ve = np.concatenate(all_vel_err) if all_vel_err else np.zeros(0)
    out["rmse_pos_visible_m"] = float(np.sqrt(np.nanmean(pe ** 2))) if np.isfinite(pe).any() else None
    out["rmse_vel_visible_ms"] = float(np.sqrt(np.nanmean(ve ** 2))) if np.isfinite(ve).any() else None
    infl = [r.get("person_infl", 0.0) for r in rows]
    out["mean_infl_m"] = float(np.mean(infl)); out["max_infl_m"] = float(np.max(infl))
    out["unseen_cap_frac"] = float(np.mean([r.get("unseen_capped", False) for r in rows]))
    out["mean_n_tracks"] = float(np.mean([len(l["tracks"]) for l in log]))
    return out


# ------------------------------------------------------------------------------------------------------------- CLI
def _job(a):
    cond, seed, max_s, out, walk, yield_m = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    try:
        summ, _ = run_episode_pt(cond, seed, max_s=max_s, walk=walk, yield_m=yield_m)
    except Exception as e:                                   # keep the sweep going; record the failure
        import traceback
        summ = dict(cond=cond, seed=seed, error=repr(e), tb=traceback.format_exc())
    fn = Path(out) / f"{cond.replace(':', '_').replace('+', '_').replace('@', '_')}_s{seed}.json"
    fn.write_text(json.dumps(summ, indent=1, default=str))
    print(f"done {cond} seed {seed} in {time.time() - t0:.0f}s contact={summ.get('person_contact')} minpd={summ.get('min_person_distance_m')}", flush=True)
    return summ


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def aggregate(out):
    import glob
    byc = {}
    for f in sorted(glob.glob(str(Path(out) / "*_s*.json"))):
        j = json.loads(Path(f).read_text())
        if "error" in j:
            continue
        byc.setdefault(j["cond"], []).append(j)
    hdr = f"{'cond':28s} {'n':>3s} {'contact':>8s} {'minPD m':>8s} {'interv%':>8s} {'done':>5s} {'t_task':>7s} {'rmse_p':>7s} {'miss%':>6s} {'ttfd s':>7s}"
    print(hdr)
    for c, js in byc.items():
        def med(f, default=np.nan):
            v = [f(j) for j in js]
            v = [x for x in v if x is not None]
            return float(np.median(v)) if v else default
        ttfd = [p["time_to_first_detection_s"] for j in js for p in j["persons"].values() if p["time_to_first_detection_s"] is not None]
        miss = [p["missed_tick_rate"] for j in js for p in j["persons"].values() if p["missed_tick_rate"] is not None]
        print(f"{c:28s} {len(js):3d} {sum(j['person_contact'] for j in js):3d}/{len(js):<4d} {min(j['min_person_distance_m'] for j in js):8.3f} "
              f"{100 * np.mean([j['intervention_fraction_gt2pct'] for j in js]):8.1f} {sum(bool(j['task_done']) for j in js):3d}/{len(js)} "
              f"{med(lambda j: j['task_time_s']):7.1f} {med(lambda j: j['rmse_pos_visible_m']):7.3f} {100 * (np.mean(miss) if miss else np.nan):6.1f} "
              f"{(np.median(ttfd) if ttfd else np.nan):7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-s", type=float, default=40.0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seeds", default="0-8")
    ap.add_argument("--conds", default="truth,seg,depth,param:0.05:3:0.0,param:0.15:3:0.1,param:0.30:6:0.3")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--yield-m", type=float, default=None, help="walker yield distance [m] (default 0.15 = stock; 0 = oblivious walker)")
    ap.add_argument("--human-mesh", default=None, choices=[None, "articulated"], help="visual Rocketbox humans on the scripted people (single run only; default: capsules)")
    ap.add_argument("--no-walk", action="store_true", help="use the stock TeleopSim (the p_walker clock never starts: the walker stands still)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.aggregate:
        aggregate(out)
        return
    if a.sweep:
        import multiprocessing as mp
        jobs = [(c, s, a.max_s, str(out), not a.no_walk, a.yield_m) for s in parse_seeds(a.seeds) for c in a.conds.split(",")]
        with mp.Pool(a.workers, maxtasksperchild=3) as pool:
            for _ in pool.imap_unordered(_job, jobs):
                pass
        aggregate(out)
        return
    vp = out / f"pt_{a.cond.replace(':', '_')}_s{a.seed}.mp4" if a.video else None
    summ, rows = run_episode_pt(a.cond, a.seed, max_s=a.max_s, video=vp, log_every=25, verbose=True, walk=not a.no_walk, yield_m=a.yield_m, human_mesh=a.human_mesh)
    fn = out / f"{a.cond.replace(':', '_').replace('+', '_').replace('@', '_')}_s{a.seed}.json"
    fn.write_text(json.dumps(summ, indent=1, default=str))
    keys = ["cond", "seed", "aborted_stuck", "task_done", "task_time_s", "person_contact", "person_contact_ticks", "min_person_distance_m", "intervention_fraction_gt2pct",
            "wall_collision_ticks", "min_wall_clearance_m", "spills_beads_out", "rmse_pos_visible_m", "rmse_vel_visible_ms", "false_track_ticks",
            "mean_infl_m", "max_infl_m", "unseen_cap_frac", "wall_s"]
    print(json.dumps({k: summ.get(k) for k in keys}, indent=1, default=str))
    print(json.dumps(summ["persons"], indent=1))


if __name__ == "__main__":
    main()
