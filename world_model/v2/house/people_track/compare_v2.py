"""people_track/compare_v2.py -- side-by-side detector evaluation on ONE recorded trajectory.  SIMULATION ONLY.

h5 `carry` episode (walking person, SloppyOperator, TRUTH shield so the trajectory does not depend on the detector under test; pose_source="est" so
the localizer runs), a sensor frame every 1/30 s.  Every detector sees the SAME frame and feeds its OWN MultiTracker:
    base         detectors.DepthDetector  (PRIVILEGED: true-model background, true camera pose)           -- the shortcut being fixed
    perc         PerceivedDepthDetector   (perceived-map background, ESTIMATED pose, pose-uncertainty margin)
    perc_nomargin   same, margin = bg_m only (shows what the margin buys)
    perc_zhi     same with the ceiling opened to 2.1 m (the map has no data above 1.15 m)
    neural       NeuralDetector (RGB -> detect_server.py -> box -> median depth -> axis; map-free; estimated-pose world frame)
    neural_ray   same, box-centre ray at the median depth only
Metrics (truth = sim.people_state, evaluation only): sensor-seen recall (>= 6 person depth pixels), false detections (no person within 0.75 m),
false detections near walls (true wall EDT < 0.5 m), 3-D axis error of matched detections (a) in the TRUE world frame after undoing the pose
estimate (= detector error alone) and (b) as reported (estimated frame vs true world = + localization error), tracker RMSE and false tracks.

  python -m world_model.v2.house.people_track.compare_v2 --seeds 0,1 --dets base,perc,neural --out OUT.json [--human articulated] [--video v.mp4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from world_model.v2.house.teleop import operator as OP, shield as SH
from world_model.v2.house.teleop.sim import TICK
from . import camera as CAM
from . import detectors as DET
from . import detect_v2 as D2
from .sim_fix import TeleopSimWalking
from .tracker import MultiTracker, TrackerCfg

MATCH_M = 0.75
_CLIENTS = {}                      # one connection per server socket for the whole process (the server serves one client at a time)
OUT = Path("/data/local/ss5772/molmospaces/percept/people2")


def make_dets(names, sim, cam, sock=None, res=(640, 360), thr=0.5):
    """detector variants (all share one RGB render + one inference per frame):
        base            detectors.DepthDetector (PRIVILEGED true model + true pose)
        perc            PerceivedDepthDetector (perceived-map background, est. pose, pose margin)       perc_nomargin: no margin      perc_g: + perceived 2-D grid filter
        neural          boxes -> box-depth points -> visible-surface centroid + (pi/4) 0.22 + circle fit (the r=0.22 prior)
        neural_sXX      boxes -> box-depth points -> visible-surface centroid + XX cm along the ray (no circle prior; XX = person half-depth)
        neural_ray      ray through the box centre at the median depth + (pi/4) 0.22
        fusion[_sXX]    neural boxes gate the perc_g depth foreground (box AND foreground depth points form the cluster); same centroid options"""
    from world_model.v2.house.loc import mapper as MP
    out = {}
    pmap = None
    src = None

    def pm():
        nonlocal pmap
        if pmap is None:
            pmap = MP.PerceivedMap.load(MP.map_path(sim.scene))
        return pmap

    for n in names:
        if n == "base":
            out[n] = DET.DepthDetector(occ=sim.occ, use_static_map=False, bg=True)
        elif n.startswith("perc"):
            kw = {}
            if n == "perc_nomargin":
                kw["loc_margin"] = False; kw["map_m"] = 0.0
            if n == "perc_g":
                kw["use_static_map"] = True
            out[n] = D2.PerceivedDepthDetector(pm(), **kw)
        elif n.startswith(("neural", "fusion")):
            if src is None:
                src = D2.shared_box_source(sim, cam, sock, res, thr)
            kind, _, tail = n.partition("_")
            kw = {}
            if tail == "ray":
                kw["centroid"] = "ray"
            elif tail.startswith("s"):
                kw["centroid"], kw["r_off"] = "surf", float(tail[1:]) / 100.0
            fus = D2.PerceivedDepthDetector(pm(), use_static_map=True) if kind == "fusion" else None
            out[n] = D2.NeuralDetector(src, fusion=fus, name=n, **kw)
        else:
            raise ValueError(n)
    return out


def run_one(seed, names, sock, max_s=40.0, res=(640, 360), thr=0.5, scene="h5", video=None, human=False, rate_hz=30.0, yield_m=None):
    if human:
        from .human_vis import TeleopSimHuman, HeadCameraHuman
        SimCls, CamCls = TeleopSimHuman, HeadCameraHuman
    else:
        SimCls, CamCls = TeleopSimWalking, CAM.HeadCamera
    sim = SimCls(scene, grasped=True, seed=seed, with_people=True, counter=False, desk=False, pose_source="est", occ_source="perceived")
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    shield = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw)
    shield.docks = [wp["cafe_counter_pick"][:2]]
    cam = CamCls(sim, CAM.head_spec(W=160, H=90))
    vis = DET.Visibility(cam)
    dets = make_dets(names, sim, cam, sock, res, thr)
    trk = {n: MultiTracker(TrackerCfg()) for n in names}
    rec = {n: dict(frames=[], ms=[], trk=[]) for n in names}
    fr_truth = []
    r = rate_hz * TICK
    vframes = []
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        truth = [(xy.copy(), v.copy()) for xy, v in sim.people_state]
        for n in names:
            trk[n].predict_to(k)
        due = int(math.floor((k + 1) * r)) > int(math.floor(k * r))
        bt, be, dy = D2.est_correction(sim)
        if due:
            frame = cam.render(k, seed + 7)
            npix = [int((frame["person"] == i).sum()) for i in range(len(truth))]
            vis_now = [vis.of(sim, xy) for xy, _ in truth]
            fr_truth.append(dict(k=k, cam_o=frame["origin"][:2].copy(), truth=[t[0] for t in truth], npix=npix, vis=[v[1] for v in vis_now], bt=bt, be=be, dy=dy,
                                 rng=[v[2] for v in vis_now]))
            rgb_img = None
            for n in names:
                t0 = time.perf_counter()
                ds = dets[n].detect(frame, cam, sim)
                rec[n]["ms"].append((time.perf_counter() - t0) * 1000)
                extra = dict(dets[n].last) if n.startswith(("neural", "fusion")) else {}
                rec[n]["frames"].append(dict(k=k, dets=[(d.xy.copy(), d.truth) for d in ds], **{kk: extra[kk] for kk in ("srv_ms", "render_ms", "n_boxes") if kk in extra}))
                trk[n].process_frame(ds, k)
                if video and n == "neural" and len(rec[n]["frames"]) % 3 == 0:
                    L = dets[n].last
                    vframes.append(dict(k=k, img=L["img"].copy(), boxes=list(L["boxes"]), scores=list(L["scores"]),
                                        dets=[to_true(d.xy, bt, be, dy) for d in ds], truth=[t[0] for t in truth], rng=[v[2] for v in vis_now],
                                        npix=list(npix), t=sim.t))
        for n in names:
            if not due:
                trk[n].predict_frame_tick(k)
            rec[n]["trk"].append((k, [(t.id, t.x.copy()) for t in trk[n].output(True)], bt, be, dy, [t[0] for t in truth]))
        safe, info = shield.filter(sim, cmd)
        sim.tick(safe)
        if k % 250 == 0:
            print(f'  [seed {seed}] t={sim.t:5.1f} ph={op.phase}', flush=True)
        if op.done:
            break
    return sim, rec, fr_truth, dets, vframes


def to_true(xy, bt, be, dy):
    c, s = math.cos(-dy), math.sin(-dy)
    d = np.asarray(xy) - be
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]) + bt


def metrics(sim, names, rec, fr_truth):
    res = {}
    occ = sim.occ                      # TRUE occupancy EDT (evaluation only)
    npers = len(fr_truth[0]["truth"])
    for n in names:
        fr = rec[n]["frames"]
        nfr = len(fr)
        seen_n = seen_h = vis_n = vis_h = 0
        n_det = n_fp = n_fp_wall = 0
        e_true, e_rep, rng_m, rbias, lerr = [], [], [], [], []
        fp_pos = []
        per_p_seen = [0] * npers; per_p_hit = [0] * npers
        for f, ft in zip(fr, fr_truth):
            bt, be, dy = ft["bt"], ft["be"], ft["dy"]
            dm = [((np.asarray(x) if n == "base" else to_true(x, bt, be, dy)), x, tr) for x, tr in f["dets"]]   # base: true-frame detections
            n_det += len(dm)
            for xt, xe, _ in dm:
                dd = [float(np.hypot(*(xt - p))) for p in ft["truth"]]
                if not dd or min(dd) > MATCH_M:
                    n_fp += 1
                    fp_pos.append(xt.round(2).tolist())
                    if float(occ.lookup(np.array([xt[0]]), np.array([xt[1]]))[0]) < 0.5:
                        n_fp_wall += 1
            for i, p in enumerate(ft["truth"]):
                ds = [(float(np.hypot(*(xt - p))), xt, xe) for xt, xe, _ in dm]
                hit = [d for d in ds if d[0] < MATCH_M]
                if ft["npix"][i] >= 6:
                    seen_n += 1; per_p_seen[i] += 1
                    if hit:
                        seen_h += 1; per_p_hit[i] += 1
                        b = min(hit, key=lambda d: d[0])
                        e_true.append(b[0]); e_rep.append(float(np.hypot(*(b[2] - p)))); rng_m.append(ft["rng"][i])
                        u = (np.asarray(p) - ft["cam_o"]); u = u / (np.linalg.norm(u) + 1e-9)
                        # b[1] = detection mapped to the true frame
                        dv = b[1] - np.asarray(p)
                        rbias.append(float(dv @ u)); lerr.append(float(dv @ np.array([-u[1], u[0]])))
                if ft["vis"][i]:
                    vis_n += 1
                    vis_h += int(bool(hit))
        # tracker
        pe_t, pe_r = [], []
        false_ticks = 0
        ids_ok, ids_all = set(), set()
        for k, tr, bt, be, dy, truth in rec[n]["trk"]:
            for tid, x in tr:
                ids_all.add(tid)
                xt = x[:2] if n == "base" else to_true(x[:2], bt, be, dy)
                dd = [float(np.hypot(*(xt - p))) for p in truth]
                if dd and min(dd) < 1.5:
                    ids_ok.add(tid)
                else:
                    false_ticks += 1
            for i, p in enumerate(truth):
                best = None
                for tid, x in tr:
                    xt = x[:2] if n == "base" else to_true(x[:2], bt, be, dy)
                    d = float(np.hypot(*(xt - p)))
                    if best is None or d < best[0]:
                        best = (d, x)
                if best and best[0] < MATCH_M and 0:
                    pass
                if best and best[0] < MATCH_M:
                    pe_t.append(best[0]); pe_r.append(float(np.hypot(*(best[1][:2] - p))))
        a = lambda v: np.asarray(v, float)
        res[n] = dict(
            n_frames=nfr, sensor_seen_frames=seen_n, recall_seen=seen_h / seen_n if seen_n else None,
            recall_seen_per_person=[(h / s if s else None, s) for h, s in zip(per_p_hit, per_p_seen)],
            recall_visible=vis_h / vis_n if vis_n else None, vis_frames=vis_n,
            n_dets=n_det, n_false_dets=n_fp, fp_per_frame=n_fp / max(nfr, 1), n_false_dets_near_wall=n_fp_wall,
            fp_positions_first=fp_pos[:12],
            centroid_rmse_true_m=float(np.sqrt(np.mean(a(e_true) ** 2))) if e_true else None,
            centroid_med_true_m=float(np.median(e_true)) if e_true else None,
            ray_bias_mean_m=float(np.mean(rbias)) if rbias else None, ray_bias_std_m=float(np.std(rbias)) if rbias else None,
            lateral_rms_m=float(np.sqrt(np.mean(np.square(lerr)))) if lerr else None,
            centroid_rmse_reported_m=float(np.sqrt(np.mean(a(e_rep) ** 2))) if e_rep else None,
            track_rmse_true_m=float(np.sqrt(np.mean(a(pe_t) ** 2))) if pe_t else None,
            track_rmse_reported_m=float(np.sqrt(np.mean(a(pe_r) ** 2))) if pe_r else None, n_track_matches=len(pe_t),
            false_track_ticks=false_ticks, n_false_track_ids=len(ids_all - ids_ok), n_track_ids=len(ids_all),
            det_ms_mean=float(np.mean(rec[n]["ms"])) if rec[n]["ms"] else None, det_ms_p95=float(np.percentile(rec[n]["ms"], 95)) if rec[n]["ms"] else None)
        if n.startswith(("neural", "fusion")):
            res[n]["srv_ms_mean"] = float(np.mean([f["srv_ms"] for f in fr])); res[n]["render_ms_mean"] = float(np.mean([f["render_ms"] for f in fr]))
    if sim.loc is not None:
        res["_loc"] = {k: v for k, v in sim.loc.errors().items() if k in ("ate_rmse_m", "ate_max_m", "yaw_err_rms_rad", "yaw_err_max_rad", "dist_m")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--dets", default="base,perc,neural")
    ap.add_argument("--sock", default=os.environ.get("DETECT_SOCK", ""))
    ap.add_argument("--res", default="640x360")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max-s", type=float, default=40.0)
    ap.add_argument("--out", default=str(OUT / "compare.json"))
    ap.add_argument("--video", default=None)
    ap.add_argument("--human", action="store_true", help="articulated Rocketbox visual humans (human_vis.TeleopSimHuman)")
    a = ap.parse_args()
    names = a.dets.split(",")
    res = tuple(int(v) for v in a.res.split("x"))
    allr = {}
    for s in [int(v) for v in a.seeds.split(",")]:
        t0 = time.time()
        sim, rec, frt, dets, vf = run_one(s, names, a.sock, a.max_s, res, a.thr, video=a.video if s == int(a.seeds.split(",")[0]) else None, human=a.human)
        allr[f"seed{s}"] = metrics(sim, names, rec, frt)
        print(f"seed {s} done in {time.time() - t0:.0f}s", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(allr, open(a.out, "w"), indent=1, default=float)
        if a.video and vf:
            from .video_v2 import write_video
            write_video(a.video, vf)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(allr, open(a.out, "w"), indent=1, default=float)
    for sd, r in allr.items():
        for n, m in r.items():
            if n.startswith("_"):
                print(sd, n, m); continue
            print(sd, f"{n:14s} recall_seen={m['recall_seen']} fp/frame={m['fp_per_frame']:.3f} fp_wall={m['n_false_dets_near_wall']} cent_true={m['centroid_rmse_true_m']} "
                  f"trkRMSE(true/rep)={m['track_rmse_true_m']}/{m['track_rmse_reported_m']} false_trk_ids={m['n_false_track_ids']}/{m['n_track_ids']} ms={m['det_ms_mean']:.1f}")


if __name__ == "__main__":
    main()
