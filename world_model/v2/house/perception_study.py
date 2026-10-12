"""perception_study.py -- Phase 2 part B experiments: point-cloud perception inside a MolmoSpaces house.  SIMULATION ONLY (CPU, ray casting).

    python -m world_model.v2.house.perception_study carry   [--house 17]     # static carry pose: every rig, ROI cost, tracker error
    python -m world_model.v2.house.perception_study path    [--house 17]     # base driven through a doorway, with / without the house
    python -m world_model.v2.house.perception_study survey  [--house 17]     # N random base poses, with / without the house (paired)
    python -m world_model.v2.house.perception_study zono    [--workers 6]    # object zonotopes + segmentation in clutter

Observations only.  Outputs: /data/local/ss5772/molmospaces/pb/results/*.json (copies of the small summaries in house/results/).
The ball is placed kinematically on the (welded, moving) tray; no physics step is taken, so the ball is exactly where the study puts it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import mujoco

from world_model.v2.house import assets as A
from world_model.v2.house import build as B
from world_model.v2.house import perception_bridge as PB
from world_model.v2.house import phase1 as P1

OUT = Path("/data/local/ss5772/molmospaces/pb/results")
HERE = Path(__file__).resolve().parent
HOLD = dict(house=17, spawn=(3.55, 10.05), yaw=0.3)       # the Phase 1 pose
T_LOSS = 10


def build_scene(house=17, spawn=None, yaw=None, hold=True):
    spawn = HOLD["spawn"] if spawn is None else spawn
    yaw = HOLD["yaw"] if yaw is None else yaw
    C = P1.new_hold(house, spawn, yaw, 0.005) if hold else B.compose(house, spawn, yaw=yaw, dt=0.005)
    return PB.HouseScene(C)


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=float("nan"), rms=float("nan"), p95=float("nan"))
    return dict(n=int(len(x)), mean=float(x.mean()), rms=float(np.sqrt((x ** 2).mean())), p95=float(np.percentile(x, 95)))


# Feasible ball region on the tray for the Phase-1 hold pose (house 17, lift-independent; measured with mj_geomDistance ball vs robot
# geoms): the left arm reaches over the tray, so for tray-frame x > -0.01 the ball centre is INSIDE arm geometry (to -70 mm) and for
# x < -0.07 it hits the right hand.  Clearance >= 5 mm holds for x in [-0.07, -0.02], |y| <= 0.07.  (My first Phase-2B tracker runs used
# x in +-0.07 and so put the ball inside the left arm for about half of the ticks: superseded by the numbers produced with this path.)
FEAS_X, FEAS_Y = (-0.07, -0.02), 0.06


def ball_path(t, speed=0.17, ax=0.020, ay=0.05):
    """prescribed ball motion on the tray [m, tray frame], inside the feasible region; peak speed about `speed` m/s."""
    w = speed / (1.3 * ay)
    return (-0.045 + ax * np.sin(0.7 * w * t * 0.02), ay * np.sin(1.3 * w * t * 0.02 + 0.7))


def run_track(sc, rig, n_ticks, state_fn, rate=30.0, lat=3, mask="cheap", max_rays=None, seed=0, warm=20, eval_los=False, house_hidden=False, focus=False, **trk_kw):
    """state_fn(t) places the base/tray/ball for tick t.  Returns dict of metrics (errors are KF position error vs the true ball, tray frame)."""
    trk = PB.HouseTracker(sc, rig, rate, lat, mask=mask, seed=seed, max_rays=max_rays, eval_los=eval_los, focus=focus, **trk_kw)
    state_fn(0)
    trk.init(0)
    err, lost, caps, err_alive = [], [], 0, []
    for t in range(1, n_ticks):
        state_fn(t)
        trk.step(t)
        if t >= warm:
            e = trk.pipe.kf.x[:2] - sc.ball_in_tray()[:2]
            err.append(float(np.linalg.norm(e)) * 1e3)
            lost.append(trk.pipe.kf.since(t) > T_LOSS)
            if not lost[-1]:
                err_alive.append(err[-1])
    cost = [r for r in trk.cost if r["tick"] >= warm]
    ncam = len(trk.cams)
    out = dict(focus=focus, focused_frac=float(np.mean([r["focused"] for r in cost])) if cost else 0.0, rig=rig, err_mm=stats(err), lost_frac=float(np.mean(lost)) if lost else float("nan"),
               err_alive_mm=stats(err_alive), n_caps=len(cost),
               valid_frac=float(np.mean([r["valid"] for r in cost])) if cost else float("nan"),
               fit_err_mm=stats([np.linalg.norm(r["raw_xy"] - r["truth_xy"]) * 1e3 for r in cost if r["valid"] > 0.5]),
               ball_hits_med=[float(np.median([r["n_ball_hits"][i] for r in cost])) if cost else 0.0 for i in range(ncam)],
               inliers_med=float(np.median([r["n_inl"] for r in cost])) if cost else 0.0,
               rays_med=[float(np.median([r["n_rays"][i] for r in cost])) if cost else 0.0 for i in range(ncam)],
               cam_ms=[stats([r["t_cam"][i] for r in cost]) for i in range(ncam)] if cost else [],
               fit_ms=stats([r["t_fit"] for r in cost]), cams=[c.spec.name for c in trk.cams])
    if eval_los:
        out["los_blocked_frac"] = [float(np.mean([r["los"][i][0] for r in cost])) if cost else float("nan") for i in range(ncam)]
        out["los_house_blocked_frac"] = [float(np.mean([r["los_h"][i][0] for r in cost])) if cost else float("nan") for i in range(ncam)]
        hc = {}
        for r in cost:
            for i, (b, c) in enumerate(r["los_h"]):
                if b:
                    hc.setdefault(trk.cams[i].spec.name, {}).setdefault(c, 0)
                    hc[trk.cams[i].spec.name][c] += 1
        out["los_house_blockers"] = hc
        cats = {}
        for r in cost:
            for i, (b, c) in enumerate(r["los"]):
                if b:
                    cats.setdefault(trk.cams[i].spec.name, {}).setdefault(c, 0)
                    cats[trk.cams[i].spec.name][c] += 1
        out["los_blockers"] = cats
    return out, trk


# ------------------------------------------------------------------------------------------------------------- carry
def cmd_carry(a):
    sc = build_scene(a.house)
    base0 = (np.array(sc.C.info["spawn"]), sc.C.info["yaw"])
    res = []
    balls = {"centre": (0, 0), "x+": (0.09, 0.0), "x-": (-0.09, 0.0), "y+": (0.0, 0.05), "y-": (0.0, -0.05)}
    rigs = ["head", "head_aim", "wrist_l", "wrist_r", "wrist_both", "head+wrist", "head_aim+wrist"]
    modes = [(None, False), (a.max_rays, False), (a.max_rays, True)] if a.modes == "all" else [(a.max_rays, True)]
    for rig in rigs:
        for (rate, lat) in ((30.0, 3),) if a.modes != "all" else ((50.0, 0), (30.0, 3)):
            for mr, fc in modes:
                if (mr is not None or fc) and rig not in ("head_aim+wrist", "head_aim", "head+wrist", "wrist_r", "head"):
                    continue
                static, moving = [], []
                for nm, bxy in balls.items():
                    f = lambda t, bxy=bxy: sc.place_base(base0[0], base0[1], bxy)
                    r, _ = run_track(sc, rig, 60, f, rate, lat, max_rays=mr, seed=1, focus=fc)
                    r["ball"] = nm
                    static.append(r)
                f = lambda t: sc.place_base(base0[0], base0[1], ball_path(t))
                r, _ = run_track(sc, rig, 200, f, rate, lat, max_rays=mr, seed=2, warm=25, focus=fc, eval_los=True)
                moving.append(r)
                res.append(dict(rig=rig, rate=rate, lat=lat, max_rays=mr, focus=fc, static=static, moving=moving))
                print(rig, rate, lat, mr, fc, "static err mean(avg over 5 balls) %.1f mm | moving err mean/rms/p95 %.1f/%.1f/%.1f valid %.2f cam_ms %s" % (
                    np.nanmean([x["err_mm"]["mean"] for x in static]), moving[0]["err_mm"]["mean"], moving[0]["err_mm"]["rms"],
                    moving[0]["err_mm"]["p95"], moving[0]["valid_frac"], [round(c["mean"], 1) for c in moving[0]["cam_ms"]]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"carry_{a.modes}_h{a.house}.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------------- doorway path
def door_line(sc, k=0):
    """(centre xy, unit normal n (xy), width) of the k-th ProcTHOR doorway, from the metadata body position and the wall0 direction."""
    idx = sc.idx
    sj = {dr["id"]: dr for dr in A.spec_json(idx)["doors"]}
    out = []
    for key, v in sc.md.items():
        if v["category"] not in ("Doorway", "Doorframe") or v["object_id"] not in sj:
            continue
        dr = sj[v["object_id"]]
        mm = re.match(r"wall\|\d+\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)", dr["wall0"])
        x0, z0, x1, z1 = (float(g) for g in mm.groups())
        wd = np.array([x1 - x0, z1 - z0])
        wd /= np.linalg.norm(wd)
        root = sc.m.body(next(iter(v["name_map"]["bodies"]))).id
        c = sc.d.xpos[root][:2].copy()
        h = dr["holePolygon"]
        out.append((c, np.array([-wd[1], wd[0]]), abs(h[1]["x"] - h[0]["x"]), v["object_id"]))
    return out[k]


def cmd_path(a):
    sc = build_scene(a.house)
    mujoco.mj_forward(sc.m, sc.d)
    c, n, width, did = door_line(sc, a.door)
    t_ = np.array([-n[1], n[0]])
    c = c + t_ * a.lateral                      # lateral offset along the wall: 0 = through the opening, else into the wall
    yaw = float(np.arctan2(n[1], n[0]))
    if a.flip:
        n, yaw = -n, yaw + np.pi
    s0, s1 = -a.span, a.span
    speed = 0.5
    n_ticks = int((s1 - s0) / speed / 0.02)
    print("door", did, "centre", c, "normal", n, "width", width, "ticks", n_ticks, flush=True)
    res = dict(house=a.house, door=did, centre=c.tolist(), normal=n.tolist(), width=width, speed=speed, conds={})
    pen = []
    for cond in ("house", "no_house"):
        for rig in ("head_aim+wrist", "head+wrist"):
            def f(t):
                s = s0 + speed * 0.02 * t
                sc.place_base(c + n * s, yaw, ball_path(t))
            ctx = sc.hide("house") if cond == "no_house" else sc.hide("doors")
            with ctx:
                if cond == "no_house":
                    pass
                r, trk = run_track(sc, rig, n_ticks, f, 30.0, 3, seed=3, warm=10, eval_los=True, max_rays=a.max_rays, focus=a.focus)
            # per-capture records with the along-path coordinate
            rows = []
            for rec in trk.cost:
                s = s0 + speed * 0.02 * rec["tick"]
                rows.append(dict(s=s, valid=rec["valid"], ball_hits=rec["n_ball_hits"], los=[l[1] if l[0] else "" for l in rec["los"]],
                                 los_h=[l[1] if l[0] else "" for l in rec["los_h"]],
                                 t_cam=rec["t_cam"], fit_err=float(np.linalg.norm(rec["raw_xy"] - rec["truth_xy"]) * 1e3) if rec["valid"] > .5 else None))
            r["captures"] = rows
            res["conds"][f"{cond}|{rig}"] = r
            print(cond, rig, "err mean/rms/p95 %.1f/%.1f/%.1f valid %.2f lost %.2f blocked %s" % (
                r["err_mm"]["mean"], r["err_mm"]["rms"], r["err_mm"]["p95"], r["valid_frac"], r["lost_frac"], r["los_blocked_frac"]), flush=True)
    # footprint penetration along the path (house present)
    with sc.hide("doors"):
        for s in np.arange(s0, s1 + 1e-9, 0.1):
            sc.place_base(c + n * s, yaw)
            pen.append((float(s), sc.house_contacts()))
    res["penetrations"] = pen
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"path_h{a.house}_d{a.door}{'f' if a.flip else ''}{'_lat%.1f' % a.lateral if a.lateral else ''}{a.tag}.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------------- survey
def cmd_survey(a):
    sc = build_scene(a.house)
    rng = np.random.default_rng(a.seed)
    polys = B.room_polys(a.house)
    poses = []
    tries = 0
    while len(poses) < a.n and tries < 4000:
        tries += 1
        rid, rt, poly = polys[rng.integers(len(polys))]
        P = np.array(poly)
        xy = rng.uniform(P.min(0), P.max(0))
        if not B._in_poly(xy, poly):
            continue
        yaw = float(rng.uniform(-np.pi, np.pi))
        with sc.hide("doors"):
            sc.place_base(xy, yaw)
            if sc.house_contacts() > 0:
                continue
            # skip poses where a camera origin is inside solid house geometry (cheap test on the robot-frame head position)
        poses.append((xy, yaw, rt))
    print("poses", len(poses), "of", tries, "tries", flush=True)
    rows = []
    for i, (xy, yaw, rt) in enumerate(poses):
        row = dict(i=i, xy=xy.tolist(), yaw=yaw, room=rt)
        for cond in ("house", "no_house"):
            ctx = sc.hide("house") if cond == "no_house" else sc.hide("doors")
            with ctx:
                for rig in ("head_aim+wrist", "head_aim", "head+wrist"):
                    r, _ = run_track(sc, rig, 70, lambda t: sc.place_base(xy, yaw, ball_path(t)), 30.0, 3, seed=4 + i, warm=25,
                                     eval_los=(cond == "house"), max_rays=a.max_rays, focus=a.focus)
                    r.pop("cam_ms", None)
                    row[f"{cond}|{rig}"] = r
        rows.append(row)
        h, n_ = row["house|head_aim+wrist"], row["no_house|head_aim+wrist"]
        print(i, "%.1f,%.1f %s" % (xy[0], xy[1], rt), "fused err house %.1f no_house %.1f | head_aim blocked %.2f" % (
            h["err_mm"]["mean"], n_["err_mm"]["mean"], row["house|head_aim"]["los_blocked_frac"][0]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"survey_h{a.house}{a.tag}.json").write_text(json.dumps(rows, indent=1))


# ------------------------------------------------------------------------------------------------------------- zonotopes in clutter
BIG = {'Chair', 'Sofa', 'Bed', 'DiningTable', 'Desk', 'CounterTop', 'Fridge', 'ShelvingUnit', 'Dresser', 'TVStand', 'Doorway', 'Window',
       'Painting', 'Wall', 'Floor', 'Television', 'SideTable', 'CoffeeTable', 'Cabinet', 'Shelf', 'Bathtub', 'Toilet', 'Sink', 'Stool',
       'ArmChair', 'Ottoman', 'GarbageCan', 'HousePlant', 'Doorframe'}
ZRIGS = ["head_aim", "head_aim+wrist", "wrist_both", "head_hi"]
ZCONDS = ["nominal", "n3mm"]
ZMETH = ["ubox", "hz2", "split_pa_fs"]


def find_clusters(sc, want=2, min_ext=0.03):
    """clusters of 3-5 small annotated objects (metadata) within 0.30 m of each other on one surface, each with a g4 hull and an extent
    >= min_ext.  -> list of lists of metadata keys."""
    m, d = sc.m, sc.d
    mujoco.mj_forward(m, d)
    cand = []
    for k, v in sc.md.items():
        if v["category"] in BIG:
            continue
        bs = PB.object_bodies(sc, k)
        gs = PB.object_geoms(sc, k)
        if not bs or not gs:
            continue
        p = d.xpos[bs[0]].copy()
        try:
            lo, hi = PB.MeshObj(sc, k, n_boundary=200).aabb()
        except Exception:
            continue
        if (hi - lo).max() < min_ext or not (0.55 < lo[2] < 1.3):
            continue
        cand.append((k, v["category"], (lo + hi) / 2, lo, hi))
    used, out = set(), []
    for i, (k, c, p, lo, hi) in enumerate(cand):
        if i in used:
            continue
        mem = [j for j, x in enumerate(cand) if j not in used and np.linalg.norm(x[2][:2] - p[:2]) < 0.30 and abs(x[3][2] - lo[2]) < 0.10]
        if len(mem) >= 3:
            mem = mem[:5]
            for j in mem:
                used.add(j)
            out.append([cand[j][0] for j in mem])
    out.sort(key=lambda c: -len(c))
    return out[:want]


def zono_trial(args):
    """one (cluster, yaw, rig, cond, seed): capture, preprocess, segment (gt / dbscan_xyz / dbscan_xyz_plane), fit, score."""
    from world_model.v2.perception_zono import segment as SG, methods as MT
    from world_model.v2.perception_zono.score import Truth
    sc, keys, objs, Ts, yaw, rig, cond, seed = args[:8]
    cams = args[8] if len(args) > 8 else None           # robot-mounted HouseCameras (lift study); None = the transplanted tabletop rig
    rng = np.random.default_rng(seed)
    lo = np.min([o.aabb()[0] for o in objs], 0)
    hi = np.max([o.aabb()[1] for o in objs], 0)
    ctr = (lo + hi) / 2
    z0 = float(lo[2])
    rho = float(np.linalg.norm(hi - lo) / 2) + 0.05
    specs = cams if cams is not None else PB.rig_in_frame(rig, np.array([ctr[0], ctr[1], z0]), yaw, cond=cond)
    labels = {}
    for i, k in enumerate(keys):
        for g in PB.object_geoms(sc, k):
            labels[g] = i + 1
    t0 = time.perf_counter()
    cl = PB.capture_cloud(sc, specs, ctr, rho, labels, seed=seed)
    t_cap = (time.perf_counter() - t0) * 1e3
    pp = SG.preprocess(cl)
    gt = pp["label"]
    rec = dict(robot_occ=(float(np.mean([PB.robot_occluded_frac(sc, cams[0], T_.surf[:300]) for T_ in Ts])) if cams is not None else None),
               keys=keys, yaw=float(yaw), rig=rig, cond=cond, seed=int(seed), n_pts=int(len(pp["pts"])), n_rays=int(cl.n_rays),
               t_cap_ms=t_cap, n_obj_pts={str(k): int((gt == i + 1).sum()) for i, k in enumerate(keys)}, n_other=int((gt == 0).sum()), segs={})
    for seg in ("gt", "dbscan_xyz", "dbscan_xyz_plane"):
        if seg == "gt":
            sets = {i + 1: np.flatnonzero(gt == i + 1) for i in range(len(keys))}
            sm = None
        else:
            plane = (np.array([0.0, 0.0, 1.0]), z0 + float(rng.normal(0, 0.002)), 0.003) if seg.endswith("plane") else None
            lab, info = SG.unsup_segments(pp, cl.views, rng, plane=plane)
            sm = SG.seg_metrics(gt, lab)
            sets = SG.match_clusters(gt, lab, list(range(1, len(keys) + 1)))
        r = dict(seg=sm, objs={})
        for i, k in enumerate(keys):
            idx = sets[i + 1]
            if idx is None or len(idx) < 8:
                r["objs"][k] = dict(fail="no_points", n=0 if idx is None else int(len(idx)))
                continue
            P = pp["pts"][idx]
            T = Ts[i]
            T.set_visible(P)
            vis = T.vis
            ro = dict(n=int(len(P)), purity=float((gt[idx] == i + 1).mean()), coverage=float(vis.mean()), methods={})
            for mn in ZMETH:
                U = MT.BASE[mn](P, cl.views)
                inn = U.contains(T.surf)
                ro["methods"][mn] = dict(contain=float(inn.mean()), contain_vis=float(inn[vis].mean()) if vis.any() else float("nan"), oc_hull=float(U.volume_sum() / T.vol_hull),
                                         leak_p95_mm=float(np.percentile(U.outside_depth(T.surf), 95) * 1e3))
            r["objs"][k] = ro
        rec["segs"][seg] = r
    return rec


_W = {}


def _zono_worker_init(house):
    pass


def zono_house(house, out_q=None, yaws=(0, 90, 180, 270), seeds=2, want=2):
    from world_model.v2.perception_zono.score import Truth
    t0 = time.time()
    spots = None
    m0, d0 = B.load_house(house, True)
    mujoco.mj_forward(m0, d0)
    spot = B.find_clear_spots(house, m0, d0, top=1)[0]
    del m0, d0
    C = B.compose(house, (spot["x"], spot["y"]), yaw=0.0, dt=0.005)
    sc = PB.HouseScene(C)
    clusters = find_clusters(sc, want)
    print(f"house {house}: spawn {spot['x']:.1f},{spot['y']:.1f}; clusters {[[sc.md[k]['category'] for k in c] for c in clusters]} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for ci, keys in enumerate(clusters):
        objs = [PB.MeshObj(sc, k, oid=i + 1) for i, k in enumerate(keys)]
        rng = np.random.default_rng(100 + ci)
        from world_model.v2.perception_zono.score import Truth
        Ts = [Truth(o, rng, n_surf=3000, n_q=0, n_mc=60000) for o in objs]
        for yaw in yaws:
            for rig in ZRIGS:
                for cond in ZCONDS:
                    for sd in range(seeds):
                        seed = (house * 1000 + ci * 100 + int(yaw) // 90 * 10 + sd) & 0x7FFFFFFF
                        rec = zono_trial((sc, keys, objs, Ts, np.deg2rad(yaw), rig, cond, seed))
                        rec.update(house=house, cluster=ci, cats=[sc.md[k]["category"] for k in keys])
                        rows.append(rec)
        print(f"house {house} cluster {ci} done {len(rows)} trials ({time.time() - t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"zono_h{house}.json").write_text(json.dumps(rows))
    return rows


def cmd_zono(a):
    zono_house(a.house, yaws=tuple(a.yaws), seeds=a.seeds, want=a.want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["carry", "path", "survey", "zono"])
    ap.add_argument("--yaws", type=float, nargs="*", default=[0, 90, 180, 270])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--want", type=int, default=2)
    ap.add_argument("--house", type=int, default=17)
    ap.add_argument("--door", type=int, default=0)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--span", type=float, default=2.5)
    ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--modes", default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rays", type=int, default=None)
    a = ap.parse_args()
    {"carry": cmd_carry, "path": cmd_path, "survey": cmd_survey, "zono": cmd_zono}[a.cmd](a)


if __name__ == "__main__":
    main()
