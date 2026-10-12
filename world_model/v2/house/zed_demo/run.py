"""run.py -- end-to-end SIMULATION demo: ZED stereo -> SGBM -> point cloud -> segmentation -> zonotopes -> the robot places the tray with them -> video.

    cd /common/users/ss5772/bimanual
    PYTHONPATH=. python -m world_model.v2.house.zed_demo.run [--out DIR] [--pitch 30] [--no-video]

Outputs (default /data/local/ss5772/molmospaces/zed_demo): zed_demo.mp4, results.json, frames/*.png (key frames), stereo_left/right/disparity PNGs.
Never imports YOR/robot/base.py, YOR/extra/*, the ZED SDK or ROS.  CPU parts are pinned to 1 thread."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("ZED_DEMO_OUT", "/data/local/ss5772/molmospaces/zed_demo"))
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="tiny video (code-path test)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    os.environ["ZED_DEMO_OUT"] = str(out)
    from world_model.v2.house.zed_demo import rig as Z
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        g = Z.pick_gpu()
        if g is None:
            sys.exit("no free non-zero GPU for EGL")
        os.environ["CUDA_VISIBLE_DEVICES"] = g
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import mujoco
    import imageio
    cv2.setNumThreads(1)
    from world_model.v2.house import phase1 as P1
    from world_model.v2.house.office import scene as S
    from world_model.v2.house.zed_demo import stereo as ST, perceive as PC, plan as PL, execute as EX, viz as VZ, video as V
    t_all = time.time()
    R = {}
    print("[run] building scene + robot + stereo rig ...", flush=True)
    C, gt_clutter = Z.build_composed(pitch_deg=a.pitch, dt=EX.DT)
    m, d = C.m, C.d
    rg = Z.ZedRig(m, seed=a.seed)
    K = rg.K
    cls = P1.geom_class(m)
    keys = PC.gt_keys(m)
    base0 = np.array(C.info["spawn"], float)
    yaw0 = float(C.info["yaw"])
    tray_id = m.body("tray").id
    tray0 = d.xpos[tray_id].copy()
    snap0 = EX.snapshot(d)
    pay = PL.payload_zonotopes(C, cls)
    R["setup"] = dict(baseline_m=Z.ZED_BASELINE, fovy_deg=Z.ZED_FOVY, resolution=[K["W"], K["H"]], f_px=K["f"], head_pos_robot=list(Z.HEAD_POS),
                      head_pitch_deg=a.pitch, base_xy=base0.tolist(), base_yaw=yaw0, n_payload_zonotopes=len(pay),
                      clutter_gt=gt_clutter, hold_pose_note="hold_q from build.set_hold_state IK (seeded)")

    # ---------------------------------------------------------------------------------------- perception (+ one retry if the plan is infeasible)
    sgbm = ST.make_sgbm()
    best = None
    for it in range(a.tries):
        o = rg.capture(d)
        res = PC.run(m, d, rg, o, cls, keys, base0, yaw0, seed=a.seed + it, sgbm=sgbm)
        obst = [(f"Z{i}:{z['gt_key']}", z["Zi"]) for i, z in enumerate(res["zonos"])]
        sup = [z for z in res["zonos"] if z["kind"] == "support"]
        if not sup:
            print(f"[run] try {it}: no support plane found, retry", flush=True)
            continue
        info = PL.plan_task(pay, obst, sup[0]["Z"], tray0[:2])
        print(f"[run] try {it}: {sum(z['kind'] == 'obj' for z in res['zonos'])} clusters, plan feasible cells {info['n_feasible']}, "
              f"perceive {res['timing']['perceive_s']:.1f}s", flush=True)
        best = (o, res, obst, sup[0], info, it)
        if info["n_feasible"] > 0 and info.get("path_t") is not None:
            break
    o, res, obst, sup, info, tries_used = best
    R["perception_tries"] = tries_used + 1
    left0, right0 = o["left"], o["right"]
    origin0, R0 = rg.left_pose(d)
    imageio.imwrite(str(out / "stereo_left.png"), left0)
    imageio.imwrite(str(out / "stereo_right.png"), right0)
    imageio.imwrite(str(out / "stereo_disparity.png"), ST.colormap_disp(res["disp"]))

    # ---------------------------------------------------------------------------------------- repeatability over fresh noisy captures (no sweep: 5 captures)
    rep = []
    for it in range(5):
        o_, r_ = rg.capture(d), None
        r_ = PC.run(m, d, rg, o_, cls, keys, base0, yaw0, seed=100 + it, sgbm=sgbm)
        ob_ = [(f"Z{i}", z["Zi"]) for i, z in enumerate(r_["zonos"])]
        sp_ = [z for z in r_["zonos"] if z["kind"] == "support"]
        i_ = PL.plan_task(pay, ob_, sp_[0]["Z"], tray0[:2]) if sp_ else None
        rc_ = PC.gt_recovery(r_)
        rep.append(dict(n_clusters=sum(z["kind"] == "obj" for z in r_["zonos"]), recovered=f"{sum(v['recovered'] for v in rc_.values())}/{len(rc_)}",
                        fill_scene=float(r_["ev_scene"]["fill_all"]), p95_mm=1000 * r_["ev_scene"]["p95_abs"],
                        spot_offset=None if i_ is None else i_["spot_t"].tolist(), n_feasible=None if i_ is None else i_["n_feasible"]))
    R["repeatability_5_fresh_captures"] = rep
    print("[run] repeatability:", json.dumps(rep), flush=True)

    # ---------------------------------------------------------------------------------------- stereo accuracy vs MuJoCo GT
    rmask = res["rmask"]
    gt_eval = o["gt_depth"].copy()
    gt_eval[rmask] = np.nan
    gt_eval[:, :192] = np.nan
    ev_scene = ST.evaluate(res["depth"], gt_eval)
    ev_full = ST.evaluate(res["depth"], o["gt_depth"])
    fa, fs = V.fill_metrics(res["disp"], o["gt_seg"], cls)
    R["stereo_vs_gt"] = dict(scene_only_active_region=ev_scene, all_pixels_incl_robot=dict(fill_all=ev_full["fill_all"], rmse=ev_full["rmse"], p95_abs=ev_full["p95_abs"]),
                             fill_active_region_all_px=fa, fill_active_region_scene_px=fs, sgbm_s=res["timing"]["sgbm_s"],
                             note="error = stereo z-depth - MuJoCo z-depth (left camera) on pixels where both exist; active region = columns >= 192 (numDisparities); robot body excluded")
    R["timing_s"] = res["timing"]

    # ---------------------------------------------------------------------------------------- segmentation / zonotopes vs GT (eval only)
    rec = PC.gt_recovery(res)
    clutter_keys = ["bottle_a", "mug_desk1", "book_a", "pencup_a", "box_a", "monitor_1", "lamp_1"]
    cont_all = PC.gt_containment(m, d, keys, res["zonos"], clutter_keys)
    cont_task = PC.gt_containment(m, d, keys, res["zonos"], clutter_keys, zmax=1.05)
    zl = []
    for i, z in enumerate(res["zonos"]):
        lo, hi = z["Zi"].interval_hull()
        zl.append(dict(id=i, kind=z["kind"], n_pts=z["n_pts"], gt_key=z["gt_key"], purity=z["purity"], n_gen=int(z["Z"].n_gen), delta_m=z["delta"],
                       range_m=z["range_m"], contains_segment_pts=z["contains_pts"], aabb_lo=lo.tolist(), aabb_hi=hi.tolist()))
    n_obj_clusters = sum(z["kind"] == "obj" for z in res["zonos"])
    n_rec = sum(v["recovered"] for v in rec.values())
    nz_big = [z for z in res["zonos"] if z["kind"] == "obj" and z["n_pts"] >= 250]
    contain_vals = [v["contained"] for v in cont_task.values() if v.get("contained") is not None]
    R["segmentation"] = dict(n_clusters=n_obj_clusters, n_zonotopes_displayed=len(nz_big), gt_objects_eval=rec, n_gt_objects=len(rec), n_recovered_iou05=n_rec,
                             min_zonotope_segment_containment=min(z["contains_pts"] for z in res["zonos"]))
    R["zonotope_gt_containment"] = dict(all_z=cont_all, task_slab_z_le_1p05=cont_task, mean_task_slab=float(np.mean(contain_vals)),
                                        note="fraction of GT collision-geom sample points (corners/rings/mesh verts) inside the INFLATED zonotope(s) matched to that object; unseen sides are covered only by the assumed extrusions")
    R["zonotopes"] = zl

    # ---------------------------------------------------------------------------------------- task: plan, exact check, counterfactual
    spot_t = info["spot_t"]
    path_t = info["path_t"]
    ex_c = [PL.exact_clearance(pay, obst, t) for t in path_t[::max(len(path_t) // 25, 1)]]
    t_naive = np.array([0.0, 0.405])                                        # tray centre -> (4.2, 5.36): desk centre line, 0.2 m in from the front edge
    naive_path_t = np.stack([np.linspace(0, t_naive[0], 60), np.linspace(0, t_naive[1], 60)], 1)
    naive_c_map = float(info["Cm"].at(naive_path_t).min())
    naive_ex = [PL.exact_clearance(pay, obst, t) for t in naive_path_t[::6]]
    R["plan"] = dict(spot_tray_xy=(tray0[:2] + spot_t).tolist(), spot_offset_t=spot_t.tolist(), zonotope_clearance_at_spot_m=info["c_at_spot"],
                     n_feasible_spots=info["n_feasible"], path_len_m=float(np.linalg.norm(np.diff(path_t, axis=0), axis=1).sum()),
                     path_min_exact_zonotope_clearance_m=float(min(x[0] for x in ex_c)), path_limiting_pair=min(ex_c, key=lambda x: x[0])[1],
                     clearance_map_s=info["t_map_s"],
                     naive_target_tray_xy=(tray0[:2] + t_naive).tolist(), naive_map_clearance_min_m=naive_c_map,
                     naive_exact_zonotope_clearance_min_m=float(min(x[0] for x in naive_ex)), naive_limiting_pair=min(naive_ex, key=lambda x: x[0])[1],
                     payload_zonotope_groups=[n for n, _ in pay])
    print("[run] plan:", json.dumps({k: R["plan"][k] for k in ("spot_tray_xy", "zonotope_clearance_at_spot_m", "path_len_m", "path_min_exact_zonotope_clearance_m",
                                                                "naive_map_clearance_min_m", "naive_exact_zonotope_clearance_min_m")}), flush=True)
    traj = EX.make_traj(path_t, base0, yaw0)
    print(f"[run] executing FL-MPC carry ({traj.meta['duration_s']:.1f}s traj) ...", flush=True)
    t0 = time.time()
    ex_sum, frames_exec = EX.run_tracking(C, traj, keys, seed=a.seed, progress=lambda k, n: print(f"   step {k}/{n}", flush=True))
    ex_sum["wall_s"] = time.time() - t0
    # restore the start state and replay the naive straight line kinematically
    EX.restore(m, d, snap0)
    d.qfrc_applied[:] = 0
    naive_path_world = np.asarray(naive_path_t) + base0[None]
    naive_traj = EX.make_traj(naive_path_t, base0, yaw0)
    nv_sum, frames_naive = EX.replay_kinematic(C, naive_traj, keys)
    # exact check also for the CHOSEN path in the same kinematic way (independent of the controller): distances from MuJoCo for the planned traj
    EX.restore(m, d, snap0)
    ex_kin, _ = EX.replay_kinematic(C, traj, keys, dist_every=10)
    R["execution"] = {k: v for k, v in ex_sum.items() if k not in ("trace_t", "trace_d")}
    R["execution"]["kinematic_replay_of_same_path_min_clearance_m"] = ex_kin["min_clearance_m"]
    R["naive_counterfactual"] = {k: v for k, v in nv_sum.items() if k not in ("trace_t", "trace_d")}
    print("[run] exec:", json.dumps({k: R["execution"][k] for k in ("min_clearance_m", "min_clearance_pair", "n_contacts_total", "ball_lost", "tray_tilt_max_deg", "n_infeasible",
                                                                    "final_tray_xy", "ctl_ms_median")}), flush=True)
    print("[run] naive:", json.dumps({k: R["naive_counterfactual"][k] for k in ("min_clearance_m", "min_clearance_pair", "n_contacts_total", "collides",
                                                                              "penetrating_pairs")}), flush=True)
    # ---- independent audit: every collision-geom point of the robot (arms, hands, tray, ball, chassis) vs the INFLATED zonotope union, over the executed frames
    from world_model.v2.perception_zono.zonoutil import ZUnion
    U = ZUnion([z["Zi"] for z in res["zonos"] if z["kind"] == "obj"] + [sup["Zi"]])
    d2 = mujoco.MjData(m)
    own = EX.ClearMon(C, keys).own

    def audit(frames):
        mins = []
        for f in frames[::6]:
            EX.restore(m, d2, f["snap"])
            Pr = np.concatenate([PC.geom_pts(m, d2, g) for g in own])
            mins.append(float(U.outside_depth(Pr).min()))
        return min(mins), mins

    a_ex, _ = audit(frames_exec)
    a_nv, _ = audit(frames_naive)
    R["execution"]["robot_pts_vs_inflated_zonotopes_min_lowerbound_m"] = a_ex
    R["naive_counterfactual"]["robot_pts_vs_inflated_zonotopes_min_lowerbound_m"] = a_nv
    print(f"[run] audit: robot points vs inflated zonotopes  executed >= {a_ex * 1000:.0f} mm   naive {a_nv * 1000:.0f} mm (0 = inside)", flush=True)
    EX.restore(m, d, snap0)
    R["wall_s_before_video"] = time.time() - t_all
    (out / "results.json").write_text(json.dumps(R, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    if a.no_video:
        print("[run] --no-video: done", flush=True)
        sys.stdout.flush()
        os._exit(0)

    # ---------------------------------------------------------------------------------------- video context
    ctx = V.Ctx()
    ctx.quick = a.quick
    ctx.m, ctx.rig, ctx.K, ctx.cls, ctx.res, ctx.sgbm = m, rg, K, cls, res, sgbm
    ctx.dv = mujoco.MjData(m)
    ctx.r3 = mujoco.Renderer(m, V.PH, V.PW, max_geom=60000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (4.1, 5.05, 0.80)
    cam.distance, cam.elevation, cam.azimuth = 2.55, -27.0, 70.0
    ctx.cam3 = cam
    ctx.yaw0, ctx.snap0, ctx.left0, ctx.right0, ctx.origin0, ctx.R0 = yaw0, snap0, left0, right0, origin0, R0
    ctx.z_desk = float(res["z_desk"])
    ctx.cv = VZ.CloudView(target=(4.15, 5.35, 0.85), dist=2.15, elev=24.0)
    ctx.frames_dir = out / "frames"
    ctx.frames_exec, ctx.frames_naive = frames_exec, frames_naive
    ctx.fill_active, ctx.fill_scene = fa, fs
    dv_ok = res["disp"][np.isfinite(res["disp"])]
    ctx.dmin_show, ctx.dmax_show = float(np.percentile(dv_ok, 1)), float(np.percentile(dv_ok, 99))
    ctx.n_clusters, ctx.rec_n, ctx.rec_total = n_obj_clusters, n_rec, len(rec)
    ctx.contain_task = float(np.mean(contain_vals)) if contain_vals else 0.0
    ctx.support_zi = sup["Zi"]
    ctx.plan_c_spot, ctx.naive_c = info["c_at_spot"], naive_c_map
    ctx.path_len = R["plan"]["path_len_m"]
    # display zonotopes (clusters with >= 250 pts), sorted left to right; colours
    nz_big = sorted(nz_big, key=lambda z: z["Z"].c[0])
    ctx.zonos, ctx.cluster_color = [], {}
    ci = 0
    for z in nz_big:
        if z["gt_key"] == "desk_1":
            col = (170, 130, 95)
        else:
            col = VZ.color_of(ci)
            ci += 1
        ctx.zonos.append(dict(z=z, color=col, label=f"Z{len(ctx.zonos) + 1} {z['gt_key'].replace('_a', '').replace('_desk1', '').replace('_1', '')}"))
        ctx.cluster_color[z["k"]] = col
    for z in res["zonos"]:
        if z["kind"] == "obj":
            ctx.cluster_color.setdefault(z["k"], (110, 110, 125))
    # clouds
    depth = res["depth"]
    Pall, pall = ST.depth_to_cloud(depth, K, origin0, R0)
    sel = (Pall[:, 0] > 3.0) & (Pall[:, 0] < 5.4) & (Pall[:, 1] > 4.2) & (Pall[:, 1] < 6.1) & (Pall[:, 2] > -0.05) & (Pall[:, 2] < 1.5)
    Pall, pall = Pall[sel][::2], pall[sel][::2]
    flat = left0.reshape(-1, 3)
    ctx.raw_P, ctx.raw_C = Pall, flat[pall]
    ctx.vox_P = res["P_voxel"]
    rgbv = flat[res["pix_voxel"]]
    ctx.vox_dim_C = (0.35 * rgbv).astype(np.uint8)
    pl = res["plane_lab"]
    colp = rgbv.copy()
    colp_c = np.empty_like(rgbv)
    for i, p in enumerate(res["planes"]):
        colp[pl == i] = (0.45 * np.array((90, 110, 160) if p["kind"] == "horizontal" else (110, 110, 120))).astype(np.uint8)
    ctx.vox_plane_C = colp
    cc = (0.5 * np.array([110, 110, 125])).astype(np.uint8) * np.ones((len(ctx.vox_P), 1), np.uint8)
    cc = colp.copy()
    cc[pl >= 0] = (0.3 * np.array([120, 130, 160])).astype(np.uint8)
    idx_keep = np.flatnonzero(res["keep"])
    for j, l in enumerate(res["lab_obj"]):
        cc[idx_keep[j]] = ctx.cluster_color[int(l)] if l >= 0 else (60, 60, 70)
    ctx.vox_cluster_C = cc
    # plan visuals
    xs, ys = info["xs"], info["ys"]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    msk = info["reach"] & info["on_desk"]
    pts = np.stack([tray0[0] + X[msk], tray0[1] + Y[msk], np.full(msk.sum(), ctx.z_desk + 0.012)], 1)
    sc = np.clip(info["Cm"].C[msk] / 0.12, 0, 1)
    colr = cv2.applyColorMap((sc * 255).astype(np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)[:, 0, ::-1]
    ctx.heat = [(pts, colr)]
    ctx.spot_xyz = np.array([tray0[0] + spot_t[0], tray0[1] + spot_t[1], 0.85])
    ctx.spot_xyz_yaw = ctx.spot_xyz.copy()
    ctx.naive_spot_yaw = np.array([tray0[0] + t_naive[0], tray0[1] + t_naive[1], 0.85])
    ctx.path_world = np.column_stack([tray0[0] + path_t[:, 0], tray0[1] + path_t[:, 1], np.full(len(path_t), 0.86)])
    ctx.naive_world = np.column_stack([tray0[0] + naive_path_t[:, 0], tray0[1] + naive_path_t[:, 1], np.full(len(naive_path_t), 0.86)])
    ctx.end_card_fn = lambda th: _end_card(ctx, R, left0, res, th)
    print("[run] rendering video ...", flush=True)
    n = V.render_video(ctx, out / ("zed_demo_quick.mp4" if a.quick else "zed_demo.mp4"), log=lambda s: print(s, flush=True))
    R["video"] = dict(path=str(out / ("zed_demo_quick.mp4" if a.quick else "zed_demo.mp4")), frames=n, seconds=n / V.FPS, fps=V.FPS, size=[V.W, V.H])
    R["wall_s_total"] = time.time() - t_all
    (out / "results.json").write_text(json.dumps(R, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print(f"[run] done in {time.time() - t_all:.0f}s", flush=True)
    sys.stdout.flush()
    os._exit(0)                                   # skip the noisy EGL destructors at interpreter shutdown


def _end_card(ctx, R, left0, res, thumbs=None):
    from world_model.v2.house.zed_demo import video as V
    sv = R["stereo_vs_gt"]["scene_only_active_region"]
    sg, pl, ex, nv, zc = R["segmentation"], R["plan"], R["execution"], R["naive_counterfactual"], R["zonotope_gt_containment"]
    G = (110, 235, 150)
    Rr = (255, 120, 110)
    lines = [
        ("STEREO (vs MuJoCo ground-truth depth, scene pixels, active region)", (150, 200, 255)),
        f"  fill {100 * R['stereo_vs_gt']['fill_active_region_scene_px']:.0f}%   median |err| {1000 * sv['median_abs']:.1f} mm   p95 {1000 * sv['p95_abs']:.1f} mm   RMSE {1000 * sv['rmse_inliers_lt5cm']:.1f} mm inliers / {1000 * sv['rmse']:.0f} mm all ({100 * sv['outlier_frac_gt5cm']:.1f}% > 5 cm)   SGBM {R['stereo_vs_gt']['sgbm_s']:.2f} s",
        ("PERCEPTION", (150, 200, 255)),
        f"  {sg['n_clusters']} clusters ({sg['n_zonotopes_displayed']} shown);  GT objects recovered {sg['n_recovered_iou05']}/{sg['n_gt_objects']} (IoU>=0.5);  zonotope contains its segment points: {100 * sg['min_zonotope_segment_containment']:.0f}%",
        f"  GT collision geoms inside the inflated zonotopes (task slab z<1.05 m): {100 * zc['mean_task_slab']:.0f}% mean over 7 objects;  perception total {R['timing_s']['total_s']:.1f} s (1 thread)",
        ("TASK: place the tray on the cluttered desk", (150, 200, 255)),
        (f"  chosen spot ({pl['spot_tray_xy'][0]:.2f}, {pl['spot_tray_xy'][1]:.2f}) m;  executed with FL-MPC: min TRUE clearance {1000 * ex['min_clearance_m']:.0f} mm ({ex['min_clearance_pair'].split('|')[-1].replace('_col', '')}),  contacts {ex['n_contacts_total']},  tray tilt max {ex['tray_tilt_max_deg']:.1f} deg,  ball {'kept' if not ex['ball_lost'] else 'lost'}", G),
        (f"  NAIVE straight line to the desk centre: min clearance {1000 * nv['min_clearance_m']:.0f} mm  ->  {'COLLISION' if nv['collides'] else 'no collision'}  ({nv['min_clearance_pair'].split('|')[-1].replace('_col', '')}), contact frames {nv['n_contacts_total']}", Rr if nv["collides"] else G),
        ("ASSUMPTIONS", (150, 200, 255)),
        "  ZED 2 baseline 0.12 m, HD720, fovy 70 deg, rectified by construction; head = proposed mast mount (confirm on hardware); speckle textures added.",
        "  Unseen object sides: 8 cm extrusion along the view ray + shadow volume to the desk plane.  GT used for evaluation and label text only.",
    ]
    return V.card(lines, "Result: stereo -> zonotopes -> safe placement", "ZED stereo (sim) / SGBM / point cloud / DBSCAN / zonotopes / FL-MPC", thumbs,
                  "Simulation only: no hardware, ZED SDK, ROS or network.")


if __name__ == "__main__":
    main()
