"""bench_pose.py -- FoundationPose model-free on the carried tray + cup over a recorded h5 carry (capture.py pose).  mujoco env; model runs in the server process.

    python -m world_model.v2.house.foundation.bench_pose --gpu 6 --depth gt --views 16 --tag gt
Steps: READY (warm-up) -> onboarding of tray and cup (NOF; masks = masks.floor_gate_mask, non-privileged) -> register on frame 0 (first-frame mask from the rendered
segmentation = PRIVILEGED stand-in for a SAM/VLM prompt) -> track every 10 Hz frame.  Errors vs MuJoCo truth (EVALUATION ONLY).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from . import masks as M
from .pose_fp import Client

OUT = Path(os.environ.get("FOUNDATION_OUT", "/data/local/ss5772/molmospaces/percept/foundation"))
FLIP = np.diag([1.0, -1.0, -1.0])


def T_of(p, R):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R).reshape(3, 3)
    T[:3, 3] = p
    return T


def gpu_mem_mb(pid):
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        for ln in o.strip().splitlines():
            a, b = (x.strip() for x in ln.split(","))
            if int(a) == int(pid):
                return int(b)
    except Exception:
        pass
    return None


def rot_angle(R):
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def ang(u, v):
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


ROI_W, ROI_H = 640, 360


def crop_roi(rgb, dep, mask, K, uv, w=None, h=None):
    w, h = w or ROI_W, h or ROI_H
    """crop a w x h window centred on pixel uv (clamped to the image) and shift the principal point: FoundationPose then runs at native pixel scale on a small image
    (its memory grows with the input size: a full 1280x720 frame OOMs the 11 GB card; the upstream demos run <= 640x480)."""
    H, W = rgb.shape[:2]
    u0 = int(np.clip(round(uv[0] - w / 2), 0, W - w))
    v0 = int(np.clip(round(uv[1] - h / 2), 0, H - h))
    K2 = np.array(K, float).copy()
    K2[0, 2] -= u0
    K2[1, 2] -= v0
    return rgb[v0:v0 + h, u0:u0 + w], dep[v0:v0 + h, u0:u0 + w], (None if mask is None else mask[v0:v0 + h, u0:u0 + w]), K2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--depth", default="gt", choices=["gt", "fs"])
    ap.add_argument("--fs-depth-file", default=None, help="npz with 'depth' (N,H,W) float32 NaN=invalid for the sequence frames (made by make_fs_depth.py)")
    ap.add_argument("--views", type=int, default=16)
    ap.add_argument("--n-step", type=int, default=None)
    ap.add_argument("--objs", nargs="*", default=["tray", "cup"])
    ap.add_argument("--seq", default="seq_s0_1280x720.npz")
    ap.add_argument("--max-frames", type=int, default=10 ** 9)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--reg-iters", type=int, default=5)
    ap.add_argument("--mesh-dir", default=None, help="load meshes from here (e.g. a CAD-box mesh in the onboarding frame) instead of the NOF meshes")
    ap.add_argument("--init", default="register", choices=["register", "prior"])
    ap.add_argument("--prior-mm", type=float, default=20.0)
    ap.add_argument("--prior-deg", type=float, default=5.0)
    ap.add_argument("--up-prior", type=int, default=1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--mesh-cache", action="store_true", help="reuse meshes saved by an earlier run instead of re-running the NOF")
    ap.add_argument("--roi", default="640x360")
    a = ap.parse_args()
    global ROI_W, ROI_H
    ROI_W, ROI_H = (int(v) for v in a.roi.split("x"))
    seq = dict(np.load(OUT / "pose" / a.seq))      # materialise once (NpzFile re-reads the whole array on every access)
    N = min(len(seq["left"]), a.max_frames)
    H, W = seq["left"].shape[1:3]
    f = float(seq["f"])
    Kseq = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
    res = dict(args=vars(a))
    t0 = time.time()
    cl = Client(sizes=[f"{ROI_W}x{ROI_H}"], gpu=a.gpu)
    warm = cl.warmup()
    res["ready_s"] = time.time() - t0
    res["warmup"] = warm
    print(f"READY {res['ready_s']:.1f}s {json.dumps(warm)[:300]}", flush=True)
    truth_depth = seq["gt_depth"]
    if a.depth == "fs":
        truth_depth = np.load(a.fs_depth_file)["depth"]
    info = {}
    meshdir = OUT / "onboard"
    for obj in a.objs:
        z = dict(np.load(OUT / "onboard" / f"{obj}.npz"))
        K = z["K"]
        sel = np.linspace(0, len(z["rgb"]) - 1, a.views).astype(int)
        hmin, rad = (0.003, 0.25) if obj == "tray" else (0.003, 0.12)
        seed = z["seed_xy"]
        shift = np.eye(4)
        shift[:2, 3] = seed
        rgbs, deps, mks, poses, ious = [], [], [], [], []
        for i in sel:
            mk = M.floor_gate_mask(z["depth"][i], K, z["cam_in_world"][i], seed, float(z["floor_z"]), hmin, 0.4, rad)
            ious.append(M.iou(mk, np.isin(z["seg"][i], z["obj_geoms"])))          # EVALUATION ONLY
            rgbs.append(z["rgb"][i])
            deps.append(z["depth"][i])
            mks.append(mk)
            poses.append(np.linalg.inv(shift) @ z["cam_in_world"][i])
        mp = Path(a.mesh_dir) / f"{obj}_mesh.obj" if a.mesh_dir else meshdir / f"{obj}_mesh.obj"
        if a.mesh_cache and mp.exists():
            t1 = time.perf_counter()
            r = cl.load_mesh(obj, mp)
            r["onboard_s"] = time.perf_counter() - t1
            r["cached"] = True
        else:
            r = cl.onboard(obj, rgbs, deps, mks, poses, K, n_step=a.n_step, save_mesh=str(mp))
        r["mask_iou_mean"], r["mask_iou_min"] = float(np.mean(ious)), float(np.min(ious))
        r["n_views"] = len(sel)
        info[obj] = r
        print(f"[onboard {obj}] {r['onboard_s']:.0f}s extents {np.round(r['extents'], 4)} mask IoU {np.mean(ious):.3f}", flush=True)
    res["onboard"] = info
    res["gpu_mem_mb_after_onboard"] = gpu_mem_mb(cl.server_pid)

    # ------------------------------------------------------------------ sequence
    per = {}
    for obj in a.objs:
        z = dict(np.load(OUT / "onboard" / f"{obj}.npz"))
        seed = z["seed_xy"]
        shift = np.eye(4)
        shift[:2, 3] = seed
        T0 = z["obj_world"]                                             # EVAL ONLY: true body pose at scan time
        Tw_mesh0 = np.linalg.inv(T0) @ shift                            # world<-mesh = T_t @ inv(T0) @ shift
        cm = np.mean(np.array(info[obj]["bounds"]), 0)                  # object centre in mesh coordinates
        P, Rr = (seq["tray_p"], seq["tray_R"]) if obj == "tray" else (seq["cup_p"], seq["cup_R"])
        geoms = seq["tray_geoms"] if obj == "tray" else seq["cup_geoms"]
        rec = dict(ms_score=[], t_err=[], r_err=[], axis_err=[], tilt_err=[], score=[], ms=[], vis=[], tracked_ms=[], reg=None)
        # first frame
        for fr in range(N):
            Tcw_mj = T_of(seq["origin"][fr], seq["R"][fr])
            Tcw = Tcw_mj.copy()
            Tcw[:3, :3] = Tcw_mj[:3, :3] @ FLIP                          # OpenCV camera
            seg = seq["gt_seg"][fr]
            m_obj = np.isin(seg, geoms)
            rec["vis"].append(int(m_obj.sum()))
            rgb, dep = seq["left"][fr], truth_depth[fr]
            if fr == 0 and a.init == "prior":
                # FK-prior emulation: truth perturbed by `prior_mm` / `prior_deg` (what arm forward kinematics gives, ~cm / few deg); no mask, no global search
                T_t = T_of(P[fr], Rr[fr])
                Tc_true0 = np.linalg.inv(Tcw) @ T_t @ Tw_mesh0
                rng = np.random.default_rng(3)
                ax = rng.normal(size=3)
                ax /= np.linalg.norm(ax)
                th = np.radians(a.prior_deg)
                Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                Rp = np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx
                prior = Tc_true0.copy()
                prior[:3, :3] = Rp @ prior[:3, :3]
                prior[:3, 3] += rng.normal(size=3) / np.sqrt(3) * a.prior_mm * 1e-3
                cc = Tc_true0[:3, :3] @ cm + Tc_true0[:3, 3]
                uv = (Kseq[0, 0] * cc[0] / cc[2] + Kseq[0, 2], Kseq[1, 1] * cc[1] / cc[2] + Kseq[1, 2])
                r_, d_, _, K_ = crop_roi(rgb, dep, None, Kseq, uv)
                pose, sc, ms = cl.init_from_prior(obj, r_, d_, K_, prior, iters=a.reg_iters)
                rec["reg"] = dict(ms=ms, score=sc, server=cl.last_info, prior_mm=a.prior_mm, prior_deg=a.prior_deg)
            elif fr == 0:
                ys, xs = np.nonzero(m_obj)
                r_, d_, m_, K_ = crop_roi(rgb, dep, m_obj, Kseq, (xs.mean(), ys.mean()))
                up_cam = np.linalg.inv(Tcw)[:3, :3] @ np.array([0.0, 0.0, 1.0])      # gravity-up in the camera frame (IMU + FK; not truth of the object)
                pose, sc, ms = cl.register(obj, r_, d_, m_, K_, iters=a.reg_iters, up_cam=(up_cam if a.up_prior else None))
                rec["reg"] = dict(ms=ms, score=sc, server=cl.last_info)
                rec["track_after_reg_ms"] = None
            else:
                want = (fr % 5 == 0)
                cc = pose[:3, :3] @ cm + pose[:3, 3]                         # previous estimate -> ROI centre (no truth)
                uv = (Kseq[0, 0] * cc[0] / max(cc[2], 0.1) + Kseq[0, 2], Kseq[1, 1] * cc[1] / max(cc[2], 0.1) + Kseq[1, 2])
                r_, d_, _, K_ = crop_roi(rgb, dep, None, Kseq, uv)
                pose, sc, ms = cl.track(obj, r_, d_, K_, iters=a.iters, score=want)
                (rec["ms_score"] if want else rec["ms"]).append(ms)
                if not want:
                    rec["tracked_ms"].append(cl.last_info["infer"])
            # truth in the mesh frame (see module doc): T_cam_mesh_true = inv(Tcw) @ T_t @ inv(T0) @ shift
            T_t = T_of(P[fr], Rr[fr])
            Tc_true = np.linalg.inv(Tcw) @ T_t @ Tw_mesh0
            ce, ct = (pose[:3, :3] @ cm + pose[:3, 3]), (Tc_true[:3, :3] @ cm + Tc_true[:3, 3])
            Rrel = pose[:3, :3].T @ Tc_true[:3, :3]
            rec["t_err"].append(float(np.linalg.norm(ce - ct)))
            rec["r_err"].append(rot_angle(Rrel))
            rec["axis_err"].append(ang(pose[:3, 2], Tc_true[:3, 2]))     # symmetric-invariant: the object's up axis (z of the scan frame)
            zw_est = Tcw[:3, :3] @ pose[:3, 2]
            zw_true = Tcw[:3, :3] @ Tc_true[:3, 2]
            rec["tilt_err"].append(abs(ang(zw_est, [0, 0, 1]) - ang(zw_true, [0, 0, 1])))   # tilt from world vertical
            rec["score"].append(float(sc))
        te, ae, re_, ti = (np.array(rec[k]) for k in ("t_err", "axis_err", "r_err", "tilt_err"))
        bad = (te > 0.03) | (ae > 15)
        first_bad = int(np.argmax(bad)) if bad.any() else None
        tm = np.array(rec["ms"])
        per[obj] = dict(
            frames=N, vis_px_median=float(np.median(rec["vis"])), vis_px_min=int(np.min(rec["vis"])), reg=rec["reg"],
            t_err_mm=dict(median=1e3 * float(np.median(te)), p95=1e3 * float(np.percentile(te, 95)), max=1e3 * float(te.max())),
            rot_err_deg=dict(median=float(np.median(re_)), p95=float(np.percentile(re_, 95))),
            axis_err_deg=dict(median=float(np.median(ae)), p95=float(np.percentile(ae, 95)), max=float(ae.max())),
            tilt_err_deg=dict(median=float(np.median(ti)), p95=float(np.percentile(ti, 95)), max=float(ti.max())),
            fail_frac=float(bad.mean()), first_fail_frame=first_bad,
            track_ms=dict(mean=float(tm.mean()), median=float(np.median(tm)), p95=float(np.percentile(tm, 95)), max=float(tm.max())),
            track_server_ms_median=float(np.median(rec["tracked_ms"])), track_score_ms_median=float(np.median(rec["ms_score"])),
            score_median=float(np.nanmedian(rec["score"][1:])), score_min=float(np.nanmin(rec["score"][1:])))
        per[obj]["series"] = dict(t_err=te.tolist(), axis_err=ae.tolist(), score=rec["score"], vis=rec["vis"])
        print(f"[{obj}] t_err med {per[obj]['t_err_mm']['median']:.1f} mm p95 {per[obj]['t_err_mm']['p95']:.1f}; axis err med {np.median(ae):.1f} p95 {np.percentile(ae, 95):.1f} deg; "
              f"fail {bad.mean() * 100:.0f}% first {first_bad}; track {np.median(tm):.0f} ms (server {np.median(rec['tracked_ms']):.0f})", flush=True)
    res["per_obj"] = per
    # pure tracking time without scoring (what the rate budget needs) on the cup / tray alternately
    for obj in a.objs[:1]:
        pass
    res["gpu_mem_mb"] = gpu_mem_mb(cl.server_pid)
    (OUT / "bench").mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.depth
    (OUT / "bench" / f"pose_{tag}.json").write_text(json.dumps(res, indent=1))
    cl.close()
    os._exit(0)


if __name__ == "__main__":
    main()
