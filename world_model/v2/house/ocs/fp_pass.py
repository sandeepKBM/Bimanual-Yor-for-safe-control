"""ocs/fp_pass.py -- FoundationPose (model-free) measurement pass over one OCS carry recording.  SIMULATION ONLY.

    export PYTHONPATH=/common/users/ss5772/bimanual:/common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang:/common/users/ss5772/bimanual/experiments/realtime
    CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl python -m world_model.v2.house.ocs.fp_onboard                      # once  -> /data/local/ss5772/molmospaces/ocs/fp/cup_sim_mesh/cup_sim.obj
    CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl python -m world_model.v2.house.ocs.fp_pass --seed S [--max-ticks N] [--W 320 --H 180]

For every wrist_due tick of /data/local/ss5772/molmospaces/ocs/rec/carry_s{S}.pkl.gz and each wrist side (left, right):
  1. fp_render: kinematic re-render of the TRUE state (what the camera sees) -> RGB + noisy depth (wrist_l/wrist_r depth model, robot self-filtered).
  2. NON-PRIVILEGED mask: depth points -> tray frame with T_tray<-cam = inv(FK tray) * FK wrist camera (ENCODER forward kinematics only), keep points with height above the tray
     top in [0.005, cup_h + 0.03] inside the tray rectangle shrunk by 2 cm; largest component.  mask px < min_px -> no measurement (occluded / out of view).
  3. FoundationPose: register (first time, after a loss, or after > GAP_REG consecutive empty frames; gravity-up prior = FK tray normal in the camera frame) else track (iters=2).
     Loss = the cup base centre lies > 2 cm outside the mask's 3-D bounding box, or jumped > 3 cm since the last accepted measurement -> re-register on this frame.
  4. ob_in_cam -> TRAY frame (same FK transform); yaw canonicalised (cup is rotationally symmetric): keep the measured z-axis, x = tray x projected orthogonal to it.
Truth enters only (a) as what the renderer draws and (b) in the evaluation printed by --eval / at the end of the run.

Output  <OUT>/fp_s{S}.npz : ticks (M,) int64 | ok (M,2) bool [left,right] | p_tray (M,2,3) | R_tray (M,2,3,3) (NaN where not ok) | mask_px (M,2) int64 | mode (M,2) int8 0 none/1 register/2 track
        | ms (M,2) FoundationPose ms per call;  --cam head: index 0 = head camera, index 1 always not-ok, meta['cam']='head' (sum if register after a failed track) | meta (json string)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np

from world_model.v2.house.ocs import fp_render as FR, util as U

OUT = Path(os.environ.get("OCS_FP_OUT", "/data/local/ss5772/molmospaces/ocs/fp"))
REC = Path(os.environ.get("OCS_OUT", "/data/local/ss5772/molmospaces/ocs")) / "rec"
MESH = OUT / "cup_sim_mesh" / "cup_sim.obj"
GAP_REG = 5                       # consecutive empty frames after which the next measurement re-registers instead of tracking
BOX_TOL, JUMP_TOL = 0.02, 0.03    # loss thresholds [m]


def box_dist(p, pts):
    """distance from point p to the axis-aligned 3-D bounding box of pts (0 inside)."""
    lo, hi = pts.min(0), pts.max(0)
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


def draw_estimate(img, K, T_cam_ob, r, h, color):
    """wireframe of the cup (base circle, rim circle, axis) projected with K; T_cam_ob = ob_in_cam (OpenCV)."""
    th = np.linspace(0, 2 * np.pi, 40)
    for z in (0.0, h):
        P = np.stack([r * np.cos(th), r * np.sin(th), np.full_like(th, z), np.ones_like(th)], 0)
        Pc = (T_cam_ob @ P)[:3]
        if (Pc[2] > 0.01).all():
            uv = (K @ Pc)[:2] / Pc[2]
            cv2.polylines(img, [np.round(uv.T).astype(np.int32).reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)
    a = (T_cam_ob @ np.array([[0, 0, 0, 1], [0, 0, h, 1.0]]).T)[:3]
    if (a[2] > 0.01).all():
        uv = (K @ a)[:2] / a[2]
        cv2.line(img, tuple(int(round(v)) for v in uv[:, 0]), tuple(int(round(v)) for v in uv[:, 1]), color, 1, cv2.LINE_AA)
    return img


def panel(o, mask, T_est, ok, K, r, h, crop=None):
    """rgb | rgb + mask (green) + estimate wireframe (yellow accepted / red rejected) | depth.  crop = (cx, cy, half_w, half_h, scale): crop + upscale around a pixel (head camera)."""
    rgb = cv2.cvtColor(o["rgb"], cv2.COLOR_RGB2BGR)
    ov = rgb.copy()
    ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    d = (np.clip(o["depth"] / (0.4 if crop is None else 1.0), 0, 1) * 255).astype(np.uint8)
    d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    if crop is None:
        if T_est is not None:
            draw_estimate(ov, K, T_est, r, h, (0, 255, 255) if ok else (0, 0, 255))
        return np.hstack([rgb, ov, d])
    cx, cy, hw, hh, sc = crop
    H, W = rgb.shape[:2]
    x0, y0 = int(np.clip(cx - hw, 0, W - 2 * hw)), int(np.clip(cy - hh, 0, H - 2 * hh))
    outs = []
    for im in (rgb, ov, d):
        c = cv2.resize(im[y0:y0 + 2 * hh, x0:x0 + 2 * hw], None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        outs.append(c)
    if T_est is not None:                                              # draw the wireframe on the upscaled crop (anti-aliased) with shifted/scaled intrinsics
        K2 = K.copy()
        K2[0, 2] -= x0
        K2[1, 2] -= y0
        K2 = np.diag([sc, sc, 1.0]) @ K2
        draw_estimate(outs[1], K2, T_est, r, h, (0, 255, 255) if ok else (0, 0, 255))
    return np.hstack(outs)


def run(seed, max_ticks=None, W=320, H=180, iters=2, min_px=400, gpu=6, mesh=MESH, out=OUT, png_ticks=(), png_dir=None, zmin=None, verbose=True, quiet_every=20, debug=False, cam="wrist"):
    assert gpu == 6, "this job is restricted to GPU 6"
    from world_model.v2.house.ocs.views import Recording, ArmFK
    from world_model.v2.house.foundation.pose_fp import Client
    rec = Recording.load(REC / f"carry_s{seed}.pkl.gz")
    A = rec.A
    t_all = time.time()
    fr = FR.FPRender(seed, W, H, cam=cam)
    head = cam == "head"
    if zmin is not None and not head:                                    # what-if override of the assumed wrist depth minimum range (default = depth_sim wrist spec: 0.10 m)
        from dataclasses import replace
        fr.spec = {s: replace(sp, z_min=float(zmin)) for s, sp in fr.spec.items()}
    fk = ArmFK(fr.sim)                                      # encoder FK (same object recipe as Recording.fk(), built from the render sim: no second scene build)
    cup_r, cup_h = U.cup_dims("sim")
    def crop_of(mask):                                          # head view: cup is ~40 px, crop 160x120 around the mask (or the cup region of the previous frame) and 3x upscale
        if not head:
            return None
        ys, xs = np.nonzero(mask)
        c = (xs.mean(), ys.mean()) if len(xs) else last_c[0]
        last_c[0] = c
        return (c[0], c[1], 80, 60, 3)
    last_c = [(W / 2, H / 2)]
    ticks = np.flatnonzero(A["wrist_due"])
    if max_ticks is not None:
        ticks = ticks[ticks < max_ticks]
    M = len(ticks)
    ok = np.zeros((M, 2), bool)
    p_tray = np.full((M, 2, 3), np.nan)
    R_tray = np.full((M, 2, 3, 3), np.nan)
    mask_px = np.zeros((M, 2), np.int64)
    mode = np.zeros((M, 2), np.int8)
    ms = np.zeros((M, 2))
    t_render = t_fp = 0.0
    cl = Client(sizes=[f"{W}x{H}"], gpu=gpu)
    try:
        t0 = time.time()
        cl.warmup()
        print(f"[fp_pass] seed {seed}: FP server READY {time.time() - t0:.0f}s; render ready; {M} wrist ticks", flush=True)
        names = {"left": "cup_L", "right": "cup_R", "head": "cup_H"}
        for s in fr.sides:
            cl.load_mesh(names[s], str(mesh))
        st = {s: dict(tracking=False, last_p=None, gap=0) for s in fr.sides}
        n_reg = {s: 0 for s in fr.sides}
        for i, k in enumerate(ticks):
            fr.set_tick(rec, k)
            qL, qR, lift = A["qL"][k], A["qR"][k], float(A["lift_enc"][k])
            pt, Rt = fk.tray(qL, qR, lift)
            for si, s in enumerate(fr.sides):                              # head: index 0 = head, index 1 stays not-ok (schema (M,2,...))
                t1 = time.time()
                o = fr.render(s, k, seed=seed)
                t_render += time.time() - t1
                org, Rc = FR.head_cam_base(lift) if head else fk.wrist_cam(s, qL, qR, lift)       # head: KNOWN mount + ENCODER lift (base frame); wrist: encoder FK
                T_tc = FR.cv_cam_in_tray(Rt, pt, org, Rc)                   # OpenCV camera pose in the FK tray frame
                K = o["K"]
                dm = cv2.medianBlur(o["depth"], 3)                           # 3x3 median for the GATE only (FoundationPose gets the raw noisy depth)
                mask, pts = FR.tray_gate_mask(dm, K, T_tc, U.HX, U.HY, shrink=0.02, h_lo=0.005, h_hi=cup_h + 0.03, min_px=min_px)
                mask_px[i, si] = int(mask.sum())
                s_ = st[s]
                if mask_px[i, si] < min_px:
                    s_["gap"] += 1
                    if (png_dir is not None) and int(k) in png_ticks:
                        cv2.imwrite(str(Path(png_dir) / f"s{seed}_t{int(k):04d}_{s}.png"), panel(o, mask, None, False, K, cup_r, cup_h, crop_of(mask)))
                    continue
                up_cam = T_tc[:3, :3].T @ np.array([0.0, 0.0, 1.0])         # FK tray normal in the camera frame (gravity-up proxy; no truth)
                need_reg = (not s_["tracking"]) or s_["gap"] > GAP_REG
                t1 = time.time()
                pose, used = None, 0
                try:
                    if need_reg:
                        pose, _, m_ = cl.register(names[s], o["rgb"], o["depth"], mask, K, iters=5, up_cam=up_cam)
                        used = 1
                    else:
                        pose, _, m_ = cl.track(names[s], o["rgb"], o["depth"], K, iters=iters)
                        used = 2
                    ms[i, si] += m_
                except RuntimeError as e:
                    print(f"[fp_pass] {s} tick {k}: FoundationPose error {str(e)[:120]}", flush=True)
                    s_["tracking"] = False
                    s_["gap"] += 1
                    continue
                s_["gap"] = 0

                def to_tray(pose_):
                    T = T_tc @ pose_
                    return T[:3, 3].copy(), T[:3, :3].copy()

                p, R = to_tray(pose)
                lost = box_dist(p, pts) > BOX_TOL or (s_["last_p"] is not None and used == 2 and np.linalg.norm(p - s_["last_p"]) > JUMP_TOL)
                if lost and used == 2:                                       # re-register on this frame
                    try:
                        pose, _, m_ = cl.register(names[s], o["rgb"], o["depth"], mask, K, iters=5, up_cam=up_cam)
                        ms[i, si] += m_
                        used = 1
                        p, R = to_tray(pose)
                        lost = box_dist(p, pts) > BOX_TOL
                    except RuntimeError as e:
                        print(f"[fp_pass] {s} tick {k}: re-register error {str(e)[:120]}", flush=True)
                        lost = True
                t_fp += time.time() - t1
                if debug:                                                    # TRUTH used for printing only
                    Rt_true = A["tray_R"][k].reshape(3, 3)
                    pe = 1e3 * np.linalg.norm(p - Rt_true.T @ (A["cup_p"][k] - A["tray_p"][k]))
                    te = math.degrees(math.acos(float(np.clip(R[:, 2] @ (Rt_true.T @ A["cup_R"][k].reshape(3, 3)[:, 2]), -1, 1))))
                    print(f"[dbg] t{int(k)} {s} used={used} box_dist={1e3 * box_dist(p, pts):.0f}mm lost={lost} err_true={pe:.1f}mm tilt_err={te:.1f}deg mask={mask_px[i, si]} p={np.round(p, 3).tolist()}", flush=True)
                mode[i, si] = used
                n_reg[s] += int(used == 1)
                if lost:
                    s_["tracking"] = False
                    ok[i, si] = False
                else:
                    s_["tracking"], s_["last_p"] = True, p
                    ok[i, si] = True
                    p_tray[i, si], R_tray[i, si] = p, FR.canonical_yaw(R)
                if (png_dir is not None) and int(k) in png_ticks:
                    cv2.imwrite(str(Path(png_dir) / f"s{seed}_t{int(k):04d}_{s}.png"), panel(o, mask, pose, ok[i, si], K, cup_r, cup_h, crop_of(mask)))
            if verbose and (i % quiet_every == 0 or i == M - 1):
                print(f"[fp_pass] {i + 1}/{M} tick {int(k)}  ok L/R {int(ok[:i + 1, 0].sum())}/{int(ok[:i + 1, 1].sum())}  reg {n_reg}  "
                      f"render {1e3 * t_render / (2 * (i + 1)):.0f} ms/frame  FP(+mask) {1e3 * t_fp / max(1, (mode[:i + 1] > 0).sum()):.0f} ms/call  wall {time.time() - t_all:.0f}s", flush=True)
    finally:
        cl.close()
        fr.close()
    meta = dict(W=W, H=H, cam=cam, K=fr.K["head" if head else "right"].tolist(), iters=iters, mesh=str(mesh), gpu=gpu, min_px=min_px, seed=seed, n_wrist_ticks=int(M),
                z_min=float(fr.spec["head" if head else "right"].z_min), gap_reg=GAP_REG, box_tol=BOX_TOL, jump_tol=JUMP_TOL, wall_s=time.time() - t_all, render_s=t_render, fp_s=t_fp,
                notes=("RGB-D = kinematic GL re-render of TRUE state (beads rigid with cup) + wrist_l/r depth noise model (robot self-filtered); mask = tray-plane gate from ENCODER FK "
                       "(height above tray top in [0.005, cup_h+0.03], tray rect shrunk 2 cm, largest component of 3x3-median depth); register uses a FK-tray-normal gravity-up prior; "
                       "poses in the FK TRAY frame, yaw canonicalised (x = tray x projected orthogonal to the measured z); mesh frame = cup base frame (model-free NOF mesh, fp_onboard)."))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fp_s{seed}.npz"
    np.savez(path, ticks=ticks.astype(np.int64), ok=ok, p_tray=p_tray, R_tray=R_tray, mask_px=mask_px, mode=mode, ms=ms, meta=json.dumps(meta))
    print(f"[fp_pass] saved {path}  wall {time.time() - t_all:.0f}s", flush=True)
    return str(path)


# ====================================================================================== evaluation (TRUTH used here only)
def evaluate(path, rec=None, window=None, fk_frame=True):
    """per side: median / p95 position error [mm] and tilt error [deg] of the ok measurements vs the TRUE cup pose in the TRUE tray frame (tray_R.T @ (cup_p - tray_p)), fraction ok,
    register count, mean ms.  window = (k0, k1) restricts to ticks in [k0, k1].  fk_frame=True also reports the same errors against the truth expressed in the ENCODER-FK tray
    frame (the frame the measurement is expressed in: removes the FK-tray / grasp-slip offset, i.e. isolates the perception error): keys *_fk."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    if rec is None:
        from world_model.v2.house.ocs.views import Recording
        rec = Recording.load(REC / f"carry_s{meta['seed']}.pkl.gz")
    A = rec.A
    fk = rec.fk() if fk_frame else None
    ticks = z["ticks"]
    sel = np.ones(len(ticks), bool) if window is None else (ticks >= window[0]) & (ticks <= window[1])
    res = {}
    for si, s in enumerate(("head", "unused") if meta.get("cam") == "head" else FR.SIDES):
        rows = np.flatnonzero(sel & z["ok"][:, si])
        e_pos, e_tilt, e_pos_fk, e_tilt_fk = [], [], [], []
        for r in rows:
            k = ticks[r]
            Rt_true = A["tray_R"][k].reshape(3, 3)
            p_true = Rt_true.T @ (A["cup_p"][k] - A["tray_p"][k])
            z_true = Rt_true.T @ A["cup_R"][k].reshape(3, 3)[:, 2]
            zm = z["R_tray"][r, si][:, 2]
            e_pos.append(1e3 * np.linalg.norm(z["p_tray"][r, si] - p_true))
            e_tilt.append(math.degrees(math.acos(float(np.clip(zm @ z_true, -1, 1)))))
            if fk is not None:
                pt, Rf = fk.tray(A["qL"][k], A["qR"][k], float(A["lift_enc"][k]))
                c, sn = math.cos(A["base"][k, 2]), math.sin(A["base"][k, 2])
                Rb = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])                      # base -> world
                p_base = Rb.T @ (A["cup_p"][k] - np.array([A["base"][k, 0], A["base"][k, 1], 0.0]))
                e_pos_fk.append(1e3 * np.linalg.norm(z["p_tray"][r, si] - Rf.T @ (p_base - pt)))
                zf = Rf.T @ Rb.T @ A["cup_R"][k].reshape(3, 3)[:, 2]
                e_tilt_fk.append(math.degrees(math.acos(float(np.clip(zm @ zf, -1, 1)))))
        n = int(sel.sum())
        q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
        call = sel & (z["mode"][:, si] > 0)
        res[s] = dict(n_frames=n, n_ok=int(len(rows)), frac_ok=len(rows) / max(n, 1), frac_mask=float((sel & (z["mask_px"][:, si] > 0)).sum() / max(n, 1)),
                      n_register=int(((z["mode"][:, si] == 1) & sel).sum()), n_track=int(((z["mode"][:, si] == 2) & sel).sum()),
                      pos_mm_med=q(e_pos, 50), pos_mm_p95=q(e_pos, 95), tilt_deg_med=q(e_tilt, 50), tilt_deg_p95=q(e_tilt, 95),
                      pos_mm_med_fk=q(e_pos_fk, 50), pos_mm_p95_fk=q(e_pos_fk, 95), tilt_deg_med_fk=q(e_tilt_fk, 50), tilt_deg_p95_fk=q(e_tilt_fk, 95),
                      ms_mean=float(z["ms"][call, si].mean()) if call.any() else None, mask_px_med=float(np.median(z["mask_px"][sel, si])) if n else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--cam", choices=["wrist", "head"], default="wrist", help="wrist: the two arm cameras (default); head: the torso ZED left eye (640x360 unless --W/--H given)")
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--min-px", type=int, default=None, help="min mask pixels for a measurement (default 400 at 320x180 wrist, 150 at 640x360 head)")
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--mesh", default=str(MESH))
    ap.add_argument("--out", default=None, help="default OUT (wrist) / OUT/head (head)")
    ap.add_argument("--zmin", type=float, default=None, help="what-if: override the wrist depth min range (spec 0.10 m)")
    ap.add_argument("--png-ticks", type=int, nargs="*", default=[])
    ap.add_argument("--png-dir", default=None)
    ap.add_argument("--debug", action="store_true", help="print per-call truth errors (evaluation only)")
    ap.add_argument("--eval", action="store_true", help="print the truth-based evaluation after the run (and for window 200-240)")
    ap.add_argument("--eval-window", type=int, nargs=2, default=[200, 240])
    ap.add_argument("--eval-only", default=None, help="path of an existing fp_s*.npz: evaluate and exit")
    a = ap.parse_args()
    if a.eval_only:
        for w in (None, tuple(a.eval_window)):
            print(w, json.dumps(evaluate(a.eval_only, window=w, fk_frame=False)))
        return
    if a.png_dir:
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
    head = a.cam == "head"
    W, H = a.W or (640 if head else 320), a.H or (360 if head else 180)
    min_px = a.min_px or (150 if head else 400)
    out = a.out or str(OUT / "head" if head else OUT)
    path = run(a.seed, a.max_ticks, W, H, a.iters, min_px, a.gpu, a.mesh, out, tuple(a.png_ticks), a.png_dir, a.zmin, debug=a.debug, cam=a.cam)
    if a.eval:
        for w in (None, tuple(a.eval_window)):
            print("window", w, json.dumps(evaluate(path, window=w, fk_frame=False)), flush=True)


if __name__ == "__main__":
    main()
