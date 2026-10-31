"""ocs/fp_tray.py -- VISION measurement of the held tray's pose from the head-ZED depth (non-privileged), then the FoundationPose cup pose re-expressed in the
VISUAL tray frame.  SIMULATION ONLY.

Why: fp/head measures the cup ~1 mm accurately, but in the ENCODER-FK tray frame, which assumes the nominal grasp; the true tray sits up to ~4 cm off it, so the
cup-in-tray position the E4 'inside' atom sees (ocs.consumers._rel_entity: the cup's OWN parent-frame pose, tray_top = nominal 30x20 cm rectangle) inherits that
offset.  Measure the tray from the same depth image and the cup is expressed relative to the MEASURED tray.

    export PYTHONPATH=...; CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl python -m world_model.v2.house.ocs.fp_tray --seed S [--eval]  [--eval-only <fp_s*.npz>]
    reads  /data/local/ss5772/molmospaces/ocs/rec/carry_s{S}.pkl.gz (state to RENDER + encoders) and fp/head/fp_s{S}.npz (cup pose in the FK tray frame)
    writes /data/local/ss5772/molmospaces/ocs/fp/head_vtray/fp_s{S}.npz : the fp/head keys (p_tray / R_tray now in the VISUAL tray frame) + p_tray_fk, R_tray_fk,
           vtray_ok (M,), T_fk_vtray (M,4,4) [visual tray pose in the FK tray frame], vtray_inliers (M,), vtray_ms (M,), vtray_cost (M,)

Method (fit_tray; inputs = the noisy depth point cloud in the FK tray frame + the camera origin; the FK pose only GATES and INITIALISES):
  1. gate: points with height above the FK tray plane in [-0.04, +0.015] m and inside the nominal rectangle grown by 8 cm (admits a ~4-5 cm grasp offset).
  2. RANSAC plane (3-point, 250 hypotheses, |d| < 1 cm, normal within 20 deg of the FK normal) + 2 least-squares refits on its inliers -> tray-top normal n and height c.
     The tray top is the plane z = 0 of the tray frame (tray_env.tray_xml: box tray_top size HX HY TRAY_T at z = -TRAY_T, testbeds/b_tray_ball/tray_env.py:103).
  3. footprint (x, y, yaw) in the fitted plane, tray half sizes HX = 0.15, HY = 0.10 (testbeds/b_tray_ball/tray_env.py:31 = ocs/util.py:14; the h5 carry tray
     is built by teleop/scene.py:267, lip=False, flat 30 x 20 cm plate).  Every depth point is classified against the plane: ON (|h| < 1.2 cm), BEHIND (h < -3 cm: the
     background seen past the tray edge, localised by the ray's intersection with the plane), IN FRONT (cup / beads / fingers: ignored).  Invalid pixels (robot self-filter,
     dropout, grazing) are unknown: occlusion removes evidence, it never creates any.  Cost(x, y, yaw) = sum over ON points of the distance outside the rectangle
     + sum over BEHIND points of the distance inside it (both truncated at 3 cm) + a negligible tie-break prior toward the FK pose; coarse grid (+/-9 cm, +/-9 deg) then Nelder-Mead.
  4. fit accepted iff the plane + ON support are large enough and the result is inside the gate; else the FK value is kept and vtray_ok = False.
Truth is used ONLY by evaluate().
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

from world_model.v2.house.ocs import util as U

OUT = Path(os.environ.get("OCS_FP_OUT", "/data/local/ss5772/molmospaces/ocs/fp"))
REC = Path(os.environ.get("OCS_OUT", "/data/local/ss5772/molmospaces/ocs")) / "rec"
HX, HY = U.HX, U.HY


class Cfg:
    gate_z = (-0.04, 0.015)           # height above the FK plane [m] of the points fed to RANSAC
    gate_xy = 0.08                    # rectangle growth of the RANSAC gate [m] (>= grasp offset ~4-5 cm)
    ransac_iters = 250
    ransac_tol = 0.010
    max_tilt = math.radians(20.0)
    min_plane_pts = 300
    on_tol = 0.012                    # |h| of an ON-plane point
    behind_h = -0.03                  # h of a BEHIND point
    window = 0.10                     # classification window around the FK footprint [m]
    cap = 0.03
    plate_t = 0.006                   # plate thickness [m] (2 * TRAY_T, tray_env.py:32): its side face is visible on the camera side and widens the apparent footprint there
    eps_on, eps_bh, w_on = 0.0, 0.002, 2.0   # hinge margins [m] and ON weight: tuned on the calibration seeds 0-2 only (median xy error 0.07 mm, p95 2.6 mm; held-out 3-5: 0.16 / 3.4 mm)
    max_pts = 1500                    # subsample of classified points used in the footprint search
    min_on = 150
    grid_xy = np.arange(-0.08, 0.0801, 0.02)
    grid_yaw = np.radians(np.arange(-8.0, 8.01, 4.0))
    prior_xy, prior_yaw = 50.0, 5.0   # tie-break prior weights on dx^2 + dy^2 [m^2], yaw^2 [rad^2]
    max_off = 0.10                    # accepted |dx|, |dy| [m]
    max_yaw = math.radians(15.0)


def _rot_basis(n):
    """in-plane basis (a, b) of a plane with unit normal n: a = FK x projected on the plane, b = n x a."""
    a = np.array([1.0, 0.0, 0.0]) - n[0] * n
    a /= np.linalg.norm(a)
    return a, np.cross(n, a)


def _lsq_plane(P):
    c0 = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c0, full_matrices=False)
    n = Vt[2]
    if n[2] < 0:
        n = -n
    return n, float(n @ c0)


def ransac_plane(P, cfg, rng):
    """-> (n, c, n_inliers) of the dominant near-horizontal plane n.x = c in P (N,3) [FK tray frame], or None."""
    N = len(P)
    if N < cfg.min_plane_pts:
        return None
    best, best_cnt = None, 0
    idx = rng.integers(0, N, size=(cfg.ransac_iters, 3))
    cos_max = math.cos(cfg.max_tilt)
    for tri in idx:
        p0, p1, p2 = P[tri]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        if n[2] < cos_max:
            continue
        c = float(n @ p0)
        cnt = int((np.abs(P @ n - c) < cfg.ransac_tol).sum())
        if cnt > best_cnt:
            best, best_cnt = (n, c), cnt
    if best is None or best_cnt < cfg.min_plane_pts:
        return None
    n, c = best
    for _ in range(2):
        inl = np.abs(P @ n - c) < cfg.ransac_tol
        if inl.sum() < cfg.min_plane_pts:
            return None
        n, c = _lsq_plane(P[inl])
    return n, c, int((np.abs(P @ n - c) < cfg.ransac_tol).sum())


def _sd_rect(uv, G):
    """signed distance (<0 inside) of in-plane points uv (N,2) to the rectangles of the poses G (n,3) = (dx, dy, psi): centre (dx, dy), rotated psi, half sizes HX, HY -> (n, N)."""
    cs, sn = np.cos(G[:, 2])[:, None], np.sin(G[:, 2])[:, None]
    qx, qy = uv[None, :, 0] - G[:, 0:1], uv[None, :, 1] - G[:, 1:2]
    x, y = cs * qx + sn * qy, -sn * qx + cs * qy
    ex, ey = np.abs(x) - HX, np.abs(y) - HY
    return np.hypot(np.maximum(ex, 0.0), np.maximum(ey, 0.0)) + np.minimum(np.maximum(ex, ey), 0.0)


def _costs(G, on, bh, cfg):
    """cost of every pose row of G (n,3).  on / bh = (N,4) [u, v, eu, ev]: in-plane hit of the pixel's ray with the top plane and the displacement (eu, ev) of the ray from the
    top plane to the plate BOTTOM (= thickness * cot(elevation) along the ray's in-plane direction: the plate has a visible side face on the camera side).
      ON  pixel: some point of the ray inside the plate (top .. bottom) lies in the rectangle  ->  cost = distance outside = min over {top, bottom} hit of max(sd, 0) (+ eps_on)
      BEHIND pixel: the ray passes below the plate's bottom edge or beyond the edge  ->  the bottom hit must lie outside  ->  cost = max(-sd(bottom hit), 0) (+ eps_bh)"""
    c = np.zeros(len(G))
    for i0 in range(0, len(G), 64):
        g = G[i0:i0 + 64]
        t = np.zeros(len(g))
        if len(on):
            sd = np.minimum(_sd_rect(on[:, :2], g), _sd_rect(on[:, :2] + on[:, 2:], g))
            t += np.minimum(np.maximum(sd + cfg.eps_on, 0.0), cfg.cap).sum(1) * cfg.w_on
        if len(bh):
            sd = _sd_rect(bh[:, :2] + bh[:, 2:], g)
            t += np.minimum(np.maximum(-sd + cfg.eps_bh, 0.0), cfg.cap).sum(1)
        c[i0:i0 + 64] = t + cfg.prior_xy * (g[:, 0] ** 2 + g[:, 1] ** 2) + cfg.prior_yaw * g[:, 2] ** 2
    return c


def fit_tray(P, cam_o, cfg: Cfg | None = None, rng=None):
    """P (N,3) valid depth points and cam_o (3,) camera origin, both in the FK tray frame (the FK pose only gates / initialises).
    -> dict(ok, T (4,4) visual tray pose in the FK tray frame, n_plane, n_on, n_behind, cost, why) ; T = identity when not ok."""
    cfg = cfg or Cfg()
    rng = rng if rng is not None else np.random.default_rng(0)
    res = dict(ok=False, T=np.eye(4), n_plane=0, n_on=0, n_behind=0, cost=np.nan, why="")
    gate = (P[:, 2] >= cfg.gate_z[0]) & (P[:, 2] <= cfg.gate_z[1]) & (np.abs(P[:, 0]) <= HX + cfg.gate_xy) & (np.abs(P[:, 1]) <= HY + cfg.gate_xy)
    Pg = P[gate]
    pl = ransac_plane(Pg, cfg, rng)
    if pl is None:
        res["why"] = "no plane"
        return res
    n, c, n_pl = pl
    res["n_plane"] = n_pl
    a, b = _rot_basis(n)
    # ---- classify every point against the fitted plane
    h = P @ n - c
    on = np.abs(h) < cfg.on_tol
    bh = h < cfg.behind_h
    # in-plane location of a pixel = where its RAY meets the fitted plane (not the noisy measured point: depth noise acts ALONG the ray, i.e. mostly sideways at grazing
    # incidence, and would smear the footprint boundary outward; the ray is exact, only the ON / BEHIND class depends on the depth)
    sel = on | bh
    Ps = P[sel]
    d = Ps - cam_o
    den = d @ n
    s = (c - float(cam_o @ n)) / np.where(np.abs(den) < 1e-9, -1e-9, den)
    okr = (s > 0.0) & (s < 1.5)
    hit = cam_o + s[:, None] * d
    ru = d / np.linalg.norm(d, axis=1, keepdims=True)
    sin_el = np.maximum(np.abs(ru @ n), 0.15)
    g_in = ru - (ru @ n)[:, None] * n                                  # ray direction in the plane (away from the camera)
    ext = cfg.plate_t * g_in / sin_el[:, None]                          # = thickness * cot(elevation) * unit in-plane direction
    uv = np.stack([hit @ a, hit @ b, ext @ a, ext @ b], 1)
    isbh = bh[sel]
    uv_on, uv_bh = uv[okr & ~isbh], uv[okr & isbh]
    w = cfg.window + 0.0
    inwin = lambda uv: (np.abs(uv[:, 0]) <= HX + cfg.gate_xy + w) & (np.abs(uv[:, 1]) <= HY + cfg.gate_xy + w)
    uv_on, uv_bh = uv_on[inwin(uv_on)], uv_bh[inwin(uv_bh)]
    res["n_on"], res["n_behind"] = len(uv_on), len(uv_bh)
    if len(uv_on) < cfg.min_on:
        res["why"] = "few ON points"
        return res
    sub = lambda uv: uv[rng.choice(len(uv), cfg.max_pts, replace=False)] if len(uv) > cfg.max_pts else uv
    uv_on, uv_bh = sub(uv_on), sub(uv_bh)
    gx, gy, gp = np.meshgrid(cfg.grid_xy, cfg.grid_xy, cfg.grid_yaw, indexing="ij")
    G = np.stack([gx.ravel(), gy.ravel(), gp.ravel()], 1)
    cg = _costs(G, uv_on, uv_bh, cfg)
    x0 = G[int(np.argmin(cg))]
    from scipy.optimize import minimize
    r = minimize(lambda x: float(_costs(x[None], uv_on, uv_bh, cfg)[0]), x0, method="Nelder-Mead", options=dict(xatol=5e-4, fatol=1e-6, initial_simplex=np.array(
        [x0, x0 + [0.01, 0, 0], x0 + [0, 0.01, 0], x0 + [0, 0, math.radians(2.0)]])))
    dx, dy, psi = r.x
    res["cost"] = float(r.fun)
    if abs(dx) > cfg.max_off or abs(dy) > cfg.max_off or abs(psi) > cfg.max_yaw:
        res["why"] = "fit outside gate"
        return res
    cp, sp = math.cos(psi), math.sin(psi)
    T = np.eye(4)
    T[:3, 0] = cp * a + sp * b
    T[:3, 1] = -sp * a + cp * b
    T[:3, 2] = n
    T[:3, 3] = c * n + dx * a + dy * b
    res.update(ok=True, T=T, why="ok")
    return res


# ====================================================================================== image -> cloud
def cloud_fk(depth, K, T_tc):
    """valid (depth > 0) pixels of a z-depth image -> (N,3) points in the FK tray frame, camera origin (3,)."""
    from world_model.v2.house.ocs import fp_render as FR
    P = FR.backproject(np.nan_to_num(depth, nan=0.0), K)
    m = depth > 0.0
    return P[m] @ T_tc[:3, :3].T + T_tc[:3, 3], T_tc[:3, 3].copy()


def re_express(T_fk_v, p_fk, R_fk):
    """cup pose (p, R) in the FK tray frame -> in the visual tray frame; the cup is rotationally symmetric: yaw canonicalised to the (visual) tray x as fp_pass does."""
    from world_model.v2.house.ocs import fp_render as FR
    Rv, pv = T_fk_v[:3, :3].T, T_fk_v[:3, :3].T @ (np.asarray(p_fk) - T_fk_v[:3, 3])
    return pv, FR.canonical_yaw(Rv @ np.asarray(R_fk, float))


def process_seed(seed, out=None, W=640, H=360, max_ticks=None, fp_dir=None, cfg=None, verbose=True):
    import cv2
    from world_model.v2.house.ocs import fp_render as FR
    from world_model.v2.house.ocs.views import Recording, ArmFK
    fp_dir = Path(fp_dir or OUT / "head")
    out = Path(out or OUT / "head_vtray")
    z = dict(np.load(fp_dir / f"fp_s{seed}.npz"))
    meta = json.loads(str(z["meta"]))
    assert meta["cam"] == "head" and (meta["W"], meta["H"]) == (W, H)
    rec = Recording.load(REC / f"carry_s{seed}.pkl.gz")
    A = rec.A
    fr = FR.FPRender(seed, W, H, cam="head")                          # same camera / 640x360 / noise model / RNG seed as fp_pass --cam head
    fk = ArmFK(fr.sim)
    cfg = cfg or Cfg()
    ticks = z["ticks"]
    M = len(ticks) if max_ticks is None else int(np.sum(ticks < max_ticks))
    ok = np.zeros(M, bool)
    Tfv = np.tile(np.eye(4), (M, 1, 1))
    ninl = np.zeros(M, np.int64)
    ms = np.zeros(M)
    cost = np.full(M, np.nan)
    p_new, R_new = z["p_tray"][:M].copy(), z["R_tray"][:M].copy()
    t0 = time.time()
    try:
        for i in range(M):
            k = int(ticks[i])
            fr.set_tick(rec, k)
            o = fr.render("head", k, seed=seed, seg=True)
            lift = float(A["lift_enc"][k])
            pt, Rt = fk.tray(A["qL"][k], A["qR"][k], lift)
            org, Rc = FR.head_cam_base(lift)
            T_tc = FR.cv_cam_in_tray(Rt, pt, org, Rc)
            dm = cv2.medianBlur(o["depth"], 3)
            P, co = cloud_fk(dm, o["K"], T_tc)
            t1 = time.time()
            r = fit_tray(P, co, cfg, np.random.default_rng(k))
            ms[i] = 1e3 * (time.time() - t1)
            ok[i], Tfv[i], ninl[i], cost[i] = r["ok"], r["T"], r["n_plane"], r["cost"]
            if r["ok"] and z["ok"][i, 0]:
                p_new[i, 0], R_new[i, 0] = re_express(r["T"], z["p_tray"][i, 0], z["R_tray"][i, 0])
            if verbose and (i % 25 == 0 or i == M - 1):
                print(f"[fp_tray] s{seed} {i + 1}/{M} tick {k} fit ok {int(ok[:i + 1].sum())}/{i + 1}  fit {ms[:i + 1].mean():.0f} ms  wall {time.time() - t0:.0f}s", flush=True)
    finally:
        fr.close()
    z2 = {k_: (v[:M] if (isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(ticks)) else v) for k_, v in z.items()}
    z2.update(p_tray=p_new, R_tray=R_new, p_tray_fk=z["p_tray"][:M], R_tray_fk=z["R_tray"][:M], vtray_ok=ok, T_fk_vtray=Tfv, vtray_inliers=ninl, vtray_ms=ms, vtray_cost=cost)
    meta.update(vtray=True, vtray_note="p_tray / R_tray = cup pose in the VISUAL tray frame (fp_tray.fit_tray); *_fk = original FK-tray-frame values", vtray_wall_s=time.time() - t0)
    z2["meta"] = json.dumps(meta)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fp_s{seed}.npz"
    np.savez(path, **z2)
    print(f"[fp_tray] saved {path}  wall {time.time() - t0:.0f}s", flush=True)
    return str(path)


# ====================================================================================== evaluation (TRUTH used here only)
def _yaw_tilt(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0])), math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))


def evaluate(path, rec=None):
    """per seed: visual-tray vs FK-tray error against the TRUE tray (expressed in the FK tray frame), cup-in-visual-tray vs cup-in-FK-tray error against the truth cup-in-true-tray."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    from world_model.v2.house.ocs.views import Recording
    rec = rec or Recording.load(REC / f"carry_s{meta['seed']}.pkl.gz")
    A, fk = rec.A, rec.fk()
    ticks, ok = z["ticks"], z["vtray_ok"]
    M = len(ok)
    pv, pf, yv, yf, tv, tf, ev, ef, evx, efx = [], [], [], [], [], [], [], [], [], []
    for i in range(M):
        k = int(ticks[i])
        pt, Rf = fk.tray(A["qL"][k], A["qR"][k], float(A["lift_enc"][k]))
        c_, s_ = math.cos(A["base"][k, 2]), math.sin(A["base"][k, 2])
        Rb = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]])
        pt_true, Rt_true_b = Rb.T @ (A["tray_p"][k] - np.array([A["base"][k, 0], A["base"][k, 1], 0.0])), Rb.T @ A["tray_R"][k].reshape(3, 3)
        T_true_fk = np.eye(4)                                              # TRUE tray pose in the FK tray frame
        T_true_fk[:3, :3], T_true_fk[:3, 3] = Rf.T @ Rt_true_b, Rf.T @ (pt_true - pt)
        e_f = np.linalg.norm(T_true_fk[:3, 3])
        yaw_f, tilt_f = _yaw_tilt(T_true_fk[:3, :3])
        if not ok[i]:
            continue
        Tv = z["T_fk_vtray"][i]
        Dp = np.linalg.inv(T_true_fk) @ Tv                                  # visual tray in the true tray frame
        pv.append(1e3 * np.linalg.norm(Dp[:3, 3]))
        yaw_v, tilt_v = _yaw_tilt(Dp[:3, :3])
        yv.append(abs(yaw_v))
        tv.append(tilt_v)
        pf.append(1e3 * e_f)
        yf.append(abs(yaw_f))
        tf.append(tilt_f)
        if z["ok"][i, 0]:
            Rt_true = A["tray_R"][k].reshape(3, 3)
            ptrue = Rt_true.T @ (A["cup_p"][k] - A["tray_p"][k])
            ev.append(1e3 * np.linalg.norm(z["p_tray"][i, 0] - ptrue))
            ef.append(1e3 * np.linalg.norm(z["p_tray_fk"][i, 0] - ptrue))
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
    return dict(n=M, fit_cov=float(ok.mean()), ms_med=float(np.median(z["vtray_ms"])), ms_mean=float(z["vtray_ms"].mean()),
                tray_pos_mm_vis=[q(pv, 50), q(pv, 95)], tray_pos_mm_fk=[q(pf, 50), q(pf, 95)], tray_yaw_deg_vis=[q(yv, 50), q(yv, 95)], tray_yaw_deg_fk=[q(yf, 50), q(yf, 95)],
                tray_tilt_deg_vis=[q(tv, 50), q(tv, 95)], tray_tilt_deg_fk=[q(tf, 50), q(tf, 95)],
                cup_in_tray_mm_vis=[q(ev, 50), q(ev, 95)], cup_in_tray_mm_fk=[q(ef, 50), q(ef, 95)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fp-dir", default=None)
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-only", default=None)
    a = ap.parse_args()
    if a.eval_only:
        print(json.dumps(evaluate(a.eval_only)), flush=True)
        return
    p = process_seed(a.seed, a.out, max_ticks=a.max_ticks, fp_dir=a.fp_dir)
    if a.eval:
        print("EVAL", a.seed, json.dumps(evaluate(p)), flush=True)


if __name__ == "__main__":
    main()
