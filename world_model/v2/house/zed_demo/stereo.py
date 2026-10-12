"""stereo.py -- rectified stereo pair -> disparity (cv2.StereoSGBM + left-right consistency) -> metric depth -> world-frame point cloud.  SIMULATION ONLY.

Z = f * B / d  (z-depth along the optical axis, d in px of the rectified pair).  MuJoCo camera frame: x right, y up, camera looks along -z, so a pixel
(u, v) at depth Z maps to camera coords  ((u - cx) Z / f, -(v - cy) Z / f, -Z)  and to the world through the camera pose (origin, R_cam2world).
"""
from __future__ import annotations

import time

import numpy as np
import cv2

SGBM_DEFAULT = dict(minDisparity=0, numDisparities=192, blockSize=5, uniquenessRatio=8, speckleWindowSize=120, speckleRange=2, disp12MaxDiff=1,
                    preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_HH)


def make_sgbm(**kw):
    p = dict(SGBM_DEFAULT)
    p.update(kw)
    bs = p["blockSize"]
    p.setdefault("P1", 8 * bs * bs)
    p.setdefault("P2", 32 * bs * bs)
    return cv2.StereoSGBM_create(**p)


def compute_disparity(left, right, sgbm=None):
    """left/right uint8 RGB (H,W,3) -> (disp float32 px with NaN = invalid, seconds).  Uses grey images (P1/P2 for 1 channel)."""
    sgbm = sgbm or make_sgbm()
    gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    t = time.perf_counter()
    raw = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    dt = time.perf_counter() - t
    raw[raw < sgbm.getMinDisparity() + 0.5] = np.nan            # SGBM marks invalid as minDisparity-1
    raw[raw <= 0.5] = np.nan
    return raw, dt


def depth_from_disparity(disp, K):
    with np.errstate(invalid="ignore", divide="ignore"):
        return K["f"] * K["B"] / disp


def pixel_rays(K):
    """unit-z camera-frame direction scale (x/z, y/z) per pixel: (H, W, 2) with y up (MuJoCo)."""
    u = np.arange(K["W"], dtype=np.float32)
    v = np.arange(K["H"], dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - K["cx"]) / K["f"], -(vv - K["cy"]) / K["f"]], -1)


def depth_to_cloud(depth, K, origin, R, mask=None):
    """z-depth map (H,W) -> world points (n,3) + flat pixel index (n,).  mask: extra bool (H,W) pixels to keep."""
    ok = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        ok &= mask
    idx = np.flatnonzero(ok.ravel())
    xy = pixel_rays(K).reshape(-1, 2)[idx]
    z = depth.ravel()[idx]
    pc = np.stack([xy[:, 0] * z, xy[:, 1] * z, -z], 1)
    return (pc @ R.T + origin).astype(np.float64), idx


def project(P, K, origin, R):
    """world points (n,3) -> pixel (u,v) (n,2) and z-depth (n,)."""
    pc = (np.asarray(P, float) - origin) @ R
    z = -pc[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = K["cx"] + K["f"] * pc[:, 0] / z
        v = K["cy"] - K["f"] * pc[:, 1] / z
    return np.stack([u, v], 1), z


# ------------------------------------------------------------------------------------------------ evaluation vs MuJoCo ground truth
RANGE_BINS = (0.3, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)


def evaluate(depth, gt, robot_mask=None):
    """depth: stereo depth (NaN invalid); gt: MuJoCo z-depth (left camera).  Error = depth - gt on pixels where both are valid.
    Returns overall + per-range-bin mean(signed) / RMSE / p95|err| / fill rate (valid stereo / pixels with GT in the bin), and fill over all pixels."""
    gvalid = np.isfinite(gt) & (gt > 0.05) & (gt < 30.0)
    sv = np.isfinite(depth)
    err = depth - gt
    out = dict(fill_all=float(sv[gvalid].mean()), n_px=int(gvalid.sum()))
    if robot_mask is not None:
        sc = gvalid & ~robot_mask
        out["fill_scene_only"] = float(sv[sc].mean())
    both = gvalid & sv
    e = err[both]
    inl = np.abs(e) < 0.05
    out.update(rmse_inliers_lt5cm=float(np.sqrt((e[inl] ** 2).mean())), mean_err=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())), p95_abs=float(np.percentile(np.abs(e), 95)),
               median_abs=float(np.median(np.abs(e))))
    bins = []
    edges = RANGE_BINS
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = gvalid & (gt >= lo) & (gt < hi)
        sb = sel & sv
        if sel.sum() < 50:
            continue
        eb = err[sb]
        bins.append(dict(lo=lo, hi=hi, n=int(sel.sum()), fill=float(sb.sum() / sel.sum()),
                         mean=float(eb.mean()) if len(eb) else None, rmse=float(np.sqrt((eb ** 2).mean())) if len(eb) else None,
                         p95=float(np.percentile(np.abs(eb), 95)) if len(eb) else None,
                         rel_rmse=float(np.sqrt(((eb / gt[sb]) ** 2).mean())) if len(eb) else None))
    out["bins"] = bins
    # robust (MAD) statistics: the matching also makes some gross outliers at depth discontinuities
    out["mad_sigma"] = float(1.4826 * np.median(np.abs(e - np.median(e))))
    out["outlier_frac_gt5cm"] = float((np.abs(e) > 0.05).mean())
    return out


def error_at_range(ev, z, key="p95"):
    """measured stereo error [m] at range z from an `evaluate` result (bin lookup, last bin extrapolated by z^2 scaling)."""
    b = ev["bins"]
    for r in b:
        if r["lo"] <= z < r["hi"]:
            return float(r[key])
    r = b[-1] if z >= b[-1]["hi"] else b[0]
    return float(r[key] * (z / (0.5 * (r["lo"] + r["hi"]))) ** 2)


def colormap_disp(disp, dmax=None, invalid=(20, 20, 24)):
    """RGB uint8 image of a disparity map (TURBO; invalid = dark)."""
    v = np.isfinite(disp)
    dmax = dmax or float(np.nanpercentile(disp[v], 99)) if v.any() else 1.0
    dn = np.clip(np.nan_to_num(disp, nan=0.0) / dmax, 0, 1)
    img = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
    img[~v] = invalid
    return img
