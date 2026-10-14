"""grasp/estimate.py -- wrist-camera estimate of the TRAY pose (the "handle" the pads grip) from one noisy depth frame.  SIMULATION ONLY.

The tray is a 0.30 x 0.20 m, 6 mm plate (tray_env HX, HY, TRAY_T); the handle is its edge band (left hand: the +x edge, right hand: the -x
edge, tray-frame x).  Given a PRIOR tray pose (the operator brought the hands to within ~3.5 cm / 6 deg, so the prior is the nominal
gripper-to-tray relpose composed with the current hand pose) the estimator fits the tray RECTANGLE (pose: 3 translations + 3 rotations)
to the point cloud by maximising a render-and-compare occupancy score, robust to the cup, the shelf, outliers and partial views:

    S(pose) = sum_in w(z) 1[|x|<HX+m, |y|<HY+m]          points on the plate top surface (|z| < eps)           (positive evidence)
              - W_NEG  #{z < -4.5 mm, inside footprint}   a return BELOW the plate where the plate should be   (negative evidence:
                                                           what makes the edge observable -- the shelf is 6 mm lower)
              - W_BAND #{3.5 < z < 9 mm, inside footprint} thin band just above the plate (a misplaced plate one step too low)
              - 0.5 * LAMBDA * |pose - prior|^2_Sigma      weak prior (the along-edge position / roll are weakly observable)

Segmentation modes (the rectangle score is the same; only which points may count as plate points differs):
    "priv"  PRIVILEGED upper bound: plate points = hits whose geom id is the tray (sim segmentation mask).  Not available on the robot.
    "depth" DEPTH-ONLY: no labels.  Points are cropped to the prior volume; a plane is RANSAC-fitted near the prior plate height to
            initialise (z, roll, pitch); every point with |z| < eps counts; the rectangle fit then does the rest.  Reads nothing but
            the (noisy) point cloud, the camera pose from the arm kinematics, and the prior.
Output: pose (world), confidence in [0,1], the spread of the best hypotheses (std dx, dy, dz, yaw) and diagnostics.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY

EPS_IN = 0.0025
MARGIN = 0.004
Z_NEG = -0.0035
Z_BAND_HI = 0.009
W_NEG, W_BAND = 1.5, 0.7
CROP = (HX + 0.10, HY + 0.10, -0.05, 0.09)            # |x|, |y|, z range in the prior tray frame
SIG0 = np.array([0.025, 0.025, 0.008, math.radians(5.0), math.radians(0.6), math.radians(0.6)])   # initial search spread
SIG_PRIOR = np.array([0.04, 0.04, 0.012, math.radians(10.0), math.radians(1.0), math.radians(1.0)])
LAMBDA = 25.0
MAX_PTS = 700
BOUNDS = np.array([0.07, 0.07, 0.012, math.radians(12.0), math.radians(1.5), math.radians(1.5)])   # hard search box around the (LEVELLED) prior: operator tolerance
# 3.5 cm / 6 deg yaw + margin.  Roll / pitch are NOT searched freely: the tray rests / is carried level (gravity direction from the base IMU +
# arm kinematics); a free tilt lets the fit lay a ramp across the 6 mm plate-to-shelf step (found and fixed on the right wrist camera).
RING = 0.06                                       # plate-level points within this distance OUTSIDE the footprint are unexplained plate (penalised)
W_OUT = 1.0


@dataclass
class PoseEst:
    ok: bool
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))          # tray origin (top-surface centre), world
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    conf: float = 0.0
    n_in: float = 0.0
    n_neg: float = 0.0
    n_pts: int = 0
    spread: np.ndarray = field(default_factory=lambda: np.full(4, np.inf))      # std over the best hypotheses: dx, dy, dz [m], yaw [rad]
    score: float = 0.0
    ms: float = 0.0
    n_out: float = 0.0
    seg: str = ""
    delta: np.ndarray = field(default_factory=lambda: np.zeros(6))      # fitted pose in the prior frame (dx,dy,dz,yaw,pitch,roll)


def _Rzyx(psi, th, ph):
    cz, sz, cy, sy, cx, sx = np.cos(psi), np.sin(psi), np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
    # R = Rz(psi) Ry(th) Rx(ph), batched over leading dims
    R = np.empty(np.shape(psi) + (3, 3))
    R[..., 0, 0] = cz * cy
    R[..., 0, 1] = cz * sy * sx - sz * cx
    R[..., 0, 2] = cz * sy * cx + sz * sx
    R[..., 1, 0] = sz * cy
    R[..., 1, 1] = sz * sy * sx + cz * cx
    R[..., 1, 2] = sz * sy * cx - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cy * cx
    return R


def ransac_plane(P, rng, z0=0.0, band=0.025, tol=0.003, iters=120, z_weight=40.0, z_lo=None):
    """plane z = a x + b y + c through points near the prior plate height; returns (a,b,c,inlier_mask) or None.  Ties between the
    plate (z ~ 0) and the shelf (z ~ -6 mm) are broken toward the plane nearest the prior height (the hands were brought to ~2 cm)."""
    sel = np.flatnonzero((np.abs(P[:, 2] - z0) < band) & ((P[:, 2] - z0 > z_lo) if z_lo is not None else True))   # z_lo: ignore everything lower than the plate by more than the hands' height tolerance (the 12 mm lower counter slab)
    if len(sel) < 30:
        return None
    Q = P[sel]
    best, best_sc = None, -1e9
    for _ in range(iters):
        i = rng.choice(len(Q), 3, replace=False)
        A = np.c_[Q[i, :2], np.ones(3)]
        try:
            abc = np.linalg.solve(A, Q[i, 2])
        except np.linalg.LinAlgError:
            continue
        if abs(abc[0]) > 0.2 or abs(abc[1]) > 0.2:
            continue
        r = Q[:, 2] - (Q[:, :2] @ abc[:2] + abc[2])
        inl = np.abs(r) < tol
        sc = inl.sum() - z_weight * abs(abc[2] - z0) / 0.006        # preference for the prior height (z_weight 40: small; the assist uses 400)
        if sc > best_sc:
            best_sc, best = sc, abc
    if best is None:
        return None
    # LS refit on the inliers
    for _ in range(2):
        r = Q[:, 2] - (Q[:, :2] @ best[:2] + best[2])
        inl = np.abs(r) < tol
        if inl.sum() < 10:
            return None
        A = np.c_[Q[inl, :2], np.ones(inl.sum())]
        best = np.linalg.lstsq(A, Q[inl, 2], rcond=None)[0]
    return best


def _score(P, label, H, W_neg=W_NEG, W_band=W_BAND, use_band=True):
    """P (n,3) prior-frame points; H: dict of arrays (h,) dx,dy,dz,psi,th,ph; label (n,) bool or None.  -> (h,) score, (h,) n_in, (h,) n_neg"""
    R = _Rzyx(H["psi"], H["th"], H["ph"])                              # (h,3,3)
    t = np.stack([H["dx"], H["dy"], H["dz"]], -1)                      # (h,3)
    Pt = np.einsum("hji,hnj->hni", R, P[None, :, :] - t[:, None, :])               # R^T (p - t): points in the hypothesis tray frame
    x, y, z = Pt[..., 0], Pt[..., 1], Pt[..., 2]
    foot_in = (np.abs(x) < HX + MARGIN) & (np.abs(y) < HY + MARGIN)
    foot_core = (np.abs(x) < HX - 0.002) & (np.abs(y) < HY - 0.002)
    ring = (np.abs(x) < HX + RING) & (np.abs(y) < HY + RING) & ~foot_in
    w = np.clip(1.0 - (z / EPS_IN) ** 2, 0.0, 1.0)
    if label is None:
        out = (w * ring).sum(1)
        inl = (w * foot_in).sum(1)
        neg = ((z < Z_NEG) & foot_core).sum(1)
        band = ((z > -Z_NEG) & (z < Z_BAND_HI) & foot_core).sum(1) if use_band else 0.0
    else:
        out = (w * ring * label[None]).sum(1)
        inl = (w * foot_in * label[None]).sum(1)
        neg = ((z < Z_NEG) & foot_core & ~label[None]).sum(1)
        band = 0.0
    return inl - W_neg * neg - W_band * band - W_OUT * out, inl, neg, out


def fit_rectangle(P, label, init=None, rng=None, iters=9, n_hyp=260, sig0=SIG0, prior=None, bounds=BOUNDS):
    """pyramid random search.  init: 6-vector start (default zeros).  prior: 6-vector centre of the weak Gaussian prior (default zeros)."""
    rng = np.random.default_rng(0) if rng is None else rng
    c = np.zeros(6) if init is None else np.asarray(init, float).copy()
    pr = np.zeros(6) if prior is None else np.asarray(prior, float)
    sig = np.asarray(sig0, float).copy()
    best_s, best = -1e18, c.copy()
    best_in = best_neg = best_out = 0.0
    pop = None
    for it in range(iters):
        D = rng.normal(size=(n_hyp, 6)) * sig
        D[0] = 0.0
        X = c[None] + D
        H = dict(dx=X[:, 0], dy=X[:, 1], dz=X[:, 2], psi=X[:, 3], th=X[:, 4], ph=X[:, 5])
        s, inl, neg, out = _score(P, label, H)
        s = s - 0.5 * LAMBDA * (((X - pr) / SIG_PRIOR) ** 2).sum(1)
        s = np.where((np.abs(X) > bounds).any(1), -1e9, s)
        j = int(np.argmax(s))
        if s[j] > best_s:
            best_s, best = float(s[j]), X[j].copy()
            best_in, best_neg, best_out = float(inl[j]), float(neg[j]), float(out[j])
        c = best.copy()
        sig = sig * 0.62
        pop = (X, s)
    X, s = pop
    top = np.argsort(-s)[:max(8, n_hyp // 12)]
    spread = np.array([X[top, 0].std(), X[top, 1].std(), X[top, 2].std(), X[top, 3].std()])
    return best, best_s, best_in, best_neg, best_out, spread


def estimate_tray(frame, prior_p, prior_R, mode="depth", tray_geoms=None, seed=0, cfg=None, z_weight=40.0, bounds_xy=None, z_lo=None, bounds_z=None):
    """frame: WristFrame.  prior_p/prior_R: PRIOR tray pose (world).  mode: 'priv' | 'depth'."""
    t0 = time.perf_counter()
    prior_p, prior_R = np.asarray(prior_p, float), np.asarray(prior_R, float)
    prior_R = level_R(prior_R)
    rng = np.random.default_rng(seed)
    if len(frame.pts) == 0:
        return PoseEst(False, seg=mode, ms=(time.perf_counter() - t0) * 1e3)
    P = (frame.pts - prior_p) @ prior_R
    cx, cy, zlo, zhi = CROP
    keep = (np.abs(P[:, 0]) < cx) & (np.abs(P[:, 1]) < cy) & (P[:, 2] > zlo) & (P[:, 2] < zhi)
    P = P[keep]
    if len(P) < 30:
        return PoseEst(False, n_pts=len(P), seg=mode, ms=(time.perf_counter() - t0) * 1e3)
    init = np.zeros(6)
    label = None
    if mode == "priv":
        label = np.isin(frame.geom[keep], tray_geoms)
        if label.sum() < 15:
            return PoseEst(False, n_pts=len(P), seg=mode, ms=(time.perf_counter() - t0) * 1e3)
        Q = P[label]
        # plane of the labelled points initialises z / pitch / roll (2 trimmed LS passes)
        A = np.c_[Q[:, :2], np.ones(len(Q))]
        abc = np.linalg.lstsq(A, Q[:, 2], rcond=None)[0]
        for _ in range(2):
            ok = np.abs(Q[:, 2] - A @ abc) < 0.006
            if ok.sum() < 10:
                break
            abc = np.linalg.lstsq(A[ok], Q[ok, 2], rcond=None)[0]
        init[2] = abc[2]                   # height only: the tilt is fixed to the level prior (see BOUNDS)
    else:
        abc = ransac_plane(P, rng, z_weight=z_weight, z_lo=z_lo)    # z_weight high: do not lock onto the 12 mm lower counter slab when it fills more of the view than the plate
        if abc is not None:
            init[2] = abc[2]
    if len(P) > MAX_PTS:                                   # cap the search cost: stride subsample (the counts are rescaled below)
        sub = np.sort(rng.choice(len(P), MAX_PTS, replace=False))
        scale = len(P) / MAX_PTS
        P, label = P[sub], (None if label is None else label[sub])
    else:
        scale = 1.0
    bnd = BOUNDS.copy()
    if bounds_xy is not None:             # the assist's capture range (+ margin): a hypothesis footprint at the edge of the old 7 cm box has its 'ring' outside the cropped data,
        bnd[:2] = bounds_xy               # so a plane lock on the (12 mm lower, larger) counter slab went unpenalised
    if bounds_z is not None:              # ... and the rectangle may not slide down onto the counter either (the hands expect the plate height to a few mm)
        bnd[2] = bounds_z
    init = np.clip(init, -bnd * 0.9, bnd * 0.9)
    best, sc, n_in, n_neg, n_out, spread = fit_rectangle(P, label, init=init, rng=rng, iters=9, n_hyp=220, bounds=bnd)
    n_in, n_neg, n_out = n_in * scale, n_neg * scale, n_out * scale
    R_h = _Rzyx(best[3], best[4], best[5])
    p = prior_p + prior_R @ best[:3]
    R = prior_R @ R_h
    # confidence: enough plate evidence and a well-localised fit
    ev = min(1.0, n_in / 150.0)
    loc = math.exp(-float(np.hypot(spread[0], spread[1])) / 0.012) * math.exp(-spread[3] / math.radians(4.0))
    neg_pen = 1.0 / (1.0 + max(0.0, n_neg - 12.0) / 30.0)
    expl = n_in / max(n_in + n_out, 1.0)                                   # fraction of the plate-level points the rectangle explains
    at_bound = bool((np.abs(best) > 0.95 * bnd).any())
    conf = float(ev * loc * neg_pen * expl ** 2 * (0.3 if at_bound else 1.0))
    return PoseEst(True, p, R, conf, n_in, n_neg, len(P), spread, sc, (time.perf_counter() - t0) * 1e3, n_out, mode, best)


def level_R(R):
    """rotation about the world z axis by the yaw of R's x axis (tilt removed): the tray's gravity-aligned prior orientation."""
    psi = math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose_error(p_est, R_est, p_gt, R_gt):
    """(position error [mm] 3-vector in the GT tray frame, norm, rotation error [deg])"""
    e = R_gt.T @ (np.asarray(p_est) - np.asarray(p_gt))
    Rr = R_gt.T @ R_est
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rr) - 1) / 2))))
    return e * 1e3, float(np.linalg.norm(e) * 1e3), ang
