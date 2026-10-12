"""people_track/detectors.py -- person detectors on the robot's simulated head camera.  SIMULATION ONLY.

All detectors return a list of tracker.Det (world xy of the person's AXIS, 2x2 covariance, #points).

(a) SegDetector   PRIVILEGED upper bound: the MuJoCo segmentation (person geom ids of the hit pixels) selects the person's points.  Depth noise,
                  occlusion by walls/furniture/robot, FOV and range are all real; only the "which pixels are a person" step is free.
(b) DepthDetector NON-privileged, depth only: noisy world points -> drop the floor (z < z_floor) and anything above z_ceil -> drop points that lie on the
                  static occupancy map (EDT(x,y) < d0 + k*sigma_z(range): walls/furniture; the map is `sim.occ`, the shield's own map; optional) and BACKGROUND SUBTRACTION against the prior 3-D house model (the same rays re-cast without people; the
                  robot is localised in a known house) -> 2-D bin
                  (cell `bin_m`) -> connected components -> keep components that look like a standing person (>= min_pts points, footprint diameter in
                  [w_min, w_max], top of the visible points above z_top_min) -> centroid.
(c) ParametricDetector  truth + N(0, sigma^2) noise, frame dropout p, detections gated by the REAL camera FOV (frustum) and line of sight
                  (rays against walls / furniture; the robot itself is ignored).  For sensitivity sweeps (latency lives in perception.py).

Both camera detectors place the person AXIS behind the visible surface: the camera sees the front of a vertical cylinder of radius r, the mean
depth of that front surface is (pi/4) r from the axis, so the axis = mean point + (pi/4) r along the camera->point direction on the ground plane
(r = scene.PERSON_R = 0.22: a person is a capsule, assumed known = a person-size prior, NOT ground truth of any particular walker).  When >= 8 points lie
on the torso (0.30 < z < 1.45) a fixed-radius circle fit (ridge toward that guess) replaces the centroid rule: it stays correct for partial views.

Detection covariance (camera detectors): along the viewing ray  sigma_r^2 = sigma_z(zbar)^2 / n_eff + sigma_model^2 (sigma_model 0.05 m: cylinder-offset
and partial-body bias), across it sigma_l^2 = (sigma_l0 + 0.5 * sigma_pix_range)^2 (sigma_l0 = 0.04, one pixel at that range).  n_eff = min(n_pts, 16)
(neighbouring pixels share the same stereo error).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy import ndimage

from world_model.v2.house.teleop import scene as SC
from .tracker import Det

PERSON_R = SC.PERSON_R
AXIS_OFFSET = (math.pi / 4.0) * PERSON_R


def _det_cov(cam, origin, pts_xy, zbar, n, trunc=False):
    u = pts_xy - origin[:2]
    rng = float(np.linalg.norm(u)) + 1e-9
    u = u / rng
    sig_r2 = float(cam.spec.sigma_z(zbar)) ** 2 / max(min(n, 16), 1) + 0.05 ** 2
    pix = rng / cam.spec.f_px
    sig_l2 = (0.04 + 0.5 * pix) ** 2
    if trunc:                                    # cluster cut by the image border: only part of the body is seen, the centroid is biased
        sig_l2 += 0.12 ** 2
        sig_r2 += 0.08 ** 2
    Rm = np.array([[u[0], -u[1]], [u[1], u[0]]])
    return Rm @ np.diag([sig_r2, sig_l2]) @ Rm.T


def _circle_fit(P, c0, R=PERSON_R, lam=0.5, iters=8):
    """fixed-radius circle (the person's cylinder cross-section) fitted to the visible-surface points: Gauss-Newton on r_i = |p_i - c| - R with a
    ridge toward the centroid+offset guess c0.  Robust to PARTIAL views (image border, occlusion) where the centroid is biased."""
    c = c0.copy()
    for _ in range(iters):
        d = P - c
        r = np.linalg.norm(d, axis=1) + 1e-9
        U = d / r[:, None]
        res = r - R
        w = 1.0 / np.maximum(1.0, np.abs(res) / 0.05)                  # Huber-like down-weighting of outliers
        A = (U * w[:, None]).T @ U + lam * np.eye(2)
        step = np.linalg.solve(A, (U * w[:, None]).T @ res - lam * (c - c0))
        c = c + step
        if np.linalg.norm(step) < 1e-4:
            break
    return c


def _axis_from_points(cam, origin, P, pix=None):
    """P: (n,3) points on the person's visible surface -> (axis xy, cov).  Centroid + (pi/4) r along the ray (always), refined by a fixed-radius
    circle fit on the torso-height points when there are >= 8 of them (and the fit stays within 0.35 m of the centroid guess)."""
    c = P[:, :2].mean(0)
    u = c - origin[:2]
    nrm = np.linalg.norm(u) + 1e-9
    axis = c + AXIS_OFFSET * u / nrm
    Pz = P[(P[:, 2] > 0.30) & (P[:, 2] < 1.45)]
    if len(Pz) >= 8:
        c2 = _circle_fit(Pz[:, :2], axis)
        if np.linalg.norm(c2 - axis) < 0.35:
            axis = c2
    zbar = float(np.mean(np.linalg.norm(P - origin, axis=1)))
    trunc = False
    if pix is not None and len(pix):
        W, H = cam.spec.W, cam.spec.H
        trunc = bool(pix[:, 1].min() <= 1 or pix[:, 1].max() >= W - 2 or pix[:, 0].min() <= 1 or pix[:, 0].max() >= H - 2)
    return axis, _det_cov(cam, origin, c, zbar, len(P), trunc)


class SegDetector:
    kind = "seg"

    def __init__(self, min_pts=4):
        self.min_pts = min_pts

    def detect(self, frame, cam, sim, **_):
        dets = []
        for pi in np.unique(frame["person"][frame["person"] >= 0]):
            msk = frame["person"] == pi
            if msk.sum() < self.min_pts:
                continue
            axis, cov = _axis_from_points(cam, frame["origin"], frame["pts"][msk], frame["pix"][msk])
            dets.append(Det(axis, cov, int(msk.sum()), truth=int(pi), kind=self.kind))
        return dets


class DepthDetector:
    kind = "depth"

    def __init__(self, z_floor=0.10, z_ceil=2.1, d0=0.06, k_sig=3.0, bin_m=0.10, min_pts=6, w_min=0.12, w_max=0.9, z_top_min=0.25,
                 occ=None, use_static_map=True, r_max=4.5, bg=True, bg_k=3.0, bg_m=0.06):
        self.z_floor, self.z_ceil, self.d0, self.k_sig, self.bin_m = z_floor, z_ceil, d0, k_sig, bin_m
        self.min_pts, self.w_min, self.w_max, self.z_top_min = min_pts, w_min, w_max, z_top_min
        self.occ = occ
        self.use_static_map = use_static_map
        self.r_max = r_max
        self.bg, self.bg_k, self.bg_m = bg, bg_k, bg_m
        self.last_n_static_rejected = 0

    def detect(self, frame, cam, sim, **_):
        P = frame["pts"]
        if len(P) == 0:
            return []
        z = P[:, 2]
        keep = (z > self.z_floor) & (z < self.z_ceil)
        occ = self.occ if self.occ is not None else getattr(sim, "occ", None)
        self.last_n_static_rejected = 0
        if self.bg and keep.any():
            # background subtraction against the PRIOR house model: re-cast the same rays against the static model (no people) and drop
            # every return that lies at the static surface (|z - z_static| < bg_k sigma_z + bg_m).  A person is in FRONT of the static surface.
            zs = cam.static_depth(frame, np.flatnonzero(keep))
            sig = cam.spec.sigma_z(frame["z"][keep])
            stat = np.isfinite(zs) & (np.abs(frame["z"][keep] - zs) < self.bg_k * sig + self.bg_m)
            ki = np.flatnonzero(keep)
            self.last_n_static_rejected += int(stat.sum())
            keep[ki[stat]] = False
        if self.use_static_map and occ is not None:
            sig = cam.spec.sigma_z(frame["z"])
            edt = occ.lookup(P[:, 0], P[:, 1])
            stat = edt < (self.d0 + self.k_sig * sig)
            self.last_n_static_rejected = int((keep & stat).sum())
            keep &= ~stat
        P = P[keep]
        pers = frame["person"][keep]
        pix_k = frame["pix"][keep]
        if len(P) < self.min_pts:
            return []
        b = np.floor(P[:, :2] / self.bin_m).astype(int)
        b0 = b.min(0)
        bi = b - b0
        grid = np.zeros(tuple(bi.max(0) + 3), bool)
        grid[bi[:, 0] + 1, bi[:, 1] + 1] = True
        lab, nl = ndimage.label(grid, structure=np.ones((3, 3)))
        lp = lab[bi[:, 0] + 1, bi[:, 1] + 1]
        dets = []
        for c in range(1, nl + 1):
            msk = lp == c
            n = int(msk.sum())
            if n < self.min_pts:
                continue
            Pc = P[msk]
            ext = Pc[:, :2].max(0) - Pc[:, :2].min(0)
            diam = float(np.hypot(*ext))
            if diam > self.w_max * 1.4 or max(ext) > self.w_max or Pc[:, 2].max() < self.z_top_min:
                continue
            if diam < self.w_min:
                continue
            if float(np.mean(np.linalg.norm(Pc - frame["origin"], axis=1))) > self.r_max:     # beyond the range where sigma_z (~5 cm at 4 m) separates a person from a wall
                continue
            axis, cov = _axis_from_points(cam, frame["origin"], Pc, pix_k[msk])
            pc = pers[msk]
            truth = int(np.bincount(pc[pc >= 0]).argmax()) if (pc >= 0).sum() > 0.5 * n else -1      # EVALUATION ONLY
            dets.append(Det(axis, cov, n, truth=truth, kind=self.kind))
        return dets


class Visibility:
    """ground-truth visibility of the sim's people for the metrics and for the parametric detector: FOV (frustum, any of 3 torso heights) and
    line of sight (rays against walls / furniture / floor; the robot's own geoms are skipped through)."""

    HEIGHTS = (0.45, 0.9, 1.35)

    def __init__(self, cam):
        self.cam = cam
        self._mask = np.zeros(6, np.uint8); self._mask[[3, 4]] = 1

    def los_clear(self, sim, origin, P):
        m, d = sim.m, sim.d
        v = P - origin
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            return True
        v = v / L
        o = origin.copy()
        travelled = 0.0
        gid = np.zeros(1, np.int32)
        for _ in range(6):
            dist = mujoco.mj_ray(m, d, o, v, self._mask, 1, -1, gid)
            if gid[0] < 0 or dist < 0:
                return True
            if travelled + dist >= L - PERSON_R:
                return True
            if self.cam.self_geom[gid[0]]:
                o = o + v * (dist + 2e-3)
                travelled += dist + 2e-3
                continue
            return False
        return True

    def of(self, sim, xy):
        """-> (in_fov, visible, range_m)."""
        origin, R = self.cam.pose()
        pts = np.array([[xy[0], xy[1], h] for h in self.HEIGHTS])
        fov = self.cam.in_fov(pts, origin, R)
        rng = float(np.hypot(xy[0] - origin[0], xy[1] - origin[1]))
        if not fov.any():
            return False, False, rng
        for p, f in zip(pts, fov):
            if f and self.los_clear(sim, origin, p):
                return True, True, rng
        return True, False, rng


class ParametricDetector:
    kind = "param"

    def __init__(self, sigma=0.10, p_drop=0.0, seed=0):
        self.sigma, self.p_drop = float(sigma), float(p_drop)
        self.rng = np.random.default_rng(1234 + seed)

    def detect(self, frame, cam, sim, vis=None, truth=None, **_):
        dets = []
        for i, (xy, _v) in enumerate(truth):
            if not vis[i][1]:
                continue
            if self.rng.random() < self.p_drop:
                continue
            z = np.asarray(xy, float) + self.rng.normal(0, self.sigma, 2)
            dets.append(Det(z, np.eye(2) * max(self.sigma, 0.02) ** 2, 0, truth=i, kind=self.kind))
        return dets
