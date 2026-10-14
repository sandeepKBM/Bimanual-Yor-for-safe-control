"""stereo/adapters.py -- make the stereo depth a DROP-IN source for the existing consumers (no edit to their files).  SIMULATION ONLY.

  frame_to_head_frame(fr, ...)   stereo Frame -> the `head_frame` dict that ocs.PeopleProducer / people_track.DepthDetector consume
                                 (tick, origin, R, base_true, pts, z, pix, zs, person): depth decimated to the tracker grid, robot pixels removed by the FK
                                 self-mask, `zs` = the static house model re-cast along the same rays (background subtraction against the prior map)
  stereo_spec(cfg, grid, noise)  the depth_sim.CamSpec the detectors read (f_px / sigma_z) for the stereo grid, noise parameters = the FITTED values
  StereoPeopleProducer           ocs.PeopleProducer with that spec (everything else, tracker, covariance, set uncertainty: unchanged)
  StereoHeadDepth                loc.mapper.HeadDepth with capture() backed by the stereo sensor on a house-only twin (same interface, same place())
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import mujoco

from .cfg import StereoCfg


# ====================================================================================== rays of the decimated grid
def grid_rays(cfg: StereoCfg, stride: int):
    """pixel rows / cols (in stereo pixels) of the decimated grid (centre sample of each stride x stride block) and the unit-z camera-frame ray scales."""
    rows = np.arange(stride // 2, cfg.H, stride)
    cols = np.arange(stride // 2, cfg.W, stride)
    K = cfg.K()
    uu, vv = np.meshgrid(cols.astype(np.float64), rows.astype(np.float64))
    x = (uu - K["cx"]) / K["f"]
    y = -(vv - K["cy"]) / K["f"]
    return rows, cols, x, y


def static_depth_grid(m, d, origin, R, x, y, mask=(3, 4)):
    """expected z-depth of the STATIC house model (people not cast) along the rays of the grid (inf = no hit); same cast people_track.camera.static_depth does."""
    dirs = np.stack([x, y, -np.ones_like(x)], -1).reshape(-1, 3)
    n = np.linalg.norm(dirs, axis=1)
    dw = np.ascontiguousarray((dirs / n[:, None]) @ R.T)
    gm = np.zeros(6, np.uint8)
    gm[list(mask)] = 1
    geom = np.full(len(dw), -1, np.int32)
    dist = np.full(len(dw), -1.0)
    mujoco.mj_multiRay(m, d, origin, dw.reshape(-1), gm, True, -1, geom, dist, None, len(dw), 8.0)
    z = np.where((geom >= 0) & (dist > 0), dist / n, np.inf)
    return z.reshape(x.shape)


def frame_to_head_frame(fr, cfg: StereoCfg, stride=4, sim=None, zs_grid=None, base_true=None, person_grid=None, z_lim=(0.10, 2.1)):
    """stereo Frame -> head_frame dict.  Pixels in the self-mask (robot / tray / cup, FK-known) are dropped; z (world) outside z_lim dropped (the rows the
    depth detector would drop anyway).  zs_grid: (h, w) static depth on the decimated grid (computed from `sim` when None).  person_grid: (h, w) int
    truth person index per grid cell (EVALUATION ONLY; -1 elsewhere)."""
    rows, cols, x, y = grid_rays(cfg, stride)
    o, R = fr["origin"], fr["R"]
    z = fr["depth"][np.ix_(rows, cols)].astype(np.float64)
    ok = np.isfinite(z)
    if "self_mask" in fr:
        ok &= ~fr["self_mask"][np.ix_(rows, cols)]
    if zs_grid is None and "static_depth" in fr:
        zs_grid = np.where(fr["static_depth"][np.ix_(rows, cols)] > 0, fr["static_depth"][np.ix_(rows, cols)], np.inf)
    if zs_grid is None:
        zs_grid = static_depth_grid(sim.m, sim.d, o, R, x, y)
    pc = np.stack([x * z, y * z, -z], -1)
    P = pc @ R.T + o
    ok &= (P[..., 2] > z_lim[0]) & (P[..., 2] < z_lim[1])
    ri, ci = np.nonzero(ok)
    person = np.full(len(ri), -1, np.int8) if person_grid is None else person_grid[ri, ci].astype(np.int8)
    return dict(tick=int(fr["tick"]), origin=o.copy(), R=R.copy(), base_true=np.asarray(base_true if base_true is not None else (0, 0, 0), float),
                pts=P[ri, ci].astype(np.float32), z=z[ri, ci].astype(np.float32), pix=np.stack([ri, ci], 1).astype(np.uint8),
                zs=np.asarray(zs_grid)[ri, ci].astype(np.float32), person=person)


def to_est_frame(hf, base_true, base_est):
    """re-place a head_frame captured with the TRUE base pose into the frame of an estimated base pose: rigid planar transform T_est o T_true^-1 on the points,
    the camera origin and the rotation (the operation ocs.producers._to_est_frame applies, here at CAPTURE time)."""
    bt, be = np.asarray(base_true, float), np.asarray(base_est, float)
    dyaw = be[2] - bt[2]
    c, s = math.cos(dyaw), math.sin(dyaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    o = dict(hf)
    o["pts"] = ((hf["pts"].astype(np.float64) - np.array([bt[0], bt[1], 0.0])) @ Rz.T + np.array([be[0], be[1], 0.0])).astype(np.float32)
    o["origin"] = (np.asarray(hf["origin"], float) - np.array([bt[0], bt[1], 0.0])) @ Rz.T + np.array([be[0], be[1], 0.0])
    o["R"] = Rz @ np.asarray(hf["R"], float)
    return o


def stereo_spec(cfg: StereoCfg, stride=4, sigma_disp_px=0.25, sigma_floor=5e-4, p_drop=0.02, p_out=0.002):
    """depth_sim.CamSpec of the DECIMATED grid: f_px = f / stride (grid focal), native_H = the stereo image height (f_native = the stereo f), noise = fitted."""
    from world_model.v2.house.people_track import camera as CAM
    gH, gW = len(range(stride // 2, cfg.H, stride)), len(range(stride // 2, cfg.W, stride))
    return replace(CAM.head_spec(), fovy=cfg.fovy, W=gW, H=gH, native_H=cfg.H, baseline=cfg.baseline, sigma_disp_px=sigma_disp_px, sigma_floor=sigma_floor,
                   z_min=cfg.z_min, z_max=cfg.z_max, p_drop=p_drop, p_out=p_out, name="zed_stereo")


MIN_PTS_STEREO = 40


def make_people_producer(spec, min_pts=MIN_PTS_STEREO, **kw):
    """ocs.PeopleProducer with the stereo grid spec (sigma_z / f_px used by the detector's background threshold and detection covariance) and a larger minimum
    cluster size: SGBM leaves scattered gross outliers (depth-edge flying pixels, repeated texture) that form 6-40-point clusters; a person has hundreds
    (min_pts 40 chosen on seeds 0-1: false clusters' 90th percentile = 27-42 pts, true detections keep 96 %; evaluated on held-out seeds 2-3).  min_pts=6 = the
    ray-cast default (people_track.DepthDetector)."""
    from world_model.v2.house.ocs.producers import PeopleProducer, _ReplayCam
    pp = PeopleProducer(**kw)
    pp.spec = spec
    pp.cam = _ReplayCam(spec)
    pp.det.min_pts = int(min_pts)
    return pp


# ====================================================================================== mapper
class StereoHeadDepth:
    """loc.mapper.HeadDepth drop-in whose capture() is a stereo render + SGBM of a house-only twin (`sensor` = StereoSensor(root=None)).
    Same outputs: (ok, pts_cam (fwd z, left, up of the CYCLOPEAN centre axes), nofree, axes); place() is the mapper's own.
    Invalid pixels are NOT free-space evidence (a stereo miss is not a no-return): nofree is all False."""

    def __init__(self, base_hd, sensor, stride=2):
        self.hd, self.s, self.stride = base_hd, sensor, stride
        cfg = sensor.cfg
        self.rows, self.cols, x, y = grid_rays(cfg, stride)
        self.x, self.y = x, y
        self.last = None
        self.ms = []

    def __getattr__(self, n):
        return getattr(self.hd, n)

    def capture(self, true_pose):
        hd, s = self.hd, self.s
        o, fwd, left, up = hd.axes(true_pose)
        R = np.stack([-left, up, -fwd], 1)                       # camera -> world (right, up, back)
        s.set_center_pose(o, R)
        fr = s.capture(0, 0.0)
        self.ms.append(fr["ms"])
        self.last = fr
        z = fr["depth"][np.ix_(self.rows, self.cols)].astype(np.float64).ravel()
        ok = np.isfinite(z) & (z <= hd.cfg.z_map_max)
        z = np.where(ok, z, 0.0)
        xs, ys = self.x.ravel(), self.y.ravel()
        # left-eye coordinates -> cyclopean axes: the left eye sits B/2 to the LEFT of the centre
        pts = np.stack([z, -xs * z + 0.5 * s.cfg.baseline, ys * z], 1)
        return ok, pts, np.zeros(len(z), bool), (o, fwd, left, up)
