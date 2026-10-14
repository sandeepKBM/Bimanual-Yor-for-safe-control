"""people_track/detect_v2.py -- people detectors WITHOUT the two privileged shortcuts of detectors.DepthDetector.  SIMULATION ONLY.

detectors.DepthDetector reads the TRUE house model (background = the same rays re-cast through the exact static MuJoCo model from the TRUE camera
pose) and puts detections in the world frame with the TRUE camera pose (cam.pose()).  Here:

  PerceivedDepthDetector   * world frame: the camera pose is the TRUE pose composed with the (estimated - true) base-pose correction, i.e. the camera
                             is mounted at the ESTIMATED base pose (sim.loc.pose, house/loc PoseEstimator; known mount + lift): pts = o_est + R_est v_cam
                           * background: rays from the ESTIMATED camera pose marched through the PERCEIVED map (house/loc PerceivedMap 3-D log-odds from
                             a head-ZED mapping pass), predicted depth = first occupied voxel.  A return is "static" when |z - z_pred| < k sigma_z +
                             m_bg + map_m + loc_k * (sigma_xy + r * tan(sigma_yaw))   (pose-uncertainty-inflated, the estimate is not exact)
                           * optional static filter uses the PERCEIVED occupancy grid (sim.occ_ctrl), not sim.occ
                           * the map only contains the obstacle band z in [obs_zlo, obs_zhi] (0.15..1.15 m): candidate points are limited to that band
                             (floor and everything above the map's ceiling are not modelled)
                           Truth enters ONLY as evaluation labels (frame["person"]).  Detections are in the ESTIMATED world frame (as the shield sees
                           them: it uses ctrl_pose).
  NeuralDetector           * map-free: head-camera RGB (MuJoCo EGL render at the head pose) -> OUT-OF-PROCESS COCO person detector (detect_server.py,
                             unix socket) -> boxes -> median sensor depth in the box -> 3-D axis (pose-estimate world frame) -> the existing tracker.
"""
from __future__ import annotations

import math
import os
import socket
import struct
import subprocess
import time

import mujoco
import numpy as np
from numba import njit

from . import detectors as DET
from .tracker import Det


# ------------------------------------------------------------------------------------------------ pose-estimate world frame
def est_correction(sim):
    """planar rigid map true-base-frame -> estimated-base-frame: p_est = Rz(dyaw) (p - b_true) + b_est."""
    bt = sim.pose()
    be = sim.loc.pose if getattr(sim, "loc", None) is not None else bt
    dy = float(be[2] - bt[2])
    return bt[:2].copy(), np.asarray(be[:2], float).copy(), dy


def to_est_frame(frame, sim):
    """frame (true-pose world points) -> frame2 with points / camera origin / rotation re-expressed with the ESTIMATED base pose (camera rigidly
    mounted on the base: known extrinsics + lift)."""
    bt, be, dy = est_correction(sim)
    c, s = math.cos(dy), math.sin(dy)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def mp(P):
        P = np.atleast_2d(P).copy()
        P[:, :2] = (P[:, :2] - bt) @ Rz[:2, :2].T + be
        return P
    f2 = dict(frame)
    f2["pts"] = mp(frame["pts"])
    f2["origin"] = mp(frame["origin"])[0]
    f2["R"] = Rz @ frame["R"]
    return f2


def _rz_cam(origin, R, corr):
    """camera origin / rotation re-expressed with the estimated base pose (corr = est_correction(sim))."""
    bt, be, dy = corr
    c, s = math.cos(dy), math.sin(dy)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    o = np.asarray(origin, float).copy()
    o[:2] = Rz[:2, :2] @ (o[:2] - bt) + be
    return o, Rz @ R


def loc_sigmas(sim):
    """(sigma_xy, sigma_yaw) the localizer would report (model-based proxy; the yaw one from the same drift model: sig_drift_yaw sqrt(dist))."""
    loc = sim.loc
    if loc is None:
        return 0.0, 0.0
    c = loc.cfg
    return float(loc.sigma_xy), float(0.005 + c.vio_drift_yaw * math.sqrt(loc.dist_since_reloc))


# ------------------------------------------------------------------------------------------------ background prediction from the perceived map
@njit(cache=True)
def _march(occ, res, x0, y0, zlo, dz, o, dirs, cth, zmax, step, out):
    nz, ny, nx = occ.shape
    n = dirs.shape[0]
    nstep = int(zmax / step)
    for i in range(n):
        out[i] = np.inf
        dx, dy, dz_ = dirs[i, 0], dirs[i, 1], dirs[i, 2]
        for k in range(1, nstep):
            t = k * step
            ix = int((o[0] + dx * t - x0) / res)
            iy = int((o[1] + dy * t - y0) / res)
            if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
                continue
            z = o[2] + dz_ * t
            iz = int((z - zlo) / dz)
            if z < zlo or iz >= nz:
                continue
            if occ[iz, iy, ix]:
                out[i] = t * cth[i]
                break


class MapBackground:
    """expected z-depth of the STATIC scene as perceived (PerceivedMap 3-D log-odds, 5 cm height bins) along arbitrary rays from an arbitrary origin."""

    def __init__(self, pmap, step=0.03, min_hits=1):
        cfg = pmap.cfg
        self.res, self.x0, self.y0 = float(pmap.res), float(pmap.x0), float(pmap.y0)
        self.zlo, self.zhi, self.dz = float(cfg.get("obs_zlo", 0.15)), float(cfg.get("obs_zhi", 1.15)), float(cfg.get("dz", 0.05))
        self.occ = np.ascontiguousarray(pmap.logodds > cfg.get("occ_thresh", 0.5))
        self.step = step
        self.pmap = pmap

    def depth(self, origin, dirs_world, cth, zmax):
        out = np.empty(len(dirs_world))
        _march(self.occ, self.res, self.x0, self.y0, self.zlo, self.dz, np.asarray(origin, float), np.ascontiguousarray(dirs_world, float),
               np.ascontiguousarray(cth, float), float(zmax), self.step, out)
        return out


class _CamProxy:
    """HeadCamera with static_depth replaced by the perceived-map prediction from the ESTIMATED pose."""

    def __init__(self, cam, bg):
        self._cam, self._bg = cam, bg

    def __getattr__(self, n):
        return getattr(self._cam, n)

    def static_depth(self, frame, idx):
        cam = self._cam
        rc = frame["pix"][idx]
        dirs = cam.dirs[rc[:, 0], rc[:, 1]]
        cth = cam.cos_th[rc[:, 0], rc[:, 1]]
        dw = dirs @ frame["R"].T
        return self._bg.depth(frame["origin"], dw, cth, cam.spec.z_max * 1.2)


class PerceivedDepthDetector(DET.DepthDetector):
    kind = "depth_perc"

    def __init__(self, pmap, occ=None, use_static_map=False, z_floor=None, z_ceil=None, map_m=0.04, loc_k=2.0, loc_margin=True, bg_m=0.06, **kw):
        bg = MapBackground(pmap)
        zf = bg.zlo if z_floor is None else z_floor
        zc = bg.zhi if z_ceil is None else z_ceil
        super().__init__(z_floor=zf, z_ceil=zc, occ=occ, use_static_map=use_static_map, bg=True, bg_m=bg_m, **kw)
        self.bgm = bg
        self.map_m, self.loc_k, self.loc_margin = map_m, loc_k, loc_margin
        self._bg_m0 = bg_m
        self.last_margin = bg_m

    def foreground(self, frame, cam, sim, f2=None):
        """boolean mask over frame['pts']: the points that survive the foreground tests of detect() (z band, perceived-map background subtraction with the
        pose-uncertainty margin, optional perceived-grid filter) -- the evidence for 'something non-static is here'."""
        f2 = to_est_frame(frame, sim) if f2 is None else f2
        sxy, syaw = loc_sigmas(sim)
        rmean = float(np.mean(frame["z"])) if len(frame["z"]) else 2.0
        extra = self.map_m + (self.loc_k * (sxy + rmean * math.tan(min(syaw, 0.3))) if self.loc_margin else 0.0)
        P = f2["pts"]
        z = P[:, 2]
        keep = (z > self.z_floor) & (z < self.z_ceil)
        if keep.any():
            ki = np.flatnonzero(keep)
            zs = _CamProxy(cam, self.bgm).static_depth(f2, ki)
            sig = cam.spec.sigma_z(frame["z"][keep])
            stat = np.isfinite(zs) & (np.abs(frame["z"][keep] - zs) < self.bg_k * sig + self._bg_m0 + extra)
            keep[ki[stat]] = False
        if self.use_static_map:
            occ = self.occ if self.occ is not None else getattr(sim, "occ_ctrl", None)
            if occ is not None:
                sig = cam.spec.sigma_z(frame["z"])
                keep &= ~(occ.lookup(P[:, 0], P[:, 1]) < (self.d0 + self.k_sig * sig))
        return keep

    def detect(self, frame, cam, sim, **_):
        f2 = to_est_frame(frame, sim)
        sxy, syaw = loc_sigmas(sim)
        rmean = float(np.mean(frame["z"])) if len(frame["z"]) else 2.0
        extra = self.map_m + (self.loc_k * (sxy + rmean * math.tan(min(syaw, 0.3))) if self.loc_margin else 0.0)
        self.bg_m = self._bg_m0 + extra
        self.last_margin = self.bg_m
        if self.occ is None and self.use_static_map:
            self.occ = getattr(sim, "occ_ctrl", None)      # the PERCEIVED grid (control-path map), not sim.occ
        return super().detect(f2, _CamProxy(cam, self.bgm), sim)


# ------------------------------------------------------------------------------------------------ neural detector (client side)
class DetectorClient:
    """unix-socket client of detect_server.py.  `start()` launches the server in the detect env and BLOCKS until it signals READY (warm-up done)."""

    DPY = "/data/local/ss5772/conda_envs/detect/bin/python"
    SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect_server.py")

    def __init__(self, sock):
        self.sock_path = sock
        self.proc = None
        self.s = None

    def start(self, res="320x180,640x360", model="frcnn_v2", gpu="5", warm=20, timeout=600, log=None):
        for f in (self.sock_path, self.sock_path + ".ready"):
            if os.path.exists(f):
                os.unlink(f)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TORCH_HOME="/data/local/ss5772/models/torch", OMP_NUM_THREADS="2",
                   PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        lf = open(log, "w") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen([self.DPY, self.SERVER, "--sock", self.sock_path, "--res", res, "--model", model, "--warm", str(warm)],
                                     env=env, stdout=lf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while not os.path.exists(self.sock_path + ".ready"):
            if self.proc.poll() is not None:
                raise RuntimeError(f"detect server died (rc={self.proc.returncode}); see {log}")
            if time.time() - t0 > timeout:
                raise TimeoutError("detect server not ready")
            time.sleep(0.2)
        self.connect()
        return open(self.sock_path + ".ready").read().strip()

    def connect(self):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(self.sock_path)
        self.s.sendall(b"P")
        assert self._recv(5) == b"READY"

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            b = self.s.recv(n - len(buf))
            if not b:
                raise ConnectionError
            buf += b
        return bytes(buf)

    def infer(self, rgb):
        H, W, _ = rgb.shape
        a = np.ascontiguousarray(rgb, np.uint8)
        self.s.sendall(b"F" + struct.pack("<HHI", W, H, a.nbytes) + a.tobytes())
        n, ms = struct.unpack("<If", self._recv(8))
        bx = np.frombuffer(self._recv(20 * n), "<f4").reshape(n, 5).copy() if n else np.zeros((0, 5), np.float32)
        return bx, ms

    def close(self):
        """disconnect only (the server is shared by every worker)."""
        try:
            if self.s:
                self.s.close()
        except OSError:
            pass

    def shutdown(self):
        """ask the server to exit (and reap it when this client launched it)."""
        try:
            if self.s:
                self.s.sendall(b"Q")
                self.s.close()
        except OSError:
            pass
        if self.proc:
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class HeadRGB:
    """MuJoCo EGL render of the sim at the head-camera pose (same pose / fov as the depth sensor)."""

    def __init__(self, sim, cam, W=640, H=360):
        self.sim, self.cam, self.W, self.H = sim, cam, W, H
        m = sim.m
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
        self.r = mujoco.Renderer(m, H, W)
        self.last_ms = 0.0

    def render(self):
        m = self.sim.m
        t0 = time.perf_counter()
        o, R = self.cam.pose()
        g = m.vis.global_.fovy
        m.vis.global_.fovy = self.cam.spec.fovy
        try:
            self.r.update_scene(self.sim.d)
        finally:
            m.vis.global_.fovy = g
        sc = self.r.scene
        for e in range(2):
            c = sc.camera[e]
            c.pos[:] = o
            c.forward[:] = -R[:, 2]
            c.up[:] = R[:, 1]
        img = self.r.render()
        self.last_ms = (time.perf_counter() - t0) * 1000
        return img


class BoxSource:
    """RGB render + out-of-process inference ONCE per sensor frame (cached on frame['tick']); shared by every neural / fusion variant on the same sim."""

    def __init__(self, client: DetectorClient, rgb: HeadRGB, thr=0.5):
        self.cl, self.rgb, self.thr = client, rgb, thr
        self._tick, self._out = None, None

    def get(self, frame, sim):
        if self._tick == frame.get("tick") and self._out is not None:
            return self._out
        img = self.rgb.render()
        bx, srv_ms = self.cl.infer(img)
        bx = bx[bx[:, 4] >= self.thr]
        self._tick, self._out = frame.get("tick"), (img, bx, srv_ms, self.rgb.last_ms)
        return self._out


def shared_box_source(sim, cam, sock, res=(640, 360), thr=0.5):
    """one BoxSource per (sim, socket, resolution): the stack of every detector variant of a run shares the single inference per frame."""
    key = ("_box_source", sock, tuple(res), thr)
    src = sim.__dict__.get("_box_sources", {}).get(key)
    if src is None:
        cl = DetectorClient(sock)
        cl.connect()
        src = BoxSource(cl, HeadRGB(sim, cam, *res), thr)
        sim.__dict__.setdefault("_box_sources", {})[key] = src
    return src


class NeuralDetector:
    """boxes (BoxSource) -> median sensor depth in the box -> 3-D axis.
    centroid: 'points'  box pixels at the median depth (+-band) -> detectors._axis_from_points (visible-surface centroid + (pi/4) r_prior offset, circle fit)
              'surf'    same points, axis = visible-surface centroid + r_off along the viewing ray  (r_off = the person's half-depth; NO r=0.22 prior)
              'ray'     ray through the box centre at the median depth + (pi/4) r_prior
    fusion:   a PerceivedDepthDetector -> only the depth points that survive ITS foreground test (z band, perceived-map background, perceived-grid filter)
              AND lie inside the neural box form the cluster (neural boxes gate the depth clusters; box-only evidence on static clutter is dropped)."""
    kind = "neural"

    def __init__(self, source: BoxSource, centroid="points", r_off=0.10, band=0.45, r_max=5.0, min_pts=4, fusion=None, name=None):
        self.src, self.centroid, self.r_off, self.band, self.r_max, self.min_pts, self.fusion = source, centroid, r_off, band, r_max, min_pts, fusion
        self.last = dict(n_boxes=0, n_no_depth=0, srv_ms=0.0, render_ms=0.0, boxes=[], scores=[])
        self.kind = name or ("fusion" if fusion is not None else "neural")

    def detect(self, frame, cam, sim, **_):
        img, bx, srv_ms, rms = self.src.get(frame, sim)
        H, W = img.shape[:2]
        sp = cam.spec
        sx, sy = sp.W / W, sp.H / H
        f2 = to_est_frame(frame, sim)
        pix = frame["pix"]
        zs = frame["z"]
        fg = None
        if self.fusion is not None:
            fg = self.fusion.foreground(frame, cam, sim, f2)
        dets, nod = [], 0
        boxes, scores = [], []
        for x1, y1, x2, y2, sc in bx:
            boxes.append((x1, y1, x2, y2)); scores.append(float(sc))
            w, h = x2 - x1, y2 - y1
            cx = (pix[:, 1] + 0.5) / sx
            cy = (pix[:, 0] + 0.5) / sy
            inb = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
            if fg is not None:
                inb &= fg
            core = (cx >= x1 + 0.2 * w) & (cx <= x2 - 0.2 * w) & (cy >= y1 + 0.2 * h) & (cy <= y1 + 0.7 * h) & (fg if fg is not None else True)
            if core.sum() < 2:
                core = inb
            if core.sum() < 2:
                nod += 1
                continue
            zmed = float(np.median(zs[core]))
            sel = inb & (np.abs(zs - zmed) < self.band)
            P = f2["pts"][sel]
            org = f2["origin"]
            if fg is not None and len(P) < self.min_pts:
                nod += 1
                continue
            if len(P) < self.min_pts or self.centroid == "ray":
                u, v = 0.5 * (x1 + x2) * sx - 0.5, 0.5 * (y1 + y2) * sy - 0.5
                d = np.array([(u + 0.5 - sp.W / 2) / sp.f_px, -(v + 0.5 - sp.H / 2) / sp.f_px, -1.0])
                d /= np.linalg.norm(d)
                pw = org + f2["R"] @ d * (zmed / -d[2])
                un = (pw[:2] - org[:2]) / (np.linalg.norm(pw[:2] - org[:2]) + 1e-9)
                axis = pw[:2] + DET.AXIS_OFFSET * un
                cov = DET._det_cov(cam, org, pw[:2], zmed, 1, False)
            elif self.centroid == "surf":
                c = P[:, :2].mean(0)
                un = (c - org[:2]) / (np.linalg.norm(c - org[:2]) + 1e-9)
                axis = c + self.r_off * un
                _, cov = DET._axis_from_points(cam, org, P, pix[sel])
            else:
                axis, cov = DET._axis_from_points(cam, org, P, pix[sel])
            if float(np.hypot(*(axis - org[:2]))) > self.r_max:
                continue
            pp = frame["person"][sel]
            truth = int(np.bincount(pp[pp >= 0]).argmax()) if (pp >= 0).sum() > 0.4 * max(sel.sum(), 1) else -1      # EVALUATION ONLY
            dets.append(Det(axis, cov, int(len(P)), truth=truth, kind=self.kind))
        self.last = dict(n_boxes=len(boxes), n_no_depth=nod, srv_ms=srv_ms, render_ms=rms, boxes=boxes, scores=scores, img=img)
        return dets


# ------------------------------------------------------------------------------------------------ drop-in perception stack (no edits of the existing stack)
def make_stack(sim, cfg=None, detector="depth_perc", pmap=None, client=None, rgb_res=(640, 360), score_thr=0.5, human=False, **kw):
    """PerceptionStack (people_track.perception) whose detector is swapped AFTER construction (kept for scripts; the stack itself now accepts
    cfg.detector in ('depth_perc', 'neural', 'fusion', 'full')).  NOT bit-identical to detector='depth' by design (that one reads the truth)."""
    from . import perception as PC
    from dataclasses import replace
    cfg = replace(cfg or PC.PerceptionCfg(), detector=detector)
    return PC.PerceptionStack(sim, cfg)


def detector_from_cfg(sim, cam, c):
    """PerceptionStack hook: detector for cfg.detector in ('depth_perc', 'neural', 'fusion').
    depth_perc: perceived-map background + estimated pose; the perceived 2-D grid filter is ON when cfg.static_map in ('occ', 'both').
    neural: box-depth axis (centroid from cfg.neural_centroid: points | surf | ray).   fusion: neural boxes gating depth_perc(+grid) foreground."""
    from world_model.v2.house.loc import mapper as MP
    grid = c.static_map in ("occ", "both")
    if c.detector == "depth_perc":
        return PerceivedDepthDetector(MP.PerceivedMap.load(MP.map_path(sim.scene)), use_static_map=grid, r_max=c.r_max)
    sock = c.neural_sock or os.environ.get("DETECT_SOCK", "")
    src = shared_box_source(sim, cam, sock, c.neural_res, c.neural_thr)
    kw = dict(centroid=c.neural_centroid, r_off=c.neural_r_off, r_max=c.r_max)
    if c.detector == "neural":
        return NeuralDetector(src, **kw)
    pd = PerceivedDepthDetector(MP.PerceivedMap.load(MP.map_path(sim.scene)), use_static_map=grid, r_max=c.r_max)
    return NeuralDetector(src, fusion=pd, **kw)
