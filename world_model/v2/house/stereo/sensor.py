"""stereo/sensor.py -- render the head-ZED stereo pair from a state snapshot, SGBM -> depth + validity, FK self-mask.  SIMULATION ONLY.

StereoSensor is the single implementation used (a) in-process on stored snapshots (calibration study, replay), (b) inside the separate process of link.py.
Input = a state snapshot (qpos, mocap_pos, mocap_quat, eq_active, sim time, tick); the camera pose is derived from the robot root body + the lift sliders
exactly like people_track.camera.HeadCamera.pose() (root o (mount + [0,0,lift]), pitched head), then the left / right eyes are offset +-B/2 along the image x axis.

Output Frame (dict)
  tick, t_sim            snapshot identity / time of the exposure
  depth (H,W) float32    z-depth along the left optical axis, NaN = invalid (no match / left-right check failed / outside [z_min, z_max])
  origin (3,), R (3,3)   left camera centre / camera-to-world rotation (MuJoCo: looks along -z, x right, y up)
  K                      dict(f, cx, cy, W, H, B)
  self_mask (H,W) bool   pixels showing the robot / tray / cup (FK-known geometry), dilated -- only when cfg.self_mask
  cls (H,W) uint8        0 other, 1 robot link, 2 tray, 3 cup / beads  (the FK-known class of the pixel; from a segmentation render of the KNOWN model)
  left (H,W,3) uint8     when cfg.keep_rgb
  gt_depth, gt_seg       EVALUATION ONLY (cfg.gt)
  ms                     dict(render, stereo (backend), post, total) wall ms of this capture
"""
from __future__ import annotations

import time

import numpy as np
import cv2
import mujoco

from .cfg import StereoCfg, head_axes

from .backends import make_backend

_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)       # NOT the enum object: comparing an ndarray with the enum is ~100x slower


def add_noise(img, rng, read_sigma, shot_k):
    f = img.astype(np.float32)
    sd = np.sqrt(read_sigma ** 2 + shot_k * f)
    return np.clip(f + rng.standard_normal(f.shape, dtype=np.float32) * sd, 0, 255).astype(np.uint8)


class StereoSensor:
    def __init__(self, m: mujoco.MjModel, cfg: StereoCfg, root: str | None = "yor_root", backend=None):
        cv2.setNumThreads(int(cfg.cv_threads))
        self.m, self.cfg = m, cfg
        self.d = mujoco.MjData(m)
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, cfg.W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, cfg.H)
        self.r = mujoco.Renderer(m, cfg.H, cfg.W)                 # RGB: anti-aliased (offsamples as in the model)
        self.r_seg = None
        if cfg.self_mask or cfg.gt or cfg.static_depth:           # segmentation ids are corrupted by multisampling on edge pixels: a separate no-AA context
            ns = m.vis.quality.offsamples
            m.vis.quality.offsamples = 0
            self.r_seg = mujoco.Renderer(m, cfg.H, cfg.W)
            m.vis.quality.offsamples = ns
        self.backend = backend or make_backend(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.root = m.body(root).id if root else None             # None: house-only twin, the caller sets the camera pose (set_center_pose)
        self.pose_override = None
        self.cam = {"l": m.camera("zed_left").id, "r": m.camera("zed_right").id}
        m.cam_fovy[list(self.cam.values())] = cfg.fovy
        self.Rc = head_axes(cfg)                                  # camera -> robot
        self.slider = [int(m.jnt_qposadr[m.joint(f"Slider_{i}").id]) for i in (1, 2)] if root else []
        self.K = cfg.K()
        # class of every geom: 1 robot link, 2 tray, 3 cup / bead (the robot's own known geometry), 0 anything else
        cls = np.zeros(m.ngeom, np.uint8)
        anc = np.zeros(m.nbody, bool)
        for b in range(m.nbody):
            p = b
            while p > 0 and self.root is not None:
                if p == self.root:
                    anc[b] = True
                    break
                p = int(m.body_parentid[p])
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            nm = m.body(b).name or ""
            if nm == "tray":
                cls[g] = 2
            elif nm == "cup" or nm.startswith("ball_"):
                cls[g] = 3
            elif anc[b]:
                cls[g] = 1
        self.geom_cls = cls
        self.person_geoms = {}
        for b in range(m.nbody):
            nm = m.body(b).name or ""
            if nm.startswith("p_"):
                for g in np.flatnonzero(m.geom_bodyid == b):
                    self.person_geoms[int(g)] = nm
        self.ms_hist = []

    # ------------------------------------------------------------------ state / pose
    def load(self, qpos, mocap_pos, mocap_quat, eq_active):
        d = self.d
        d.qpos[:] = qpos
        d.mocap_pos[:] = mocap_pos
        d.mocap_quat[:] = mocap_quat
        d.eq_active[:] = np.asarray(eq_active).astype(d.eq_active.dtype)
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_camlight(self.m, d)

    def lift(self):
        return float(sum(self.d.qpos[a] for a in self.slider))

    def poses(self):
        """-> (origin_l, origin_r, R) world poses of the two eyes from the loaded state."""
        d, c = self.d, self.cfg
        if self.pose_override is not None:
            o, R = self.pose_override
            x = R[:, 0]
            return o - 0.5 * c.baseline * x, o + 0.5 * c.baseline * x, R
        Rb = d.xmat[self.root].reshape(3, 3)
        lift = self.lift() if c.lift_mounted else 0.0
        o = d.xpos[self.root] + Rb @ (np.asarray(c.mount) + np.array([0.0, 0.0, lift]))
        R = Rb @ self.Rc
        x = R[:, 0]
        return o - 0.5 * c.baseline * x, o + 0.5 * c.baseline * x, R

    def set_center_pose(self, origin, R_cam2world):
        """free camera (house-only twin / mapping): cyclopean centre + camera-to-world rotation (columns right, up, back)."""
        self.pose_override = (np.asarray(origin, float), np.asarray(R_cam2world, float))

    def _place_cams(self):
        oL, oR, R = self.poses()
        for k, o in (("l", oL), ("r", oR)):
            self.d.cam_xpos[self.cam[k]] = o
            self.d.cam_xmat[self.cam[k]] = R.ravel()
        return oL, oR, R

    def _render(self, which, depth=False, seg=False):
        r = self.r_seg if (seg or (depth and self.r_seg is not None)) else self.r
        r.update_scene(self.d, camera=self.cam[which])
        if depth:
            r.enable_depth_rendering()
        if seg:
            r.enable_segmentation_rendering()
        out = r.render().copy()
        if depth:
            r.disable_depth_rendering()
        if seg:
            r.disable_segmentation_rendering()
        return out

    def static_depth(self):
        """z-depth (H,W) of the PEOPLE-FREE twin from the left camera (people mocap bodies moved away): the prior house model as the sensor itself would see it
        (same visual geometry as the render, so background subtraction compares render with render, not render with collision hulls)."""
        m, d = self.m, self.d
        ids = sorted({int(m.body_mocapid[b]) for b in range(m.nbody) if (m.body(b).name or "").startswith("p_") and m.body_mocapid[b] >= 0})
        if not ids:
            return self._render("l", depth=True)
        old = d.mocap_pos[ids].copy()
        d.mocap_pos[ids, 2] -= 100.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        self._place_cams()
        out = self._render("l", depth=True)
        d.mocap_pos[ids] = old
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        self._place_cams()
        return out

    # ------------------------------------------------------------------ capture
    def capture(self, tick=0, t_sim=0.0) -> dict:
        c = self.cfg
        t0 = time.perf_counter()
        oL, oR, R = self._place_cams()
        L = self._render("l")
        Rr = self._render("r")
        L, Rr = add_noise(L, self.rng, c.read_sigma, c.shot_k), add_noise(Rr, self.rng, c.read_sigma, c.shot_k)
        t1 = time.perf_counter()
        depth, _ok, _ms_b = self.backend.depth(L, Rr, self.K, c.baseline)
        t2 = time.perf_counter()
        out = dict(tick=int(tick), t_sim=float(t_sim), depth=depth, origin=oL.copy(), R=R.copy(), K=dict(self.K))
        if c.self_mask or c.gt:
            seg = self._render("l", seg=True)
            gid = np.where(seg[..., 1] == _GEOM, seg[..., 0], -1)
            cls = np.where(gid >= 0, self.geom_cls[np.clip(gid, 0, None)], 0).astype(np.uint8)
            out["cls"] = cls
            if c.self_mask:
                mk = (cls > 0).astype(np.uint8)
                if c.self_dilate > 0:
                    mk = cv2.dilate(mk, np.ones((2 * c.self_dilate + 1,) * 2, np.uint8))
                out["self_mask"] = mk.astype(bool)
            if c.gt:
                out["gt_seg"] = gid.astype(np.int32)
        if c.static_depth:
            out["static_depth"] = self.static_depth()
        if c.gt:
            out["gt_depth"] = self._render("l", depth=True)
        if c.keep_rgb:
            out["left"] = L
        if c.keep_pair:
            out["pair"] = (L, Rr)
        t3 = time.perf_counter()
        out["ms"] = dict(render=1e3 * (t1 - t0), stereo=1e3 * (t2 - t1), post=1e3 * (t3 - t2), total=1e3 * (t3 - t0))
        self.ms_hist.append(out["ms"])
        return out

    def warmup(self):
        """pay every start-up cost before operation: backend (SGBM buffers / CUDA context + weights) AND a full render + capture on the loaded state."""
        self.backend.warmup()
        for _ in range(2):
            self.capture(-1, 0.0)
        self.ms_hist.clear()

    def close(self):
        try:
            self.r.close()
            if self.r_seg is not None:
                self.r_seg.close()
        except Exception:
            pass
