"""grasp/wrist.py -- simulated wrist-camera depth for the friction-grasp assist.  SIMULATION ONLY (ray casting, CPU, no GL, no driver).

Uses the MJCF cameras `left_arm_egocentric` / `right_arm_egocentric` of the composed rig and the depth noise model of
`tray_rep.perception.depth_sim` (CamSpec wrist_l/wrist_r: stereo-law sigma_z with 2 mm floor, grazing-incidence invalid, 2 % dropout,
rare gross outliers, 1 mm quantisation, valid range [0.10, 2.0] m, 160 x 90 grid).  The wrist camera has NO documented depth on the real
robot (CamSpec.assumed_depth=True): every result built on this is a what-if on an ASSUMED depth-capable wrist camera.

Differences from house/perception_bridge.HouseCamera (not imported: it needs a `ball` body and mutates geom groups of the model):
  * geom groups are NOT modified: the mask {0 (tray/cup/beads/shelf/house visuals), 3 (robot collision), 4 (house collision hulls)} is
    passed to mj_multiRay.
  * self-filter: a pixel whose first hit is a robot geom (own gripper, either arm, base) is dropped -- equivalent to rendering the
    known robot kinematics and masking it (what perception_bridge does with a second ray pass).  Uses only the robot's own model.
Frame.geom (the hit geom id) is for EVALUATION and for the PRIVILEGED segmenter only; the depth-only path never reads it.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

from world_model.v2.tray_rep.perception import depth_sim as D

CUTOFF_MULT = 1.2
MASK_GROUPS = (0, 3, 4)
CAM_NAME = {"left": "left_arm_egocentric", "right": "right_arm_egocentric"}
SPEC_NAME = {"left": "wrist_l", "right": "wrist_r"}


@dataclass
class WristFrame:
    side: str
    tick: int
    t: float
    cam_pos: np.ndarray                  # camera centre, world
    cam_R: np.ndarray                    # camera-to-world rotation (MuJoCo camera: looks along -z)
    grip_pos: np.ndarray                 # gripper_base pose at capture (world)
    grip_R: np.ndarray
    pts: np.ndarray                      # (n,3) noisy world points (after self-filter)
    geom: np.ndarray                     # (n,) hit geom id  [EVAL / privileged segmenter ONLY]
    z: np.ndarray                        # (n,) z-depth
    n_rays: int = 0
    n_self: int = 0


class WristCam:
    def __init__(self, m: mujoco.MjModel, side: str, spec: D.CamSpec | None = None):
        self.m, self.side = m, side
        base = D.CAMERAS[SPEC_NAME[side]] if spec is None else spec
        self.spec = D.CamSpec(**{**base.__dict__, "mount": "model"})
        self.cam = D.DepthCamera(self.spec)
        self.cam_id = int(m.camera(CAM_NAME[side]).id)
        self.body_excl = int(m.cam_bodyid[self.cam_id])
        self.grip_body = int(m.body(f"{side}_arm_gripper_base").id)
        mask = np.zeros(6, np.uint8)
        mask[list(MASK_GROUPS)] = 1
        self.mask = mask
        # robot geoms (everything under yor_root) for the self-filter
        root = int(m.body("yor_root").id)
        anc = np.zeros(m.nbody, bool)
        for b in range(m.nbody):
            x = b
            while x > 0:
                if x == root:
                    anc[b] = True
                    break
                x = int(m.body_parentid[x])
        self.is_robot_geom = anc[m.geom_bodyid]
        self.tray_geoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "tray"], int)
        self.is_cup_geom = np.array([m.body(m.geom_bodyid[g]).name == "cup" for g in range(m.ngeom)])

    def pose(self, d):
        return d.cam_xpos[self.cam_id].copy(), d.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def roi_for_box(self, d, centre, R_box, half, pad_px=3):
        """pixel window (v0,v1,u0,u1) containing an oriented box (centre, rotation, half sizes) -- the prior volume of the tray; the full
        grid if a corner is behind / too close to the camera."""
        s = self.spec
        origin, R = self.pose(d)
        c = np.array([[i * half[0], j * half[1], k * half[2]] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        pc = (np.asarray(centre) + c @ np.asarray(R_box).T - origin) @ R
        if (pc[:, 2] >= -0.03).any():
            return (0, s.H, 0, s.W)
        zc = -pc[:, 2]
        u = s.W / 2.0 + s.f_px * pc[:, 0] / zc - 0.5
        v = s.H / 2.0 - s.f_px * pc[:, 1] / zc - 0.5
        u0, u1 = max(int(np.floor(u.min())) - pad_px, 0), min(int(np.ceil(u.max())) + 1 + pad_px, s.W)
        v0, v1 = max(int(np.floor(v.min())) - pad_px, 0), min(int(np.ceil(v.max())) + 1 + pad_px, s.H)
        if u1 <= u0 or v1 <= v0:
            return (0, 0, 0, 0)
        return (v0, v1, u0, u1)

    def capture(self, d, tick: int, seed: int = 0, t: float = 0.0, roi=None) -> WristFrame:
        """roi=(v0,v1,u0,u1) optional pixel window (default: the full 160x90 grid)."""
        s, m = self.spec, self.m
        origin, R = self.pose(d)
        v0, v1, u0, u1 = (0, s.H, 0, s.W) if roi is None else roi
        dirs = self.cam.dirs[v0:v1, u0:u1].reshape(-1, 3)
        cth = self.cam.cos_th[v0:v1, u0:u1].reshape(-1)
        n = len(dirs)
        if n == 0:
            return WristFrame(self.side, tick, t, origin, R, d.xpos[self.grip_body].copy(), d.xmat[self.grip_body].reshape(3, 3).copy(),
                              np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0), 0, 0)
        dw = np.ascontiguousarray(dirs @ R.T)
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        nrm = np.zeros(3 * n)
        mujoco.mj_multiRay(m, d, origin, dw.reshape(-1), self.mask, True, self.body_excl, geom, dist, nrm, n, s.z_max * CUTOFF_MULT)
        hit = (geom >= 0) & (dist > 0)
        idx = np.flatnonzero(hit)
        gb = int(self.grip_body)
        gp, gR = d.xpos[gb].copy(), d.xmat[gb].reshape(3, 3).copy()
        empty = WristFrame(self.side, tick, t, origin, R, gp, gR, np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0), n, 0)
        if len(idx) == 0:
            return empty
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, int(tick), zlib.crc32(s.name.encode())]))
        k = len(idx)
        z = dist[idx] * cth[idx]
        nh = nrm.reshape(-1, 3)[idx]
        cosinc = np.abs(np.einsum("ij,ij->i", nh, dw[idx]))
        e = rng.normal(size=k)
        u_drop, u_out = rng.random(k), rng.random(k)
        e_out = rng.uniform(0.05, 0.20, k) * np.where(rng.random(k) < 0.5, -1, 1)
        zn = z + s.sigma_z(z) * e
        zn = np.where(u_out < s.p_out, z + e_out, zn)
        if s.quant_m > 0:
            zn = np.round(zn / s.quant_m) * s.quant_m
        ok = (cosinc >= s.cos_min) & (u_drop >= s.p_drop) & (zn >= s.z_min) & (zn <= s.z_max)
        robot = self.is_robot_geom[geom[idx]]
        n_self = int((ok & robot).sum())
        ok &= ~robot                                   # self-filter (known robot kinematics)
        sel = idx[ok]
        zn = zn[ok]
        pts = origin + (zn / cth[sel])[:, None] * dw[sel]
        return WristFrame(self.side, tick, t, origin, R, gp, gR, pts, geom[sel].copy(), zn.copy(), n, n_self)

    def seg_cup(self, d, nominal_range=0.07):
        """RGB-equivalent cup track: cast the full pixel grid (no depth noise, NO minimum range: an RGB image has none) and mark the pixels
        whose first hit is the red cup.  Stands in for a red-colour threshold on the wrist RGB image (the cup is the only saturated red object
        in the scene); the real RGB render (EGL) is used by the video recorder.  Returns (n_px, centroid_u, centroid_v, mean_range) or None."""
        s, m = self.spec, self.m
        origin, R = self.pose(d)
        dirs = self.cam.dirs.reshape(-1, 3)
        n = len(dirs)
        dw = np.ascontiguousarray(dirs @ R.T)
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        mujoco.mj_multiRay(m, d, origin, dw.reshape(-1), self.mask, True, self.body_excl, geom, dist, None, n, 1.0)
        hit = geom >= 0
        cup = np.zeros(n, bool)
        cup[hit] = self.is_cup_geom[geom[hit]]
        if cup.sum() < 20:
            return None
        v, u = np.divmod(np.flatnonzero(cup), s.W)
        return int(cup.sum()), float(u.mean()), float(v.mean()), float(dist[cup].mean())

    # ---- RGB path (PROPRIO-ONLY): a colour image and a colour threshold -- no geom-id mask is read by the segmenter
    def rgb_image(self, d, tick=0, seed=0, sigma=0.03):
        """(H,W,3) float RGB in [0,1] of the wrist view: ray-cast the full pixel grid and shade each hit with the model's surface colour
        (geom_rgba, alpha ignored) * Lambert(|n . view|) + sensor noise.  Own-gripper hits are mid grey (the robot's real colour is not the cup's).
        Stand-in for a GL render (the video recorder uses the real EGL render); `seg_cup_rgb` accepts either."""
        s, m = self.spec, self.m
        origin, R = self.pose(d)
        dirs = self.cam.dirs.reshape(-1, 3)
        n = len(dirs)
        dw = np.ascontiguousarray(dirs @ R.T)
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        nrm = np.zeros(3 * n)
        mujoco.mj_multiRay(m, d, origin, dw.reshape(-1), self.mask, True, self.body_excl, geom, dist, nrm, n, 1.0)
        img = np.full((n, 3), 0.55)                                    # background / wall
        hit = geom >= 0
        col = m.geom_rgba[geom[hit], :3].astype(float)
        col[self.is_robot_geom[geom[hit]]] = 0.45
        shade = 0.35 + 0.65 * np.abs(np.einsum("ij,ij->i", nrm.reshape(-1, 3)[hit], dw[hit]))
        img[hit] = col * shade[:, None]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, int(tick), zlib.crc32(s.name.encode()), 5]))
        img = np.clip(img + rng.normal(0, sigma, img.shape), 0.0, 1.0)
        return img.reshape(s.H, s.W, 3)

    @staticmethod
    def seg_cup_rgb(img, min_px=20):
        """colour threshold for the saturated red cup on an RGB image (float [0,1] or uint8): r > 0.25, r > 1.8 g, r > 1.8 b.
        Returns (n_px, centroid_u, centroid_v, nan) or None -- same tuple as seg_cup (the range is unobservable in an RGB image)."""
        a = np.asarray(img, float)
        if a.max() > 1.5:
            a = a / 255.0
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        mask = (r > 0.25) & (r > 1.8 * g) & (r > 1.8 * b)
        if mask.sum() < min_px:
            return None
        v, u = np.nonzero(mask)
        return int(mask.sum()), float(u.mean()), float(v.mean()), float("nan")

    def seg_cup_proprio(self, d, tick=0, seed=0):
        return self.seg_cup_rgb(self.rgb_image(d, tick=tick, seed=seed))

    def depth_image(self, fr: WristFrame):
        """dense (H,W) z-depth image (nan = no return) rebuilt from a full-grid frame, for insets/debugging."""
        s = self.spec
        img = np.full((s.H, s.W), np.nan)
        pc = (fr.pts - fr.cam_pos) @ fr.cam_R
        zc = -pc[:, 2]
        u = np.clip(np.round(s.W / 2.0 + s.f_px * pc[:, 0] / zc - 0.5).astype(int), 0, s.W - 1)
        v = np.clip(np.round(s.H / 2.0 - s.f_px * pc[:, 1] / zc - 0.5).astype(int), 0, s.H - 1)
        img[v, u] = zc
        return img
