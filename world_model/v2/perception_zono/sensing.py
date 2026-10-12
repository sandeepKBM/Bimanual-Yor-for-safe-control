"""sensing.py -- ray-cast rig for the zonotope study. SIMULATION ONLY (mj_multiRay on an MJCF scene; no camera driver, no hardware).

Reuses tray_rep.perception.depth_sim: CamSpec / CAMERAS / RIGS (head_aim, wrist_l, wrist_r), DepthCamera (rays, pose, ROI), the noise law
sigma_z(z), grazing mask, dropout, outliers, quantisation. The capture here is a re-implementation of DepthCamera.capture that ALSO returns
the dense z-depth map of each camera (inf = ray hit nothing = free, nan = invalid pixel = unknown), for free-space carving.

Noise conditions (noise_spec):
  ideal    no noise, no dropout, no outliers, no quantisation, no grazing mask
  nominal  the repo's camera specs as they are (head ~1.1 mm at 0.7 m; wrist 2 mm floor + virtual 6 cm stereo; 2 % dropout; 0.2 % outliers)
  n3mm     iid 3 mm depth noise on every camera (sigma_floor = 3 mm, no stereo term); dropout / outliers / quantisation as nominal
Tray frame == world frame shifted: tray home (0,0,1), tray_R = I.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import mujoco

from world_model.v2.tray_rep.perception.depth_sim import CAMERAS, RIGS, DepthCamera

TRAY_POS = np.array([0.0, 0.0, 1.0])
TRAY_R = np.eye(3)


def _lookat_spec(name, pos, target, base="head_aim"):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], 1)
    return replace(CAMERAS[base], name=name, pos=tuple(pos), R=tuple(R.ravel()),
                   note="WHAT-IF placement (not a scene camera): head_aim optics raised to 45 deg elevation")


# what-if: the head_aim optics, 0.45 m up and 0.45 m back from the tray (45 deg elevation); shows how much the 8-degree grazing view costs
_CAMS = dict(CAMERAS)
_CAMS["head_hi"] = _lookat_spec("head_hi", TRAY_POS + [0.45, 0.0, 0.45], TRAY_POS + [0.0, 0.0, 0.03])
_RIGS = dict(RIGS)
_RIGS["head_hi"] = ("head_hi",)
_RIGS["head_hi+wrist"] = ("head_hi", "wrist_l", "wrist_r")


def noise_spec(spec, cond, W=480, H=270):
    kw = dict(W=W, H=H)
    if cond == "ideal":
        kw.update(sigma_floor=0.0, sigma_disp_px=0.0, p_drop=0.0, p_out=0.0, quant_m=0.0, cos_min=0.0)
    elif cond == "n3mm":
        kw.update(sigma_floor=3e-3, sigma_disp_px=0.0)
    elif cond != "nominal":
        raise KeyError(cond)
    return replace(spec, **kw)


@dataclass
class CamView:
    name: str
    origin: np.ndarray
    R: np.ndarray                   # camera-to-world
    f_px: float
    W: int
    H: int
    roi: tuple                      # (u0, u1, v0, v1)
    z: np.ndarray                   # (h, w): measured z-depth; inf = no hit (free); nan = invalid / unknown
    z_min: float = 0.0


@dataclass
class Cloud:
    pts: np.ndarray                 # (n, 3) world, noisy
    label: np.ndarray               # (n,) ground-truth object id of the hit geom (0 = tray)   EVAL ONLY
    cam: np.ndarray                 # (n,) camera index
    views: list
    n_rays: int = 0


def rig_cameras(rig, cond, W=480, H=270):
    return [noise_spec(_CAMS[n], cond, W, H) for n in _RIGS[rig]]


def capture(scene, rig, cond, seed, W=480, H=270, tick=0):
    """cast the rig at the scene. Returns a Cloud (noisy points + GT labels + dense depth maps)."""
    specs = rig_cameras(rig, cond, W, H)
    ctr, rho = scene.center_radius()
    rho = rho + 0.05
    P, L, C, views = [], [], [], []
    nr = 0
    for ci, s in enumerate(specs):
        cam = DepthCamera(s)
        origin, R = cam.pose(TRAY_POS, TRAY_R)
        u0, u1, v0, v1 = cam.roi(origin, R, ctr, rho)
        if u1 <= u0 or v1 <= v0:
            continue
        dirs = cam.dirs[v0:v1, u0:u1].reshape(-1, 3)
        cth = cam.cos_th[v0:v1, u0:u1].reshape(-1)
        n = len(dirs)
        nr += n
        dw = np.ascontiguousarray(dirs @ R.T)
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        nrm = np.zeros(3 * n)
        mujoco.mj_multiRay(scene.m, scene.d, origin, dw.reshape(-1), None, True, -1, geom, dist, nrm, n, s.z_max * 1.6)
        hit = (geom >= 0) & (dist > 0)
        z = np.full(n, np.nan)
        z[~hit] = np.inf                                               # a ray that left the scene: known free
        idx = np.flatnonzero(hit)
        if len(idx):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, int(tick), ci]))
            k = len(idx)
            zt = dist[idx] * cth[idx]
            nh = nrm.reshape(-1, 3)[idx]
            cosinc = np.abs(np.einsum("ij,ij->i", nh, dw[idx]))
            e = rng.normal(size=k)
            u_drop, u_out = rng.random(k), rng.random(k)
            e_out = rng.uniform(0.05, 0.20, k) * np.where(rng.random(k) < 0.5, -1, 1)
            zn = zt + s.sigma_z(zt) * e
            zn = np.where(u_out < s.p_out, zt + e_out, zn)
            if s.quant_m > 0:
                zn = np.round(zn / s.quant_m) * s.quant_m
            ok = (cosinc >= s.cos_min) & (u_drop >= s.p_drop) & (zn >= s.z_min) & (zn <= s.z_max)
            sel = idx[ok]
            z[sel] = zn[ok]
            P.append(origin + (zn[ok] / cth[sel])[:, None] * dw[sel])
            L.append(scene.geom_label[geom[sel]])
            C.append(np.full(len(sel), ci, np.int32))
        views.append(CamView(s.name, origin, R, s.f_px, s.W, s.H, (u0, u1, v0, v1), z.reshape(v1 - v0, u1 - u0), s.z_min))
    if P:
        return Cloud(np.concatenate(P), np.concatenate(L), np.concatenate(C), views, nr)
    return Cloud(np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0, np.int32), views, nr)


def known_free(views, X, margin=0.004):
    """True where the point X (n,3) lies strictly in front of the measured surface of at least one camera (ray-carved free space):
    the pixel it projects into has a valid return farther than the point's z-depth + margin (or no return at all)."""
    X = np.asarray(X)
    free = np.zeros(len(X), bool)
    for v in views:
        pc = (X - v.origin) @ v.R                         # camera frame, camera looks along -z
        zc = -pc[:, 2]
        front = zc > 0.05
        u = v.W / 2.0 + v.f_px * pc[:, 0] / np.where(front, zc, 1.0) - 0.5
        vv = v.H / 2.0 - v.f_px * pc[:, 1] / np.where(front, zc, 1.0) - 0.5
        ui, vi = np.rint(u).astype(int), np.rint(vv).astype(int)
        u0, u1, v0, v1 = v.roi
        inb = front & (ui >= u0) & (ui < u1) & (vi >= v0) & (vi < v1)
        zm = np.full(len(X), np.nan)
        zm[inb] = v.z[vi[inb] - v0, ui[inb] - u0]
        free |= inb & (zm > zc + margin) & (zc > v.z_min)
    return free
