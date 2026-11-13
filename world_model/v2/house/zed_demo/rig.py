"""rig.py -- simulated ZED 2 stereo rig on the office-scene robot head.  SIMULATION ONLY (MuJoCo rendering; no ZED SDK, no robot code).

What is simulated
  * Two ideal pinhole cameras with PARALLEL optical axes (= already rectified), baseline B = 0.12 m (ASSUMED ZED 2 nominal baseline), fovy 70 deg
    (scripts/zed_camera.ZED_FOVY_DEG), HD720 1280x720.  Mount = the office-compose head pose (ASSUMED torso/mast mount, robot frame
    (0.05, -0.20955, 1.15) at lift 0, pitched down `pitch_deg`) -- a PROPOSED mount, confirm on hardware.  Rides with the mocap base (not the lift).
  * Scene textures in the office spec are stripped (flat colours) and block matching is hopeless on flat colour, so THIS module builds its OWN COPY of
    the office spec in which every visual geom gets a procedural multi-scale speckle texture (PNG written to the output dir, tinted by the existing
    material/geom colour).  The shared scene.py / thor.py files are not edited.
  * mild sensor noise on the 8-bit RGB (Gaussian read noise + signal-dependent shot noise).
  * ground truth for evaluation only: z-depth of the left camera and a geom-id segmentation image.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import mujoco

ZED_BASELINE = 0.12                         # m   ASSUMED (ZED 2 datasheet nominal 120 mm)
ZED_FOVY = 70.0                             # deg scripts/zed_camera.ZED_FOVY_DEG
ZED_W, ZED_H = 1280, 720                    # HD720
HEAD_POS = (0.05, -0.20955, 1.15)           # robot frame, lift 0 (office compose HEAD_POS_LIFT0)
HEAD_PITCH_DEG = 36.3                       # overridden by the caller
OUT = Path(os.environ.get("ZED_DEMO_OUT", "data/molmospaces/zed_demo"))


def intrinsics(W=ZED_W, H=ZED_H, fovy=ZED_FOVY):
    f = (H / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return dict(f=f, cx=(W - 1) / 2.0, cy=(H - 1) / 2.0, W=W, H=H, B=ZED_BASELINE)


# ------------------------------------------------------------------------------------------------ textures
def make_speckle_png(path: Path, size=512, seed=0, lo=0.45, hi=1.0):
    """multi-scale correlated noise in [lo, hi] (grey, 3 channels); tiles seamlessly (wrapped gaussian filtering)."""
    import cv2
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    for sig, w in ((1.2, 1.0), (3.0, 1.0), (8.0, 0.6), (24.0, 0.3)):
        n = rng.normal(size=(size, size)).astype(np.float32)
        n = gaussian_filter(n, sig, mode="wrap")
        acc += w * n / max(float(n.std()), 1e-6)
    acc = (acc - acc.mean()) / (acc.std() + 1e-6)
    g = np.clip(0.5 * (lo + hi) + 0.30 * (hi - lo) * acc * 1.6, lo, hi)
    img = np.repeat((g * 255).astype(np.uint8)[:, :, None], 3, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def apply_speckle(spec: mujoco.MjSpec, tex_dir: Path, tile_m=0.4):
    """give every VISUAL geom (and every material) the speckle texture.  Collision geoms (groups 3, 4, 5) are left alone."""
    png = make_speckle_png(Path(tex_dir) / "speckle.png")
    spec.add_texture(name="zed_speckle", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
    for mat in spec.materials:
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "zed_speckle"
        mat.texuniform = True
        mat.texrepeat = [1.0 / tile_m, 1.0 / tile_m]
    base = spec.add_material(name="zed_speckle_mat", rgba=[1, 1, 1, 1])
    base.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "zed_speckle"
    base.texuniform = True
    base.texrepeat = [1.0 / tile_m, 1.0 / tile_m]

    def walk(b):
        for g in b.geoms:
            if g.group in (3, 4, 5):
                continue
            if g.type in (mujoco.mjtGeom.mjGEOM_PLANE,) or True:
                if not g.material:
                    g.material = "zed_speckle_mat"
        for c in b.bodies:
            walk(c)
    walk(spec.worldbody)


# ------------------------------------------------------------------------------------------------ scene + cameras
def _cyl(body, name, xy, r, z0, h, rgba, shape="cylinder"):
    """static cylinder: visual + collision twin (house structural bits)."""
    typ = mujoco.mjtGeom.mjGEOM_CYLINDER
    pos = [xy[0], xy[1], z0 + h / 2]
    body.add_geom(name=f"{name}_vis", type=typ, size=[r, h / 2, 0], pos=pos, rgba=list(rgba), group=1, contype=0, conaffinity=0)
    body.add_geom(name=f"{name}_col", type=typ, size=[r, h / 2, 0], pos=pos, rgba=[0, 1, 0, 0.3], group=4, contype=8, conaffinity=15,
                  friction=[0.9, 0.9, 0.001])


def add_clutter(spec: mujoco.MjSpec, top: float):
    """extra desk_1 clutter (static): a bottle, a mug, a book, a pen cup, a box.  Returns {name: dict(kind, centre, half)} (GT, EVAL ONLY)."""
    from world_model.v2.house.office import scene as S
    from world_model.v2.house.office import thor as T
    W = spec.worldbody
    gt = {}
    # tall bottle straight on the naive centre line of the delivery desk (the naive tray target (4.2, 5.36) lands on it)
    _cyl(W, "bottle_a", (4.28, 5.30), 0.036, top, 0.27, (0.15, 0.55, 0.30, 1))
    _cyl(W, "bottle_a_neck", (4.28, 5.30), 0.017, top + 0.27, 0.07, (0.15, 0.55, 0.30, 1))
    gt["bottle_a"] = dict(centre=[4.28, 5.30, top + 0.17], half=[0.036, 0.036, 0.17])
    _, inf = T.place(spec, *S.MUG, "mug_desk1", (4.64, 5.40), 0.0, z=top)
    gt["mug_desk1"] = dict(centre=inf["centre"], half=(np.array(inf["size"]) / 2).tolist())
    S.add_rect(W, "book_a", 4.86, 5.14, 5.28, 5.40, top, top + 0.10, (0.75, 0.25, 0.20, 1))
    gt["book_a"] = dict(centre=[5.00, 5.34, top + 0.05], half=[0.14, 0.06, 0.05])
    _cyl(W, "pencup_a", (3.84, 5.62), 0.045, top, 0.11, (0.25, 0.30, 0.65, 1))
    gt["pencup_a"] = dict(centre=[3.84, 5.62, top + 0.055], half=[0.045, 0.045, 0.055])
    S.add_rect(W, "box_a", 3.30, 3.52, 5.22, 5.44, top, top + 0.18, (0.85, 0.70, 0.30, 1))
    gt["box_a"] = dict(centre=[3.41, 5.33, top + 0.09], half=[0.11, 0.11, 0.09])
    return gt


def _zed_cams(root, pos=HEAD_POS, pitch_deg=HEAD_PITCH_DEG, fovy=ZED_FOVY, name="head_cam", baseline=ZED_BASELINE):
    """replacement for compose._head_cam: a rectified stereo pair (left/right, parallel axes) on the mocap root."""
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    up = np.array([math.sin(p), 0.0, math.cos(p)])
    z = -fwd
    x = np.cross(up, z)                     # camera x = image right (robot -y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, up, z], 1).ravel())
    pos = np.asarray(pos, float)
    for nm, s in (("zed_left", -0.5), ("zed_right", +0.5)):
        root.add_camera(name=nm, pos=(pos + s * baseline * x).tolist(), quat=q.tolist(), fovy=fovy)


def build_composed(pitch_deg=HEAD_PITCH_DEG, textures=True, clutter=True, dt=0.005, tex_dir=None, spawn_xy=None, yaw=None):
    """office scene + clutter + speckle textures + robot (lift, tray, ball) + ZED stereo cameras.  Returns (C, gt_clutter)."""
    from world_model.v2.house.office import scene as S, compose as CP
    spec = S.load_office_spec()
    meta = S.office_meta()
    gt = add_clutter(spec, meta["desk_top_height"]) if clutter else {}
    if textures:
        apply_speckle(spec, Path(tex_dir or OUT / "tex"))
    wp = meta["waypoints"]["lab_desk_place"]
    old = CP._head_cam
    CP._head_cam = _zed_cams
    try:
        C = CP.new_hold_office(spawn_xy=spawn_xy if spawn_xy is not None else wp["xy"], yaw=yaw if yaw is not None else wp["heading"],
                               dt=dt, lift_h=0.0, servos_off=True, head_cam=True, head_pitch_deg=pitch_deg, head_fovy=ZED_FOVY, spec=spec)
    finally:
        CP._head_cam = old
    return C, gt


# ------------------------------------------------------------------------------------------------ capture
def add_sensor_noise(img, rng, read_sigma=1.5, shot_k=0.35):
    """mild 8-bit sensor noise: Gaussian read noise + signal-dependent (shot-like) noise.  img uint8 (H,W,3)."""
    f = img.astype(np.float32)
    sd = np.sqrt(read_sigma ** 2 + shot_k * f)
    return np.clip(f + rng.normal(size=f.shape).astype(np.float32) * sd, 0, 255).astype(np.uint8)


class ZedRig:
    """renders the stereo pair (+ GT depth / segmentation of the left eye) from a composed model."""

    def __init__(self, m, W=ZED_W, H=ZED_H, seed=0):
        self.m, self.W, self.H = m, W, H
        self.K = intrinsics(W, H)
        self.r = mujoco.Renderer(m, height=H, width=W)
        self.rng = np.random.default_rng(seed)
        self.cam_l = m.camera("zed_left").id
        self.cam_r = m.camera("zed_right").id

    def _render(self, d, cam, depth=False, seg=False):
        self.r.update_scene(d, camera=cam)
        if depth:
            self.r.enable_depth_rendering()
        if seg:
            self.r.enable_segmentation_rendering()
        out = self.r.render().copy()
        if depth:
            self.r.disable_depth_rendering()
        if seg:
            self.r.disable_segmentation_rendering()
        return out

    def capture(self, d, gt=True, noise=True):
        L = self._render(d, self.cam_l)
        R = self._render(d, self.cam_r)
        if noise:
            L, R = add_sensor_noise(L, self.rng), add_sensor_noise(R, self.rng)
        out = dict(left=L, right=R)
        if gt:
            out["gt_depth"] = self._render(d, self.cam_l, depth=True)
            out["gt_seg"] = self._render(d, self.cam_l, seg=True)
        return out

    def left_pose(self, d):
        """(origin (3,), R_cam2world (3,3)) of the left camera: camera looks along -z, x right, y up (MuJoCo)."""
        return d.cam_xpos[self.cam_l].copy(), d.cam_xmat[self.cam_l].reshape(3, 3).copy()

    def close(self):
        self.r.close()


def pick_gpu():
    import subprocess
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout
        for line in o.strip().splitlines():
            i, mem, util = (int(v) for v in line.replace(" ", "").split(","))
            if i != 0 and mem < 500 and util < 20:
                return str(i)
    except Exception:
        pass
    return None
