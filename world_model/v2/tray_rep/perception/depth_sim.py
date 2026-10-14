"""depth_sim.py -- a simulated depth camera for the tray + ball testbed (ray casting, CPU, no GL, no GPU).
SIMULATION ONLY. No camera driver, no ZED SDK, no hardware, no network.

What it does
  * `Scene` is a tiny MuJoCo model with two mocap bodies: the tray (box, optional back lip) and the ball (sphere).
    It is separate from the physics model. `set_pose` places both and runs mj_kinematics.
  * `DepthCamera.capture` casts one ray per pixel with `mujoco.mj_multiRay` and turns the hits into a noisy point cloud.
  * `DepthSensor` adds the timing: a frame clock (sample-and-hold, e.g. 30 Hz against the 50 Hz tick) and a delivery
    delay in ticks.

Noise model (every number is a field of `CamSpec`; the values are ASSUMED unless the README says otherwise)
  1. z-depth noise  sigma_z = sqrt(floor^2 + (z^2 / (f_native * baseline) * sigma_disp_px)^2)   (stereo law)
     f_native = (native_H / 2) / tan(fovy / 2): the focal length in pixels of the REAL sensor (e.g. HD720), not of the
     decimated grid we cast.
  2. invalid pixel when the ray meets the surface at a grazing angle: |cos(incidence)| < cos_min (from the ray normal).
  3. random dropout of a hit (probability p_drop) and rare gross outliers (probability p_out, depth shifted by +-5..20 cm).
  4. depth quantisation (step quant_m) and a valid range [z_min, z_max].
  5. frame clock + latency (DepthSensor).
Each frame draws its noise from a generator seeded by (seed, capture tick), so a snapshot restore replays the same noise.

Geometry ids come back with every hit. They are for EVALUATION only (the estimator never reads them).

Frames. The world frame here is the physics world of `b_tray_ball/tray_env` (tray home at (0, 0, 1)). The tray frame has its
origin on the tray top surface, z = tray normal. Cameras are of two kinds:
  mount 'world': fixed in the world (the head camera; the tray moves in front of it).
  mount 'tray' : rigid with the tray (the wrist cameras in a rigid bimanual grasp).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import mujoco

from world_model.v2.tray_rep import common  # noqa: F401  (thread pins, sys.path)
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY, TRAY_T, LIP_H, LIP_T, DT_WM

RADIUS = 0.03

# ----------------------------------------------------------------------------------------------------------------
# camera specs and rigs
# ----------------------------------------------------------------------------------------------------------------
# Carry pose used to map robot-frame cameras into the testbed world (authority/recommend.json, front grip, 'mid_mid'):
TRAY_O_ROBOT = np.array([0.4, -0.209550, 0.95])      # tray origin in the robot/scene world frame [m]
TRAY_HOME = np.array([0.0, 0.0, 1.0])                # tray home in the testbed world (tray_env)


def _xyaxes_to_R(xy):
    """MuJoCo <camera xyaxes=...> -> 3x3 camera-to-world rotation (columns = camera x, y, z axes)."""
    x = np.array(xy[:3], float); x /= np.linalg.norm(x)
    y = np.array(xy[3:], float); y -= (y @ x) * x; y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], 1)


# (a) proposed ZED, scenes/scene_yor_canonical.xml line 59: pos 1.1 -0.21 1.05, xyaxes 0 1 0 -0.5924 0 0.8057, fovy 55
_HEAD_POS_ROBOT = np.array([1.1, -0.21, 1.05])
_HEAD_R = _xyaxes_to_R([0, 1, 0, -0.5924, 0, 0.8057])

# (b) YOR wrist cameras, piper_{left,right}.xml <camera name="egocentric" fovy="53.1">, forward kinematics of the front/mid_mid
# carry pose (authority/robot_authority.py Rig, YOR/robot/yor-description/robot-welded-base-and-lift.mjcf). In a rigid
# grasp the camera pose in the tray frame is the same for every carry pose (verified for 7 front poses: identical).
_WRIST = {
    "wrist_l": (np.array([-0.123004, 0.088512, 0.014978]),
                np.array([[-0.000011, -0.258800, -0.965931], [-0.000804, -0.965931, 0.258800], [-1.0, 0.000780, -0.000197]])),
    "wrist_r": (np.array([-0.123000, -0.088500, -0.014972]),
                np.array([[0.0, -0.258819, -0.965926], [0.000796, 0.965926, -0.258819], [1.0, -0.000769, 0.000206]])),
}

# (c) real ZED nav camera (scripts/zed_camera.py MOUNT NOTE, derived from YOR/robot/zed_pub_node.py): low forward camera,
# 0.2143 m forward, 0.0603 m left of the base origin, ~0.2 m up (ASSUMED height), pitched down 23.8 deg. Diagnostic only.
_NAV_POS_ROBOT = np.array([0.2143, -0.209550 + 0.0603, 0.25])
_pit = np.deg2rad(23.8)


def _nav_R():
    """camera looks along +x pitched DOWN by 23.8 deg; image right = -y (robot left is +y)."""
    fwd = np.array([np.cos(_pit), 0.0, -np.sin(_pit)])
    right = np.array([0.0, -1.0, 0.0])
    up = np.cross(right, fwd)
    return np.stack([right, up, -fwd], 1)


@dataclass(frozen=True)
class CamSpec:
    name: str
    mount: str                       # 'world' | 'tray'
    pos: tuple                       # world pos (mount 'world') or tray-frame pos (mount 'tray')
    R: tuple                         # 3x3 camera-to-world (or camera-to-tray) rotation, flattened row-major
    fovy: float                      # vertical field of view [deg] (MuJoCo convention)
    W: int = 160                     # simulated grid
    H: int = 90
    native_H: int = 720              # real sensor rows (HD720) used for the stereo focal length
    baseline: float = 0.12           # stereo baseline [m]   (ZED 2 = 0.12; ZED mini = 0.063)  ASSUMED model
    sigma_disp_px: float = 0.25      # disparity noise at native resolution [px]               ASSUMED
    sigma_floor: float = 5e-4        # noise floor [m]
    z_min: float = 0.30              # min valid depth [m] (ZED 2: 0.3)
    z_max: float = 5.0
    cos_min: float = float(np.cos(np.deg2rad(80.0)))   # grazing mask: invalid when incidence > 80 deg   ASSUMED
    p_drop: float = 0.02             # random dropout of a hit                                    ASSUMED
    p_out: float = 0.002             # rare gross outlier                                         ASSUMED
    quant_m: float = 1e-3            # depth quantisation step [m]                                ASSUMED
    rate_hz: float = 30.0            # frame rate                                                 ASSUMED
    latency_ticks: int = 3           # capture -> delivery delay in 50 Hz ticks (60 ms)           ASSUMED
    assumed_depth: bool = False      # True: the real device has NO documented depth; depth is a what-if
    note: str = ""

    @property
    def f_px(self):
        return (self.H / 2.0) / np.tan(np.deg2rad(self.fovy) / 2.0)

    @property
    def f_native(self):
        return (self.native_H / 2.0) / np.tan(np.deg2rad(self.fovy) / 2.0)

    def sigma_z(self, z):
        z = np.asarray(z, float)
        return np.sqrt(self.sigma_floor ** 2 + (z * z / (self.f_native * self.baseline) * self.sigma_disp_px) ** 2)


CARRY_O_ROBOT = {"mid_mid": TRAY_O_ROBOT, "mid_low": np.array([0.4, -0.209550, 0.80])}   # authority poses (front grip)


def _head_spec(carry="mid_mid", name="head", **kw):
    """the head camera is fixed in the robot frame; the testbed world is the robot frame shifted so that the tray home
    sits at the chosen carry pose (mid_mid: tray centre 0.95 m high; mid_low: 0.80 m)."""
    pos = _HEAD_POS_ROBOT - CARRY_O_ROBOT[carry] + TRAY_HOME
    base = dict(name=name, mount="world", pos=tuple(pos), R=tuple(_HEAD_R.ravel()), fovy=55.0,
                note="proposed torso ZED (scene_yor_canonical.xml line 59); mount is a proposal; model/baseline ASSUMED")
    base.update(kw)
    return CamSpec(**base)


def _head_aim_spec(carry="mid_mid", name="head_aim", **kw):
    """PLACEMENT WHAT-IF: the proposed head position, re-aimed at the tray home (look-at, world up). Not a scene camera."""
    pos = _HEAD_POS_ROBOT - CARRY_O_ROBOT[carry] + TRAY_HOME
    fwd = (TRAY_HOME + np.array([0.0, 0.0, 0.02])) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], 1)
    base = dict(name=name, mount="world", pos=tuple(pos), R=tuple(R.ravel()), fovy=55.0,
                note="what-if: the proposed ZED position, pitched to look at the tray (the scene camera looks at the table)")
    base.update(kw)
    return CamSpec(**base)


def _wrist_spec(side, **kw):
    p, R = _WRIST[side]
    base = dict(name=side, mount="tray", pos=tuple(p), R=tuple(R.ravel()), fovy=53.1, native_H=720, baseline=0.06,
                sigma_disp_px=0.25, sigma_floor=2e-3, z_min=0.10, z_max=2.0, assumed_depth=True,
                note="YOR wrist camera 'egocentric' (iPhone mesh in the MJCF). Depth is NOT documented: ASSUMED depth-capable; "
                     "noise = 2 mm floor + virtual 6 cm stereo; min range 10 cm ASSUMED")
    base.update(kw)
    return CamSpec(**base)


def _nav_spec(**kw):
    pos = _NAV_POS_ROBOT - TRAY_O_ROBOT + TRAY_HOME
    base = dict(name="zed_nav", mount="world", pos=tuple(pos), R=tuple(_nav_R().ravel()), fovy=70.0, W=160, H=90,
                note="real ZED nav mount derived from zed_pub_node.py extrinsic (height ASSUMED 0.25 m); diagnostic: expected blind")
    base.update(kw)
    return CamSpec(**base)


CAMERAS = {"head": _head_spec(), "head_low": _head_spec("mid_low", "head_low"),
           "head_aim": _head_aim_spec(), "wrist_l": _wrist_spec("wrist_l"), "wrist_r": _wrist_spec("wrist_r"),
           "zed_nav": _nav_spec()}
RIGS = {"head": ("head",), "wrist_l": ("wrist_l",), "wrist_r": ("wrist_r",), "wrist_both": ("wrist_l", "wrist_r"),
        "head+wrist": ("head", "wrist_l", "wrist_r"), "head_low": ("head_low",), "head_aim": ("head_aim",), "head_aim+wrist": ("head_aim", "wrist_l", "wrist_r"),
        "head_low+wrist": ("head_low", "wrist_l", "wrist_r"), "zed_nav": ("zed_nav",)}


def rig_specs(rig, **overrides):
    """CamSpec tuple for a rig name; `overrides` (e.g. rate_hz=15, latency_ticks=0, W=320, H=180) apply to every camera."""
    return tuple(replace(CAMERAS[n], **overrides) if overrides else CAMERAS[n] for n in RIGS[rig])


# ----------------------------------------------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------------------------------------------
class Scene:
    """Tray (box, optional back lip) + ball (sphere) as two mocap bodies in a separate model. Used for ray casting only."""

    def __init__(self, radius=RADIUS, lip=False):
        self.radius, self.lip = float(radius), bool(lip)
        lip_xml = (f'<geom name="lip" type="box" size="{LIP_T} {HY} {LIP_H / 2}" pos="{-HX + LIP_T:.5f} 0 {LIP_H / 2:.5f}"/>'
                   if lip else "")
        xml = f"""<mujoco><worldbody>
  <body name="tray" mocap="true"><geom name="tray_top" type="box" size="{HX} {HY} {TRAY_T}" pos="0 0 {-TRAY_T}"/>{lip_xml}</body>
  <body name="ball" mocap="true"><geom name="ball_g" type="sphere" size="{radius}"/></body>
</worldbody></mujoco>"""
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.gid_tray = [self.m.geom("tray_top").id] + ([self.m.geom("lip").id] if lip else [])
        self.gid_ball = self.m.geom("ball_g").id
        self.body_tray, self.body_ball = self.m.body("tray").id, self.m.body("ball").id
        self.mocap_tray, self.mocap_ball = int(self.m.body_mocapid[self.body_tray]), int(self.m.body_mocapid[self.body_ball])

    def set_pose(self, tray_pos, tray_quat, ball_pos):
        self.d.mocap_pos[self.mocap_tray] = tray_pos
        self.d.mocap_quat[self.mocap_tray] = tray_quat
        self.d.mocap_pos[self.mocap_ball] = ball_pos
        self.d.mocap_quat[self.mocap_ball] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_kinematics(self.m, self.d)


def quat_to_R(q):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


# ----------------------------------------------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Frame:
    """One captured cloud. pts_w (n,3) world points (after noise); geom (n,) ids [EVAL ONLY]; origin (3,) camera centre;
    tray_pos / tray_R: tray pose at capture time (the arm encoders give it with a timestamp); stats for the cost model."""
    pts_w: np.ndarray
    geom: np.ndarray
    origin: np.ndarray
    cam_R: np.ndarray
    tray_pos: np.ndarray
    tray_R: np.ndarray
    tick: int
    n_rays: int = 0
    cam: str = ""
    z_depth: np.ndarray = field(default_factory=lambda: np.zeros(0))     # z-depth of each returned point
    dir_w: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # unit ray direction (world) of each point

    def to_tray(self):
        """(points, origin) in the tray frame (tray pose at capture time)."""
        R = self.tray_R
        return (self.pts_w - self.tray_pos) @ R, (self.origin - self.tray_pos) @ R


class DepthCamera:
    def __init__(self, spec: CamSpec):
        self.spec = spec
        W, H, f = spec.W, spec.H, spec.f_px
        u = (np.arange(W) + 0.5 - W / 2.0) / f
        v = -(np.arange(H) + 0.5 - H / 2.0) / f                      # image row increases downward
        X, Y = np.meshgrid(u, v)                                      # (H, W)
        d = np.stack([X, Y, -np.ones_like(X)], -1)                    # MuJoCo camera looks along -z
        self.n_cam = np.linalg.norm(d, axis=-1)
        self.dirs = d / self.n_cam[..., None]                         # unit rays in the camera frame (H, W, 3)
        self.cos_th = 1.0 / self.n_cam                                # z-depth = ray length * cos_th
        self._R_cam = np.asarray(spec.R, float).reshape(3, 3)
        self._pos = np.asarray(spec.pos, float)

    # ---- pose
    def pose(self, tray_pos, tray_R):
        """camera centre and camera-to-world rotation for the given tray pose."""
        if self.spec.mount == "tray":
            return tray_pos + tray_R @ self._pos, tray_R @ self._R_cam
        return self._pos, self._R_cam

    def roi(self, origin, R, centre, rho):
        """pixel box (u0, u1, v0, v1) [half-open] that contains a sphere (centre, rho). Empty box if it is outside the view."""
        s = self.spec
        pc = R.T @ (np.asarray(centre) - origin)
        dist = float(np.linalg.norm(pc))
        if pc[2] >= 0:                                                # behind the camera
            return 0, 0, 0, 0
        if dist <= rho * 1.05:
            return 0, s.W, 0, s.H
        z = -pc[2]
        uc = s.W / 2.0 + s.f_px * pc[0] / z - 0.5
        vc = s.H / 2.0 - s.f_px * pc[1] / z - 0.5
        alpha = np.arcsin(min(rho / dist, 0.999))
        cth2 = max((z / dist) ** 2, 0.1)                              # off-axis stretch of the projected disc
        r_px = s.f_px * np.tan(alpha) / cth2 * 1.1 + 2.0
        u0, u1 = int(np.floor(uc - r_px)), int(np.ceil(uc + r_px)) + 1
        v0, v1 = int(np.floor(vc - r_px)), int(np.ceil(vc + r_px)) + 1
        u0, u1, v0, v1 = max(u0, 0), min(u1, s.W), max(v0, 0), min(v1, s.H)
        if u1 <= u0 or v1 <= v0:
            return 0, 0, 0, 0
        return u0, u1, v0, v1

    # ---- capture
    def capture(self, scene: Scene, tray_pos, tray_R, tick, seed, use_roi=True, roi_pad=0.06):
        """Cast rays (inside the ROI around the tray when `use_roi`) and return a Frame with noisy world points."""
        s = self.spec
        origin, R = self.pose(tray_pos, tray_R)
        rho = float(np.hypot(HX, HY)) + scene.radius * 2.0 + roi_pad
        centre = tray_pos + tray_R @ np.array([0.0, 0.0, scene.radius])
        u0, u1, v0, v1 = self.roi(origin, R, centre, rho) if use_roi else (0, s.W, 0, s.H)
        empty = Frame(np.zeros((0, 3)), np.zeros(0, np.int32), origin, R, np.array(tray_pos, float), np.array(tray_R, float),
                      int(tick), 0, s.name)
        if u1 <= u0 or v1 <= v0:
            return empty
        dirs = self.dirs[v0:v1, u0:u1].reshape(-1, 3)
        cth = self.cos_th[v0:v1, u0:u1].reshape(-1)
        n = len(dirs)
        dw = np.ascontiguousarray(dirs @ R.T)                          # world rays
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        nrm = np.zeros(3 * n)
        mujoco.mj_multiRay(scene.m, scene.d, origin, dw.reshape(-1), None, True, -1, geom, dist, nrm, n, s.z_max * 1.6)
        hit = (geom >= 0) & (dist > 0)
        idx = np.flatnonzero(hit)
        if len(idx) == 0:
            empty.n_rays = n
            return empty
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, int(tick)]))
        k = len(idx)
        z = dist[idx] * cth[idx]                                       # true z-depth
        nh = nrm.reshape(-1, 3)[idx]
        cosinc = np.abs(np.einsum("ij,ij->i", nh, dw[idx]))
        e = rng.normal(size=k)
        u_drop, u_out, e_out = rng.random(k), rng.random(k), rng.uniform(0.05, 0.20, k) * np.where(rng.random(k) < 0.5, -1, 1)
        zn = z + s.sigma_z(z) * e
        zn = np.where(u_out < s.p_out, z + e_out, zn)
        if s.quant_m > 0:
            zn = np.round(zn / s.quant_m) * s.quant_m
        ok = (cosinc >= s.cos_min) & (u_drop >= s.p_drop) & (zn >= s.z_min) & (zn <= s.z_max)
        sel = idx[ok]
        zn = zn[ok]
        pts = origin + (zn / cth[sel])[:, None] * dw[sel]
        return Frame(pts, geom[sel].copy(), origin, R, np.array(tray_pos, float), np.array(tray_R, float), int(tick), n,
                     s.name, zn.copy(), dw[sel].copy())


# ----------------------------------------------------------------------------------------------------------------
# sensor timing: frame clock + delivery delay
# ----------------------------------------------------------------------------------------------------------------
class DepthSensor:
    """Frame clock and delivery queue. `step(tick, capture_fn)` returns the list of (capture_tick, payload) delivered at this
    tick. Payloads are fixed-length float vectors (so the whole state packs into one array for snapshots).
    capture_fn() -> payload runs only on ticks where the clock fires. rate 50 = every tick; 30 = 3 of every 5 ticks."""

    SLOTS = 8

    def __init__(self, rate_hz=30.0, latency_ticks=3, payload_len=8):
        self.rate, self.lat, self.P = float(rate_hz), int(latency_ticks), int(payload_len)
        self.width = 1 + self.SLOTS * (2 + self.P)
        self.reset()

    def reset(self):
        self.acc = 0.0
        self.q = []                                                    # [(due, t_capture, payload)]

    def step(self, tick, capture_fn):
        self.acc += self.rate / 50.0
        if self.acc >= 1.0 - 1e-9:
            self.acc -= 1.0
            self.q.append((tick + self.lat, tick, np.asarray(capture_fn(), np.float64)))
        out = [(tc, p) for (due, tc, p) in self.q if due <= tick]
        self.q = [x for x in self.q if x[0] > tick]
        return out

    def pack(self):
        a = np.zeros(self.width)
        a[0] = self.acc
        assert len(self.q) <= self.SLOTS
        for i, (due, tc, p) in enumerate(self.q):
            o = 1 + i * (2 + self.P)
            a[o], a[o + 1], a[o + 2:o + 2 + self.P] = due + 1, tc, p            # due+1 so that 0 means "empty"
        return a

    def unpack(self, a):
        a = np.asarray(a, np.float64)
        self.acc = float(a[0])
        self.q = []
        for i in range(self.SLOTS):
            o = 1 + i * (2 + self.P)
            if a[o] > 0:
                self.q.append((int(round(a[o])) - 1, int(round(a[o + 1])), a[o + 2:o + 2 + self.P].copy()))
