"""stereo/cfg.py -- configuration of the simulated head ZED stereo sensor.  SIMULATION ONLY.

Geometry = zed_demo/rig.py (rectified pair, parallel axes, baseline 0.12 m ASSUMED ZED 2 nominal, fovy 70 deg) on the head mount of people_track/camera.py
(robot frame (0.05, -0.20955, 1.15 + lift), pitched down; PROPOSED mount, confirm on hardware).  Resolution is the knob that sets the cost:
SGBM time scales with W*H*numDisparities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np

HEAD_POS = (0.05, -0.20955, 1.15)       # robot frame at lift 0 (office compose HEAD_POS_LIFT0)
HEAD_PITCH_DEG = 36.3                   # perception_bridge.HEAD_PITCH_CURRENT (the people tracker / teleop head)
ZED_FOVY = 70.0
ZED_BASELINE = 0.12


@dataclass
class StereoCfg:
    W: int = 640
    H: int = 360
    fovy: float = ZED_FOVY
    baseline: float = ZED_BASELINE
    pitch_deg: float = HEAD_PITCH_DEG
    mount: tuple = HEAD_POS
    lift_mounted: bool = True           # head rides the lift (z = 1.15 + lift)
    rate_hz: float = 10.0
    z_min: float = 0.30                 # ZED 2 min range; sets numDisparities
    z_max: float = 5.0
    backend: str = "sgbm"               # 'sgbm' (CPU, here) | 'foundation' (house/foundation/stereo_fs.py, GPU 5 or 6 only)
    # SGBM
    block: int = 5
    uniqueness: int = 15                # sweep.py: 15 + speckle filter 400/1 = fewer wrong matches at equal fill
    speckle_win: int = 400
    speckle_range: int = 1
    disp12: int = 1
    pad_left: bool = True               # pad the pair on the left by numDisparities (cv2 SGBM leaves that strip invalid)
    prefilter_cap: int = 31
    min_texture: float = 0.0            # >0: invalidate pixels whose 9x9 grey std in the left image is below this (the sensor noise alone gives std ~7)
    mode: str = "hh"                    # 'hh' (full 8-path, zed_demo default) | '3way' | 'sgbm' (5-path)
    num_disp: int | None = None         # None -> ceil(f*B/z_min / 16) * 16
    # twin / sensor
    tex: str = "speckle"                # 'speckle' (all visual geoms) | 'objects' (everything but walls / floors) | 'none' (flat colours: the textureless case)
    read_sigma: float = 1.5             # 8-bit sensor noise (zed_demo.rig.add_sensor_noise)
    shot_k: float = 0.35
    self_mask: bool = True              # FK self-mask: robot / tray / cup pixels (known kinematics) dilated by `self_dilate`
    self_dilate: int = 2
    static_depth: bool = False          # also render the people-free twin depth (background subtraction against the sensor's own prior model)
    keep_pair: bool = False             # return (left, right) RGB (sweeps)
    keep_rgb: bool = False              # return the left RGB (payload estimator needs it)
    gt: bool = False                    # EVALUATION ONLY: also render GT left depth + geom-id segmentation
    cv_threads: int = 1                 # cv2.setNumThreads (1 = honest single-core cost)
    seed: int = 0

    @property
    def f(self) -> float:
        return (self.H / 2.0) / math.tan(math.radians(self.fovy) / 2.0)

    def n_disp(self) -> int:
        if self.num_disp:
            return int(self.num_disp)
        return int(math.ceil(self.f * self.baseline / self.z_min / 16.0) * 16)

    def K(self) -> dict:
        return dict(f=self.f, cx=(self.W - 1) / 2.0, cy=(self.H - 1) / 2.0, W=self.W, H=self.H, B=self.baseline)

    def asdict(self):
        d = asdict(self)
        d["mount"] = list(self.mount)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["mount"] = tuple(d["mount"])
        return StereoCfg(**d)


def head_axes(cfg: StereoCfg):
    """camera-to-robot rotation (columns = image right, up, back) of the pitched head, same construction as zed_demo.rig._zed_cams."""
    p = math.radians(cfg.pitch_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    up = np.array([math.sin(p), 0.0, math.cos(p)])
    z = -fwd
    x = np.cross(up, z)
    return np.stack([x, up, z], 1)
