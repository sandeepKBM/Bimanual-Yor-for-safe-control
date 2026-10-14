"""stereo/backends.py -- pluggable stereo depth backends behind ONE interface.  SIMULATION ONLY.

    backend.depth(left_rgb, right_rgb, K, baseline) -> (depth_m (H,W) float32 [NaN where invalid], valid_mask (H,W) bool, ms float)
    backend.warmup()  -> pay every start-up cost (cv2 allocation, CUDA context, weights, first-call autotune) before operation; idempotent
    backend.name

  'sgbm'        cv2.StereoSGBM (CPU, this package; reuses zed_demo/stereo parameters)
  'foundation'  house/foundation/stereo_fs.py (another agent): FoundationStereo in its own process.  Expected module API (any one of):
                    stereo_fs.depth(left_rgb, right_rgb, K, baseline) -> (depth_m, valid_mask, ms)           (module-level function)
                    stereo_fs.Client(...).depth(...) / .warmup() / .ready                                    (client object)
                stereo_fs signals READY after its own warm-up; we wait for it in warmup().
GPU: stay on 5 or 6 (0-3 belong to another project, 4 / 7 are saturated by another user).
"""
from __future__ import annotations

import importlib
import time

import numpy as np
import cv2

from .cfg import StereoCfg

_MODES = {"hh": cv2.STEREO_SGBM_MODE_HH, "3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY, "sgbm": cv2.STEREO_SGBM_MODE_SGBM}


class SGBMBackend:
    name = "sgbm"

    def __init__(self, cfg: StereoCfg):
        self.cfg = cfg
        cv2.setNumThreads(int(cfg.cv_threads))
        bs = cfg.block
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=cfg.n_disp(), blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=cfg.disp12,
                                          preFilterCap=cfg.prefilter_cap, uniquenessRatio=cfg.uniqueness, speckleWindowSize=cfg.speckle_win,
                                          speckleRange=cfg.speckle_range, mode=_MODES[cfg.mode])
        self.ready = False

    def warmup(self, n=3):
        rng = np.random.default_rng(0)
        g = rng.integers(0, 255, (self.cfg.H, self.cfg.W, 3), dtype=np.uint8)
        for _ in range(n):
            self.depth(g, np.roll(g, -4, axis=1), self.cfg.K(), self.cfg.baseline)
        self.ready = True

    def depth(self, left, right, K, baseline):
        c = self.cfg
        t = time.perf_counter()
        gl, gr = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        p = c.n_disp() if c.pad_left else 0
        if p:
            # OpenCV SGBM leaves the leftmost numDisparities columns invalid (no room for the full disparity range).  Replicate-pad both images on the left
            # and crop: pixels there can then match at any disparity <= x (a real stereo SDK has no such strip).
            gl = cv2.copyMakeBorder(gl, 0, 0, p, 0, cv2.BORDER_REPLICATE)
            gr = cv2.copyMakeBorder(gr, 0, 0, p, 0, cv2.BORDER_REPLICATE)
        disp = self.sgbm.compute(gl, gr)[:, p:].astype(np.float32) / 16.0
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(disp > 0.5, K["f"] * baseline / disp, np.nan)
        z[(z < c.z_min) | (z > c.z_max)] = np.nan
        if c.min_texture > 0:                 # confidence gate: a flat patch cannot be matched, whatever the matcher answers (it answers with noise)
            g = gl[:, p:].astype(np.float32)
            mu = cv2.boxFilter(g, -1, (9, 9))
            sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (9, 9)) - mu * mu, 0.0))
            z[sd < c.min_texture] = np.nan
        z = z.astype(np.float32)
        return z, np.isfinite(z), 1e3 * (time.perf_counter() - t)


class FoundationBackend:
    name = "foundation"

    def __init__(self, cfg: StereoCfg, **kw):
        mod = importlib.import_module("world_model.v2.house.foundation.stereo_fs")      # raises ImportError until the other agent's module exists
        self.mod = mod
        kw = {"W": cfg.W, "H": cfg.H, **kw}
        self.obj = mod.Client(**kw) if hasattr(mod, "Client") else None       # gpu=5|6 (Client default 5); the server pins its own CUDA_VISIBLE_DEVICES
        self.cfg = cfg
        self.ready = False

    def warmup(self, timeout=300.0):
        o = self.obj
        if o is not None and hasattr(o, "warmup"):
            o.warmup()
        t0 = time.time()
        while o is not None and hasattr(o, "ready") and not o.ready and time.time() - t0 < timeout:
            time.sleep(0.2)
        z = np.zeros((self.cfg.H, self.cfg.W, 3), np.uint8)
        self.depth(z, z, self.cfg.K(), self.cfg.baseline)                # first call pays any lazy init
        self.ready = True

    def depth(self, left, right, K, baseline):
        fn = self.obj.depth if self.obj is not None else self.mod.depth
        z, ok, ms = fn(left, right, K, baseline)
        z = np.asarray(z, np.float32).copy()
        ok = np.asarray(ok, bool) & np.isfinite(z) & (z >= self.cfg.z_min) & (z <= self.cfg.z_max)
        z[~ok] = np.nan
        return z, ok, float(ms)


def make_backend(cfg: StereoCfg, **kw):
    if cfg.backend == "sgbm":
        return SGBMBackend(cfg)
    if cfg.backend == "foundation":
        return FoundationBackend(cfg, **kw)
    raise ValueError(cfg.backend)
