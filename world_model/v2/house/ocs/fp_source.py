"""ocs/fp_source.py -- FoundationPose (model-free) measurement source for ocs.anchored.  SIMULATION ONLY.

RecordedFPSource replays the per-tick output of ocs/fp_pass.py (wrist-camera RGB-D re-rendered from a recording, non-privileged support-plane
mask, FoundationPose register/track, pose converted to the FK tray frame with the yaw of the symmetric cup canonicalised):
    fp_s{seed}.npz: ticks (M,), ok (M,2), p_tray (M,2,3), R_tray (M,2,3,3), mask_px (M,2), mode (M,2), ms (M,2), meta (json)
A tick with both sides ok averages the positions and takes the rotation of the side with the larger mask.  The set radius / rotation bound
come from the calibration keys <prefix>_fp_pos / <prefix>_fp_rot (split conformal on the calibration seeds, exp_e234.fit_calib) or the
priors.  The measurement is SE(3): AnchoredProducer then uses the measured rotation instead of the upright contact prior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .state import Source
from .anchored import Measurement, MeasurementSource

SIDES = ("left", "right")


class RecordedFPSource(MeasurementSource):
    name = "foundationpose"
    source = Source.WRIST

    def __init__(self, npz_path, calib=None, prefix: str = "cup", pos_prior: float = 0.006, rot_prior: float = 0.05):
        d = np.load(npz_path, allow_pickle=False)
        self.path = str(npz_path)
        self.ticks = np.asarray(d["ticks"], np.int64)
        self.ok, self.p, self.R, self.px = d["ok"], d["p_tray"], d["R_tray"], d["mask_px"]
        self.mode = d["mode"]
        self.meta = json.loads(str(d["meta"])) if "meta" in d else {}
        self.row = {int(k): i for i, k in enumerate(self.ticks)}
        self.calib, self.prefix, self.pos_prior, self.rot_prior = calib, prefix, pos_prior, rot_prior

    def measure(self, view, ws, model, t):
        i = self.row.get(int(view.k))
        if i is None or not self.ok[i].any():
            return None
        sides = [j for j in range(2) if self.ok[i, j]]
        p = np.mean([self.p[i, j] for j in sides], 0)
        jR = max(sides, key=lambda j: int(self.px[i, j]))
        R = np.asarray(self.R[i, jR], float)
        if model.on_support:
            p = np.array([p[0], p[1], 0.0])
        c = self.calib
        r = c.get(f"{self.prefix}_fp_pos", self.pos_prior) if c is not None else self.pos_prior
        rot = c.get(f"{self.prefix}_fp_rot", self.rot_prior) if c is not None else self.rot_prior
        return Measurement(t, p, float(r), R, float(rot), self.source,
                           dict(fp_p=p.copy(), fp_R=R.copy(), fp_t=float(t), fp_sides=[SIDES[j] for j in sides],
                                fp_mode=[int(self.mode[i, j]) for j in sides]))


def fp_path(seed, fp_dir=None) -> Path:
    import os
    base = Path(fp_dir or os.environ.get("OCS_FP_DIR", "data/molmospaces/ocs/fp"))
    return base / f"fp_s{int(seed)}.npz"
