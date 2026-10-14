"""stereo/study_static.py -- recompute the stereo head_frame (sg_*) of cached study frames with the people-free TWIN depth as the static model (zs).  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.study_static --tags speckle_s1_640x360 ...
The first study version cast the static model on the COLLISION hulls (as people_track.camera.static_depth does for the ray-cast camera); the stereo sensor sees the
VISUAL geometry, so rugs / door slabs / plants that the hulls do not contain showed up as foreground 'objects' and made false tracks.  Background subtraction against the
sensor's own people-free render removes that model mismatch (a real system subtracts a map built by the same sensor).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from .cfg import StereoCfg
from . import twin as TW, adapters as AD
from .sensor import StereoSensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    from world_model.v2.house.teleop.sim import TeleopSim
    for tag in a.tags:
        tex, sd, res = tag.split("_")
        seed = int(sd[1:])
        W, H = (int(v) for v in res.split("x"))
        snap = np.load(TW.OUT / "snap" / f"carry_s{seed}.npz", allow_pickle=True)
        sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time")
        cfg = StereoCfg(W=W, H=H, tex=tex, static_depth=True, self_mask=False, gt=False)
        s = StereoSensor(TW.build_twin_model(sim, cfg), cfg)
        t0 = time.time()
        files = sorted((TW.OUT / "cache" / tag).glob("fr_*.npz"))
        for f in files:
            z = dict(np.load(f))
            i = int(z["i"])
            s.load(snap["qpos"][i], snap["mpos"][i], snap["mquat"][i], snap["eq"][i])
            s._place_cams()
            sd_img = s.static_depth()
            fr = dict(tick=int(z["k"]), depth=np.where(z["depth_mm"] > 0, z["depth_mm"] / 1000.0, np.nan).astype(np.float32), origin=z["origin"], R=z["R"],
                      self_mask=(z["cls"] > 0), static_depth=sd_img)
            # re-apply the dilated self mask exactly as the sensor does
            import cv2
            fr["self_mask"] = cv2.dilate((z["cls"] > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            rows, cols, _, _ = AD.grid_rays(cfg, 4)
            pg = z["person_g"][np.ix_(rows, cols)]
            hf = AD.frame_to_head_frame(fr, cfg, stride=4, base_true=snap["base"][i], person_grid=np.where(pg == 255, -1, pg))
            for k in ("origin", "R", "pts", "z", "pix", "zs", "person"):
                z[f"sg_{k}"] = hf[k]
            z["static_mm"] = np.nan_to_num(sd_img * 1000, nan=0).clip(0, 65535).astype(np.uint16)
            np.savez_compressed(f, **z)
        print(f"[static] {tag}: {len(files)} frames in {time.time() - t0:.0f}s", flush=True)
        s.close()
    os._exit(0)


if __name__ == "__main__":
    main()
