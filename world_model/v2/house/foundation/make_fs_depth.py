"""make_fs_depth.py -- derive a 640x360 sequence whose depth is FoundationStereo's (not the MuJoCo buffer): downsample the saved 1280x720 pair 2x (area), run the
stereo backend, write a seq npz bench_pose.py can read (key 'gt_depth' then holds the STEREO depth; truth poses unchanged).   mujoco env.
    python -m world_model.v2.house.foundation.make_fs_depth --src scripted_s0_1280x720.npz --dst scripted_fs_640x360.npz --gpu 5
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

OUT = Path(os.environ.get("FOUNDATION_OUT", "/data/local/ss5772/molmospaces/percept/foundation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="scripted_s0_1280x720.npz")
    ap.add_argument("--dst", default="scripted_fs_640x360.npz")
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--engine", default="graph")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    from world_model.v2.house.foundation.stereo_fs import Client
    z = dict(np.load(OUT / "pose" / a.src))
    N = len(z["left"])
    W, H = 640, 360
    cl = Client(W=W, H=H, gpu=a.gpu, engine=a.engine, iters=a.iters)
    cl.warmup()
    f = float(z["f"]) / 2
    K = dict(f=f, cx=(W - 1) / 2, cy=(H - 1) / 2, W=W, H=H, B=float(z["B"]))
    left, dep, seg, gt = [], [], [], []
    for i in range(0, N, a.stride):
        L = cv2.resize(z["left"][i], (W, H), interpolation=cv2.INTER_AREA)
        R = cv2.resize(z["right"][i], (W, H), interpolation=cv2.INTER_AREA)
        d, ok, ms = cl.depth(L, R, K, float(z["B"]))
        left.append(L)
        dep.append(np.where((d > 0.3) & (d < 5.0), d, 0).astype(np.float32))
        seg.append(cv2.resize(z["gt_seg"][i].astype(np.int16), (W, H), interpolation=cv2.INTER_NEAREST))
        gt.append(cv2.resize(z["gt_depth"][i], (W, H), interpolation=cv2.INTER_NEAREST))
        if i % 50 == 0:
            print(i, f"{ms:.0f} ms", flush=True)
    sel = np.arange(0, N, a.stride)
    keep = {k: z[k][sel] for k in ("t", "tray_p", "tray_R", "cup_p", "cup_R", "origin", "R")}
    np.savez(OUT / "pose" / a.dst, f=f, B=float(z["B"]), left=np.array(left), right=np.array(left), gt_depth=np.array(dep), true_depth=np.array(gt), gt_seg=np.array(seg),
             tray_geoms=z["tray_geoms"], cup_geoms=z["cup_geoms"], **keep)
    err = [np.nanmedian(np.abs(dep[i][(dep[i] > 0) & (gt[i] > 0)] - gt[i][(dep[i] > 0) & (gt[i] > 0)])) for i in range(len(dep))]
    print("median |FS depth - truth| over all valid pixels:", float(np.median(err)))
    cl.close()
    os._exit(0)


if __name__ == "__main__":
    main()
