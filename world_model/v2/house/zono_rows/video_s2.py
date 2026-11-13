"""video_s2.py -- short (<= 10 s) MP4 of the S2 closed-loop run WITH the zonotope rows (frames saved by smoke_s2 run --frames).  SIMULATION ONLY.

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. python -m world_model.v2.house.zono_rows.video_s2 [--tag ""]
Draws the perceived zonotope edges (inflated, as used by the rows) in the MuJoCo scene.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path

OUT = Path(os.environ.get("ZONO_ROWS_OUT", "data/molmospaces/zono_rows"))
os.environ.setdefault("ZED_DEMO_OUT", str(OUT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--W", type=int, default=800)
    ap.add_argument("--H", type=int, default=600)
    a = ap.parse_args()
    import cv2
    from world_model.v2.house.zed_demo import rig as Z, execute as EX, perceive as PC
    from world_model.v2.tray_rep.sets.zono import Zonotope
    frames = pickle.load(open(OUT / f"s2_frames{a.tag}.pkl", "rb"))
    allres = json.loads((OUT / f"s2_results{a.tag}.json").read_text())
    res = allres["with_rows"]
    nores = allres.get("without_rows")
    z = np.load(OUT / "s2_zonos.npz")
    meta = json.loads((OUT / "s2_zonos_meta.json").read_text())["zonos"]
    C, _ = Z.build_composed(pitch_deg=30.0, dt=EX.DT, textures=True)
    m = C.m
    dv = mujoco.MjData(m)
    r = mujoco.Renderer(m, a.H, a.W, max_geom=60000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (4.1, 5.05, 0.80)
    cam.distance, cam.elevation, cam.azimuth = 2.3, -27.0, 70.0
    edges = []
    for i, mt in enumerate(meta):
        Zi = Zonotope(z[f"c{i}"], z[f"G{i}"])
        lo, hi = Zi.interval_hull()
        if hi[2] < 0.72 or lo[2] > 1.6 or not (3.0 < lo[0] < 5.5 and 4.4 < lo[1] < 6.2):
            continue
        V, E = PC.zono_edges(Zi)
        col = (1.0, 0.45, 0.1, 1.0) if mt["gt_key"] == "bottle_a" else (0.2, 0.7, 1.0, 1.0)
        edges += [(V[i0], V[j0], col) for i0, j0 in E]
    tr = np.array(res["trace"])
    out = OUT / f"s2_with_rows{a.tag}.mp4"
    w = imageio.get_writer(str(out), fps=25, codec="libx264", quality=7, macro_block_size=1)
    fr = frames[::a.stride]
    for n, f in enumerate(fr):
        dv.qpos[:] = f["qpos"]; dv.mocap_pos[:] = f["mocap_pos"]; dv.mocap_quat[:] = f["mocap_quat"]
        mujoco.mj_forward(m, dv)
        r.update_scene(dv, camera=cam)
        sc = r.scene
        for p0, p1, col in edges:
            if sc.ngeom >= sc.maxgeom - 1:
                break
            g = sc.geoms[sc.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(col, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, p0, p1)
            sc.ngeom += 1
        img = r.render().copy()
        t = n * a.stride * EX.FRAME_EVERY * EX.DT - EX.SETTLE_S
        cv2.putText(img, "FL-MPC + perceived zonotope HOCBF rows (sim)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"naive straight-line target is INSIDE the bottle | t={t:5.1f}s", (12, a.H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        txt = (f"run: tray-bottle clearance >= {1000 * res['bottle_min_dist_by_robot_part_m']['T']:.0f} mm, bottle peak force {res['bottle_peak_normal_force_N']:.1f} N"
               + (f"  (same run, no rows: tray crushes the bottle, {nores['bottle_peak_normal_force_N']:.0f} N)" if nores else ""))
        cv2.putText(img, txt, (12, a.H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        w.append_data(img)
    w.close()
    print("wrote", out, len(fr), "frames", len(fr) / 25, "s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
