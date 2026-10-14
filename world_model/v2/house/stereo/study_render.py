"""stereo/study_render.py -- STAGE 1 of the calibration study: replay recorded snapshots through (a) the textured stereo twin (SGBM) and (b) the ray-cast
head cameras, cache everything the analysis needs.  SIMULATION ONLY (GPU for the 3 renders per frame; run with CUDA_VISIBLE_DEVICES=5).

    python -m world_model.v2.house.stereo.study_render --seed 0 --tex speckle --every 1 [--res 640 360]

Per snapshot i -> <OUT>/cache/<tag>/fr_<i>.npz with:
  depth_mm, gt_mm (uint16, 0 = invalid)   stereo depth / MuJoCo GT depth of the LEFT camera (full stereo resolution)
  cls (uint8) FK class, person_g (uint8: person index, 255 none; evaluation only), left (RGB)
  sg_*   head_frame of the stereo depth on the tracker grid (adapters.frame_to_head_frame, stride 4)
  rz_*   head_frame of the RAY-CAST camera with the SAME geometry (fovy 70, 160x90 grid, native_H = stereo H) and the depth_sim noise model
  rd_*   head_frame of the RAY-CAST camera of the default tracker spec (fovy 55, 160x90, native_H 720)
  vis_zed (npeople, 3) = (in_fov, line-of-sight, range) of the ZED-geometry camera (EVALUATION ONLY)
  FK tray prior: tray_fk_p / tray_fk_R in the BASE frame (ArmFK on the arm encoders (+ their noise) + lift), qL qR lift
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from .cfg import StereoCfg
from . import twin as TW, adapters as AD
from .sensor import StereoSensor

ENC_Q, ENC_V = 1e-4, 2e-3
HEAD_SEED = 7


def rc_head_frame(cam, sim, k, seed):
    """the ray-cast head_frame (same keys / filter as ocs.record.LiveSensors.head_frame)."""
    fr = cam.render(k, seed)
    z = fr["pts"][:, 2]
    keep = (z > 0.10) & (z < 2.1)
    sub = dict(origin=fr["origin"], R=fr["R"], pts=fr["pts"][keep], z=fr["z"][keep], pix=fr["pix"][keep], person=fr["person"][keep], geom=None, tick=k)
    zs = cam.static_depth(sub, np.arange(len(sub["pts"])))
    return dict(origin=fr["origin"].copy(), R=fr["R"].copy(), pts=sub["pts"].astype(np.float32), z=sub["z"].astype(np.float32), pix=sub["pix"].astype(np.uint8),
                zs=zs.astype(np.float32), person=sub["person"].astype(np.int8))


def pack(prefix, hf, out):
    for k in ("origin", "R", "pts", "z", "pix", "zs", "person"):
        out[f"{prefix}_{k}"] = hf[k]


def run(seed, tex, every, res, out_root, rate_stride=4, max_frames=None, gt=True):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.people_track import camera as CAM, detectors as DET
    from world_model.v2.house.ocs.views import ArmFK
    t0 = time.time()
    snap = np.load(Path(out_root) / "snap" / f"carry_s{seed}.npz", allow_pickle=True)
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time")
    cfg = StereoCfg(W=res[0], H=res[1], tex=tex, gt=gt, keep_rgb=True, static_depth=True, seed=seed)
    m_tw = TW.build_twin_model(sim, cfg)
    sens = StereoSensor(m_tw, cfg)
    d = sim.d
    sens.load(snap["qpos"][0], snap["mpos"][0], snap["mquat"][0], snap["eq"][0])
    sens.warmup()
    fk = ArmFK(sim)
    cam_z = CAM.HeadCamera(sim, replace(CAM.head_spec(), fovy=cfg.fovy, W=160, H=90, native_H=cfg.H, name="head_zed"))
    cam_d = CAM.HeadCamera(sim, CAM.head_spec(W=160, H=90))
    vis = DET.Visibility(cam_z)
    names = [p.name for p in sim.scene.people]
    pers_geom = {g: names.index(b) for g, b in sens.person_geoms.items() if b in names}
    tag = f"{tex}_s{seed}_{res[0]}x{res[1]}"
    od = Path(out_root) / "cache" / tag
    od.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(70_000 + seed)
    n = len(snap["t"])
    idx = list(range(0, n, every))[:max_frames]
    print(f"[render] {tag}: {len(idx)} frames; setup {time.time() - t0:.0f}s", flush=True)
    ms = []
    _rows, _cols, gx, gy = AD.grid_rays(cfg, rate_stride)
    for j, i in enumerate(idx):
        d.qpos[:], d.mocap_pos[:], d.mocap_quat[:], d.eq_active[:] = snap["qpos"][i], snap["mpos"][i], snap["mquat"][i], snap["eq"][i]
        d.qvel[:] = 0.0
        mujoco.mj_forward(sim.m, d)
        k = int(snap["k"][i])
        sens.load(snap["qpos"][i], snap["mpos"][i], snap["mquat"][i], snap["eq"][i])
        fr = sens.capture(k, float(snap["t"][i]))
        ms.append(fr["ms"]["total"])
        ctl = sim.ctl
        qL, qR = d.qpos[ctl.lq] + rng.normal(0, ENC_Q, 6), d.qpos[ctl.rq] + rng.normal(0, ENC_Q, 6)
        lift = sim.lift_actual() + rng.normal(0, 1e-4)
        tp, tR = fk.tray(qL, qR, lift)
        pg = np.full(fr["gt_seg"].shape, 255, np.uint8)
        for g, pi in pers_geom.items():
            pg[fr["gt_seg"] == g] = pi
        hf_s = AD.frame_to_head_frame(fr, cfg, stride=rate_stride, sim=sim, base_true=snap["base"][i], person_grid=np.where(pg[np.ix_(_rows, _cols)] == 255, -1, pg[np.ix_(_rows, _cols)]))
        hf_z = rc_head_frame(cam_z, sim, k, HEAD_SEED)
        hf_d = rc_head_frame(cam_d, sim, k, HEAD_SEED)
        vz = np.array([vis.of(sim, xy) for xy in snap["ppl_xy"][i]], float).reshape(-1, 3)
        o = dict(i=i, k=k, t=float(snap["t"][i]), depth_mm=np.nan_to_num(fr["depth"] * 1000, nan=0).clip(0, 65535).astype(np.uint16),
                 gt_mm=np.nan_to_num(fr["gt_depth"] * 1000, nan=0).clip(0, 65535).astype(np.uint16), cls=fr["cls"], person_g=pg, left=fr["left"],
                 origin=fr["origin"], R=fr["R"], vis_zed=vz, static_mm=np.nan_to_num(fr["static_depth"] * 1000, nan=0).clip(0, 65535).astype(np.uint16), tray_fk_p=tp, tray_fk_R=tR, qL=qL, qR=qR, lift=lift, ms=np.array([fr["ms"][k_] for k_ in ("render", "stereo", "post", "total")]))
        pack("sg", hf_s, o)
        pack("rz", hf_z, o)
        pack("rd", hf_d, o)
        np.savez_compressed(od / f"fr_{i:04d}.npz", **o)
        if j % 20 == 0:
            print(f"[render] {tag} {j}/{len(idx)}  {np.mean(ms):.0f} ms/frame  elapsed {time.time() - t0:.0f}s", flush=True)
    sens.close()
    print(f"[render] {tag} done {len(idx)} frames in {time.time() - t0:.0f}s -> {od}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tex", default="speckle")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--res", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=str(TW.OUT))
    a = ap.parse_args()
    run(a.seed, a.tex, a.every, tuple(a.res), a.out, max_frames=a.max_frames)
    os._exit(0)          # skip the EGL teardown exceptions


if __name__ == "__main__":
    main()
