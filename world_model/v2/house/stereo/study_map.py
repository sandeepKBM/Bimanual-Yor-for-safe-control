"""stereo/study_map.py -- the mapper (loc/mapper.py) with the stereo depth as its sensor.  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.study_map --tex speckle objects none [--res 640 360]

Same scripted mapping pass (exploration_path, PoseEstimator drift, log-odds integration, MapCfg) as loc.mapper.build_map; only the depth sensor differs:
  raycast : loc.mapper.HeadDepth (ray-cast + CamSpec noise)             stereo : adapters.StereoHeadDepth (render pair -> SGBM on the house-only twin)
Reports loc.mapper.map_error vs the true occupancy grid for each (recall / false-free / spurious / signed EDT error).  A stereo miss is NOT free-space evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from .cfg import StereoCfg
from . import twin as TW, adapters as AD
from .sensor import StereoSensor


def build_map_stereo(scene, cfg, scfg: StereoCfg, loc_cfg=None, verbose=True):
    from world_model.v2.house.loc import mapper as MP
    from world_model.v2.house.loc.estimator import LocCfg, PoseEstimator, scaled
    from world_model.v2.house.teleop import scene as SC
    t0 = time.time()
    m_house = TW.build_house_twin(scene, scfg)
    sens = StereoSensor(m_house, scfg, root=None)
    d_ref = mujoco.MjData(m_house)
    sens.load(d_ref.qpos, d_ref.mocap_pos, d_ref.mocap_quat, d_ref.eq_active)
    from .cfg import head_axes
    sens.set_center_pose(np.array(scene.waypoints['lab_start'][:2] + (1.15,), float), head_axes(scfg))      # any pose: the warm-up needs one
    sens.warmup()
    true = SC.load_occupancy(scene)
    ny, nx = int(round(true.occ.shape[0] * true.res / cfg.res)), int(round(true.occ.shape[1] * true.res / cfg.res))
    L = np.zeros((int(np.ceil((cfg.obs_zhi - cfg.obs_zlo) / cfg.dz)), ny, nx), np.float32)
    # the mapper's own sensor object provides axes() / place() / spec; capture() is replaced by the stereo one
    mh = mujoco.MjModel.from_xml_string("<mujoco/>")
    hd = MP.HeadDepth(mh, mujoco.MjData(mh), cfg, seed=cfg.seed)
    sd = AD.StereoHeadDepth(hd, sens, stride=max(scfg.W // cfg.W, 1))
    route = np.asarray(scene.out_route, float)
    poses = MP.exploration_path(route, cfg)
    est = PoseEstimator(scaled(loc_cfg or LocCfg(), cfg.map_loc_scale), seed=cfg.mapping_loc_seed, true_pose0=poses[0])
    every = int(round(1.0 / (cfg.frame_hz * 0.02)))
    nf = 0
    for k, p in enumerate(poses[1:], 1):
        est.update(k * 0.02, p, 0.02)
        if k % every:
            continue
        ok, pts, nofree, _ = sd.capture(p)
        o, ex, ey, ez, hit, fx, fy, fz = hd.place(ok, pts, nofree, est.pose)
        MP._integrate(L, cfg.res, true.x0, true.y0, float(o[0]), float(o[1]), float(o[2]), ex, ey, ez, hit, cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ, cfg.l_free,
                      cfg.l_clamp, fx, fy, fz, cfg.dz)
        nf += 1
    ms = float(np.mean([h["total"] for h in sd.ms]))
    meta = dict(frames=nf, wall_s=time.time() - t0, capture_ms=ms, map_ate=est.errors(), route=route.tolist())
    if verbose:
        print(f"[map-stereo {scfg.tex}] {nf} frames, {meta['wall_s']:.0f}s wall, {ms:.0f} ms/capture", flush=True)
    sens.close()
    return MP.PerceivedMap(cfg.res, true.x0, true.y0, L, cfg.asdict(), meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", nargs="*", default=["speckle", "objects", "none"])
    ap.add_argument("--res", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--no-raycast", action="store_true")
    a = ap.parse_args()
    from world_model.v2.house.loc import mapper as MP
    from world_model.v2.house.teleop import scene as SC
    scene = SC.get_scene("h5")
    cfg = MP.MapCfg(W=a.res[0] // 2, H=a.res[1] // 2)
    true = SC.load_occupancy(scene)
    route = np.asarray(scene.out_route, float)
    res = {}
    pj = TW.OUT / "map_study.json"
    if pj.exists():
        res = json.loads(pj.read_text())
    if not a.no_raycast:
        pm = MP.build_map(scene, cfg, verbose=True)
        res["raycast"] = dict(err=MP.map_error(pm.to_occgrid(), true, route), meta={k: v for k, v in pm.meta.items() if k != "route"})
        pm.save(TW.OUT / "map_raycast.npz")
        print("raycast", json.dumps(res["raycast"]["err"]), flush=True)
    for tex in a.tex:
        scfg = StereoCfg(W=a.res[0], H=a.res[1], tex=tex, self_mask=False)
        pm = build_map_stereo(scene, cfg, scfg)
        res[f"stereo_{tex}"] = dict(err=MP.map_error(pm.to_occgrid(), true, route), meta={k: v for k, v in pm.meta.items() if k != "route"})
        pm.save(TW.OUT / f"map_stereo_{tex}.npz")
        print(f"stereo_{tex}", json.dumps(res[f"stereo_{tex}"]["err"]), flush=True)
    (TW.OUT / "map_study.json").write_text(json.dumps(res, indent=1, default=float))
    os._exit(0)


if __name__ == "__main__":
    main()
