"""gpusense/bench.py -- steady-state ms/frame, CPU ray-cast vs GPU, plus warm-up timings.  SIMULATION ONLY.

  python -m world_model.v2.house.gpusense.bench sensors   # people 160x90 / wrist 160x90 / head 640x360: ray-cast vs GPU (in-process and via the sensing process)
  python -m world_model.v2.house.gpusense.bench mapping   # log-odds update + occupancy mask + EDT(+gradient): numba/scipy vs GPU
Env as in AGENTS.md (CUDA_VISIBLE_DEVICES=6, MUJOCO_GL=egl, thread pins).  Results -> OUT/bench_{sensors,mapping}.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

OUT = Path("/data/local/ss5772/molmospaces/percept/gpu")
TMP = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "gpusense"


def _med(f, n=15, warm=2):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(1e3 * (time.perf_counter() - t))
    return float(np.median(ts)), float(np.min(ts)), float(np.percentile(ts, 90))


def sensors():
    import mujoco
    import torch
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP
    from world_model.v2.house.people_track import camera as CAM
    from world_model.v2.house.grasp import wrist as WR
    from world_model.v2.house.gpusense import core as C
    from world_model.v2.house.gpusense.adapters import Hub, GpuHeadCamera, GpuWristCam
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sim = TeleopSim("h5", seed=0)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(0, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None, back_route=sc.back_route,
                           back_mode=sc.back_mode)
    for _ in range(200):
        sim.tick(op.step(sim.observe()))
    res = {}
    spec160 = CAM.head_spec()
    spec640 = CAM.head_spec(W=640, H=360)
    # ---------------- CPU ray-cast
    hc160, hc640 = CAM.HeadCamera(sim, spec160), CAM.HeadCamera(sim, spec640)
    wl = WR.WristCam(sim.m, "left")
    res["cpu_people_160x90_ms"] = _med(lambda: hc160.render(1, 0), 10)
    res["cpu_wrist_160x90_ms"] = _med(lambda: wl.capture(sim.d, 1, 0), 10)
    res["cpu_head_640x360_ms"] = _med(lambda: hc640.render(1, 0), 3, 1)
    print("cpu", res, flush=True)
    # ---------------- GPU in-process
    path = C.snapshot_model_bytes(sim.m, str(TMP / "bench_model.mjb"))
    pg = np.array(sorted(hc160.person_geoms), int)
    person_of = np.full(sim.m.ngeom, -1, int)
    for g, p in hc160.person_geoms.items():
        person_of[g] = p
    cfgs = [C.GpuCamCfg("head", spec160, CAM.G_CAST, ((pg, 5),), (), CAM.CUTOFF_MULT, hc160.self_geom.copy(), person_of),
            C.GpuCamCfg("head640", spec640, CAM.G_CAST, ((pg, 5),), (), CAM.CUTOFF_MULT, hc160.self_geom.copy(), person_of),
            C.GpuCamCfg("head_static", spec160, (3, 4), (), (), CAM.CUTOFF_MULT * 1.5, None, None, kind="seg"),
            C.GpuCamCfg("wrist_left", wl.spec, WR.MASK_GROUPS, (), tuple(int(g) for g in np.flatnonzero(sim.m.geom_bodyid == wl.body_excl)), WR.CUTOFF_MULT,
                        wl.is_robot_geom.copy(), None)]
    t0 = time.perf_counter()
    core = C.SenseCore(path, cfgs)
    t_ctor = time.perf_counter() - t0
    # cold: first render of each camera (context + framebuffer + first kernels) -- this is the warm-up cost
    state = (sim.d.qpos.copy(), sim.d.mocap_pos.copy(), sim.d.mocap_quat.copy(), sim.d.eq_active.copy())
    core.set_state(*state)
    o160, R160 = hc160.pose()
    ow, Rw = wl.pose(sim.d)
    cold = {}
    for name, (o, R) in (("head", (o160, R160)), ("head640", (o160, R160)), ("wrist_left", (ow, Rw))):
        t = time.perf_counter()
        core.run_job(dict(cam=name, kind="cloud", origin=o, R=R, tick=0, seed=0))
        cold[name] = 1e3 * (time.perf_counter() - t)
    res["gpu_cold_first_job_ms"] = cold
    res["gpu_ctor_s"] = t_ctor
    t = time.perf_counter()
    core.warmup()
    res["gpu_warmup_call_s"] = time.perf_counter() - t
    for name, (o, R), lab in (("head", (o160, R160), "people_160x90"), ("wrist_left", (ow, Rw), "wrist_160x90"), ("head640", (o160, R160), "head_640x360")):
        job = dict(cam=name, kind="cloud", origin=o, R=R, tick=1, seed=0)
        res[f"gpu_{lab}_ms"] = _med(lambda: core.run_job(job), 40)
        c = core._cam(name)
        res[f"gpu_{lab}_render_readback_ms"] = _med(lambda: c.depth_geom(core.d, o, R), 40)        # GL render + readPixels + upload + decode
    res["gpu_people_with_static_ms"] = _med(lambda: core.run_job(dict(cam="head", kind="cloud", origin=o160, R=R160, tick=1, seed=0, want_static="head_static")), 40)
    res["gpu_set_state_ms"] = _med(lambda: core.set_state(*state), 40)
    core.close()
    print("gpu in-process", {k: v for k, v in res.items() if k.startswith("gpu")}, flush=True)
    # ---------------- GPU via the sensing process
    t = time.perf_counter()
    hub = Hub(sim, wrists=("left",), tmpdir=str(TMP))
    res["hub_ready"] = hub.ready
    res["hub_wall_s"] = time.perf_counter() - t
    gh = GpuHeadCamera(sim, spec160, hub)
    res["proc_people_roundtrip_ms"] = _med(lambda: gh.render(1, 0), 40)
    gw = GpuWristCam(sim.m, "left", hub)
    res["proc_wrist_roundtrip_ms"] = _med(lambda: gw.capture(sim.d, 1, 0), 40)

    def _async():
        h = [gh.submit(1, 0) for _ in range(4)]
        for x in h:
            gh.collect(x)
    res["proc_people_pipelined_per_frame_ms"] = tuple(v / 4 for v in _med(_async, 20))
    res["proc_parent_submit_ms"] = _med(lambda: hub.proc.conn.send(("jobs", -1, (sim.d.qpos.copy(), sim.d.mocap_pos.copy(), sim.d.mocap_quat.copy(), sim.d.eq_active.copy()), [])), 40)
    hub.close()
    json.dump(res, open(OUT / "bench_sensors.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


def mapping():
    import math
    import mujoco
    import torch
    from scipy import ndimage
    from world_model.v2.house import build as B
    from world_model.v2.house.loc import mapper as MP
    from world_model.v2.house.loc.estimator import LocCfg, PoseEstimator, scaled
    from world_model.v2.house.teleop import scene as SC, geom as GM
    from world_model.v2.house.gpusense import mapping as GMAP, core as C
    from world_model.v2.tray_rep.perception import depth_sim as D
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    scene = SC.get_scene("h5")
    cfg = MP.MapCfg()
    spec = B.load_house_spec(scene.house, True)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    path = C.snapshot_model_bytes(m, str(TMP / "bench_house.mjb"))
    true = SC.load_occupancy(scene)
    ny, nx = int(round(true.occ.shape[0] * true.res / cfg.res)), int(round(true.occ.shape[1] * true.res / cfg.res))
    poses = MP.exploration_path(np.asarray(scene.out_route, float), cfg)
    est = PoseEstimator(scaled(LocCfg(), cfg.map_loc_scale), seed=cfg.mapping_loc_seed, true_pose0=poses[0])
    cs = D.CamSpec(name="zed_map", mount="world", pos=(0, 0, 0), R=tuple(np.eye(3).ravel()), fovy=cfg.fovy, W=cfg.W, H=cfg.H, z_max=cfg.z_map_max)
    t0 = time.perf_counter()
    core = C.SenseCore(path, [C.GpuCamCfg("map", cs, (3, 4), kind="map")])
    res = {"grid": [ny, nx], "frame": [cfg.H, cfg.W]}
    core.warmup()
    gm = GMAP.GpuMapper(core, cfg, true.x0, true.y0, ny, nx, "map")
    gm.update(poses[1], poses[1])
    GMAP.GpuOccGrid(cfg.res, true.x0, true.y0, gm.occ_mask())            # compiles the EDT kernels
    torch.cuda.synchronize()
    res["gpu_warmup_total_s"] = time.perf_counter() - t0
    gm.L.zero_()
    cam = MP.HeadDepth(m, d, cfg, seed=0)
    L = np.zeros((int(math.ceil((cfg.obs_zhi - cfg.obs_zlo) / cfg.dz)), ny, nx), np.float32)
    sel = [(k, p) for k, p in enumerate(poses[1:], 1) if k % int(round(1.0 / (cfg.frame_hz * 0.02))) == 0][::8][:40]
    ep = []
    for k, p in sel:
        est.update(k * 0.02, p, 0.02)
        ep.append(est.pose.copy())
    # JIT warm for numba
    ok, pts, nofree, _ = cam.capture(sel[0][1])
    o, ex, ey, ez, hit, fx, fy, fz = cam.place(ok, pts, nofree, ep[0])
    MP._integrate(L, cfg.res, true.x0, true.y0, float(o[0]), float(o[1]), float(o[2]), ex, ey, ez, hit, cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ, cfg.l_free, cfg.l_clamp, fx, fy, fz, cfg.dz)
    L[:] = 0
    cap, plc, integ = [], [], []
    for (k, p), e in zip(sel, ep):
        t = time.perf_counter(); ok, pts, nofree, _ = cam.capture(p); t1 = time.perf_counter()
        o, ex, ey, ez, hit, fx, fy, fz = cam.place(ok, pts, nofree, e); t2 = time.perf_counter()
        MP._integrate(L, cfg.res, true.x0, true.y0, float(o[0]), float(o[1]), float(o[2]), ex, ey, ez, hit, cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ, cfg.l_free, cfg.l_clamp, fx, fy, fz, cfg.dz)
        t3 = time.perf_counter()
        cap.append(1e3 * (t1 - t)); plc.append(1e3 * (t2 - t1)); integ.append(1e3 * (t3 - t2))
    res["cpu_capture_ms"] = float(np.median(cap)); res["cpu_place_ms"] = float(np.median(plc)); res["cpu_integrate_ms"] = float(np.median(integ))
    res["cpu_update_total_ms"] = res["cpu_capture_ms"] + res["cpu_place_ms"] + res["cpu_integrate_ms"]
    pm = MP.PerceivedMap(cfg.res, true.x0, true.y0, L, cfg.asdict(), {})
    t = time.perf_counter(); mask = pm.occ_mask(); res["cpu_occ_mask_ms"] = 1e3 * (time.perf_counter() - t)
    t = time.perf_counter(); edt = ndimage.distance_transform_edt(~mask) * cfg.res; res["cpu_edt_ms"] = 1e3 * (time.perf_counter() - t)
    from world_model.v2.house.teleop.scene import OccGrid
    g = OccGrid(cfg.res, true.x0, true.y0, mask, edt)
    t = time.perf_counter(); GM.edt_grad(g); res["cpu_grad_ms"] = 1e3 * (time.perf_counter() - t)
    res["cpu_to_occgrid_total_ms"] = res["cpu_occ_mask_ms"] + res["cpu_edt_ms"] + res["cpu_grad_ms"]
    # GPU
    for exact in (True, False):
        gm.L.zero_()
        capt, intg = [], []
        for (k, p), e in zip(sel, ep):
            torch.cuda.synchronize(); t = time.perf_counter()
            ok_, zn_, nf_ = gm.capture(p); torch.cuda.synchronize(); t1 = time.perf_counter()
            # integrate with the chosen mode
            _orig = GMAP.integrate_gpu
            GMAP.integrate_gpu = lambda *a, _o=_orig, **kw: _o(*a, exact=exact, **kw)
            try:
                gm.place_integrate(ok_, zn_, nf_, e)
            finally:
                GMAP.integrate_gpu = _orig
            torch.cuda.synchronize(); t2 = time.perf_counter()
            capt.append(1e3 * (t1 - t)); intg.append(1e3 * (t2 - t1))
        tag = "exact" if exact else "phased"
        res[f"gpu_capture_ms"] = float(np.median(capt))
        res[f"gpu_place_integrate_{tag}_ms"] = float(np.median(intg))
    torch.cuda.synchronize()
    ts = {"mask": [], "edt": [], "grad": [], "host": []}
    for _ in range(15):
        torch.cuda.synchronize(); t = time.perf_counter()
        mk = gm.occ_mask(); torch.cuda.synchronize(); t1 = time.perf_counter()
        e = GMAP.edt_gpu(mk, cfg.res); torch.cuda.synchronize(); t2 = time.perf_counter()
        gx, gy = GMAP.grad_gpu(e, cfg.res); torch.cuda.synchronize(); t3 = time.perf_counter()
        go = GMAP.GpuOccGrid.__new__(GMAP.GpuOccGrid)
        go.res, go.x0, go.y0, go.occ, go.edt, go.gx, go.gy = cfg.res, true.x0, true.y0, mk, e, gx, gy
        go.to_host(); t4 = time.perf_counter()
        ts["mask"].append(1e3 * (t1 - t)); ts["edt"].append(1e3 * (t2 - t1)); ts["grad"].append(1e3 * (t3 - t2)); ts["host"].append(1e3 * (t4 - t3))
    for k, v in ts.items():
        res[f"gpu_{k}_ms"] = float(np.median(v[2:]))
    res["gpu_update_total_exact_ms"] = res["gpu_capture_ms"] + res["gpu_place_integrate_exact_ms"]
    res["gpu_update_total_phased_ms"] = res["gpu_capture_ms"] + res["gpu_place_integrate_phased_ms"]
    res["gpu_to_occgrid_total_ms"] = res["gpu_mask_ms"] + res["gpu_edt_ms"] + res["gpu_grad_ms"]
    # EDT on the full-size true grid (883 x 1048)
    og = torch.as_tensor(true.occ, device="cuda")
    GMAP.edt_gpu(og, true.res)
    res["gpu_edt_883x1048_ms"] = _med(lambda: (GMAP.edt_gpu(og, true.res), torch.cuda.synchronize()), 15)[0]
    res["cpu_edt_883x1048_ms"] = _med(lambda: ndimage.distance_transform_edt(~true.occ), 5, 1)[0]
    core.close()
    json.dump(res, open(OUT / "bench_mapping.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"sensors": sensors, "mapping": mapping}[sys.argv[1]]()
