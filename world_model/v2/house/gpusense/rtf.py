"""gpusense/rtf.py -- teleop real-time factor with the perception load (SIMULATION ONLY).

The teleop carry episode (run.run_episode's loop, shield ON) is driven here with the perception sensors added OUTSIDE the owner's files:
  people head camera 30 Hz (3-tick latency), both wrist cameras 30 Hz (3-tick latency; worst case = a grasp is active the whole run),
  occupancy map 5 Hz (log-odds update; at each update the occupancy mask + EDT + gradient are produced).
Conditions:   none   no perception (reference)
              cpu    ray-cast sensors + numba/scipy mapping, in the sim thread (what the code does today)
              gpu    sensing PROCESS, asynchronous: submit at the capture tick, collect at the delivery tick (render hides in the latency)
              gpusync  sensing process, submit + immediate collect (no overlap)
RTF = simulated seconds / wall seconds of the loop (sim.tick + shield), perception stage time reported separately.

  python -m world_model.v2.house.gpusense.rtf --cond none,cpu,gpu,gpusync --max-s 30
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

OUT = Path("/data/local/ss5772/molmospaces/percept/gpu")
TICK = 0.02
LAT = 3


def due(k, hz):
    r = hz * TICK
    return math.floor((k + 1) * r) > math.floor(k * r)


def run(cond, max_s, seed=0, scene="h5", map_hz=5.0, cam_hz=30.0):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP, shield as SH
    from world_model.v2.house.people_track import camera as CAM
    from world_model.v2.house.grasp import wrist as WR
    from world_model.v2.house.loc import mapper as MP
    sim = TeleopSim(scene, grasped=True, seed=seed, with_people=True, counter=False, desk=False)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    shield = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw)
    shield.docks = [wp["cafe_counter_pick"][:2]]
    info = dict(cond=cond)
    hub = None
    cfg = MP.MapCfg()
    if cond == "cpu":
        head = CAM.HeadCamera(sim)
        wcams = {s: WR.WristCam(sim.m, s) for s in ("left", "right")}
        import mujoco
        from world_model.v2.house.loc.estimator import LocCfg
        hd = MP.HeadDepth(sim.m, sim.d, cfg, seed=0)
        ref = sim.occ
        ny, nx = int(round(ref.occ.shape[0] * ref.res / cfg.res)), int(round(ref.occ.shape[1] * ref.res / cfg.res))
        L = np.zeros((int(math.ceil((cfg.obs_zhi - cfg.obs_zlo) / cfg.dz)), ny, nx), np.float32)
        # numba JIT before the clock starts (warm-up of the CPU path is also excluded)
        ok, pts, nofree, _ = hd.capture(sim.pose())
        o, ex, ey, ez, hit, fx, fy, fz = hd.place(ok, pts, nofree, sim.pose())
        MP._integrate(L, cfg.res, ref.x0, ref.y0, float(o[0]), float(o[1]), float(o[2]), ex, ey, ez, hit, cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ, cfg.l_free, cfg.l_clamp, fx, fy, fz, cfg.dz)
        L[:] = 0
    elif cond in ("gpu", "gpusync"):
        from world_model.v2.house.gpusense.adapters import Hub, GpuHeadCamera, GpuWristCam, MapService
        t = time.perf_counter()
        hub = Hub(sim, wrists=("left", "right"), map_cfg=cfg, tmpdir=str(OUT))
        info["hub_ready_wall_s"] = time.perf_counter() - t
        info["hub_ready"] = hub.ready
        head = GpuHeadCamera(sim, None, hub)
        head.want_static = False                  # sensor stage only (the CPU path's render() has no static pass either)
        wcams = {s: GpuWristCam(sim.m, s, hub) for s in ("left", "right")}
        msvc = MapService(sim, hub)
        # first-use costs (context, graph capture, JIT) are in Hub READY; one throw-away round trip each so the clock starts hot
        head.render(0, 0)
        for w in wcams.values():
            w.capture(sim.d, 0, 0)
        msvc.update(sim.pose(), publish=True)
        msvc.poll(block=True)
    n = int(max_s / TICK)
    pend_h, pend_w, pend_m = deque(), deque(), deque()
    t_perc = 0.0
    t_stall = 0.0
    n_head = n_wrist = n_map = n_grid = 0
    per_tick = []
    t_start = time.perf_counter()
    for k in range(n):
        t0 = time.perf_counter()
        obs = sim.observe()
        cmd = op.step(obs)
        safe, _ = shield.filter(sim, cmd)
        sim.tick(safe)
        t1 = time.perf_counter()
        if cond != "none":
            kk = sim.k
            # ---- deliveries (async)
            if cond == "gpu":
                while pend_h and pend_h[0][0] <= kk:
                    _, h = pend_h.popleft()
                    ts = time.perf_counter(); head.collect(h); t_stall += time.perf_counter() - ts
                while pend_w and pend_w[0][0] <= kk:
                    _, s, h = pend_w.popleft()
                    ts = time.perf_counter(); wcams[s].finish_capture(h); t_stall += time.perf_counter() - ts
                if pend_m:
                    g = msvc.poll(block=False)
                    if g is not None:
                        n_grid += 1
                    if not msvc.pending:
                        pend_m.clear()
            # ---- captures
            if due(k, cam_hz):
                n_head += 1
                if cond == "cpu":
                    head.render(kk, 7)
                elif cond == "gpu":
                    pend_h.append((kk + LAT, head.submit(kk, 7)))
                else:
                    head.render(kk, 7)
                for s, w in wcams.items():
                    n_wrist += 1
                    if cond == "cpu":
                        w.capture(sim.d, kk, 0)
                    elif cond == "gpu":
                        pend_w.append((kk + LAT, s, w.submit_capture(sim.d, kk, 0)))
                    else:
                        w.capture(sim.d, kk, 0)
            if due(k, map_hz):
                n_map += 1
                p = sim.pose()
                if cond == "cpu":
                    ok, pts, nofree, _ = hd.capture(p)
                    o, ex, ey, ez, hit, fx, fy, fz = hd.place(ok, pts, nofree, p)
                    MP._integrate(L, cfg.res, ref.x0, ref.y0, float(o[0]), float(o[1]), float(o[2]), ex, ey, ez, hit, cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ, cfg.l_free, cfg.l_clamp, fx, fy, fz, cfg.dz)
                    pm = MP.PerceivedMap(cfg.res, ref.x0, ref.y0, L, cfg.asdict(), {})
                    g = pm.to_occgrid()
                    from world_model.v2.house.teleop import geom as GM
                    GM.edt_grad(g)
                    n_grid += 1
                elif cond == "gpu":
                    msvc.update(p, publish=True)
                    pend_m.append(1)
                else:
                    msvc.update(p, publish=True)
                    msvc.poll(block=True)
                    n_grid += 1
        t2 = time.perf_counter()
        per_tick.append((t1 - t0, t2 - t1))
    wall = time.perf_counter() - t_start
    pt = np.array(per_tick)
    info.update(sim_s=n * TICK, wall_s=wall, rtf=n * TICK / wall, ms_per_tick=1e3 * wall / n, sim_tick_ms=1e3 * pt[:, 0].mean(), perception_ms_per_tick=1e3 * pt[:, 1].mean(),
                perception_ms_p99_tick=1e3 * float(np.percentile(pt[:, 1], 99)), perception_ms_max_tick=1e3 * float(pt[:, 1].max()), tick_over_20ms=int(((pt.sum(1)) > TICK).sum()),
                n_head=n_head, n_wrist=n_wrist, n_map=n_map, n_grid=n_grid, stall_ms_total=1e3 * t_stall)
    if hub is not None:
        for role, pr in hub.procs.items():
            st = pr.stats
            info[f"proc_{role}"] = dict(job_ms_mean=float(np.mean(st["job_ms"])) if st["job_ms"] else None, n_jobs=len(st["job_ms"]),
                                        rtt_ms_mean=float(np.mean(st["rtt_ms"])) if st["rtt_ms"] else None,
                                        rtt_ms_p95=float(np.percentile(st["rtt_ms"], 95)) if st["rtt_ms"] else None,
                                        stall_ms_total=float(np.sum(st["stall_ms"])) if st["stall_ms"] else 0.0)
        hub.close()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="none,cpu,gpu,gpusync")
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--out", default=str(OUT / "rtf.json"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for c in a.cond.split(","):
        r = run(c, a.max_s)
        print(json.dumps(r, default=str), flush=True)
        res.append(r)
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
