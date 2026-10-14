"""stereo/bench.py -- cost of the head-stereo sensor.  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.bench parts      # ms per frame by component and resolution (render, backend, self-mask seg), 1 CPU thread for cv2
    python -m world_model.v2.house.stereo.bench rtf        # sim real-time factor: no stereo | in-process (blocking) | separate process (StereoLink), 10 Hz

parts: each frame = 2 anti-aliased RGB renders + backend (SGBM) + 1 segmentation render (the FK self-mask); mean over n frames after warm-up.
rtf:   the teleop carry loop (operator + shield + TeleopSim.tick at 50 Hz); factor = sim seconds / wall seconds.  GPU: CUDA_VISIBLE_DEVICES 5 or 6 only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from .cfg import StereoCfg
from . import twin as TW
from .sensor import StereoSensor

TICK = 0.02


def _sim(seed=0):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP, shield as SH
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time")
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None, back_route=sc.back_route,
                           back_mode=sc.back_mode)
    shield = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw)
    shield.docks = [wp["cafe_counter_pick"][:2]]
    return sim, op, shield


def parts(out, n=12):
    sim, op, shield = _sim()
    rows = []
    for (W, H) in ((320, 180), (640, 360), (1280, 720)):
        for mode in ("hh", "3way", "sgbm"):
            for thr in (1, 4):
                cfg = StereoCfg(W=W, H=H, mode=mode, cv_threads=thr, self_mask=True)
                m = TW.build_twin_model(sim, cfg)
                s = StereoSensor(m, cfg)
                s.load(sim.d.qpos, sim.d.mocap_pos, sim.d.mocap_quat, sim.d.eq_active)
                s.warmup()
                for _ in range(n):
                    s.capture(0, 0.0)
                ms = {k: float(np.mean([h[k] for h in s.ms_hist])) for k in ("render", "stereo", "post", "total")}
                ms["total_wo_selfmask"] = ms["total"] - ms["post"]
                rows.append(dict(W=W, H=H, mode=mode, cv_threads=thr, n_disp=cfg.n_disp(), **ms))
                print(rows[-1], flush=True)
                s.close()
    Path(out).write_text(json.dumps(rows, indent=1))


def rtf(out, sim_s=8.0, rate_hz=10.0, res=(640, 360), mode="hh"):
    sim, op, shield = _sim()
    ticks = int(sim_s / TICK)
    res_rows = {}

    def loop(hook):
        t0 = time.perf_counter()
        for k in range(ticks):
            obs = sim.observe()
            cmd = op.step(obs)
            hook(k)
            safe, _ = shield.filter(sim, cmd)
            sim.tick(safe)
        return time.perf_counter() - t0

    cfg = StereoCfg(W=res[0], H=res[1], mode=mode, rate_hz=rate_hz, self_mask=True)
    # (a) none
    sim, op, shield = _sim()
    wa = loop(lambda k: None)
    res_rows["no_stereo"] = dict(wall_s=wa, rtf=sim_s / wa)
    # (b) in-process, blocking
    sim, op, shield = _sim()
    m = TW.build_twin_model(sim, cfg)
    s = StereoSensor(m, cfg)
    s.load(sim.d.qpos, sim.d.mocap_pos, sim.d.mocap_quat, sim.d.eq_active)
    s.warmup()
    every = int(round(1.0 / (rate_hz * TICK)))
    lat = []

    def hook_in(k):
        if k % every == 0:
            s.load(sim.d.qpos, sim.d.mocap_pos, sim.d.mocap_quat, sim.d.eq_active)
            fr = s.capture(sim.k, sim.t)
            lat.append(fr["ms"]["total"])
    wb = loop(hook_in)
    res_rows["in_process"] = dict(wall_s=wb, rtf=sim_s / wb, capture_ms=float(np.mean(lat)), frames=len(lat))
    s.close()
    # (c) separate process
    from .link import StereoLink
    sim, op, shield = _sim()
    path = TW.save_model(TW.build_twin_model(sim, cfg), TW.OUT / f"twin_bench_{os.getpid()}.mjb")
    t0 = time.perf_counter()
    link = StereoLink(sim.m, path, cfg, sim=sim)
    ready_s = time.perf_counter() - t0

    def hook_out(k):
        link.publish(sim)
    wc = loop(hook_out)
    time.sleep(0.5)
    fr = list(link.hist)
    res_rows["separate_process"] = dict(wall_s=wc, rtf=sim_s / wc, frames=len(fr), ready_s=ready_s, child_ready_ms=link.ready_ms,
                                        capture_ms=float(np.mean([f["ms"]["total"] for f in fr])) if fr else None,
                                        pub_to_recv_ms=float(np.mean([1e3 * (f["wall_recv"] - f["wall_pub"]) for f in fr])) if fr else None,
                                        achieved_hz_simtime=len(fr) / sim_s, max_hz_wallclock=1e3 / max(float(np.mean([f['ms']['total'] for f in fr])), 1e-9) if fr else None)
    link.close()
    res_rows["cfg"] = cfg.asdict()
    res_rows["sim_s"] = sim_s
    Path(out).write_text(json.dumps(res_rows, indent=1))
    print(json.dumps(res_rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["parts", "rtf"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--sim-s", type=float, default=8.0)
    ap.add_argument("--mode", default="3way")
    a = ap.parse_args()
    if a.what == "parts":
        parts(a.out or str(TW.OUT / "bench_parts.json"))
    else:
        rtf(a.out or str(TW.OUT / f"bench_rtf_{a.res[0]}x{a.res[1]}_{int(a.rate)}hz_{a.mode}.json"), a.sim_s, a.rate, tuple(a.res), a.mode)
    os._exit(0)


if __name__ == "__main__":
    main()
