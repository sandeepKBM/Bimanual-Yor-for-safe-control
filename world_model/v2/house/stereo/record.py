"""stereo/record.py -- record STATE SNAPSHOTS (10 Hz) of a teleop run so the stereo study / replay renders them offline.  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.record carry --seeds 0 1 2          # h5 carry, shield ON (truth people), walking people (people_clock="time")

Per snapshot (the state at the START of a 50 Hz tick, what the camera would expose): qpos, mocap_pos, mocap_quat, eq_active (= everything the renderer
needs), sim time / tick, and TRUTH for evaluation only (base pose, lift, tray + cup world pose, people xy / v).  Output <OUT>/snap/carry_s<seed>.npz.
CPU only (no rendering).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")

import numpy as np

from .twin import OUT

TICK = 0.02
EVERY = 5            # 10 Hz


def record_carry(seed, out=OUT, max_s=30.0, tail_s=1.0, people_clock="time"):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP, shield as SH
    t_wall = time.time()
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock=people_clock)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    shield = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw)
    shield.docks = [wp["cafe_counter_pick"][:2]]
    rows = {}

    def add(**kw):
        for k, v in kw.items():
            rows.setdefault(k, []).append(v)

    done_t = None
    d = sim.d
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        if k % EVERY == 0:
            npl = len(sim.people_state)
            add(t=sim.t, k=sim.k, qpos=d.qpos.copy(), mpos=d.mocap_pos.copy(), mquat=d.mocap_quat.copy(), eq=d.eq_active.copy(),
                base=np.array([sim.base.x, sim.base.y, sim.base.yaw]), lift=sim.lift_actual(),
                tray_p=d.xpos[sim.tray].copy(), tray_R=d.xmat[sim.tray].copy(), cup_p=d.xpos[sim.cup].copy(), cup_R=d.xmat[sim.cup].copy(),
                ppl_xy=np.array([xy for xy, _ in sim.people_state], float).reshape(npl, 2), ppl_v=np.array([v for _, v in sim.people_state], float).reshape(npl, 2))
        safe, info = shield.filter(sim, cmd)
        o = sim.tick(safe)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
    A = {k: np.asarray(v) for k, v in rows.items()}
    A["people_names"] = np.array([p.name for p in sc.people if p.dynamic] + [p.name for p in sc.people if not p.dynamic])
    path = Path(out) / "snap" / f"carry_s{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **A)
    print(f"[stereo.record] carry seed {seed}: {len(A['t'])} snapshots, {time.time() - t_wall:.0f}s wall -> {path}", flush=True)
    return str(path)


def _job(a):
    return record_carry(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["carry"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.procs > 1 and len(a.seeds) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(a.procs, len(a.seeds)), maxtasksperchild=1) as pool:
            pool.map(_job, [(s, Path(a.out)) for s in a.seeds])
    else:
        for s in a.seeds:
            record_carry(s, Path(a.out))


if __name__ == "__main__":
    main()
