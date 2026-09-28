#!/usr/bin/env python
"""check_init_cup_orientation.py -- cheap audit (NO episode run): for each seed,
build the flagship FL-MPC 'crossing' episode (faked grasp, hard geometry used
by experiments/wm_requirement/collect_ladder.py) and report the cup's own +z axis
expressed in the left-EE frame (from cup_rel_quat0, the fixed pin offset) and
its world tilt at the lift pose. A cup that fell over during init is carried
ON ITS SIDE for the whole episode (world tilt ~90 deg) -> trivially spills when
the balls are released. Simulation only.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, argparse
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def one(seed):
    import mujoco
    from scripts import make_cup_collect_safety_demo as demo
    info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                              cup_jitter_m=0.04, n_cubes=3)
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(info["cup_rel_quat0"], float))
    z_in_ee = R.reshape(3, 3)[:, 2]
    model = info["model"] if "model" in info else None
    tilt_lift = float("nan")
    if model is not None:
        d = mujoco.MjData(model)
        d.qpos[info["left_qposadr"]] = info["q_lift"]
        mujoco.mj_forward(model, d)
        Ree = d.xmat[info["left_ee_body"]].reshape(3, 3)
        tilt_lift = float(np.degrees(np.arccos(np.clip((Ree @ z_in_ee)[2], -1, 1))))
    return seed, z_in_ee, tilt_lift


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-24")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as p:
        for seed, z, t in sorted(p.map(one, range(lo, hi + 1))):
            print(f"seed {seed:4d} cup+z in EE frame = {np.round(z, 3)}  world tilt at lift pose = {t:6.1f} deg", flush=True)
