#!/usr/bin/env python
"""diag_init_tilt.py -- READ-ONLY root-cause probe for the faked-grasp init tip-over.

SIMULATION ONLY (MuJoCo, offline). Does not edit any existing file.

Re-runs eval/scene_setup.init_cup_grasped_state's EXACT sequence step by step
(copied, not imported, so we can instrument it) on the flagship cup_collect
model and logs, per seed:
  * the ball->cup relative offset that the jitter introduces,
  * worst ball/cup-wall penetration right after the teleport,
  * cup world tilt at the end of the gravity-OFF close phase and after the
    gravity-ON settle,
  * whether any finger pad ever touches the cup,
for two variants:
  orig  : scene_setup's code (jitter perturbs ONLY old_cup_pos -> ball delta),
  fixed : jitter applied consistently (balls translated by the cup's TRUE delta,
          i.e. balls stay in the declared cup-relative layout).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "experiments" / "ab_smooth_vs_bangbang"))


def tilt_deg(q):
    import mujoco
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q)
    return float(np.degrees(np.arccos(np.clip(R[8], -1, 1))))


def one(args):
    seed, variant, jit = args
    import mujoco
    import sim_common as sc
    from scripts import cup_collect_task_utils as cct
    from scripts import cup_task_utils as ct
    model, _ = cct.load_yor_cup_collect_model(keep_zone_marker=False)
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    _, lq, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    data.qpos[lq] = sc.LEFT_GRASP_Q
    for i in range(1, 7):
        data.ctrl[model.actuator(f"left_arm_joint{i}_pos").id] = sc.LEFT_GRASP_Q[i - 1]
    mujoco.mj_forward(model, data)
    p7, p8 = model.geom("left_arm_link7_pad").id, model.geom("left_arm_link8_pad").id
    pad_mid = 0.5 * (data.geom_xpos[p7] + data.geom_xpos[p8])
    cup = model.body("cup").id
    cq = int(model.jnt_qposadr[model.body_jntadr[cup]])
    true_old = data.qpos[cq:cq + 3].copy()
    j = np.array([rng.uniform(-jit, jit), rng.uniform(-jit, jit), 0.0]) if jit > 0 else np.zeros(3)
    old = true_old + j
    new = pad_mid - np.array([0, 0, 0.55 * ct.CUP_HEIGHT])
    delta = (new - old) if variant == "orig" else (new - true_old)
    data.qpos[cq:cq + 3] = new; data.qpos[cq + 3:cq + 7] = [1, 0, 0, 0]
    balls = ct.ball_body_ids(model)
    for b in balls:
        a = int(model.jnt_qposadr[model.body_jntadr[b]])
        data.qpos[a:a + 3] += delta
    mujoco.mj_forward(model, data)
    # ball-offset relative to cup axis and worst penetration of balls into cup geoms
    cup_geoms = set(np.where(model.geom_bodyid == cup)[0].tolist())
    ball_geoms = set(int(model.body_geomadr[b]) for b in balls)
    pen = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in cup_geoms and c.geom2 in ball_geoms) or (c.geom2 in cup_geoms and c.geom1 in ball_geoms):
            pen = max(pen, -float(c.dist))
    nsp0 = ct.count_spilled(model, data)
    g = model.opt.gravity.copy(); model.opt.gravity[:] = 0
    data.ctrl[model.actuator("left_arm_joint7_pos").id] = 0.0
    data.ctrl[model.actuator("left_arm_joint8_pos").id] = 0.0
    pad_touch = False
    pads = {p7, p8}
    tilt_t = []
    for k in range(800):
        if k == 500:
            model.opt.gravity[:] = g
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            if (c.geom1 in pads and c.geom2 in cup_geoms) or (c.geom2 in pads and c.geom1 in cup_geoms):
                pad_touch = True
        if k in (10, 50, 499, 799):
            tilt_t.append(tilt_deg(data.qpos[cq + 3:cq + 7]))
    return dict(seed=seed, variant=variant, off_mm=1000 * float(np.linalg.norm((delta - (new - true_old))[:2])),
                pen_mm=1000 * pen, spilled_at_teleport=nsp0, tilt10=tilt_t[0], tilt50=tilt_t[1],
                tilt_grav_off_end=tilt_t[2], tilt_final=tilt_t[3], pad_touch=pad_touch,
                spilled_final=ct.count_spilled(model, data))


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-24"); ap.add_argument("--jitter", type=float, default=0.04)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    jobs = [(s, v, a.jitter) for v in ("orig", "fixed") for s in range(lo, hi + 1)]
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4) as p:
        res = p.map(one, jobs)
    for v in ("orig", "fixed"):
        rs = [r for r in res if r["variant"] == v]
        print(f"== variant {v}")
        for r in rs:
            print(f"  seed {r['seed']:3d} ball-offset {r['off_mm']:5.1f}mm pen {r['pen_mm']:5.1f}mm "
                  f"spill@tp {r['spilled_at_teleport']} tilt@10 {r['tilt10']:6.1f} @50 {r['tilt50']:6.1f} "
                  f"@g-off-end {r['tilt_grav_off_end']:6.1f} final {r['tilt_final']:6.1f} pad_touch {r['pad_touch']} "
                  f"spilled_final {r['spilled_final']}", flush=True)
        tip = [r["tilt_final"] > 45 for r in rs]
        print(f"  -> tipped(>45deg) {sum(tip)}/{len(rs)}; corr(offset, tip) = "
              f"{np.corrcoef([r['off_mm'] for r in rs], tip)[0,1] if 0 < sum(tip) < len(rs) else float('nan'):.2f}")
