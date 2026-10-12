"""lift_study.py -- checks and the perception sweep for the lift-enabled house model.  SIMULATION ONLY.

    python -m world_model.v2.house.lift_study checks       # FL-MPC tau vs canonical, spawn contacts, 5-s carry (each lift height)
    python -m world_model.v2.house.lift_study track  --lift 0.2     # ball tracker, mounts x pitches, at one lift height
    python -m world_model.v2.house.lift_study zono   --house 0      # object segmentation / zonotopes with the robot head
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

from world_model.v2.house import build as B
from world_model.v2.house import lift as L
from world_model.v2.house import phase1 as P1

OUT = Path("/data/local/ss5772/molmospaces/pb/results")
HEIGHTS = (0.0, 0.2, 0.416)
SPOT = dict(house=17, spawn=(3.55, 10.05), yaw=0.3)


_KEEP = []          # sim_common.all_pair_barriers caches by model identity: keep every model alive so ids are never reused


def check_tau(h, n_states=6, seed=0):
    """FL-MPC torque, M and c on the lift-enabled composed model at lift h vs the canonical robot-only scene, same arm states."""
    import sim_common as sc
    import mpc_fl
    canon = sc.load_model()
    _KEEP.append(canon)
    dc = mujoco.MjData(canon)
    rng = np.random.default_rng(seed)
    _, clq, cld = sc.joint_addrs(canon, sc.LEFT_ARM_JOINTS)
    _, crq, crd = sc.joint_addrs(canon, sc.RIGHT_ARM_JOINTS)
    flc = mpc_fl.FLMPC(canon, **P1.FL_KW)
    NP = flc.NP
    states = [dict(ql=rng.normal(0, 0.4, 6) + np.array([0, 1.5, -0.6, 0, -0.9, 0.78]), qr=rng.normal(0, 0.4, 6) + np.array([0, 1.5, -0.6, 0, -0.9, -0.78]),
                   vl=rng.normal(0, 0.6, 6), vr=rng.normal(0, 0.6, 6), dql=rng.normal(0, 0.05, 6), dqr=rng.normal(0, 0.1, 6),
                   ap=(rng.normal(0, 1, 6), rng.normal(0, 1, 6))) for _ in range(n_states)]
    z = np.zeros((NP + 1, 6))

    def run(m, d, fl, lq, ld, rq, rd, s):
        d.qpos[lq] = s["ql"]; d.qpos[rq] = s["qr"]; d.qvel[ld] = s["vl"]; d.qvel[rd] = s["vr"]
        for sd in ("left", "right"):
            d.qpos[m.jnt_qposadr[m.joint(f"{sd}_arm_joint7").id]] = 0.0175
            d.qpos[m.jnt_qposadr[m.joint(f"{sd}_arm_joint8").id]] = -0.0175
        mujoco.mj_forward(m, d)
        fl.a_ego_prev, fl.a_int_prev = s["ap"][0].copy(), s["ap"][1].copy()
        qer = np.tile(d.qpos[lq] + s["dql"], (NP + 1, 1)); qir = np.tile(d.qpos[rq] + s["dqr"], (NP + 1, 1))
        pairs = sc.all_pair_barriers(m, d, ld, rd)
        te, ti, ok = fl.step(m, d, lq, ld, rq, rd, qer, z, qir, z, pairs)
        return te, ti, fl.last_M_e.copy(), fl.last_c_e.copy(), fl.last_M_i.copy(), fl.last_c_i.copy(), ok
    ref = [run(canon, dc, flc, clq, cld, crq, crd, s) for s in states]
    C = L.compose_lift(SPOT["house"], SPOT["spawn"], SPOT["yaw"], with_tray=False, dt=0.005)
    L.set_lift(C, h)
    _KEEP.append(C)
    m, d = C.m, C.d
    _, lq, ld = sc.joint_addrs(m, sc.LEFT_ARM_JOINTS)
    _, rq, rd = sc.joint_addrs(m, sc.RIGHT_ARM_JOINTS)
    fl = mpc_fl.FLMPC(m, **P1.FL_KW)
    dm = dict(lift=h, tau=0.0, M=0.0, c=0.0, flag_mismatch=0)
    for s, r in zip(states, ref):
        h_before = d.qpos[m.jnt_qposadr[m.joint("Slider_1").id]]
        o = run(m, d, fl, lq, ld, rq, rd, s)
        dm["tau"] = max(dm["tau"], float(np.abs(o[0] - r[0]).max()), float(np.abs(o[1] - r[1]).max()))
        dm["M"] = max(dm["M"], float(np.abs(o[2] - r[2]).max()), float(np.abs(o[4] - r[4]).max()))
        dm["c"] = max(dm["c"], float(np.abs(o[3] - r[3]).max()), float(np.abs(o[5] - r[5]).max()))
        dm["flag_mismatch"] += int(o[6]) != int(r[6])
    dm["max_abs_tau_ref"] = float(max(np.abs(r[0]).max() for r in ref))
    return dm


def check_spawn_carry(h, seconds=5.0):
    C = L.new_hold_lift(SPOT["house"], SPOT["spawn"], SPOT["yaw"], 0.005, lift_h=h)
    _KEEP.append(C)
    a1 = P1.acc1(C)
    car = P1.carry(C, seconds)
    keep = ["ball_max_xy_offset_m", "ball_min_z_in_tray_m", "tray_max_tilt_deg", "max_joint_drift_rad", "tray_max_drift_m", "ball_lost_at_s",
            "n_infeasible", "passed_ball", "passed_no_armarm", "passed_no_armhouse", "phys_ms_median", "ctl_ms_median", "contact_step_counts"]
    return dict(lift=h, lift_after=L.lift_state(C), spawn_contacts_with_house=a1["contacts_with_house"], ncon_total=a1["ncon_total"],
                min_robot_house_dist_m=a1["min_robot_house_dist_m"], nearest=a1["nearest"], carry={k: car[k] for k in keep})


def cmd_checks(a):
    res = dict(tau=[], spawn_carry=[])
    for h in HEIGHTS:
        t = check_tau(h)
        print("tau", t, flush=True)
        res["tau"].append(t)
    for h in HEIGHTS:
        c = check_spawn_carry(h)
        print("carry", json.dumps(c), flush=True)
        res["spawn_carry"].append(c)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lift_checks.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "track", "zono"])
    ap.add_argument("--lift", type=float, default=0.0)
    ap.add_argument("--house", type=int, default=0)
    ap.add_argument("--want", type=int, default=2)
    a = ap.parse_args()
    {"checks": cmd_checks}[a.cmd](a) if a.cmd == "checks" else None
    if a.cmd == "track":
        from world_model.v2.house import lift_track
        lift_track.cmd_track(a)
    if a.cmd == "zono":
        from world_model.v2.house import lift_track
        lift_track.cmd_zono(a)


if __name__ == "__main__":
    main()
