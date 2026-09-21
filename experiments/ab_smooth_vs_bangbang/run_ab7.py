"""v7: does killing chatter (collision checked every physics step) + a
grasp-fragility-derived (not hand-tuned) priority rule let the joint MPC
cleanly beat v4-C now? Same scenario/pipeline as v6 (run_ab6.py), swapping
mpc6.JointMPCFast for mpc7.JointMPCv7. A and v4-C are re-run byte-for-byte
via run_ab4.run_condition for reference rows.

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang/run_ab7.py

Writes logs/v7_{A,C,MPCV7}.npz + logs/v7_summary.json (+ printed table).
"""
from __future__ import annotations

import os
import json
import time
import numpy as np
import mujoco

import sim_common as sc
import run_ab4 as ab4
import mpc7 as mpc_mod
from controller_core.constrained_box_qp import numba_warmup as _box_qp_numba_warmup

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def run_mpcv7_condition(capture_steps=None, capture_cb=None):
    model = sc.load_model()
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)

    ctrl = sc.EgoController(model, q0_left=q0_left)
    joint_mpc = mpc_mod.JointMPCv7(model, ctrl, q0_left)

    mpc_mod.warmup()
    _box_qp_numba_warmup()

    left_geoms, right_geoms = ab4.left_right_geom_sets(model)
    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id

    ee_body = model.body("left_arm_gripper_base").id
    right_ee_body = model.body("right_arm_gripper_base").id
    box_body = model.body("test_box").id
    box_geom = model.geom("test_box_geom").id
    pad7 = model.geom("left_arm_link7_pad").id
    pad8 = model.geom("left_arm_link8_pad").id

    tau_filter_ego = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA,
                                             np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    tau_filter_int = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA,
                                             np.full(6, sc.FILTER_RATE_LIMIT_NM_S))

    n_steps = sc.N_STEPS
    log = {k: np.zeros(n_steps) for k in
           ["t", "h", "dist", "hdot", "psi1", "psi2", "qp_feasible_ego",
            "qp_feasible_int", "contact_force", "collision", "box_z",
            "box_drift_trans", "box_drift_rot", "int_track_err", "int_qdes_norm",
            "solve_time", "assemble_time", "qp_time", "barrier_time", "m_rows",
            "fragility"]}
    log["tau_ego"] = np.zeros((n_steps, 6))
    log["tau_int"] = np.zeros((n_steps, 6))
    log["q_ego"] = np.zeros((n_steps, 6))
    log["q_int"] = np.zeros((n_steps, 6))
    log["q_int_des"] = np.zeros((n_steps, 6))
    log["qvel_ego"] = np.zeros((n_steps, 6))
    log["ee_pos"] = np.zeros((n_steps, 3))
    log["box_pos"] = np.zeros((n_steps, 3))
    log["right_ee_pos"] = np.zeros((n_steps, 3))

    box_pos_rel0 = None
    box_quat_rel0 = None

    for k in range(n_steps):
        t = k * sc.DT
        q_int_des = ab4.intruder_ctrl_nominal(t)
        qd_int_des = ab4.intruder_ctrl_nominal_dot(t)
        q_int_des_traj = np.stack([ab4.intruder_ctrl_nominal(t + tt * sc.DT)
                                    for tt in range(mpc_mod.NP + 1)])
        qd_int_des_traj = np.stack([ab4.intruder_ctrl_nominal_dot(t + tt * sc.DT)
                                     for tt in range(mpc_mod.NP + 1)])

        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0
        mujoco.mj_forward(model, data)

        # ---- measured grasp state (BEFORE this step's QP, so this step's
        # priority reacts to the grasp state resulting from the LAST physics
        # step -- same causal ordering as `all_pair_barriers` below) ----
        R_gb = data.xmat[ee_body].reshape(3, 3)
        rel_pos = R_gb.T @ (data.xpos[box_body] - data.xpos[ee_body])
        gb_quat = data.xquat[ee_body].copy()
        box_quat = data.xquat[box_body].copy()
        rel_quat = ab4.quat_mul(ab4.quat_conj(gb_quat), box_quat)
        if box_pos_rel0 is None:
            box_pos_rel0 = rel_pos.copy()
            box_quat_rel0 = rel_quat.copy()
        box_drift_trans = float(np.linalg.norm(rel_pos - box_pos_rel0))
        drift_quat = ab4.quat_mul(ab4.quat_conj(box_quat_rel0), rel_quat)
        box_drift_rot = float(ab4.quat_angle(drift_quat))
        fragility = mpc_mod.grasp_fragility(model, data, box_geom, pad7, pad8,
                                             box_drift_trans, box_drift_rot, mujoco)

        tb0 = time.time()
        bar_all = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)
        barrier_time = time.time() - tb0
        worst = min(bar_all, key=lambda p: p["h"])
        h, dist, hdot = worst["h"], worst["dist"], worst["hdot"]
        psi1 = hdot + sc.ALPHA1 * h

        tau_ego_qp, tau_int_qp, feasible, solve_time = joint_mpc.step(
            model, data, q_int_des_traj, qd_int_des_traj, bar_all, fragility)

        tau_ego_cmd = tau_filter_ego.apply(tau_ego_qp, sc.DT)
        tau_ego_cmd = np.clip(tau_ego_cmd, -sc.TAU_MAX, sc.TAU_MAX)
        tau_int_cmd = tau_filter_int.apply(tau_int_qp, sc.DT)
        tau_int_cmd = np.clip(tau_int_cmd, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.left_dofadr] = tau_ego_cmd
        data.qfrc_applied[ctrl.right_dofadr] = tau_int_cmd

        log["t"][k] = t
        log["h"][k] = h
        log["dist"][k] = dist
        log["hdot"][k] = hdot
        log["psi1"][k] = psi1
        log["qp_feasible_ego"][k] = float(feasible)
        log["qp_feasible_int"][k] = float(feasible)
        log["solve_time"][k] = solve_time
        log["assemble_time"][k] = joint_mpc.last_assemble_time
        log["qp_time"][k] = joint_mpc.last_qp_time
        log["barrier_time"][k] = barrier_time
        log["m_rows"][k] = joint_mpc.last_m_rows
        log["fragility"][k] = fragility
        log["tau_ego"][k] = tau_ego_cmd
        log["tau_int"][k] = tau_int_cmd
        log["q_ego"][k] = data.qpos[ctrl.left_qposadr]
        log["q_int"][k] = data.qpos[ctrl.right_qposadr]
        log["q_int_des"][k] = q_int_des
        log["int_track_err"][k] = float(np.linalg.norm(data.qpos[ctrl.right_qposadr] - q_int_des))
        log["int_qdes_norm"][k] = float(np.linalg.norm(q_int_des - sc.RIGHT_HOME_CTRL))
        log["qvel_ego"][k] = data.qvel[ctrl.left_dofadr]
        log["ee_pos"][k] = data.xpos[ee_body]
        log["box_pos"][k] = data.xpos[box_body]
        log["right_ee_pos"][k] = data.xpos[right_ee_body]
        log["box_z"][k] = data.xpos[box_body][2]
        log["box_drift_trans"][k] = box_drift_trans
        log["box_drift_rot"][k] = box_drift_rot

        total_force = 0.0
        collision = False
        for i in range(data.ncon):
            con = data.contact[i]
            geoms = {con.geom1, con.geom2}
            if box_geom in geoms and (pad7 in geoms or pad8 in geoms):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total_force += float(np.linalg.norm(force[:3]))
            if (con.geom1 in left_geoms and con.geom2 in right_geoms) or \
               (con.geom2 in left_geoms and con.geom1 in right_geoms):
                collision = True
        log["contact_force"][k] = total_force
        log["collision"][k] = float(collision)

        if capture_steps is not None and k in capture_steps and capture_cb is not None:
            capture_cb("MPCV7", k, model, data)

        mujoco.mj_step(model, data)

        if (k % 100) == 0:
            print(f"  MPCV7 step {k}/{n_steps}  t={t:.2f}  h={h:.4f}  frag={fragility:.2f}  "
                  f"solve={solve_time*1000:.1f}ms (asm={joint_mpc.last_assemble_time*1000:.1f}"
                  f"+qp={joint_mpc.last_qp_time*1000:.1f})  m={joint_mpc.last_m_rows}  feas={feasible}")

    np.savez(os.path.join(LOG_DIR, "v7_MPCV7.npz"), **log)
    return log


def compute_metrics_mpcv7(log: dict) -> dict:
    m = ab4.compute_metrics("MPCV7", log)
    m["solve_time_mean_ms"] = float(np.mean(log["solve_time"]) * 1000.0)
    m["solve_time_max_ms"] = float(np.max(log["solve_time"]) * 1000.0)
    m["assemble_time_mean_ms"] = float(np.mean(log["assemble_time"]) * 1000.0)
    m["qp_time_mean_ms"] = float(np.mean(log["qp_time"]) * 1000.0)
    m["barrier_time_mean_ms"] = float(np.mean(log["barrier_time"]) * 1000.0)
    m["grand_total_mean_ms"] = float(np.mean(log["solve_time"] + log["barrier_time"]) * 1000.0)
    m["grand_total_max_ms"] = float(np.max(log["solve_time"] + log["barrier_time"]) * 1000.0)
    m["m_rows_mean"] = float(np.mean(log["m_rows"]))
    m["m_rows_max"] = float(np.max(log["m_rows"]))
    m["qp_infeasible_frac"] = float(1.0 - np.mean(log["qp_feasible_ego"]))
    m["fragility_mean"] = float(np.mean(log["fragility"]))
    m["fragility_max"] = float(np.max(log["fragility"]))
    return m


def main():
    summary = {}

    print("=== re-running v4-A (adversarial baseline, reference) ===")
    log_a = ab4.run_condition("A")
    m_a = ab4.compute_metrics("A", log_a)
    summary["A"] = m_a
    np.savez(os.path.join(LOG_DIR, "v7_A.npz"), **log_a)

    print("=== re-running v4-C (decentralized, hand-tuned ego-priority, reference) ===")
    log_c = ab4.run_condition("C")
    m_c = ab4.compute_metrics("C", log_c)
    summary["v4-C"] = m_c
    np.savez(os.path.join(LOG_DIR, "v7_C.npz"), **log_c)

    print("=== running v7-MPCV7 (every-step collision check + fragility-scaled priority) ===")
    t0 = time.time()
    log_mpc = run_mpcv7_condition()
    wall_s = time.time() - t0
    m_mpc = compute_metrics_mpcv7(log_mpc)
    m_mpc["wall_time_s"] = wall_s
    summary["MPC-v7"] = m_mpc

    home = summary["A"]["home_ee_gap"]
    adv_min = summary["A"]["min_ee_gap"]
    denom = max(home - adv_min, 1e-6)
    for c in ["A", "v4-C", "MPC-v7"]:
        prog = (home - summary[c]["min_ee_gap"]) / denom
        summary[c]["intruder_progress_frac_vs_A"] = float(prog)

    with open(os.path.join(LOG_DIR, "v7_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY TABLE (A | v4-C | MPC-v7) ===")
    keys = list(summary["MPC-v7"].keys())
    for k in keys:
        if k == "condition":
            continue
        row = "  ".join(f"{c}={summary[c].get(k, 'n/a')!s:>10}" for c in ["A", "v4-C", "MPC-v7"])
        print(f"{k:28s}  {row}")


if __name__ == "__main__":
    main()
