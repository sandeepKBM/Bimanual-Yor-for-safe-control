"""v4-HO experiment: add HIGHER-ORDER smoothness to v4-C's per-step
decentralized-cooperative HOCBF-QP (RESULTS_v4.md's winner: collision-free,
box-retained (drift 1.6cm), lowest chatter of A/B/C at 0.68 hdot
sign-changes/s) and re-test against the SAME A (adversarial) / C (v4-C
baseline) rows, byte-for-byte reused from run_ab4.py, plus a new C_HO row.

v4-C's smoothness today is only FIRST-order: a low-pass + rate-limit filter
(sc.TorqueCommandFilter) applied AFTER each arm's QP solves for a raw torque
tau_qp. This adds, inside BOTH arms' per-step QPs (ego HOCBF-QP and
intruder HOCBF-QP -- v4-C runs one each), two second-order smoothness
mechanisms on the QP's own decision variable tau_qp, minimally and cheaply
(still a per-step box+linear-inequality QP, NOT a horizon MPC):

  1. Slew-rate CONSTRAINT (hard bound): |tau_t - tau_{t-1}| <= RHO_SLEW per
     joint. Implemented for free -- no new inequality rows needed -- by
     tightening the box bounds passed to solve_constrained_box_qp each step:
     lower = max(-TAU_MAX, tau_prev - RHO_SLEW), upper = min(TAU_MAX,
     tau_prev + RHO_SLEW). The existing psi2>=0 safety row is UNCHANGED (same
     a_ineq/b_ineq construction as run_ab4.py) -- so the hard safety
     guarantee is unaffected by this addition; only the box the QP searches
     within shrinks.
  2. Snap PENALTY (soft, in the objective): W_SNAP * ||tau_t - 2*tau_{t-1} +
     tau_{t-2}||^2, i.e. a weighted least-squares term (I, 2*tau_prev -
     tau_prev2, W_SNAP) added alongside the existing task term via
     sc.build_weighted_least_squares_qp's already-multi-term interface (no
     new QP-building code needed). Requires keeping each arm's last TWO raw
     QP torques as running state (tau_prev, tau_prev2), reset to "no
     constraint/penalty yet" for the first 1-2 steps of the run (no history).

tau_t here is each arm's RAW QP output (the decision variable), tracked
BEFORE the existing low-pass/rate-limit command filter, which still runs
identically downstream for both C and C_HO -- isolates the comparison to
"does smoothing the QP's own solution help" vs. any change to the existing
first-order filter.

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang/run_ab4ho.py

Writes experiments/ab_smooth_vs_bangbang/logs/v4ho_{A,C,C_HO}.npz +
logs/v4ho_summary.json (+ printed table).
"""

from __future__ import annotations

import os
import json
import time
import numpy as np
import mujoco

import sim_common as sc
import run_ab4 as ab4

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CONDITIONS = ["A", "C", "C_HO"]

# ---- higher-order smoothness hyperparameters (new for this file) ----------
RHO_SLEW = 20.0     # [Nm] hard per-joint per-step slew bound on the RAW QP
                    # torque, applied to BOTH arms' QPs (half of TAU_MAX=40 --
                    # loose enough that the hard psi2>=0 safety row can still
                    # always be satisfied within the shrunk box, tight enough
                    # to forbid a full-scale one-step jump).
W_SNAP = 0.02       # weight on ||tau_t - 2 tau_{t-1} + tau_{t-2}||^2 relative
                    # to the task term's weight of 1.0 -- deliberately small:
                    # this only breaks ties within whatever the hard psi2>=0
                    # row + slew box still allow, it never competes with
                    # safety or the task term for dominance.


def run_condition_c_ho(capture_steps=None, capture_cb=None):
    """v4-C (cooperative, ego-priority) with higher-order smoothness added to
    BOTH arms' per-step QPs. Mirrors run_ab4.run_condition's C branch
    exactly (same alpha=30/30, same INTRUDER_EXTRA_MARGIN=0.06, same
    task/barrier construction, same downstream TorqueCommandFilter) except
    for the RHO_SLEW box-tightening + W_SNAP objective term below."""
    condition = "C_HO"
    model = sc.load_model()
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)

    ctrl = sc.EgoController(model, q0_left=q0_left)
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

    alpha1_int = alpha2_int = ab4.INTRUDER_ALPHA["C"]
    extra_margin_int = ab4.INTRUDER_EXTRA_MARGIN["C"]

    n_steps = sc.N_STEPS
    log = {k: np.zeros(n_steps) for k in
           ["t", "h", "dist", "hdot", "psi1", "psi2", "qp_feasible_ego",
            "qp_feasible_int", "contact_force", "collision", "box_z",
            "box_drift_trans", "box_drift_rot", "int_track_err", "int_qdes_norm",
            "solve_time_s"]}
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

    # higher-order state: last two RAW QP torques per arm (None = no history yet)
    tau_prev_ego = tau_prev2_ego = None
    tau_prev_int = tau_prev2_int = None

    for k in range(n_steps):
        t = k * sc.DT
        q_int_des = ab4.intruder_ctrl_nominal(t)
        qd_int_des = ab4.intruder_ctrl_nominal_dot(t)
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0

        mujoco.mj_forward(model, data)

        tau_task, M_ll, bias_l = ctrl.task_torque(data)
        bar = ctrl.barrier(data)
        h, hdot, n_vec = bar["h"], bar["hdot"], bar["n"]
        psi1 = hdot + sc.ALPHA1 * h

        M_ll_inv = np.linalg.inv(M_ll)
        pairs = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)

        qacc_int_known = data.qacc[ctrl.right_dofadr]
        qacc_ego_known = data.qacc[ctrl.left_dofadr]

        t_solve0 = time.perf_counter()

        # ---- ego HOCBF-QP (same a_ineq/b_ineq construction as run_ab4.py's
        # C -- the hard safety row is untouched by the HO addition) ----
        a_rows, b_rows = [], []
        for p in pairs:
            n_p, Je_p, Ji_p = p["n"], p["Je"], p["Ji"]
            JeDot_p, JiDot_p = p["JeDot"], p["JiDot"]
            v_ego_p, v_int_p = p["v_ego"], p["v_int"]
            h_p, hdot_p, dist_p = p["h"], p["hdot"], p["dist"]
            a_int_p = Ji_p @ qacc_int_known + JiDot_p @ data.qvel[ctrl.right_dofadr]
            ndot_p = (np.eye(3) - np.outer(n_p, n_p)) @ (v_ego_p - v_int_p) / max(dist_p, 1e-6)
            hddot_rest_p = float(ndot_p @ (v_ego_p - v_int_p)
                                  + n_p @ (JeDot_p @ data.qvel[ctrl.left_dofadr])
                                  - n_p @ a_int_p)
            psi1_p = hdot_p + sc.ALPHA1 * h_p
            A_h_p = n_p @ Je_p @ M_ll_inv
            const_term_p = (-float(n_p @ Je_p @ M_ll_inv @ bias_l) + hddot_rest_p
                            + sc.ALPHA1 * hdot_p + sc.ALPHA2 * psi1_p)
            a_rows.append(-A_h_p)
            b_rows.append(const_term_p)
        a_ineq = np.stack(a_rows, axis=0)
        b_ineq = np.array(b_rows)

        # ---- HO addition (ego): slew box + snap objective term ----
        terms_ego = [(np.eye(6), tau_task, 1.0)]
        if tau_prev_ego is not None and tau_prev2_ego is not None:
            terms_ego.append((np.eye(6), 2.0 * tau_prev_ego - tau_prev2_ego, W_SNAP))
        H_qp, f_qp = sc.build_weighted_least_squares_qp(terms_ego)
        if tau_prev_ego is not None:
            lower = np.maximum(-sc.TAU_MAX, tau_prev_ego - RHO_SLEW)
            upper = np.minimum(sc.TAU_MAX, tau_prev_ego + RHO_SLEW)
        else:
            lower = -sc.TAU_MAX * np.ones(6)
            upper = sc.TAU_MAX * np.ones(6)
        tau_ego_qp, _, feasible_ego = sc.solve_constrained_box_qp(
            H_qp, f_qp, lower, upper, a_ineq, b_ineq)
        tau_prev2_ego = tau_prev_ego
        tau_prev_ego = tau_ego_qp.copy()

        tau_ego_cmd = tau_filter_ego.apply(tau_ego_qp, sc.DT)
        tau_ego_cmd = np.clip(tau_ego_cmd, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.left_dofadr] = tau_ego_cmd

        # ---- intruder task torque (identical to run_ab4.py's C) ----
        M_full = sc.full_mass_matrix(model, data)
        M_rr = M_full[np.ix_(ctrl.right_dofadr, ctrl.right_dofadr)]
        bias_r = data.qfrc_bias[ctrl.right_dofadr]
        q_int = data.qpos[ctrl.right_qposadr]
        qvel_int = data.qvel[ctrl.right_dofadr]
        qacc_int_des = (ab4.INTRUDER_KP_TASK * (q_int_des - q_int)
                         + ab4.INTRUDER_KD_TASK * (qd_int_des - qvel_int))
        tau_task_int = M_rr @ qacc_int_des + bias_r
        M_rr_inv = np.linalg.inv(M_rr)

        # ---- intruder HOCBF-QP (same mirror-derived construction as
        # run_ab4.py's C) ----
        ai_rows, bi_rows = [], []
        for p in pairs:
            n_p, Je_p, Ji_p = p["n"], p["Je"], p["Ji"]
            JeDot_p, JiDot_p = p["JeDot"], p["JiDot"]
            v_ego_p, v_int_p = p["v_ego"], p["v_int"]
            h_p, hdot_p, dist_p = p["h"], p["hdot"], p["dist"]
            h_p_int = h_p - extra_margin_int
            a_ego_p = Je_p @ qacc_ego_known + JeDot_p @ data.qvel[ctrl.left_dofadr]
            ndot_p = (np.eye(3) - np.outer(n_p, n_p)) @ (v_ego_p - v_int_p) / max(dist_p, 1e-6)
            hddot_rest_int_p = float(ndot_p @ (v_ego_p - v_int_p)
                                      + n_p @ a_ego_p
                                      - n_p @ (JiDot_p @ qvel_int))
            psi1_p_int = hdot_p + alpha1_int * h_p_int
            A_h_int_p = n_p @ Ji_p @ M_rr_inv
            const_term_int_p = (float(n_p @ Ji_p @ M_rr_inv @ bias_r) + hddot_rest_int_p
                                 + alpha1_int * hdot_p + alpha2_int * psi1_p_int)
            ai_rows.append(A_h_int_p)
            bi_rows.append(const_term_int_p)
        ai_ineq = np.stack(ai_rows, axis=0)
        bi_ineq = np.array(bi_rows)

        # ---- HO addition (intruder): slew box + snap objective term ----
        terms_int = [(np.eye(6), tau_task_int, 1.0)]
        if tau_prev_int is not None and tau_prev2_int is not None:
            terms_int.append((np.eye(6), 2.0 * tau_prev_int - tau_prev2_int, W_SNAP))
        Hi_qp, fi_qp = sc.build_weighted_least_squares_qp(terms_int)
        if tau_prev_int is not None:
            lower_i = np.maximum(-sc.TAU_MAX, tau_prev_int - RHO_SLEW)
            upper_i = np.minimum(sc.TAU_MAX, tau_prev_int + RHO_SLEW)
        else:
            lower_i = -sc.TAU_MAX * np.ones(6)
            upper_i = sc.TAU_MAX * np.ones(6)
        tau_int_qp, _, feasible_int = sc.solve_constrained_box_qp(
            Hi_qp, fi_qp, lower_i, upper_i, ai_ineq, bi_ineq)
        tau_prev2_int = tau_prev_int
        tau_prev_int = tau_int_qp.copy()

        tau_int_cmd = tau_filter_int.apply(tau_int_qp, sc.DT)
        tau_int_cmd = np.clip(tau_int_cmd, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.right_dofadr] = tau_int_cmd

        solve_time_s = time.perf_counter() - t_solve0

        # ---- logging (identical fields to run_ab4.py) ----
        log["t"][k] = t
        log["h"][k] = h
        log["dist"][k] = bar["dist"]
        log["hdot"][k] = hdot
        log["psi1"][k] = psi1
        log["qp_feasible_ego"][k] = float(feasible_ego)
        log["qp_feasible_int"][k] = float(feasible_int)
        log["solve_time_s"][k] = solve_time_s
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

        R_gb = data.xmat[ee_body].reshape(3, 3)
        rel_pos = R_gb.T @ (data.xpos[box_body] - data.xpos[ee_body])
        gb_quat = data.xquat[ee_body].copy()
        box_quat = data.xquat[box_body].copy()
        rel_quat = ab4.quat_mul(ab4.quat_conj(gb_quat), box_quat)
        if box_pos_rel0 is None:
            box_pos_rel0 = rel_pos.copy()
            box_quat_rel0 = rel_quat.copy()
        log["box_drift_trans"][k] = float(np.linalg.norm(rel_pos - box_pos_rel0))
        drift_quat = ab4.quat_mul(ab4.quat_conj(box_quat_rel0), rel_quat)
        log["box_drift_rot"][k] = float(ab4.quat_angle(drift_quat))

        if capture_steps is not None and k in capture_steps and capture_cb is not None:
            capture_cb(condition, k, model, data)

        mujoco.mj_step(model, data)

    np.savez(os.path.join(LOG_DIR, "v4ho_C_HO.npz"), **log)
    return log


def run_and_time(condition):
    """Run a byte-identical-to-run_ab4 condition (A or C), timed the same
    coarse wall-clock-per-step way as run_condition_c_ho (no per-QP timing
    hooks exist in run_ab4.py, and it is NOT modified here -- see docstring
    caveats in the report)."""
    t0 = time.perf_counter()
    log = ab4.run_condition(condition)
    total_s = time.perf_counter() - t0
    log["solve_time_s"] = np.full(sc.N_STEPS, total_s / sc.N_STEPS)
    np.savez(os.path.join(LOG_DIR, f"v4ho_{condition}.npz"), **log)
    return log


def main():
    summary = {}
    logs = {}
    for condition in CONDITIONS:
        print(f"=== running v4ho-{condition} ===")
        if condition == "C_HO":
            log = run_condition_c_ho()
        else:
            log = run_and_time(condition)
        logs[condition] = log
        m = ab4.compute_metrics(condition, log)
        m["avg_step_time_ms"] = float(np.mean(log["solve_time_s"])) * 1000.0
        summary[condition] = m
        print(json.dumps(m, indent=2))

    home = summary["A"]["home_ee_gap"]
    adv_min = summary["A"]["min_ee_gap"]
    denom = max(home - adv_min, 1e-6)
    for c in CONDITIONS:
        prog = (home - summary[c]["min_ee_gap"]) / denom
        summary[c]["intruder_progress_frac_vs_A"] = float(prog)

    with open(os.path.join(LOG_DIR, "v4ho_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY TABLE (A | C | C_HO) ===")
    keys = list(summary["A"].keys())
    for k in keys:
        if k == "condition":
            continue
        row = "  ".join(f"{c}={summary[c][k]!s:>10}" for c in CONDITIONS)
        print(f"{k:28s}  {row}")


if __name__ == "__main__":
    main()
