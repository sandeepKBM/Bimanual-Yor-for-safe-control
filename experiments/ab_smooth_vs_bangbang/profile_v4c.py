"""ab6: measure v4-C's OWN decentralized QP-assembly+solve cost per step
(excluding the shared all_pair_barriers call), for an honest side-by-side
"solve time" comparison against mpc6's joint MPC (RESULTS_v6.md Sec.5).
Replicates run_ab4.run_condition("C")'s per-step ego+intruder QP construction
verbatim, just wrapped in timers -- does not modify run_ab4.py.
"""
from __future__ import annotations
import time
import numpy as np
import mujoco

import sim_common as sc
import run_ab4 as ab4
from controller_core.constrained_box_qp import numba_warmup as _box_qp_numba_warmup

N_PROFILE = 300
INTRUDER_EXTRA_MARGIN = ab4.INTRUDER_EXTRA_MARGIN if hasattr(ab4, "INTRUDER_EXTRA_MARGIN") else 0.06


def main():
    model = sc.load_model()
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)
    ctrl = sc.EgoController(model, q0_left=q0_left)

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id

    alpha1_int = sc.ALPHA1
    alpha2_int = sc.ALPHA2
    extra_margin_int = 0.06  # v4-C's tuned value

    _box_qp_numba_warmup()

    t_barriers = t_qp_ego = t_qp_int = 0.0
    n = 0
    max_ego = max_int = 0.0
    for k in range(N_PROFILE):
        t = k * sc.DT
        q_int_des = ab4.intruder_ctrl_nominal(t)
        qd_int_des = ab4.intruder_ctrl_nominal_dot(t)
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0
        mujoco.mj_forward(model, data)

        tau_task, M_ll, bias_l = ctrl.task_torque(data)
        M_ll_inv = np.linalg.inv(M_ll)

        tb0 = time.perf_counter()
        pairs = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)
        tb1 = time.perf_counter()
        t_barriers += (tb1 - tb0)

        qacc_int_known = data.qacc[ctrl.right_dofadr]
        qacc_ego_known = data.qacc[ctrl.left_dofadr]

        tqe0 = time.perf_counter()
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
        H_qp, f_qp = sc.build_weighted_least_squares_qp([(np.eye(6), tau_task, 1.0)])
        lower = -sc.TAU_MAX * np.ones(6)
        upper = sc.TAU_MAX * np.ones(6)
        tau_ego_qp, _, feasible_ego = sc.solve_constrained_box_qp(
            H_qp, f_qp, lower, upper, a_ineq, b_ineq)
        tqe1 = time.perf_counter()
        t_qp_ego += (tqe1 - tqe0)
        max_ego = max(max_ego, tqe1 - tqe0)

        M_full = sc.full_mass_matrix(model, data)
        M_rr = M_full[np.ix_(ctrl.right_dofadr, ctrl.right_dofadr)]
        bias_r = data.qfrc_bias[ctrl.right_dofadr]
        q_int = data.qpos[ctrl.right_qposadr]
        qvel_int = data.qvel[ctrl.right_dofadr]
        qacc_int_des = (ab4.INTRUDER_KP_TASK * (q_int_des - q_int)
                         + ab4.INTRUDER_KD_TASK * (qd_int_des - qvel_int))
        tau_task_int = M_rr @ qacc_int_des + bias_r
        M_rr_inv = np.linalg.inv(M_rr)

        tqi0 = time.perf_counter()
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
        Hi_qp, fi_qp = sc.build_weighted_least_squares_qp([(np.eye(6), tau_task_int, 1.0)])
        tau_int_qp, _, feasible_int = sc.solve_constrained_box_qp(
            Hi_qp, fi_qp, lower, upper, ai_ineq, bi_ineq)
        tqi1 = time.perf_counter()
        t_qp_int += (tqi1 - tqi0)
        max_int = max(max_int, tqi1 - tqi0)
        n += 1

        tau_ego_cmd = np.clip(tau_ego_qp, -sc.TAU_MAX, sc.TAU_MAX)
        tau_int_cmd = np.clip(tau_int_qp, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.left_dofadr] = tau_ego_cmd
        data.qfrc_applied[ctrl.right_dofadr] = tau_int_cmd
        mujoco.mj_step(model, data)

    print(f"n={n}  m_rows(all pairs)={a_ineq.shape[0]}")
    print(f"all_pair_barriers (shared): {1000*t_barriers/n:.3f} ms/step")
    print(f"ego QP assemble+solve:      {1000*t_qp_ego/n:.3f} ms/step")
    print(f"int QP assemble+solve:      {1000*t_qp_int/n:.3f} ms/step")
    print(f"v4-C controller-only total: {1000*(t_qp_ego+t_qp_int)/n:.3f} ms/step  (max ego={1000*max_ego:.3f} int={1000*max_int:.3f})")
    print(f"v4-C grand total (+barriers): {1000*(t_qp_ego+t_qp_int+t_barriers)/n:.3f} ms/step")


if __name__ == "__main__":
    main()
