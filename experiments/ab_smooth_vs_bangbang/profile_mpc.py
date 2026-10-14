"""ab6: profile the v5 JointMPC.step() to find where the 16.8ms/step went.
Instruments the SAME code path as run_ab5.run_mpc_condition, with manual
perf_counter section timers inserted around: (a) MuJoCo calls (full_mass_matrix,
payload_terms, body_jac_full -- mj_fullM/mj_jac under the hood), (b) the Python
QP-assembly (sensitivity/_propagate recursion + collision/grasp row building),
(c) the constrained_box_qp solve itself. Does NOT modify mpc.py; monkeypatches
timers around the same calls mpc.JointMPC.step makes, replicated inline here
so the original module is left untouched for a fair "before" baseline.
"""
from __future__ import annotations
import time
import numpy as np
import mujoco

import sim_common as sc
import run_ab4 as ab4
import mpc as mpc_mod

N_PROFILE = 300


def main():
    model = sc.load_model()
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)
    ctrl = sc.EgoController(model, q0_left=q0_left)
    joint_mpc = mpc_mod.JointMPC(model, ctrl, q0_left)

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id

    tau_filter_ego = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    tau_filter_int = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))

    t_mujoco = t_assembly = t_qp = t_barriers = 0.0
    t_total = 0.0
    n = 0

    for k in range(N_PROFILE):
        t = k * sc.DT
        q_int_des_traj = np.stack([ab4.intruder_ctrl_nominal(t + tt * sc.DT) for tt in range(mpc_mod.NP + 1)])
        qd_int_des_traj = np.stack([ab4.intruder_ctrl_nominal_dot(t + tt * sc.DT) for tt in range(mpc_mod.NP + 1)])

        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0
        mujoco.mj_forward(model, data)

        t0 = time.perf_counter()
        bar_all = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)
        t1 = time.perf_counter()
        t_barriers += (t1 - t0)

        # ---- replicate JointMPC.step, timed section-by-section ----
        t_step0 = time.perf_counter()
        q_ego0 = data.qpos[ctrl.left_qposadr].copy()
        qd_ego0 = data.qvel[ctrl.left_dofadr].copy()
        q_int0 = data.qpos[ctrl.right_qposadr].copy()
        qd_int0 = data.qvel[ctrl.right_dofadr].copy()

        tA = time.perf_counter()
        M_full = sc.full_mass_matrix(model, data)
        M_ll = M_full[np.ix_(ctrl.left_dofadr, ctrl.left_dofadr)]
        M_rr = M_full[np.ix_(ctrl.right_dofadr, ctrl.right_dofadr)]
        bias_l = data.qfrc_bias[ctrl.left_dofadr].copy()
        bias_r = data.qfrc_bias[ctrl.right_dofadr].copy()
        M_box, bias_box = ctrl.payload_terms(data)
        M_ll = M_ll + M_box
        bias_l = bias_l + bias_box
        Minv_ll = np.linalg.inv(M_ll)
        Minv_rr = np.linalg.inv(M_rr)
        Jp_full, Jr_full = sc.body_jac_full(model, data, ctrl.ee_body_id)
        tB = time.perf_counter()
        t_mujoco += (tB - tA)

        from mpc import _sensitivity
        qf_e, qdf_e, Phq_e, Phqd_e = _sensitivity(q_ego0, qd_ego0, Minv_ll, bias_l)
        qf_i, qdf_i, Phq_i, Phqd_i = _sensitivity(q_int0, qd_int0, Minv_rr, bias_r)

        terms = []
        for tt in range(1, mpc_mod.NP + 1):
            terms.append((joint_mpc._pad_e(Phq_e[tt]), joint_mpc.q0_left - qf_e[tt], mpc_mod.W_POS_EGO))
            terms.append((joint_mpc._pad_e(Phqd_e[tt]), -qdf_e[tt], mpc_mod.W_VEL_EGO))
            terms.append((joint_mpc._pad_i(Phq_i[tt]), q_int_des_traj[tt] - qf_i[tt], mpc_mod.W_POS_INT))
            terms.append((joint_mpc._pad_i(Phqd_i[tt]), qd_int_des_traj[tt] - qdf_i[tt], mpc_mod.W_VEL_INT))
        terms.append((joint_mpc._block_sel_ego(0), joint_mpc.tau_ego_prev, mpc_mod.W_JERK))
        terms.append((joint_mpc._block_sel_int(0), joint_mpc.tau_int_prev, mpc_mod.W_JERK))
        for b in range(1, mpc_mod.NC):
            terms.append((joint_mpc._block_sel_ego(b) - joint_mpc._block_sel_ego(b - 1), np.zeros(6), mpc_mod.W_JERK))
            terms.append((joint_mpc._block_sel_int(b) - joint_mpc._block_sel_int(b - 1), np.zeros(6), mpc_mod.W_JERK))
        H, f = sc.build_weighted_least_squares_qp(terms, reg=mpc_mod.REG_EFFORT, n=joint_mpc.ntot)

        pairs_sorted = sorted(bar_all, key=lambda p: p["h"])[:mpc_mod.K_PAIRS]
        a_rows, b_rows = [], []
        for p in pairs_sorted:
            n_p, Je_p, Ji_p, h0 = p["n"], p["Je"], p["Ji"], p["h"]
            c_e = n_p @ Je_p
            c_i = n_p @ Ji_p
            for tt in range(1, mpc_mod.NP + 1):
                row_e = -c_e @ Phq_e[tt]
                row_i = c_i @ Phq_i[tt]
                a_row = np.concatenate([row_e, row_i])
                const = h0 + c_e @ (qf_e[tt] - q_ego0) - c_i @ (qf_i[tt] - q_int0)
                a_rows.append(a_row)
                b_rows.append(const)

        Jp_ee = Jp_full[:, ctrl.left_dofadr]
        Jr_ee = Jr_full[:, ctrl.left_dofadr]
        for tt in (mpc_mod.NP // 2, mpc_mod.NP):
            for J_ee, slack in ((Jp_ee, mpc_mod.EE_SLACK_POS), (Jr_ee, mpc_mod.EE_SLACK_ROT)):
                d_free = J_ee @ (qf_e[tt] - joint_mpc.q0_left)
                Phi_ee = J_ee @ Phq_e[tt]
                row_full = joint_mpc._pad_e(Phi_ee)
                for i in range(3):
                    a_rows.append(row_full[i])
                    b_rows.append(slack - d_free[i])
                    a_rows.append(-row_full[i])
                    b_rows.append(slack + d_free[i])

        a_ineq = np.stack(a_rows, axis=0)
        b_ineq = np.array(b_rows)
        tC = time.perf_counter()
        t_assembly += (tC - tB)

        lower = -sc.TAU_MAX * np.ones(joint_mpc.ntot)
        upper = sc.TAU_MAX * np.ones(joint_mpc.ntot)
        x, _, feasible = sc.solve_constrained_box_qp(
            H, f, lower, upper, a_ineq, b_ineq,
            dual_sweeps=mpc_mod.DUAL_SWEEPS, dual_root_iters=mpc_mod.DUAL_ROOT_ITERS)
        tD = time.perf_counter()
        t_qp += (tD - tC)
        t_total += (tD - t_step0)
        n += 1

        tau_ego = x[0:6]
        tau_int = x[joint_mpc.n_e:joint_mpc.n_e + 6]
        joint_mpc.tau_ego_prev = tau_ego.copy()
        joint_mpc.tau_int_prev = tau_int.copy()

        tau_ego_cmd = np.clip(tau_filter_ego.apply(tau_ego, sc.DT), -sc.TAU_MAX, sc.TAU_MAX)
        tau_int_cmd = np.clip(tau_filter_int.apply(tau_int, sc.DT), -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.left_dofadr] = tau_ego_cmd
        data.qfrc_applied[ctrl.right_dofadr] = tau_int_cmd
        mujoco.mj_step(model, data)

    print(f"n={n} steps profiled")
    print(f"all_pair_barriers (shared, outside joint_mpc.step): {1000*t_barriers/n:.3f} ms/step")
    print(f"MuJoCo calls (fullM/payload/body_jac) inside step:  {1000*t_mujoco/n:.3f} ms/step")
    print(f"Python QP-assembly (sensitivity+rows) inside step:  {1000*t_assembly/n:.3f} ms/step")
    print(f"constrained_box_qp solve:                           {1000*t_qp/n:.3f} ms/step")
    print(f"TOTAL joint_mpc.step-equivalent:                    {1000*t_total/n:.3f} ms/step")
    print(f"  (+ barriers, full per-physics-step total ~= {1000*(t_total+t_barriers)/n:.3f} ms/step)")


if __name__ == "__main__":
    main()
