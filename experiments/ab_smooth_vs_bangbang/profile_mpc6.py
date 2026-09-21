"""ab6: profile mpc6.JointMPCFast at (a) v5's original horizon size (NC=4,
L=2,NP=8) for an apples-to-apples numba-speedup number, and (b) mpc6's own
extended-horizon constants (module-level NC/NP), reporting assemble/QP/total
ms per step for each. Numba warmup (JIT compile) is run once before timing.
"""
from __future__ import annotations
import time
import numpy as np
import mujoco

import sim_common as sc
import run_ab4 as ab4
import mpc6
from controller_core.constrained_box_qp import numba_warmup as _box_qp_numba_warmup

N_PROFILE = 300


def run_config(NC, L_PER_BLOCK, K_PAIRS, label):
    # monkeypatch mpc6 module constants for this config (NP recomputed)
    mpc6.NC = NC
    mpc6.L_PER_BLOCK = L_PER_BLOCK
    mpc6.NP = NC * L_PER_BLOCK
    mpc6.K_PAIRS = K_PAIRS

    model = sc.load_model()
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)
    ctrl = sc.EgoController(model, q0_left=q0_left)
    joint_mpc = mpc6.JointMPCFast(model, ctrl, q0_left)

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id

    mpc6.warmup()  # JIT compile mpc6's own kernels before timing
    _box_qp_numba_warmup()  # JIT compile the reused constrained_box_qp kernel too

    t_assemble = t_qp = t_total = t_barriers = 0.0
    solve_times = []
    n = 0
    for k in range(N_PROFILE):
        t = k * sc.DT
        NPc = mpc6.NP
        q_int_des_traj = np.stack([ab4.intruder_ctrl_nominal(t + tt * sc.DT) for tt in range(NPc + 1)])
        qd_int_des_traj = np.stack([ab4.intruder_ctrl_nominal_dot(t + tt * sc.DT) for tt in range(NPc + 1)])

        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0
        mujoco.mj_forward(model, data)

        tb0 = time.perf_counter()
        bar_all = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)
        tb1 = time.perf_counter()
        t_barriers += (tb1 - tb0)

        tau_ego, tau_int, feasible, solve_time = joint_mpc.step(
            model, data, q_int_des_traj, qd_int_des_traj, bar_all)

        t_assemble += joint_mpc.last_assemble_time
        t_qp += joint_mpc.last_qp_time
        t_total += solve_time
        solve_times.append(solve_time)
        n += 1

        tau_ego_cmd = np.clip(tau_ego, -sc.TAU_MAX, sc.TAU_MAX)
        tau_int_cmd = np.clip(tau_int, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.left_dofadr] = tau_ego_cmd
        data.qfrc_applied[ctrl.right_dofadr] = tau_int_cmd
        mujoco.mj_step(model, data)

    solve_times = np.array(solve_times) * 1000.0
    print(f"=== {label}: NC={NC} L={L_PER_BLOCK} NP={NC*L_PER_BLOCK} K_PAIRS={K_PAIRS} m_rows={joint_mpc.last_m_rows} ===")
    print(f"  assemble: {1000*t_assemble/n:.3f} ms/step")
    print(f"  qp solve: {1000*t_qp/n:.3f} ms/step")
    print(f"  total (assemble+qp, joint_mpc.step): mean={1000*t_total/n:.3f}  max={solve_times.max():.3f} ms/step")
    print(f"  all_pair_barriers (shared, outside step): {1000*t_barriers/n:.3f} ms/step")
    print(f"  grand total (barriers+step): mean={1000*(t_total+t_barriers)/n:.3f} ms/step")


def main():
    run_config(NC=4, L_PER_BLOCK=2, K_PAIRS=5, label="A: matched v5 size (NP=8, n=48)")
    run_config(NC=4, L_PER_BLOCK=6, K_PAIRS=5, label="D: NC=4 (n=48), L=6 -> NP=24, 120ms")
    run_config(NC=4, L_PER_BLOCK=8, K_PAIRS=5, label="E: NC=4 (n=48), L=8 -> NP=32, 160ms")
    run_config(NC=4, L_PER_BLOCK=10, K_PAIRS=5, label="F: NC=4 (n=48), L=10 -> NP=40, 200ms")
    run_config(NC=6, L_PER_BLOCK=6, K_PAIRS=5, label="G: NC=6 (n=72), L=6 -> NP=36, 180ms")


if __name__ == "__main__":
    main()
