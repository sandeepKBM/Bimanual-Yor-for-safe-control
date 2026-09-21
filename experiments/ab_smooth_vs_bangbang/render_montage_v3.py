"""v3: Render initial / near-collision / after frames for C1, C2, C3 (the
three ego safety laws; C0 -- no safety -- is left out of the montage per the
task spec, same as v1's C1-vs-C2 framing) and stitch them into a comparison
montage PNG under /common/users/ss5772/bimanual/artifacts/ab3_render_montage.png.

Separate script from render_montage.py (v1, C1-vs-C2 only, writes
ab_render_montage.png) so the v1 montage is never overwritten.

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang/render_montage_v3.py
"""
from __future__ import annotations

import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim_common as sc
from run_ab import left_right_geom_sets

ART_DIR = "/common/users/ss5772/bimanual/artifacts"
os.makedirs(ART_DIR, exist_ok=True)

# Steps to capture (seconds): initial, near the deepest-penetration moment of
# the v3 oscillating intrusion, and after it ramps out. (v3 episode is 6.0s.)
CAPTURE_T = [0.0, 1.1, 5.95]


def render(model, data, width=960, height=720):
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    # v3: re-centered on the canonical scene's shared workspace (table/box at
    # 0.42,-0.21,0.55-0.57; arms mounted at z~0.67), vs. v1/v2's floor-mounted
    # rig framing (0.30, 0.05, 0.40).
    cam.lookat = np.array([0.42, -0.21, 0.65])
    cam.distance = 0.9
    cam.azimuth = 160
    cam.elevation = -15
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    renderer.close()
    return pixels


def run_and_capture(condition: str):
    model = sc.load_model()
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)
    ctrl = sc.EgoController(model, q0_left=q0_left)

    right_act_ids = [model.actuator(f"right_arm_joint{i}_pos").id for i in range(1, 7)]
    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id
    tau_filter = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))

    frames = {}
    capture_steps = sorted(set(int(round(t / sc.DT)) for t in CAPTURE_T))

    for k in range(sc.N_STEPS):
        t = k * sc.DT
        data.ctrl[right_act_ids] = sc.intruder_ctrl(t)
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0
        mujoco.mj_forward(model, data)

        tau_task, M_ll, bias_l = ctrl.task_torque(data)
        bar = ctrl.barrier(data)
        h, hdot, n = bar["h"], bar["hdot"], bar["n"]
        Je = bar["Je"]
        psi1 = hdot + sc.ALPHA1 * h

        if condition == "C1":
            if h < sc.H_TRIGGER:
                g_joint = Je.T @ n
                nrm = np.linalg.norm(g_joint)
                tau_cmd = sc.TAU_MAX * g_joint / nrm if nrm > 1e-9 else np.zeros(6)
            else:
                tau_cmd = tau_task
        elif condition == "C2":
            M_ll_inv = np.linalg.inv(M_ll)
            pairs = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)
            a_rows, b_rows = [], []
            for p in pairs:
                n_p, Je_p, Ji_p = p["n"], p["Je"], p["Ji"]
                JeDot_p, JiDot_p = p["JeDot"], p["JiDot"]
                v_ego_p, v_int_p = p["v_ego"], p["v_int"]
                h_p, hdot_p, dist_p = p["h"], p["hdot"], p["dist"]
                qacc_int = data.qacc[ctrl.right_dofadr]
                a_int_p = Ji_p @ qacc_int + JiDot_p @ data.qvel[ctrl.right_dofadr]
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
            tau_qp, lam, feasible = sc.solve_constrained_box_qp(H_qp, f_qp, lower, upper, a_ineq, b_ineq)
            tau_cmd = tau_filter.apply(tau_qp, sc.DT)
        elif condition == "C3":
            pairs = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)
            qacc_f = ctrl.fabric_qacc(data, pairs)
            tau_f = M_ll @ qacc_f + bias_l
            tau_cmd = tau_filter.apply(tau_f, sc.DT)
        else:
            raise ValueError(condition)

        tau_cmd = np.clip(tau_cmd, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.left_dofadr] = tau_cmd

        if k in capture_steps:
            frames[k] = render(model, data)

        mujoco.mj_step(model, data)

    return frames, capture_steps


def main():
    fig, axes = plt.subplots(3, len(CAPTURE_T), figsize=(4.6 * len(CAPTURE_T), 12.6))
    for row, condition in enumerate(["C1", "C2", "C3"]):
        frames, capture_steps = run_and_capture(condition)
        for col, (t_target, k) in enumerate(zip(CAPTURE_T, capture_steps)):
            ax = axes[row, col]
            ax.imshow(frames[k])
            ax.axis("off")
            label = {"C1": "C1 bang-bang (hard)", "C2": "C2 smooth HOCBF-QP (hard)",
                     "C3": "C3 geometric fabric (soft)"}[condition]
            phase = {0: "initial", 1: "peak intrusion", 2: "after"}[col]
            ax.set_title(f"{label} -- {phase} (t={k*sc.DT:.2f}s)", fontsize=11)
    fig.suptitle("v3: C1 (bang-bang) vs C2 (smooth HOCBF-QP) vs C3 (geometric fabric) -- "
                 "aggressive oscillating intruder", fontsize=13)
    fig.tight_layout()
    out = os.path.join(ART_DIR, "ab3_render_montage.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
