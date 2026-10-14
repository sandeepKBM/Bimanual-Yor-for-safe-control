"""A/B/C/D (v3) experiment: smooth HOCBF-QP safety filter (C2) vs bang-bang
HJ-style fail-safe (C1) vs no safety (C0) vs a soft geometric-fabric filter
(C3), single-arm pick + intrusion (E1) on scenes/scene_yor_canonical.xml (the
proper YOR robot, arms 0.185m apart -- see RESULTS_v3.md for the full
re-point changelog vs v2's scene_bimanual_pickup.xml).

v2 additions vs v1 (see RESULTS_v2.md): C3 (geometric fabric), a more
aggressive/oscillating intruder (sim_common.py), and grasped-box payload
coupling into every condition's effective mass matrix / bias.

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang/run_ab.py

Writes per-condition raw per-timestep logs to
experiments/ab_smooth_vs_bangbang/logs/{C0,C1,C2,C3}.npz and a summary metric
table to experiments/ab_smooth_vs_bangbang/logs/summary.json (+ printed table).
"""

from __future__ import annotations

import os
import json
import numpy as np
import mujoco

import sim_common as sc


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_angle(q):
    w = np.clip(abs(q[0]), -1.0, 1.0)
    return 2.0 * np.arccos(w)


def left_right_geom_sets(model):
    left_geoms, right_geoms = set(), set()
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = model.body(bid).name
        if bname.startswith("left_arm_"):
            left_geoms.add(gi)
        elif bname.startswith("right_arm_"):
            right_geoms.add(gi)
    return left_geoms, right_geoms


def run_condition(condition: str):
    model = sc.load_model()
    # Grasp init happens FIRST, with the left arm's own position servos still
    # active (so the arm holds the 'home' pose while the fingers close on the
    # teleported box, exactly as in scripts/test_pickup_scene.py). Only AFTER
    # the grasp is established do we bypass the 6 left ARM joint position
    # servos and hand control to the ego torque controller -- q0 (the task
    # target) is captured at that hand-off instant, i.e. the arm's settled
    # 'home' pose while already holding the box.
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)

    ctrl = sc.EgoController(model, q0_left=q0_left)

    left_geoms, right_geoms = left_right_geom_sets(model)

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id
    right_act_ids = [model.actuator(f"right_arm_joint{i}_pos").id for i in range(1, 7)]

    ee_body = model.body("left_arm_gripper_base").id
    box_body = model.body("test_box").id
    box_geom = model.geom("test_box_geom").id
    pad7 = model.geom("left_arm_link7_pad").id
    pad8 = model.geom("left_arm_link8_pad").id

    tau_filter = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA,
                                         np.full(6, sc.FILTER_RATE_LIMIT_NM_S))

    n_steps = sc.N_STEPS
    log = {k: np.zeros(n_steps) for k in
           ["t", "h", "dist", "hdot", "psi1", "psi2", "bangbang_active",
            "qp_feasible", "fabric_repulsion_active", "contact_force", "collision", "box_z",
            "box_drift_trans", "box_drift_rot"]}
    log["tau"] = np.zeros((n_steps, 6))
    log["q"] = np.zeros((n_steps, 6))
    log["qvel"] = np.zeros((n_steps, 6))
    log["ee_pos"] = np.zeros((n_steps, 3))
    log["box_pos"] = np.zeros((n_steps, 3))
    log["right_ee_pos"] = np.zeros((n_steps, 3))

    box_pos_rel0 = None
    box_quat_rel0 = None
    tau_prev = np.zeros(6)

    for k in range(n_steps):
        t = k * sc.DT
        data.ctrl[right_act_ids] = sc.intruder_ctrl(t)
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0

        mujoco.mj_forward(model, data)

        tau_task, M_ll, bias_l = ctrl.task_torque(data)
        bar = ctrl.barrier(data)
        h, hdot, n = bar["h"], bar["hdot"], bar["n"]
        Je, Ji = bar["Je"], bar["Ji"]
        v_ego, v_int = bar["v_ego"], bar["v_int"]
        eid, iid = bar["eid"], bar["iid"]
        psi1 = hdot + sc.ALPHA1 * h

        bangbang_active = False
        qp_feasible = True
        fabric_repulsion_active = False
        psi2_val = np.nan

        if condition == "C0":
            tau_cmd = tau_task

        elif condition == "C1":
            if h < sc.H_TRIGGER:
                bangbang_active = True
                g_joint = Je.T @ n  # generalized force ascending h fastest
                nrm = np.linalg.norm(g_joint)
                if nrm > 1e-9:
                    tau_cmd = sc.TAU_MAX * g_joint / nrm
                else:
                    tau_cmd = np.zeros(6)
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
            # diagnostic: psi2 at tau_task for the (logged) globally-closest pair
            psi2_val = float(-a_ineq[int(np.argmin([p["h"] for p in pairs]))] @ tau_task
                              + b_ineq[int(np.argmin([p["h"] for p in pairs]))])

            H_qp, f_qp = sc.build_weighted_least_squares_qp(
                [(np.eye(6), tau_task, 1.0)])
            lower = -sc.TAU_MAX * np.ones(6)
            upper = sc.TAU_MAX * np.ones(6)
            tau_qp, lam, feasible = sc.solve_constrained_box_qp(
                H_qp, f_qp, lower, upper, a_ineq, b_ineq)
            qp_feasible = bool(feasible)
            tau_cmd = tau_filter.apply(tau_qp, sc.DT)

        elif condition == "C3":
            pairs = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)
            fabric_repulsion_active = any(p["h"] < sc.FABRIC_REPULSION_MARGIN for p in pairs)
            qacc_f = ctrl.fabric_qacc(data, pairs)
            tau_f = M_ll @ qacc_f + bias_l
            tau_cmd = tau_filter.apply(tau_f, sc.DT)
        else:
            raise ValueError(condition)

        tau_cmd = np.clip(tau_cmd, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.left_dofadr] = tau_cmd

        # ---- logging (pre-step state / this-step command) ----
        log["t"][k] = t
        log["h"][k] = h
        log["dist"][k] = bar["dist"]
        log["hdot"][k] = hdot
        log["psi1"][k] = psi1
        log["psi2"][k] = psi2_val
        log["bangbang_active"][k] = float(bangbang_active)
        log["qp_feasible"][k] = float(qp_feasible)
        log["fabric_repulsion_active"][k] = float(fabric_repulsion_active)
        log["tau"][k] = tau_cmd
        log["q"][k] = data.qpos[ctrl.left_qposadr]
        log["qvel"][k] = data.qvel[ctrl.left_dofadr]
        log["ee_pos"][k] = data.xpos[ee_body]
        log["box_pos"][k] = data.xpos[box_body]
        log["right_ee_pos"][k] = data.xpos[model.body("right_arm_gripper_base").id]
        log["box_z"][k] = data.xpos[box_body][2]

        # finger-box contact force
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

        # box pose relative to gripper (translation + rotation drift)
        R_gb = data.xmat[ee_body].reshape(3, 3)
        rel_pos = R_gb.T @ (data.xpos[box_body] - data.xpos[ee_body])
        gb_quat = data.xquat[ee_body].copy()
        box_quat = data.xquat[box_body].copy()
        rel_quat = quat_mul(quat_conj(gb_quat), box_quat)
        if box_pos_rel0 is None:
            box_pos_rel0 = rel_pos.copy()
            box_quat_rel0 = rel_quat.copy()
        log["box_drift_trans"][k] = float(np.linalg.norm(rel_pos - box_pos_rel0))
        drift_quat = quat_mul(quat_conj(box_quat_rel0), rel_quat)
        log["box_drift_rot"][k] = float(quat_angle(drift_quat))

        mujoco.mj_step(model, data)
        tau_prev = tau_cmd

    np.savez(os.path.join(LOG_DIR, f"{condition}.npz"), **log)
    return log


def compute_metrics(condition: str, log: dict) -> dict:
    dt = sc.DT
    ee_pos = log["ee_pos"]
    q = log["q"]

    rms_ee_jerk, max_ee_jerk = sc.jerk_metrics(ee_pos, dt)
    rms_joint_jerk, max_joint_jerk = sc.jerk_metrics(q, dt)

    v_ee = np.gradient(ee_pos, dt, axis=0)
    speed = np.linalg.norm(v_ee, axis=1)
    sparc_val = sc.sparc(speed, dt)

    table_top_z = 0.55  # scenes/YOR_CANONICAL.md: table_top surface z (v2 was 0.20, old floor-mounted rig)
    box_dropped = bool(np.min(log["box_z"]) < table_top_z - 0.01) or \
        bool(np.max(log["box_drift_trans"]) > 0.05)
    min_contact_force = float(np.min(log["contact_force"]))

    min_h = float(np.min(log["h"]))
    min_dist = float(np.min(log["dist"]))
    collision = bool(np.any(log["collision"] > 0.5))

    tau = log["tau"]
    tau_rate = np.diff(tau, axis=0) / dt
    rms_torque = float(np.sqrt(np.mean(tau ** 2)))
    rms_torque_rate = float(np.sqrt(np.mean(tau_rate ** 2))) if len(tau_rate) else 0.0

    frac_bangbang_active = float(np.mean(log["bangbang_active"])) if "bangbang_active" in log else 0.0
    qp_infeasible_frac = float(np.mean(1.0 - log["qp_feasible"])) if "qp_feasible" in log else 0.0
    frac_fabric_repulsion_active = float(np.mean(log["fabric_repulsion_active"])) \
        if "fabric_repulsion_active" in log else 0.0

    safety_type = {
        "C0": "none",
        "C1": "hard switch (bang-bang HJ-style, discontinuous)",
        "C2": "hard constraint (smooth HOCBF-QP, psi2>=0 enforced)",
        "C3": "SOFT (geometric fabric -- encouraged, never enforced)",
    }[condition]

    return dict(
        condition=condition,
        safety_type=safety_type,
        rms_ee_jerk=rms_ee_jerk, max_ee_jerk=max_ee_jerk,
        rms_joint_jerk=rms_joint_jerk, max_joint_jerk=max_joint_jerk,
        sparc=sparc_val,
        box_drift_trans_max=float(np.max(log["box_drift_trans"])),
        box_drift_rot_max_deg=float(np.max(log["box_drift_rot"]) * 180.0 / np.pi),
        min_contact_force=min_contact_force,
        box_dropped=box_dropped,
        min_h=min_h, min_dist=min_dist, collision=collision,
        rms_torque=rms_torque, rms_torque_rate=rms_torque_rate,
        frac_bangbang_active=frac_bangbang_active,
        qp_infeasible_frac=qp_infeasible_frac,
        frac_fabric_repulsion_active=frac_fabric_repulsion_active,
    )


CONDITIONS = ["C0", "C1", "C2", "C3"]


def main():
    summary = {}
    for condition in CONDITIONS:
        print(f"=== running {condition} ===")
        log = run_condition(condition)
        m = compute_metrics(condition, log)
        summary[condition] = m
        print(json.dumps(m, indent=2))

    with open(os.path.join(LOG_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY TABLE ===")
    keys = list(summary["C0"].keys())
    for k in keys:
        if k == "condition":
            continue
        row = "  ".join(f"{c}={summary[c][k]!s:>10}" for c in CONDITIONS)
        print(f"{k:28s}  {row}")


if __name__ == "__main__":
    main()
