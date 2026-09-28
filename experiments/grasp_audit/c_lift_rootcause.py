"""Root-cause the '1.3 cm cup slip during a gentle 10 cm lift' of scripts/smoke_realgrasp_lift.py.
Replicates the smoke EXACTLY (grasp_cup_real F=8, transport_cfg OSC, 500-step ramp + 400 hold) and
decomposes its slip metric. SIMULATION ONLY.

variants (argv[1]): orig | fixedosc (world-frame orientation error monkeypatched) | motor | fixedosc_motor
"""
from __future__ import annotations

import json
import sys

import numpy as np
import mujoco

from common import REPO, quat_mul, quat_conj, quat_angle, rel_pose, jaw_geoms, contact_stats  # noqa: F401

sys.path.insert(0, str(REPO))
from scripts.cup_collect_task_utils import load_yor_cup_collect_model  # noqa: E402
from scripts import cup_task_utils as ct  # noqa: E402
from control import grasp_routines as gr  # noqa: E402
from control import cartesian_impedance as ci  # noqa: E402
from control import cube_grasp as cg  # noqa: E402
from control.arm_controller import ArmController  # noqa: E402
from control.osc_core import quat_normalize_wxyz  # noqa: E402


def _world_err(quat_des, quat_cur):
    qe = quat_mul(quat_normalize_wxyz(quat_cur), quat_conj(quat_normalize_wxyz(quat_des)))
    qe /= np.linalg.norm(qe)
    if qe[0] < 0:
        qe = -qe
    return 2.0 * qe[1:4]


def main(var):
    if "fixedosc" in var:
        ci.orientation_error_vec_wxyz = _world_err
    model, data = load_yor_cup_collect_model(restore_finger_object_collision=True)
    if "motor" in var:
        # same close/settle as grasp_cup_real but finger actuators as force motors
        model.opt.noslip_iterations = 2
        gr.place_cup_in_open_gripper(model, data, side="left", z_frac=0.5, carry_balls=True)
        cg.set_finger_motor_mode(model, "left", 20.0)
        arm = ArmController(model, data, "left", cfg=gr.transport_cfg(), grip_mode="motor_direct")
        arm.hold_pose()
        for k in range(200):
            arm.set_grip_force(min(1.0, (k + 1) / 100) * 8.0); arm.step(); mujoco.mj_step(model, data)
        arm.set_grip_force(8.0)
        for _ in range(200):
            arm.step(); mujoco.mj_step(model, data)
    else:
        cfg = None
        if "stiff" in var:
            from control.cartesian_impedance import CartesianImpedanceConfig
            import sim_common as sc
            cfg = CartesianImpedanceConfig(n_joints=6, tau_max_nm=np.full(6, sc.TAU_MAX), kp_pos=np.full(3, 900.0),
                                           kd_pos=np.full(3, 60.0), kp_rot=900.0, kd_rot=60.0, kp_posture=1.0,
                                           kd_posture=2.0, kd_joint=0.5, use_osc_inertia=True)
        if "nofric" in var:
            import sim_common as sc
            _, _, dadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
            model.dof_frictionloss[dadr] = 0.0
        arm, res = gr.grasp_cup_real(model, data, grip_force=8.0, cfg=cfg)
    cup = model.body(ct.CUP_BODY_NAME).id
    ee = arm.idx.ee_body_id
    pad7, pad8 = arm.idx.pad_geom_ids
    jaw7, jaw8 = jaw_geoms(model, "left")

    def pad_mid():
        return 0.5 * (data.geom_xpos[pad7] + data.geom_xpos[pad8])
    off0 = (data.xpos[cup] - pad_mid()).copy()
    ee_q0, cup_q0 = data.xquat[ee].copy(), data.xquat[cup].copy()
    R0 = data.xmat[ee].reshape(3, 3).copy()
    rp0, rq0 = rel_pose(data, ee, cup)
    # pad-mid in EE frame (to check finger motion) and cup-vs-padmid in EE frame
    pm_local0 = R0.T @ (pad_mid() - data.xpos[ee])
    cupvpm_local0 = R0.T @ off0
    ee_start = data.xpos[ee].copy()
    n7, n8, f7, f8, util, dmin = contact_stats(model, data, cup, jaw7, jaw8)
    table_contacts0 = [(model.geom(data.contact[i].geom1).name, model.geom(data.contact[i].geom2).name)
                       for i in range(data.ncon)
                       if "table" in (model.geom(data.contact[i].geom1).name + model.geom(data.contact[i].geom2).name)
                       and ("left_arm" in model.body(model.geom_bodyid[data.contact[i].geom1]).name
                            or "left_arm" in model.body(model.geom_bodyid[data.contact[i].geom2]).name
                            or "cup" in model.geom(data.contact[i].geom1).name + model.geom(data.contact[i].geom2).name)]
    lift = np.array([0.0, 0.0, 0.10])
    hist = []
    for k in range(900):
        frac = min(1.0, (k + 1) / 500)
        arm.set_ee_target(ee_start + frac * lift, quat=ee_q0)
        arm.set_grip_force(8.0)
        arm.step()
        mujoco.mj_step(model, data)
        R = data.xmat[ee].reshape(3, 3)
        rp, rq = rel_pose(data, ee, cup)
        off = data.xpos[cup] - pad_mid()
        hist.append(dict(
            k=k,
            smoke_slip_mm=float(np.linalg.norm(off - off0) * 1e3),
            rot_only_pred_mm=float(np.linalg.norm((R @ R0.T - np.eye(3)) @ off0) * 1e3),
            ee_frame_slip_mm=float(np.linalg.norm(rp - rp0) * 1e3),
            cup_vs_padmid_eeframe_mm=float(np.linalg.norm(R.T @ off - cupvpm_local0) * 1e3),
            padmid_move_eeframe_mm=float(np.linalg.norm(R.T @ (pad_mid() - data.xpos[ee]) - pm_local0) * 1e3),
            rel_rot_deg=float(np.degrees(quat_angle(quat_mul(quat_conj(rq0), rq)))),
            ee_tilt_deg=float(np.degrees(quat_angle(quat_mul(quat_conj(ee_q0), data.xquat[ee])))),
            smoke_relrot_metric_deg=abs(float(np.degrees(quat_angle(quat_mul(quat_conj(cup_q0), data.xquat[cup]))))
                                        - float(np.degrees(quat_angle(quat_mul(quat_conj(ee_q0), data.xquat[ee]))))),
            ee_pos_err_mm=float(np.linalg.norm(data.xpos[ee] - (ee_start + frac * lift)) * 1e3),
        ))
    f = hist[-1]
    first = next((h["k"] for h in hist if h["ee_frame_slip_mm"] > 2.0), None)
    out = dict(variant=var, grasp_contacts=[n7, n8], grasp_fn=[f7, f8], grasp_pen_mm=-dmin * 1e3,
               table_contacts_at_grasp=table_contacts0[:6], final=f,
               max_ee_frame_slip_mm=max(h["ee_frame_slip_mm"] for h in hist),
               max_smoke_slip_mm=max(h["smoke_slip_mm"] for h in hist),
               max_ee_tilt_deg=max(h["ee_tilt_deg"] for h in hist),
               first_step_eeframe_slip_gt2mm=first,
               slip_at=[{k: round(v, 2) for k, v in hist[i].items()} for i in (0, 10, 50, 100, 250, 499, 899)])
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
