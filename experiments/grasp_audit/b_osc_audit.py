"""Part B: OSC Cartesian controller (control/arm_controller.py + cartesian_impedance.py)
characterisation on the LEFT YOR arm at the cup-grasp pose. SIMULATION ONLY.

Variants:
  orig        : code as-is (orientation error = 2*vec(conj(q_des) q_cur), a BODY-frame vector,
                used as a WORLD-frame moment -- suspected frame bug)
  fixed       : same law with the orientation error expressed in the WORLD frame
                (2*vec(q_cur conj(q_des))), monkeypatched in this process only
Configs: default (kp_rot 80/kd_rot 18), transport (80/120), and fixed+stiffer gains.
Tests: hold drift, 5 cm position steps, 10 deg orientation steps, circle tracking,
z-sine bandwidth, torque use. No cup (pure controller characterisation).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import mujoco

from common import REPO, sc, quat_mul, quat_conj, quat_angle, axis_angle_quat  # noqa: F401

sys.path.insert(0, str(REPO))
from scripts.cup_collect_task_utils import load_yor_cup_collect_model  # noqa: E402
from control import grasp_routines as gr  # noqa: E402
from control import cartesian_impedance as ci  # noqa: E402
from control.cartesian_impedance import CartesianImpedanceConfig  # noqa: E402
from control.arm_controller import ArmController  # noqa: E402
from control.osc_core import quat_normalize_wxyz  # noqa: E402

_ORIG_ERR = ci.orientation_error_vec_wxyz


def _world_err(quat_des, quat_cur):
    qd = quat_normalize_wxyz(quat_des)
    qc = quat_normalize_wxyz(quat_cur)
    qe = quat_mul(qc, quat_conj(qd))   # world-frame des->cur
    qe = qe / np.linalg.norm(qe)
    if qe[0] < 0:
        qe = -qe
    return 2.0 * qe[1:4]


def cfg(kp_rot, kd_rot, kp_pos=225.0, kd_pos=30.0):
    return CartesianImpedanceConfig(n_joints=6, tau_max_nm=np.full(6, sc.TAU_MAX),
                                    kp_pos=np.full(3, kp_pos), kd_pos=np.full(3, kd_pos),
                                    kp_rot=kp_rot, kd_rot=kd_rot, kp_posture=1.0, kd_posture=2.0,
                                    kd_joint=0.5, use_osc_inertia=True)


def setup(c, fixed, zero_fricloss=False):
    ci.orientation_error_vec_wxyz = _world_err if fixed else _ORIG_ERR
    model, data = load_yor_cup_collect_model(restore_finger_object_collision=True)
    gr.place_cup_in_open_gripper(model, data, carry_balls=True)
    # move the cup far away so this is a pure-controller test
    cj = model.joint("cup_free")
    data.qpos[cj.qposadr[0]:cj.qposadr[0] + 3] += np.array([0.0, 0.0, -5.0])
    cup = model.body("cup").id
    for g in range(model.ngeom):   # take the cup (+ balls) out of collision entirely
        if model.geom_bodyid[g] == cup or model.body(model.geom_bodyid[g]).name.startswith("ball_"):
            model.geom_contype[g] = 0; model.geom_conaffinity[g] = 0
    if zero_fricloss:
        _, _, dadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
        model.dof_frictionloss[dadr] = 0.0
    sc.disable_left_arm_position_servos(model)
    mujoco.mj_forward(model, data)
    # LEFT_GRASP_Q has the left finger (link8) 6.8 mm INTO the table top -> first raise the EE
    # 8 cm with an accurate task-space inverse-dynamics driver so every OSC test starts in free space.
    from common import TaskSpaceID
    ts = TaskSpaceID(model, data, "left")
    ee = ts.ee
    p0, q0 = data.xpos[ee].copy(), data.xquat[ee].copy()
    for k in range(300):
        s = min(1.0, k / 200.0); s = s * s * (3 - 2 * s)
        ts.step(p0 + np.array([0, 0, 0.08 * s]), q0)
        mujoco.mj_step(model, data)
    _, _, dadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    data.qvel[dadr] = 0.0
    mujoco.mj_forward(model, data)
    arm = ArmController(model, data, "left", cfg=c)
    return model, data, arm


def run(model, data, arm, ref_fn, T):
    ee = arm.idx.ee_body_id
    n = int(T / model.opt.timestep)
    P = np.zeros((n, 3)); PR = np.zeros((n, 3)); ANG = np.zeros(n); TAU = np.zeros(n)
    for k in range(n):
        t = k * model.opt.timestep
        p_ref, q_ref, v_ref = ref_fn(t)
        arm.set_ee_target(p_ref, quat=q_ref, vel=v_ref)
        arm.set_grip_opening(1.0)
        arm.step()
        TAU[k] = np.max(np.abs(data.qfrc_applied[arm.idx.dofadr]))
        mujoco.mj_step(model, data)
        P[k] = data.xpos[ee]; PR[k] = p_ref
        ANG[k] = quat_angle(quat_mul(quat_conj(q_ref), data.xquat[ee]))
        if not np.all(np.isfinite(data.qpos)):
            break
    return P, PR, ANG, TAU


def step_metrics(y, y0, y1, dt):
    """Scalar step response along the step direction."""
    s = (y - y0) / (y1 - y0)
    try:
        t10 = np.argmax(s >= 0.1) * dt
        t90 = np.argmax(s >= 0.9) * dt if np.any(s >= 0.9) else np.nan
    except Exception:
        t10 = t90 = np.nan
    over = max(0.0, float(np.max(s) - 1.0))
    band = np.abs(s - 1.0) > 0.02
    ts = (np.nonzero(band)[0][-1] + 1) * dt if np.any(band) else 0.0
    return dict(rise_s=float(t90 - t10), overshoot_pct=100 * over, settle2pct_s=float(ts),
                ss_err_mm=float(abs(1.0 - s[-1]) * abs(y1 - y0) * 1e3))


def audit(name, c, fixed, zero_fricloss=False):
    out = dict(variant=name)
    dt = 0.005
    # 1) hold drift
    model, data, arm = setup(c, fixed, zero_fricloss)
    ee = arm.idx.ee_body_id
    p0, q0 = data.xpos[ee].copy(), data.xquat[ee].copy()
    P, PR, ANG, TAU = run(model, data, arm, lambda t: (p0, q0, None), 4.0)
    out["hold_4s_tilt_deg_max"] = float(np.degrees(ANG.max()))
    out["hold_4s_pos_err_mm_max"] = float(np.max(np.linalg.norm(P - p0, axis=1)) * 1e3)
    # 2) position steps
    for ax in range(3):
        model, data, arm = setup(c, fixed, zero_fricloss)
        p0, q0 = data.xpos[ee].copy(), data.xquat[ee].copy()
        p1 = p0.copy(); p1[ax] += 0.05
        P, PR, ANG, TAU = run(model, data, arm, lambda t: (p1, q0, None), 3.0)
        m = step_metrics(P[:, ax], p0[ax], p1[ax], dt)
        m["tilt_during_deg_max"] = float(np.degrees(ANG.max()))
        m["tau_peak_nm"] = float(TAU.max())
        out[f"pstep_{'xyz'[ax]}"] = m
    # 3) orientation steps (10 deg about world axes)
    for ax in range(3):
        model, data, arm = setup(c, fixed, zero_fricloss)
        p0, q0 = data.xpos[ee].copy(), data.xquat[ee].copy()
        e = np.zeros(3); e[ax] = 1.0
        q1 = quat_mul(axis_angle_quat(e, np.radians(10.0)), q0)
        P, PR, ANG, TAU = run(model, data, arm, lambda t: (p0, q1, None), 3.0)
        err_deg = np.degrees(ANG)
        s = 1.0 - err_deg / 10.0
        band = np.abs(err_deg) > 0.2
        out[f"rstep_{'xyz'[ax]}"] = dict(
            t90_s=float(np.argmax(s >= 0.9) * dt) if np.any(s >= 0.9) else None,
            final_err_deg=float(err_deg[-1]), min_err_deg=float(err_deg.min()),
            settle_0p2deg_s=float((np.nonzero(band)[0][-1] + 1) * dt) if np.any(band) else 0.0,
            pos_disturb_mm=float(np.max(np.linalg.norm(P - p0, axis=1)) * 1e3), tau_peak_nm=float(TAU.max()))
    # 4) circle tracking in the horizontal plane, r=5 cm, 0.5 Hz (v=0.157 m/s), orientation held
    model, data, arm = setup(c, fixed, zero_fricloss)
    p0, q0 = data.xpos[ee].copy(), data.xquat[ee].copy()
    r, f = 0.05, 0.5
    w = 2 * np.pi * f

    def circ(t):
        ramp = min(1.0, t / 1.0)
        p = p0 + ramp * r * np.array([np.cos(w * t) - 1.0, np.sin(w * t), 0.0])
        v = ramp * r * w * np.array([-np.sin(w * t), np.cos(w * t), 0.0])
        return p, q0, v
    P, PR, ANG, TAU = run(model, data, arm, circ, 5.0)
    e = np.linalg.norm(P - PR, axis=1)[200:]
    out["circle_r5cm_0p5Hz"] = dict(rms_mm=float(np.sqrt(np.mean(e ** 2)) * 1e3), max_mm=float(e.max() * 1e3),
                                   tilt_deg_max=float(np.degrees(ANG.max())), tau_peak_nm=float(TAU.max()))
    # 5) z bandwidth: 1 cm sine at several freqs, gain/phase from projection
    bw = {}
    for fz in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        model, data, arm = setup(c, fixed, zero_fricloss)
        p0, q0 = data.xpos[ee].copy(), data.xquat[ee].copy()
        wz = 2 * np.pi * fz
        T = max(3.0, 6.0 / fz)
        P, PR, ANG, TAU = run(model, data, arm,
                              lambda t: (p0 + np.array([0, 0, 0.01 * np.sin(wz * t)]), q0,
                                         np.array([0, 0, 0.01 * wz * np.cos(wz * t)])), T)
        n = len(P); tt = np.arange(n) * dt
        sel = tt > T / 2
        y = P[sel, 2] - p0[2]
        cs = np.exp(-1j * wz * tt[sel])
        G = (2 * np.mean(y * cs)) / (2 * np.mean(0.01 * np.sin(wz * tt[sel]) * cs))
        bw[str(fz)] = dict(gain_db=float(20 * np.log10(abs(G))), phase_deg=float(np.degrees(np.angle(G))))
    out["z_freq_resp"] = bw
    return out


if __name__ == "__main__":
    variants = [
        ("orig_default_80_18", cfg(80, 18), False),
        ("orig_transport_80_120", cfg(80, 120), False),
        ("fixed_transport_80_120", cfg(80, 120), True),
        ("fixed_crit_225_30", cfg(225, 30), True),
        ("fixed_crit_900_60", cfg(900, 60), True),
        ("fixed_crit_2500_100", cfg(2500, 100), True),
        ("fixed_crit_900_60_pos900", cfg(900, 60, 900, 60), True),
        ("fixed_crit_225_30_nofricloss", cfg(225, 30), True, True),
    ]
    which = sys.argv[1] if len(sys.argv) > 1 else None
    res = []
    for v in variants:
        if which and v[0] != which:
            continue
        r = audit(v[0], *v[1:])
        print(json.dumps(r), flush=True)
        res.append(r)
