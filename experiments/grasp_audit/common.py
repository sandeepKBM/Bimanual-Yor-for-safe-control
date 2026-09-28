"""Shared helpers for the grasp/EE audit (SIMULATION ONLY; no hardware)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "scripts", REPO / "experiments" / "ab_smooth_vs_bangbang"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sim_common as sc  # noqa: E402


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_angle(q):
    q = q / np.linalg.norm(q)
    return 2.0 * np.arccos(min(1.0, abs(q[0])))


def world_rot_err(q_ref, q_cur):
    """World-frame rotation vector taking CURRENT to REFERENCE (restoring direction)."""
    qe = quat_mul(q_ref, quat_conj(q_cur))
    if qe[0] < 0:
        qe = -qe
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, qe, 1.0)
    return v


def axis_angle_quat(axis, ang):
    q = np.zeros(4)
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    mujoco.mju_axisAngle2Quat(q, ax, float(ang))
    return q


def body_vel(model, data, bid):
    v6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, v6, 0)
    return v6[3:].copy(), v6[:3].copy()   # lin, ang (world)


def rel_pose(data, ee_bid, obj_bid):
    R = data.xmat[ee_bid].reshape(3, 3)
    p = R.T @ (data.xpos[obj_bid] - data.xpos[ee_bid])
    q = quat_mul(quat_conj(data.xquat[ee_bid]), data.xquat[obj_bid])
    return p, q


class TaskSpaceID:
    """Cartesian inverse-dynamics (feedback-linearized) excitation driver for one YOR arm.
    qdd = J^+ (a_ref + Kp e + Kd edot - Jdot qd);  tau = M qdd + (bias - passive) [+ payload bias].
    Used ONLY as a stiff, accurate EE-motion generator for the grasp stress rig (the measured
    EE acceleration, not the commanded one, is what gets reported)."""

    def __init__(self, model, data, side, kp=900.0, kd=60.0, kp_r=900.0, kd_r=60.0,
                 held_bid=None, lam=1e-3):
        self.m, self.d = model, data
        joints = sc.LEFT_ARM_JOINTS if side == "left" else sc.RIGHT_ARM_JOINTS
        _, self.qadr, self.dadr = sc.joint_addrs(model, joints)
        self.ee = model.body(f"{side}_arm_gripper_base").id
        self.kp, self.kd, self.kp_r, self.kd_r = kp, kd, kp_r, kd_r
        self.held_bid = held_bid
        self.lam = lam
        self.q0 = data.qpos[self.qadr].copy()
        self.max_tau = 0.0
        self.sat_steps = 0

    def step(self, p_ref, q_ref, v_ref=np.zeros(3), w_ref=np.zeros(3), a_ref=np.zeros(3), al_ref=np.zeros(3)):
        m, d = self.m, self.d
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, d.xpos[self.ee], self.ee)
        jpd = np.zeros((3, m.nv)); jrd = np.zeros((3, m.nv))
        mujoco.mj_jacDot(m, d, jpd, jrd, d.xpos[self.ee], self.ee)
        J = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
        Jd = np.vstack([jpd[:, self.dadr], jrd[:, self.dadr]])
        qd = d.qvel[self.dadr]
        v = J @ qd
        e_p = p_ref - d.xpos[self.ee]
        e_r = world_rot_err(q_ref, d.xquat[self.ee])
        a_t = np.concatenate([a_ref + self.kp * e_p + self.kd * (v_ref - v[:3]),
                              al_ref + self.kp_r * e_r + self.kd_r * (w_ref - v[3:])])
        Jpinv = J.T @ np.linalg.inv(J @ J.T + self.lam ** 2 * np.eye(6))
        qdd = Jpinv @ (a_t - Jd @ qd)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, M, d.qM)
        Ma = M[np.ix_(self.dadr, self.dadr)]
        tau = Ma @ qdd + d.qfrc_bias[self.dadr] - d.qfrc_passive[self.dadr]
        if self.held_bid is not None:
            Mb, bb = sc.payload_reflected_terms(m, d, self.ee, self.held_bid, self.dadr)
            tau = tau + Mb @ qdd + bb
        if np.any(np.abs(tau) > sc.TAU_MAX):
            self.sat_steps += 1
        tau = np.clip(tau, -sc.TAU_MAX, sc.TAU_MAX)
        self.max_tau = max(self.max_tau, float(np.max(np.abs(tau))))
        d.qfrc_applied[self.dadr] = tau
        return e_p, e_r


def jaw_geoms(model, side):
    out = []
    for fb in (f"{side}_arm_link7", f"{side}_arm_link8"):
        bid = model.body(fb).id
        out.append(set(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid
                       and (model.geom_contype[g] or model.geom_conaffinity[g])))
    return out


def contact_stats(model, data, obj_bid, jaw7, jaw8):
    """Per-step grasp contact metrics between the object body and each jaw."""
    n7 = n8 = 0
    f7 = f8 = 0.0
    util_max = 0.0
    dist_min = 1.0
    f6 = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
        if b1 == obj_bid:
            other = con.geom2
        elif b2 == obj_bid:
            other = con.geom1
        else:
            continue
        in7, in8 = other in jaw7, other in jaw8
        if not (in7 or in8):
            continue
        mujoco.mj_contactForce(model, data, c, f6)
        fn = abs(f6[0])
        mu = con.friction
        if con.dim >= 3 and fn > 1e-9:
            s = (f6[1] / mu[0]) ** 2 + (f6[2] / mu[1]) ** 2
            if con.dim >= 4:
                s += (f6[3] / mu[2]) ** 2
            util_max = max(util_max, float(np.sqrt(s) / fn))
        dist_min = min(dist_min, float(con.dist))
        if in7:
            n7 += 1; f7 += fn
        else:
            n8 += 1; f8 += fn
    return n7, n8, f7, f8, util_max, dist_min
