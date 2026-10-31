"""pick/ik.py -- multi-start damped-least-squares IK for ONE Piper arm on the robot-only kinematic model (build._ik_model), robot frame.  SIMULATION ONLY.

Same residual as build.solve_hold_ik (ee_grasp site position + approach axis (gripper_base z) + jaw axis (finger slide axis, up to sign)), but
  * warm-started from caller-supplied seeds (the current joint vector first) -> typically 1 seed converges when the pose is near,
  * returns the solution nearest to the first seed among the converged ones, plus the residuals,
  * joint limits = the MJCF ranges with a 0.03 rad margin.
No MuJoCo contact model here: collision checking is done by pick.skill against the live sim model.
"""
from __future__ import annotations

import numpy as np
import mujoco

from world_model.v2.house import build as B

QMARGIN = 0.03
SITE_TO_PAD = 0.1 - 0.0908      # ee_grasp site is 0.100 m along the approach axis, the pad centres 0.0908 m (model: left_arm_link7_pad in gripper_base frame)


def _arm(side):
    m = B._ik_model()
    pre = f"{side}_arm_"
    qadr, dadr, jids = B._jaddr(m, [f"{pre}joint{i}" for i in range(1, 7)])
    return m, qadr, dadr, jids, m.site(f"{pre}ee_grasp").id, m.body(f"{pre}gripper_base").id, m.joint(f"{pre}joint7").id


def limits(side):
    m, qadr, dadr, jids, *_ = _arm(side)
    return m.jnt_range[jids, 0] + QMARGIN, m.jnt_range[jids, 1] - QMARGIN


def fk(side, q):
    """-> (site position, gripper_base rotation (3,3)) in the robot frame."""
    m, qadr, dadr, jids, site, gb, j7 = _arm(side)
    dd = mujoco.MjData(m)
    dd.qpos[qadr] = q
    mujoco.mj_kinematics(m, dd)
    return dd.site_xpos[site].copy(), dd.xmat[gb].reshape(3, 3).copy()


def solve(site_pos, approach, jaw, side, seeds_q=(), n_random=14, iters=250, tol=(3e-4, 3e-3, 3e-2), rng_seed=0):
    """-> (q (6,), dict(pos_err, app_err, jaw_err, ok, n_seed)) ; q None when no seed converged (the best residual is still reported)."""
    m, qadr, dadr, jids, site, gb, j7 = _arm(side)
    lo, hi = limits(side)
    approach = np.asarray(approach, float) / np.linalg.norm(approach)
    jaw = np.asarray(jaw, float) / np.linalg.norm(jaw)
    q_home = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78 if side == "left" else -0.78])
    rng = np.random.default_rng(rng_seed)
    seeds = [np.clip(np.asarray(s, float), lo, hi) for s in seeds_q] + [q_home]
    seeds += [np.clip(q_home + rng.normal(0, 0.6, 6), lo, hi) for _ in range(n_random)]
    dd = mujoco.MjData(m)
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    best = (1e9, None, None)
    sols = []
    for si, q0 in enumerate(seeds):
        q = q0.copy()
        for it in range(iters):
            dd.qpos[qadr] = q
            mujoco.mj_kinematics(m, dd)
            mujoco.mj_comPos(m, dd)
            p = dd.site_xpos[site]
            app = dd.xmat[gb].reshape(3, 3)[:, 2]
            jx = dd.xaxis[j7]
            sgn = 1.0 if jx @ jaw >= 0 else -1.0
            e_p = site_pos - p
            e_a = approach - app                       # vector difference (NOT a cross product: build.solve_hold_ik's cross residual is blind to an antiparallel axis)
            e_j = sgn * jaw - jx
            if np.linalg.norm(e_p) < tol[0] and np.linalg.norm(e_a) < tol[1] and np.linalg.norm(e_j) < tol[2]:
                break
            mujoco.mj_jacSite(m, dd, jp, jr, site)
            Ja = -B.skew(app) @ jr[:, dadr]              # d app = w x app = -[app]x w
            Jj = -B.skew(jx) @ jr[:, dadr]
            J = np.vstack([jp[:, dadr], 0.5 * Ja, 0.3 * Jj])
            err = np.concatenate([e_p, 0.5 * e_a, 0.3 * e_j])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(9), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        dd.qpos[qadr] = q
        mujoco.mj_kinematics(m, dd)
        p = dd.site_xpos[site]
        app = dd.xmat[gb].reshape(3, 3)[:, 2]
        jx = dd.xaxis[j7]
        e = (float(np.linalg.norm(site_pos - p)), float(np.linalg.norm(approach - app)), float(np.linalg.norm((1 if jx @ jaw >= 0 else -1) * jaw - jx)))
        score = e[0] + e[1] + 0.3 * e[2]
        if score < best[0]:
            best = (score, q.copy(), e)
        if e[0] < tol[0] * 3 and e[1] < tol[1] * 3 and e[2] < tol[2] * 3:
            sols.append((q.copy(), e, si))
            if si == 0 or len(sols) >= 2:
                break
    if sols:
        sols.sort(key=lambda s: np.abs(s[0] - seeds[0]).max())
        q, e, si = sols[0]
        return q, dict(pos_err=e[0], app_err=e[1], jaw_err=e[2], ok=True, n_seed=si)
    _, q, e = best
    return None, dict(pos_err=e[0], app_err=e[1], jaw_err=e[2], ok=False, n_seed=-1)
