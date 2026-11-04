"""ik.py -- 7-DoF IK for the bimanual tray hold with a NULLSPACE POSTURE task, self-motion (elbow) continuation, hold-state setup.

Task per arm (same 6 constraints as house/build.solve_hold_ik for the 6-DoF Piper):  ee_grasp site at `hand`, gripper_base z
(approach) along `approach`, gripper_base y (finger-closing axis) along +-`jaw`.  With 7 joints the arm has a 1-D
self-motion manifold; the posture task resolves it:  dq = J^+ e + (I - J^+ J) k (q_post - q).
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import config as C
from . import adapters as A
from . import gripper as G

# ---- tray hold geometry: SAME numbers as world_model/v2/house/build.py (robot-local frame)
TRAY_LOCAL = (0.45, -0.21, 0.85)          # tray centre: ~0.45 m ahead, 0.85 m high (house/phase1 hold pose)
TRAY_HX, TRAY_HY = 0.15, 0.10
INSET = 0.012
FINGER_OPEN = 0.35                        # opening fraction at hold (earlier form: 17.5 mm of 50 mm per-finger travel)
HOLD_MODES = {"side": ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))}   # left approach, right approach, jaw axis


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def tray_hold_targets(tray_local=TRAY_LOCAL, mode="side", inset=INSET, hx=TRAY_HX):
    c = np.array(tray_local, float)
    al, ar, jw = (np.array(v, float) for v in HOLD_MODES[mode])
    h = hx - inset                        # across=True: hands on the SHORT rims (|y| = 0.138), hand separation 0.276 m
    return c, c + [0.0, h, 0.0], c + [0.0, -h, 0.0], al, ar, jw


class ArmIK:
    def __init__(self, m, side):
        self.m, self.side = m, side
        self.d = mujoco.MjData(m)
        self.qa, self.da = A.arm_addrs(m, side)
        self.site = m.site(f"{side}_arm_ee_grasp").id
        self.gb = m.body(f"{side}_arm_gripper_base").id
        self.lo, self.hi = C.URDF_LIMITS[:, 0].copy(), C.URDF_LIMITS[:, 1].copy()
        self.q_post = 0.5 * (self.lo + self.hi)        # default posture: mid-range

    def set_q(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def pose(self, q):
        self.set_q(q)
        return self.d.site_xpos[self.site].copy(), self.d.xmat[self.gb].reshape(3, 3).copy()

    def jac6(self, q):
        """(6,7) [lin; rot] Jacobian of the ee_grasp site (world frame)."""
        self.set_q(q)
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
        return np.vstack([jp[:, self.da], jr[:, self.da]])

    def solve(self, hand, approach, jaw, q0, q_post=None, k_post=0.5, iters=400, lam=1e-7, tol=(1e-5, 1e-4, 1e-3)):
        """DLS + nullspace posture.  -> (q, (pos_err, app_err, jaw_err)).  q is clipped to the URDF limits each iteration."""
        qp = self.q_post if q_post is None else q_post
        q = np.array(q0, float)
        m, d = self.m, self.d
        for it in range(iters):
            self.set_q(q)
            p = d.site_xpos[self.site]
            Rg = d.xmat[self.gb].reshape(3, 3)
            app, jw = Rg[:, 2], Rg[:, 1]
            sgn = 1.0 if jw @ jaw >= 0 else -1.0
            e_p = hand - p
            e_a = np.cross(app, approach)
            e_j = np.cross(jw, sgn * jaw)
            if np.linalg.norm(e_p) < tol[0] and np.linalg.norm(e_a) < tol[1] and np.linalg.norm(e_j) < tol[2]:
                break
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            Ja = skew(approach) @ skew(app) @ jr[:, self.da]
            Jj = skew(sgn * jaw) @ skew(jw) @ jr[:, self.da]
            J = np.vstack([jp[:, self.da], 0.5 * Ja, 0.3 * Jj])
            err = np.concatenate([e_p, -0.5 * e_a, -0.3 * e_j])
            Jpinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(9))
            dq = Jpinv @ err + (np.eye(7) - Jpinv @ J) @ (k_post * (qp - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        self.set_q(q)
        p = d.site_xpos[self.site]
        Rg = d.xmat[self.gb].reshape(3, 3)
        sgn = 1.0 if Rg[:, 1] @ jaw >= 0 else -1.0
        return q, (float(np.linalg.norm(hand - p)), float(np.linalg.norm(np.cross(Rg[:, 2], approach))),
                   float(np.linalg.norm(np.cross(Rg[:, 1], sgn * jaw))))

    def collect(self, hand, approach, jaw, n_sol=24, seeds=100, seed=0, q_post=None, ok=(2e-4, 2e-3, 3e-2)):
        rng = np.random.default_rng(seed)
        sols = []
        for s in range(seeds):
            q0 = 0.5 * (self.lo + self.hi) if s == 0 else rng.uniform(self.lo, self.hi)
            # random posture target per seed -> the nullspace task drives different points of the 1-D self-motion manifold
            qpt = q_post if (q_post is not None and s == 0) else (rng.uniform(self.lo, self.hi) if s else self.q_post)
            q, e = self.solve(hand, approach, jaw, q0, qpt, iters=300)
            if e[0] < ok[0] and e[1] < ok[1] and e[2] < ok[2] and all(np.abs(q - o[0]).max() > 0.08 for o in sols):
                sols.append((q, e))
            if len(sols) >= n_sol:
                break
        return sols

    def margin(self, q):
        """min over joints of normalised distance to the nearest limit (1 = mid-range, 0 = at a limit)."""
        return float(np.min(np.minimum(q - self.lo, self.hi - q) / (0.5 * (self.hi - self.lo))))

    def elbow(self, q):
        self.set_q(q)
        return self.d.xpos[self.m.body(f"{self.side}_arm_link4").id].copy()


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def capture_weld(m, d, eq_name, body1, body2):
    """weld target := body2's CURRENT pose in body1's frame; activate (no snap).  Same as house/build.capture_weld."""
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    b1, b2 = m.body(body1).id, m.body(body2).id
    R1 = d.xmat[b1].reshape(3, 3)
    rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
    rq = _qmul(d.xquat[b1] * np.array([1, -1, -1, -1]), d.xquat[b2])
    m.eq_data[eq, :] = 0.0
    m.eq_data[eq, 3:6] = rel
    m.eq_data[eq, 6:10] = rq / np.linalg.norm(rq)
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


def set_hold_state(m, d, tray_local=TRAY_LOCAL, finger=FINGER_OPEN, q_post=None, min_h=0.05, n_sol=24, seeds=100, verbose=False):
    """IK both arms to pinch the tray's two short rims, place tray level between the hands and the ball on it, capture
    both weld relposes (zero initial violation) and activate them.  Picks, among converged IK pairs with zero contacts and
    arm-arm proxy h >= min_h, the pair with the largest joint-limit margin.  Returns info dict."""
    c, hl, hr, al, ar, jw = tray_hold_targets(tray_local)
    ikL, ikR = ArmIK(m, "left"), ArmIK(m, "right")
    solsL = ikL.collect(hl, al, jw, n_sol, seeds, 0, q_post)
    solsR = ikR.collect(hr, ar, jw, n_sol, seeds, 1, q_post)
    pb = A.PairBarriers(m)
    best = None
    for (q1, e1) in solsL:
        for (q2, e2) in solsR:
            d.qpos[ikL.qa] = q1
            d.qpos[ikR.qa] = q2
            mujoco.mj_forward(m, d)
            if d.ncon != 0:
                continue
            h = pb.min_h(d)
            if h < min_h:
                continue
            score = min(ikL.margin(q1), ikR.margin(q2))
            if best is None or score > best[0]:
                best = (score, q1, e1, q2, e2, h)
    if best is None:
        raise RuntimeError(f"no contact-free IK pair (solutions L/R {len(solsL)}/{len(solsR)})")
    score, ql, el, qr, er, h = best
    d.qpos[ikL.qa] = ql
    d.qpos[ikR.qa] = qr
    for side in C.SIDES:
        G.set_opening(m, d, side, finger)     # finger = opening fraction (1 = open)
    ta = m.jnt_qposadr[m.joint("tray_free").id]
    d.qpos[ta:ta + 3] = c
    d.qpos[ta + 3:ta + 7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]          # yaw +90 deg: long axis across the robot
    from world_model.v2.testbeds.b_tray_ball import tray_env as TE
    bp = TE.BallParams()
    ba = m.jnt_qposadr[m.joint("ball_free").id]
    d.qpos[ba:ba + 3] = c + np.array([0.0, 0.0, bp.radius + 0.0005])
    d.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]
    d.ctrl[:] = 0
    for side in C.SIDES:     # keep position servos (if not disabled) pointing at the hold pose
        q = ql if side == "left" else qr
        for i in range(7):
            d.ctrl[m.actuator(f"{side}_arm_joint{i+1}_pos").id] = q[i]
        d.ctrl[m.actuator(f"{side}_arm_gripper_pos").id] = G.open_frac_to_ctrl(finger)
    mujoco.mj_forward(m, d)
    for side in C.SIDES:
        capture_weld(m, d, f"hold_{side}", f"{side}_arm_gripper_base", "tray")
    mujoco.mj_forward(m, d)
    return dict(hold_q=dict(left=ql.tolist(), right=qr.tolist()), ik_err=dict(left=el, right=er), n_solutions=(len(solsL), len(solsR)),
                min_limit_margin=float(score), min_pair_h=float(h), finger=finger, tray_local=list(c))
