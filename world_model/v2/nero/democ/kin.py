"""democ/kin.py -- kinematic helpers shared by grasp_smoke / turnover_sweep / tests (SIMULATION ONLY, CPU).

* full 6-D pose IK (no jaw-sign ambiguity: the gripper frame is fully constrained so that a closed-chain rotation of the box
  about the hand-to-hand axis y is tracked CONTINUOUSLY, unlike ik.ArmIK.solve which treats +-jaw as equivalent)
* box grasp frames (hand site targets + gripper_base rotations) for a given turnover angle
* pre-grasp search (contact-free IK pair, open fingers) and state setup
"""
from __future__ import annotations

import numpy as np
import mujoco

from .. import config as C
from .. import adapters as A
from .. import ik as IK
from ..nullspace import rotvec_err
from . import scene as S

FINGER_OPEN = 0.04          # per-finger travel at pre-grasp: gap 0.08 m vs 0.06 m box thickness (1 cm clearance per side)
INSET = 0.02                # ee_grasp site sits 2 cm inside the box end face (pad overlap ~3.8 cm along approach)
NAMES_OK = ("democ_box_geom", "democ_table_top", "democ_table_pedestal", "floor")


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def grasp_rot(approach, jaw):
    """gripper_base rotation with z = approach, y = jaw (finger-closing axis), x = y x z."""
    z = np.asarray(approach, float); y = np.asarray(jaw, float)
    return np.stack([np.cross(y, z), y, z], axis=1)


def box_grasp_targets(p: S.DemoCParams, theta=0.0, jaw_sign=(1.0, 1.0), inset=INSET):
    """{side: (site_pos, R_gripper)} for the box rotated by `theta` about world y through its centre.
    left hand at +y end (approach -y), right hand at -y end (approach +y); jaw axis = box z axis rotated by theta."""
    c = np.array(p.box_centre, float)
    h = p.box_half[1] - inset
    out = {}
    for side, sy, ap, js in (("left", +1.0, (0, -1.0, 0), jaw_sign[0]), ("right", -1.0, (0, 1.0, 0), jaw_sign[1])):
        R0 = grasp_rot(ap, (0, 0, js))
        out[side] = (c + np.array([0.0, sy * h, 0.0]), Ry(theta) @ R0)
    return out


class PoseIK:
    def __init__(self, m, side):
        self.ik = IK.ArmIK(m, side)
        self.m, self.side = m, side
        self.lo, self.hi = self.ik.lo, self.ik.hi
        self.q_post = 0.5 * (self.lo + self.hi)

    def err(self, q, p, R):
        pc, Rc = self.ik.pose(q)
        return np.concatenate([p - pc, rotvec_err(Rc, R)])

    def solve(self, p, R, q0, q_post=None, k_post=0.3, iters=300, tol=1e-10, lam=1e-9, max_step=0.2):
        """DLS pose IK + nullspace posture; q clipped to the URDF limits.  -> (q, |e|_2 over [pos m; rotvec rad])"""
        qp = self.q_post if q_post is None else q_post
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        for _ in range(iters):
            e = self.err(q, p, R)
            J = self.ik.jac6(q)
            Jp = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
            en = np.linalg.norm(e)
            dq = Jp @ e
            dn = (np.eye(7) - Jp @ J) @ (k_post * (qp - q))
            if en < tol and np.linalg.norm(dn) < 1e-7:
                break
            qn = np.clip(q + np.clip(dq + dn, -max_step, max_step), self.lo, self.hi)
            if en < tol:                              # converged in task space: only polish nullspace if it does not hurt
                if np.linalg.norm(self.err(qn, p, R)) > 10 * tol:
                    break
            q = qn
        return q, float(np.linalg.norm(self.err(q, p, R)))

    def margin(self, q):
        """(min joint-limit distance [rad], joint index 1..7)"""
        dist = np.minimum(q - self.lo, self.hi - q)
        j = int(np.argmin(dist))
        return float(dist[j]), j + 1

    def sigma_min(self, q):
        return float(np.linalg.svd(self.ik.jac6(q), compute_uv=False)[-1])


def collect_pose(pik: PoseIK, p, R, n_sol=12, seeds=60, seed=0, ok=1e-6):
    rng = np.random.default_rng(seed)
    sols = []
    for s in range(seeds):
        q0 = pik.q_post if s == 0 else rng.uniform(pik.lo, pik.hi)
        qpt = pik.q_post if s == 0 else rng.uniform(pik.lo, pik.hi)
        q, e = pik.solve(p, R, q0, qpt, iters=250)
        if e < ok and all(np.abs(q - o[0]).max() > 0.08 for o in sols):
            sols.append((q, e))
        if len(sols) >= n_sol:
            break
    return sols


def set_state(m, d, p: S.DemoCParams, ql, qr, finger=FINGER_OPEN, box_pose=None):
    """arms at ql/qr, fingers at `finger`, box resting at its nominal pose, ctrl = hold targets."""
    qa_l, _ = A.arm_addrs(m, "left"); qa_r, _ = A.arm_addrs(m, "right")
    d.qpos[qa_l] = ql; d.qpos[qa_r] = qr
    for side in C.SIDES:
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_gripper_joint1").id]] = finger
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_gripper_joint2").id]] = -finger
        q = ql if side == "left" else qr
        for i in range(7):
            d.ctrl[m.actuator(f"{side}_arm_joint{i+1}_pos").id] = q[i]
        d.ctrl[m.actuator(f"{side}_arm_gripper_pos").id] = finger
    ba = m.jnt_qposadr[m.joint("democ_box_free").id]
    d.qpos[ba:ba + 3] = p.box_centre
    d.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)


def bad_contacts(m, d):
    """contacts that are not box<->table/pedestal (any contact involving a robot geom, or box vs robot, is 'bad')."""
    ok = {m.geom(n).id for n in NAMES_OK}
    return [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon] if not (c.geom1 in ok and c.geom2 in ok)]


def find_pregrasp(m, p: S.DemoCParams, n_sol=12, seeds=60, min_h=0.03, finger=FINGER_OPEN, theta=0.0):
    """Best contact-free (L, R) IK pair at turnover angle `theta`: tries both jaw signs per hand; max joint-limit margin among
    pairs with no robot contact and arm-arm proxy h >= min_h.  -> dict"""
    d = mujoco.MjData(m)
    pb = A.PairBarriers(m)
    sols = {}
    for side in C.SIDES:
        pik = PoseIK(m, side)
        for js in (+1.0, -1.0):
            tg = box_grasp_targets(p, theta, (js, js))[side]
            sols[(side, js)] = [(pik, q, e) for q, e in collect_pose(pik, tg[0], tg[1], n_sol, seeds, seed=0 if side == "left" else 1)]
    best = None
    for js in (+1.0, -1.0):
        for jr in (+1.0, -1.0):
            for (pl, ql, el) in sols[("left", js)]:
                for (pr, qr, er) in sols[("right", jr)]:
                    set_state(m, d, p, ql, qr, finger)
                    if bad_contacts(m, d):
                        continue
                    h = pb.min_h(d)
                    if h < min_h:
                        continue
                    score = min(pl.margin(ql)[0], pr.margin(qr)[0])
                    if best is None or score > best["score"]:
                        best = dict(score=score, ql=ql, qr=qr, el=el, er=er, h=h, jaw_sign=(js, jr))
    if best is None:
        raise RuntimeError("no contact-free pre-grasp IK pair (solutions per side/sign: %s)"
                           % {f"{k[0]}{int(k[1]):+d}": len(v) for k, v in sols.items()})
    return best
