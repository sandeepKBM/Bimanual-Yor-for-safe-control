"""nullspace.py -- self-motion (elbow) demo: hold the gripper pose (hence the welded tray pose) fixed and move the elbow
through the 1-D nullspace of the 6x7 arm Jacobian by +-X cm.

Kinematic part: continuation q <- q + ds*n(q) (n = null vector of the 6x7 [lin;rot] Jacobian at the ee_grasp site), each step
followed by Newton projection back onto the fixed-pose manifold (error < 1e-10).  Elbow point = origin of `{side}_arm_link4`
(joint-4 axis).  Tray-pose error = pose of the tray implied by that hand through the captured weld relpose, vs the hold pose.

    python -m world_model.v2.nero.nullspace [--dynamic]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from . import config as C
from . import adapters as A
from . import ik as IK


def rotvec_err(R, R0):
    """world-frame small rotation vector taking R to R0 (R0 R^T)."""
    E = R0 @ R.T
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def null_vec(J):
    return np.linalg.svd(J)[2][-1]


class SelfMotion:
    def __init__(self, m, side, q_hold, weld_data=None):
        self.m, self.side = m, side
        self.ik = IK.ArmIK(m, side)
        self.q0 = np.asarray(q_hold, float).copy()
        self.p0, self.R0 = self.ik.pose(self.q0)             # ee_grasp site pos, gripper_base rot
        self.e0 = self.ik.elbow(self.q0)
        eq = m.eq_data[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"hold_{side}")]
        self.rel = eq[3:6].copy()                              # tray centre in gripper_base frame (weld relpose)
        self.gb = m.body(f"{side}_arm_gripper_base").id

    def tray_pose(self, q):
        """tray centre + rotation implied by this hand (weld relpose: tray_pos = x_gb + R_gb rel; tray_R = R_gb R_rel)."""
        self.ik.set_q(q)
        d = self.ik.d
        Rg = d.xmat[self.gb].reshape(3, 3).copy()
        return d.xpos[self.gb] + Rg @ self.rel, Rg                # rotation error of tray == rotation error of gripper_base

    def pose_err(self, q):
        p, R = self.ik.pose(q)
        return self.p0 - p, rotvec_err(R, self.R0)

    def project(self, q, iters=20, lam=1e-10):
        for _ in range(iters):
            ep, er = self.pose_err(q)
            e = np.concatenate([ep, er])
            if np.linalg.norm(e) < 1e-12:
                break
            J = self.ik.jac6(q)
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e), self.ik.lo, self.ik.hi)
        return q

    def branch(self, sign, max_shift=0.25, ds=0.004, margin=1e-3):
        """walk the self-motion manifold in direction `sign` until the elbow moved `max_shift` or a joint limit (within margin)."""
        q = self.q0.copy()
        n_prev = None
        path = [(0.0, q.copy())]
        for _ in range(int(3.0 / ds)):
            J = self.ik.jac6(q)
            n = null_vec(J)
            if n_prev is None:
                # fix the branch orientation by the elbow motion direction: sign * (n . d elbow/dq) > 0 along world z? use joint-3 (index 2) sign
                n = n if n[2] * sign >= 0 else -n
            else:
                n = n if n @ n_prev >= 0 else -n
            n_prev = n
            qn = self.project(q + ds * n)
            if np.any(qn <= self.ik.lo + margin) or np.any(qn >= self.ik.hi - margin):
                break
            q = qn
            shift = float(np.linalg.norm(self.ik.elbow(q) - self.e0))
            path.append((shift, q.copy()))
            if shift >= max_shift:
                break
        return path

    def report(self, path, levels=(0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20), d_check=None, pb=None):
        out = []
        p_tray0, _ = self.tray_pose(self.q0)
        for X in levels:
            hit = [(s, q) for s, q in path if s >= X]
            if not hit:
                out.append(dict(requested_cm=100 * X, reached=False))
                continue
            s, q = hit[0]
            ep, er = self.pose_err(q)
            pt, _ = self.tray_pose(q)
            row = dict(requested_cm=100 * X, reached=True, elbow_shift_cm=100 * s,
                       hand_pos_err_mm=1e3 * float(np.linalg.norm(ep)), hand_rot_err_mrad=1e3 * float(np.linalg.norm(er)),
                       tray_pos_err_mm=1e3 * float(np.linalg.norm(pt - p_tray0)),
                       tray_rot_err_mrad=1e3 * float(np.linalg.norm(er)),
                       elbow_delta_xyz_cm=(100 * (self.ik.elbow(q) - self.e0)).round(2).tolist(),
                       dq_rad=np.abs(q - self.q0).max().round(4).item(), limit_margin=self.ik.margin(q))
            out.append(row)
        return out


def run_kinematic(hold_q, m=None, d=None):
    """-> dict of {side: {+: rows, -: rows, max_reach}}; also whole-path worst pose error and contacts/min-h along the paths."""
    if m is None:
        m = A.compose_scene()
        d = mujoco.MjData(m)
        info = IK.set_hold_state(m, d)
        hold_q = info["hold_q"]
    pb = A.PairBarriers(m)
    res = {}
    d2 = mujoco.MjData(m)
    for side in C.SIDES:
        sm = SelfMotion(m, side, hold_q[side])
        other = "right" if side == "left" else "left"
        qo = np.asarray(hold_q[other])
        qa_o, _ = A.arm_addrs(m, other)
        qa_s, _ = A.arm_addrs(m, side)
        r = {}
        for sign, key in ((+1, "plus"), (-1, "minus")):
            path = sm.branch(sign)
            worst_pos = worst_rot = 0.0
            ncon_max = 0
            hmin = 1e9
            for s, q in path:
                ep, er = sm.pose_err(q)
                worst_pos, worst_rot = max(worst_pos, np.linalg.norm(ep)), max(worst_rot, np.linalg.norm(er))
                d2.qpos[:] = d.qpos
                d2.qpos[qa_s] = q
                d2.qpos[qa_o] = qo
                mujoco.mj_forward(m, d2)
                ncon_max = max(ncon_max, int(d2.ncon))
                hmin = min(hmin, pb.min_h(d2))
            r[key] = dict(max_elbow_shift_cm=100 * path[-1][0], n_steps=len(path), rows=sm.report(path),
                          worst_hand_pos_err_mm=1e3 * float(worst_pos), worst_hand_rot_err_mrad=1e3 * float(worst_rot),
                          max_contacts_along_path_excl_weld=ncon_max, min_proxy_h_along_path_m=float(hmin),
                          path_q_end=path[-1][1].round(4).tolist())
        res[side] = r
    return res


def run_dynamic(side="left", shift_cm=4.0, t_ramp=2.0, t_hold=1.0, sign=+1, t_settle=1.0):
    """Closed-loop: tray welded, FL-MPC (NV=7) tracks a joint reference that ramps the `side` arm along the self-motion path to
    an elbow shift of `shift_cm` while the other arm keeps its hold pose.  Reports tray pose deviation from the (settled) hold pose."""
    from . import hold_test as H
    tau = C.tau_limits()
    m = A.compose_scene(tau=tau)
    d = mujoco.MjData(m)
    info = IK.set_hold_state(m, d)
    A.disable_arm_servos(m)
    sm = SelfMotion(m, side, info["hold_q"][side])
    path = sm.branch(sign, max_shift=shift_cm / 100.0 + 0.01)
    hit = [(s, q) for s, q in path if s >= shift_cm / 100.0]
    if not hit:
        return dict(error=f"elbow shift {shift_cm} cm not reachable (max {100*path[-1][0]:.1f} cm)")
    q_goal = hit[0][1]
    ctl = H.Ctl(m, info["hold_q"], tau)
    NP = ctl.fl.NP
    dt = m.opt.timestep
    q_hold = np.array(info["hold_q"][side])
    tray = m.body("tray").id
    n_tot = int((t_settle + t_ramp + t_hold) / dt)
    n_settle = int(t_settle / dt)

    def qref(t):
        a = np.clip(t / t_ramp, 0.0, 1.0)
        a = a * a * (3 - 2 * a)                         # smoothstep
        return q_hold + a * (q_goal - q_hold)

    def qdref(t):
        if t <= 0 or t >= t_ramp:
            return np.zeros(7)
        a = t / t_ramp
        return (6 * a * (1 - a) / t_ramp) * (q_goal - q_hold)

    sm_check = SelfMotion(m, side, q_goal)
    dev, rotdev, hand_err = [], [], []
    tray0 = None
    for k in range(n_tot):
        t = max(0.0, k * dt - t_settle)
        refs = np.array([qref(t + i * dt) for i in range(NP + 1)])
        vrefs = np.array([qdref(t + i * dt) for i in range(NP + 1)])
        mujoco.mj_forward(m, d)
        pairs = ctl.pairs(d)
        qe_ref, qe_vref, qi_ref, qi_vref = (refs, vrefs, ctl.qi_ref, ctl.z) if side == "left" else (ctl.qe_ref, ctl.z, refs, vrefs)
        te, ti, ok = ctl.fl.step(m, d, ctl.lq, ctl.ld, ctl.rq, ctl.rd, qe_ref, qe_vref, qi_ref, qi_vref, pairs)
        if not ok:
            ctl.n_infeasible += 1
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        mujoco.mj_step(m, d)
        if k == n_settle - 1:        # baseline = tray pose after the initial weld sag has settled (reference still = hold pose)
            tray0 = d.xpos[tray].copy(); R0 = d.xmat[tray].reshape(3, 3).copy()
        if k < n_settle:
            continue
        dev.append(float(np.linalg.norm(d.xpos[tray] - tray0)))
        rotdev.append(float(np.linalg.norm(rotvec_err(d.xmat[tray].reshape(3, 3), R0))))
    q_end = d.qpos[ctl.lq if side == "left" else ctl.rq].copy()
    elbow_end = sm.ik.elbow(q_end)
    return dict(side=side, shift_cm_requested=shift_cm, elbow_shift_achieved_cm=100 * float(np.linalg.norm(elbow_end - sm.e0)),
                n_infeasible=ctl.n_infeasible, tray_pos_dev_max_mm=1e3 * max(dev), tray_pos_dev_final_mm=1e3 * dev[-1],
                tray_rot_dev_max_mrad=1e3 * max(rotdev), tray_rot_dev_final_mrad=1e3 * rotdev[-1],
                ball_on_tray=bool(np.linalg.norm((d.xpos[m.body("ball").id] - d.xpos[tray])[:2]) < 0.1),
                q_err_final_rad=float(np.abs(q_end - q_goal).max()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "nullspace.json"))
    a = ap.parse_args(argv)
    res = dict(kinematic=run_kinematic(None))
    if a.dynamic:
        res["dynamic"] = [run_dynamic("left", 4.0, sign=s) for s in (+1, -1)] + [run_dynamic("right", 4.0, sign=+1)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for side, r in res["kinematic"].items():
        for k, v in r.items():
            print(side, k, f"max_elbow_shift={v['max_elbow_shift_cm']:.1f}cm worst_hand_err={v['worst_hand_pos_err_mm']:.2e}mm/{v['worst_hand_rot_err_mrad']:.2e}mrad ncon={v['max_contacts_along_path_excl_weld']} min_h={v['min_proxy_h_along_path_m']:.3f}")
            for row in v["rows"]:
                if row.get("reached"):
                    print("   ", {k2: (round(x, 5) if isinstance(x, float) else x) for k2, x in row.items()})
    if a.dynamic:
        print(json.dumps(res["dynamic"], indent=1))


if __name__ == "__main__":
    main()
