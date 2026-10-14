"""grasp/hands.py -- per-hand Cartesian targets -> DLS IK -> joint references of the unmodified FL-MPC.  SIMULATION ONLY.

The teleop ObjectFrame moves BOTH hands rigidly with one object frame.  The assisted grasp needs the hands to move independently (each one
follows its own wrist-camera estimate), so this module reuses ObjectFrame's FK / Jacobian / joint-limit machinery (`of._hands`, `of._jac`,
`of.jlo/jhi`, the same damping and step clip) but with a separate target per hand.  The result is written through `ArmCtl.set_ref`, i.e.
the same path the operator's tray jog uses; there is no teleport and FL-MPC (arm-arm clearance, torque box) stays in the loop.

All poses here are in the ROBOT frame of ObjectFrame (root frame at episode start, lift removed), p (3,), R (3,3).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from world_model.v2.house.teleop.arm import _log_R, _rotvec_to_R

V_MAX, W_MAX = 0.10, 0.60            # target slew [m/s], [rad/s]


def rot_err(R_t, R):
    return _log_R(R_t @ R.T)


class HandServo:
    def __init__(self, sim):
        self.sim = sim
        self.of = sim.of
        self.m = sim.m
        self.tgt = [None, None]           # commanded (slewed) target per hand: (p, R) robot frame
        self.goal = [None, None]          # final target per hand
        self.active = False
        self.ik_resid = [0.0, 0.0]
        self.corr = [(np.zeros(3), np.zeros(3)), (np.zeros(3), np.zeros(3))]   # integral correction (dp, drotvec) per hand
        self.ki = 1.2                     # [1/s] integral gain on (goal - ACTUAL hand pose): absorbs the FL-MPC's steady tracking offset
        self.corr_lim = (0.05, math.radians(14.0))
        self.gate_integral = False        # fast mode: integrate only while the GOAL is (nearly) still -- the lag accumulated during a move would overshoot at its end
        self._goal_prev = [None, None]
        self.moving = [False, False]
        self.v_max, self.w_max = V_MAX, W_MAX     # target slew limits [m/s], [rad/s] (the assist raises them in its fast mode)
        self.w_rot = 1.0                  # weight of the rotation error in the IK (1 = as the operator's jog)
        # true joint limits with a 5 mrad margin (the operator jog keeps 30 mrad; the hold pose itself sits at -1.188 vs a -1.17 jog limit)
        rng = lambda s_: np.array([self.m.jnt_range[self.m.joint(f"{s_}_arm_joint{i}").id] for i in range(1, 7)])
        self.jlo = [rng(s_)[:, 0] + 0.005 for s_ in ("left", "right")]
        self.jhi = [rng(s_)[:, 1] - 0.005 for s_ in ("left", "right")]

    # ---- measurement (true kinematics = encoders + known geometry)
    def hand_pose(self, k):
        d, m, of = self.sim.d, self.m, self.of
        b = (of.bL, of.bR)[k]
        root = m.body("yor_root").id
        Rr = d.xmat[root].reshape(3, 3)
        pr = d.xpos[root]
        lift = self.sim.lift_actual()
        p = Rr.T @ (d.xpos[b] - pr) - np.array([0.0, 0.0, lift])
        R = Rr.T @ d.xmat[b].reshape(3, 3)
        # ObjectFrame's robot frame is the root frame at episode start (yaw/pos of the mocap at init); the root frame here is the
        # same robot frame because the base never rotates relative to itself -> identical coordinates.
        return p, R

    def set_goal(self, k, p, R, jump=False):
        self.goal[k] = (np.array(p, float), np.array(R, float))
        if self.tgt[k] is None or jump:
            self.tgt[k] = (np.array(p, float), np.array(R, float))

    def start_from_current(self):
        self.corr = [(np.zeros(3), np.zeros(3)), (np.zeros(3), np.zeros(3))]
        for k in (0, 1):
            p, R = self.hand_pose(k)
            self.tgt[k] = (p.copy(), R.copy())
            self.goal[k] = (p.copy(), R.copy())
        self.active = True

    def at_goal(self, k, tol_p=0.003, tol_r=math.radians(1.0)):
        p, R = self.hand_pose(k)
        gp, gR = self.goal[k]
        return np.linalg.norm(gp - p) < tol_p and np.linalg.norm(rot_err(gR, R)) < tol_r

    def goal_error(self, k):
        p, R = self.hand_pose(k)
        gp, gR = self.goal[k]
        return float(np.linalg.norm(gp - p)), float(np.degrees(np.linalg.norm(rot_err(gR, R))))

    # ---- one 50 Hz update: slew the target toward the goal, IK, write FL-MPC references
    def step(self, dt, slow=1.0):
        if not self.active:
            return
        of = self.of
        for k in (0, 1):
            if self.goal[k] is None:
                continue
            tp, tR = self.tgt[k]
            gp, gR = self.goal[k]
            dp = gp - tp
            n = np.linalg.norm(dp)
            lim = self.v_max * slow * dt
            tp = tp + (dp if n <= lim else dp * lim / n)
            w = rot_err(gR, tR)
            nw = np.linalg.norm(w)
            limw = self.w_max * slow * dt
            tR = (_rotvec_to_R(w if nw <= limw else w * limw / nw)) @ tR
            self.tgt[k] = (tp, tR)
            if self.gate_integral:
                pg = self._goal_prev[k]
                self.moving[k] = pg is not None and (np.linalg.norm(gp - pg[0]) / dt > 0.02 or np.linalg.norm(rot_err(gR, pg[1])) / dt > 0.15)
                self._goal_prev[k] = (gp.copy(), gR.copy())
            # integral action on the ACTUAL hand pose (encoders): the FL-MPC leaves a joint-space steady offset that the camera loop must remove
            p, R = self.hand_pose(k)
            cp, cr = self.corr[k]
            ki = 0.0 if (self.gate_integral and self.moving[k]) else self.ki
            cp = np.clip(cp + ki * dt * (tp - p), -self.corr_lim[0], self.corr_lim[0])
            cr = cr + ki * dt * rot_err(tR, R)
            nr = np.linalg.norm(cr)
            if nr > self.corr_lim[1]:
                cr = cr * self.corr_lim[1] / nr
            self.corr[k] = (cp, cr)
        self._ik()

    def _ik(self, iters=3):
        of = self.of
        for _ in range(iters):
            hands = of._hands(of.qL, of.qR)
            for k, (b, dofs, attr) in enumerate(((of.bL, of.dofL, "qL"), (of.bR, of.dofR, "qR"))):
                if self.tgt[k] is None:
                    continue
                p, R = hands[k]
                tp, tR = self.tgt[k]
                cp, cr = self.corr[k]
                tp, tR = tp + cp, _rotvec_to_R(cr) @ tR
                e = np.concatenate([tp - p, self.w_rot * _log_R(tR @ R.T)])
                self.ik_resid[k] = float(np.linalg.norm(e[:3]))
                J = of._jac(b, dofs)
                dq = J.T @ np.linalg.solve(J @ J.T + of.damp ** 2 * np.eye(6), e)
                q = np.clip(getattr(of, attr) + np.clip(dq, -0.05, 0.05), self.jlo[k], self.jhi[k])
                setattr(of, attr, q)
        of.ctl.set_ref(of.qL, of.qR)

    def rebase_object_frame(self, tray_world=None):
        """After a verified grasp: make the operator's tray-frame jog continue from the CURRENT hand poses (no jump).
        hand_i = G * T_i with T_i taken at the current hands; G = the tray centre in the robot frame."""
        of = self.of
        hands = of._hands(of.qL, of.qR)
        of.hand0 = [(p.copy(), R.copy()) for p, R in hands]
        tray = self.sim.d.xpos[self.sim.tray] if tray_world is None else np.asarray(tray_world, float)       # tray_world: proprio (hand FK + grasp offset)
        root = self.m.body("yor_root").id
        Rr = self.sim.d.xmat[root].reshape(3, 3)
        of.G0p = Rr.T @ (tray - self.sim.d.xpos[root]) - np.array([0.0, 0.0, self.sim.lift_actual()])
        of.dp = np.zeros(3)
        of.dR = np.eye(3)
        self.active = False
