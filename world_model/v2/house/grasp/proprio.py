"""grasp/proprio.py -- PROPRIO-ONLY switch and the gripper / torque sensor models for the friction-grasp assist.  SIMULATION ONLY.

BIMANUAL_PROPRIO_ONLY=1 (env, default OFF = bit-identical) makes GraspAssist read only what the real robot has:
  * gripper finger gap  (`gripper_pos`, finger encoders)           -> GripperSensor.gap
  * gripper effort      (`gripper_effort`, the finger motor force)   -> GripperSensor.effort
  * measured joint torque (Piper JointState.torque)                  -> PayloadEstimator
  * wrist-camera depth / RGB, hand forward kinematics from the arm encoders.
It never reads d.contact / mj_contactForce, the true masses, or d.xpos[tray] for a decision (those stay in instrumentation-only metrics).
Sensor numbers below are ASSUMED (the Piper SDK documents the fields, not their noise): they only need to be realistic, the tests sweep them.
"""
from __future__ import annotations

import os

import numpy as np

GAP_SIGMA = 2.0e-4        # finger-gap encoder noise [m]
GAP_QUANT = 1.0e-4        # [m]
EFF_SIGMA = 0.30          # finger motor force noise [N]
EFF_QUANT = 0.10          # [N]
TAU_SIGMA = 0.08          # measured joint torque noise [Nm] (per control tick, after the 5 ms substep average)
TAU_QUANT = 0.05          # [Nm]


def proprio_env() -> bool:
    return os.environ.get("BIMANUAL_PROPRIO_ONLY", "0") == "1"


def _q(x, step):
    return np.round(np.asarray(x, float) / step) * step


class GripperSensor:
    """finger gap (qpos7 - qpos8) and effort (|motor force| summed over the two fingers) with noise + quantisation."""

    def __init__(self, sim, fa, fact, seed=0):
        self.sim, self.fa, self.fact = sim, fa, fact
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, 7771]))

    def gap(self, s):
        d = self.sim.d
        a7, a8 = self.fa[s]
        return float(_q(d.qpos[a7] - d.qpos[a8] + self.rng.normal(0, GAP_SIGMA), GAP_QUANT))

    def effort(self, s):
        d = self.sim.d
        c7, c8 = self.fact[s]
        f = abs(float(d.actuator_force[c7])) + abs(float(d.actuator_force[c8]))
        return float(max(0.0, _q(f + self.rng.normal(0, EFF_SIGMA), EFF_QUANT)))


class PayloadEstimator:
    """per-hand payload mass from measured joint torque vs the model, for a hand holding the payload quasi-statically.

    MuJoCo:  M qacc + qfrc_bias = tau + qfrc_passive + qfrc_constraint, and the payload's pull on the pads enters as J_pt^T (-m g_vec).  With the
    hand at rest (qacc ~ 0):   r := tau_meas - (qfrc_bias - qfrc_passive)  =  J_pt^T e_z  (m g)   =: j m.   Least squares over the window:
    m_hat = <j, mean r> / <j, j>.  Same convention as grasp/ctl.FFArmCtl.ff_torque (weight acts at `pt` on the gripper_base body)."""

    def __init__(self, m, ld, rd, gb, seed=0):
        import mujoco
        self.mj = mujoco
        self.m = m
        self.dofs = (np.asarray(ld), np.asarray(rd))
        self.gb = gb
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, 9917]))
        self._jp = np.zeros((3, m.nv))
        self.reset()

    def reset(self):
        self.r = [[], []]
        self.j = [[], []]
        self.tau_acc = [np.zeros(6), np.zeros(6)]
        self.n_acc = 0

    def accumulate_substep(self, d):
        """call once per physics step AFTER qfrc_applied is set: running mean of the applied (= measured) arm torque."""
        self.tau_acc[0] += d.qfrc_applied[self.dofs[0]]
        self.tau_acc[1] += d.qfrc_applied[self.dofs[1]]
        self.n_acc += 1

    def end_tick(self, d, pt, still_tol=0.03, collect=False):
        """close the tick average; if `collect` and the arms are still, store the torque residual and the load direction at point `pt`
        (pt = (2,3) per-hand points, or (3,))."""
        if self.n_acc == 0:
            return
        tau = [t / self.n_acc for t in self.tau_acc]
        self.tau_acc = [np.zeros(6), np.zeros(6)]
        self.n_acc = 0
        if not collect:
            return
        pt = np.asarray(pt, float)
        for k in (0, 1):
            dofs = self.dofs[k]
            if np.abs(d.qvel[dofs]).max() > still_tol:
                continue
            tm = tau[k] + self.rng.normal(0, TAU_SIGMA, 6)
            tm = _q(tm, TAU_QUANT)
            c = d.qfrc_bias[dofs] - d.qfrc_passive[dofs]
            p = pt if pt.ndim == 1 else pt[k]
            self.mj.mj_jac(self.m, d, self._jp, None, p, self.gb[k])
            j = self._jp[:, dofs].T @ np.array([0.0, 0.0, -float(self.m.opt.gravity[2])])
            self.r[k].append(tm - c)
            self.j[k].append(j)

    def estimate(self):
        """(m_hat per hand (2,), n samples (2,)); nan where there is no sample."""
        out, n = np.full(2, np.nan), np.zeros(2, int)
        for k in (0, 1):
            if not self.r[k]:
                continue
            r = np.mean(self.r[k], 0)
            j = np.mean(self.j[k], 0)
            out[k] = float(j @ r / (j @ j + 1e-12))
            n[k] = len(self.r[k])
        return out, n
