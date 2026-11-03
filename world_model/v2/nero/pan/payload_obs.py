"""pan/payload_obs.py -- payload-aware lift: a joint-space load observer feeding FL-MPC's payload hooks.  SIMULATION ONLY.

USER 11-01 ("build the payload-aware lift"): with a hidden load the one-claw lift sagged (the controller knew only the empty pan),
e.g. 1.4-2.9 cm of a 5 cm lift at 40 N grip (results/contact_model_2026-11-01/claw40/RESULT.txt).

Observer (standard disturbance observer, robot-side signals only):
    tau_ext_raw = tau_cmd(previous step) - M(q) qdd_fd - bias(q, qd) - passive(q, qd)
    tau_ext     = first-order low-pass of tau_ext_raw, time constant TAU_F
  qdd_fd = finite difference of joint velocities (what the robot has), M / bias / passive = the robot's own arm model (MuJoCo's,
  i.e. a PERFECT arm model: ASSUMED).  tau_cmd = the torque the controller commanded (motor current -> torque assumed exact: ASSUMED).
  tau_ext therefore contains the load carried by the claw, contact with the stove/table through the pan, AND the joint frictionloss
  stiction (0.3 Nm, a MuJoCo constraint the robot cannot separate) -- that last part is an honest error source, left in.
Feed-forward: FL-MPC gets bias_extra = -tau_ext (so tau = M a + c - tau_ext: the load is cancelled when the estimate is right) and
M_extra = the empty-pan inertia (nominal, as before).  mpc_fl.py is NOT edited; only its existing M/bias_extra hooks are used.
The load estimate in task space (for logging / the certificate later): wrench at the gripper base = pinv(J^T) tau_ext; m_hat = -F_z/g.

Bounds (labels per AGENTS.md):
  TAU_F = 0.10 s          ASSUMED  (filter time constant; slower = smoother, more sag during fast moves)
  SIGMA_TAU = 0.0 Nm      ASSUMED  (joint-torque sensing noise; pass sigma_tau to test)
  arm model exact         ASSUMED  (no inertial-parameter error)
"""
from __future__ import annotations

import mujoco
import numpy as np

TAU_F = 0.10        # s, ASSUMED
SIGMA_TAU = 0.0     # Nm, ASSUMED


class LoadObserver:
    """per-arm external joint torque estimate.  Call update(m, d, tau_cmd_prev) once per control step AFTER mj_forward."""

    def __init__(self, m, dofadr, dt, tau_f=TAU_F, sigma_tau=SIGMA_TAU, rng=None):
        self.dof = np.asarray(dofadr, int)
        self.dt, self.alpha = float(dt), float(dt / (tau_f + dt))
        self.sigma, self.rng = float(sigma_tau), rng if rng is not None else np.random.default_rng(0)
        self.qd_prev = None
        self.est = np.zeros(len(self.dof))
        self._M = np.zeros((m.nv, m.nv))

    def update(self, m, d, tau_cmd_prev):
        qd = d.qvel[self.dof].copy()
        if self.qd_prev is None:
            self.qd_prev = qd
            return self.est
        qdd = (qd - self.qd_prev) / self.dt
        self.qd_prev = qd
        mujoco.mj_fullM(m, self._M, d.qM)
        Ma = self._M[np.ix_(self.dof, self.dof)] @ qdd
        tau = np.asarray(tau_cmd_prev, float) + d.qfrc_actuator[self.dof]      # arm servos are zeroed; kept for safety
        if self.sigma:
            tau = tau + self.rng.normal(0.0, self.sigma, tau.shape)
        # MuJoCo: M qdd + bias = tau_applied + passive + tau_constraint  ->  tau_constraint = M qdd + bias - tau_applied - passive
        con = Ma + d.qfrc_bias[self.dof] - tau - d.qfrc_passive[self.dof]
        self.est += self.alpha * (con - self.est)                             # est = constraint (load) torque on the arm
        return self.est

    def wrench(self, m, d, body_id):
        """least-squares wrench [F (3), M (3)] at the body origin, world frame, from the current estimate."""
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, body_id)
        J = np.vstack([jp, jr])[:, self.dof]
        return np.linalg.lstsq(J.T, self.est, rcond=None)[0]

    def mass_hat(self, m, d, body_id):
        return float(self.wrench(m, d, body_id)[2] / m.opt.gravity[2])      # F_z / g_z (g_z < 0, load pulls down)
