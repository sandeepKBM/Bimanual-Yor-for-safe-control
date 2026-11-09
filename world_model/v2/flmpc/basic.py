"""flmpc_basic -- the BASIC FL-MPC arm tracker, the one the research stack may use (USER 11-09: "FL MPC is a skill to be used", "remove those
stuff from FLMPC just the basic one"; docs/status/decisions.md).

What it is: feedback linearization tau = M(q) a + c(q, qd) (exact double integrator, payload terms M/bias extras allowed) + a joint
acceleration-planning MPC that tracks the plan's joint references (same cost, horizon, gains, numba kernel as mpc_fl.FLMPC).
What it is NOT: it has NO safety rows.  Arm-arm clearance (pairs), table/floor and joint-limit rows, cup tilt / leash, ee speed caps, adaptive
yield, the slack-QP fallback and the torque-safe scale are NOT built.  Safety lives only in the checker.

Kept limits (the only constraints):
  * actuator torque box  |M a0 + c| <= tau_max_vec        # MODEL: MJCF forcerange = URDF effort (nero/config.tau_limit(); 100 Nm, USER 11-05).
    With tau_max_vec=None the scalar _sc.TAU_MAX (100 Nm, MODEL: YOR MJCF) is used.
  * acceleration box |a| <= a_box                         # ASSUMED 800 rad/s^2 (inherited FLMPC default, no source; raised to the user)
Infeasible QP (e.g. gravity bias alone beyond the actuator limit) = what a motor does: solve the unconstrained tracking problem (acceleration
box only) and SATURATE the torque at the actuator limit; those steps are counted (n_sat_steps = infeasible QP steps, n_tau_clipped = steps where
the saturation actually changed the torque).  `last_fallback` is always 0 and no fallback is ever called.

`mpc_fl.FLMPC` is untouched (teleop, Sept cup demo, latent_hj, house keep the full FL-MPC); this is a subclass that overrides step().
"""
from __future__ import annotations

import numpy as np

from . import _sc as sc
from .mpc_fl import FLMPC, _assemble_fl
from .qp.box_qp import solve_box_qp

_RULING = "USER 11-09: FL-MPC is the BASIC tracker only (no safety rows; safety lives in the checker)"


class FLMPCBasic(FLMPC):
    def __init__(self, model, NC=3, L=8, NV=6, DT=None,
                 W_POS_EGO=3000.0, W_VEL_EGO=15.0, W_POS_INT=600.0, W_VEL_INT=8.0,
                 R_ACC=1e-4, W_JERK=1e-3, a_box=800.0,        # a_box ASSUMED (see module docstring)
                 delay_steps=0, tau_max_vec=None):
        super().__init__(model, NC=NC, L=L, NV=NV, DT=DT, W_POS_EGO=W_POS_EGO, W_VEL_EGO=W_VEL_EGO, W_POS_INT=W_POS_INT,
                         W_VEL_INT=W_VEL_INT, R_ACC=R_ACC, W_JERK=W_JERK, a_box=a_box, delay_steps=delay_steps,
                         tau_max_vec=tau_max_vec)
        NV, NP = self.NV, self.NP                       # compile the zero-pair kernel specialization now (not on the first control step)
        qt = np.zeros((NP + 1, NV))
        _assemble_fl(self.Phi_q_all, self.Phi_qd_all, qt, qt, qt, qt, qt, qt, qt, qt, np.zeros(NV), np.zeros(NV),
                     np.zeros((0, 3)), np.zeros((0, 3, NV)), np.zeros((0, 3, NV)), np.zeros(0), self.coll_steps,
                     np.zeros(NV), np.zeros(NV), np.arange(NP + 1) * self.DT, np.eye(NV), np.zeros(NV), np.eye(NV), np.zeros(NV),
                     self.NC, NV, self.n_blk, self.W['pe'], self.W['ve'], self.W['pi'], self.W['vi'], self.W['r'], self.W['j'],
                     self.H_MARGIN, sc.TAU_MAX)
        self.n_sat_steps = 0         # QP infeasible -> unconstrained tracking solve, torque saturated
        self.n_tau_clipped = 0       # steps on which the saturation changed the torque (> 1e-6 Nm)

    def step(self, model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
             q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
             pairs_all=None, M_ego_extra=None, bias_ego_extra=None, M_int_extra=None, bias_int_extra=None, **removed):
        """One control step; same positional signature as FLMPC.step for the kept parts.  `pairs_all` is accepted and IGNORED (no clearance rows).
        Any removed feature (table_rows, tilt / leash / cone, speed cap, q_lim / qd_lim, ego_hold=False ...) given non-None/False raises."""
        for k, v in removed.items():
            if v is not None and not (k == "ego_hold" and v is True):
                raise ValueError(f"FLMPCBasic.step: '{k}' is a removed FL-MPC feature ({_RULING})")
        NV, NC, NP, DT = self.NV, self.NC, self.NP, self.DT
        n_blk = self.n_blk
        ntot = 2 * n_blk

        q0_e = data.qpos[left_qposadr].copy()
        qd0_e = data.qvel[left_dofadr].copy()
        q0_i = data.qpos[right_qposadr].copy()
        qd0_i = data.qvel[right_dofadr].copy()
        M_full = sc.full_mass_matrix(model, data)
        M_e = M_full[np.ix_(left_dofadr, left_dofadr)].copy()
        M_i = M_full[np.ix_(right_dofadr, right_dofadr)].copy()
        c_e = (data.qfrc_bias[left_dofadr] - data.qfrc_passive[left_dofadr]).copy()
        c_i = (data.qfrc_bias[right_dofadr] - data.qfrc_passive[right_dofadr]).copy()
        if M_ego_extra is not None:
            M_e = M_e + M_ego_extra
        if bias_ego_extra is not None:
            c_e = c_e + bias_ego_extra
        if M_int_extra is not None:
            M_i = M_i + M_int_extra
        if bias_int_extra is not None:
            c_i = c_i + bias_int_extra
        self.last_M_e, self.last_c_e, self.last_M_i, self.last_c_i = M_e, c_e, M_i, c_i

        if self.delay_steps > 0:
            q0_e, qd0_e, q0_i, qd0_i, _, _ = self._predict_over_delay(q0_e, qd0_e, q0_i, qd0_i)
        tvec = np.arange(NP + 1) * DT
        qnom_e = q0_e[None, :] + np.outer(tvec, qd0_e)
        qnom_i = q0_i[None, :] + np.outer(tvec, qd0_i)
        qdnom_e = np.tile(qd0_e, (NP + 1, 1))
        qdnom_i = np.tile(qd0_i, (NP + 1, 1))

        # zero clearance pairs: the kernel then builds tracking cost + effort/jerk + the 4*NV torque rows only
        n_arr = np.zeros((0, 3))
        Je_arr = np.zeros((0, 3, NV))
        Ji_arr = np.zeros((0, 3, NV))
        h0_arr = np.zeros(0)
        H, f, a_ineq, b_ineq = _assemble_fl(
            self.Phi_q_all, self.Phi_qd_all, qnom_e, qdnom_e, qnom_i, qdnom_i,
            q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, self.a_ego_prev, self.a_int_prev,
            n_arr, Je_arr, Ji_arr, h0_arr, self.coll_steps, qd0_e, qd0_i, tvec,
            M_e, c_e, M_i, c_i, NC, NV, n_blk,
            self.W['pe'], self.W['ve'], self.W['pi'], self.W['vi'], self.W['r'], self.W['j'],
            self.H_MARGIN, sc.TAU_MAX)
        tau_lim = sc.TAU_MAX * np.ones(NV) if self.tau_max_vec is None else self.tau_max_vec   # MODEL: actuator limit
        for j in range(NV):          # torque rows (first 4*NV rows, no collision rows before them); NO |c_j| relaxation: a motor saturates
            b_ineq[2 * j] = tau_lim[j] - c_e[j]
            b_ineq[2 * j + 1] = tau_lim[j] + c_e[j]
            b_ineq[2 * NV + 2 * j] = tau_lim[j] - c_i[j]
            b_ineq[2 * NV + 2 * j + 1] = tau_lim[j] + c_i[j]
        lower = -self.a_box * np.ones(ntot)       # ASSUMED a_box
        upper = self.a_box * np.ones(ntot)
        x, _, feasible = sc.solve_constrained_box_qp(H, f, lower, upper, np.ascontiguousarray(a_ineq), np.ascontiguousarray(b_ineq))
        self.last_feasible = bool(feasible)
        self.last_fallback = 0
        if not feasible:
            x = solve_box_qp(H, f, lower, upper)          # unconstrained tracking (acceleration box only); torque saturated below
            self.n_sat_steps += 1
        a_e0 = x[0:NV].copy()
        a_i0 = x[n_blk:n_blk + NV].copy()
        self.a_ego_prev = a_e0.copy()
        self.a_int_prev = a_i0.copy()
        if self.delay_steps > 0:
            self._a_queue.append((a_e0.copy(), a_i0.copy()))
            if len(self._a_queue) > self.delay_steps:
                self._a_queue.pop(0)
        tau_e = M_e @ a_e0 + c_e
        tau_i = M_i @ a_i0 + c_i
        te_c, ti_c = np.clip(tau_e, -tau_lim, tau_lim), np.clip(tau_i, -tau_lim, tau_lim)      # the motor's own saturation
        if max(float(np.max(np.abs(te_c - tau_e))), float(np.max(np.abs(ti_c - tau_i)))) > 1e-6:
            self.n_tau_clipped += 1
        return te_c, ti_c, bool(feasible)
