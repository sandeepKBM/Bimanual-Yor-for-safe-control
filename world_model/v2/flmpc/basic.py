"""flmpc_basic -- the BASIC FL-MPC arm tracker, the one the research stack may use (USER 11-09: "FL MPC is a skill to be used", "remove those
stuff from FLMPC just the basic one"; docs/status/decisions.md).

What it is: feedback linearization tau = M(q) a + c(q, qd) (exact double integrator, payload terms M/bias extras allowed) + a joint
acceleration-planning MPC that tracks the plan's joint references (same cost, horizon, gains, numba kernel as mpc_fl.FLMPC).
What it is NOT: it has NO safety rows.  Arm-arm clearance (pairs), table/floor and joint-limit rows, cup tilt / leash, ee speed caps, adaptive
yield, the slack-QP fallback and the torque-safe scale are NOT built.  Safety lives only in the checker.

Kept limits (the only constraints):
  * actuator torque box  |M a0 + c| <= tau_max_vec        # MODEL: MJCF forcerange = URDF effort (nero/config.tau_limit(); 100 Nm, USER 11-05).
    With tau_max_vec=None the scalar _sc.TAU_MAX (100 Nm, MODEL: YOR MJCF) is used.
  * joint speed |qd_j(t)| <= qd_max_j over the horizon    # OFFICIAL config.MANUAL["joint_speed_deg_s"] (NERO manual, deg/s -> rad/s); the
    CALLER passes `qd_max` (hold_test.Ctl reads it from config; None = no speed rows).  Rows sit at step 1 and at the end of each control
    block (acceleration is piecewise constant per block, so velocity is piecewise linear and these are its extremes).
  * NO acceleration limit: NERO publishes none (config.MANUAL); acceleration is bounded by the torque box.  The QP solver needs finite variable
    bounds, so |a| <= A_SOLVER_BOUND (a numerical bound, not a limit); `n_a_bound_steps` counts steps on which it bound (expected 0).
Infeasible QP (torque or speed rows cannot be met, e.g. gravity bias alone beyond the actuator limit) = what a motor does: solve the unconstrained
tracking problem (solver bound only) and SATURATE the torque at the actuator limit; those steps are counted (n_sat_steps = infeasible QP steps,
n_tau_clipped = steps where the saturation actually changed the torque).  `n_speed_binding_steps` = feasible steps on which a speed row is active.
`last_fallback` is always 0 and no fallback is ever called.

`mpc_fl.FLMPC` is untouched (teleop, Sept cup demo, latent_hj, house keep the full FL-MPC); this is a subclass that overrides step().
"""
from __future__ import annotations

import numpy as np

from . import _sc as sc
from .mpc_fl import FLMPC, _assemble_fl
from .qp.box_qp import solve_box_qp

RESID_TOL = 1e-3          # Nm / rad/s: a returned x violating a torque or speed row by more than this counts as infeasible
A_SOLVER_BOUND = 1e5      # rad/s^2: solver bound, not a limit (box QP needs finite bounds); counted when it binds
_RULING = "USER 11-09: FL-MPC is the BASIC tracker only (no safety rows; safety lives in the checker)"


def _osqp_solve(H, f, lower, upper, A, b):
    """min 0.5 x'Hx + f'x  s.t.  A x <= b, lower <= x <= upper  (OSQP).  None when infeasible / no solution."""
    import osqp
    from scipy import sparse
    n = len(f)
    Hs = 0.5 * (H + H.T)
    prob = osqp.OSQP()
    prob.setup(P=sparse.csc_matrix(np.triu(Hs)), q=f, A=sparse.csc_matrix(np.vstack([A, np.eye(n)])),
               l=np.concatenate([np.full(len(b), -np.inf), lower]), u=np.concatenate([b, upper]),
               eps_abs=1e-8, eps_rel=1e-8, max_iter=50000, polishing=True, verbose=False)
    res = prob.solve()
    if res.info.status_val not in (1, 2) or res.x is None or not np.all(np.isfinite(res.x)):      # solved / solved inaccurate
        return None
    return np.asarray(res.x, float)


class FLMPCBasic(FLMPC):
    def __init__(self, model, NC=3, L=8, NV=6, DT=None,
                 W_POS_EGO=3000.0, W_VEL_EGO=15.0, W_POS_INT=600.0, W_VEL_INT=8.0,
                 R_ACC=1e-4, W_JERK=1e-3,
                 delay_steps=0, tau_max_vec=None, qd_max=None):       # qd_max: rad/s, scalar / (NV,) / (2, NV); OFFICIAL config.MANUAL, passed by the caller
        super().__init__(model, NC=NC, L=L, NV=NV, DT=DT, W_POS_EGO=W_POS_EGO, W_VEL_EGO=W_VEL_EGO, W_POS_INT=W_POS_INT,
                         W_VEL_INT=W_VEL_INT, R_ACC=R_ACC, W_JERK=W_JERK, a_box=A_SOLVER_BOUND, delay_steps=delay_steps,
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
        self.n_a_bound_steps = 0     # steps on which the numerical solver bound on |a| was active (expected 0)
        self.n_speed_binding_steps = 0   # feasible steps on which a speed row is active (within 1e-3 rad/s)
        self.n_speed_resolves = 0        # steps on which the speed rows were violated by the torque-only solution (re-solved in OSQP)
        self.qd_max = None if qd_max is None else np.broadcast_to(np.asarray(qd_max, float), (2, NV)).copy()
        self.speed_steps = sorted(set([1] + list(range(self.L, self.NP + 1, self.L))))
        self._sp_A = None            # (n_speed, 2*n_blk) rows, built once; right-hand side = +-qd_max -+ qd0 per step
        if self.qd_max is not None:
            ntot = 2 * self.n_blk
            rows = []
            for c0 in (0, self.n_blk):
                for t in self.speed_steps:
                    Ad = self.Phi_qd_all[t]
                    for j in range(NV):
                        r = np.zeros(ntot); r[c0:c0 + self.n_blk] = Ad[j]
                        rows.append(r); rows.append(-r)
            self._sp_A = np.array(rows)

    def step(self, model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
             q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
             pairs_all=None, M_ego_extra=None, bias_ego_extra=None, M_int_extra=None, bias_int_extra=None, **removed):
        """One control step; same positional signature as FLMPC.step for the kept parts.  `pairs_all` is accepted and IGNORED (no clearance rows).
        Any removed feature (table_rows, tilt / leash / cone, ee speed cap, q_lim / qd_lim (the joint speed limit is the constructor's qd_max), ego_hold=False ...) given non-None/False raises."""
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
        n_speed = 0
        if self.qd_max is not None:               # OFFICIAL config.MANUAL joint speed: Phi_qd_all[t] @ a_block + qd0 within +-qd_max, both arms
            qd0 = (qd0_e, qd0_i)
            rhs = []
            for k in range(2):
                for t in self.speed_steps:
                    for j in range(NV):
                        rhs.append(self.qd_max[k, j] - qd0[k][j]); rhs.append(self.qd_max[k, j] + qd0[k][j])
            sp_a, sp_b = self._sp_A, np.array(rhs)
            n_speed = len(rhs)
        lower = -self.a_box * np.ones(ntot)       # solver bound, not a limit
        upper = self.a_box * np.ones(ntot)
        # 1) torque box only, with the same dual-ascent solver as FLMPC (identical to before when no speed row is active)
        x, _, feasible = sc.solve_constrained_box_qp(H, f, lower, upper, np.ascontiguousarray(a_ineq), np.ascontiguousarray(b_ineq))
        # the solver's flag is not trusted alone (memory: QP 'feasible' != constraints met): check the residual of every row (Nm / rad/s)
        if feasible and float(np.max(a_ineq @ x - b_ineq)) > RESID_TOL:
            feasible = False
        if n_speed and feasible and float(np.max(sp_a @ x - sp_b)) > 1e-6:
            # 2) a speed row is violated: re-solve with torque + speed rows in OSQP (the dual-ascent solver does not converge on the many
            #    small-norm speed rows: it left torque rows violated by Nm in a test).  Same objective, bounds |a| <= solver bound.
            x2 = _osqp_solve(H, f, lower, upper, np.vstack([a_ineq, sp_a]), np.concatenate([b_ineq, sp_b]))
            feasible = x2 is not None and float(np.max(np.vstack([a_ineq, sp_a]) @ x2 - np.concatenate([b_ineq, sp_b]))) <= RESID_TOL
            if feasible:
                x = x2
            self.n_speed_resolves += 1
        self.last_feasible = bool(feasible)
        self.last_fallback = 0
        if not feasible:
            x = solve_box_qp(H, f, lower, upper)          # unconstrained tracking (acceleration box only); torque saturated below
            self.n_sat_steps += 1
        elif n_speed and float(np.min(sp_b - sp_a @ x)) < 1e-3:
            self.n_speed_binding_steps += 1
        if float(np.max(np.abs(x))) >= 0.999 * self.a_box:
            self.n_a_bound_steps += 1
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
