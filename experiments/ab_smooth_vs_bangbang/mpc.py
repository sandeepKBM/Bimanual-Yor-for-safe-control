"""v5: a JOINT, grasp-aware, short-horizon cooperative MPC over BOTH arms'
torques -- see RESULTS_v5.md for the full write-up and motivation (v4's
Sec.6 "principled next step": remove the hand-tuned decentralized ego-
priority margin by solving a single combined psi2>=0-style constraint over
BOTH arms' torques at once, so priority EMERGES from the ego's grasp
constraint pinning it, rather than being hand-set via an asymmetric margin).

FORMULATION (single QP per control step, receding horizon):
  - Decision variable x = [tau_ego_block(0..NC-1); tau_int_block(0..NC-1)]
    in R^{2*NC*6} -- PIECEWISE-CONSTANT torque over NC control blocks of
    L_PER_BLOCK physics steps each (a standard MPC "control horizon <<
    prediction horizon" trick to keep the decision dimension small; NP =
    NC*L_PER_BLOCK physics steps of PREDICTION, i.e. NP*DT seconds of
    look-ahead).
  - Dynamics: LINEARIZED/frozen -- mass matrix M and bias (Coriolis+gravity,
    ego's WITH the grasped box's mass/inertia reflected in via
    sim_common.payload_reflected_terms, exactly as the reactive C2/v4
    controllers already do) are evaluated ONCE at the current state and
    held constant across the horizon (a "zeroth-order-hold Jacobian"
    approximation, standard for a horizon this short: NP*DT <= 60ms here).
    Under this freeze, qacc(t) = Minv@(tau_block(t)-bias) is affine in the
    decision variable, so q(t)/qdot(t) (double-integrator recursion) are
    each EXACTLY affine in x (not a finite-difference approximation --
    _sensitivity below derives the affine map analytically via
    superposition, since the recursion truly is linear once M/bias are
    frozen).
  - Objective: quadratic tracking of (a) the ego's grasp-hold joint pose
    q0_left + zero joint velocity at every predicted step, (b) the
    intruder's own trackable nominal goal trajectory (run_ab4's
    intruder_ctrl_nominal, oscillation-free -- same de-tuning rationale as
    v4, this is the intruder's REFERENCE not a hack), (c) an inter-block
    jerk penalty (block-to-block torque difference, continuous with the
    PREVIOUS step's applied block-0 torque) + a small effort regularizer
    -- reuses sim_common.build_weighted_least_squares_qp for every one of
    these terms (same helper v1-v4 already use for their single-stage QPs).
  - Constraints (all LINEAR in x given the frozen-Jacobian affine map, so
    solvable by the SAME general-purpose box+linear-inequality QP already
    in this repo, sim_common.solve_constrained_box_qp -- no new solver
    written):
      * per-arm torque box bounds (TAU_MAX), enforced as the QP's box
        bounds directly, not inequality rows.
      * ARM-ARM COLLISION: for the K_PAIRS nearest (smallest current h)
        sphere pairs (from sim_common.all_pair_barriers), require the
        FIRST-ORDER-linearized barrier h_lin(t) = h0 + n.(Je@dq_ego(t) -
        Ji@dq_int(t)) >= 0 at EVERY predicted step t=1..NP -- using n/Je/Ji
        frozen at t=0 (same short-horizon-freeze approximation as the
        dynamics). Crucially: h_lin(t) >= 0 uses EXACTLY the same D_MIN
        that both v4's ego and intruder QPs already shared symmetrically
        (sim_common.D_MIN) -- NO extra asymmetric margin is added to
        either side. If priority emerges, it must come from the grasp
        constraint below pinning the ego, not from this term.
      * EGO GRASP (hard-ish): the ego EE's (gripper_base) predicted
        Cartesian translation AND small-angle rotation deviation from its
        pose at the CURRENT step, evaluated at two predicted steps (mid-
        horizon + terminal) via the frozen EE Jacobian, must stay within
        EE_SLACK_POS / EE_SLACK_ROT -- a genuine linear inequality (not a
        soft cost), so grasp retention is a hard constraint the QP must
        satisfy (or report infeasible), matching the prompt's "ego GRASP
        constraint ... as a hard-ish constraint".
  - Solved via sim_common.solve_constrained_box_qp (dual-ascent box-QP,
    already in this repo, used by every prior condition's reactive QP) --
    NOT a new solver. Only the FIRST control block's torque is applied each
    physics step (receding horizon: replanned every step from the true
    MuJoCo state, not open-loop over the horizon).

This is a genuine (if short-horizon, linearized) joint/centralized
optimization -- BOTH arms' torques are decision variables of ONE QP with a
SHARED collision-avoidance constraint set, unlike v4's decentralized
construction (two separate QPs, each treating the other arm's currently-
measured acceleration as a fixed disturbance).
"""
from __future__ import annotations

import numpy as np
import sim_common as sc

# ---------------------------------------------------------------------------
# Horizon / QP sizing (measured for real-time plausibility -- see
# RESULTS_v5.md "solve time" section; these constants were reduced from an
# initial NP=9/K_PAIRS=6 pass after a synthetic scaling probe showed the
# reused dual-ascent QP solver costs ~17-43ms at m~66 rows even with its
# numba fast path -- NOT a rewrite of the solver, just a size budget chosen
# to keep this experiment's wall-clock tractable while still being a
# genuine multi-row linear-inequality QP).
# ---------------------------------------------------------------------------
NC = 4                 # control blocks (piecewise-constant torque per block)
L_PER_BLOCK = 2         # physics steps held constant per block
NP = NC * L_PER_BLOCK   # prediction horizon steps (=8 -> 40ms @ DT=0.005)
K_PAIRS = 5             # nearest collision sphere-pairs enforced at every predicted step
# EE_SLACK_POS/ROT: ego EE translational/rotational hard-ish grasp cap (2
# stages x 3 axes x 2 signs). NOT a collision-priority margin (that's the
# thing this experiment deliberately avoids hand-tuning) -- this is a grasp-
# tolerance design parameter, analogous to any position-controller's
# acceptable-tracking-error band. IMPORTANT, HONEST FINDING (see
# RESULTS_v5.md): a tight cap (0.015m/0.03rad, the first value tried, "keep
# the grasp essentially rigid") makes the joint QP infeasible 68-89% of the
# time once the intruder is close (the ego's grasp is pinned so tightly
# that avoidance has nowhere to go but the intruder alone, which cannot
# decelerate enough within this horizon/torque budget to satisfy the
# collision constraint every predicted step) -- and, like v4's rejected
# high-alpha attempt, this is NOT a graceful degradation: once infeasible,
# the dual-ascent solver's saturated multipliers drive both arms' commanded
# torque to bang between box-bound extremes, producing catastrophic jerk
# and a full box flip (worse than doing nothing). These looser values were
# found (by a 3-point manual probe, not a fine sweep) to be the boundary
# where infeasibility drops back near 0% -- SLACK-TUNING WAS REQUIRED to
# avoid the infeasibility cascade, exactly the kind of caveat the prompt
# asked to report honestly.
EE_SLACK_POS = 0.05     # [m]
EE_SLACK_ROT = 0.2      # [rad, small-angle]
DUAL_SWEEPS = 2
DUAL_ROOT_ITERS = 5

W_POS_EGO = 3000.0
W_VEL_EGO = 15.0
W_POS_INT = 600.0
W_VEL_INT = 8.0
W_JERK = 0.02
REG_EFFORT = 5.0e-4

NV = 6  # dof per arm


def _propagate(q0, qd0, Minv, bias, tau_blocks):
    """tau_blocks: (NC, NV). Double-integrator recursion with FROZEN Minv/bias
    (same M/bias for every predicted step -- the short-horizon-freeze
    approximation this whole module relies on). Returns q_traj, qd_traj,
    each (NP+1, NV) including t=0 (the given initial condition)."""
    q = np.zeros((NP + 1, NV))
    qd = np.zeros((NP + 1, NV))
    q[0] = q0
    qd[0] = qd0
    for t in range(NP):
        b = t // L_PER_BLOCK
        qacc = Minv @ (tau_blocks[b] - bias)
        qd[t + 1] = qd[t] + qacc * sc.DT
        q[t + 1] = q[t] + qd[t] * sc.DT + 0.5 * qacc * sc.DT ** 2
    return q, qd


def _sensitivity(q0, qd0, Minv, bias):
    """Analytic affine map (via superposition -- exact, not finite-difference,
    since _propagate is EXACTLY linear in tau_blocks once Minv/bias are
    frozen): q(t) = q_free[t] + Phi_q[t] @ tau_stack, qd(t) = qd_free[t] +
    Phi_qd[t] @ tau_stack, where tau_stack in R^{NC*NV} is the flattened
    (block, joint) decision vector. q_free/qd_free = the trajectory with
    tau_stack=0 (isolates the bias/gravity-only response); each Phi column
    = the trajectory with q0=qd0=bias=0 and tau_stack=one unit basis vector
    (isolates exactly the linear part, since with zero initial state and
    zero bias the recursion is homogeneous-linear in tau)."""
    zero_blocks = np.zeros((NC, NV))
    q_free, qd_free = _propagate(q0, qd0, Minv, bias, zero_blocks)
    Phi_q = np.zeros((NP + 1, NV, NC * NV))
    Phi_qd = np.zeros((NP + 1, NV, NC * NV))
    for b in range(NC):
        for j in range(NV):
            unit = np.zeros((NC, NV))
            unit[b, j] = 1.0
            q_u, qd_u = _propagate(np.zeros(NV), np.zeros(NV), Minv, np.zeros(NV), unit)
            col = b * NV + j
            Phi_q[:, :, col] = q_u
            Phi_qd[:, :, col] = qd_u
    return q_free, qd_free, Phi_q, Phi_qd


class JointMPC:
    """Joint cooperative MPC controller: one QP per physics step, both arms'
    torques as decision variables. See module docstring for the full
    formulation."""

    def __init__(self, model, ctrl, q0_left):
        self.model = model
        self.ctrl = ctrl
        self.q0_left = q0_left.copy()
        self.n_e = NC * NV
        self.n_i = NC * NV
        self.ntot = self.n_e + self.n_i
        self.tau_ego_prev = np.zeros(NV)
        self.tau_int_prev = np.zeros(NV)
        self.last_solve_time = 0.0
        self.last_m_rows = 0

    def _pad_e(self, mat):
        return np.hstack([mat, np.zeros((mat.shape[0], self.n_i))])

    def _pad_i(self, mat):
        return np.hstack([np.zeros((mat.shape[0], self.n_e)), mat])

    def _block_sel_ego(self, b):
        m = np.zeros((NV, self.ntot))
        m[:, b * NV:(b + 1) * NV] = np.eye(NV)
        return m

    def _block_sel_int(self, b):
        m = np.zeros((NV, self.ntot))
        m[:, self.n_e + b * NV:self.n_e + (b + 1) * NV] = np.eye(NV)
        return m

    def step(self, model, data, q_int_des_traj, qd_int_des_traj, pairs_all):
        """q_int_des_traj/qd_int_des_traj: (NP+1, NV) arrays -- the intruder's
        OWN scripted nominal reference at t, t+DT, ..., t+NP*DT (known in
        advance -- it is the intruder's own goal schedule, not a sensed
        quantity, same epistemic status as v4's B/C intruder QP already
        assumed by using its own reference).
        pairs_all: output of sim_common.all_pair_barriers(...) for the
        CURRENT step (already computed by the caller for logging -- reused
        here, not recomputed, to avoid a second ~17ms all_pair_barriers
        call).

        Returns (tau_ego(6,), tau_int(6,), feasible(bool), solve_time_s).
        """
        import time
        t_start = time.time()
        ctrl = self.ctrl

        q_ego0 = data.qpos[ctrl.left_qposadr].copy()
        qd_ego0 = data.qvel[ctrl.left_dofadr].copy()
        q_int0 = data.qpos[ctrl.right_qposadr].copy()
        qd_int0 = data.qvel[ctrl.right_dofadr].copy()

        M_full = sc.full_mass_matrix(model, data)
        M_ll = M_full[np.ix_(ctrl.left_dofadr, ctrl.left_dofadr)]
        M_rr = M_full[np.ix_(ctrl.right_dofadr, ctrl.right_dofadr)]
        bias_l = data.qfrc_bias[ctrl.left_dofadr].copy()
        bias_r = data.qfrc_bias[ctrl.right_dofadr].copy()
        M_box, bias_box = ctrl.payload_terms(data)  # grasped-box dynamics reflected into the ego only
        M_ll = M_ll + M_box
        bias_l = bias_l + bias_box
        Minv_ll = np.linalg.inv(M_ll)
        Minv_rr = np.linalg.inv(M_rr)

        qf_e, qdf_e, Phq_e, Phqd_e = _sensitivity(q_ego0, qd_ego0, Minv_ll, bias_l)
        qf_i, qdf_i, Phq_i, Phqd_i = _sensitivity(q_int0, qd_int0, Minv_rr, bias_r)

        # ---- objective terms ----
        terms = []
        for t in range(1, NP + 1):
            terms.append((self._pad_e(Phq_e[t]), self.q0_left - qf_e[t], W_POS_EGO))
            terms.append((self._pad_e(Phqd_e[t]), -qdf_e[t], W_VEL_EGO))
            terms.append((self._pad_i(Phq_i[t]), q_int_des_traj[t] - qf_i[t], W_POS_INT))
            terms.append((self._pad_i(Phqd_i[t]), qd_int_des_traj[t] - qdf_i[t], W_VEL_INT))
        # jerk / continuity (block-to-block + continuity with previous step's applied block-0)
        terms.append((self._block_sel_ego(0), self.tau_ego_prev, W_JERK))
        terms.append((self._block_sel_int(0), self.tau_int_prev, W_JERK))
        for b in range(1, NC):
            terms.append((self._block_sel_ego(b) - self._block_sel_ego(b - 1), np.zeros(NV), W_JERK))
            terms.append((self._block_sel_int(b) - self._block_sel_int(b - 1), np.zeros(NV), W_JERK))

        H, f = sc.build_weighted_least_squares_qp(terms, reg=REG_EFFORT, n=self.ntot)

        # ---- collision constraints: K nearest pairs, every predicted step ----
        pairs_sorted = sorted(pairs_all, key=lambda p: p["h"])[:K_PAIRS]
        a_rows, b_rows = [], []
        for p in pairs_sorted:
            n_p, Je_p, Ji_p, h0 = p["n"], p["Je"], p["Ji"], p["h"]
            c_e = n_p @ Je_p       # (NV,)
            c_i = n_p @ Ji_p       # (NV,)
            for t in range(1, NP + 1):
                row_e = -c_e @ Phq_e[t]
                row_i = c_i @ Phq_i[t]
                a_row = np.concatenate([row_e, row_i])
                const = h0 + c_e @ (qf_e[t] - q_ego0) - c_i @ (qf_i[t] - q_int0)
                a_rows.append(a_row)
                b_rows.append(const)

        # ---- ego grasp (hard-ish): EE translation + small-angle rotation ----
        Jp_full, Jr_full = sc.body_jac_full(model, data, ctrl.ee_body_id)
        Jp_ee = Jp_full[:, ctrl.left_dofadr]
        Jr_ee = Jr_full[:, ctrl.left_dofadr]
        for t in (NP // 2, NP):
            for J_ee, slack in ((Jp_ee, EE_SLACK_POS), (Jr_ee, EE_SLACK_ROT)):
                d_free = J_ee @ (qf_e[t] - self.q0_left)   # (3,)
                Phi_ee = J_ee @ Phq_e[t]                    # (3, n_e)
                row_full = self._pad_e(Phi_ee)               # (3, ntot)
                for i in range(3):
                    a_rows.append(row_full[i])
                    b_rows.append(slack - d_free[i])
                    a_rows.append(-row_full[i])
                    b_rows.append(slack + d_free[i])

        a_ineq = np.stack(a_rows, axis=0)
        b_ineq = np.array(b_rows)
        self.last_m_rows = a_ineq.shape[0]

        lower = -sc.TAU_MAX * np.ones(self.ntot)
        upper = sc.TAU_MAX * np.ones(self.ntot)
        x, _, feasible = sc.solve_constrained_box_qp(
            H, f, lower, upper, a_ineq, b_ineq,
            dual_sweeps=DUAL_SWEEPS, dual_root_iters=DUAL_ROOT_ITERS)

        tau_ego = x[0:NV].copy()
        tau_int = x[self.n_e:self.n_e + NV].copy()
        self.tau_ego_prev = tau_ego.copy()
        self.tau_int_prev = tau_int.copy()

        self.last_solve_time = time.time() - t_start
        return tau_ego, tau_int, bool(feasible), self.last_solve_time
