"""v6: numba-accelerated joint cooperative MPC + explicit priority cost.

See RESULTS_v6.md for the full write-up. This module is a from-scratch
REIMPLEMENTATION of v5's mpc.py (kept untouched at mpc.py for reference/
rollback), motivated by v5's own profiling finding: `constrained_box_qp`
(real_Cartpole) was ALREADY numba, but the Python ASSEMBLY around it (the
linearized-dynamics sensitivity recursion + per-stage constraint-row
building, all done with per-call numpy dispatch and a generic padded-matrix
"terms" builder) was the dominant cost (measured ~7.0ms/step of v5's total
~15-17ms, see profile_mpc.py / RESULTS_v6.md Sec.1).

WHAT CHANGED vs mpc.py (v5):
  1. The sensitivity recursion (_sensitivity_core) and ALL constraint/
     objective assembly (_assemble) are now single `@njit` kernels operating
     on raw arrays -- no per-term np.hstack/padded-matrix allocation (v5's
     `_pad_e`/`_pad_i` + `sc.build_weighted_least_squares_qp` generic
     "list-of-terms" builder is bypassed entirely; H/f are accumulated
     directly into their ego/int block-diagonal sub-blocks, which is exact
     since the OBJECTIVE has no ego-intruder cross terms -- verified by
     inspection of v5's term list, none mix `_pad_e`/`_pad_i`).
  2. MuJoCo calls (mj_fullM, mj_jac, payload_terms) stay OUTSIDE njit,
     exactly as before -- only their OUTPUT arrays (M, bias, Jacobians) are
     passed into the compiled kernels.
  3. Horizon extended using the resulting speed budget (see Sec.3 of
     RESULTS_v6.md for the measured tradeoff): NC/L_PER_BLOCK below.
  4. NEW: an explicit "grasp-holder right-of-way" priority cost (Sec.4) --
     v5 found priority does NOT emerge for free from the grasp HARD
     constraint alone (loose slack -> ego absorbed avoidance; tight slack ->
     infeasible). This adds:
       (a) asymmetric effort regularization: REG_EFFORT_EGO >> REG_EFFORT_INT
           (moving the ego's torque away from 0 is expensive; moving the
           intruder's is cheap),
       (b) asymmetric jerk weight: W_JERK_EGO > W_JERK_INT (same idea, in
           the block-to-block-torque-change term),
       (c) an explicit EGO EE-TRANSLATION HOLD soft cost (W_EE_HOLD) at
           EVERY predicted step (not just the 2 discrete grasp-cap stages)
           -- this is new relative to v5, which only constrained the EE
           pose at 2 stages; here it is additionally COSTED continuously,
           so the QP is penalized for planning ANY ego EE drift, not just
           for exceeding the hard cap. This is the "principled ego-grasp-
           preservation weight" alternative the task asked for, on top of
           (not instead of) v5's existing hard EE_SLACK_POS/ROT constraint
           (kept, same values, so grasp retention is still hard-capped).
     None of (a)/(b)/(c) touch the shared, symmetric collision barrier
     (still D_MIN, no asymmetric margin there) -- priority is expressed
     ONLY via how expensive/costly each arm's OWN motion is, not by
     weakening the intruder's safety margin. This is the "grasp-holder
     right-of-way" the task asked to make explicit and honest.
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit

import sim_common as sc

# ---------------------------------------------------------------------------
# Horizon sizing -- extended using the numba speedup (see RESULTS_v6.md Sec.3
# for the measured solve-time-vs-horizon tradeoff that picked these values).
# ---------------------------------------------------------------------------
# NC (control blocks) is the DOMINANT driver of constrained_box_qp's solve
# cost (empirically: n=2*NC*NV decision dims -- see profile_mpc6.py sweep in
# RESULTS_v6.md Sec.3 -- NC=4->n=48 costs ~8ms/step even at NP=40 (200ms
# horizon); NC=6->n=72 costs ~25ms/step at NP=36; NC=8->n=96 costs ~50ms/step
# even with ONLY 64 constraint rows). So the horizon is extended by making
# blocks LONGER (bigger L_PER_BLOCK), NOT by adding more blocks -- this
# reaches further in TIME (and gives the tracking objective a longer horizon
# to anticipate over) at v5's original NC=4/n=48 decision-dimension cost.
NC = 4                   # control blocks (SAME as v5 -- keep n small)
L_PER_BLOCK = 8           # physics steps held constant per block (v5: 2)
NP = NC * L_PER_BLOCK    # = 32 -> 160ms look-ahead @ DT=0.005 (v5: 8 -> 40ms)
K_PAIRS = 5              # nearest collision sphere-pairs (unchanged from v5)
# Collision-CONSTRAINT rows are only checked once per control block (at each
# block's END, t=L,2L,...,NP), NOT at every one of the NP_ physics-step
# predictions -- see _assemble's docstring comment: this keeps the
# constraint row count m (and therefore constrained_box_qp's solve time,
# which scales with m) from growing with the full physics-step resolution
# of the horizon, while the OBJECTIVE/tracking cost still uses every step
# (so the anticipation benefit of the longer horizon is not lost, only the
# collision-row COUNT is capped).


def _coll_steps():
    return np.arange(L_PER_BLOCK, NP + 1, L_PER_BLOCK, dtype=np.int64)

EE_SLACK_POS = 0.05      # [m] hard-ish grasp cap (unchanged from v5's tuned value)
EE_SLACK_ROT = 0.2       # [rad, small-angle]
DUAL_SWEEPS = 2
DUAL_ROOT_ITERS = 5

# ---- objective weights ----
W_POS_EGO = 3000.0
W_VEL_EGO = 15.0
W_POS_INT = 600.0
W_VEL_INT = 8.0

# ---- EXPLICIT PRIORITY ("grasp-holder right-of-way") ----
# Asymmetric effort/jerk cost: cheap for the intruder to move, expensive for
# the ego -- see module docstring point 4(a)/(b).
REG_EFFORT_EGO = 5.0e-3
REG_EFFORT_INT = 5.0e-5
W_JERK_EGO = 0.08
W_JERK_INT = 0.01
# Explicit soft EE-hold cost -- point 4(c). Units: cost is W_EE_HOLD per
# (meter of predicted EE drift)^2, summed over all NP predicted steps (so
# larger than a single-stage weight would need to be, since it acts at
# every step, not just 2).
W_EE_HOLD = 4.0e4

NV = 6


# =============================================================================
# Numba kernels. All operate on plain float64 arrays; no MuJoCo calls inside.
# =============================================================================

@njit(cache=True)
def _sensitivity_core(q0, qd0, Minv, bias, NC_, NV_, NP_, L_, DT):
    """Exact affine sensitivity of the frozen-dynamics double-integrator
    recursion to the piecewise-constant torque decision vector -- same
    algorithm as v5's mpc.py `_sensitivity` (unit-column simulation via
    superposition), just fused into one compiled kernel operating on whole
    NV-column blocks at once (block b's NV unit responses share the same
    on/off timing, so they can be propagated together as one NVxNV matrix
    recursion instead of NV separate NV-vector recursions).

    Returns qf, qdf: (NP+1, NV) free response (tau_stack=0).
            Phi_q, Phi_qd: (NP+1, NV, NC*NV) -- q(t) = qf[t] + Phi_q[t] @ x,
            qd(t) = qdf[t] + Phi_qd[t] @ x, x = flattened (block, joint)
            decision vector.
    """
    qf = np.zeros((NP_ + 1, NV_))
    qdf = np.zeros((NP_ + 1, NV_))
    qf[0] = q0
    qdf[0] = qd0
    qacc_free = Minv @ (-bias)
    for t in range(NP_):
        qdf[t + 1] = qdf[t] + qacc_free * DT
        qf[t + 1] = qf[t] + qdf[t] * DT + 0.5 * qacc_free * DT * DT

    n_blk = NC_ * NV_
    Phi_q = np.zeros((NP_ + 1, NV_, n_blk))
    Phi_qd = np.zeros((NP_ + 1, NV_, n_blk))
    for b in range(NC_):
        t_start = b * L_
        t_end = (b + 1) * L_
        q_local = np.zeros((NV_, NV_))
        qd_local = np.zeros((NV_, NV_))
        for t in range(t_start, NP_):
            if t < t_end:
                qacc = Minv  # each column j of Minv IS the unit-torque response accel for joint j
            else:
                qacc = np.zeros((NV_, NV_))
            q_new = q_local + qd_local * DT + 0.5 * qacc * DT * DT
            qd_new = qd_local + qacc * DT
            Phi_q[t + 1, :, b * NV_:(b + 1) * NV_] = q_new
            Phi_qd[t + 1, :, b * NV_:(b + 1) * NV_] = qd_new
            q_local = q_new
            qd_local = qd_new
    return qf, qdf, Phi_q, Phi_qd


@njit(cache=True)
def _assemble(q0_e, qd0_e, Minv_e, bias_e,
              q0_i, qd0_i, Minv_i, bias_i,
              q0_left, q_int_des_traj, qd_int_des_traj,
              tau_ego_prev, tau_int_prev,
              Jp_ee, Jr_ee,
              n_arr, Je_arr, Ji_arr, h0_arr, coll_steps,
              NC_, NV_, NP_, L_, K_, DT,
              W_POS_EGO_, W_VEL_EGO_, W_POS_INT_, W_VEL_INT_,
              W_JERK_EGO_, W_JERK_INT_, W_EE_HOLD_,
              REG_EFFORT_EGO_, REG_EFFORT_INT_,
              EE_SLACK_POS_, EE_SLACK_ROT_):
    """Fused objective + constraint assembly. Returns H (ntot,ntot),
    f (ntot,), a_ineq (m,ntot), b_ineq (m,) ready for solve_constrained_box_qp.
    ntot = 2*NC_*NV_ (ego block stacked before intruder block, matching v5's
    layout so downstream indexing (`x[0:NV]`, `x[n_e:n_e+NV]`) is unchanged).
    """
    n_e = NC_ * NV_
    n_i = NC_ * NV_
    ntot = n_e + n_i

    qf_e, qdf_e, Phq_e, Phqd_e = _sensitivity_core(q0_e, qd0_e, Minv_e, bias_e, NC_, NV_, NP_, L_, DT)
    qf_i, qdf_i, Phq_i, Phqd_i = _sensitivity_core(q0_i, qd0_i, Minv_i, bias_i, NC_, NV_, NP_, L_, DT)

    H = np.zeros((ntot, ntot))
    f = np.zeros(ntot)

    # ---- Tikhonov effort regularization (ASYMMETRIC -- priority 4a) ----
    for j in range(n_e):
        H[j, j] += 2.0 * REG_EFFORT_EGO_
    for j in range(n_i):
        H[n_e + j, n_e + j] += 2.0 * REG_EFFORT_INT_

    # ---- ego pos/vel tracking + EE-hold, intruder pos/vel tracking ----
    for t in range(1, NP_ + 1):
        Ae = Phq_e[t]              # (NV, n_e)
        be = q0_left - qf_e[t]      # (NV,)
        AeT = Ae.T
        H[0:n_e, 0:n_e] += 2.0 * W_POS_EGO_ * (AeT @ Ae)
        f[0:n_e] += -2.0 * W_POS_EGO_ * (AeT @ be)

        Aev = Phqd_e[t]
        bev = -qdf_e[t]
        AevT = Aev.T
        H[0:n_e, 0:n_e] += 2.0 * W_VEL_EGO_ * (AevT @ Aev)
        f[0:n_e] += -2.0 * W_VEL_EGO_ * (AevT @ bev)

        # explicit EE-hold soft cost (priority 4c): translational drift only
        Aee = Jp_ee @ Phq_e[t]                 # (3, n_e)
        bee = -(Jp_ee @ (qf_e[t] - q0_left))     # (3,)
        AeeT = Aee.T
        H[0:n_e, 0:n_e] += 2.0 * W_EE_HOLD_ * (AeeT @ Aee)
        f[0:n_e] += -2.0 * W_EE_HOLD_ * (AeeT @ bee)

        Ai = Phq_i[t]
        bi = q_int_des_traj[t] - qf_i[t]
        AiT = Ai.T
        H[n_e:ntot, n_e:ntot] += 2.0 * W_POS_INT_ * (AiT @ Ai)
        f[n_e:ntot] += -2.0 * W_POS_INT_ * (AiT @ bi)

        Aiv = Phqd_i[t]
        biv = qd_int_des_traj[t] - qdf_i[t]
        AivT = Aiv.T
        H[n_e:ntot, n_e:ntot] += 2.0 * W_VEL_INT_ * (AivT @ Aiv)
        f[n_e:ntot] += -2.0 * W_VEL_INT_ * (AivT @ biv)

    # ---- jerk / block-continuity (ASYMMETRIC -- priority 4b) ----
    for j in range(NV_):
        H[j, j] += 2.0 * W_JERK_EGO_
        f[j] += -2.0 * W_JERK_EGO_ * tau_ego_prev[j]
        H[n_e + j, n_e + j] += 2.0 * W_JERK_INT_
        f[n_e + j] += -2.0 * W_JERK_INT_ * tau_int_prev[j]
    for b in range(1, NC_):
        i0 = b * NV_
        i1 = (b - 1) * NV_
        for j in range(NV_):
            H[i0 + j, i0 + j] += 2.0 * W_JERK_EGO_
            H[i1 + j, i1 + j] += 2.0 * W_JERK_EGO_
            H[i0 + j, i1 + j] += -2.0 * W_JERK_EGO_
            H[i1 + j, i0 + j] += -2.0 * W_JERK_EGO_
            oi0 = n_e + i0 + j
            oi1 = n_e + i1 + j
            H[oi0, oi0] += 2.0 * W_JERK_INT_
            H[oi1, oi1] += 2.0 * W_JERK_INT_
            H[oi0, oi1] += -2.0 * W_JERK_INT_
            H[oi1, oi0] += -2.0 * W_JERK_INT_

    # ---- collision constraints: K nearest pairs, at `coll_steps` predicted
    # steps only (NOT every one of NP_ -- see module docstring point 3: with
    # the numba speedup alone, a longer NP_ made the OBJECTIVE/tracking
    # horizon longer for free, but the reused constrained_box_qp's cost
    # scales with the CONSTRAINT ROW COUNT m, so collision rows are
    # subsampled to `coll_steps` (e.g. one check per control block, at each
    # block's END) to keep m -- and therefore solve time -- from scaling
    # with the full physics-step-resolution horizon. This is a genuine
    # approximation (a barrier violation strictly BETWEEN two checked steps
    # would not be caught), documented and measured in RESULTS_v6.md.
    NCS = coll_steps.shape[0]
    m_coll = K_ * NCS
    m_grasp = 2 * 2 * 3 * 2  # 2 stages x (pos,rot) x 3 axes x 2 signs = 24
    m = m_coll + m_grasp
    a_ineq = np.zeros((m, ntot))
    b_ineq = np.zeros(m)

    row = 0
    for k in range(K_):
        c_e = n_arr[k] @ Je_arr[k]   # (NV,)
        c_i = n_arr[k] @ Ji_arr[k]   # (NV,)
        h0 = h0_arr[k]
        for si in range(NCS):
            t = coll_steps[si]
            row_e = -(c_e @ Phq_e[t])   # (n_e,)
            row_i = c_i @ Phq_i[t]       # (n_i,)
            a_ineq[row, 0:n_e] = row_e
            a_ineq[row, n_e:ntot] = row_i
            b_ineq[row] = h0 + c_e @ (qf_e[t] - q0_e) - c_i @ (qf_i[t] - q0_i)
            row += 1

    # ---- ego grasp (hard-ish): EE translation + small-angle rotation ----
    stages = (NP_ // 2, NP_)
    for si in range(2):
        t = stages[si]
        for which in range(2):
            J_ee = Jp_ee if which == 0 else Jr_ee
            slack = EE_SLACK_POS_ if which == 0 else EE_SLACK_ROT_
            d_free = J_ee @ (qf_e[t] - q0_left)   # (3,)
            Phi_ee = J_ee @ Phq_e[t]                # (3, n_e)
            for i in range(3):
                a_ineq[row, 0:n_e] = Phi_ee[i]
                b_ineq[row] = slack - d_free[i]
                row += 1
                a_ineq[row, 0:n_e] = -Phi_ee[i]
                b_ineq[row] = slack + d_free[i]
                row += 1

    return H, f, a_ineq, b_ineq


class JointMPCFast:
    """v6 fast joint cooperative MPC controller -- numba-assembled QP, longer
    horizon, explicit priority cost. See module docstring for what changed
    relative to mpc.py (v5)."""

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
        self.last_assemble_time = 0.0
        self.last_qp_time = 0.0
        self.last_m_rows = 0

    def step(self, model, data, q_int_des_traj, qd_int_des_traj, pairs_all):
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
        M_box, bias_box = ctrl.payload_terms(data)
        M_ll = M_ll + M_box
        bias_l = bias_l + bias_box
        Minv_ll = np.linalg.inv(M_ll)
        Minv_rr = np.linalg.inv(M_rr)

        Jp_full, Jr_full = sc.body_jac_full(model, data, ctrl.ee_body_id)
        Jp_ee = np.ascontiguousarray(Jp_full[:, ctrl.left_dofadr])
        Jr_ee = np.ascontiguousarray(Jr_full[:, ctrl.left_dofadr])

        pairs_sorted = sorted(pairs_all, key=lambda p: p["h"])[:K_PAIRS]
        n_arr = np.stack([p["n"] for p in pairs_sorted])
        Je_arr = np.stack([p["Je"] for p in pairs_sorted])
        Ji_arr = np.stack([p["Ji"] for p in pairs_sorted])
        h0_arr = np.array([p["h"] for p in pairs_sorted])
        coll_steps = _coll_steps()

        t_assemble0 = time.time()
        H, f, a_ineq, b_ineq = _assemble(
            q_ego0, qd_ego0, Minv_ll, bias_l,
            q_int0, qd_int0, Minv_rr, bias_r,
            self.q0_left, q_int_des_traj, qd_int_des_traj,
            self.tau_ego_prev, self.tau_int_prev,
            Jp_ee, Jr_ee,
            n_arr, Je_arr, Ji_arr, h0_arr, coll_steps,
            NC, NV, NP, L_PER_BLOCK, K_PAIRS, sc.DT,
            W_POS_EGO, W_VEL_EGO, W_POS_INT, W_VEL_INT,
            W_JERK_EGO, W_JERK_INT, W_EE_HOLD,
            REG_EFFORT_EGO, REG_EFFORT_INT,
            EE_SLACK_POS, EE_SLACK_ROT)
        self.last_assemble_time = time.time() - t_assemble0
        self.last_m_rows = a_ineq.shape[0]

        lower = -sc.TAU_MAX * np.ones(self.ntot)
        upper = sc.TAU_MAX * np.ones(self.ntot)
        t_qp0 = time.time()
        x, _, feasible = sc.solve_constrained_box_qp(
            H, f, lower, upper, a_ineq, b_ineq,
            dual_sweeps=DUAL_SWEEPS, dual_root_iters=DUAL_ROOT_ITERS)
        self.last_qp_time = time.time() - t_qp0

        tau_ego = x[0:NV].copy()
        tau_int = x[self.n_e:self.n_e + NV].copy()
        self.tau_ego_prev = tau_ego.copy()
        self.tau_int_prev = tau_int.copy()

        self.last_solve_time = time.time() - t_start
        return tau_ego, tau_int, bool(feasible), self.last_solve_time


def warmup():
    """Trigger numba JIT compilation once, outside timed runs (cache=True
    also persists the compiled code to disk across processes)."""
    NVt = NV
    q0 = np.zeros(NVt)
    qd0 = np.zeros(NVt)
    Minv = np.eye(NVt)
    bias = np.zeros(NVt)
    _sensitivity_core(q0, qd0, Minv, bias, NC, NVt, NP, L_PER_BLOCK, sc.DT)
    n_arr = np.zeros((K_PAIRS, 3))
    n_arr[:, 0] = 1.0
    Je_arr = np.zeros((K_PAIRS, 3, NVt))
    Ji_arr = np.zeros((K_PAIRS, 3, NVt))
    h0_arr = np.ones(K_PAIRS) * 0.1
    q_traj = np.zeros((NP + 1, NVt))
    qd_traj = np.zeros((NP + 1, NVt))
    Jp_ee = np.zeros((3, NVt))
    Jr_ee = np.zeros((3, NVt))
    coll_steps = _coll_steps()
    _assemble(q0, qd0, Minv, bias, q0, qd0, Minv, bias,
              q0, q_traj, qd_traj, np.zeros(NVt), np.zeros(NVt),
              Jp_ee, Jr_ee, n_arr, Je_arr, Ji_arr, h0_arr, coll_steps,
              NC, NVt, NP, L_PER_BLOCK, K_PAIRS, sc.DT,
              W_POS_EGO, W_VEL_EGO, W_POS_INT, W_VEL_INT,
              W_JERK_EGO, W_JERK_INT, W_EE_HOLD,
              REG_EFFORT_EGO, REG_EFFORT_INT,
              EE_SLACK_POS, EE_SLACK_ROT)
