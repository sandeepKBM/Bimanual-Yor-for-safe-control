"""Feedback-linearized cooperative MPC (prototype).

Motivation (see docs/design/mpc_formulation.md §5 + review 2026-09-08): the
torque-space MPC (mpc7) must LINEARIZE the nonlinear dynamics q̈=M(q)^-1(τ-c)
over the horizon, and that linearization is only valid on the near horizon
(open-loop response ~doubles every 40ms), so a long horizon injected 3.4-rad
garbage and the plan collided.

Feedback linearization removes the linearization ENTIRELY. Choose per arm
    τ = M(q)·a + c(q,q̇)        ⇒        q̈ = a      (EXACT double integrator)
and let the MPC plan the acceleration a. Then:
  * the horizon prediction q(t) = q0 + q̇0·t + Φ_q(t)·a is EXACT and LINEAR --
    Φ is the double-integrator sensitivity, a CONSTANT precomputed matrix
    (Minv ≡ I), no per-step MuJoCo rollout, no divergence, any horizon length.
  * the arm-arm barrier over the horizon uses that exact q(t) (first-order in
    the barrier geometry only), giving genuine anticipation (yield early).
  * torque limits are the state-dependent rows |M(q0)·a0 + c| ≤ τ_max (linear
    in the applied first-block accel a0).
  * the realized torque τ = M(q0)·a0 + c cancels gravity/Coriolis exactly (in
    sim, where M,c are known from MuJoCo; on hardware add a robustness margin).

This is a JOINT QP over BOTH arms' acceleration profiles, so the coupled
collision constraint makes both arms cooperatively yield (balance set by the
per-arm tracking weights, not an asymmetric margin).

Numba optimization (2026-09-08, see mpc7.py's `_assemble` for the same
pattern): Φ_q(t)/Φ_qd(t) are CONSTANT -- they depend only on cq/cqd, NC, L,
NV, DT, never on state -- so the original prototype's `_phi_q(t)`/`_phi_qd(t)`
(rebuilt from scratch, in Python, on EVERY call inside the tracking-cost and
collision-row loops) was pure waste. `FLMPC.__init__` now builds the full
stacked tensors `Phi_q_all`/`Phi_qd_all` (NP+1, NV, NC*NV) ONCE (by literally
calling the original `_phi_q`/`_phi_qd` methods once each, at construction
time, so the cached tensor is bit-identical to what the prototype rebuilt
per-call), and the numeric assembly of the dominant cost terms (tracking
cost, effort/jerk regularization, anticipatory arm-arm clearance rows,
torque-box rows) is done by the `@njit` kernel `_assemble_fl` below, which
takes those precomputed tensors plus the per-step MuJoCo-derived numpy
arrays (M/c, references, collision-pair arrays) and replicates the ORIGINAL
numpy code's math/row order/signs exactly (see `_assemble_fl`'s docstring
and the equivalence test in the job scratch dir). The two OPTIONAL,
low-row-count extras (cup-upright tilt rows, table-floor rows) stay in plain
Python in `step()` -- they are cheap and default-off, so keeping them
un-jitted minimizes the risk of a subtle refactor bug in a rarely-exercised
path while the njit kernel captures essentially all of the original
prototype's ~44-69ms/step cost. MuJoCo calls (full_mass_matrix,
body_jac_full, qfrc_bias/passive) necessarily stay in Python (can't be
njit'd); their outputs are gathered into plain numpy arrays before the
kernel call, same as mpc7.JointMPCv7.step already does for its own
collision-pair arrays.
"""
from __future__ import annotations
import numpy as np
from numba import njit
import sim_common as sc


def di_coeffs(NC, L, NP, DT):
    """Double-integrator scalar sensitivity coefficients (each joint is an
    independent q̈=a double integrator, so the per-block map is scalar·I).
    Returns cq (NP+1, NC), cqd (NP+1, NC): q(t)/q̇(t) response to a UNIT constant
    acceleration held during block b's active window [b·L,(b+1)·L). Matches the
    explicit integration mpc7._sensitivity_core used (with Minv=I)."""
    cq = np.zeros((NP + 1, NC))
    cqd = np.zeros((NP + 1, NC))
    for b in range(NC):
        q_loc = 0.0
        qd_loc = 0.0
        for t in range(b * L, NP):
            acc = 1.0 if t < (b + 1) * L else 0.0
            qd_new = qd_loc + acc * DT
            q_new = q_loc + qd_new * DT          # SEMI-IMPLICIT Euler (matches MuJoCo's default integrator)
            cq[t + 1, b] = q_new
            cqd[t + 1, b] = qd_new
            q_loc, qd_loc = q_new, qd_new
    return cq, cqd


@njit(cache=True)
def _assemble_fl(Phi_q_all, Phi_qd_all,
                  qnom_e, qdnom_e, qnom_i, qdnom_i,
                  q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
                  a_ego_prev, a_int_prev,
                  n_arr, Je_arr, Ji_arr, h0_arr, coll_steps,
                  qd0_e, qd0_i, tvec,
                  M_e, c_e, M_i, c_i,
                  NC, NV, n_blk,
                  W_pe, W_ve, W_pi, W_vi, W_r, W_j,
                  H_MARGIN, TAU_MAX):
    """Numba assembly kernel: builds H, f, a_ineq, b_ineq EXACTLY as the
    original numpy FLMPC.step did, using PRECOMPUTED Phi_q_all/Phi_qd_all
    (NP+1, NV, n_blk) in place of the original's per-t `_phi_q(t)`/
    `_phi_qd(t)` rebuild. Covers:
      (a) tracking cost (ego+int pos/vel, over t=1..NP)
      (b) effort + jerk regularization (per block, both arms)
      (c) anticipatory arm-arm clearance rows (K pairs x coll_steps)
      (d) torque-box rows |M_e@a_e0+c_e|<=TAU_MAX, |M_i@a_i0+c_i|<=TAU_MAX
    in that exact row order (collision rows first, then torque rows) --
    matching the original's `a_rows`/`b_rows` append order for (1) and (2).
    The optional cup-upright tilt / table-floor rows ((3)/(4) in the
    original) are appended by FLMPC.step in plain Python afterward (cheap,
    low row count, default-off) -- see FLMPC.step."""
    NP = Phi_q_all.shape[0] - 1
    ntot = 2 * n_blk
    H = np.zeros((ntot, ntot))
    f = np.zeros(ntot)

    # --- tracking cost (exact prediction q=qnom+Phi@a) ---
    for t in range(1, NP + 1):
        Aq = Phi_q_all[t]
        Aqd = Phi_qd_all[t]
        AqT = Aq.T
        AqdT = Aqd.T

        be = q_ego_ref[t] - qnom_e[t]
        H[0:n_blk, 0:n_blk] += 2.0 * W_pe * (AqT @ Aq)
        f[0:n_blk] += -2.0 * W_pe * (AqT @ be)
        bev = qd_ego_ref[t] - qdnom_e[t]
        H[0:n_blk, 0:n_blk] += 2.0 * W_ve * (AqdT @ Aqd)
        f[0:n_blk] += -2.0 * W_ve * (AqdT @ bev)

        bi = q_int_ref[t] - qnom_i[t]
        H[n_blk:ntot, n_blk:ntot] += 2.0 * W_pi * (AqT @ Aq)
        f[n_blk:ntot] += -2.0 * W_pi * (AqT @ bi)
        biv = qd_int_ref[t] - qdnom_i[t]
        H[n_blk:ntot, n_blk:ntot] += 2.0 * W_vi * (AqdT @ Aqd)
        f[n_blk:ntot] += -2.0 * W_vi * (AqdT @ biv)

    # --- effort regularization (all blocks, both arms) ---
    for j in range(ntot):
        H[j, j] += 2.0 * W_r

    # --- jerk regularization: ego block (base=0) ---
    for j in range(NV):
        H[j, j] += 2.0 * W_j
        f[j] += -2.0 * W_j * a_ego_prev[j]
    for b in range(1, NC):
        i0 = b * NV
        i1 = (b - 1) * NV
        for j in range(NV):
            H[i0 + j, i0 + j] += 2.0 * W_j
            H[i1 + j, i1 + j] += 2.0 * W_j
            H[i0 + j, i1 + j] += -2.0 * W_j
            H[i1 + j, i0 + j] += -2.0 * W_j

    # --- jerk regularization: int block (base=n_blk) ---
    for j in range(NV):
        H[n_blk + j, n_blk + j] += 2.0 * W_j
        f[n_blk + j] += -2.0 * W_j * a_int_prev[j]
    for b in range(1, NC):
        i0 = n_blk + b * NV
        i1 = n_blk + (b - 1) * NV
        for j in range(NV):
            H[i0 + j, i0 + j] += 2.0 * W_j
            H[i1 + j, i1 + j] += 2.0 * W_j
            H[i0 + j, i1 + j] += -2.0 * W_j
            H[i1 + j, i0 + j] += -2.0 * W_j

    # --- constraint rows: (1) anticipatory arm-arm clearance, (2) torque box ---
    K = n_arr.shape[0]
    NCS = coll_steps.shape[0]
    m_coll = K * NCS
    m_torque = 4 * NV
    m = m_coll + m_torque
    a_ineq = np.zeros((m, ntot))
    b_ineq = np.zeros(m)

    row = 0
    for k in range(K):
        n_k = n_arr[k]
        ce = n_k @ Je_arr[k]
        ci = n_k @ Ji_arr[k]
        h0 = h0_arr[k]
        for si in range(NCS):
            t = coll_steps[si]
            Aq = Phi_q_all[t]
            a_ineq[row, 0:n_blk] = -(ce @ Aq)
            a_ineq[row, n_blk:ntot] = ci @ Aq
            drift = ce @ (qd0_e * tvec[t]) - ci @ (qd0_i * tvec[t])
            b_ineq[row] = h0 - H_MARGIN + drift
            row += 1

    for j in range(NV):
        a_ineq[row, 0:NV] = M_e[j]
        b_ineq[row] = TAU_MAX - c_e[j]
        row += 1
        a_ineq[row, 0:NV] = -M_e[j]
        b_ineq[row] = TAU_MAX + c_e[j]
        row += 1
    for j in range(NV):
        a_ineq[row, n_blk:n_blk + NV] = M_i[j]
        b_ineq[row] = TAU_MAX - c_i[j]
        row += 1
        a_ineq[row, n_blk:n_blk + NV] = -M_i[j]
        b_ineq[row] = TAU_MAX + c_i[j]
        row += 1

    return H, f, a_ineq, b_ineq


class FLMPC:
    def __init__(self, model, NC=3, L=8, NV=6, DT=None,
                 W_POS_EGO=3000.0, W_VEL_EGO=15.0, W_POS_INT=600.0, W_VEL_INT=8.0,
                 R_ACC=1e-4, W_JERK=1e-3, K_PAIRS=4, H_MARGIN=0.02,
                 coll_stride=2, a_box=800.0,
                 left_ee_body=None, tilt_max_rad=None):
        self.model = model
        self.NC, self.L, self.NV = NC, L, NV
        self.NP = NC * L
        self.DT = sc.DT if DT is None else DT
        self.W = dict(pe=W_POS_EGO, ve=W_VEL_EGO, pi=W_POS_INT, vi=W_VEL_INT,
                      r=R_ACC, j=W_JERK)
        self.K_PAIRS = K_PAIRS
        self.H_MARGIN = H_MARGIN
        self.coll_stride = coll_stride
        self.a_box = a_box
        self.cq, self.cqd = di_coeffs(NC, L, self.NP, self.DT)
        self.n_blk = NC * NV
        self.a_ego_prev = np.zeros(NV)
        self.a_int_prev = np.zeros(NV)
        self.last_feasible = True
        # --- CUP-UPRIGHT constraint config (optional, default OFF so existing
        # callers/tests are unaffected). left_ee_body: the LEFT EE body id whose
        # ROTATION about world x,y is bounded (this is the body the cup is
        # rigidly re-attached to every step, see eval/scene_setup.
        # reattach_cup_rigidly -- constraining its tilt directly bounds the
        # cup's tilt, since the cup<-EE offset is fixed). tilt_max_rad: the
        # +-tilt bound [rad]. Either may also be supplied per-call to step().
        self.left_ee_body = left_ee_body
        self.tilt_max_rad = tilt_max_rad
        # world-"up" expressed in the EE body's own local frame, captured ONCE
        # (lazily, on the first step() call with the constraint enabled) --
        # defines "vertical" for tilt0 below. Not knowable at __init__ time
        # (no `data` yet).
        self._cup_axis_local = None

        # --- numba precompute (see module docstring): Phi_q(t)/Phi_qd(t) are
        # CONSTANT -- build the full stacked tensors ONCE, by calling the
        # original per-t methods once each (so the cache is bit-identical to
        # what the prototype rebuilt on every call/every t), plus the fixed
        # horizon-step index arrays the anticipatory-clearance and
        # cup-upright rows iterate over. ---
        self.Phi_q_all = np.ascontiguousarray(
            np.stack([self._phi_q(t) for t in range(self.NP + 1)]))
        self.Phi_qd_all = np.ascontiguousarray(
            np.stack([self._phi_qd(t) for t in range(self.NP + 1)]))
        self.coll_steps = np.arange(self.coll_stride, self.NP + 1, self.coll_stride,
                                     dtype=np.int64)
        self.tilt_steps = np.arange(self.L, self.NP + 1, self.L, dtype=np.int64)
        self._warmup()

    def _phi_q(self, t):
        """(NV, NC*NV) double-integrator position sensitivity at step t."""
        NV, NC = self.NV, self.NC
        Phi = np.zeros((NV, NC * NV))
        for b in range(NC):
            Phi[:, b * NV:(b + 1) * NV] = self.cq[t, b] * np.eye(NV)
        return Phi

    def _phi_qd(self, t):
        NV, NC = self.NV, self.NC
        Phi = np.zeros((NV, NC * NV))
        for b in range(NC):
            Phi[:, b * NV:(b + 1) * NV] = self.cqd[t, b] * np.eye(NV)
        return Phi

    def _warmup(self):
        """Trigger numba JIT compilation of `_assemble_fl` ONCE, here at
        construction time, on dummy (zero-valued, correctly-shaped) arrays,
        so the first REAL step() call doesn't pay the JIT compile cost.
        Purely a compile-triggering call -- its output is discarded."""
        NV, NC, NP, n_blk = self.NV, self.NC, self.NP, self.n_blk
        K = max(self.K_PAIRS, 1)
        qtraj = np.zeros((NP + 1, NV))
        n_arr = np.zeros((K, 3)); n_arr[:, 0] = 1.0
        Je_arr = np.zeros((K, 3, NV))
        Ji_arr = np.zeros((K, 3, NV))
        h0_arr = np.ones(K) * 0.1
        tvec = np.arange(NP + 1) * self.DT
        _assemble_fl(
            self.Phi_q_all, self.Phi_qd_all,
            qtraj, qtraj, qtraj, qtraj,
            qtraj, qtraj, qtraj, qtraj,
            np.zeros(NV), np.zeros(NV),
            n_arr, Je_arr, Ji_arr, h0_arr, self.coll_steps,
            np.zeros(NV), np.zeros(NV), tvec,
            np.eye(NV), np.zeros(NV), np.eye(NV), np.zeros(NV),
            NC, NV, n_blk,
            self.W['pe'], self.W['ve'], self.W['pi'], self.W['vi'], self.W['r'], self.W['j'],
            self.H_MARGIN, sc.TAU_MAX)

    def step(self, model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
             q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
             pairs_all, M_ego_extra=None, bias_ego_extra=None,
             left_ee_body=None, tilt_max_rad=None, table_rows=None):
        """One control step. q_*_ref / qd_*_ref: (NP+1, NV) reference
        trajectories (index 0 = now). table_rows: optional
        (a_tab_e, b_tab_e, a_tab_i, b_tab_i) per-arm TABLE-FLOOR HOCBF rows in
        the usual TORQUE convention (a_tab @ tau <= b_tab -- e.g. straight
        from sim_common.table_plane_barrier_rows; SAME format/tuple order
        mpc7.JointMPCv7.step's own `table_rows` kwarg consumes on its
        first-block torque columns). Default None -> no rows added, so
        existing callers/tests that don't pass this are bit-for-bit
        unaffected. Returns (tau_ego, tau_int, feasible)."""
        NV, NC, NP, DT = self.NV, self.NC, self.NP, self.DT
        n_blk = self.n_blk
        ntot = 2 * n_blk
        eslice = slice(0, n_blk)
        islice = slice(n_blk, ntot)

        # current states -- positions via qposadr, velocities via dofadr
        # (the YOR arms have distinct qpos vs dof addresses).
        q0_e = data.qpos[left_qposadr].copy()
        qd0_e = data.qvel[left_dofadr].copy()
        q0_i = data.qpos[right_qposadr].copy()
        qd0_i = data.qvel[right_dofadr].copy()

        # feedback-linearization terms M(q0), c(q0,q̇0) per arm (exact, from MuJoCo)
        M_full = sc.full_mass_matrix(model, data)
        M_e = M_full[np.ix_(left_dofadr, left_dofadr)].copy()
        M_i = M_full[np.ix_(right_dofadr, right_dofadr)].copy()
        # c = qfrc_bias - qfrc_passive so tau = M*a + c makes q̈ = a EXACTLY
        # (cancels gravity/Coriolis AND joint damping/springs; MuJoCo EOM is
        # M q̈ = qfrc_applied + qfrc_passive - qfrc_bias).
        c_e = (data.qfrc_bias[left_dofadr] - data.qfrc_passive[left_dofadr]).copy()
        c_i = (data.qfrc_bias[right_dofadr] - data.qfrc_passive[right_dofadr]).copy()
        if M_ego_extra is not None:
            M_e = M_e + M_ego_extra
        if bias_ego_extra is not None:
            c_e = c_e + bias_ego_extra

        # nominal (a=0) drift trajectory, EXACT double integrator
        tvec = np.arange(NP + 1) * DT
        qnom_e = q0_e[None, :] + np.outer(tvec, qd0_e)
        qnom_i = q0_i[None, :] + np.outer(tvec, qd0_i)
        qdnom_e = np.tile(qd0_e, (NP + 1, 1))
        qdnom_i = np.tile(qd0_i, (NP + 1, 1))

        # collision pairs -> stacked arrays (top K_PAIRS closest), same
        # gather-in-Python-then-hand-to-njit pattern as mpc7.JointMPCv7.step.
        pairs = sorted(pairs_all, key=lambda p: p["h"])[:self.K_PAIRS]
        n_arr = np.stack([p["n"] for p in pairs])
        Je_arr = np.stack([p["Je"] for p in pairs])
        Ji_arr = np.stack([p["Ji"] for p in pairs])
        h0_arr = np.array([p["h"] for p in pairs])

        # --- numba-assembled: tracking cost + effort/jerk cost + (1)
        # anticipatory clearance rows + (2) torque-box rows. ---
        H, f, a_ineq, b_ineq = _assemble_fl(
            self.Phi_q_all, self.Phi_qd_all,
            qnom_e, qdnom_e, qnom_i, qdnom_i,
            q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
            self.a_ego_prev, self.a_int_prev,
            n_arr, Je_arr, Ji_arr, h0_arr, self.coll_steps,
            qd0_e, qd0_i, tvec,
            M_e, c_e, M_i, c_i,
            NC, NV, n_blk,
            self.W['pe'], self.W['ve'], self.W['pi'], self.W['vi'], self.W['r'], self.W['j'],
            self.H_MARGIN, sc.TAU_MAX)

        # --- (3) CUP-UPRIGHT (optional, plain Python -- cheap/default-off):
        # bound the LEFT EE's predicted small-rotation about world x,y over a
        # few horizon steps to +-tilt_max_rad. The cup is kinematically rigid
        # on this EE (see eval/scene_setup.reattach_cup_rigidly: cup pose =
        # EE pose (*) a FIXED offset every step), so bounding the EE's tilt
        # directly bounds the cup's tilt -- while leaving EE TRANSLATION (and
        # hence its ability to yield to the intruder) completely
        # unconstrained. Off by default (both left_ee_body and tilt_max_rad
        # None => no rows added, so existing callers/tests that don't pass
        # these args are bit-for-bit unaffected). Reuses the precomputed
        # Phi_q_all (identical values to the original's per-call _phi_q(t)).
        extra_a, extra_b = [], []
        ee_body = left_ee_body if left_ee_body is not None else self.left_ee_body
        tilt_max = tilt_max_rad if tilt_max_rad is not None else self.tilt_max_rad
        if ee_body is not None and tilt_max is not None:
            if self._cup_axis_local is None:
                # capture world-"up" in the EE's local frame ONCE, at whatever
                # orientation the EE has on this first call -- this defines
                # "vertical" (tilt0=0 there) for every later call.
                R_ee0 = data.xmat[ee_body].reshape(3, 3).copy()
                self._cup_axis_local = R_ee0.T @ np.array([0.0, 0.0, 1.0])
            R_ee = data.xmat[ee_body].reshape(3, 3)
            cup_axis_world = R_ee @ self._cup_axis_local
            # small-angle rotation vector (world x,y) taking world-up "up" to
            # the EE's CURRENT cup axis "b": b ~= up + omega x up, so with
            # up=[0,0,1]:  omega_x = -b_y,  omega_y = b_x.
            tilt0 = np.array([-cup_axis_world[1], cup_axis_world[0]])
            Jr_ee = sc.body_jac_full(model, data, ee_body)[1][:, left_dofadr]  # (3,NV) at t=0
            for t in self.tilt_steps:   # a few horizon steps (block boundaries)
                Aq = self.Phi_q_all[t]
                drift_t = qd0_e * tvec[t]
                for ax in (0, 1):   # world x, y (tilt about the horizontal axes)
                    row_coef = Jr_ee[ax] @ Aq            # (n_blk,) -- ego block columns only
                    drift = tilt0[ax] + Jr_ee[ax] @ drift_t
                    r_up = np.zeros(ntot); r_up[eslice] = row_coef
                    extra_a.append(r_up); extra_b.append(tilt_max - drift)
                    r_lo = np.zeros(ntot); r_lo[eslice] = -row_coef
                    extra_a.append(r_lo); extra_b.append(tilt_max + drift)

        # --- (4) TABLE FLOOR (optional, plain Python -- cheap/default-off,
        # table_rows=None default so existing callers/tests are unaffected).
        # The caller builds these rows in the usual TORQUE convention on that
        # arm's OWN M_inv/bias (which may already include a payload
        # reflection, matching M_e/c_e above) -- FL substitutes the exact
        # tau = M_arm @ a_arm0 + c_arm relation for the applied FIRST-BLOCK
        # accel to get the equivalent ACCELERATION-space row:
        #   a_tab @ (M_arm @ a_arm0 + c_arm) <= b_tab
        #   =>  (a_tab @ M_arm) @ a_arm0  <=  b_tab - a_tab @ c_arm
        if table_rows is not None:
            a_tab_e, b_tab_e, a_tab_i, b_tab_i = table_rows
            for a_tab, b_tab, M_a, c_a, s in ((a_tab_e, b_tab_e, M_e, c_e, eslice),
                                               (a_tab_i, b_tab_i, M_i, c_i, islice)):
                base = s.start
                for r in range(a_tab.shape[0]):
                    row = np.zeros(ntot)
                    row[base:base + NV] = a_tab[r] @ M_a
                    extra_a.append(row)
                    extra_b.append(float(b_tab[r] - a_tab[r] @ c_a))

        if extra_a:
            a_ineq = np.vstack([a_ineq, np.stack(extra_a)])
            b_ineq = np.concatenate([b_ineq, np.array(extra_b)])
        a_ineq = np.ascontiguousarray(a_ineq)
        b_ineq = np.ascontiguousarray(b_ineq)
        lower = -self.a_box * np.ones(ntot)
        upper = self.a_box * np.ones(ntot)
        x, _, feasible = sc.solve_constrained_box_qp(H, f, lower, upper, a_ineq, b_ineq)
        self.last_feasible = bool(feasible)

        a_e0 = x[0:NV].copy()
        a_i0 = x[n_blk:n_blk + NV].copy()
        self.a_ego_prev = a_e0.copy()
        self.a_int_prev = a_i0.copy()
        # realize the feedback-linearizing torque
        tau_e = M_e @ a_e0 + c_e
        tau_i = M_i @ a_i0 + c_i
        return tau_e, tau_i, bool(feasible)
