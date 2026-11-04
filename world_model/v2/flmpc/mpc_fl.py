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
and the equivalence test in the job scratch dir). The OPTIONAL, low-row-count
extras (cup-upright tilt rows, cup-arm displacement-leash rows, table-floor
rows) stay in plain Python in `step()` -- they are cheap and default-off, so
keeping them
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
from . import _sc as sc  # 2026-11-03: was `import sim_common as sc` (sys.path); same 5 items, see _sc.py


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
    The optional cup-upright tilt / cup-arm displacement-leash / table-floor
    rows ((3)/(4)/(5) in FLMPC.step) are appended by FLMPC.step in plain
    Python afterward (cheap, low row count, default-off) -- see FLMPC.step."""
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
                 left_ee_body=None, tilt_max_rad=None,
                 leash_ee_body=None, leash_max_m=None,
                 qp_fallback_rho=1.0e5, qp_fallback_eps=1.0e-3, qp_fallback_smax=5.0,
                 adaptive_yield=False, H_MARGIN_URGENT=0.06, URGENCY_HDOT_REF=0.15,
                 W_POS_EGO_MIN=None, LEASH_MAX_URGENT_M=None, CUP_TILT_SAFE_FRAC=0.6,
                 cup_cone_facets=0, g_accel=9.81,
                 ee_speed_cap=None, ee_speed_body_int=None, speed_cap_steps=None,
                 delay_steps=0, tau_max_vec=None, cone_mode="effgrav"):
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
        # --- GRACEFUL QP-INFEASIBILITY FALLBACK config (see step()'s
        # docstring + `_relaxed_fallback` below for the full rationale). Only
        # used on a step where the HARD QP reports feasible=False; feasible
        # steps are completely unaffected (bit-identical to before this
        # fallback was added). ---
        self.RHO_SLACK = float(qp_fallback_rho)
        self.EPS_SLACK = float(qp_fallback_eps)
        self.S_MAX = float(qp_fallback_smax)
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
        # --- CUP EFFECTIVE-GRAVITY CONE (optional, default OFF: cup_cone_facets=0
        # => the per-axis small-angle tilt BOX below is used, byte-identical to
        # before). When cup_cone_facets=K>0, the cup-upright rows instead bound
        # the cup axis to within tilt_max of the EFFECTIVE-gravity up direction
        # up_eff = (a_cup - g)/||a_cup - g|| (not just world-vertical), via K
        # linear tangent facets of the L2 cone -- so it (a) accounts for
        # ACCELERATION-induced effective-gravity tilt (the dynamic spill driver
        # a pure-orientation box is blind to) and (b) is an L2 cone, not a
        # per-axis L-inf box (which let the COMBINED tilt magnitude exceed
        # tilt_max by up to sqrt(2)). See step()'s block (3). g_accel = |g|.
        self.cup_cone_facets = int(cup_cone_facets)
        self.G_ACCEL = float(g_accel)
        # world-"up" expressed in the EE body's own local frame, captured ONCE
        # (lazily, on the first step() call with the constraint enabled) --
        # defines "vertical" for tilt0 below. Not knowable at __init__ time
        # (no `data` yet).
        self._cup_axis_local = None

        # --- CUP-ARM DISPLACEMENT LEASH config (optional, default OFF so
        # existing callers/tests are unaffected). leash_ee_body: the LEFT EE
        # body id whose Cartesian displacement from the hold pose (q_ego_ref[0],
        # i.e. q_lift) is bounded to +-leash_max_m per world axis over a few
        # horizon steps. This is what lets W_POS_EGO be lowered (so the ego
        # arm actually cooperates with the collision-avoidance/table rows
        # instead of being glued in place by a crude high hold weight) while
        # still guaranteeing the cup arm never wanders arbitrarily far from
        # its hold pose. Either may also be supplied per-call to step().
        self.leash_ee_body = leash_ee_body
        self.leash_max_m = leash_max_m

        # --- ADAPTIVE COOPERATIVE YIELDING (optional, default OFF --
        # adaptive_yield=False reproduces EXACTLY the prior behavior: no
        # extra branches taken in step(), H_MARGIN/W_POS_EGO/leash_max_m
        # used unmodified). See step()'s docstring for the full derivation;
        # summary of the two mechanisms below (2026-09-09 review note /
        # flmpc-adaptive-yield, motivated by the flmpc-qp-infeasibility-
        # cliff note in this module's docstring: sweeps showed collisions
        # come from the anticipatory clearance rows going QP-infeasible in
        # crowded geometry, and once infeasible a collision is essentially
        # unavoidable -- the fallback only softens the torque, it can't
        # regain clearance. The rate-lever that fixed this in the sweep was
        # SLOWING DOWN, i.e. giving the rows more margin/time to react before
        # the pair actually closes -- this adaptive layer buys the same
        # effect locally/on-demand instead of globally, by widening the
        # margin (and, for the cup arm specifically, its willingness to move)
        # exactly when a pair is closing fast and clearance is running out).
        #
        # (1) URGENCY u in [0,1], from the WORST (nearest) of the K_PAIRS
        #     pairs already selected below in step() -- pairs[0] after the
        #     existing `sorted(pairs_all, key=lambda p: p["h"])[:K_PAIRS]`
        #     (global closest, since sorting-then-truncating preserves the
        #     global min at index 0). Two independent axes, multiplied so
        #     BOTH "close" and "closing fast" are required (matches the task
        #     spec: high h0 or hdot>=0 (separating/stationary) => u=0):
        #       u_h    = clip((H_MARGIN_URGENT - h0) / (H_MARGIN_URGENT - H_MARGIN), 0, 1)
        #                -- 0 once h0 reaches the "urgent" band H_MARGIN_URGENT,
        #                1 once h0 has shrunk down to (or past) the BASE
        #                margin H_MARGIN (i.e. already at/through the normal
        #                anticipatory threshold -- the edge of the
        #                infeasibility cliff).
        #       u_rate = clip(max(0, -hdot) / URGENCY_HDOT_REF, 0, 1)
        #                -- 0 when the pair isn't closing (hdot>=0); ramps to
        #                1 as the closing speed reaches URGENCY_HDOT_REF [m/s].
        #       u      = u_h * u_rate
        #     hdot itself: use the pair's own precomputed `hdot` field
        #     (sim_common.all_pair_barriers already returns it -- n@(v_ego-
        #     v_int)) when present; else recompute it from the pair's own
        #     n/v_ego/v_int (also always present in that dict); else (some
        #     other caller's pair dict has neither) fall back to a
        #     finite-difference estimate from the PREVIOUS step's worst h0
        #     (self._prev_worst_h0, an approximation across possibly-
        #     different physical pairs -- last resort only); else 0 (treat as
        #     not closing -- never fabricates urgency from missing data).
        #
        # (2) H_MARGIN_eff = H_MARGIN + u*(H_MARGIN_URGENT - H_MARGIN):
        #     the SAME scalar `_assemble_fl` already takes as its `H_MARGIN`
        #     argument (kernel untouched -- this just swaps which Python
        #     float gets passed in, exactly like the fallback/tilt/leash
        #     rows already do their work in plain Python around the kernel).
        #     Bounded in [H_MARGIN, H_MARGIN_URGENT] by construction (u in
        #     [0,1]) -- pushes BOTH arms apart earlier/harder as a pair
        #     closes fast, before the hard-infeasible corner is entered.
        #
        # (3) ADAPTIVE CUP SHARE: how much of that yielding falls on the CUP
        #     (ego/left) arm vs the collecting (int/right) arm, via a single
        #     gain g = u*(1 - tilt_frac) in [0,1] (tilt_frac below):
        #       W_pos_ego_eff = W_POS_EGO - g*(W_POS_EGO - W_POS_EGO_MIN)
        #         -- lowers the cup arm's hold-tracking weight (so its own
        #         acceleration is cheaper for the joint QP to spend on
        #         yielding) as urgency rises; None (default) => disabled,
        #         W_pos_ego_eff == W_POS_EGO always (byte-identical).
        #       leash_max_eff = leash_max_m + g*(LEASH_MAX_URGENT_M - leash_max_m)
        #         -- enlarges how far the cup EE may wander from its hold
        #         pose while urgency is high; None (default, or leash not
        #         configured at all) => disabled, leash_max_eff == leash_max_m.
        #     tilt_frac = clip(||tilt0|| / (CUP_TILT_SAFE_FRAC * tilt_max_rad), 0, 1)
        #     -- 0 (cup comfortably upright) => g == u (full urgency-driven
        #     loosening); ramps to 1 (cup already tilted past
        #     CUP_TILT_SAFE_FRAC of its hard bound) => g -> 0, so
        #     W_pos_ego_eff/leash_max_eff relax back toward their BASELINE
        #     (never below -- g in [0,1] makes both effective values a
        #     convex combination of the baseline and the loosened bound), i.e.
        #     the cup arm reverts to holding and the collecting arm does more
        #     of the yielding. If the cup-upright constraint isn't engaged
        #     (left_ee_body/tilt_max_rad both None) tilt_frac defaults to 0
        #     (no tilt signal => assume safe, g == u) rather than blocking the
        #     margin-only adaptation.
        #
        # All of (1)-(3) collapse to a no-op when adaptive_yield=False (the
        # default) or when the relevant *_MIN/*_URGENT_M kwarg is left None
        # -- existing callers (run_ab7.py, this module's own _warmup, any
        # caller that doesn't pass these new kwargs) see EXACTLY the prior
        # behavior. ---
        self.adaptive_yield = bool(adaptive_yield)
        self.H_MARGIN_URGENT = float(H_MARGIN_URGENT)
        self.URGENCY_HDOT_REF = float(URGENCY_HDOT_REF)
        self.W_POS_EGO_MIN = None if W_POS_EGO_MIN is None else float(W_POS_EGO_MIN)
        self.LEASH_MAX_URGENT_M = None if LEASH_MAX_URGENT_M is None else float(LEASH_MAX_URGENT_M)
        self.CUP_TILT_SAFE_FRAC = float(CUP_TILT_SAFE_FRAC)
        self._prev_worst_h0 = None          # finite-difference hdot fallback only
        self.last_urgency = 0.0             # u, for introspection/logging only
        self.last_cup_share_gain = 0.0       # g, for introspection/logging only
        self.last_H_MARGIN_eff = self.H_MARGIN
        self.last_W_pos_ego_eff = self.W['pe']

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

        # --- 2026-09-25 CONTROLLER FIXES (all default OFF -> step() byte-identical) ---
        # F1 EE SPEED CAP (int/right arm): ee_speed_cap=V [m/s] + ee_speed_body_int
        # (body id) -> rows |Jp_int @ qd_int(t)| <= V per world axis, plus one row
        # along the current EE motion direction u (u . v(t) <= V), at the horizon
        # steps `speed_cap_steps` (default: step coll_stride + every block boundary).
        # qd_int(t) = qd0 + Phi_qd(t) @ a_int is EXACT for the FL double integrator
        # (Jp frozen at t=0). Slackened together with the collision rows on an
        # infeasible step (_relaxed_fallback); the torque box stays HARD.
        self.ee_speed_cap = None if ee_speed_cap is None else float(ee_speed_cap)
        self.ee_speed_body_int = ee_speed_body_int
        if speed_cap_steps is None:
            speed_cap_steps = sorted(set([self.coll_stride] + list(range(self.L, self.NP + 1, self.L))))
        self.speed_cap_steps = np.asarray(speed_cap_steps, dtype=np.int64)
        # F5 DELAY COMPENSATION: delay_steps=d>0 -> the state the plan acts on is
        # forward-predicted d steps with the already-committed accelerations
        # (queue of the last d applied (a_ego, a_int); EXACT for the double
        # integrator, semi-implicit Euler like MuJoCo). The caller must sample the
        # references d steps ahead (index 0 = when this command takes effect).
        self.delay_steps = int(delay_steps)
        self._a_queue = []                   # oldest first, len <= delay_steps
        # F6 PER-JOINT TORQUE LIMIT: tau_max_vec (NV,) replaces the scalar
        # sc.TAU_MAX in the torque-box rows and _torque_safe_scale. If gravity/bias
        # alone exceeds a joint's limit (|c_j| > tau_max_j), a=0 is NOT feasible; that
        # joint's box is explicitly relaxed to |c_j| (the controller then adds no
        # torque beyond the bias on that joint) and counted in n_taumax_relaxed.
        self.tau_max_vec = None if tau_max_vec is None else np.asarray(tau_max_vec, dtype=np.float64).copy()
        self.n_taumax_relaxed = 0            # steps on which >=1 joint's box had to be relaxed
        # F4 CONE MODE: "effgrav" (the original K-facet effective-gravity cone,
        # facet RHS tilt_max -> a CIRCUMSCRIBED polygon, up to tilt_max/cos(pi/K)) or
        # "true" (K facets on the geometric/true cup tilt ONLY, no acceleration term,
        # RHS tilt_max*cos(pi/K) -> an INSCRIBED polygon, so ||tilt|| <= tilt_max).
        assert cone_mode in ("effgrav", "true"), cone_mode
        self.cone_mode = cone_mode
        # fallback bookkeeping (introspection/logging only)
        self.last_fallback = 0               # 0 feasible, 1 relaxed-slack QP, 2 a=0
        self.n_relaxed = 0
        self.n_zero_fallback = 0
        self.n_limit_steps = 0               # 11-03 opt-in joint position/speed rows: steps that carried them
        self.n_limit_slack = 0               #   ... steps whose returned x violated one of them by > 1e-6 (slack used / row unmet)
        self.n_limit_fallback = 0            #   ... steps on which the hard QP was infeasible while limit rows were present
        self.n_scale_shrunk = 0              # 11-03 counter only: infeasible steps on which _torque_safe_scale actually shrank a0 (lam < 1)
        self.last_M_e = self.last_c_e = self.last_M_i = self.last_c_i = None
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

    def _relaxed_fallback(self, H, f, lower, upper, a_ineq, b_ineq, m_coll, slack_idx=None):
        """GRACEFUL QP-INFEASIBILITY FALLBACK (see step()'s docstring). Called
        ONLY when the hard QP built in step() reports feasible=False (the
        collision-clearance rows + the hard torque-box rows have no common
        solution -- see the module's 2026-09-08 review note /
        flmpc-qp-infeasibility-cliff: this is the FL-MPC's only failure mode,
        and the OLD handling (using solve_constrained_box_qp's own
        infeasible-problem `x`, which the dual ascent has driven to whatever
        extreme point best fights the unsatisfiable row -- still inside the
        acceleration box, but the resulting REALIZED torque tau=M@a+c is NOT
        box-limited and can wildly exceed TAU_MAX) is what flings the arm.

        Formulation chosen: SLACKEN ONLY THE COLLISION-CLEARANCE ROWS (the
        leading `m_coll` rows of a_ineq/b_ineq -- see _assemble_fl's
        docstring for the row order: collision rows first, then the torque
        box, then step()'s optional cup-tilt/leash/table rows). Augment the
        decision vector with ONE nonnegative slack s_k per collision row and
        turn each hard row `a_k @ x <= b_k` into the always-satisfiable
        `a_k @ x - s_k <= b_k` (s_k IS that row's clearance-constraint
        violation, in the row's own units, if x can't satisfy it outright).
        ALL other rows (torque box, cup-tilt, leash, table-floor) are passed
        through completely UNCHANGED -- hard, never relaxed -- so the torque
        box is enforced exactly as strictly as on a feasible step; the
        actuator limit is never the thing that gets traded away.

        Cost added on top of the ORIGINAL tracking/effort/jerk cost (H, f,
        unchanged): RHO_SLACK*sum(s) [linear -- LARGE, dominates the
        tracking-cost gradients, so the solver strongly prefers reducing the
        worst clearance violation over tracking the reference -- this is what
        makes the result "best clearance-improving" rather than merely
        "won't fling"] + EPS_SLACK*sum(s^2) [tiny quadratic term, purely so
        the augmented Hessian block on the slack columns is > 0 for the inner
        box-QP's projected-gradient iteration; negligible next to
        RHO_SLACK*s for any s worth mentioning]. Because a slack can always
        absorb any positive `a_k@x - b_k` (S_MAX is generous -- far above any
        one-step clearance violation this controller would ever see), the
        AUGMENTED problem is feasible whenever the HARD rows alone (torque
        box + optional tilt/leash/table) are jointly satisfiable by SOME x in
        the box -- which is the normal case (a=0 always clears the torque
        box for a physically-reachable M(q),c(q,q̇); the optional rows are
        satisfied at the drift trajectory in every case exercised so far).
        On the vanishingly rare case that even THAT fails, step() itself
        falls back one level further, to a=0 (see step()'s tail).

        Returns (x, feasible2): x has the ORIGINAL `ntot` dims (the slack
        columns are dropped before returning -- callers never see them),
        feasible2 is solve_constrained_box_qp's own feasibility flag on the
        AUGMENTED problem."""
        ntot = H.shape[0]
        m_total = a_ineq.shape[0]
        if slack_idx is not None:
            # F1: slacken an explicit row set (collision rows + EE speed-cap rows).
            m_coll = len(slack_idx)
        aug_n = ntot + m_coll

        H_aug = np.zeros((aug_n, aug_n))
        H_aug[:ntot, :ntot] = H
        for j in range(m_coll):
            H_aug[ntot + j, ntot + j] = 2.0 * self.EPS_SLACK
        f_aug = np.concatenate([f, self.RHO_SLACK * np.ones(m_coll)])

        lower_aug = np.concatenate([lower, np.zeros(m_coll)])
        upper_aug = np.concatenate([upper, self.S_MAX * np.ones(m_coll)])

        a_aug = np.zeros((m_total, aug_n))
        a_aug[:, :ntot] = a_ineq
        for j in range(m_coll):
            r = j if slack_idx is None else int(slack_idx[j])
            a_aug[r, ntot + j] = -1.0   # a_k@x - s_k <= b_k, collision rows only (+ speed rows, F1)
        b_aug = b_ineq

        y, _, feasible2 = sc.solve_constrained_box_qp(H_aug, f_aug, lower_aug, upper_aug,
                                                        a_aug, b_aug)
        return y[:ntot], bool(feasible2)

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
             left_ee_body=None, tilt_max_rad=None, table_rows=None,
             leash_ee_body=None, leash_max_m=None, cup_cone_facets=None,
             ego_hold=True, int_ee_body=None, v_ee_max=None,
             M_int_extra=None, bias_int_extra=None, q_lim=None, qd_lim=None, lim_steps=None):
        """One control step. q_*_ref / qd_*_ref: (NP+1, NV) reference
        trajectories (index 0 = now). table_rows: optional
        (a_tab_e, b_tab_e, a_tab_i, b_tab_i) per-arm TABLE-FLOOR HOCBF rows in
        the usual TORQUE convention (a_tab @ tau <= b_tab -- e.g. straight
        from sim_common.table_plane_barrier_rows; SAME format/tuple order
        mpc7.JointMPCv7.step's own `table_rows` kwarg consumes on its
        first-block torque columns). Default None -> no rows added, so
        existing callers/tests that don't pass this are bit-for-bit
        unaffected. leash_ee_body/leash_max_m: optional CUP-ARM DISPLACEMENT
        LEASH -- see the block below step()'s main body; default None -> no
        rows added. q_lim=(lo, hi) / qd_lim=v (11-03, opt-in, default None -> no rows, bit-identical):
        JOINT POSITION / SPEED rows on BOTH arms, lo <= q_j(t) <= hi and |qd_j(t)| <= v_j at the horizon steps `lim_steps`
        (default {1} + speed_cap_steps), from the exact double-integrator prediction q(t) = qnom(t) + Phi_q(t) a,
        qd(t) = qd0 + Phi_qd(t) a.  lo, hi, v: (NV,) (both arms) or (2, NV) (ego, int); +-inf / non-finite entries -> no row.  The rows are
        slackened with the collision rows on an infeasible step (_relaxed_fallback); the torque box stays HARD.  Counters:
        n_limit_steps (steps that had limit rows), n_limit_slack (steps whose returned x violates a limit row by > 1e-6, i.e. slack used).
        Returns (tau_ego, tau_int, feasible)."""
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
        # Same payload hook for the RIGHT (int) arm, e.g. each hand's share of a box held by both (default None -> unchanged).
        # Added here, so every later use of M_i/c_i (torque box, table rows, fallback, tau = M_i a + c_i) sees the payload.
        if M_int_extra is not None:
            M_i = M_i + M_int_extra
        if bias_int_extra is not None:
            c_i = c_i + bias_int_extra
        self.last_M_e, self.last_c_e, self.last_M_i, self.last_c_i = M_e, c_e, M_i, c_i

        # F5 DELAY COMPENSATION (default off: delay_steps=0 -> skipped entirely).
        # Plan from the state the command will act on: advance (q, qd) over the
        # d in-flight steps with the already-committed accelerations (exact for
        # the double integrator; padded with a=0 at start-up).
        dq_e_pred = dq_i_pred = None
        if self.delay_steps > 0:
            q0_e, qd0_e, q0_i, qd0_i, dq_e_pred, dq_i_pred = self._predict_over_delay(q0_e, qd0_e, q0_i, qd0_i)

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
        if dq_e_pred is not None:
            # F5: first-order barrier value at the predicted state (same constant
            # n, J linearization the clearance rows already use over the horizon).
            for k in range(h0_arr.shape[0]):
                h0_arr[k] = (h0_arr[k] + float((n_arr[k] @ Je_arr[k]) @ dq_e_pred)
                             - float((n_arr[k] @ Ji_arr[k]) @ dq_i_pred))

        # --- ADAPTIVE COOPERATIVE YIELDING (optional, default OFF -- see
        # __init__'s docstring for the full derivation/rationale). Computed
        # here, in plain Python, from the SAME worst (nearest) pair the
        # kernel's clearance rows use below (pairs[0] -- global closest,
        # since `pairs` is already sorted-then-truncated above), BEFORE the
        # numba kernel call, so H_MARGIN_eff/W_pe_eff can be fed straight
        # into it. adaptive_yield=False (default) => u=g=0, H_MARGIN_eff==
        # self.H_MARGIN, W_pe_eff==self.W['pe'] exactly -- byte-identical to
        # the pre-adaptive kernel call. ---
        u = 0.0
        H_MARGIN_eff = self.H_MARGIN
        if self.adaptive_yield and len(pairs) > 0:
            worst = pairs[0]
            h0_worst = float(worst["h"])
            if "hdot" in worst:
                hdot_worst = float(worst["hdot"])
            elif "n" in worst and "v_ego" in worst and "v_int" in worst:
                hdot_worst = float(worst["n"] @ (worst["v_ego"] - worst["v_int"]))
            elif self._prev_worst_h0 is not None:
                hdot_worst = (h0_worst - self._prev_worst_h0) / DT
            else:
                hdot_worst = 0.0
            self._prev_worst_h0 = h0_worst
            denom_h = max(self.H_MARGIN_URGENT - self.H_MARGIN, 1e-9)
            u_h = float(np.clip((self.H_MARGIN_URGENT - h0_worst) / denom_h, 0.0, 1.0))
            u_rate = float(np.clip(max(0.0, -hdot_worst) / max(self.URGENCY_HDOT_REF, 1e-9), 0.0, 1.0))
            u = u_h * u_rate
            H_MARGIN_eff = self.H_MARGIN + u * (self.H_MARGIN_URGENT - self.H_MARGIN)
        self.last_urgency = u
        self.last_H_MARGIN_eff = H_MARGIN_eff

        # ADAPTIVE CUP SHARE gain g = u*(1-tilt_frac) -- needs a tilt-
        # magnitude read on the cup (left) EE, computed the SAME way block
        # (3) below computes its `tilt0` constraint-row anchor (and caches
        # self._cup_axis_local so block (3) does not recapture it); done
        # here only to feed g, block (3) still builds the actual tilt rows.
        g = 0.0
        tilt0_now = None
        ee_body_early = left_ee_body if left_ee_body is not None else self.left_ee_body
        tilt_max_early = tilt_max_rad if tilt_max_rad is not None else self.tilt_max_rad
        if self.adaptive_yield and ee_body_early is not None and tilt_max_early is not None:
            if self._cup_axis_local is None:
                R_ee0 = data.xmat[ee_body_early].reshape(3, 3).copy()
                self._cup_axis_local = R_ee0.T @ np.array([0.0, 0.0, 1.0])
            R_ee = data.xmat[ee_body_early].reshape(3, 3)
            cup_axis_world = R_ee @ self._cup_axis_local
            tilt0_now = np.array([-cup_axis_world[1], cup_axis_world[0]])
        if self.adaptive_yield:
            tilt_frac = 0.0
            if tilt0_now is not None and tilt_max_early:
                tilt_frac = float(np.clip(
                    np.linalg.norm(tilt0_now) / (self.CUP_TILT_SAFE_FRAC * tilt_max_early), 0.0, 1.0))
            g = float(np.clip(u * (1.0 - tilt_frac), 0.0, 1.0))
        self.last_cup_share_gain = g

        W_pe_eff = self.W['pe']
        if self.adaptive_yield and self.W_POS_EGO_MIN is not None:
            W_pe_eff = self.W['pe'] - g * (self.W['pe'] - self.W_POS_EGO_MIN)
        # EGO POSITION-HOLD RELEASE (ego_hold=False): drop the ego (left/cup) arm's
        # position-tracking term entirely (W_pe -> 0), so the arm has NO restoring
        # pull toward q_ego_ref. It then only (a) damps its velocity toward
        # qd_ego_ref (=0 for a hold -> comes to rest wherever it is), (b) avoids
        # collisions, and (c) stays smooth (effort/jerk). Net behavior: the arm
        # holds its current pose until the other arm threatens, moves AWAY to stay
        # safe, and STAYS there -- it never springs back to its initial pose. The
        # cup-arm leash (if configured) still hard-bounds how far it can wander, so
        # "move away" stays "move away SAFELY". Use ego_hold=True during any initial
        # positioning/lift (where q_ego_ref DOES need to drive the arm into place),
        # then ego_hold=False for the reactive phase. NOTE: ego_hold=False FULLY
        # supersedes the adaptive_yield partial W_pe release computed just above
        # (a full release is the stronger, explicit intent of ego_hold=False) --
        # so with ego_hold=False the W_POS_EGO_MIN/cup-share-gain tuning is inert
        # by design. (adaptive_yield is off/deprecated by default anyway.)
        if not ego_hold:
            W_pe_eff = 0.0
        self.last_W_pos_ego_eff = W_pe_eff

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
            W_pe_eff, self.W['ve'], self.W['pi'], self.W['vi'], self.W['r'], self.W['j'],
            H_MARGIN_eff, sc.TAU_MAX)
        m_coll_rows = int(n_arr.shape[0]) * int(self.coll_steps.shape[0])
        tau_lim_e = tau_lim_i = sc.TAU_MAX        # scalar (default) or per-joint vector (F6)
        if self.tau_max_vec is not None:
            # F6: per-joint torque-box RHS (the kernel's rows are rebuilt in place;
            # coefficient rows M_e/M_i are unchanged). Guard a=0 feasibility.
            tau_lim_e = np.maximum(self.tau_max_vec, np.abs(c_e))
            tau_lim_i = np.maximum(self.tau_max_vec, np.abs(c_i))
            if np.any(tau_lim_e > self.tau_max_vec) or np.any(tau_lim_i > self.tau_max_vec):
                self.n_taumax_relaxed += 1
            base = m_coll_rows
            for j in range(NV):
                b_ineq[base + 2 * j] = tau_lim_e[j] - c_e[j]
                b_ineq[base + 2 * j + 1] = tau_lim_e[j] + c_e[j]
                b_ineq[base + 2 * NV + 2 * j] = tau_lim_i[j] - c_i[j]
                b_ineq[base + 2 * NV + 2 * j + 1] = tau_lim_i[j] + c_i[j]

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
            if tilt0_now is not None:
                # already computed above in the ADAPTIVE COOPERATIVE YIELDING
                # block (same ee_body/tilt_max resolution) -- reuse rather
                # than recapturing/recomputing; bit-identical to recomputing
                # it here (tilt0_now is None whenever adaptive_yield=False,
                # so that case falls through to the original computation
                # below unchanged).
                tilt0 = tilt0_now
            else:
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
            Jr_ee = sc.body_jac_full(model, data, ee_body)[1][:, left_dofadr]  # (3,NV) rot Jac at t=0
            if dq_e_pred is not None:
                tilt0 = tilt0 + Jr_ee[0:2] @ dq_e_pred     # F5: tilt at the predicted state
            n_facets = self.cup_cone_facets if cup_cone_facets is None else int(cup_cone_facets)
            if n_facets and n_facets > 0 and self.cone_mode == "true":
                # --- F4 TRUE-TILT CONE: K facets on the geometric tilt ONLY, RHS
                # tilt_max*cos(pi/K) -> INSCRIBED polygon => ||tilt(t)|| <= tilt_max
                # (the per-axis box allows sqrt(2)*tilt_max on the diagonal; the
                # effective-gravity cone below additionally PERMITS tilting into
                # the cup acceleration, a/g rad, and its tangent facets
                # circumscribe the circle). ---
                rhs = tilt_max * np.cos(np.pi / n_facets)
                dirs = [(np.cos(2.0 * np.pi * k / n_facets), np.sin(2.0 * np.pi * k / n_facets))
                        for k in range(n_facets)]
                for t in self.tilt_steps:
                    Aq = self.Phi_q_all[t]
                    drift_t = qd0_e * tvec[t]
                    g0 = Jr_ee[0] @ Aq
                    g1 = Jr_ee[1] @ Aq
                    c0 = tilt0[0] + Jr_ee[0] @ drift_t
                    c1 = tilt0[1] + Jr_ee[1] @ drift_t
                    for (ux, uy) in dirs:
                        row = np.zeros(ntot)
                        row[eslice] = ux * g0 + uy * g1
                        extra_a.append(row)
                        extra_b.append(rhs - (ux * c0 + uy * c1))
            elif n_facets and n_facets > 0:
                # --- EFFECTIVE-GRAVITY CONE (K tangent facets of the L2 cone) ---
                # Spill-free iff the cup axis stays within tilt_max of the
                # EFFECTIVE-gravity "up", up_eff = (a_cup - g)/||a_cup - g||. In
                # the small-angle 2-vector (world x,y) picture:
                #   tau_geom(t) = cup-axis tilt from world-vertical (= the BOX's
                #                 quantity: tilt0 + Jr_ee @ predicted dq), and
                #   tau_grav    = up_eff's tilt from vertical ~ (1/g)*a_cup_xy,
                # so the cup-axis tilt RELATIVE to up_eff is r(t) = tau_geom(t)
                # - tau_grav, and the spill manifold is the CONE ||r(t)|| <=
                # tilt_max. a_cup (cup linear accel) is linear in the FL accel:
                # a_cup ~ Jp_ee @ a_e0 (first-block accel; the immediate command
                # we control -- the J-dot*qd drift and later-block accels are
                # omitted, a deliberate first-order approximation, so tau_grav
                # is a constant-Jacobian linear map of a_e0). The L2 cone
                # ||r|| <= tilt_max is inner-approximated by K linear facets
                # u_k . r(t) <= tilt_max, u_k = (cos, sin)(2*pi*k/K) -- a
                # conservative (inner) polygon, i.e. always safe. Each facet is
                # linear in the ego block of the decision vector:
                #   from tau_geom: sum_ax u_k[ax] * (Jr_ee[ax] @ Phi_q(t))  (all ego blocks)
                #   from -tau_grav: +(1/g) sum_ax u_k[ax] * Jp_ee[ax]       (first-block cols only)
                Jp_ee = sc.body_jac_full(model, data, ee_body)[0][:, left_dofadr]  # (3,NV) lin Jac at t=0
                inv_g = 1.0 / self.G_ACCEL
                dirs = [(np.cos(2.0 * np.pi * k / n_facets), np.sin(2.0 * np.pi * k / n_facets))
                        for k in range(n_facets)]
                for t in self.tilt_steps:
                    Aq = self.Phi_q_all[t]
                    drift_t = qd0_e * tvec[t]
                    for (ux, uy) in dirs:
                        # tau_geom(t) contribution (all ego-block columns via Phi)
                        coef_geom = ux * (Jr_ee[0] @ Aq) + uy * (Jr_ee[1] @ Aq)   # (n_blk,)
                        # -tau_grav contribution: +(1/g)*u . Jp_ee_xy, on the
                        # FIRST-block accel a_e0 (first NV ego columns).
                        grav_first = inv_g * (ux * Jp_ee[0] + uy * Jp_ee[1])      # (NV,)
                        row = np.zeros(ntot)
                        row[eslice] = coef_geom
                        row[0:NV] += grav_first    # eslice starts at 0, so a_e0 == first NV cols
                        const = (ux * (tilt0[0] + Jr_ee[0] @ drift_t)
                                 + uy * (tilt0[1] + Jr_ee[1] @ drift_t))
                        extra_a.append(row)
                        extra_b.append(tilt_max - const)
            else:
                # --- original per-axis small-angle BOX (default, cup_cone_facets=0) ---
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

        # --- (4) CUP-ARM DISPLACEMENT LEASH (optional, plain Python --
        # cheap/default-off; both leash_ee_body and leash_max_m None => no
        # rows added, so existing callers/tests that don't pass these args
        # are bit-for-bit unaffected). Bounds the LEFT EE's predicted
        # Cartesian displacement from the HOLD pose q_ego_ref[0] (=q_lift) to
        # +-leash_max_m per world axis over a few horizon steps -- this is
        # what lets W_POS_EGO be lowered (see FLMPC.__init__'s docstring) so
        # the ego arm actually cooperates with the collision-avoidance/table
        # rows *within* the leash, while still guaranteeing it never wanders
        # arbitrarily far from its hold pose (bounded, not glued). Linear in
        # a_e (ego block columns only): with the exact prediction
        # q_e(t) = qnom_e[t] + Phi_q(t)@a_e, the displacement from hold is
        #   d(t) = Jp_leash @ (q_e(t) - q_ego_ref[0])
        #        = Jp_leash @ (qnom_e[t] - q_ego_ref[0]) + (Jp_leash@Phi_q(t)) @ a_e
        # Jp_leash (3xNV) is the EE's linear Jacobian at t=0 (frozen over the
        # horizon -- same first-order approximation the cup-upright rows
        # above use for Jr_ee). SAME a_rows/b_rows plain-Python convention as
        # (3) above.
        leash_body = leash_ee_body if leash_ee_body is not None else self.leash_ee_body
        leash_max = leash_max_m if leash_max_m is not None else self.leash_max_m
        # ADAPTIVE CUP SHARE (leash half): enlarge the leash toward
        # LEASH_MAX_URGENT_M by the SAME gain `g` used for W_pe_eff above, so
        # the cup arm is allowed to wander further from its hold pose (help
        # avoid) exactly when urgency is high and the cup is not already
        # near its tilt bound. Disabled (leash_max_eff == leash_max exactly)
        # unless adaptive_yield AND LEASH_MAX_URGENT_M is configured.
        leash_max_eff = leash_max
        if self.adaptive_yield and leash_max is not None and self.LEASH_MAX_URGENT_M is not None:
            leash_max_eff = leash_max + g * (self.LEASH_MAX_URGENT_M - leash_max)
        if leash_body is not None and leash_max is not None:
            Jp_leash = sc.body_jac_full(model, data, leash_body)[0][:, left_dofadr]  # (3,NV) at t=0
            hold_e = q_ego_ref[0]
            for t in self.tilt_steps:   # a few horizon steps (block boundaries)
                Aq = self.Phi_q_all[t]
                const_t = Jp_leash @ (qnom_e[t] - hold_e)   # (3,) drift-only displacement at step t
                JA = Jp_leash @ Aq                          # (3, n_blk)
                for ax in range(3):   # world x, y, z
                    row_coef = JA[ax]
                    r_up = np.zeros(ntot); r_up[eslice] = row_coef
                    extra_a.append(r_up); extra_b.append(leash_max_eff - const_t[ax])
                    r_lo = np.zeros(ntot); r_lo[eslice] = -row_coef
                    extra_a.append(r_lo); extra_b.append(leash_max_eff + const_t[ax])

        # --- (5) TABLE FLOOR (optional, plain Python -- cheap/default-off,
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

        # --- (6) F1 EE SPEED CAP on the int/right arm (optional, default off:
        # ee_speed_cap None and v_ee_max None -> no rows, slack_idx stays None).
        # v(t) = Jp_int @ (qd0_i + Phi_qd(t) @ a_int)  (exact FL prediction, Jp
        # frozen at t=0). Rows: +-v_ax(t) <= V (3 axes) and u.v(t) <= V along the
        # current EE motion direction u (tightens the sqrt(3) box corner exactly
        # where the arm is heading).
        slack_idx = None
        v_cap = v_ee_max if v_ee_max is not None else self.ee_speed_cap
        sp_body = int_ee_body if int_ee_body is not None else self.ee_speed_body_int
        if v_cap is not None and sp_body is not None:
            Jp_int = sc.body_jac_full(model, data, sp_body)[0][:, right_dofadr]   # (3,NV)
            v_now = Jp_int @ qd0_i
            v_norm = float(np.linalg.norm(v_now))
            u_dir = v_now / v_norm if v_norm > 1e-3 else None
            row0 = a_ineq.shape[0] + len(extra_a)
            n_sp = 0
            for t in self.speed_cap_steps:
                JA = Jp_int @ self.Phi_qd_all[t]          # (3, n_blk)
                for ax in range(3):
                    r_up = np.zeros(ntot); r_up[islice] = JA[ax]
                    extra_a.append(r_up); extra_b.append(v_cap - v_now[ax])
                    r_lo = np.zeros(ntot); r_lo[islice] = -JA[ax]
                    extra_a.append(r_lo); extra_b.append(v_cap + v_now[ax])
                    n_sp += 2
                if u_dir is not None:
                    r_u = np.zeros(ntot); r_u[islice] = u_dir @ JA
                    extra_a.append(r_u); extra_b.append(v_cap - float(u_dir @ v_now))
                    n_sp += 1
            slack_idx = np.concatenate([np.arange(m_coll_rows), np.arange(row0, row0 + n_sp)]).astype(np.int64)

        # --- (7) JOINT POSITION / SPEED LIMIT rows (11-03, optional; q_lim and qd_lim None -> nothing added, bit-identical) ---
        lim_rows = None
        if q_lim is not None or qd_lim is not None:
            steps = sorted(set([1] + [int(x) for x in self.speed_cap_steps])) if lim_steps is None else [int(x) for x in lim_steps]
            row0 = a_ineq.shape[0] + len(extra_a)
            la, lb = [], []
            for k, (sl, qn, qdv) in enumerate(((eslice, qnom_e, qd0_e), (islice, qnom_i, qd0_i))):
                if q_lim is not None:
                    lo = np.broadcast_to(np.asarray(q_lim[0], float), (2, NV))[k]
                    hi = np.broadcast_to(np.asarray(q_lim[1], float), (2, NV))[k]
                    for t in steps:
                        Aq = self.Phi_q_all[t]
                        for j in range(NV):
                            if np.isfinite(hi[j]):
                                r = np.zeros(ntot); r[sl] = Aq[j]; la.append(r); lb.append(float(hi[j] - qn[t][j]))
                            if np.isfinite(lo[j]):
                                r = np.zeros(ntot); r[sl] = -Aq[j]; la.append(r); lb.append(float(qn[t][j] - lo[j]))
                if qd_lim is not None:
                    vm = np.broadcast_to(np.asarray(qd_lim, float), (2, NV))[k]
                    for t in steps:
                        Ad = self.Phi_qd_all[t]
                        for j in range(NV):
                            if np.isfinite(vm[j]):
                                r = np.zeros(ntot); r[sl] = Ad[j]; la.append(r); lb.append(float(vm[j] - qdv[j]))
                                r = np.zeros(ntot); r[sl] = -Ad[j]; la.append(r); lb.append(float(vm[j] + qdv[j]))
            if la:
                extra_a.extend(la); extra_b.extend(lb)
                lim_rows = (np.stack(la), np.array(lb))
                lim_idx = np.arange(row0, row0 + len(la)).astype(np.int64)
                base_idx = np.arange(m_coll_rows) if slack_idx is None else slack_idx
                slack_idx = np.concatenate([base_idx, lim_idx]).astype(np.int64)
                self.n_limit_steps += 1

        if extra_a:
            a_ineq = np.vstack([a_ineq, np.stack(extra_a)])
            b_ineq = np.concatenate([b_ineq, np.array(extra_b)])
        a_ineq = np.ascontiguousarray(a_ineq)
        b_ineq = np.ascontiguousarray(b_ineq)
        lower = -self.a_box * np.ones(ntot)
        upper = self.a_box * np.ones(ntot)
        x, _, feasible = sc.solve_constrained_box_qp(H, f, lower, upper, a_ineq, b_ineq)
        self.last_feasible = bool(feasible)

        if not feasible:
            # GRACEFUL QP-INFEASIBILITY FALLBACK (see _relaxed_fallback's
            # docstring + the module docstring's 2026-09-08 review note /
            # flmpc-qp-infeasibility-cliff). `x` above is the RAW dual-ascent
            # result on an infeasible problem -- still inside the
            # acceleration box, but NOT torque-box-consistent (the dual
            # ascent pushed it to whatever extreme point best fights the
            # unsatisfiable collision row; tau=M@a+c from that point can
            # wildly exceed TAU_MAX -- this is the fling). Discard it and
            # re-solve the collision-clearance-relaxed problem instead, which
            # keeps the torque box HARD and returns the least-clearance-
            # violating torque within it.
            m_coll = int(n_arr.shape[0]) * int(self.coll_steps.shape[0])
            if slack_idx is None:
                x_relaxed, feasible2 = self._relaxed_fallback(H, f, lower, upper, a_ineq, b_ineq, m_coll)
            else:
                x_relaxed, feasible2 = self._relaxed_fallback(H, f, lower, upper, a_ineq, b_ineq, m_coll,
                                                              slack_idx=slack_idx)
            if feasible2:
                x = x_relaxed
                self.last_fallback = 1
                self.n_relaxed += 1
            else:
                self.last_fallback = 2
                self.n_zero_fallback += 1
                # Ultimate fallback (should essentially never fire -- see
                # _relaxed_fallback's docstring for when it could): command
                # ZERO acceleration. tau = c_e/c_i then is pure
                # gravity/Coriolis/damping compensation, which trivially
                # satisfies the (state-consistent-by-construction) torque
                # box and adds no directional effort at all -- never flings,
                # never amplifies velocity, though it no longer actively
                # improves clearance either.
                x = np.zeros(ntot)

        if lim_rows is not None:
            self.n_limit_fallback += int(not feasible)
            self.n_limit_slack += int(float(np.max(lim_rows[0] @ x - lim_rows[1])) > 1e-6)
        a_e0 = x[0:NV].copy()
        a_i0 = x[n_blk:n_blk + NV].copy()
        if not feasible:
            # FINAL EXACT TORQUE-BOX BACKSTOP (defense in depth). The
            # relaxed-fallback QP above enforces the torque-box rows only to
            # solve_constrained_box_qp's own numerical tolerance (bounded
            # dual-ascent bisection) -- on the larger AUGMENTED problem, at
            # high joint velocity/bias torque this was observed (smoke test)
            # to occasionally clear TAU_MAX by a fraction of a percent.
            # "Never exceed the torque box" is this fallback's top priority,
            # so exactly clamp with a closed-form scale-toward-zero:
            # tau(lambda) = c + lambda*(M@a0) is AFFINE in lambda, and
            # lambda=0 (a0=0, tau=c) is, by construction, the torque-box-
            # feasible point every physically-reachable state admits -- so
            # the largest lambda in [0,1] keeping every joint's
            # |tau(lambda)|<=TAU_MAX is exact and cheap (no re-solve). This
            # can only SHRINK the fallback's commanded acceleration toward
            # zero (never redirect/amplify it), so it tightens the torque
            # guarantee to exact without undoing the fallback's
            # "improve clearance, don't fling" property.
            _a_e_in, _a_i_in = a_e0, a_i0                      # 11-03 counter only (no logic change)
            a_e0 = self._torque_safe_scale(M_e, c_e, a_e0, tau_lim_e)
            a_i0 = self._torque_safe_scale(M_i, c_i, a_i0, tau_lim_i)
            if not (np.array_equal(a_e0, _a_e_in) and np.array_equal(a_i0, _a_i_in)):
                self.n_scale_shrunk += 1
        else:
            self.last_fallback = 0
        self.a_ego_prev = a_e0.copy()
        self.a_int_prev = a_i0.copy()
        if self.delay_steps > 0:
            # F5: this command is now in flight (callers may overwrite it with the
            # accel the plant will actually see, e.g. post torque-filter).
            self._a_queue.append((a_e0.copy(), a_i0.copy()))
            if len(self._a_queue) > self.delay_steps:
                self._a_queue.pop(0)
        # realize the feedback-linearizing torque
        tau_e = M_e @ a_e0 + c_e
        tau_i = M_i @ a_i0 + c_i
        return tau_e, tau_i, bool(feasible)

    # ---------------- F5 delay-compensation helpers ----------------------
    def _predict_over_delay(self, q_e, qd_e, q_i, qd_i):
        """Advance both arms' (q, qd) over `delay_steps` in-flight control steps
        with the committed accelerations (oldest first; missing entries at
        start-up = 0). Semi-implicit Euler, the same integrator di_coeffs and
        MuJoCo use -> exact for the FL double integrator."""
        d = self.delay_steps
        queue = [(np.zeros(self.NV), np.zeros(self.NV))] * (d - len(self._a_queue)) + list(self._a_queue)
        qe, qde, qi, qdi = q_e.copy(), qd_e.copy(), q_i.copy(), qd_i.copy()
        for a_e, a_i in queue:
            qde = qde + a_e * self.DT
            qe = qe + qde * self.DT
            qdi = qdi + a_i * self.DT
            qi = qi + qdi * self.DT
        return qe, qde, qi, qdi, qe - q_e, qi - q_i

    def replace_last_applied(self, a_e, a_i):
        """F5: overwrite the newest in-flight entry with the accel the plant will
        actually receive (e.g. after the caller's torque filter / clip)."""
        if self.delay_steps > 0 and self._a_queue:
            self._a_queue[-1] = (np.asarray(a_e, float).copy(), np.asarray(a_i, float).copy())

    def push_applied(self, a_e, a_i):
        """F5: record a command issued by ANOTHER controller (e.g. the demo's
        lift branch) so the in-flight queue is correct when FL-MPC takes over."""
        if self.delay_steps > 0:
            self._a_queue.append((np.asarray(a_e, float).copy(), np.asarray(a_i, float).copy()))
            if len(self._a_queue) > self.delay_steps:
                self._a_queue.pop(0)

    @staticmethod
    def _torque_safe_scale(M_arm, c_arm, a0, tau_max):
        """Largest lambda in [0,1] such that |c_arm + lambda*(M_arm@a0)| <=
        tau_max componentwise -- exact closed form (tau(lambda) is affine in
        lambda; lambda=0 gives tau=c_arm, the pure gravity/Coriolis/damping
        hold torque, torque-box-feasible for any physically-reachable state).
        Returns lambda*a0 (a scaled-down copy of a0 -- never a different
        direction, so it cannot flip the fallback's clearance-improving
        acceleration into a clearance-worsening one, only shrink its
        magnitude toward the always-safe a=0 point)."""
        delta = M_arm @ a0
        lam = 1.0
        vec = np.ndim(tau_max) > 0          # F6: per-joint limits (scalar path unchanged)
        for j in range(delta.shape[0]):
            d = float(delta[j])
            if d == 0.0:
                continue
            tm = float(tau_max[j]) if vec else tau_max
            lam_j = ((tm - c_arm[j]) / d) if d > 0.0 else ((-tm - c_arm[j]) / d)
            if lam_j < lam:
                lam = lam_j
        lam = max(0.0, min(1.0, lam))
        return lam * a0
