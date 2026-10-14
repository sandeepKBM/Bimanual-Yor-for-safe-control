"""v7: fixes the two remaining v6 weaknesses (see RESULTS_v6.md Sec.6):
  (1) chatter (32.2 hdot sign-changes/s, worst of A/v4-C/v6) -- caused by the
      collision HOCBF being checked only at NC=4 block-boundary steps of the
      160ms predicted horizon, not every physics step, so the barrier could
      dip (and the gradient direction flip) between checks and between
      consecutive replans.
  (2) hand-tuned, constant priority asymmetry (REG_EFFORT_EGO=100x INT,
      W_JERK_EGO=8x INT, fixed W_EE_HOLD) -- not derived from any measured
      grasp state, just single hand-picked values (RESULTS_v6.md Sec.7).

FIX 1 -- collision constraint at every physics step:
  `_coll_steps()` now returns ALL of 1..NP (every one of the 32 predicted
  physics steps), not just the NC=4 block-end steps. `_assemble`'s row-count
  m grows (m_coll = K_PAIRS*NP = 160 vs v6's K_PAIRS*NC = 20) but its BODY is
  unchanged -- it already loops generically over however many `coll_steps`
  it is given. This removes the "gap between checks" chatter mechanism
  identified in RESULTS_v6.md by construction, at a measured solve-time
  cost (see RESULTS_v7.md). K_PAIRS is reduced from 5 to 3 to keep the row
  count (and therefore constrained_box_qp's solve time, which scales with m)
  tractable -- documented as a real trade (fewer simultaneous nearest pairs
  tracked, same collision-avoidance mechanism).

FIX 2 -- grasp-state-scaled priority (no magic constant):
  A scalar GRASP FRAGILITY in [0,1] is computed each step from measured
  grasp/contact state (`grasp_fragility()` below), combining:
    (a) slip margin: max, over the pad7/box and pad8/box contacts, of
        ||tangential force|| / (MU_GRASP * normal force) -- how close the
        real grasp is to Coulomb slip, MU_GRASP=1.0 taken directly from the
        scene's own pad/box contact friction spec
        (scenes/scene_yor_canonical.xml geom friction="1.0 0.01 0.0001", the
        first coefficient IS the sliding-friction mu MuJoCo uses for
        condim=4 contacts) -- NOT a tuned value.
        If no grasp contact exists this step, slip margin defaults to 1.0
        (maximally cautious: the grasp already has zero measured margin).
    (b) drift margin: box_drift_trans / EE_SLACK_POS and
        box_drift_rot / EE_SLACK_ROT -- how far the box has ACTUALLY moved
        relative to the gripper (measured from MuJoCo state, same quantity
        already used as the metric in RESULTS_v4-v6) relative to the
        existing hard slack cap.
  fragility = clip(max(slip_ratio, drift_pos_ratio, drift_rot_ratio), 0, 1).
  Priority weights then INTERPOLATE from a SYMMETRIC baseline (ego ==
  intruder, zero built-in authority) at fragility=0 up to v6's own
  validated ceiling values at fragility=1:
    REG_EFFORT_EGO(t) = REG_BASE * (1 + 99*fragility)   -> 5e-5 .. 5e-3
    W_JERK_EGO(t)      = W_JERK_INT_BASE*(1 + 7*fragility) -> 0.01 .. 0.08
    W_EE_HOLD(t)        = W_EE_HOLD_MAX * fragility        -> 0 .. 4e4
  i.e. the ego earns right-of-way IN PROPORTION TO how close the grasp
  actually is to slipping/exceeding its cap, not by a fixed multiplier
  applied regardless of state. The GAINS (99, 7) are chosen only so the
  fragility=1 ceiling reproduces v6's already-validated endpoint values (so
  the new rule is a strict generalization of v6's constants, not a
  re-tuning); the intruder's own weights (REG_EFFORT_INT, W_JERK_INT) stay
  fixed at the same BASE the ego relaxes to when fragility=0. The hard
  EE_SLACK_POS/ROT cap is UNCHANGED and always active regardless of
  fragility -- fragility only shapes the SOFT pre-emptive cost, never the
  hard safety cap.
"""
from __future__ import annotations

import time
import numpy as np
import mujoco
from numba import njit

import sim_common as sc

# ---------------------------------------------------------------------------
# Horizon sizing (same structure as mpc6.py; K_PAIRS reduced, see FIX 1).
# ---------------------------------------------------------------------------
NC = 2
L_PER_BLOCK = 8
NP = NC * L_PER_BLOCK    # FIX 4c: 16 -> 80ms look-ahead @ DT=0.005. Shortened from 32/160ms:
                          # with the now-EXACT mj_step nominal, the linear control sensitivity is
                          # accurate only on the near horizon (open-loop response ~doubles every 40ms;
                          # measured err 36 mrad@20ms -> 811@80ms -> 3409@160ms). 160ms summed 3.4-rad
                          # garbage into the QP cost/constraints and wrecked the plan (v7c collided);
                          # 80ms keeps the linearization valid while retaining real foresight over the
                          # crossing. NC=2 control blocks (2x8x5ms).
K_PAIRS = 3               # reduced from v6's 5 -- see FIX 1 docstring


def _coll_steps():
    """FIX 1: every one of the NP predicted physics steps, not just the NC
    block-boundary steps (v6's `_coll_steps` returned
    np.arange(L_PER_BLOCK, NP+1, L_PER_BLOCK), i.e. only 4 of 32)."""
    return np.arange(1, NP + 1, dtype=np.int64)


EE_SLACK_POS = 0.05      # [m] hard grasp cap (unchanged from v5/v6)
EE_SLACK_ROT = 0.2       # [rad, small-angle]
DUAL_SWEEPS = 2
DUAL_ROOT_ITERS = 5

# ---- objective weights (unchanged from v6) ----
W_POS_EGO = 3000.0
W_VEL_EGO = 15.0
W_POS_INT = 600.0
W_VEL_INT = 8.0

# ---- FIX 2: grasp-fragility-scaled priority ----
# Symmetric BASE (what ego relaxes to at fragility=0) == intruder's fixed
# values (v6's own REG_EFFORT_INT / W_JERK_INT, unchanged).
REG_EFFORT_BASE = 5.0e-5
REG_EFFORT_INT = 5.0e-5
W_JERK_BASE = 0.01
W_JERK_INT = 0.01
W_EE_HOLD_MAX = 4.0e4
# Gains chosen so fragility=1 reproduces v6's validated ceiling exactly:
#   REG_EFFORT_BASE*(1+GAIN_REG) = 5e-3 (v6's REG_EFFORT_EGO)  -> GAIN_REG=99
#   W_JERK_BASE*(1+GAIN_JERK)    = 0.08 (v6's W_JERK_EGO)      -> GAIN_JERK=7
GAIN_REG = 99.0
GAIN_JERK = 7.0

MU_GRASP = 1.0  # scene_yor_canonical.xml pad/box contact friction (first coeff)

NV = 6


def grasp_fragility(model, data, box_geom, pad7, pad8, box_drift_trans, box_drift_rot,
                     mujoco_mod):
    """FIX 2: scalar in [0,1] from measured grasp/contact state. See module
    docstring for the combination rule."""
    slip_ratio = 0.0
    found_contact = False
    for i in range(data.ncon):
        con = data.contact[i]
        geoms = {con.geom1, con.geom2}
        if box_geom in geoms and (pad7 in geoms or pad8 in geoms):
            found_contact = True
            force = np.zeros(6)
            mujoco_mod.mj_contactForce(model, data, i, force)
            normal = abs(float(force[0]))
            tangent = float(np.hypot(force[1], force[2]))
            ratio = tangent / (MU_GRASP * normal + 1e-6)
            slip_ratio = max(slip_ratio, ratio)
    if not found_contact:
        slip_ratio = 1.0  # no measured grasp contact -> zero measured margin
    drift_pos_ratio = box_drift_trans / EE_SLACK_POS
    drift_rot_ratio = box_drift_rot / EE_SLACK_ROT
    frag = max(slip_ratio, drift_pos_ratio, drift_rot_ratio)
    return float(np.clip(frag, 0.0, 1.0))


# =============================================================================
# Numba kernels.
#
# FIX 4 (2026-09-07, PREDICTION-MODEL FIX -- time-varying linearization):
# `_sensitivity_core` used to predict the whole NP=32-step (160ms) horizon
# with a SINGLE, FROZEN inverse mass matrix `Minv` and bias `bias` captured
# once at t=0 (qacc_free = Minv @ (-bias), Euler-integrated for all NP
# steps; the control->state sensitivity Phi_q/Phi_qd was also built with
# that same constant Minv in every block). Since M(q) and qfrc_bias(q, qd)
# are configuration-dependent and the arm swings substantially over 160ms
# during a crossing, this frozen-Minv nominal/sensitivity diverges from the
# true dynamics and the MPC mis-plans.
#
# Because this njit kernel cannot call MuJoCo, the nominal rollout is now
# done in PYTHON (JointMPCv7._rollout_arm, using a scratch MjData that is
# forwarded to each predicted (q_t, qd_t) to read the REAL M(q_t)/
# qfrc_bias(q_t, qd_t) along the nominal trajectory) and the resulting
# per-step qf_nom/qdf_nom/Minv_t arrays are passed in here. This kernel no
# longer computes the nominal itself -- it only builds Phi_q/Phi_qd, using
# Minv_t[t] (the REAL per-step d(qacc)/d(tau) at horizon step t) in place of
# the old constant `Minv` in the block loop. Everything else (the
# constant-acceleration-style Phi propagation within a control block) is
# unchanged from mpc6.py/the original v7.
# =============================================================================

@njit(cache=True)
def _sensitivity_core(qf_nom, qdf_nom, Minv_t, x_nom, NC_, NV_, NP_, L_, DT):
    """qf_nom, qdf_nom: (NP_+1, NV_) nominal trajectory rolled under the
    WARM-START control x_nom (see JointMPCv7._rollout_arm) with a per-step
    MuJoCo M(q_t)/bias(q_t,qd_t) rollout -- index 0 is (q0, qd0). Minv_t:
    (NP_, NV_, NV_) per-step inv(M(q_t)) along THAT controlled trajectory,
    used as the control->accel sensitivity at horizon step t.

    FIX 4b (control-aware linearization): the downstream QP models the state
    as `qf_eff[t] + Phi[t] @ x` where x is the ABSOLUTE per-block torque. We
    linearize about x_nom (the previous solution), so the true model is
    `state(x) = qf_nom_ctrl[t] + Phi[t] @ (x - x_nom)`. Folding the constant
    `-Phi[t] @ x_nom` into the nominal gives `qf_eff = qf_nom_ctrl - Phi@x_nom`,
    which lets _assemble stay UNCHANGED (it still does qf_eff + Phi@x). x_nom
    is the per-block-tiled warm-start torque, shape (NC_*NV_,). Rolling about
    the controlled trajectory (not free-fall) is what makes the time-varying
    M(q_t) actually improve prediction accuracy -- see RESULTS/review notes."""
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
                qacc = Minv_t[t]
            else:
                qacc = np.zeros((NV_, NV_))
            q_new = q_local + qd_local * DT + 0.5 * qacc * DT * DT
            qd_new = qd_local + qacc * DT
            Phi_q[t + 1, :, b * NV_:(b + 1) * NV_] = q_new
            Phi_qd[t + 1, :, b * NV_:(b + 1) * NV_] = qd_new
            q_local = q_new
            qd_local = qd_new
    # FIX 4b: shift nominal by -Phi@x_nom so state = qf_eff + Phi@x is exact
    # for the linearization about x_nom.
    qf_eff = qf_nom.copy()
    qdf_eff = qdf_nom.copy()
    for t in range(NP_ + 1):
        qf_eff[t] = qf_nom[t] - Phi_q[t] @ x_nom
        qdf_eff[t] = qdf_nom[t] - Phi_qd[t] @ x_nom
    return qf_eff, qdf_eff, Phi_q, Phi_qd


@njit(cache=True)
def _assemble(qf_nom_e, qdf_nom_e, Minv_t_e,
              qf_nom_i, qdf_nom_i, Minv_t_i,
              q0_left, q_int_des_traj, qd_int_des_traj,
              tau_ego_prev, tau_int_prev,
              Jp_ee, Jr_ee,
              n_arr, Je_arr, Ji_arr, h0_arr, coll_steps,
              NC_, NV_, NP_, L_, K_, DT,
              W_POS_EGO_, W_VEL_EGO_, W_POS_INT_, W_VEL_INT_,
              W_JERK_EGO_, W_JERK_INT_, W_EE_HOLD_,
              REG_EFFORT_EGO_, REG_EFFORT_INT_,
              EE_SLACK_POS_, EE_SLACK_ROT_):
    n_e = NC_ * NV_
    n_i = NC_ * NV_
    ntot = n_e + n_i

    q0_e = qf_nom_e[0]
    q0_i = qf_nom_i[0]

    # FIX 4b: warm-start control tiled per block -- the linearization center.
    # The nominal was rolled under this same torque in _rollout_arm.
    x_nom_e = np.zeros(n_e)
    x_nom_i = np.zeros(n_i)
    for b in range(NC_):
        x_nom_e[b * NV_:(b + 1) * NV_] = tau_ego_prev
        x_nom_i[b * NV_:(b + 1) * NV_] = tau_int_prev

    qf_e, qdf_e, Phq_e, Phqd_e = _sensitivity_core(qf_nom_e, qdf_nom_e, Minv_t_e, x_nom_e, NC_, NV_, NP_, L_, DT)
    qf_i, qdf_i, Phq_i, Phqd_i = _sensitivity_core(qf_nom_i, qdf_nom_i, Minv_t_i, x_nom_i, NC_, NV_, NP_, L_, DT)

    H = np.zeros((ntot, ntot))
    f = np.zeros(ntot)

    for j in range(n_e):
        H[j, j] += 2.0 * REG_EFFORT_EGO_
    for j in range(n_i):
        H[n_e + j, n_e + j] += 2.0 * REG_EFFORT_INT_

    for t in range(1, NP_ + 1):
        Ae = Phq_e[t]
        be = q0_left - qf_e[t]
        AeT = Ae.T
        H[0:n_e, 0:n_e] += 2.0 * W_POS_EGO_ * (AeT @ Ae)
        f[0:n_e] += -2.0 * W_POS_EGO_ * (AeT @ be)

        Aev = Phqd_e[t]
        bev = -qdf_e[t]
        AevT = Aev.T
        H[0:n_e, 0:n_e] += 2.0 * W_VEL_EGO_ * (AevT @ Aev)
        f[0:n_e] += -2.0 * W_VEL_EGO_ * (AevT @ bev)

        Aee = Jp_ee @ Phq_e[t]
        bee = -(Jp_ee @ (qf_e[t] - q0_left))
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

    NCS = coll_steps.shape[0]
    m_coll = K_ * NCS
    m_grasp = 2 * 2 * 3 * 2
    m = m_coll + m_grasp
    a_ineq = np.zeros((m, ntot))
    b_ineq = np.zeros(m)

    row = 0
    for k in range(K_):
        c_e = n_arr[k] @ Je_arr[k]
        c_i = n_arr[k] @ Ji_arr[k]
        h0 = h0_arr[k]
        for si in range(NCS):
            t = coll_steps[si]
            row_e = -(c_e @ Phq_e[t])
            row_i = c_i @ Phq_i[t]
            a_ineq[row, 0:n_e] = row_e
            a_ineq[row, n_e:ntot] = row_i
            b_ineq[row] = h0 + c_e @ (qf_e[t] - q0_e) - c_i @ (qf_i[t] - q0_i)
            row += 1

    stages = (NP_ // 2, NP_)
    for si in range(2):
        t = stages[si]
        for which in range(2):
            J_ee = Jp_ee if which == 0 else Jr_ee
            slack = EE_SLACK_POS_ if which == 0 else EE_SLACK_ROT_
            d_free = J_ee @ (qf_e[t] - q0_left)
            Phi_ee = J_ee @ Phq_e[t]
            for i in range(3):
                a_ineq[row, 0:n_e] = Phi_ee[i]
                b_ineq[row] = slack - d_free[i]
                row += 1
                a_ineq[row, 0:n_e] = -Phi_ee[i]
                b_ineq[row] = slack + d_free[i]
                row += 1

    return H, f, a_ineq, b_ineq


class JointMPCv7:
    """v7 = v6 + FIX 1 (collision checked every physics step) + FIX 2
    (grasp-fragility-scaled priority, no magic constant).

    FIX 3 (2026-09-07, opt-in, byte-identical when off) -- two extra families
    of LINEAR inequality rows are appended (in plain Python) to the assembled
    a_ineq/b_ineq just before the box-QP solve, so the njit `_assemble` kernel
    is untouched:
      (a) EE-SPEED LIMIT (v_ee_max, ee_body_id_int): a per-step Cartesian
          velocity cap on the INTRUDER arm's end-effector. The intruder's
          applied torque this step is x[n_e:n_e+NV]; the resulting next-step
          EE linear velocity is v_next = Jp_int @ (qd_int0 + DT*Minv_rr*(tau
          - bias_r)), affine in tau. We bound |v_next_axis| <= v_ee_max per
          world axis (3 axes x 2 signs = 6 rows). This is what actually
          "holds the EE speed" -- a hard constraint the reactive HOCBF-QP
          never had; the horizon cost still shapes the *approach*, this caps
          the *magnitude*. v_ee_max=None -> no rows (off).
      (b) TABLE FLOOR (table_rows): pre-built per-arm HOCBF rows from
          sim_common.table_plane_barrier_rows, passed as
          (a_ego(m_e,NV), b_ego, a_int(m_i,NV), b_int) in each arm's own
          `a @ tau <= b` convention, embedded into the ego (cols 0:NV) and
          intruder (cols n_e:n_e+NV) FIRST-block torque -- the same
          single-step guarantee on the applied action the reactive v4c QP
          gives. table_rows=None -> no rows (off).
    Both are enforced on the FIRST control block (the torque actually applied
    this step), matching how the reactive filter enforces its own rows."""

    def __init__(self, model, ctrl, q0_left, v_ee_max=None, ee_body_id_int=None):
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
        self.last_fragility = 0.0
        # FIX 3 (opt-in). Off by default -> byte-identical to the validated v7.
        self.v_ee_max = v_ee_max
        self.ee_body_id_int = ee_body_id_int
        # FIX 4: scratch MjData reused every step for the time-varying nominal
        # rollout (see _rollout_arm) -- lazily created on the first step()
        # call against whatever `model` is actually passed in.
        self._scratch = None
        self.last_rollout_time = 0.0

    def _rollout_arm(self, model, data, qposadr, dofadr, q0, qd0, M_extra, bias_extra, tau_nom):
        """FIX 4 -- PYTHON-side time-varying-linearization nominal rollout for
        ONE arm (called once for ego/left, once for intruder/right).

        Rolls the FREE (uncontrolled) nominal trajectory forward NP steps
        starting from (q0, qd0): at each step t, this arm's joints are set on
        a scratch MjData to (q_t, qd_t) -- the REST of the model (the other
        arm, the box/payload freejoint, everything else) is left at whatever
        state the real `data` currently has -- then mujoco.mj_forward is run
        and the REAL M(q_t) (that arm's own dofadr submatrix of the full mass
        matrix, via sim_common.full_mass_matrix, the same index pattern
        step() already uses for M_ll/M_rr) and REAL qfrc_bias(q_t, qd_t) are
        read off. `M_extra`/`bias_extra` (e.g. the grasped-payload reflection
        for the ego arm) are added on top, HELD CONSTANT at their t=0 value
        for the whole rollout -- see the payload-handling note in `step()`.
        qacc_t = solve(M_t, -bias_t); semi-implicit Euler integration
        (qd_{t+1} = qd_t + qacc_t*DT; q_{t+1} = q_t + qd_{t+1}*DT), matching
        MuJoCo's own default Euler integrator (scene_yor_canonical.xml sets
        no <option integrator=.../>, i.e. keeps MuJoCo's semi-implicit-Euler
        default) more closely than the old constant-acceleration formula.

        Returns (qf_nom (NP+1, NV), qdf_nom (NP+1, NV), Minv_t (NP, NV, NV))
        -- qf_nom[0]/qdf_nom[0] = q0/qd0, Minv_t[t] = inv(M_t) used to step
        from t to t+1 -- fed straight into `_sensitivity_core`.
        """
        # FIX 4c: roll the nominal with MuJoCo's OWN stepper (mj_step) applying
        # the warm-start torque tau_nom to this arm -- the TRUE evolution,
        # including passive (joint damping/spring), constraint (the grasped-cup
        # weld for the ego arm) and Coriolis/gravity forces that a hand-rolled
        # M^-1(tau-bias) Euler omits (that omission left even a free-fall
        # nominal ~700 mrad off from the real trajectory; see review notes).
        # M_t (arm submatrix + M_extra payload reflection) is still extracted
        # along this REAL trajectory for the control->accel sensitivity.
        scratch = mujoco.MjData(model)   # fresh each call; mj_step mutates the whole model state
        mujoco.mj_copyData(scratch, model, data)
        NVn = len(dofadr)
        qf = np.zeros((NP + 1, NVn))
        qdf = np.zeros((NP + 1, NVn))
        Minv_t = np.zeros((NP, NVn, NVn))
        qf[0] = q0
        qdf[0] = qd0
        for t in range(NP):
            mujoco.mj_forward(model, scratch)                    # ensure qM/derived valid at state t
            M_t = sc.full_mass_matrix(model, scratch)[np.ix_(dofadr, dofadr)] + M_extra
            Minv_t[t] = np.linalg.inv(M_t)
            scratch.qfrc_applied[:] = 0.0
            scratch.qfrc_applied[dofadr] = tau_nom               # apply the warm-start torque to THIS arm
            mujoco.mj_step(model, scratch)                        # TRUE nonlinear step t -> t+1
            qf[t + 1] = scratch.qpos[qposadr].copy()
            qdf[t + 1] = scratch.qvel[dofadr].copy()
        return qf, qdf, Minv_t

    def step(self, model, data, q_int_des_traj, qd_int_des_traj, pairs_all, fragility,
             *, table_rows=None):
        t_start = time.time()
        ctrl = self.ctrl
        self.last_fragility = float(fragility)

        # FIX 2: interpolate priority weights from the symmetric base to
        # v6's validated ceiling, in proportion to measured fragility.
        reg_effort_ego = REG_EFFORT_BASE * (1.0 + GAIN_REG * fragility)
        w_jerk_ego = W_JERK_BASE * (1.0 + GAIN_JERK * fragility)
        w_ee_hold = W_EE_HOLD_MAX * fragility

        q_ego0 = data.qpos[ctrl.left_qposadr].copy()
        qd_ego0 = data.qvel[ctrl.left_dofadr].copy()
        q_int0 = data.qpos[ctrl.right_qposadr].copy()
        qd_int0 = data.qvel[ctrl.right_dofadr].copy()

        # t=0 values, kept for: (a) FIX 3's EE-speed-cap row (Minv_rr, bias_r,
        # a single-step-ahead affine bound, not a horizon prediction -- out of
        # FIX 4's scope) and (b) the ego's payload reflection, which FIX 4
        # holds CONSTANT at its t=0 value through the rollout (see
        # _rollout_arm's docstring and the module-level FIX 4 note): the
        # grasped payload's mass/inertia don't change, and recomputing its
        # r-offset/Jacobian along the rollout would need the box's OWN
        # freejoint qpos to be propagated too (it isn't -- it's held at the
        # real `data` value in the scratch rollout, exactly like the rest of
        # the non-arm state), which would make a moving-box r-offset *less*
        # accurate, not more, over a stale frozen box position.
        M_full = sc.full_mass_matrix(model, data)
        M_rr = M_full[np.ix_(ctrl.right_dofadr, ctrl.right_dofadr)]
        bias_r = data.qfrc_bias[ctrl.right_dofadr].copy()
        Minv_rr = np.linalg.inv(M_rr)
        M_box, bias_box = ctrl.payload_terms(data)

        Jp_full, Jr_full = sc.body_jac_full(model, data, ctrl.ee_body_id)
        Jp_ee = np.ascontiguousarray(Jp_full[:, ctrl.left_dofadr])
        Jr_ee = np.ascontiguousarray(Jr_full[:, ctrl.left_dofadr])

        pairs_sorted = sorted(pairs_all, key=lambda p: p["h"])[:K_PAIRS]
        n_arr = np.stack([p["n"] for p in pairs_sorted])
        Je_arr = np.stack([p["Je"] for p in pairs_sorted])
        Ji_arr = np.stack([p["Ji"] for p in pairs_sorted])
        h0_arr = np.array([p["h"] for p in pairs_sorted])
        coll_steps = _coll_steps()

        # FIX 4: PYTHON-side time-varying-linearization nominal rollout, one
        # per arm, using the REAL MuJoCo M(q_t)/qfrc_bias(q_t, qd_t) along
        # the predicted nominal trajectory (see _rollout_arm) instead of a
        # single frozen t=0 Minv/bias for the whole 160ms horizon.
        t_rollout0 = time.time()
        qf_nom_e, qdf_nom_e, Minv_t_e = self._rollout_arm(
            model, data, ctrl.left_qposadr, ctrl.left_dofadr,
            q_ego0, qd_ego0, M_box, bias_box, self.tau_ego_prev)
        qf_nom_i, qdf_nom_i, Minv_t_i = self._rollout_arm(
            model, data, ctrl.right_qposadr, ctrl.right_dofadr,
            q_int0, qd_int0, np.zeros((NV, NV)), np.zeros(NV), self.tau_int_prev)
        self.last_rollout_time = time.time() - t_rollout0

        t_assemble0 = time.time()
        H, f, a_ineq, b_ineq = _assemble(
            qf_nom_e, qdf_nom_e, Minv_t_e,
            qf_nom_i, qdf_nom_i, Minv_t_i,
            self.q0_left, q_int_des_traj, qd_int_des_traj,
            self.tau_ego_prev, self.tau_int_prev,
            Jp_ee, Jr_ee,
            n_arr, Je_arr, Ji_arr, h0_arr, coll_steps,
            NC, NV, NP, L_PER_BLOCK, K_PAIRS, sc.DT,
            W_POS_EGO, W_VEL_EGO, W_POS_INT, W_VEL_INT,
            w_jerk_ego, W_JERK_INT, w_ee_hold,
            reg_effort_ego, REG_EFFORT_INT,
            EE_SLACK_POS, EE_SLACK_ROT)
        self.last_assemble_time = time.time() - t_assemble0

        # -- FIX 3: append opt-in EE-speed-limit + table-floor rows (Python;
        # njit kernel above untouched). Both act on the FIRST-block torque
        # (ego = cols 0:NV, intruder = cols n_e:n_e+NV). --
        extra_a, extra_b = [], []
        if self.v_ee_max is not None and self.ee_body_id_int is not None:
            import ee_speed_limit as _esl
            Jp_int_full, _ = sc.body_jac_full(model, data, self.ee_body_id_int)
            Jp_int = Jp_int_full[:, ctrl.right_dofadr]           # (3, NV) linear EE Jac
            a_vel, b_vel = _esl.ee_velocity_cap_rows(Jp_int, qd_int0, Minv_rr, bias_r,
                                                      self.v_ee_max, sc.DT)
            for r in range(a_vel.shape[0]):
                row = np.zeros(self.ntot); row[self.n_e:self.n_e + NV] = a_vel[r]
                extra_a.append(row); extra_b.append(float(b_vel[r]))
        if table_rows is not None:
            a_ego_t, b_ego_t, a_int_t, b_int_t = table_rows
            for r in range(a_ego_t.shape[0]):
                row = np.zeros(self.ntot); row[0:NV] = a_ego_t[r]
                extra_a.append(row); extra_b.append(float(b_ego_t[r]))
            for r in range(a_int_t.shape[0]):
                row = np.zeros(self.ntot); row[self.n_e:self.n_e + NV] = a_int_t[r]
                extra_a.append(row); extra_b.append(float(b_int_t[r]))
        if extra_a:
            a_ineq = np.ascontiguousarray(np.vstack([a_ineq, np.stack(extra_a)]))
            b_ineq = np.ascontiguousarray(np.concatenate([b_ineq, np.array(extra_b)]))
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
    """JIT-warms _sensitivity_core/_assemble with FIX 4's new (per-step
    nominal + per-step Minv_t) argument shapes -- dummy/frozen arrays here
    (a real MuJoCo model isn't available to `warmup()`), just to trigger
    numba compilation; JointMPCv7.step() itself always feeds real per-step
    MuJoCo-rollout arrays via _rollout_arm."""
    NVt = NV
    q0 = np.zeros(NVt)
    qd0 = np.zeros(NVt)
    Minv = np.eye(NVt)
    qf_nom = np.zeros((NP + 1, NVt))
    qdf_nom = np.zeros((NP + 1, NVt))
    Minv_t = np.stack([Minv] * NP)
    x_nom = np.zeros(NC * NVt)
    _sensitivity_core(qf_nom, qdf_nom, Minv_t, x_nom, NC, NVt, NP, L_PER_BLOCK, sc.DT)
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
    _assemble(qf_nom, qdf_nom, Minv_t, qf_nom, qdf_nom, Minv_t,
              q0, q_traj, qd_traj, np.zeros(NVt), np.zeros(NVt),
              Jp_ee, Jr_ee, n_arr, Je_arr, Ji_arr, h0_arr, coll_steps,
              NC, NVt, NP, L_PER_BLOCK, K_PAIRS, sc.DT,
              W_POS_EGO, W_VEL_EGO, W_POS_INT, W_VEL_INT,
              W_JERK_BASE, W_JERK_INT, W_EE_HOLD_MAX,
              REG_EFFORT_BASE, REG_EFFORT_INT,
              EE_SLACK_POS, EE_SLACK_ROT)
