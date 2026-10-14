"""Decentralized worst-case feedback-linearized MPC ("Option B").

This is a DECENTRALIZED variant of mpc_fl.FLMPC. The flagship FLMPC solves ONE
joint QP over both arms' acceleration profiles a = [a_left; a_right], so the
shared arm-arm clearance rows make both arms cooperatively yield. Here, instead,
EACH arm solves its OWN QP over only its own acceleration, treating the OTHER
arm as a bounded WORST-CASE disturbance (the Hamilton-Jacobi-Isaacs assumption
NeHMO/HaMMaR make) -- but using the SAME anticipatory-HOCBF machinery as FLMPC
(the analytic linear prediction q(t)=q0+qd0*t+Phi(t)*a and the clearance row
h(t)=dist-(r_e+r_i+D_MIN)), NOT a learned certificate. No comms, no shared plan:
each arm knows only the other's CURRENT state (position+velocity => nominal
drift) plus a robust "tube" bounding how far the other arm could move against it.

REUSE: this subclasses mpc_fl.FLMPC and reuses, verbatim,
  * FLMPC.__init__            (precomputed Phi_q_all / Phi_qd_all tensors,
                               di_coeffs, weights, warmup, all config)
  * mpc_fl._assemble_fl       (the numba kernel that builds the joint H, f,
                               a_ineq, b_ineq -- cost blocks are already
                               block-diagonal between the two arms and only the
                               collision rows couple them, so we slice the
                               per-arm cost/collision/torque blocks straight out
                               of the joint assembly and then DROP the other
                               arm's decision columns, adding the robust tube)
  * FLMPC._relaxed_fallback   (graceful slack-QP infeasibility fallback -- it is
                               dimension-agnostic, so we call it per-arm)
  * FLMPC._torque_safe_scale  (closed-form torque-box backstop, per arm)
Only step() is overridden.

------------------------------------------------------------------------------
ROBUSTIFICATION MATH (worst-case tube, per collision pair k, horizon time t)
------------------------------------------------------------------------------
FLMPC's clearance row for pair k enforces, at horizon step t, the exact
first-order prediction of the barrier

    h_k(t) = h0_k + ce_k.(qd0_e t + Phi_q(t) a_e) - ci_k.(qd0_i t + Phi_q(t) a_i)
             >= H_MARGIN,

    with   ce_k = n_k . Je_k   (3xNV -> NV),   ci_k = n_k . Ji_k,
    and    n_k  the unit contact normal pi->pe  (so moving pi toward pe, i.e.
           along +n, LOWERS h; moving pe toward pi, along -n, also lowers h).

In FLMPC both a_e and a_i are decision variables (one joint QP). Decentralized,
each arm keeps only its OWN acceleration as a decision variable and REPLACES the
other arm's Phi_q(t) a_other term (which it can neither see nor command) by the
WORST it could be. The other arm's true keypoint motion beyond the nominal drift
is its acceleration integrated twice; over a horizon of length t a constant
Cartesian keypoint acceleration a_close (m/s^2) in the CLOSING direction displaces
the keypoint by at most  0.5 * a_close * t^2  (double integrator). So we tighten

    h_robust_k(t) = h_nominal_k(t) - 0.5 * a_close_other,k * t^2   >= H_MARGIN,

which, because the nominal already carries BOTH arms' drift terms, is exactly the
FLMPC row for this arm (other arm's decision columns dropped) with the RHS b
reduced by  0.5 * a_close_other,k * t^2.

a_close (the meanest neighbor) is derived from the JOINT TORQUE BOX via the exact
feedback-linearization map, NOT a hand-picked Cartesian number. For the other arm
j (mass matrix M_j, bias c_j at the current state), any joint acceleration it can
produce is a_j = M_j^{-1}(tau_j - c_j) with |tau_j|_inf <= TAU_MAX. Its keypoint's
Cartesian acceleration along the closing direction is a linear functional of tau_j:

    ego arm defending (disturber = int j):  close_i = n . Ji . a_i = w_i.(tau_i - c_i),
        w_i = n . Ji . M_i^{-1}   (NV,)
        a_close_i = max_{|tau_i|<=TAU_MAX} w_i.(tau_i - c_i)
                  = TAU_MAX * ||w_i||_1  -  w_i . c_i
    int arm defending (disturber = ego e):  h grows with +n.(ego motion), so the
        worst ego motion MINIMIZES n . Je . a_e; the reduction of h is its negative:
        w_e = n . Je . M_e^{-1}
        a_close_e = -min_{|tau_e|<=TAU_MAX} w_e.(tau_e - c_e)
                  = TAU_MAX * ||w_e||_1  +  w_e . c_e

Both are clipped at >= 0 (WC_CLIP_NONNEG, default True): a disturber that
physically cannot close the gap at this state is never given "credit" that would
LOOSEN the row below its nominal -- the tube only ever tightens. WC_GAIN (default
1.0) scales the whole tube for sensitivity studies (0.0 recovers a comms-free but
non-robust decentralization; the arm-arm rows then use each other's current state
only). M_e here already includes any carried-payload reflection (M_ego_extra),
so a heavy cup correctly makes the ego arm a SLOWER disturber.

CONSEQUENCE (expected): each arm defends against the meanest possible neighbor
without coordinating, so the pair is double-counted (each side reserves the full
tube) and the controller is MORE CONSERVATIVE than the cooperative joint QP:
larger deflections / earlier yielding, and, if the tube eats the whole margin,
possible task slowdown or standstill. See the smoke report.
"""
from __future__ import annotations
import numpy as np
import sim_common as sc
from mpc_fl import FLMPC, _assemble_fl


# Module-level default worst-case tube gain, so a driver that constructs
# FLMPCDecentralized without threading a wc_gain kwarg (e.g.
# make_cup_collect_safety_demo's flmpc branch) can still sweep it for a
# conservativeness study. 1.0 = full HJI worst case (spec default).
DEFAULT_WC_GAIN = 1.0


class FLMPCDecentralized(FLMPC):
    def __init__(self, *args, wc_gain: float = None, wc_clip_nonneg: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.WC_GAIN = float(DEFAULT_WC_GAIN if wc_gain is None else wc_gain)
        self.WC_CLIP_NONNEG = bool(wc_clip_nonneg)
        # per-step introspection
        self.last_feasible_left = True
        self.last_feasible_right = True
        self.last_tube_max = 0.0

    def step(self, model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
             q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
             pairs_all, M_ego_extra=None, bias_ego_extra=None,
             left_ee_body=None, tilt_max_rad=None, table_rows=None,
             leash_ee_body=None, leash_max_m=None, cup_cone_facets=None,
             ego_hold=True):
        NV, NC, NP, DT = self.NV, self.NC, self.NP, self.DT
        n_blk = self.n_blk
        ntot = 2 * n_blk
        eslice = slice(0, n_blk)
        islice = slice(n_blk, ntot)

        n_facets = self.cup_cone_facets if cup_cone_facets is None else int(cup_cone_facets)
        if n_facets and n_facets > 0:
            raise NotImplementedError(
                "FLMPCDecentralized only implements the per-axis cup-tilt BOX "
                "(cup_cone_facets=0); the demo default is 0.")

        # ---- current state (same as FLMPC.step) ----
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

        tvec = np.arange(NP + 1) * DT
        qnom_e = q0_e[None, :] + np.outer(tvec, qd0_e)
        qnom_i = q0_i[None, :] + np.outer(tvec, qd0_i)
        qdnom_e = np.tile(qd0_e, (NP + 1, 1))
        qdnom_i = np.tile(qd0_i, (NP + 1, 1))

        pairs = sorted(pairs_all, key=lambda p: p["h"])[:self.K_PAIRS]
        n_arr = np.stack([p["n"] for p in pairs])
        Je_arr = np.stack([p["Je"] for p in pairs])
        Ji_arr = np.stack([p["Ji"] for p in pairs])
        h0_arr = np.array([p["h"] for p in pairs])

        # ego_hold=False releases the ego (cup) arm's position-hold (W_pe -> 0);
        # decentralized keeps the same convention as FLMPC.step.
        W_pe_eff = self.W['pe'] if ego_hold else 0.0
        self.last_W_pos_ego_eff = W_pe_eff

        # ---- REUSE the joint numba assembly, then slice per arm ----
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
            self.H_MARGIN, sc.TAU_MAX)

        K = n_arr.shape[0]
        NCS = self.coll_steps.shape[0]
        m_coll = K * NCS

        # cost blocks are block-diagonal (see _assemble_fl: tracking/effort/jerk
        # never write ego<->int cross terms), so the per-arm cost is exactly the
        # diagonal block.
        H_e = np.ascontiguousarray(H[eslice, eslice]); f_e = np.ascontiguousarray(f[eslice])
        H_i = np.ascontiguousarray(H[islice, islice]); f_i = np.ascontiguousarray(f[islice])

        # ---- worst-case closing accel per pair (see module docstring) ----
        Minv_e = np.linalg.inv(M_e)
        Minv_i = np.linalg.inv(M_i)
        a_close_i = np.zeros(K)   # int (disturber) closing accel, defended by EGO qp
        a_close_e = np.zeros(K)   # ego (disturber) closing accel, defended by INT qp
        for k in range(K):
            n_k = n_arr[k]
            w_i = n_k @ Ji_arr[k] @ Minv_i         # (NV,)
            aci = sc.TAU_MAX * np.abs(w_i).sum() - w_i @ c_i
            w_e = n_k @ Je_arr[k] @ Minv_e
            ace = sc.TAU_MAX * np.abs(w_e).sum() + w_e @ c_e
            if self.WC_CLIP_NONNEG:
                aci = max(0.0, aci); ace = max(0.0, ace)
            a_close_i[k] = aci; a_close_e[k] = ace
        tube_max = 0.0

        # ---- per-arm collision rows: slice ego/int coefficient columns, tighten b ----
        A_coll_e = np.zeros((m_coll, n_blk)); b_coll_e = np.zeros(m_coll)
        A_coll_i = np.zeros((m_coll, n_blk)); b_coll_i = np.zeros(m_coll)
        for row in range(m_coll):
            k = row // NCS
            si = row % NCS
            t = self.coll_steps[si]
            t2 = float(tvec[t]) ** 2
            A_coll_e[row] = a_ineq[row, eslice]        # -(ce @ Phi_q(t))
            A_coll_i[row] = a_ineq[row, islice]        # +(ci @ Phi_q(t))
            tube_e = self.WC_GAIN * 0.5 * a_close_i[k] * t2   # ego defends vs int
            tube_i = self.WC_GAIN * 0.5 * a_close_e[k] * t2   # int defends vs ego
            b_coll_e[row] = b_ineq[row] - tube_e
            b_coll_i[row] = b_ineq[row] - tube_i
            tube_max = max(tube_max, tube_e, tube_i)
        self.last_tube_max = float(tube_max)

        # ---- torque-box rows (slice out of the joint assembly) ----
        # joint layout: rows [m_coll : m_coll+2NV] ego on cols 0:NV,
        #               rows [m_coll+2NV : m_coll+4NV] int on cols n_blk:n_blk+NV.
        A_tq_e = np.ascontiguousarray(a_ineq[m_coll:m_coll + 2 * NV, 0:n_blk])
        b_tq_e = np.ascontiguousarray(b_ineq[m_coll:m_coll + 2 * NV])
        A_tq_i = np.ascontiguousarray(a_ineq[m_coll + 2 * NV:m_coll + 4 * NV, n_blk:ntot])
        b_tq_i = np.ascontiguousarray(b_ineq[m_coll + 2 * NV:m_coll + 4 * NV])

        # ---- optional EGO extras (cup-upright box + leash), plain Python
        #      (ported from FLMPC.step; single-arm width n_blk == ego block) ----
        extra_a_e, extra_b_e = [], []
        ee_body = left_ee_body if left_ee_body is not None else self.left_ee_body
        tilt_max = tilt_max_rad if tilt_max_rad is not None else self.tilt_max_rad
        if ee_body is not None and tilt_max is not None:
            if self._cup_axis_local is None:
                R_ee0 = data.xmat[ee_body].reshape(3, 3).copy()
                self._cup_axis_local = R_ee0.T @ np.array([0.0, 0.0, 1.0])
            R_ee = data.xmat[ee_body].reshape(3, 3)
            cup_axis_world = R_ee @ self._cup_axis_local
            tilt0 = np.array([-cup_axis_world[1], cup_axis_world[0]])
            Jr_ee = sc.body_jac_full(model, data, ee_body)[1][:, left_dofadr]   # (3,NV)
            for t in self.tilt_steps:
                Aq = self.Phi_q_all[t]
                drift_t = qd0_e * tvec[t]
                for ax in (0, 1):
                    row_coef = Jr_ee[ax] @ Aq                    # (n_blk,)
                    drift = tilt0[ax] + Jr_ee[ax] @ drift_t
                    extra_a_e.append(row_coef);       extra_b_e.append(tilt_max - drift)
                    extra_a_e.append(-row_coef);      extra_b_e.append(tilt_max + drift)

        leash_body = leash_ee_body if leash_ee_body is not None else self.leash_ee_body
        leash_max = leash_max_m if leash_max_m is not None else self.leash_max_m
        if leash_body is not None and leash_max is not None:
            Jp_leash = sc.body_jac_full(model, data, leash_body)[0][:, left_dofadr]  # (3,NV)
            hold_e = q_ego_ref[0]
            for t in self.tilt_steps:
                Aq = self.Phi_q_all[t]
                const_t = Jp_leash @ (qnom_e[t] - hold_e)
                JA = Jp_leash @ Aq
                for ax in range(3):
                    extra_a_e.append(JA[ax]);   extra_b_e.append(leash_max - const_t[ax])
                    extra_a_e.append(-JA[ax]);  extra_b_e.append(leash_max + const_t[ax])

        # ---- optional TABLE rows per arm (ported; ego on M_e, int on M_i) ----
        extra_a_i, extra_b_i = [], []
        if table_rows is not None:
            a_tab_e, b_tab_e, a_tab_i, b_tab_i = table_rows
            for r in range(a_tab_e.shape[0]):
                row = np.zeros(n_blk); row[0:NV] = a_tab_e[r] @ M_e   # first-block accel only
                extra_a_e.append(row)
                extra_b_e.append(float(b_tab_e[r] - a_tab_e[r] @ c_e))
            for r in range(a_tab_i.shape[0]):
                row = np.zeros(n_blk); row[0:NV] = a_tab_i[r] @ M_i
                extra_a_i.append(row)
                extra_b_i.append(float(b_tab_i[r] - a_tab_i[r] @ c_i))

        # ---- assemble + solve each arm's QP (collision rows LEADING, so the
        #      graceful fallback slackens exactly the clearance rows) ----
        a_e0, feas_e = self._solve_arm(H_e, f_e, A_coll_e, b_coll_e, A_tq_e, b_tq_e,
                                        extra_a_e, extra_b_e, m_coll, M_e, c_e)
        a_i0, feas_i = self._solve_arm(H_i, f_i, A_coll_i, b_coll_i, A_tq_i, b_tq_i,
                                        extra_a_i, extra_b_i, m_coll, M_i, c_i)

        self.last_feasible_left = feas_e
        self.last_feasible_right = feas_i
        self.last_feasible = feas_e and feas_i
        self.a_ego_prev = a_e0.copy()
        self.a_int_prev = a_i0.copy()
        tau_e = M_e @ a_e0 + c_e
        tau_i = M_i @ a_i0 + c_i
        # returned `feasible` mirrors FLMPC.step's single bool (both arms feasible)
        return tau_e, tau_i, bool(feas_e and feas_i)

    def _solve_arm(self, H, f, A_coll, b_coll, A_tq, b_tq, extra_a, extra_b,
                   m_coll, M_arm, c_arm):
        """Solve ONE arm's decentralized box-QP over its own NV*NC accel blocks.
        Rows are ordered [collision (m_coll), torque box, optional extras] so the
        reused FLMPC._relaxed_fallback slackens exactly the leading collision
        rows on an infeasible step; the closed-form torque-box backstop
        (FLMPC._torque_safe_scale) is applied to the first-block accel on any
        infeasible step, identical to FLMPC.step."""
        NV, n_blk = self.NV, self.n_blk
        rows = [A_coll, A_tq]
        bs = [b_coll, b_tq]
        if extra_a:
            rows.append(np.stack(extra_a)); bs.append(np.asarray(extra_b, dtype=float))
        a_ineq = np.ascontiguousarray(np.vstack(rows))
        b_ineq = np.ascontiguousarray(np.concatenate(bs))
        lower = -self.a_box * np.ones(n_blk)
        upper = self.a_box * np.ones(n_blk)
        x, _, feasible = sc.solve_constrained_box_qp(H, f, lower, upper, a_ineq, b_ineq)
        if not feasible:
            x_relaxed, feasible2 = self._relaxed_fallback(H, f, lower, upper, a_ineq, b_ineq, m_coll)
            x = x_relaxed if feasible2 else np.zeros(n_blk)
        a0 = x[0:NV].copy()
        if not feasible:
            a0 = self._torque_safe_scale(M_arm, c_arm, a0, sc.TAU_MAX)
        return a0, bool(feasible)
