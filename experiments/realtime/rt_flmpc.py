"""Real-time FL-MPC (phase 1): mpc_fl.FLMPC's exact optimisation problem, solved by ONE
compiled, allocation-free, time-bounded Numba entry point. SIMULATION ONLY -- nothing
in this module opens any hardware / network connection.

    RTFLMPC(model, **FLMPC kwargs)      subclass of mpc_fl.FLMPC (same ctor, same
                                        attributes/counters the demo reads); FLMPC
                                        itself is NOT modified.
      .step(...)        drop-in, SAME signature/inputs as FLMPC.step (scene `data`,
                        caller's pairs / table_rows / payload): rows + QP + fallback
                        in rt_kernels.rt_core. Used by the G1 equivalence gate.
      .configure_lean(...) / .step_lean(...)
                        the deployment path: robot-only dynamics (rt_dynamics) +
                        geometry + rows + QP + fallback in ONE njit call
                        (rt_kernels.rt_full). The demo's --controller-impl rt uses this.

Solver: Goldfarb-Idnani dual active set (rt_qp.py), capped. Semantics per step:
  hard QP OPTIMAL                        -> a = QP optimum (fallback 0)
  hard QP INFEASIBLE or ITERATION CAP    -> relaxed QP (collision + EE-speed rows get a
                                            slack s>=0, cost RHO*s + EPS*s^2, s<=S_MAX;
                                            torque box / tilt / leash / table stay HARD)
     relaxed OPTIMAL                     -> a = relaxed optimum      (fallback 1)
     relaxed INFEASIBLE or CAP           -> a = 0                    (fallback 2)
  hard not OPTIMAL                       -> exact torque-box backstop _torque_safe_scale
  == mpc_fl.FLMPC.step's semantics, with "cap hit" handled exactly like "infeasible"
  (a capped GI iterate is optimal only for a SUBSET of rows, i.e. not certified
  feasible, so it is never returned). Caps are COUNTED (n_hard_cap / n_relax_cap)
  and iterations are logged per step (hard_it_hist / relax_it_hist, .records).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, os.path.join(_REPO, "experiments", "ab_smooth_vs_bangbang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mpc_fl  # noqa: E402
import sim_common as sc  # noqa: E402
import rt_kernels as K  # noqa: E402
import rt_qp  # noqa: E402

IT_HARD = 60       # GI iteration cap, hard QP    (>= 1.5x the max observed on logged steps; see report)
IT_RELAX = 200     # GI iteration cap, relaxed QP (>= 1.5x the max observed in the stress test)
TABLE_KP_MAX = 96  # per-arm table keypoint capacity (grown on demand, outside the kernel)
FACETS_MAX = 32


def _chain_mask(model, body, dofs):
    """mask[c] = 1 iff dofs[c] lies on body's kinematic chain (== mj_jac nonzero column)."""
    bb = int(body)
    while bb and not model.body_dofnum[bb]:
        bb = int(model.body_parentid[bb])
    chain = set()
    if bb:
        i = int(model.body_dofadr[bb] + model.body_dofnum[bb] - 1)
        while i >= 0:
            chain.add(i)
            i = int(model.dof_parentid[i])
    return np.array([1 if int(d) in chain else 0 for d in dofs], np.int64)


def _topk_pairs(pairs_all, Kp):
    """== sorted(pairs_all, key=h)[:K] (FLMPC.step); for a fast_barriers.PairList the
    same stable top-K is taken from its arrays (== fast_barriers.topk_pairs) without
    materialising the ~1200 dicts (and without importing fast_barriers)."""
    a = getattr(pairs_all, "arrays", None)
    if a is None:
        return sorted(pairs_all, key=lambda p: p["h"])[:Kp]
    idx = np.argsort(a["h"], kind="stable")[:Kp]
    out = []
    for k in idx.tolist():
        e, i = divmod(k, a["I"])
        out.append(dict(h=float(a["h"][k]), n=a["n"][k], Je=a["Je"][e], Ji=a["Ji"][i]))
    return out


class RTWorkspace:
    """All scratch/output arrays of one controller (C-contiguous, preallocated)."""

    def __init__(self, NV, n_blk, Kp, E, I, ntl, ntr, m_max, ns_max):
        ntot = 2 * n_blk
        nt = ntot + ns_max
        z = np.zeros
        self.dims = (NV, n_blk, Kp, E, I, ntl, ntr, m_max, ns_max)
        self.W = (z((NV, NV)), z(NV), z((NV, NV)), z(NV), z((NV, NV)), z(NV), z((NV, NV)), z(NV),
                  z((Kp, 3)), z((Kp, 3, NV)), z((Kp, 3, NV)), z(Kp), z(Kp),
                  z(2), z((3, NV)), z((3, NV)), z((3, NV)), z((3, NV)), z((3, NV)),
                  z((ntl, NV)), z(ntl), z((ntr, NV)), z(ntr),
                  z((E, 3)), z((I, 3)), z(E * I), z(Kp, np.int64), z(3), z((3, NV)), z((3, NV)), z((6, NV)),
                  z((NV, NV)), z((NV, NV)), z((NV, NV)), z(NV), z((3, 3)), z((2, NV)),
                  z(ntot), z((m_max, ntot)), z(m_max), z(m_max, np.int64),
                  z((nt, nt)), z((nt, nt)), z(nt + 1, np.int64), z(nt + 1), z(nt), z(m_max + 2 * nt, np.int64),
                  z(nt), z(nt), z(nt), z(m_max, np.int64), z(nt), z(nt), z(nt), z(4, np.int64), z(2 * m_max))
        (self.M_e, self.c_e, self.M_i, self.c_i, self.M_tl, self.b_tl, self.M_tr, self.b_tr,
         self.n_arr, self.Je_arr, self.Ji_arr, self.h0_arr, self.hdot_arr,
         self.tilt0, self.Jr_ee, self.Jp_ee, self.Jp_leash, self.Jr_tmp, self.Jp_int,
         self.a_tab_e, self.b_tab_e, self.a_tab_i, self.b_tab_i,
         self.pe, self.pi_, self.hbuf, self.sel, self.p3, self.J3, self.Jd3, self.Jb6,
         self.Minv, self.work6, self.M_box, self.b_box, self.Iw, self.scr,
         self.f, self.A, self.b, self.slack_rows,
         self.J, self.R, self.act, self.u, self.y, self.iact,
         self.d, self.z, self.r, self.sidx, self.g_aug, self.lo, self.hi, self.stats, self.rn) = self.W
        self.oi = np.zeros(K.N_OI, np.int64)
        self.xout = z(ntot)
        self.a_e0 = z(NV)
        self.a_i0 = z(NV)
        self.tau_e = z(NV)
        self.tau_i = z(NV)
        self.tau_lim_e = z(NV)
        self.tau_lim_i = z(NV)


class RTFLMPC(mpc_fl.FLMPC):
    """See module docstring."""

    def __init__(self, model, *args, it_hard: int = IT_HARD, it_relax: int = IT_RELAX, record: bool = False,
                 **kwargs):
        super().__init__(model, *args, **kwargs)
        if self.adaptive_yield:
            raise NotImplementedError("rt_flmpc: adaptive_yield (default off, legacy ablation) is not ported")
        NV, NC, NP = self.NV, self.NC, self.NP
        self.it_hard, self.it_relax = int(it_hard), int(it_relax)
        self._ci = np.array([NC, self.L, NV, NP, self.n_blk, self.K_PAIRS, self.it_hard, self.it_relax], np.int64)
        self._cf = np.zeros(K.N_CF)
        self._cf[K.CF_WVE] = self.W["ve"]
        self._cf[K.CF_WPI] = self.W["pi"]
        self._cf[K.CF_WVI] = self.W["vi"]
        self._cf[K.CF_WR] = self.W["r"]
        self._cf[K.CF_WJ] = self.W["j"]
        self._cf[K.CF_ABOX] = self.a_box
        self._cf[K.CF_RHO] = self.RHO_SLACK
        self._cf[K.CF_EPS] = self.EPS_SLACK
        self._cf[K.CF_SMAX] = self.S_MAX
        self._cf[K.CF_GACC] = self.G_ACCEL
        # relaxed Hessian slack block: H_aug = 2*EPS, + the solver's 1e-8 Tikhonov (as in FLMPC's QP call)
        self._cf[K.CF_SIGMA] = float(np.sqrt(2.0 * self.EPS_SLACK + 1e-8))
        self._cq = np.ascontiguousarray(self.cq)
        self._cqd = np.ascontiguousarray(self.cqd)
        self._tvec = np.arange(NP + 1) * self.DT
        self._coll_steps = np.ascontiguousarray(self.coll_steps, np.int64)
        self._tilt_steps = np.ascontiguousarray(self.tilt_steps, np.int64)
        self._speed_steps = np.ascontiguousarray(self.speed_cap_steps, np.int64)
        self._pi = np.zeros(K.N_PI, np.int64)
        self._pf = np.zeros(K.N_PF)
        self._J0 = {}
        self._tau_max_arr = (np.array([sc.TAU_MAX]) if self.tau_max_vec is None
                             else np.ascontiguousarray(self.tau_max_vec, np.float64))
        # counters / logs (all steps, never silent)
        self.n_hard_cap = 0
        self.n_relax_cap = 0
        self.n_hard_infeasible = 0
        self.n_relax_infeasible = 0
        self.n_steps_rt = 0
        self.hard_it_hist = np.zeros(self.it_hard + 1, np.int64)
        self.relax_it_hist = np.zeros(self.it_relax + 1, np.int64)
        self.record = bool(record)
        self.records = []
        self.last_oi = None
        self._lean_cfg = None
        self._ws = None
        self._empty2 = np.zeros((1, NV))
        self._empty1 = np.zeros(1)
        self._zero3 = np.zeros((3, NV))
        self._ensure_ws(ntl=TABLE_KP_MAX, ntr=TABLE_KP_MAX, E=1, I=1)
        self._J0_for(self.W["pe"])
        self._J0_for(0.0)

    # ------------------------------------------------------------------ setup
    def _dims(self, ntl, ntr, nfac=FACETS_MAX):
        NV = self.NV
        m_coll = self.K_PAIRS * len(self.coll_steps)
        nts = len(self.tilt_steps)
        n_sp = 7 * len(self.speed_cap_steps)
        m_max = m_coll + 4 * NV + max(nfac, 4) * nts + 6 * nts + ntl + ntr + n_sp
        return m_max, m_coll + n_sp

    def _ensure_ws(self, ntl, ntr, E, I):
        cur = self._ws.dims if self._ws is not None else None
        if cur is not None and ntl <= cur[5] and ntr <= cur[6] and E <= cur[3] and I <= cur[4]:
            return
        if cur is not None:
            ntl, ntr, E, I = max(ntl, cur[5]), max(ntr, cur[6]), max(E, cur[3]), max(I, cur[4])
        m_max, ns_max = self._dims(ntl, ntr)
        self._ws = RTWorkspace(self.NV, self.n_blk, self.K_PAIRS, E, I, ntl, ntr, m_max, ns_max)

    def _J0_for(self, W_pe_eff):
        """J0 = L^{-T} of G = 0.5(H+H^T) + 1e-8 I. H is state-independent: it is taken
        from mpc_fl._assemble_fl itself (zero state), so it is bit-identical to the H
        the py path builds every step. Cached per W_pe_eff (ego_hold toggles it)."""
        key = float(W_pe_eff)
        J0 = self._J0.get(key)
        if J0 is None:
            NV, NP = self.NV, self.NP
            z = np.zeros((NP + 1, NV))
            H = mpc_fl._assemble_fl(self.Phi_q_all, self.Phi_qd_all, z, z, z, z, z, z, z, z,
                                    np.zeros(NV), np.zeros(NV), np.zeros((1, 3)), np.zeros((1, 3, NV)),
                                    np.zeros((1, 3, NV)), np.zeros(1), self.coll_steps, np.zeros(NV), np.zeros(NV),
                                    self._tvec, np.eye(NV), np.zeros(NV), np.eye(NV), np.zeros(NV),
                                    self.NC, NV, self.n_blk, key, self.W["ve"], self.W["pi"], self.W["vi"],
                                    self.W["r"], self.W["j"], self.H_MARGIN, sc.TAU_MAX)[0]
            n = H.shape[0]
            G = 0.5 * (H + H.T) + 1e-8 * np.eye(n)
            J0 = rt_qp.factor(G)
            self._J0[key] = J0
            self._H_cache = getattr(self, "_H_cache", {})
            self._H_cache[key] = H
        return J0

    # ------------------------------------------------------------------ bookkeeping
    def rt_log_fields(self):
        """Counters for the demo log / comprehensive_eval rows (prefix fl_rt_)."""
        return dict(fl_rt_steps=self.n_steps_rt, fl_rt_hard_cap_steps=self.n_hard_cap,
                    fl_rt_relax_cap_steps=self.n_relax_cap, fl_rt_hard_infeasible_steps=self.n_hard_infeasible,
                    fl_rt_relax_infeasible_steps=self.n_relax_infeasible,
                    fl_rt_hard_it_hist=np.trim_zeros(self.hard_it_hist, "b"),
                    fl_rt_relax_it_hist=np.trim_zeros(self.relax_it_hist, "b"),
                    fl_rt_it_caps=np.array([self.it_hard, self.it_relax]))

    def _post(self, oi, M_e, c_e, M_i, c_i, tau_lim_e, tau_lim_i, t_dyn=0.0, t_ker=0.0, t_tot=0.0):
        ws = self._ws
        st, fb = int(oi[K.OI_HARD_STATUS]), int(oi[K.OI_FALLBACK])
        feasible = st == rt_qp.ST_OPTIMAL
        self.last_feasible = feasible
        self.last_fallback = fb
        self.n_steps_rt += 1
        self.hard_it_hist[min(int(oi[K.OI_HARD_IT]), self.it_hard)] += 1
        if st == rt_qp.ST_ITERCAP:
            self.n_hard_cap += 1
        elif st == rt_qp.ST_INFEASIBLE:
            self.n_hard_infeasible += 1
        if not feasible:
            rs = int(oi[K.OI_RELAX_STATUS])
            self.relax_it_hist[min(int(oi[K.OI_RELAX_IT]), self.it_relax)] += 1
            if rs == rt_qp.ST_ITERCAP:
                self.n_relax_cap += 1
            elif rs == rt_qp.ST_INFEASIBLE:
                self.n_relax_infeasible += 1
            if fb == 1:
                self.n_relaxed += 1
            else:
                self.n_zero_fallback += 1
        self.last_M_e, self.last_c_e, self.last_M_i, self.last_c_i = M_e.copy(), c_e.copy(), M_i.copy(), c_i.copy()
        a_e0 = ws.a_e0.copy()
        a_i0 = ws.a_i0.copy()
        self.a_ego_prev = a_e0
        self.a_int_prev = a_i0.copy()
        if self.delay_steps > 0:
            self._a_queue.append((a_e0.copy(), a_i0.copy()))
            if len(self._a_queue) > self.delay_steps:
                self._a_queue.pop(0)
        self.last_oi = oi.copy()
        if self.record:
            self.records.append((st, int(oi[K.OI_HARD_IT]), int(oi[K.OI_RELAX_STATUS]), int(oi[K.OI_RELAX_IT]),
                                 int(oi[K.OI_M]), fb, t_dyn, t_ker, t_tot))
        return ws.tau_e.copy(), ws.tau_i.copy(), feasible

    def _set_params(self, tilt_on, nfac, cone_true, leash_on, tab_on, ntab_e, ntab_i, speed_on,
                    W_pe_eff, H_margin, tilt_max, leash_max, v_cap):
        pi, pf = self._pi, self._pf
        pi[K.PI_TILT_ON] = 1 if tilt_on else 0
        pi[K.PI_NFACETS] = int(nfac)
        pi[K.PI_CONE_TRUE] = 1 if cone_true else 0
        pi[K.PI_LEASH_ON] = 1 if leash_on else 0
        pi[K.PI_TAB_ON] = 1 if tab_on else 0
        pi[K.PI_SPEED_ON] = 1 if speed_on else 0
        pi[K.PI_UDIR_ON] = 0
        pi[K.PI_NTAB_E] = int(ntab_e)
        pi[K.PI_NTAB_I] = int(ntab_i)
        pi[K.PI_VEC_TAU] = 0 if self.tau_max_vec is None else 1
        pf[K.PF_WPE] = W_pe_eff
        pf[K.PF_HMARGIN] = H_margin
        pf[K.PF_TILTMAX] = 0.0 if tilt_max is None else float(tilt_max)
        pf[K.PF_LEASHMAX] = 0.0 if leash_max is None else float(leash_max)
        pf[K.PF_VCAP] = 0.0 if v_cap is None else float(v_cap)

    # ------------------------------------------------------------------ drop-in step
    def step(self, model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
             q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
             pairs_all, M_ego_extra=None, bias_ego_extra=None,
             left_ee_body=None, tilt_max_rad=None, table_rows=None,
             leash_ee_body=None, leash_max_m=None, cup_cone_facets=None,
             ego_hold=True, int_ee_body=None, v_ee_max=None):
        """SAME inputs/outputs as mpc_fl.FLMPC.step; the python-side input gathering is
        a line-for-line copy of FLMPC.step's, the rest runs in rt_kernels.rt_core."""
        t0 = time.perf_counter()
        NV = self.NV
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
        dq_e_pred = dq_i_pred = None
        if self.delay_steps > 0:
            q0_e, qd0_e, q0_i, qd0_i, dq_e_pred, dq_i_pred = self._predict_over_delay(q0_e, qd0_e, q0_i, qd0_i)
        pairs = _topk_pairs(pairs_all, self.K_PAIRS)
        n_arr = np.ascontiguousarray(np.stack([p["n"] for p in pairs]))
        Je_arr = np.ascontiguousarray(np.stack([p["Je"] for p in pairs]))
        Ji_arr = np.ascontiguousarray(np.stack([p["Ji"] for p in pairs]))
        h0_arr = np.array([p["h"] for p in pairs])
        if dq_e_pred is not None:
            for k in range(h0_arr.shape[0]):
                h0_arr[k] = (h0_arr[k] + float((n_arr[k] @ Je_arr[k]) @ dq_e_pred)
                             - float((n_arr[k] @ Ji_arr[k]) @ dq_i_pred))
        W_pe_eff = self.W["pe"] if ego_hold else 0.0
        self.last_urgency = 0.0
        self.last_H_MARGIN_eff = self.H_MARGIN
        self.last_cup_share_gain = 0.0
        self.last_W_pos_ego_eff = W_pe_eff
        tau_lim_e = self._ws.tau_lim_e
        tau_lim_i = self._ws.tau_lim_i
        if self.tau_max_vec is not None:
            tau_lim_e[:] = np.maximum(self.tau_max_vec, np.abs(c_e))
            tau_lim_i[:] = np.maximum(self.tau_max_vec, np.abs(c_i))
            if np.any(tau_lim_e > self.tau_max_vec) or np.any(tau_lim_i > self.tau_max_vec):
                self.n_taumax_relaxed += 1
        else:
            tau_lim_e[:] = sc.TAU_MAX
            tau_lim_i[:] = sc.TAU_MAX
        # (3) cup upright inputs
        tilt0 = np.zeros(2)
        Jr_ee = Jp_ee = self._zero3
        ee_body = left_ee_body if left_ee_body is not None else self.left_ee_body
        tilt_max = tilt_max_rad if tilt_max_rad is not None else self.tilt_max_rad
        tilt_on = ee_body is not None and tilt_max is not None
        n_facets = self.cup_cone_facets if cup_cone_facets is None else int(cup_cone_facets)
        if tilt_on:
            if self._cup_axis_local is None:
                R_ee0 = data.xmat[ee_body].reshape(3, 3).copy()
                self._cup_axis_local = R_ee0.T @ np.array([0.0, 0.0, 1.0])
            R_ee = data.xmat[ee_body].reshape(3, 3)
            cup_axis_world = R_ee @ self._cup_axis_local
            tilt0 = np.array([-cup_axis_world[1], cup_axis_world[0]])
            jp, jr = sc.body_jac_full(model, data, ee_body)
            Jr_ee = np.ascontiguousarray(jr[:, left_dofadr])
            Jp_ee = np.ascontiguousarray(jp[:, left_dofadr])
            if dq_e_pred is not None:
                tilt0 = tilt0 + Jr_ee[0:2] @ dq_e_pred
        # (4) leash
        leash_body = leash_ee_body if leash_ee_body is not None else self.leash_ee_body
        leash_max = leash_max_m if leash_max_m is not None else self.leash_max_m
        leash_on = leash_body is not None and leash_max is not None
        Jp_leash = self._zero3
        if leash_on:
            Jp_leash = np.ascontiguousarray(sc.body_jac_full(model, data, leash_body)[0][:, left_dofadr])
        # (5) table
        tab_on = table_rows is not None
        if tab_on:
            a_tab_e, b_tab_e, a_tab_i, b_tab_i = (np.ascontiguousarray(v, np.float64) for v in table_rows)
            self._ensure_ws(a_tab_e.shape[0], a_tab_i.shape[0], 1, 1)
        else:
            a_tab_e = a_tab_i = self._empty2
            b_tab_e = b_tab_i = self._empty1
        # (6) speed cap
        v_cap = v_ee_max if v_ee_max is not None else self.ee_speed_cap
        sp_body = int_ee_body if int_ee_body is not None else self.ee_speed_body_int
        speed_on = v_cap is not None and sp_body is not None
        Jp_int = self._zero3
        if speed_on:
            Jp_int = np.ascontiguousarray(sc.body_jac_full(model, data, sp_body)[0][:, right_dofadr])
        self._set_params(tilt_on, n_facets if (n_facets and n_facets > 0) else 0, self.cone_mode == "true",
                         leash_on, tab_on, a_tab_e.shape[0] if tab_on else 0, a_tab_i.shape[0] if tab_on else 0,
                         speed_on, W_pe_eff, self.H_MARGIN, tilt_max, leash_max, v_cap)
        J0 = self._J0_for(W_pe_eff)
        ws = self._ws
        t1 = time.perf_counter()
        K.rt_core(self._ci, self._cf, self._pi, self._pf, self._cq, self._cqd, self._tvec, self._coll_steps,
                  self._tilt_steps, self._speed_steps,
                  q0_e, qd0_e, q0_i, qd0_i, M_e, c_e, M_i, c_i,
                  np.ascontiguousarray(q_ego_ref, np.float64), np.ascontiguousarray(qd_ego_ref, np.float64),
                  np.ascontiguousarray(q_int_ref, np.float64), np.ascontiguousarray(qd_int_ref, np.float64),
                  self.a_ego_prev, self.a_int_prev,
                  n_arr, Je_arr, Ji_arr, h0_arr, tau_lim_e, tau_lim_i,
                  tilt0, Jr_ee, Jp_ee, Jp_leash, a_tab_e, b_tab_e, a_tab_i, b_tab_i, Jp_int,
                  J0, ws.f, ws.A, ws.b, ws.slack_rows, ws.J, ws.R, ws.act, ws.u, ws.y, ws.iact, ws.d, ws.z, ws.r,
                  ws.sidx, ws.g_aug, ws.lo, ws.hi, ws.stats, ws.oi, ws.xout, ws.a_e0, ws.a_i0, ws.tau_e, ws.tau_i,
                  ws.scr, ws.rn)
        t2 = time.perf_counter()
        return self._post(ws.oi, M_e, c_e, M_i, c_i, tau_lim_e, tau_lim_i, t1 - t0, t2 - t1, t2 - t0)

    # ------------------------------------------------------------------ lean (deployment) path
    def configure_lean(self, lean, scene_model, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
                       left_ee_body, cup_body=None, right_ee_body=None, speed_body=None,
                       ego_extra_kps=None, left_table_kps=None, right_table_kps=None,
                       table_z=None, table_margin=0.0, table_alpha1=sc.ALPHA1, table_alpha2=sc.ALPHA2,
                       table_kp_radius=False, table_gravfix=True, leash_body=None):
        """Static wiring (scene ids are mapped onto the robot-only model by name).
        table_gravfix: the demo's table rows use bias = qfrc_bias - qfrc_passive (ctrlfix
        "gravity") -> True; bias = qfrc_bias alone (ctrlfixes off) -> False."""
        lm = lean.model
        ldofs = lean.lean_dofs(left_dofadr)
        rdofs = lean.lean_dofs(right_dofadr)
        root = int(lm.body_rootid[lean.lean_body(left_ee_body)])
        lee = lean.lean_body(left_ee_body)
        lb = lean.lean_body(leash_body if leash_body is not None else left_ee_body)
        sb = lean.lean_body(speed_body) if speed_body is not None else lee
        rb = lean.lean_body(right_ee_body) if right_ee_body is not None else sb
        cfg = dict(lean=lean, scene_model=scene_model, ldofs=ldofs, rdofs=rdofs, root=root,
                   left_qposadr=np.asarray(left_qposadr), left_dofadr=np.asarray(left_dofadr),
                   right_qposadr=np.asarray(right_qposadr), right_dofadr=np.asarray(right_dofadr),
                   lee=lee, lb=lb, sb=sb, rb=rb, cup_body=cup_body, right_ee_body_scene=right_ee_body,
                   lmask_ee=_chain_mask(lm, lee, ldofs), lmask_leash=_chain_mask(lm, lb, ldofs),
                   rmask_speed=_chain_mask(lm, sb, rdofs), rmask_cube=_chain_mask(lm, rb, rdofs),
                   table_z=table_z, table_margin=float(table_margin), alpha1=float(table_alpha1),
                   alpha2=float(table_alpha2), kp_radius=bool(table_kp_radius), gravfix=bool(table_gravfix),
                   gvec=np.ascontiguousarray(scene_model.opt.gravity, np.float64), pack_cache={})
        self._lean_cfg = cfg
        cfg["ego_base"] = self._pack(sc.barrier_keypoints(scene_model, sc.EGO_PREFIX), ldofs)
        cfg["int_kps"] = self._pack(sc.barrier_keypoints(scene_model, sc.INT_PREFIX), rdofs)
        cfg["ego_extra"] = ego_extra_kps
        cfg["ltab"] = left_table_kps
        cfg["rtab"] = right_table_kps
        self._gcfg = np.zeros(K.N_GI, np.int64)
        self._gpar = np.zeros(K.N_GF)
        self._flg = np.zeros(K.N_FL, np.int64)
        self._cup_mI = np.zeros(4)
        self._cube_mI = np.zeros(4)
        self._eye3 = np.eye(3)
        self._z3 = np.zeros(3)
        self._zNV = np.zeros(self.NV)
        self.jit_warmup_s = self._warmup_lean()

    def _warmup_lean(self):
        """Compile / load-from-cache the rt_full entry point NOW (not on the first control
        step): one dry step_lean on the robot-only model's reference state with zero refs;
        every piece of controller state it touches is restored afterwards."""
        t0 = time.perf_counter()
        keep = {k: (v.copy() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else v))
                for k, v in self.__dict__.items()
                if k not in ("_lean_cfg", "_ws", "_J0", "_H_cache", "model", "Phi_q_all", "Phi_qd_all")}
        lean = self._lean_cfg["lean"]
        q0 = lean.model.qpos0.copy()
        v0 = np.zeros(lean.model.nv)
        z = np.zeros((self.NP + 1, self.NV))
        try:
            RTFLMPC.step_lean(self, None, z, z, z, z, sensing=(q0, v0), ego_extra_kps="cfg", right_table_kps="cfg",
                              table_floor=True, left_ee_body=None, tilt_max_rad=None)
        finally:
            self.__dict__.update(keep)
        return time.perf_counter() - t0

    def _pack(self, kps, dofs, cache=True):
        """keypoint list [(scene body, off, r)] -> (lean body ids, offs (K,3), radii, masks (K,NV)),
        cached by list identity (the demo's lists are static)."""
        cfg = self._lean_cfg
        key = (id(kps), id(dofs))
        ent = cfg["pack_cache"].get(key)
        if ent is not None and ent[0] is kps and ent[1] == len(kps):
            return ent[2]
        lean = cfg["lean"]
        bodies = np.array([lean.lean_body(b) for b, _o, _r in kps], np.int64)
        offs = np.ascontiguousarray(np.array([np.asarray(o, float) for _b, o, _r in kps]).reshape(-1, 3))
        radii = np.array([float(r) for _b, _o, r in kps])
        masks = np.ascontiguousarray(np.stack([_chain_mask(lean.model, b, dofs) for b in bodies])) \
            if len(bodies) else np.zeros((0, len(dofs)), np.int64)
        res = (bodies, offs, radii, masks)
        if cache:
            cfg["pack_cache"][key] = (kps, len(kps), res)
        return res

    def _ego_pack(self, extra):
        cfg = self._lean_cfg
        key = ("ego", id(extra))
        ent = cfg["pack_cache"].get(key)
        if ent is not None and ent[0] is extra:
            return ent[2]
        b0, o0, r0, m0 = cfg["ego_base"]
        if extra:
            b1, o1, r1, m1 = self._pack(extra, cfg["ldofs"], cache=False)
            res = (np.concatenate([b0, b1]), np.ascontiguousarray(np.concatenate([o0, o1])),
                   np.concatenate([r0, r1]), np.ascontiguousarray(np.concatenate([m0, m1])))
        else:
            res = (b0, o0, r0, m0)
        cfg["pack_cache"][key] = (extra, None, res)
        return res

    def step_lean(self, scene_data, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, *,
                  left_ee_body=None, tilt_max_rad=None, leash_ee_body=None, leash_max_m=None,
                  cup_cone_facets=None, ego_hold=True, v_ee_max=None, table_floor=True,
                  cup_pose=None, cube=None, ego_extra_kps="cfg", right_table_kps="cfg", sensing=None):
        """Deployment path. Sensing = robot joint state (+ payload pose); everything else
        from the robot-only model. cup_pose=(p(3), R(3,3)) of the held cup (scene body
        cfg['cup_body'] mass/inertia); cube=(scene_body_id, p, R) of a welded cube or None.
        sensing: optional (qpos_lean, qvel_lean) to plan from a given robot state instead of
        syncing from scene_data (rt_pipe forward prediction)."""
        t0 = time.perf_counter()
        cfg = self._lean_cfg
        lean = cfg["lean"]
        if sensing is None:
            lean.sync_from_scene(scene_data)
        else:
            lean.set_state(sensing[0], sensing[1])
        lean.forward()
        Mfull = lean.full_M()
        t1 = time.perf_counter()
        return self._step_core(lean.data, Mfull, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, t0, t1,
                               left_ee_body=left_ee_body, tilt_max_rad=tilt_max_rad, leash_ee_body=leash_ee_body,
                               leash_max_m=leash_max_m, cup_cone_facets=cup_cone_facets, ego_hold=ego_hold,
                               v_ee_max=v_ee_max, table_floor=table_floor, cup_pose=cup_pose, cube=cube,
                               ego_extra_kps=ego_extra_kps, right_table_kps=right_table_kps)

    def _step_core(self, ld, Mfull, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, t0, t1, **opts):
        """step_lean after the robot-only forward: `ld` is anything with the robot-only MjData
        fields (qpos, qvel, xpos, xmat, cdof, cdof_dot, cvel, subtree_com, qfrc_bias,
        qfrc_passive) -- the live lean MjData, or a snapshot of it (rt_pipe)."""
        args = self._prepare(ld, Mfull, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, **opts)
        return self._run(args, t0, t1)

    def _prepare(self, ld, Mfull, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, *,
                 left_ee_body=None, tilt_max_rad=None, leash_ee_body=None, leash_max_m=None,
                 cup_cone_facets=None, ego_hold=True, v_ee_max=None, table_floor=True,
                 cup_pose=None, cube=None, ego_extra_kps="cfg", right_table_kps="cfg",
                 a_prev=None, a_queue=None):
        """Python prologue of the lean step: everything rt_full needs EXCEPT the workspace, as a
        self-contained argument tuple (the small per-step parameter arrays are private copies),
        so it can be built on one thread and executed (_run) on another. a_prev=(a_e, a_i) /
        a_queue=[(a_e, a_i), ...] override self.a_*_prev / self._a_queue (rt_pipe)."""
        cfg = self._lean_cfg
        lean = cfg["lean"]
        ldofs, rdofs = cfg["ldofs"], cfg["rdofs"]
        q0_e = ld.qpos[lean.model.jnt_qposadr[lean.model.dof_jntid[ldofs]]].copy()
        qd0_e = ld.qvel[ldofs].copy()
        q0_i = ld.qpos[lean.model.jnt_qposadr[lean.model.dof_jntid[rdofs]]].copy()
        qd0_i = ld.qvel[rdofs].copy()
        flg = np.zeros(K.N_FL, np.int64)
        dq_e_pred = dq_i_pred = self._zNV
        if self.delay_steps > 0:
            if a_queue is None:
                q0_e, qd0_e, q0_i, qd0_i, dq_e_pred, dq_i_pred = self._predict_over_delay(q0_e, qd0_e, q0_i, qd0_i)
            else:
                saved = self._a_queue
                self._a_queue = list(a_queue)
                try:
                    q0_e, qd0_e, q0_i, qd0_i, dq_e_pred, dq_i_pred = self._predict_over_delay(q0_e, qd0_e, q0_i, qd0_i)
                finally:
                    self._a_queue = saved
            flg[K.FL_DELAY_ON] = 1
        # payloads
        sm = cfg["scene_model"]
        cup_p, cup_R, cup_mI = self._z3, self._eye3, np.zeros(4)
        if cup_pose is not None:
            cb = cfg["cup_body"]
            cup_mI[0] = sm.body_mass[cb]
            cup_mI[1:4] = sm.body_inertia[cb]
            cup_p = np.ascontiguousarray(cup_pose[0], np.float64)
            cup_R = np.ascontiguousarray(np.asarray(cup_pose[1], np.float64).reshape(3, 3))
            flg[K.FL_CUP_ON] = 1
        cube_p, cube_R, cube_mI = self._z3, self._eye3, np.zeros(4)
        if cube is not None:
            cbid, cp, cR = cube
            cube_mI[0] = sm.body_mass[cbid]
            cube_mI[1:4] = sm.body_inertia[cbid]
            cube_p = np.ascontiguousarray(cp, np.float64)
            cube_R = np.ascontiguousarray(np.asarray(cR, np.float64).reshape(3, 3))
            flg[K.FL_CUBE_ON] = 1
        # keypoint sets
        extra = cfg["ego_extra"] if isinstance(ego_extra_kps, str) else ego_extra_kps
        e_body, e_off, e_r, e_mask = self._ego_pack(extra)
        i_body, i_off, i_r, i_mask = cfg["int_kps"]
        rtk = cfg["rtab"] if isinstance(right_table_kps, str) else right_table_kps
        tab_on = bool(table_floor) and cfg["ltab"] is not None and rtk is not None
        if tab_on:
            lt = self._pack(cfg["ltab"], ldofs)
            rt = self._pack(rtk, rdofs)
        else:
            lt = rt = (np.zeros(1, np.int64), np.zeros((1, 3)), np.zeros(1), np.zeros((1, self.NV), np.int64))
        self._ensure_ws(len(lt[0]), len(rt[0]), len(e_body), len(i_body))
        g = np.zeros(K.N_GI, np.int64)
        g[K.GI_ROOT] = cfg["root"]
        g[K.GI_LEFT_EE] = cfg["lee"]
        g[K.GI_LEASH_BODY] = cfg["lb"]
        g[K.GI_SPEED_BODY] = cfg["sb"]
        g[K.GI_E] = len(e_body)
        g[K.GI_I] = len(i_body)
        g[K.GI_NTAB_L] = len(lt[0]) if tab_on else 0
        g[K.GI_NTAB_R] = len(rt[0]) if tab_on else 0
        g[K.GI_KP_RADIUS] = 1 if cfg["kp_radius"] else 0
        g[K.GI_TAB_GRAVFIX] = 1 if cfg["gravfix"] else 0
        gp = np.zeros(K.N_GF)
        gp[K.GF_DMIN] = sc.D_MIN
        gp[K.GF_ZFLOOR] = 0.0 if cfg["table_z"] is None else float(cfg["table_z"])
        gp[K.GF_MARGIN] = cfg["table_margin"]
        gp[K.GF_ALPHA1] = cfg["alpha1"]
        gp[K.GF_ALPHA2] = cfg["alpha2"]
        # options (same resolution as FLMPC.step)
        ee_body = left_ee_body if left_ee_body is not None else self.left_ee_body
        tilt_max = tilt_max_rad if tilt_max_rad is not None else self.tilt_max_rad
        tilt_on = ee_body is not None and tilt_max is not None
        n_facets = self.cup_cone_facets if cup_cone_facets is None else int(cup_cone_facets)
        if tilt_on and self._cup_axis_local is None:
            R_ee0 = ld.xmat[cfg["lee"]].reshape(3, 3).copy()
            self._cup_axis_local = R_ee0.T @ np.array([0.0, 0.0, 1.0])
        axis_local = self._cup_axis_local if self._cup_axis_local is not None else self._z3
        leash_body = leash_ee_body if leash_ee_body is not None else self.leash_ee_body
        leash_max = leash_max_m if leash_max_m is not None else self.leash_max_m
        leash_on = leash_body is not None and leash_max is not None
        v_cap = v_ee_max if v_ee_max is not None else self.ee_speed_cap
        speed_on = v_cap is not None and self.ee_speed_body_int is not None
        W_pe_eff = self.W["pe"] if ego_hold else 0.0
        self.last_W_pos_ego_eff = W_pe_eff
        self._set_params(tilt_on, n_facets if (n_facets and n_facets > 0) else 0, self.cone_mode == "true",
                         leash_on, tab_on, 0, 0, speed_on, W_pe_eff, self.H_MARGIN, tilt_max, leash_max, v_cap)
        J0 = self._J0_for(W_pe_eff)
        a_e_prev, a_i_prev = (self.a_ego_prev, self.a_int_prev) if a_prev is None else a_prev
        return (self._ci, self._cf, self._pi.copy(), self._pf.copy(), self._cq, self._cqd, self._tvec,
                self._coll_steps, self._tilt_steps, self._speed_steps, g, gp, flg,
                ld.xpos, ld.xmat, ld.cdof, ld.cdof_dot, ld.cvel, ld.subtree_com, ld.qvel, ld.qfrc_bias,
                ld.qfrc_passive, Mfull,
                ldofs, rdofs, cfg["lmask_ee"], cfg["lmask_leash"], cfg["rmask_speed"],
                e_body, e_off, e_r, e_mask, i_body, i_off, i_r, i_mask,
                lt[0], lt[1], lt[2], lt[3], rt[0], rt[1], rt[2], rt[3],
                cup_p, cup_R, cup_mI, cube_p, cube_R, cube_mI, cfg["rmask_cube"], cfg["gvec"], cfg["rb"],
                np.ascontiguousarray(axis_local, np.float64), q0_e, qd0_e, q0_i, qd0_i, dq_e_pred, dq_i_pred,
                np.ascontiguousarray(q_ego_ref, np.float64), np.ascontiguousarray(qd_ego_ref, np.float64),
                np.ascontiguousarray(q_int_ref, np.float64), np.ascontiguousarray(qd_int_ref, np.float64),
                np.array(a_e_prev, np.float64), np.array(a_i_prev, np.float64), self._tau_max_arr, J0)

    def _run(self, args, t0, t1):
        """Execute a prepared step: ONE compiled call (rt_kernels.rt_full) + bookkeeping."""
        ws = self._ws
        t2 = time.perf_counter()
        K.rt_full(*args, ws.W, ws.oi, ws.xout, ws.a_e0, ws.a_i0, ws.tau_e, ws.tau_i, ws.tau_lim_e, ws.tau_lim_i)
        t3 = time.perf_counter()
        if int(ws.oi[K.OI_TAUMAX_RELAXED]):
            self.n_taumax_relaxed += 1
        return self._post(ws.oi, ws.M_e, ws.c_e, ws.M_i, ws.c_i, ws.tau_lim_e, ws.tau_lim_i,
                          t1 - t0, t3 - t2, t3 - t0)
