"""Numba kernels for the real-time FL-MPC (rt_flmpc.py). SIMULATION / OFFLINE math only.

Every kernel is @njit(cache=True, nogil=True), allocation-free (all outputs go to
caller-owned, preallocated, C-contiguous workspace arrays) and replicates the math,
row ORDER and signs of mpc_fl.FLMPC.step exactly (see that method; row blocks:
[collision K*NCS][torque box 4*NV][tilt/cone][leash][table e][table i][speed cap]).

Phi_q(t) = cq[t, b] * I on block b and Phi_qd(t) = cqd[t, b] * I (mpc_fl.di_coeffs),
so every "X @ Phi(t)" product in FLMPC.step is a single scalar product per entry;
the kernels use that structure directly instead of dense (6 x n_blk) matmuls.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from rt_qp import (gi_core, gi_init_cold, gi_init_slack, ST_OPTIMAL, ST_INFEASIBLE, ST_ITERCAP)

# ---- int config indices (ci) ----
CI_NC, CI_L, CI_NV, CI_NP, CI_NBLK, CI_K, CI_IT_HARD, CI_IT_RELAX = range(8)
N_CI = 8
# ---- float config indices (cf) ----
CF_WVE, CF_WPI, CF_WVI, CF_WR, CF_WJ, CF_ABOX, CF_RHO, CF_EPS, CF_SMAX, CF_GACC, CF_SIGMA = range(11)
N_CF = 11
# ---- per-step int params (pi) ----
PI_TILT_ON, PI_NFACETS, PI_CONE_TRUE, PI_LEASH_ON, PI_TAB_ON, PI_SPEED_ON, PI_UDIR_ON, PI_NTAB_E, PI_NTAB_I, \
    PI_VEC_TAU = range(10)
N_PI = 10
# ---- per-step float params (pf) ----
PF_WPE, PF_HMARGIN, PF_TILTMAX, PF_LEASHMAX, PF_VCAP = range(5)
N_PF = 5
# ---- output info (oi) ----
OI_HARD_STATUS, OI_HARD_IT, OI_HARD_Q, OI_FALLBACK, OI_RELAX_STATUS, OI_RELAX_IT, OI_RELAX_Q, OI_M, OI_NS, \
    OI_HARD_DROPS, OI_RELAX_DROPS, OI_TAUMAX_RELAXED, OI_K = range(13)
N_OI = 13


@njit(cache=True, nogil=True)
def assemble_rows(ci, cf, pi, pf, cq, cqd, tvec, coll_steps, tilt_steps, speed_steps,
                  q0_e, qd0_e, q0_i, qd0_i, M_e, c_e, M_i, c_i,
                  q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, a_ego_prev, a_int_prev,
                  n_arr, Je_arr, Ji_arr, h0_arr,
                  tau_lim_e, tau_lim_i,
                  tilt0, Jr_ee, Jp_ee, Jp_leash, a_tab_e, b_tab_e, a_tab_i, b_tab_i, Jp_int,
                  f, A, b, slack_rows, scr):
    """f (ntot), A (m x ntot), b (m) exactly as FLMPC.step builds them (H is
    state-independent and cached by the caller). Returns (m, ns, m_coll) and fills
    slack_rows[:ns] with the rows the relaxed fallback slackens."""
    NC = ci[CI_NC]
    NV = ci[CI_NV]
    NP = ci[CI_NP]
    n_blk = ci[CI_NBLK]
    ntot = 2 * n_blk
    W_pe = pf[PF_WPE]
    W_ve = cf[CF_WVE]
    W_pi = cf[CF_WPI]
    W_vi = cf[CF_WVI]
    W_j = cf[CF_WJ]
    H_MARGIN = pf[PF_HMARGIN]

    # ---------------- f (tracking + jerk), _assemble_fl order ----------------
    for j in range(ntot):
        f[j] = 0.0
    sp_e = -2.0 * W_pe
    sv_e = -2.0 * W_ve
    sp_i = -2.0 * W_pi
    sv_i = -2.0 * W_vi
    for t in range(1, NP + 1):
        tt = tvec[t]
        for bb in range(NC):
            cqb = cq[t, bb]
            cqdb = cqd[t, bb]
            o = bb * NV
            for j in range(NV):
                be = q_ego_ref[t, j] - (q0_e[j] + tt * qd0_e[j])
                f[o + j] += sp_e * (cqb * be)
        for bb in range(NC):
            cqdb = cqd[t, bb]
            o = bb * NV
            for j in range(NV):
                bev = qd_ego_ref[t, j] - qd0_e[j]
                f[o + j] += sv_e * (cqdb * bev)
        for bb in range(NC):
            cqb = cq[t, bb]
            o = n_blk + bb * NV
            for j in range(NV):
                bi = q_int_ref[t, j] - (q0_i[j] + tt * qd0_i[j])
                f[o + j] += sp_i * (cqb * bi)
        for bb in range(NC):
            cqdb = cqd[t, bb]
            o = n_blk + bb * NV
            for j in range(NV):
                biv = qd_int_ref[t, j] - qd0_i[j]
                f[o + j] += sv_i * (cqdb * biv)
    sj = -2.0 * W_j
    for j in range(NV):
        f[j] += sj * a_ego_prev[j]
    for j in range(NV):
        f[n_blk + j] += sj * a_int_prev[j]

    # ---------------- (1) collision rows ----------------
    K = n_arr.shape[0]
    NCS = coll_steps.shape[0]
    row = 0
    ce = scr[0]
    cii = scr[1]
    for k in range(K):
        for j in range(NV):
            ce[j] = n_arr[k, 0] * Je_arr[k, 0, j] + n_arr[k, 1] * Je_arr[k, 1, j] + n_arr[k, 2] * Je_arr[k, 2, j]
            cii[j] = n_arr[k, 0] * Ji_arr[k, 0, j] + n_arr[k, 1] * Ji_arr[k, 1, j] + n_arr[k, 2] * Ji_arr[k, 2, j]
        h0 = h0_arr[k]
        for si in range(NCS):
            t = coll_steps[si]
            for c in range(ntot):
                A[row, c] = 0.0
            for bb in range(NC):
                cqb = cq[t, bb]
                for j in range(NV):
                    A[row, bb * NV + j] = -(ce[j] * cqb)
                    A[row, n_blk + bb * NV + j] = cii[j] * cqb
            de = 0.0
            di = 0.0
            for j in range(NV):
                de += ce[j] * (qd0_e[j] * tvec[t])
                di += cii[j] * (qd0_i[j] * tvec[t])
            b[row] = h0 - H_MARGIN + (de - di)
            slack_rows[row] = row
            row += 1
    m_coll = row
    ns = m_coll

    # ---------------- (2) torque box ----------------
    for j in range(NV):
        for c in range(ntot):
            A[row, c] = 0.0
            A[row + 1, c] = 0.0
        for c in range(NV):
            A[row, c] = M_e[j, c]
            A[row + 1, c] = -M_e[j, c]
        b[row] = tau_lim_e[j] - c_e[j]
        b[row + 1] = tau_lim_e[j] + c_e[j]
        row += 2
    for j in range(NV):
        for c in range(ntot):
            A[row, c] = 0.0
            A[row + 1, c] = 0.0
        for c in range(NV):
            A[row, n_blk + c] = M_i[j, c]
            A[row + 1, n_blk + c] = -M_i[j, c]
        b[row] = tau_lim_i[j] - c_i[j]
        b[row + 1] = tau_lim_i[j] + c_i[j]
        row += 2

    # ---------------- (3) cup upright: box / true cone / eff-gravity cone ----------------
    if pi[PI_TILT_ON] != 0:
        tilt_max = pf[PF_TILTMAX]
        nf = pi[PI_NFACETS]
        for ti in range(tilt_steps.shape[0]):
            t = tilt_steps[ti]
            # drift terms Jr[ax] @ (qd0_e * t)
            d0 = 0.0
            d1 = 0.0
            for j in range(NV):
                dr = qd0_e[j] * tvec[t]
                d0 += Jr_ee[0, j] * dr
                d1 += Jr_ee[1, j] * dr
            c0 = tilt0[0] + d0
            c1 = tilt0[1] + d1
            if nf > 0 and pi[PI_CONE_TRUE] != 0:
                rhs = tilt_max * math.cos(math.pi / nf)
                for kf in range(nf):
                    ux = math.cos(2.0 * math.pi * kf / nf)
                    uy = math.sin(2.0 * math.pi * kf / nf)
                    for c in range(ntot):
                        A[row, c] = 0.0
                    for bb in range(NC):
                        cqb = cq[t, bb]
                        for j in range(NV):
                            A[row, bb * NV + j] = ux * (Jr_ee[0, j] * cqb) + uy * (Jr_ee[1, j] * cqb)
                    b[row] = rhs - (ux * c0 + uy * c1)
                    row += 1
            elif nf > 0:
                inv_g = 1.0 / cf[CF_GACC]
                for kf in range(nf):
                    ux = math.cos(2.0 * math.pi * kf / nf)
                    uy = math.sin(2.0 * math.pi * kf / nf)
                    for c in range(ntot):
                        A[row, c] = 0.0
                    for bb in range(NC):
                        cqb = cq[t, bb]
                        for j in range(NV):
                            A[row, bb * NV + j] = ux * (Jr_ee[0, j] * cqb) + uy * (Jr_ee[1, j] * cqb)
                    for j in range(NV):
                        A[row, j] += inv_g * (ux * Jp_ee[0, j] + uy * Jp_ee[1, j])
                    b[row] = tilt_max - (ux * c0 + uy * c1)
                    row += 1
            else:
                for ax in range(2):
                    drift = c0 if ax == 0 else c1
                    for c in range(ntot):
                        A[row, c] = 0.0
                        A[row + 1, c] = 0.0
                    for bb in range(NC):
                        cqb = cq[t, bb]
                        for j in range(NV):
                            v = Jr_ee[ax, j] * cqb
                            A[row, bb * NV + j] = v
                            A[row + 1, bb * NV + j] = -v
                    b[row] = tilt_max - drift
                    b[row + 1] = tilt_max + drift
                    row += 2

    # ---------------- (4) leash ----------------
    if pi[PI_LEASH_ON] != 0:
        lmax = pf[PF_LEASHMAX]
        for ti in range(tilt_steps.shape[0]):
            t = tilt_steps[ti]
            tt = tvec[t]
            for ax in range(3):
                const = 0.0
                for j in range(NV):
                    const += Jp_leash[ax, j] * ((q0_e[j] + tt * qd0_e[j]) - q_ego_ref[0, j])
                for c in range(ntot):
                    A[row, c] = 0.0
                    A[row + 1, c] = 0.0
                for bb in range(NC):
                    cqb = cq[t, bb]
                    for j in range(NV):
                        v = Jp_leash[ax, j] * cqb
                        A[row, bb * NV + j] = v
                        A[row + 1, bb * NV + j] = -v
                b[row] = lmax - const
                b[row + 1] = lmax + const
                row += 2

    # ---------------- (5) table floor (torque-convention rows -> accel rows) ----------------
    if pi[PI_TAB_ON] != 0:
        for arm in range(2):
            nr = pi[PI_NTAB_E] if arm == 0 else pi[PI_NTAB_I]
            base = 0 if arm == 0 else n_blk
            for r in range(nr):
                for c in range(ntot):
                    A[row, c] = 0.0
                bt = 0.0
                for c in range(NV):
                    s = 0.0
                    for k in range(NV):
                        if arm == 0:
                            s += a_tab_e[r, k] * M_e[k, c]
                        else:
                            s += a_tab_i[r, k] * M_i[k, c]
                    A[row, base + c] = s
                for k in range(NV):
                    if arm == 0:
                        bt += a_tab_e[r, k] * c_e[k]
                    else:
                        bt += a_tab_i[r, k] * c_i[k]
                b[row] = (b_tab_e[r] if arm == 0 else b_tab_i[r]) - bt
                row += 1

    # ---------------- (6) EE speed cap (int arm) ----------------
    if pi[PI_SPEED_ON] != 0:
        vcap = pf[PF_VCAP]
        vn0 = 0.0
        vn1 = 0.0
        vn2 = 0.0
        for j in range(NV):
            vn0 += Jp_int[0, j] * qd0_i[j]
            vn1 += Jp_int[1, j] * qd0_i[j]
            vn2 += Jp_int[2, j] * qd0_i[j]
        vnorm = math.sqrt(vn0 * vn0 + vn1 * vn1 + vn2 * vn2)
        udir = vnorm > 1e-3
        u0 = vn0 / vnorm if udir else 0.0
        u1 = vn1 / vnorm if udir else 0.0
        u2 = vn2 / vnorm if udir else 0.0
        pi[PI_UDIR_ON] = 1 if udir else 0
        for si in range(speed_steps.shape[0]):
            t = speed_steps[si]
            for ax in range(3):
                vax = vn0 if ax == 0 else (vn1 if ax == 1 else vn2)
                for c in range(ntot):
                    A[row, c] = 0.0
                    A[row + 1, c] = 0.0
                for bb in range(NC):
                    cqdb = cqd[t, bb]
                    for j in range(NV):
                        v = Jp_int[ax, j] * cqdb
                        A[row, n_blk + bb * NV + j] = v
                        A[row + 1, n_blk + bb * NV + j] = -v
                b[row] = vcap - vax
                b[row + 1] = vcap + vax
                slack_rows[ns] = row
                slack_rows[ns + 1] = row + 1
                ns += 2
                row += 2
            if udir:
                for c in range(ntot):
                    A[row, c] = 0.0
                for bb in range(NC):
                    cqdb = cqd[t, bb]
                    for j in range(NV):
                        A[row, n_blk + bb * NV + j] = (u0 * (Jp_int[0, j] * cqdb) + u1 * (Jp_int[1, j] * cqdb)
                                                       + u2 * (Jp_int[2, j] * cqdb))
                b[row] = vcap - (u0 * vn0 + u1 * vn1 + u2 * vn2)
                slack_rows[ns] = row
                ns += 1
                row += 1
    return row, ns, m_coll


@njit(cache=True, nogil=True)
def torque_safe_scale(M_arm, c_arm, a0, tau_lim, out):
    """== FLMPC._torque_safe_scale (largest lambda in [0,1] with |c + lambda M a0| <= tau_lim)."""
    NV = a0.shape[0]
    lam = 1.0
    for j in range(NV):
        dj = 0.0
        for k in range(NV):
            dj += M_arm[j, k] * a0[k]
        if dj == 0.0:
            continue
        tm = tau_lim[j]
        lam_j = ((tm - c_arm[j]) / dj) if dj > 0.0 else ((-tm - c_arm[j]) / dj)
        if lam_j < lam:
            lam = lam_j
    lam = max(0.0, min(1.0, lam))
    for j in range(NV):
        out[j] = lam * a0[j]
    return lam


@njit(cache=True, nogil=True)
def solve_with_fallback(ci, cf, m, ns, ntot, A, b, f, slack_rows, J0, J, R, act, u, y, iact, d, z, r,
                        sidx, g_aug, lo, hi, stats, oi, xout, rn):
    """Hard QP (GI, capped) -> if not OPTIMAL: relaxed QP (collision + speed rows
    slackened, EVERYTHING ELSE incl. the torque box HARD; GI started from the
    all-slack-bounds-active dual-feasible point; capped) -> if that is not OPTIMAL
    either: a = 0. Writes the (ntot) solution to xout. oi gets statuses/iterations."""
    a_box = cf[CF_ABOX]
    for i in range(m):
        sidx[i] = -1
    for j in range(ntot):
        lo[j] = -a_box
        hi[j] = a_box
    gi_init_cold(ntot, J0, f, J, y, iact, m + 2 * ntot)
    st = gi_core(ntot, 0, m, A, b, sidx, f, lo, hi, J, R, act, u, y, iact, d, z, r, 0, ci[CI_IT_HARD], stats, rn)
    oi[OI_HARD_STATUS] = st
    oi[OI_HARD_IT] = stats[0]
    oi[OI_HARD_Q] = stats[1]
    oi[OI_HARD_DROPS] = stats[2]
    oi[OI_M] = m
    oi[OI_NS] = ns
    oi[OI_RELAX_STATUS] = -1
    oi[OI_RELAX_IT] = 0
    oi[OI_RELAX_Q] = 0
    oi[OI_RELAX_DROPS] = 0
    if st == ST_OPTIMAL:
        for j in range(ntot):
            xout[j] = y[j]
        oi[OI_FALLBACK] = 0
        return st
    # ---- relaxed fallback (FLMPC._relaxed_fallback semantics) ----
    for k in range(ns):
        sidx[slack_rows[k]] = k
    for j in range(ntot):
        g_aug[j] = f[j]
    for k in range(ns):
        g_aug[ntot + k] = cf[CF_RHO]
        lo[ntot + k] = 0.0
        hi[ntot + k] = cf[CF_SMAX]
    q0 = gi_init_slack(ntot, ns, m, J0, cf[CF_SIGMA], g_aug, J, R, act, u, y, iact)
    st2 = gi_core(ntot, ns, m, A, b, sidx, g_aug, lo, hi, J, R, act, u, y, iact, d, z, r, q0,
                  ci[CI_IT_RELAX], stats, rn)
    oi[OI_RELAX_STATUS] = st2
    oi[OI_RELAX_IT] = stats[0]
    oi[OI_RELAX_Q] = stats[1]
    oi[OI_RELAX_DROPS] = stats[2]
    if st2 == ST_OPTIMAL:
        for j in range(ntot):
            xout[j] = y[j]
        oi[OI_FALLBACK] = 1
    else:
        for j in range(ntot):
            xout[j] = 0.0
        oi[OI_FALLBACK] = 2
    return st


@njit(cache=True, nogil=True)
def rt_core(ci, cf, pi, pf, cq, cqd, tvec, coll_steps, tilt_steps, speed_steps,
            q0_e, qd0_e, q0_i, qd0_i, M_e, c_e, M_i, c_i,
            q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, a_ego_prev, a_int_prev,
            n_arr, Je_arr, Ji_arr, h0_arr, tau_lim_e, tau_lim_i,
            tilt0, Jr_ee, Jp_ee, Jp_leash, a_tab_e, b_tab_e, a_tab_i, b_tab_i, Jp_int,
            J0, f, A, b, slack_rows, J, R, act, u, y, iact, d, z, r, sidx, g_aug, lo, hi, stats, oi,
            xout, a_e0, a_i0, tau_e, tau_i, scr, rn):
    """ONE compiled FL-MPC step after the dynamics/geometry inputs: rows + QP + relaxed
    fallback + exact torque-box backstop + FL torque. Returns hard-QP status."""
    NV = ci[CI_NV]
    n_blk = ci[CI_NBLK]
    ntot = 2 * n_blk
    m, ns, m_coll = assemble_rows(ci, cf, pi, pf, cq, cqd, tvec, coll_steps, tilt_steps, speed_steps,
                                  q0_e, qd0_e, q0_i, qd0_i, M_e, c_e, M_i, c_i,
                                  q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, a_ego_prev, a_int_prev,
                                  n_arr, Je_arr, Ji_arr, h0_arr, tau_lim_e, tau_lim_i,
                                  tilt0, Jr_ee, Jp_ee, Jp_leash, a_tab_e, b_tab_e, a_tab_i, b_tab_i, Jp_int,
                                  f, A, b, slack_rows, scr)
    oi[OI_K] = n_arr.shape[0]
    st = solve_with_fallback(ci, cf, m, ns, ntot, A, b, f, slack_rows, J0, J, R, act, u, y, iact, d, z, r,
                             sidx, g_aug, lo, hi, stats, oi, xout, rn)
    for j in range(NV):
        a_e0[j] = xout[j]
        a_i0[j] = xout[n_blk + j]
    if st != ST_OPTIMAL:
        # FINAL EXACT TORQUE-BOX BACKSTOP (== FLMPC: applied whenever the hard QP was not solved)
        torque_safe_scale(M_e, c_e, xout[0:NV], tau_lim_e, a_e0)
        torque_safe_scale(M_i, c_i, xout[n_blk:n_blk + NV], tau_lim_i, a_i0)
    for j in range(NV):
        se = c_e[j]
        si = c_i[j]
        for k in range(NV):
            se += M_e[j, k] * a_e0[k]
            si += M_i[j, k] * a_i0[k]
        tau_e[j] = se
        tau_i[j] = si
    return st


# ===========================================================================
# geometry from a robot-only MjData (lean dynamics): keypoints, Jacobians,
# top-K arm-arm pairs, table-floor rows, payload reflection
# ===========================================================================
@njit(cache=True, nogil=True, inline="always")
def _world_pt(xpos, xmat, bid, off, p):
    for a in range(3):
        p[a] = xpos[bid, a] + (xmat[bid, 3 * a] * off[0] + xmat[bid, 3 * a + 1] * off[1]
                               + xmat[bid, 3 * a + 2] * off[2])


@njit(cache=True, nogil=True)
def _jac_pt(p, bid, root, dofs, mask, cdof, cdof_dot, cvel, subtree_com, J, Jd, want_jd):
    """mj_jac / mj_jacDot (translational) at world point p on body bid, restricted
    to the arm dof columns `dofs` (mask[c]=1 iff dofs[c] is in bid's chain). Same
    arithmetic as fast_barriers._jac_jacdot_batch (MuJoCo engine_support.c)."""
    o0 = p[0] - subtree_com[root, 0]
    o1 = p[1] - subtree_com[root, 1]
    o2 = p[2] - subtree_com[root, 2]
    w0 = cvel[bid, 0]
    w1 = cvel[bid, 1]
    w2 = cvel[bid, 2]
    c0 = o1 * w2 - o2 * w1
    c1 = o2 * w0 - o0 * w2
    c2 = o0 * w1 - o1 * w0
    pv0 = cvel[bid, 3] - c0
    pv1 = cvel[bid, 4] - c1
    pv2 = cvel[bid, 5] - c2
    for c in range(dofs.shape[0]):
        if mask[c] == 0:
            J[0, c] = 0.0
            J[1, c] = 0.0
            J[2, c] = 0.0
            if want_jd:
                Jd[0, c] = 0.0
                Jd[1, c] = 0.0
                Jd[2, c] = 0.0
            continue
        i = dofs[c]
        a0 = cdof[i, 0]
        a1 = cdof[i, 1]
        a2 = cdof[i, 2]
        J[0, c] = cdof[i, 3] + (a1 * o2 - a2 * o1)
        J[1, c] = cdof[i, 4] + (a2 * o0 - a0 * o2)
        J[2, c] = cdof[i, 5] + (a0 * o1 - a1 * o0)
        if want_jd:
            d0 = cdof_dot[i, 0]
            d1 = cdof_dot[i, 1]
            d2 = cdof_dot[i, 2]
            t10 = d1 * o2 - d2 * o1
            t11 = d2 * o0 - d0 * o2
            t12 = d0 * o1 - d1 * o0
            t20 = a1 * pv2 - a2 * pv1
            t21 = a2 * pv0 - a0 * pv2
            t22 = a0 * pv1 - a1 * pv0
            Jd[0, c] = cdof_dot[i, 3] + t10 + t20
            Jd[1, c] = cdof_dot[i, 4] + t11 + t21
            Jd[2, c] = cdof_dot[i, 5] + t12 + t22


@njit(cache=True, nogil=True)
def _body_jac(bid, root, dofs, mask, xpos, cdof, subtree_com, Jp, Jr):
    """body_jac_full(model, data, bid)[0/1][:, dofs] (origin of the body)."""
    o0 = xpos[bid, 0] - subtree_com[root, 0]
    o1 = xpos[bid, 1] - subtree_com[root, 1]
    o2 = xpos[bid, 2] - subtree_com[root, 2]
    for c in range(dofs.shape[0]):
        if mask[c] == 0:
            for a in range(3):
                Jp[a, c] = 0.0
                Jr[a, c] = 0.0
            continue
        i = dofs[c]
        a0 = cdof[i, 0]
        a1 = cdof[i, 1]
        a2 = cdof[i, 2]
        Jp[0, c] = cdof[i, 3] + (a1 * o2 - a2 * o1)
        Jp[1, c] = cdof[i, 4] + (a2 * o0 - a0 * o2)
        Jp[2, c] = cdof[i, 5] + (a0 * o1 - a1 * o0)
        Jr[0, c] = a0
        Jr[1, c] = a1
        Jr[2, c] = a2


@njit(cache=True, nogil=True)
def _inv6(M, Minv, work):
    """Gauss-Jordan inverse with partial pivoting (NV x NV, small)."""
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            work[i, j] = M[i, j]
            Minv[i, j] = 1.0 if i == j else 0.0
    for col in range(n):
        piv = col
        best = abs(work[col, col])
        for rr in range(col + 1, n):
            if abs(work[rr, col]) > best:
                best = abs(work[rr, col])
                piv = rr
        if piv != col:
            for j in range(n):
                tmp = work[col, j]
                work[col, j] = work[piv, j]
                work[piv, j] = tmp
                tmp = Minv[col, j]
                Minv[col, j] = Minv[piv, j]
                Minv[piv, j] = tmp
        dg = work[col, col]
        for j in range(n):
            work[col, j] /= dg
            Minv[col, j] /= dg
        for rr in range(n):
            if rr != col:
                fct = work[rr, col]
                if fct != 0.0:
                    for j in range(n):
                        work[rr, j] -= fct * work[col, j]
                        Minv[rr, j] -= fct * Minv[col, j]


@njit(cache=True, nogil=True)
def payload_terms(ee_bid, root, dofs, mask, xpos, cdof, subtree_com, p_obj, R_obj, mass, inertia_diag, gvec,
                  Mout, bout, Jp, Jr, Jb, Iw):
    """== sim_common.payload_reflected_terms (rigid-grasp reflection of an object's
    mass/inertia + gravity into one arm's joint space). Mout/bout are ADDED to."""
    NV = dofs.shape[0]
    _body_jac(ee_bid, root, dofs, mask, xpos, cdof, subtree_com, Jp, Jr)
    r0 = p_obj[0] - xpos[ee_bid, 0]
    r1 = p_obj[1] - xpos[ee_bid, 1]
    r2 = p_obj[2] - xpos[ee_bid, 2]
    # Jp_box = Jp - skew(r) @ Jr ; skew(r) = [[0,-r2,r1],[r2,0,-r0],[-r1,r0,0]]
    for c in range(NV):
        Jb[0, c] = Jp[0, c] - (-r2 * Jr[1, c] + r1 * Jr[2, c])
        Jb[1, c] = Jp[1, c] - (r2 * Jr[0, c] - r0 * Jr[2, c])
        Jb[2, c] = Jp[2, c] - (-r1 * Jr[0, c] + r0 * Jr[1, c])
        Jb[3, c] = Jr[0, c]
        Jb[4, c] = Jr[1, c]
        Jb[5, c] = Jr[2, c]
    # I_world = R diag(I) R^T
    for a in range(3):
        for bb in range(3):
            s = 0.0
            for k in range(3):
                s += R_obj[a, k] * inertia_diag[k] * R_obj[bb, k]
            Iw[a, bb] = s
    for i in range(NV):
        for j in range(NV):
            s = mass * (Jb[0, i] * Jb[0, j] + Jb[1, i] * Jb[1, j] + Jb[2, i] * Jb[2, j])
            for a in range(3):
                for bb in range(3):
                    s += Jb[3 + a, i] * Iw[a, bb] * Jb[3 + bb, j]
            Mout[i, j] += s
        bout[i] += -(Jb[0, i] * (mass * gvec[0]) + Jb[1, i] * (mass * gvec[1]) + Jb[2, i] * (mass * gvec[2]))


@njit(cache=True, nogil=True)
def table_rows_arm(kp_body, kp_off, kp_r, kp_mask, n_kp, root, dofs, xpos, xmat, cdof, cdof_dot, cvel,
                   subtree_com, qvel, M_tab, bias_tab, z_floor, margin, alpha1, alpha2, kp_radius,
                   a_out, b_out, Minv, work, p, J, Jd):
    """== sim_common.table_plane_barrier_rows (torque convention a @ tau <= b)."""
    NV = dofs.shape[0]
    _inv6(M_tab, Minv, work)
    z_th = z_floor - margin
    for k in range(n_kp):
        bid = kp_body[k]
        _world_pt(xpos, xmat, bid, kp_off[k], p)
        _jac_pt(p, bid, root, dofs, kp_mask[k], cdof, cdof_dot, cvel, subtree_com, J, Jd, True)
        h = p[2] - z_th if kp_radius == 0 else p[2] - kp_r[k] - z_th
        hdot = 0.0
        hdd = 0.0
        for c in range(NV):
            qd = qvel[dofs[c]]
            hdot += J[2, c] * qd
            hdd += Jd[2, c] * qd
        psi1 = hdot + alpha1 * h
        # A_h = J[2] @ Minv
        ab = 0.0
        for c in range(NV):
            s = 0.0
            for kk in range(NV):
                s += J[2, kk] * Minv[kk, c]
            a_out[k, c] = -s
            ab += s * bias_tab[c]
        b_out[k] = -ab + hdd + alpha1 * hdot + alpha2 * psi1


@njit(cache=True, nogil=True)
def topk_pairs(e_body, e_off, e_r, e_mask, E, i_body, i_off, i_r, i_mask, I, K, d_min, root, ldofs, rdofs,
               xpos, xmat, cdof, cdof_dot, cvel, subtree_com,
               pe, pi_, hbuf, sel, n_arr, Je_arr, Ji_arr, h0_arr, hdot_arr, qvel, p, Jd):
    """All E x I sphere pairs -> the K smallest h (stable: ties by flat index, ==
    np.argsort(kind='stable')[:K] / sorted()[:K]) with their n, Je, Ji, h, hdot."""
    for e in range(E):
        _world_pt(xpos, xmat, e_body[e], e_off[e], p)
        pe[e, 0] = p[0]
        pe[e, 1] = p[1]
        pe[e, 2] = p[2]
    for i in range(I):
        _world_pt(xpos, xmat, i_body[i], i_off[i], p)
        pi_[i, 0] = p[0]
        pi_[i, 1] = p[1]
        pi_[i, 2] = p[2]
    for e in range(E):
        for i in range(I):
            d0 = pe[e, 0] - pi_[i, 0]
            d1 = pe[e, 1] - pi_[i, 1]
            d2 = pe[e, 2] - pi_[i, 2]
            dist = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            hbuf[e * I + i] = dist - ((e_r[e] + i_r[i]) + d_min)
    NPR = E * I
    Kk = K if K < NPR else NPR
    for k in range(Kk):
        best = -1
        for j in range(NPR):
            already = False
            for kk in range(k):
                if sel[kk] == j:
                    already = True
                    break
            if already:
                continue
            if best < 0 or hbuf[j] < hbuf[best]:
                best = j
        sel[k] = best
    NV = ldofs.shape[0]
    for k in range(Kk):
        j = sel[k]
        e = j // I
        i = j - e * I
        d0 = pe[e, 0] - pi_[i, 0]
        d1 = pe[e, 1] - pi_[i, 1]
        d2 = pe[e, 2] - pi_[i, 2]
        dist = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        dd = dist if dist > 1e-9 else 1e-9
        n_arr[k, 0] = d0 / dd
        n_arr[k, 1] = d1 / dd
        n_arr[k, 2] = d2 / dd
        h0_arr[k] = hbuf[j]
        p[0] = pe[e, 0]
        p[1] = pe[e, 1]
        p[2] = pe[e, 2]
        _jac_pt(p, e_body[e], root, ldofs, e_mask[e], cdof, cdof_dot, cvel, subtree_com, Je_arr[k], Jd, False)
        p[0] = pi_[i, 0]
        p[1] = pi_[i, 1]
        p[2] = pi_[i, 2]
        _jac_pt(p, i_body[i], root, rdofs, i_mask[i], cdof, cdof_dot, cvel, subtree_com, Ji_arr[k], Jd, False)
        hd = 0.0
        for a in range(3):
            ve = 0.0
            vi = 0.0
            for c in range(NV):
                ve += Je_arr[k, a, c] * qvel[ldofs[c]]
                vi += Ji_arr[k, a, c] * qvel[rdofs[c]]
            hd += n_arr[k, a] * (ve - vi)
        hdot_arr[k] = hd
    return Kk


# ===========================================================================
# the ONE compiled entry point of the lean (deployment) path
# ===========================================================================
# geometry int config (gi_) indices
GI_ROOT, GI_LEFT_EE, GI_LEASH_BODY, GI_SPEED_BODY, GI_E, GI_I, GI_NTAB_L, GI_NTAB_R, GI_KP_RADIUS, \
    GI_TAB_GRAVFIX = range(10)
N_GI = 10
# geometry per-step float (gf_) indices
GF_DMIN, GF_ZFLOOR, GF_MARGIN, GF_ALPHA1, GF_ALPHA2 = range(5)
N_GF = 5
# per-step flags (pi extension, separate array fl_)
FL_CUP_ON, FL_CUBE_ON, FL_DELAY_ON = range(3)
N_FL = 3


@njit(cache=True, nogil=True)
def rt_full(ci, cf, pi, pf, cq, cqd, tvec, coll_steps, tilt_steps, speed_steps,
            gcfg, gpar, flg,
            xpos, xmat, cdof, cdof_dot, cvel, subtree_com, qvel, qfrc_bias, qfrc_passive, Mfull,
            ldofs, rdofs, lmask_ee, lmask_leash, rmask_speed,
            e_body, e_off, e_r, e_mask, i_body, i_off, i_r, i_mask,
            lt_body, lt_off, lt_r, lt_mask, rt_body, rt_off, rt_r, rt_mask,
            cup_p, cup_R, cup_mI, cube_p, cube_R, cube_mI, cube_ee_mask, gvec, cube_ee_body,
            axis_local, q0_e, qd0_e, q0_i, qd0_i, dq_e_pred, dq_i_pred,
            q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, a_ego_prev, a_int_prev, tau_max_vec,
            J0, W, oi, xout, a_e0, a_i0, tau_e, tau_i, tau_lim_e, tau_lim_i):
    """Lean-dynamics FL-MPC step: M/c (+cup payload), top-K arm-arm pairs, table
    rows (+held-cube payload), tilt/leash/speed Jacobians, then rt_core. All
    scratch in the flat workspace tuple W (see rt_flmpc.RTWorkspace)."""
    (M_e, c_e, M_i, c_i, M_tl, b_tl, M_tr, b_tr, n_arr, Je_arr, Ji_arr, h0_arr, hdot_arr,
     tilt0, Jr_ee, Jp_ee, Jp_leash, Jr_tmp, Jp_int, a_tab_e, b_tab_e, a_tab_i, b_tab_i,
     pe, pi_, hbuf, sel, p3, J3, Jd3, Jb6, Minv, work6, M_box, b_box, Iw, scr,
     f, A, b, slack_rows, J, R, act, u, y, iact, d, z, r, sidx, g_aug, lo, hi, stats, rn) = W
    NV = ldofs.shape[0]
    root = gcfg[GI_ROOT]
    # ---------------- M, c per arm (c = qfrc_bias - qfrc_passive) ----------------
    for a in range(NV):
        for bcol in range(NV):
            M_e[a, bcol] = Mfull[ldofs[a], ldofs[bcol]]
            M_i[a, bcol] = Mfull[rdofs[a], rdofs[bcol]]
        c_e[a] = qfrc_bias[ldofs[a]] - qfrc_passive[ldofs[a]]
        c_i[a] = qfrc_bias[rdofs[a]] - qfrc_passive[rdofs[a]]
    # table-row inverse-dynamics terms (demo: M_ll + M_box, bias_l [-passive if gravfix] + bias_box)
    for a in range(NV):
        for bcol in range(NV):
            M_tl[a, bcol] = M_e[a, bcol]
            M_tr[a, bcol] = M_i[a, bcol]
        if gcfg[GI_TAB_GRAVFIX] != 0:
            b_tl[a] = c_e[a]
            b_tr[a] = c_i[a]
        else:
            b_tl[a] = qfrc_bias[ldofs[a]]
            b_tr[a] = qfrc_bias[rdofs[a]]
    if flg[FL_CUP_ON] != 0:
        # ONE payload evaluation, added to BOTH (M_e,c_e) and (M_tl,b_tl): same
        # numbers the demo adds to M_ll/bias_l and passes as M_ego_extra/bias_ego_extra.
        for a in range(NV):
            for bcol in range(NV):
                M_box[a, bcol] = 0.0
            b_box[a] = 0.0
        payload_terms(gcfg[GI_LEFT_EE], root, ldofs, lmask_ee, xpos, cdof, subtree_com, cup_p, cup_R,
                      cup_mI[0], cup_mI[1:4], gvec, M_box, b_box, Jp_ee, Jr_tmp, Jb6, Iw)
        for a in range(NV):
            for bcol in range(NV):
                M_e[a, bcol] += M_box[a, bcol]
                M_tl[a, bcol] += M_box[a, bcol]
            c_e[a] += b_box[a]
            b_tl[a] += b_box[a]
    if flg[FL_CUBE_ON] != 0:
        payload_terms(cube_ee_body, root, rdofs, cube_ee_mask, xpos, cdof, subtree_com, cube_p, cube_R,
                      cube_mI[0], cube_mI[1:4], gvec, M_tr, b_tr, Jp_int, Jr_tmp, Jb6, Iw)
    # ---------------- torque limits (scalar or per-joint F6) ----------------
    if pi[PI_VEC_TAU] != 0:
        relaxed = 0
        for a in range(NV):
            ae = abs(c_e[a])
            ai = abs(c_i[a])
            tau_lim_e[a] = tau_max_vec[a] if tau_max_vec[a] >= ae else ae
            tau_lim_i[a] = tau_max_vec[a] if tau_max_vec[a] >= ai else ai
            if tau_lim_e[a] > tau_max_vec[a] or tau_lim_i[a] > tau_max_vec[a]:
                relaxed = 1
        oi[OI_TAUMAX_RELAXED] = relaxed
    else:
        for a in range(NV):
            tau_lim_e[a] = tau_max_vec[0]
            tau_lim_i[a] = tau_max_vec[0]
        oi[OI_TAUMAX_RELAXED] = 0
    # ---------------- top-K arm-arm pairs (+ F5 first-order h0 at predicted state) ----------------
    K = ci[CI_K]
    Kk = topk_pairs(e_body, e_off, e_r, e_mask, gcfg[GI_E], i_body, i_off, i_r, i_mask, gcfg[GI_I], K,
                    gpar[GF_DMIN], root, ldofs, rdofs, xpos, xmat, cdof, cdof_dot, cvel, subtree_com,
                    pe, pi_, hbuf, sel, n_arr, Je_arr, Ji_arr, h0_arr, hdot_arr, qvel, p3, Jd3)
    if flg[FL_DELAY_ON] != 0:
        for k in range(Kk):
            se = 0.0
            si = 0.0
            for j in range(NV):
                ce_j = n_arr[k, 0] * Je_arr[k, 0, j] + n_arr[k, 1] * Je_arr[k, 1, j] + n_arr[k, 2] * Je_arr[k, 2, j]
                ci_j = n_arr[k, 0] * Ji_arr[k, 0, j] + n_arr[k, 1] * Ji_arr[k, 1, j] + n_arr[k, 2] * Ji_arr[k, 2, j]
                se += ce_j * dq_e_pred[j]
                si += ci_j * dq_i_pred[j]
            h0_arr[k] = (h0_arr[k] + se) - si
    # ---------------- cup-upright tilt anchor + Jacobians ----------------
    if pi[PI_TILT_ON] != 0 or pi[PI_LEASH_ON] != 0:
        ee = gcfg[GI_LEFT_EE]
        _body_jac(ee, root, ldofs, lmask_ee, xpos, cdof, subtree_com, Jp_ee, Jr_ee)
        if pi[PI_TILT_ON] != 0:
            # cup_axis_world = R_ee @ axis_local ; tilt0 = (-caw_y, caw_x)
            caw0 = xmat[ee, 0] * axis_local[0] + xmat[ee, 1] * axis_local[1] + xmat[ee, 2] * axis_local[2]
            caw1 = xmat[ee, 3] * axis_local[0] + xmat[ee, 4] * axis_local[1] + xmat[ee, 5] * axis_local[2]
            tilt0[0] = -caw1
            tilt0[1] = caw0
            if flg[FL_DELAY_ON] != 0:
                for ax in range(2):
                    s = 0.0
                    for j in range(NV):
                        s += Jr_ee[ax, j] * dq_e_pred[j]
                    tilt0[ax] += s
        if pi[PI_LEASH_ON] != 0:
            _body_jac(gcfg[GI_LEASH_BODY], root, ldofs, lmask_leash, xpos, cdof, subtree_com, Jp_leash, Jr_tmp)
    if pi[PI_SPEED_ON] != 0:
        _body_jac(gcfg[GI_SPEED_BODY], root, rdofs, rmask_speed, xpos, cdof, subtree_com, Jp_int, Jr_tmp)
    # ---------------- table floor rows (torque convention) ----------------
    if pi[PI_TAB_ON] != 0:
        table_rows_arm(lt_body, lt_off, lt_r, lt_mask, gcfg[GI_NTAB_L], root, ldofs, xpos, xmat, cdof, cdof_dot,
                       cvel, subtree_com, qvel, M_tl, b_tl, gpar[GF_ZFLOOR], gpar[GF_MARGIN], gpar[GF_ALPHA1],
                       gpar[GF_ALPHA2], gcfg[GI_KP_RADIUS], a_tab_e, b_tab_e, Minv, work6, p3, J3, Jd3)
        table_rows_arm(rt_body, rt_off, rt_r, rt_mask, gcfg[GI_NTAB_R], root, rdofs, xpos, xmat, cdof, cdof_dot,
                       cvel, subtree_com, qvel, M_tr, b_tr, gpar[GF_ZFLOOR], gpar[GF_MARGIN], gpar[GF_ALPHA1],
                       gpar[GF_ALPHA2], gcfg[GI_KP_RADIUS], a_tab_i, b_tab_i, Minv, work6, p3, J3, Jd3)
        pi[PI_NTAB_E] = gcfg[GI_NTAB_L]
        pi[PI_NTAB_I] = gcfg[GI_NTAB_R]
    return rt_core(ci, cf, pi, pf, cq, cqd, tvec, coll_steps, tilt_steps, speed_steps,
                   q0_e, qd0_e, q0_i, qd0_i, M_e, c_e, M_i, c_i,
                   q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, a_ego_prev, a_int_prev,
                   n_arr[:Kk], Je_arr[:Kk], Ji_arr[:Kk], h0_arr[:Kk], tau_lim_e, tau_lim_i,
                   tilt0, Jr_ee, Jp_ee, Jp_leash, a_tab_e, b_tab_e, a_tab_i, b_tab_i, Jp_int,
                   J0, f, A, b, slack_rows, J, R, act, u, y, iact, d, z, r, sidx, g_aug, lo, hi, stats, oi,
                   xout, a_e0, a_i0, tau_e, tau_i, scr, rn)


# ===========================================================================
# rt_pipe: execution-time checks on the MEASURED state (main thread, cheap, nogil)
# ===========================================================================
EX_HMIN, EX_BRAKED, EX_RECHECK_FAIL, EX_SCALED, EX_LAM_E, EX_LAM_I = range(6)
N_EX = 6


@njit(cache=True, nogil=True)
def exec_check(ci, coll_steps, DT, H_MARGIN, gcfg, gpar, flg,
               xpos, xmat, cdof, cdof_dot, cvel, subtree_com, qvel, qfrc_bias, qfrc_passive, Mfull,
               ldofs, rdofs, lmask_ee, e_body, e_off, e_r, e_mask, i_body, i_off, i_r, i_mask,
               cup_p, cup_R, cup_mI, gvec, plan_x, age, force_brake, hmin_plan, recheck_tol, brake_kd,
               tau_lim_e, tau_lim_i, W, a_e, a_i, tau_e, tau_i, ex):
    """Apply a (possibly stale) plan at the measured state x_k:
      * M, c (+ cup payload) at x_k -- the FL torque tau = M a + c is always formed with the
        CURRENT dynamics;
      * barrier re-check: top-K pairs at x_k, FL prediction (semi-implicit Euler, exact for
        q''=a) of the plan's accelerations SHIFTED by `age` steps (block (age+s)//L), first-
        order clearance h(t) at the anticipatory steps; FAIL if min h < min(H_MARGIN/2,
        hmin_plan - recheck_tol) (worse than half the margin AND worse than the plan itself
        predicted);
      * force_brake (plan too old / none) or FAIL -> brake a = -brake_kd * M^-1 qd (the demo's
        INFEAS_BRAKE_KD law, in FL form);
      * torque box: if |M a + c| exceeds tau_lim -> exact _torque_safe_scale (never redirects a).
    """
    (M_e, c_e, M_i, c_i, M_tl, b_tl, M_tr, b_tr, n_arr, Je_arr, Ji_arr, h0_arr, hdot_arr,
     tilt0, Jr_ee, Jp_ee, Jp_leash, Jr_tmp, Jp_int, a_tab_e, b_tab_e, a_tab_i, b_tab_i,
     pe, pi_, hbuf, sel, p3, J3, Jd3, Jb6, Minv, work6, M_box, b_box, Iw, scr,
     f, A, b, slack_rows, J, R, act, u, y, iact, d, z, r, sidx, g_aug, lo, hi, stats, rn) = W
    NC = ci[CI_NC]
    L = ci[CI_L]
    NV = ci[CI_NV]
    NP = ci[CI_NP]
    n_blk = ci[CI_NBLK]
    root = gcfg[GI_ROOT]
    for a in range(NV):
        for bcol in range(NV):
            M_e[a, bcol] = Mfull[ldofs[a], ldofs[bcol]]
            M_i[a, bcol] = Mfull[rdofs[a], rdofs[bcol]]
        c_e[a] = qfrc_bias[ldofs[a]] - qfrc_passive[ldofs[a]]
        c_i[a] = qfrc_bias[rdofs[a]] - qfrc_passive[rdofs[a]]
    if flg[FL_CUP_ON] != 0:
        for a in range(NV):
            for bcol in range(NV):
                M_box[a, bcol] = 0.0
            b_box[a] = 0.0
        payload_terms(gcfg[GI_LEFT_EE], root, ldofs, lmask_ee, xpos, cdof, subtree_com, cup_p, cup_R,
                      cup_mI[0], cup_mI[1:4], gvec, M_box, b_box, Jp_ee, Jr_tmp, Jb6, Iw)
        for a in range(NV):
            for bcol in range(NV):
                M_e[a, bcol] += M_box[a, bcol]
            c_e[a] += b_box[a]
    braked = force_brake
    fail = 0
    hmin = 1e9
    if braked == 0:
        Kk = topk_pairs(e_body, e_off, e_r, e_mask, gcfg[GI_E], i_body, i_off, i_r, i_mask, gcfg[GI_I], ci[CI_K],
                        gpar[GF_DMIN], root, ldofs, rdofs, xpos, xmat, cdof, cdof_dot, cvel, subtree_com,
                        pe, pi_, hbuf, sel, n_arr, Je_arr, Ji_arr, h0_arr, hdot_arr, qvel, p3, Jd3)
        # predicted joint displacement per arm over the horizon (scr rows: 0 dq_e, 1 dq_i ; Jd3 as qd buffers)
        NCS = coll_steps.shape[0]
        for j in range(NV):
            scr[0, j] = 0.0
            scr[1, j] = 0.0
            Jd3[0, j] = qvel[ldofs[j]]
            Jd3[1, j] = qvel[rdofs[j]]
        si = 0
        for s in range(NP):
            bb = (age + s) // L
            if bb > NC - 1:
                bb = NC - 1
            for j in range(NV):
                Jd3[0, j] += plan_x[bb * NV + j] * DT
                Jd3[1, j] += plan_x[n_blk + bb * NV + j] * DT
                scr[0, j] += Jd3[0, j] * DT
                scr[1, j] += Jd3[1, j] * DT
            t = s + 1
            if si < NCS and coll_steps[si] == t:
                for k in range(Kk):
                    hk = h0_arr[k]
                    for j in range(NV):
                        ce_j = n_arr[k, 0] * Je_arr[k, 0, j] + n_arr[k, 1] * Je_arr[k, 1, j] + n_arr[k, 2] * Je_arr[k, 2, j]
                        ci_j = n_arr[k, 0] * Ji_arr[k, 0, j] + n_arr[k, 1] * Ji_arr[k, 1, j] + n_arr[k, 2] * Ji_arr[k, 2, j]
                        hk += ce_j * scr[0, j] - ci_j * scr[1, j]
                    if hk < hmin:
                        hmin = hk
                si += 1
        thr = 0.5 * H_MARGIN
        if hmin_plan - recheck_tol < thr:
            thr = hmin_plan - recheck_tol
        if hmin < thr:
            fail = 1
            braked = 1
    if braked != 0:
        # a = -kd * M^-1 qd  (per arm)
        _inv6(M_e, Minv, work6)
        for j in range(NV):
            s = 0.0
            for k in range(NV):
                s -= Minv[j, k] * brake_kd * qvel[ldofs[k]]
            scr[0, j] = s
        _inv6(M_i, Minv, work6)
        for j in range(NV):
            s = 0.0
            for k in range(NV):
                s -= Minv[j, k] * brake_kd * qvel[rdofs[k]]
            scr[1, j] = s
    else:
        bb = age // L
        if bb > NC - 1:
            bb = NC - 1
        for j in range(NV):
            scr[0, j] = plan_x[bb * NV + j]
            scr[1, j] = plan_x[n_blk + bb * NV + j]
    # torque box (exact scale toward a=0 only if violated)
    over = 0
    for j in range(NV):
        te = c_e[j]
        ti = c_i[j]
        for k in range(NV):
            te += M_e[j, k] * scr[0, k]
            ti += M_i[j, k] * scr[1, k]
        if abs(te) > tau_lim_e[j] + 1e-9 or abs(ti) > tau_lim_i[j] + 1e-9:
            over = 1
    lam_e = 1.0
    lam_i = 1.0
    if over != 0:
        lam_e = torque_safe_scale(M_e, c_e, scr[0], tau_lim_e, a_e)
        lam_i = torque_safe_scale(M_i, c_i, scr[1], tau_lim_i, a_i)
    else:
        for j in range(NV):
            a_e[j] = scr[0, j]
            a_i[j] = scr[1, j]
    for j in range(NV):
        te = c_e[j]
        ti = c_i[j]
        for k in range(NV):
            te += M_e[j, k] * a_e[k]
            ti += M_i[j, k] * a_i[k]
        tau_e[j] = te
        tau_i[j] = ti
    ex[EX_HMIN] = hmin
    ex[EX_BRAKED] = braked
    ex[EX_RECHECK_FAIL] = fail
    ex[EX_SCALED] = over
    ex[EX_LAM_E] = lam_e
    ex[EX_LAM_I] = lam_i
