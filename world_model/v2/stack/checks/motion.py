"""checks/motion.py -- model rows on planned joint motions (checker design 11-05 sec 3.2 joint q/qd, sec 3.3 reach + closed chain, A1).  PURE: numpy + stack only.

Every row takes arrays at the control period (a time grid t (n,), per side q / qd (n,7)).  A row that passes is guarantee "proof" (MODEL: it holds for the
planned arrays and the stated between-sample bound); a row that fails is "refuted" with a counterexample in `worst` (side, joint, sample index, time, value, limit).

Between samples (MODEL).  The arrays are the planner's output at the control period dt.  Between samples k, k+1 the joint velocity is taken piecewise linear
(acceleration bounded by the finite difference of qd), so
    |q(t) - q_k|  <=  max(|qd_k|, |qd_{k+1}|) * dt / 2          (position excursion beyond the sample values, per joint; exact for a linear-velocity segment's
                                                                  trajectory of mean speed <= max, so a conservative bound),
    |qd(t)|       <=  |qd_k| + |qdd| * dt / 2, with |qdd| = |qd_{k+1} - qd_k| / dt   (the value between samples never exceeds the larger endpoint,
                                                                  the bound is kept as the widening term |dqd| / 2).
The position row widens its margin by margin_eff = margin + max_k |qd_k| * dt / 2 per joint; the speed row widens by max_k |qd_{k+1} - qd_k| / 2.

ASSUMED constants (A1, to be raised with the user): EPS_CC_POS, EPS_CC_ROT.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.stack.checks.row import Row

EPS_CC_POS = 0.002   # ASSUMED (A1, to be raised) [m]: tolerated position disagreement of the object pose implied by the two arms
EPS_CC_ROT = 0.01    # ASSUMED rad (A1, to be raised): tolerated rotation disagreement.  The grasp is a rigid weld, so a mismatch becomes internal force
#                      (handoff 11-03: 39.6 N on the scripted box path); the bound is NOT derived from a force limit.


def _dt(t):
    t = np.asarray(t, float)
    return float(np.median(np.diff(t))) if len(t) > 1 else 0.0


def joint_pos_row(q, t, q_lo, q_hi, margin=0.0, name="joint_pos"):
    """every sample of every side within [q_lo + margin_eff, q_hi - margin_eff], margin_eff = margin + max_k |qd_k| dt / 2 per joint (MODEL, see module doc).
    q = {side: (n,7)}; qd_k is the finite-difference velocity of q (np.gradient), so only positions are needed.
    margin of the row = smallest distance to the widened limit over side/joint/sample [rad]."""
    t = np.asarray(t, float)
    dt = _dt(t)
    lo, hi = np.asarray(q_lo, float), np.asarray(q_hi, float)
    best = None
    for s, Q in q.items():
        Q = np.asarray(Q, float)
        qd = np.gradient(Q, t, axis=0) if len(t) > 1 else np.zeros_like(Q)
        w = np.abs(qd).max(0) * dt / 2                     # per joint between-sample excursion
        mlo = Q - (lo + margin + w)
        mhi = (hi - margin - w) - Q
        m = np.minimum(mlo, mhi)
        k, j = np.unravel_index(np.argmin(m), m.shape)
        if best is None or m[k, j] < best["margin"]:
            lim = float(lo[j] + margin + w[j]) if mlo[k, j] <= mhi[k, j] else float(hi[j] - margin - w[j])
            best = dict(side=s, joint=int(j), index=int(k), t=float(t[k]), value=float(Q[k, j]), limit=lim, margin=float(m[k, j]), widen=float(w[j]))
    ok = best is not None and best["margin"] >= 0.0
    return Row(name, ok=ok, margin=None if best is None else best["margin"], unit="rad", guarantee="proof" if ok else "refuted", worst=best, n=int(len(t)),
               assumptions=dict(margin=dict(v=margin, tag="ASSUMED"), limits="MODEL (URDF position limits)", between_samples="MODEL |dq|<=max|qd|dt/2"),
               why="" if ok else f"{best['side']} J{best['joint'] + 1} = {best['value']:.4f} vs limit {best['limit']:.4f} at t = {best['t']:.3f} s (sample {best['index']})")


def joint_speed_row(qd, t, qd_max, name="joint_speed"):
    """|qd| <= qd_max at every sample, widened between samples by max|qd_{k+1} - qd_k| / 2 per joint (MODEL).  margin [rad/s]."""
    t = np.asarray(t, float)
    lim = np.asarray(qd_max, float)
    best = None
    for s, V in qd.items():
        V = np.asarray(V, float)
        w = np.abs(np.diff(V, axis=0)).max(0) / 2 if len(V) > 1 else np.zeros(V.shape[1])
        m = (lim - w) - np.abs(V)
        k, j = np.unravel_index(np.argmin(m), m.shape)
        if best is None or m[k, j] < best["margin"]:
            best = dict(side=s, joint=int(j), index=int(k), t=float(t[k]), value=float(abs(V[k, j])), limit=float(lim[j] - w[j]), margin=float(m[k, j]), widen=float(w[j]))
    ok = best is not None and best["margin"] >= 0.0
    return Row(name, ok=ok, margin=None if best is None else best["margin"], unit="rad/s", guarantee="proof" if ok else "refuted", worst=best, n=int(len(t)),
               assumptions=dict(limits="MODEL (datasheet joint speed, config.MANUAL)", between_samples="MODEL |dqd|/2 widening"),
               why="" if ok else f"{best['side']} J{best['joint'] + 1} |qd| = {best['value']:.4f} vs {best['limit']:.4f} at t = {best['t']:.3f} s")


def reach_row(fk_miss_pos, fk_miss_rot, tol_pos, tol_rot, t=None, name="reach"):
    """FK-validated pose error of every sample within tolerance.  A miss REFUSES (refuted row); the old code only recorded it.  margin = min over samples of
    min(1 - pos/tol_pos, 1 - rot/tol_rot) (dimensionless, > 0 inside); worst holds the largest misses."""
    p, r = np.asarray(fk_miss_pos, float), np.asarray(fk_miss_rot, float)
    mp, mr = 1.0 - p / tol_pos, 1.0 - r / tol_rot
    m = np.minimum(mp, mr)
    k = int(np.argmin(m))
    ok = bool(m[k] >= 0.0)
    worst = dict(index=k, t=None if t is None else float(np.asarray(t)[k]), pos_miss=float(p[k]), rot_miss=float(r[k]), pos_max=float(p.max()), rot_max=float(r.max()),
                 tol_pos=float(tol_pos), tol_rot=float(tol_rot))
    return Row(name, ok=ok, margin=float(m[k]), unit="1 - miss/tol", guarantee="proof" if ok else "refuted", worst=worst, n=int(len(p)),
               assumptions=dict(tol_pos=dict(v=tol_pos, tag="ASSUMED (carry.PATH_IK_TOL)"), tol_rot=dict(v=tol_rot, tag="ASSUMED"), fk="MODEL (MuJoCo FK of the plan model)"),
               why="" if ok else f"FK miss {p[k]:.2e} m / {r[k]:.2e} rad at sample {k} exceeds {tol_pos:.1e} / {tol_rot:.1e}")


def rot_angle(R):
    """rotation angle [rad] of a (..., 3, 3) rotation."""
    R = np.asarray(R, float)
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    s = np.linalg.norm(np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1), axis=-1) / 2.0
    return np.arctan2(s, c)


def closed_chain_row(T_obj_from_left, T_obj_from_right, eps_pos=EPS_CC_POS, eps_rot=EPS_CC_ROT, t=None, name="closed_chain"):
    """both arms agree on the object pose: ||p_L - p_R|| <= eps_pos and angle(R_L^T R_R) <= eps_rot at every sample.  Inputs are (n,4,4) object poses, each implied
    by one arm (T_tcp @ inv(T_grasp_side)).  margin = min over samples of min(1 - dp/eps_pos, 1 - dr/eps_rot) (dimensionless)."""
    A, B = np.asarray(T_obj_from_left, float), np.asarray(T_obj_from_right, float)
    dp = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1)
    dr = rot_angle(np.einsum("nji,njk->nik", A[:, :3, :3], B[:, :3, :3]))
    m = np.minimum(1.0 - dp / eps_pos, 1.0 - dr / eps_rot)
    k = int(np.argmin(m))
    ok = bool(m[k] >= 0.0)
    worst = dict(index=k, t=None if t is None else float(np.asarray(t)[k]), dpos=float(dp[k]), drot=float(dr[k]), dpos_max=float(dp.max()), drot_max=float(dr.max()),
                 eps_pos=eps_pos, eps_rot=eps_rot)
    return Row(name, ok=ok, margin=float(m[k]), unit="1 - err/eps", guarantee="proof" if ok else "refuted", worst=worst, n=int(len(dp)),
               assumptions=dict(eps_pos=dict(v=eps_pos, tag="ASSUMED (A1)"), eps_rot=dict(v=eps_rot, tag="ASSUMED (A1)"),
                                note="rigid weld: a mismatch becomes internal force (11-03: 39.6 N on the scripted box path)"),
               why="" if ok else f"arms disagree on the object pose by {dp[k]:.2e} m / {dr[k]:.2e} rad at sample {k}")
