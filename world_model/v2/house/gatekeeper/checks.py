"""gatekeeper/checks.py -- Checker 1 (belief-wide admission of nominal+backup) and Checker 2 (backup ends in the task-ready set T(b)).  SIMULATION ONLY.

Notation: samples k = 0..N at t_k = k h, planned pose p_k = (x_k, y_k, th_k), body twist v_k, boundary points q_j of the footprint rectangle (robot frame),
|q_j| their distance from the base origin, i_rest = first sample with v = 0.  Belief (belief.Belief): e_xy, e_th, e_p, v_p_max, latency, mu_lo, d_wall, d_person.

CHECKER 1 (all must hold; each is the WORST CASE over the belief set, inflation is exact, no sampling of the belief):
  wall   : EDT(p_k + R(th_k) q_j) - e_xy - |q_j| e_th - fp_step/2 - sl_k  >=  d_wall(p_k)             for all k, j
           [|R(th+d) q - R(th) q| <= |q||d| and |dxy| <= e_xy => EDT (1-Lipschitz) drops by at most e_xy + |q| e_th; fp_step/2: boundary sampling;
            sl_k = 0.5 h max_{k-1,k,k+1} (|v_xy| + |wz| max|q|): rigorous inter-sample sweep]
  person : for every moving sample (k < i_rest) and person i:
           dist(c_i, Rect(p_k)) - (r0 + v_p (t_k + latency)) - d_person - 0.5 h (vpt_k + v_p)  >=  0,    r0 = PERSON_CLEAR_R + e_p
           [reachable set at t = disc(c_i, r0 + v_p t): the exact worst case of a person anywhere within v_p of its observed position]
           OPT-IN person_model="accel": additionally (t_e = t_k + latency, v_hat = velocity estimate)
              dist(c_i + v_hat t_e, Rect(p_k)) - (r0 + e_v t_e + a_p t_e^2/2) - d_person - 0.5 h (vfar_k + |v_hat| + e_v + a_p (t_e + h)) >= 0
              the person lies in BOTH discs, so the robot is safe if it is clear of EITHER: per-sample margin = max(margin_speed, margin_accel)
  cup    : |a_cup(t)| <= a_cup_max = cup_margin * min(kappa mu_lo, tan theta_spill) g     (a_cup incl. alpha x r + w x (w x r); segment start AND end)
           OPT-IN gk.cup_chord: each segment's values get + (h^2/8)(2|alpha||a_b| + 2 alpha^2 |r|) (a_cup is quadratic in t on a segment -> sound between samples)
  hw     : |a_i| <= A_HW_i and |v_i| <= V_STICK_i

CHECKER 2 (backup ends in T(b) = {v = 0, cup margin >= 0, static clearance >= d_wall at rest for all pose-belief members, passive person safety}):
  rest   : i_rest exists (v reaches 0 within t_back_max)
  static : min_j EDT(p_N + R(th_N) q_j) - e_xy - |q_j| e_th - fp_step/2  >=  d_wall(p_N)
  passive: t_touch - t_rest >= 0  with  t_touch = (dist(c_i, Rect(p_rest)) - r0)/v_p - latency   (robot is at rest before ANY person disc can reach it);
           accel model: t_touch = max(t_speed, t_accel), t_accel = positive root of a_p t^2/2 + (e_v + |v_hat|) t = dist - r0 (see t_touch_accel)
           a trajectory that is at rest from t = 0 (i_rest = 0) is passive-safe by definition.
  cup    : v_N = 0 => a_cup = 0 (and the sampled |a_cup| <= a_cup_max was verified in Checker 1)

Return: (ok, min_margin, reason, margins); margins are in their own SI unit (geometry m, passive s, cup/hw m/s^2); reason = failing (or tightest) item.
Monotone: enlarging e_xy, e_th, e_p, v_p_max, latency, d_wall, d_person or lowering mu_lo can only turn ok=True into ok=False for a FIXED trajectory.
"""
from __future__ import annotations

import math

import numpy as np

from ..teleop import geom as GM
from .traj import Traj, acup_series, acup_chord

_BCACHE = {}


def _fp_pts(fp, step):
    key = (tuple(fp["x"]), tuple(fp["y"]), step)
    if key not in _BCACHE:
        B = GM.boundary_pts(fp, step=step)
        _BCACHE[key] = (B, np.hypot(B[:, 0], B[:, 1]))
    return _BCACHE[key]


def world_poses(P_rel, pose0):
    c, s = math.cos(pose0[2]), math.sin(pose0[2])
    return np.stack([pose0[0] + c * P_rel[:, 0] - s * P_rel[:, 1], pose0[1] + s * P_rel[:, 0] + c * P_rel[:, 1], pose0[2] + P_rel[:, 2]], 1)


def _wall_dists(grid, Pw, B, nq, e_xy, e_th):
    """(N+1,) worst-case boundary clearance: min_j EDT - e_xy - |q_j| e_th."""
    c, s = np.cos(Pw[:, 2])[:, None], np.sin(Pw[:, 2])[:, None]
    X = Pw[:, 0:1] + c * B[None, :, 0] - s * B[None, :, 1]
    Y = Pw[:, 1:2] + s * B[None, :, 0] + c * B[None, :, 1]
    return (grid.lookup(X, Y) - e_xy - nq[None, :] * e_th).min(1)


def _d_wall_k(bel, Pw):
    d = np.full(len(Pw), bel.d_wall)
    for (dx, dy) in bel.docks:
        d = np.where(np.hypot(Pw[:, 0] - dx, Pw[:, 1] - dy) < bel.dock_radius, bel.d_wall_dock, d)
    return d


def _rect_dist(c_xy, Pw, fp):
    """signed distance (>= 0 outside, < 0 penetration) from point c_xy to the footprint rectangle at each pose, (N+1,)."""
    c, s = np.cos(Pw[:, 2]), np.sin(Pw[:, 2])
    dx, dy = c_xy[0] - Pw[:, 0], c_xy[1] - Pw[:, 1]
    qx, qy = c * dx + s * dy, -s * dx + c * dy
    lo = np.array([fp["x"][0], fp["y"][0]]); hi = np.array([fp["x"][1], fp["y"][1]])
    ox = np.maximum(np.maximum(lo[0] - qx, qx - hi[0]), 0.0)
    oy = np.maximum(np.maximum(lo[1] - qy, qy - hi[1]), 0.0)
    out = np.hypot(ox, oy)
    inside = (ox == 0) & (oy == 0)
    pen = np.minimum(np.minimum(qx - lo[0], hi[0] - qx), np.minimum(qy - lo[1], hi[1] - qy))
    return np.where(inside, -pen, out)


def _slack(traj, qmax):
    V = traj.V
    vpt = np.hypot(V[:, 0], V[:, 1]) + np.abs(V[:, 2]) * qmax
    m = vpt.copy()
    m[1:] = np.maximum(m[1:], vpt[:-1])
    m[:-1] = np.maximum(m[:-1], vpt[1:])
    return m, vpt


def check1(traj: Traj, bel, gk, tol_geo=None, tol_cup=1e-6):
    """belief-wide admission of the whole nominal+backup trajectory.  -> (ok, min_margin, reason, margins)."""
    tol_geo = gk.tol if tol_geo is None else tol_geo
    h = traj.h
    N = traj.N
    V = traj.V
    mg = {}
    # hardware slew / speed cap
    if N >= 1:
        A = np.abs(np.diff(V, axis=0)) / h
        mg["hw_slew"] = float(((bel.a_hw[None, :] - A) ).min())
    mg["speed_cap"] = float((bel.v_cap[None, :] - np.abs(V)).min())
    # cup (rotation terms included)
    ac = acup_series(V, h, bel.r_cup)
    if getattr(gk, "cup_chord", False) and N >= 1:                      # OPT-IN: sound between samples (draft Lemma 1, GAP-1): per-segment chord term
        cc = acup_chord(V, h, bel.r_cup)
        ac = np.concatenate([ac[:N] + cc, ac[N:2 * N] + cc, ac[2 * N:]])
    mg["cup"] = float(bel.a_cup_max - ac.max())
    # static obstacles (pose-box worst case)
    B, nq = _fp_pts(bel.fp, gk.fp_step)
    qmax = float(nq.max())
    sl_m, vpt = _slack(traj, qmax)
    sw = 0.5 * h * sl_m if gk.sweep_slack else np.zeros(N + 1)
    Pr = traj.poses()
    Pw = world_poses(Pr, bel.pose_wall)
    if bel.grid is not None:
        wd = _wall_dists(bel.grid, Pw, B, nq, bel.e_xy, bel.e_th) - 0.5 * gk.fp_step - sw - _d_wall_k(bel, Pw)
        mg["wall"] = float(wd.min())
    # people (reachable discs) while the robot is moving
    i_rest = traj.i_rest
    n_mov = N + 1 if i_rest is None else i_rest
    if len(bel.people) and n_mov > 0:
        Pp = world_poses(Pr[:n_mov], bel.pose_people)
        tk = np.arange(n_mov) * h
        vfar, _ = _slack(traj, _rect_far(bel.fp))
        best = np.inf
        accel = bel.person_model == "accel"
        for ip, c in enumerate(bel.people):
            c = np.asarray(c, float)
            r0 = GM.PERSON_CLEAR_R + _pe(bel, ip)
            lat = _lat(bel, ip)
            d = _rect_dist(c, Pp, bel.fp)
            sp = 0.5 * h * (vfar[:n_mov] + bel.v_p_max) if gk.sweep_slack else 0.0
            m = d - (r0 + bel.v_p_max * (tk + lat)) - bel.d_person - sp
            if accel:                                                     # person is in D_speed AND D_accel -> robot is safe if clear of EITHER -> max margin
                vh = _vhat(bel, ip)
                te = tk + lat                                             # radius growth time (observation age incl. the track's staleness)
                tc = tk + bel.latency_s                                   # centre advance time (a track centre is already predicted to now -> no staleness)
                ev = _ev(bel, ip)
                ctr = (c[0] + vh[0] * tc, c[1] + vh[1] * tc)
                d_a = _rect_dist(ctr, Pp, bel.fp)
                rad = r0 + ev * te + 0.5 * bel.a_p * te * te
                rate = math.hypot(vh[0], vh[1]) + ev + bel.a_p * (te + h)       # max growth of the disc surface over the next sample interval
                sp_a = 0.5 * h * (vfar[:n_mov] + rate) if gk.sweep_slack else 0.0
                m = np.maximum(m, d_a - rad - bel.d_person - sp_a)
            best = min(best, float(m.min()))
        mg["person"] = best
    return _verdict(mg, {"cup": tol_cup, "hw_slew": tol_cup, "speed_cap": tol_cup}, tol_geo)


def _pe(bel, i):
    """effective position error of person i: global e_p (+ the track's own e_p when the belief carries per-person values)."""
    return bel.e_p + bel.people_e_p[i] if i < len(bel.people_e_p) else bel.e_p


def _ev(bel, i):
    return bel.e_v + bel.people_e_v[i] if i < len(bel.people_e_v) else bel.e_v


def _lat(bel, i):
    """effective observation age of person i: latency_s (+ time since the track's last update)."""
    return bel.latency_s + bel.people_stale[i] if i < len(bel.people_stale) else bel.latency_s


def _vhat(bel, i):
    pv = bel.people_vel
    return np.asarray(pv[i], float) if i < len(pv) else np.zeros(2)


def person_radii(bel, i, t):
    """(centre, radius) of the speed disc and of the accel disc of person i at time t (after the observation, latency included by the caller):
    speed: (c, r0 + v_p t);  accel: (c + v_hat t, r0 + e_v t + a_p t^2/2).  The person is in the INTERSECTION of both."""
    c = np.asarray(bel.people[i], float)
    r0 = GM.PERSON_CLEAR_R + _pe(bel, i)
    vh = _vhat(bel, i)
    return (c, r0 + bel.v_p_max * t), (c + vh * t, r0 + _ev(bel, i) * t + 0.5 * bel.a_p * t * t)


def t_touch_accel(x, b, a_p, v_p_max):
    """time [s] for the accel-model disc to close a gap x = dist(c, Rect) - r0 given the centre may approach at <= b = e_v + |v_hat| (1-Lipschitz bound of the
    rect distance) and the radius gains a_p t^2/2:  smallest t >= 0 with a_p t^2/2 + b t >= x  ->  t = (-b + sqrt(b^2 + 2 a_p x))/a_p.  x < 0 (already touching)
    continues linearly (x / max(b, v_p_max)) so the margin stays monotone in r0."""
    if x < 0:
        return x / max(b, v_p_max, 1e-9)
    if a_p <= 1e-12:
        return x / b if b > 1e-12 else math.inf
    return (-b + math.sqrt(b * b + 2.0 * a_p * x)) / a_p


def _rect_far(fp):
    return float(max(math.hypot(x, y) for x in fp["x"] for y in fp["y"]))


def check2(traj: Traj, bel, gk, tol_geo=None, tol_cup=1e-6):
    """backup ends in the task-ready set T(b).  -> (ok, margin, reason, margins)."""
    tol_geo = gk.tol if tol_geo is None else tol_geo
    mg = {}
    i_rest = traj.i_rest
    if i_rest is None:
        return False, -1.0, "no_rest", {"no_rest": -1.0}
    mg["rest_speed"] = float(1e-6 - np.abs(traj.V[-1]).max())
    Pr = traj.poses()
    P_end = Pr[i_rest:i_rest + 1]
    if bel.grid is not None:
        B, nq = _fp_pts(bel.fp, gk.fp_step)
        Pw = world_poses(P_end, bel.pose_wall)
        mg["wall_at_rest"] = float((_wall_dists(bel.grid, Pw, B, nq, bel.e_xy, bel.e_th) - 0.5 * gk.fp_step - _d_wall_k(bel, Pw)).min())
    if len(bel.people) and i_rest > 0:
        Pp = world_poses(P_end, bel.pose_people)
        t_rest = i_rest * traj.h
        best = np.inf
        for ip, c in enumerate(bel.people):
            r0 = GM.PERSON_CLEAR_R + _pe(bel, ip)
            lat = _lat(bel, ip)
            d = float(_rect_dist(np.asarray(c, float), Pp, bel.fp)[0])
            if bel.v_p_max <= 0:
                m = np.inf if d >= r0 else -np.inf
            else:
                m = (d - r0) / bel.v_p_max - lat - t_rest
                if bel.person_model == "accel":                           # touching needs BOTH discs -> t_touch = max(t_speed, t_accel) -> margin = max
                    vh = _vhat(bel, ip)
                    ta = t_touch_accel(d - r0, _ev(bel, ip) + math.hypot(vh[0], vh[1]), bel.a_p, bel.v_p_max)
                    m = max(m, ta - lat - t_rest)
            best = min(best, m)
        mg["passive_person"] = float(min(best, 1e3))
    return _verdict(mg, {}, tol_geo)


def _verdict(mg, tols, tol_geo):
    worst, wk = np.inf, "none"
    ok = True
    for k, m in mg.items():
        tol = tols.get(k, tol_geo)
        if m < -tol:
            ok = False
        if m < worst:
            worst, wk = m, k
    return ok, float(worst), wk, mg
