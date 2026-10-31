"""gatekeeper/monitor.py -- (1) set-membership filtering of the person position discs, (2) the KEEP-rule falsification monitor.  SIMULATION ONLY.

DISC INTERSECTION  disc_enclose(c1, r1, c2, r2) -> smallest disc enclosing  D(c1,r1) ∩ D(c2,r2)  (closed form), or None if the discs are disjoint:
    d = |c2 - c1|;  d > r1 + r2            -> None (disjoint)
    d + r_small <= r_big                   -> the smaller disc (containment; concentric included)
    otherwise a = (d^2 + r1^2 - r2^2)/(2d) is the distance from c1 to the radical chord, h = sqrt(r1^2 - a^2) its half length:
      a <= 0       -> D(c1, r1)   (the lens contains the major arc of circle 1, whose smallest enclosing disc is the circle itself)
      d - a <= 0   -> D(c2, r2)   (symmetric)
      else         -> D(c1 + a u, h), u = (c2-c1)/d   (the chord is a diameter; both circular segments have sagitta r - a' <= h for a' >= 0, so the lens fits)
  Sound: every point of the intersection lies in the returned disc.  Sequential use (A ∩ B ∩ C ~ enclose(enclose(A,B), C)) stays a superset.

FILTER (GKConfig.belief_filter="intersect"), per tracked person (key = track name, else list index): the previous step's position set S = D(c, r) at t_prev is
predicted over dt = t - t_prev:  speed  D(c, r + v_p dt);  accel (person_model="accel")  D(c + v_hat_prev dt, r + e_v_prev dt + a_p dt^2/2)  (v_hat/e_v of the previous step);
the person is in both, so P = enclose(speed, accel);  new set = enclose(P, D(c_obs, e_p_obs)).  Disjoint -> n_obs_inconsistent += 1 and reset to the observation.
Only the track's OWN position error (people_e_p, or 0 for the truth source) is filtered; the global GKConfig.e_p stays an additive assumption on top.
Truth source (own radius 0): the new set is the observed point itself -> no-op (unless the point leaves the predicted disc = a model falsification, counted).
CAVEAT: if e_p is a 95 % conformal bound, intersecting many such sets lowers the joint coverage (union bound, 1 - n*alpha); `n_belief_miss` (audit) measures it.

KEEP-RULE MONITOR (GKConfig.keep_rule="monitor"; theorem T1): no candidate passed -> keep tau_j WITHOUT re-certifying it under b_k; regime 2 only if
  (a) people (only while the kept trajectory still has a MOVING part, i.e. rem.i_rest != 0; at rest S2 is vacuous):  for each current person i with observation disc
      O_i = D(c_now, R_i),  R_i = e_p_i(+global e_p) + v_p_max*latency_s:
        MATCHED  iff some commit-time person m has ALL its predicted discs (elapsed dt = t_now - t_j; speed D(c_m, e_m + v_p (dt + lat_m)) and, if the commit belief used
                 person_model="accel", D(c_m + v_hat_m (dt + latency_s), e_m + e_v_m (dt + lat_m) + a_p (dt + lat_m)^2/2)) intersecting O_i (|c_now - c| <= r + R_i + tol).
                 Matching is by "any commit-time person whose predicted set contains/meets the observation" (nearest-in-set; track ids are not carried in the Belief).
        UNMATCHED (new track, or a person that left its predicted set): falsified iff
                 min_k dist(c_now, Rect(p_k)) < PERSON_CLEAR_R + R_i + d_person + v_p_max * T_move,   over the moving samples k of the kept profile, T_move = i_rest*h
                 (a person outside this distance cannot reach the remaining motion even at v_p_max starting from now, so tau_j's future is unaffected).
  (b) pose:  |ctrl_pose_now - pose_pred_j(t_now)|_xy > e_xy_j + e_xy_now + falsify_tol   or heading likewise with e_th (the commit-time pose box propagated along tau_j; the
      current box is added because the measured pose carries its own error).
Returns (ok, reason) with reason in {"person_unmatched_near", "pose_xy", "pose_th"}.
GKConfig.runtime_rows == "cup": part (a) is OFF (people are the planner's job); only the pose part (b) remains.
"""
from __future__ import annotations

import math

import numpy as np

from ..teleop import geom as GM
from .checks import _pe, _ev, _lat, _vhat, _rect_dist, world_poses

_EPS = 1e-9


def disc_enclose(c1, r1, c2, r2):
    c1 = np.asarray(c1, float); c2 = np.asarray(c2, float)
    d = float(np.hypot(*(c2 - c1)))
    if d > r1 + r2 + _EPS:
        return None
    if d + r2 <= r1 + _EPS:
        return c2.copy(), float(r2)
    if d + r1 <= r2 + _EPS:
        return c1.copy(), float(r1)
    a = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    if a <= 0.0:
        return c1.copy(), float(r1)
    if d - a <= 0.0:
        return c2.copy(), float(r2)
    hh = math.sqrt(max(0.0, r1 * r1 - a * a))
    return c1 + (c2 - c1) * (a / d), float(hh)


class PersonSetFilter:
    def __init__(self, gk):
        self.gk = gk
        self.state = {}
        self.n_inconsistent = 0
        self.n_obs = 0
        self.sum_raw_r = 0.0
        self.sum_filt_r = 0.0

    def update(self, keys, C, R, V, EV, t):
        """keys/C/R/V/EV: per person (key, centre, own position radius, velocity estimate or None, velocity error) -> (centres, radii)."""
        gk = self.gk
        new, oc, orad = {}, [], []
        for key, c, r, v, ev in zip(keys, C, R, V, EV):
            c = np.asarray(c, float); r = float(r)
            v = np.zeros(2) if v is None else np.asarray(v, float)
            cf, rf = c, r
            st = self.state.get(key)
            if st is not None and t - st["t"] > 1e-9:
                dt = t - st["t"]
                pred = (st["c"], st["r"] + gk.v_p_max * dt)
                if gk.person_model in ("accel", "pred"):
                    pred = disc_enclose(*pred, st["c"] + st["v"] * dt, st["r"] + st["ev"] * dt + 0.5 * gk.a_p * dt * dt)
                m = None if pred is None else disc_enclose(pred[0], pred[1], c, r)
                if m is None:
                    self.n_inconsistent += 1                                  # observation outside the predicted set: model falsified -> reset to the observation
                else:
                    cf, rf = m
            self.n_obs += 1; self.sum_raw_r += r; self.sum_filt_r += rf
            new[key] = dict(t=t, c=cf, r=rf, v=v, ev=float(ev))
            oc.append(cf); orad.append(rf)
        self.state = new
        return oc, orad

    def stats(self):
        n = max(self.n_obs, 1)
        return dict(n_obs_inconsistent=self.n_inconsistent, mean_raw_radius=self.sum_raw_r / n, mean_filtered_radius=self.sum_filt_r / n)


def observation_discs(bel, gk):
    """(centres, radii) of the current observation discs: position error (global + per-person) + v_p_max * latency_s."""
    return ([np.asarray(c, float) for c in bel.people], [_pe(bel, i) + bel.v_p_max * bel.latency_s for i in range(len(bel.people))])


def predicted_discs(cb, m, dt):
    """predicted position-set discs (centre, radius) of commit-time person m after elapsed dt (observation age included)."""
    c = np.asarray(cb.people[m], float)
    pe, lat = _pe(cb, m), _lat(cb, m)
    te = dt + lat
    out = [(c, pe + cb.v_p_max * te)]
    if cb.person_model in ("accel", "pred"):
        out.append((c + _vhat(cb, m) * (dt + cb.latency_s), pe + _ev(cb, m) * te + 0.5 * cb.a_p * te * te))
    return out


def pose_pred(traj_j, cb, dt):
    """world pose of the committed profile at elapsed dt (linear in the sample grid; holds the last pose)."""
    Pw = world_poses(traj_j.poses(), cb.pose_wall)
    x = max(0.0, dt) / traj_j.h
    if x >= traj_j.N:
        return Pw[-1]
    k = int(x); f = x - k
    p = Pw[k] * (1 - f) + Pw[k + 1] * f
    return p


def falsification_monitor(cb, traj_j, t_j, rem, bel_now, obs_now, pose_now, t_now, gk):
    """-> (ok, reason).  See the module docstring."""
    dt = max(0.0, t_now - t_j)
    # (b) pose
    pp = pose_pred(traj_j, cb, dt)
    dxy = float(np.hypot(pose_now[0] - pp[0], pose_now[1] - pp[1]))
    dth = abs((float(pose_now[2]) - float(pp[2]) + math.pi) % (2 * math.pi) - math.pi)
    if dxy > cb.e_xy + bel_now.e_xy + gk.falsify_tol:
        return False, "pose_xy"
    if dth > cb.e_th + bel_now.e_th + gk.falsify_tol:
        return False, "pose_th"
    # (a) people.  runtime_rows == "cup": the people part is OFF (people are the planner's job; only the pose part of the monitor stays)
    if getattr(gk, "runtime_rows", "all") == "cup":
        return True, "ok"
    i_rest = rem.i_rest
    n_mov = rem.N + 1 if i_rest is None else i_rest
    if n_mov <= 0 or not len(obs_now[0]):
        return True, "ok"
    pred = [predicted_discs(cb, m, dt) for m in range(len(cb.people))]
    Pp = None
    for c, R in zip(*obs_now):
        matched = any(all(float(np.hypot(*(c - pc))) <= pr + R + _EPS for pc, pr in discs) for discs in pred)
        if matched:
            continue
        if Pp is None:
            Pp = world_poses(rem.poses()[:n_mov], bel_now.pose_people)
        dmin = float(_rect_dist(c, Pp, bel_now.fp).min())
        if dmin < GM.PERSON_CLEAR_R + R + bel_now.d_person + bel_now.v_p_max * n_mov * rem.h:
            return False, "person_unmatched_near"
    return True, "ok"
