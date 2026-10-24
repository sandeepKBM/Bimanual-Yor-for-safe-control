"""tamp/plan_monitor.py -- plan-prediction monitor: detect that the PEOPLE PREDICTION a plan was certified under is falsified.  SIMULATION ONLY, opt-in.

The segment planner (people_pred="time") certifies the immediate route against people at c_i + v_hat_i t.  If a person then moves differently (a standing walker starts
walking) the plan's certificate no longer describes the world.  `PlanMonitor` holds the plan's prediction [(c_i0, v_i0)] observed at t_obs and, at every gatekeeper step
while a route is active, compares it with the live belief:

    person i violates if   |c_i(t) - (c_i0 + v_i0 (t - t_obs))| > eps_p   or   |v_i(t) - v_i0| > eps_v          (kind "pos" / "vel")
    an observed person with no predicted match is a "new" violation
    a violation counts only if the observed OR predicted position is within r_rel of the REMAINING planned route polyline [s_cur, s_stop]
    flag after n_confirm consecutive violating checks; one flag per prediction (the monitor is off until set_prediction() is called again).

Pure logic (no sim, no checkers): the SegmentGatekeeper owns an instance and feeds it the belief it builds anyway.  The monitor never commits or changes anything itself.
"""
from __future__ import annotations

import math

import numpy as np

from ..gatekeeper.checks import _ev, _pe
from ..teleop import geom as GM


def dist_to_polyline(p, path, s0: float, s1: float) -> float:
    """distance from xy point p to the sub-polyline of `path` (segments.Path) between arclengths s0 <= s1."""
    p = np.asarray(p, float)
    if path.total < 1e-9:
        return float(np.hypot(*(p - path.P[0])))
    s0 = min(max(float(s0), 0.0), path.total)
    s1 = min(max(float(s1), s0), path.total)
    pts = [path.point(s0)] + [path.P[i] for i in range(len(path.P)) if s0 < path.cum[i] < s1] + [path.point(s1)]
    best = math.inf
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L2 = float(d @ d)
        f = 0.0 if L2 < 1e-12 else min(1.0, max(0.0, float((p - a) @ d) / L2))
        best = min(best, float(np.hypot(*(p - (a + f * d)))))
    if len(pts) == 1:
        best = float(np.hypot(*(p - pts[0])))
    return best


class PlanMonitor:
    def __init__(self, eps_p: float = 0.30, eps_v: float = 0.20, r_rel: float = 2.0, n_confirm: int = 2):
        self.eps_p, self.eps_v, self.r_rel, self.n_confirm = float(eps_p), float(eps_v), float(r_rel), int(n_confirm)
        self.pred = None              # [(c0 (2,), v0 (2,)), ...] observed when the plan was made
        self.t_obs = 0.0
        self.n_viol = 0               # consecutive violating checks
        self.n_checks = 0
        self.flag = None              # the flag record once raised (monitor off until set_prediction)

    @property
    def active(self) -> bool:
        return self.pred is not None and self.flag is None

    def set_prediction(self, people, t_obs: float):
        """people: [(c, v_hat), ...] the plan was certified for (positions at t_obs, constant velocity).  Re-arms the monitor."""
        self.pred = [(np.asarray(c, float)[:2].copy(), np.asarray(v, float)[:2].copy()) for c, v in people]
        self.t_obs = float(t_obs)
        self.n_viol, self.n_checks, self.flag = 0, 0, None

    def clear(self):
        self.pred, self.flag, self.n_viol = None, None, 0

    def reset_debounce(self):
        self.n_viol = 0

    def expected(self, i: int, t: float):
        c0, v0 = self.pred[i]
        return c0 + v0 * (t - self.t_obs)

    def _match(self, obs_c, t):
        """obs index -> predicted index or None (index order when the counts agree, else greedy nearest to the predicted positions)."""
        n_o, n_p = len(obs_c), len(self.pred)
        if n_o == n_p:
            return list(range(n_o))
        used, out = set(), []
        for c in obs_c:
            best, bj = math.inf, None
            for j in range(n_p):
                if j in used:
                    continue
                d = float(np.hypot(*(np.asarray(c, float)[:2] - self.expected(j, t))))
                if d < best:
                    best, bj = d, j
            if bj is not None:
                used.add(bj)
            out.append(bj)
        return out

    def check(self, people, people_vel, t: float, path, s_cur: float, s_stop: float):
        """one check at time t.  people: observed xy per person; people_vel: observed velocities (or () = unknown -> velocity test skipped).
        -> the flag record (dict) once n_confirm consecutive checks violate, else None."""
        if not self.active:
            return None
        self.n_checks += 1
        obs = [np.asarray(c, float)[:2] for c in people]
        have_v = len(people_vel) == len(obs)
        match = self._match(obs, t)
        first = None
        for i, c in enumerate(obs):
            j = match[i]
            if j is None:
                kind, pe, ve, pred_xy = "new", math.inf, math.inf, None
            else:
                pred_xy = self.expected(j, t)
                pe = float(np.hypot(*(c - pred_xy)))
                ve = float(np.hypot(*(np.asarray(people_vel[i], float)[:2] - self.pred[j][1]))) if have_v else 0.0
                if pe <= self.eps_p and ve <= self.eps_v:
                    continue
                kind = "vel" if ve > self.eps_v else "pos"
            near = dist_to_polyline(c, path, s_cur, s_stop) <= self.r_rel
            if not near and pred_xy is not None:
                near = dist_to_polyline(pred_xy, path, s_cur, s_stop) <= self.r_rel
            if near and first is None:
                first = dict(t=float(t), person=int(i), pred_index=j, pos_err=pe, vel_err=ve, kind=kind, t_obs=self.t_obs)
        if first is None:
            self.n_viol = 0
            return None
        self.n_viol += 1
        if self.n_viol < self.n_confirm:
            return None
        first["n_confirm"] = self.n_viol
        self.flag = first
        return first


def conflict_time(samples, people, vels, r0s, evs, t_f: float, d_person: float, fp=None):
    """earliest sample time t > t_f at which, for some person i, the tracked tube disc(c_i + v_i (t - t_f), r0_i + e_v_i (t - t_f)) comes within d_person of the robot footprint
    at the predicted pose of that sample.  samples: [(t, pose (x, y, yaw), v, phase), ...] ascending in t (the robot's predicted OLD-route motion).  -> time or inf."""
    fp = GM.FP if fp is None else fp
    for (t, pose, _v, _ph) in samples:
        tau = t - t_f
        if tau <= 0.0:
            tau = 0.0
        for c, v, r0, ev in zip(people, vels, r0s, evs):
            ctr = np.asarray(c, float)[:2] + np.asarray(v, float)[:2] * tau
            if GM.rect_dist_point(ctr, pose, fp)[0] - (r0 + ev * tau) <= d_person:
                return float(t)
    return math.inf


def state_at(samples, t: float):
    """(pose, v, phase) of the sample nearest to time t (holds the last sample beyond the end)."""
    best = min(samples, key=lambda x: abs(x[0] - t))
    return best[1], best[2], best[3]


def people_vels(bel):
    """tracked velocities of bel.people as 2-vectors (zeros when the belief carries none)."""
    n = len(bel.people)
    return [np.asarray(v, float)[:2] for v in bel.people_vel] if len(bel.people_vel) == n else [np.zeros(2) for _ in range(n)]


def rest_tube_conflict_time(bel, pose, t_from: float, horizon: float, dt: float = 0.25, t_f=None, clear: float = 0.0) -> float:
    """earliest time in [t_from, t_from + horizon] (grid dt) at which a footprint RESTING at `pose` comes within d_person + clear of any person's tracked tube
    disc(c_i + v_i (t - t_f), r0_i + e_v_i (t - t_f)), r0_i = PERSON_CLEAR_R + e_p_i  (conflict_time on a stationary sample list).  t_f = the time the belief's people are observed at
    (default t_from; the C2-ready tube term passes 0 because candidate times are measured from the belief time).  -> time or inf.  Shared by tamp/hop.py out_of_tubes and the
    segment ranking's tube term (tamp/segments.py)."""
    n = len(bel.people)
    t_f = t_from if t_f is None else t_f
    S = [(t_from + k * dt, tuple(float(x) for x in pose), np.zeros(3), "rest") for k in range(int(round(horizon / dt)) + 1)]
    return conflict_time(S, bel.people, people_vels(bel), [GM.PERSON_CLEAR_R + _pe(bel, i) for i in range(n)], [_ev(bel, i) for i in range(n)],
                         t_f, bel.d_person + clear, bel.fp)
