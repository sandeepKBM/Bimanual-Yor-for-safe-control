"""gatekeeper/traj.py -- candidate trajectories: nominal (u_op held for T_S, cup-safe ramp) + BACKUP (cup-safe stop).  SIMULATION ONLY.

A trajectory is a body-twist profile v_k = (vx, vy, wz) on a uniform grid h, piecewise-LINEAR in time (constant body acceleration a_k = (v_{k+1}-v_k)/h on
each segment).  Poses are integrated in the frame of the start pose; the checkers anchor them (ctrl pose for walls, true pose for people).

Cup acceleration of a point at base-frame r_c (teleop/base.cup_accel_body, written out):
    a_cup = ( a_x - alpha r_y - w (v_y + w r_x) ,  a_y + alpha r_x + w (v_x - w r_y) ),      w = wz, alpha = a_z
The planner picks, at every step, the direction  a_des = (target - v)/h  scaled into the hardware slew box (common factor k, keeps the ray), then the
largest s in [0,1] with  |a_cup(v, s a_des)| <= L  (L = plan_frac * a_cup_max)  -- a quadratic in s since a_cup is affine in a.  The same scale is
re-verified at the segment END state (centripetal term changes) and shrunk if needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS_V = 1e-9


def acup(v, a, r):
    """scalar-python cup inertial acceleration (ax, ay) in the base frame (== base.cup_accel_body)."""
    w = v[2]
    return (a[0] - a[2] * r[1] - w * (v[1] + w * r[0]), a[1] + a[2] * r[0] + w * (v[0] - w * r[1]))


def acup_series(V, h, r):
    """|cup acceleration| at the START and END of every segment of the profile V (N+1,3), plus the last sample at zero accel -> (2N+1,) norms."""
    V = np.asarray(V, float)
    if len(V) < 2:
        return np.array([math.hypot(*acup(V[0], (0, 0, 0), r))])
    A = np.diff(V, axis=0) / h
    out = []
    for Vk, Ak in ((V[:-1], A), (V[1:], A)):
        w = Vk[:, 2]
        ax = Ak[:, 0] - Ak[:, 2] * r[1] - w * (Vk[:, 1] + w * r[0])
        ay = Ak[:, 1] + Ak[:, 2] * r[0] + w * (Vk[:, 0] - w * r[1])
        out.append(np.hypot(ax, ay))
    w = V[-1, 2]
    last = math.hypot(-w * (V[-1, 1] + w * r[0]), w * (V[-1, 0] - w * r[1]))
    return np.concatenate([out[0], out[1], [last]])


def acup_chord(V, h, r):
    """(N,) per-segment bound (h^2/8)(2|alpha||a_b| + 2 alpha^2 |r|) on the excess of max_t |a_cup(t)| over the segment-END values (a_cup is quadratic in t on a segment)."""
    V = np.asarray(V, float)
    if len(V) < 2:
        return np.zeros(0)
    A = np.diff(V, axis=0) / h
    al = np.abs(A[:, 2])
    ab = np.hypot(A[:, 0], A[:, 1])
    return (h * h / 8.0) * (2.0 * al * ab + 2.0 * al * al * math.hypot(r[0], r[1]))


@dataclass
class Traj:
    V: np.ndarray                  # (N+1, 3) body twist at samples t0 + k h
    h: float
    t0: float = 0.0                # sim time of sample 0
    t_sw: float = 0.0              # s after t0 where the BACKUP starts (<= 0: already in the backup)
    T_S: float = 0.0               # switch time this candidate was planned with

    @property
    def N(self):
        return len(self.V) - 1

    @property
    def i_rest(self):
        """first sample index from which v == 0 for good (0 if at rest throughout; None if the profile does not end at rest)."""
        nz = np.where(np.abs(self.V).max(1) > EPS_V)[0]
        if len(nz) == 0:
            return 0
        return None if nz[-1] == len(self.V) - 1 else int(nz[-1] + 1)

    @property
    def t_rest(self):
        i = self.i_rest
        return None if i is None else i * self.h

    def poses(self):
        """poses (N+1,3) in the frame of the start pose (midpoint heading, like base.advance)."""
        P = np.zeros((len(self.V), 3))
        h = self.h
        for k in range(len(self.V) - 1):
            vm = 0.5 * (self.V[k] + self.V[k + 1])
            ym = P[k, 2] + 0.5 * vm[2] * h
            c, s = math.cos(ym), math.sin(ym)
            P[k + 1] = (P[k, 0] + (c * vm[0] - s * vm[1]) * h, P[k, 1] + (s * vm[0] + c * vm[1]) * h, P[k, 2] + vm[2] * h)
        return P

    def u_at(self, t):
        """twist of the profile at sim time t (linear interpolation; holds the last sample)."""
        x = (t - self.t0) / self.h
        if x <= 0:
            return self.V[0].copy()
        if x >= self.N:
            return self.V[-1].copy()
        k = int(x)
        f = x - k
        return self.V[k] * (1 - f) + self.V[k + 1] * f

    def shifted(self, t_now):
        """the remaining profile from t_now (re-sampled on the grid h, so poses re-integrate from wherever the robot really is)."""
        dt = max(0.0, t_now - self.t0)
        n = max(1, int(math.ceil(max(0.0, self.N * self.h - dt) / self.h - 1e-9)))
        V = np.array([self.u_at(t_now + k * self.h) for k in range(n + 1)])
        return Traj(V, self.h, t_now, self.t_sw - dt, self.T_S)


def _step(v, target, h, L, r, a_hw):
    """one segment toward `target`: returns (v_next, s*a_des)."""
    af = [(target[i] - v[i]) / h for i in range(3)]
    k = 1.0
    for i in range(3):
        if abs(af[i]) > a_hw[i]:
            k = min(k, a_hw[i] / abs(af[i]))
    ad = [k * x for x in af]
    c = acup(v, (0.0, 0.0, 0.0), r)
    m = acup(v, ad, r)
    m = (m[0] - c[0], m[1] - c[1])
    A = m[0] * m[0] + m[1] * m[1]
    B = 2 * (c[0] * m[0] + c[1] * m[1])
    C = c[0] * c[0] + c[1] * c[1] - L * L
    if A < 1e-18:
        s = 1.0
    else:
        if C <= 0:                                                       # start inside the disc: largest s staying inside
            s = min(1.0, (-B + math.sqrt(max(0.0, B * B - 4 * A * C))) / (2 * A))
        else:                                                            # |c| > L already (state outside the certified set): brake at the full slew box, the
            s = 1.0                                                      # checker reports the violation (no cup-safe backup exists from here)
        s = max(0.0, s)
    for _ in range(10):                                                  # verify at the segment END (centripetal term moved)
        v1 = [v[i] + s * ad[i] * h for i in range(3)]
        ae = acup(v1, [s * x for x in ad], r)
        if math.hypot(*ae) <= L * (1 + 1e-9) or s <= 1e-6:
            break
        if C > 0:
            break                                                       # start already violated by the centripetal term: nothing the scale can fix
        s *= 0.8
    v1 = [v[i] + s * ad[i] * h for i in range(3)]
    if s * k >= 1.0 - 1e-12:
        v1 = list(target)                                                # exact arrival
    return v1


def cup_safe_target(u, r, L, v_cap, scale_iter=40):
    """largest alpha in [0,1] with |a_cup(alpha u, 0)| <= L (steady-state centripetal term; ~alpha^2) and |alpha u| <= v_cap; returns alpha*u."""
    u = np.minimum(np.maximum(np.asarray(u, float), -v_cap), v_cap)
    if math.hypot(*acup(u, (0, 0, 0), r)) <= L:
        return u
    lo, hi = 0.0, 1.0
    for _ in range(scale_iter):
        mid = 0.5 * (lo + hi)
        if math.hypot(*acup(mid * u, (0, 0, 0), r)) <= L:
            lo = mid
        else:
            hi = mid
    return lo * u


def plan_candidate(v0, u_op, T_S, bel, gk, t0=0.0):
    """nominal (ramp to the cup-safe version of u_op at cup-safe accel, hold) for T_S, then the cup-safe backup to rest."""
    h = gk.h
    L = gk.plan_frac * bel.a_cup_max
    r = (float(bel.r_cup[0]), float(bel.r_cup[1]))
    a_hw = [float(x) for x in bel.a_hw]
    v = [float(x) for x in v0]
    tgt = cup_safe_target(u_op, r, gk.steady_frac * L, bel.v_cap)
    out = [list(v)]
    n_nom = int(math.ceil(T_S / h - 1e-9))
    for _ in range(n_nom):
        v = _step(v, tgt, h, L, r, a_hw)
        out.append(list(v))
    zero = (0.0, 0.0, 0.0)
    for _ in range(int(gk.t_back_max / h)):
        if max(abs(x) for x in v) <= EPS_V:
            break
        v = _step(v, zero, h, L, r, a_hw)
        if max(abs(x) for x in v) <= 1e-3:                               # snap the asymptotic tail (accel jump <= 1e-3/h)
            v = [0.0, 0.0, 0.0]
        out.append(list(v))
    return Traj(np.array(out), h, t0, n_nom * h, T_S)


def backup_from(v0, bel, gk, t0=0.0):
    """pure backup (T_S = 0) from the current twist."""
    return plan_candidate(v0, np.zeros(3), 0.0, bel, gk, t0)
