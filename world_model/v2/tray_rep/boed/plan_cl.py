"""boed/plan_cl.py -- CLOSED-LOOP nominal for the BOED harness. SIMULATION / SAVED DATA ONLY. CPU. No hardware.

New file; boed/carry.py and boed/episode.py are NOT changed. `CLPlan` follows the SAME minimum-jerk path and timing as `carry.Plan` at
speed factor I (it takes the transports (t0, T, D) from a `carry.Plan` built at the carry's start position, so the reference is the
open-loop plan's own first-pass profile), but its command is a tracking law with a time-scaled reference:

  s            reference clock [plan s]; ds/dt = r(t) in [0, 1]  (r = 1 -> the reference plays at the planned pace)
  p_r(s), v_r = p_r'(s) r, a_r = p_r''(s) r^2 + p_r'(s) rdot        (min-jerk reference, unit rate derivatives scaled by r)
  e = p_r(s) - p                                                    (tray xy tracking error)
  a = a_r + Kp e + Kd (v_r - v)                                     (then the accel VECTOR is clipped at 5*a_scale = renv.a_max)
  r_target = clip((e_hi - |e|) / (e_hi - e_lo), 0, 1);  r slews to r_target (down at `dn` 1/s, up at `up` 1/s)

so when a safety filter brakes (or the Piper authority box clips the command) the error grows, the reference slows and then PAUSES (r = 0
at |e| >= e_hi) and resumes as the error closes: braking delays the carry instead of derailing it. After the reference has played out,
the same PD law holds the reference at the goal until arrival (within 1.5 cm and |v| <= 3 cm/s) or a timeout.
Tilt: the plan's own law, unchanged (carry_u_clip lines: c* = normalize([kappa a_x, kappa a_y, g]), w = (c x c*)/tau + OU noise, clip +-2 w_scale),
driven by the COMMANDED accel a (so the tilt anti-roll follows the closed-loop acceleration, incl. the braking part of the feedback).
The OU/rng stream of the carry is consumed exactly once per tick, as in carry_u_clip.

Gains: critically damped, omega = 4 rad/s: Kp = omega^2 = 16 1/s^2, Kd = 2 omega = 8 1/s  (zeta = 1). 4 rad/s = 0.64 Hz: 78x below the 50 Hz tick
(discrete-time margin: omega dt = 0.08, explicit-Euler stable up to omega dt ~ 2 zeta), 1/omega = 0.25 s closes an error e in ~1 s, and the
feedback accel of the pause threshold e_hi = 6 cm is Kp e_hi = 0.96 m/s^2 (64% of a_max = 1.5 m/s^2 for the Piper), so the PD alone cannot
saturate the box but the feed-forward (peak ~ I^2 a_max) plus PD can at I_rel ~ 1 (clipped by the vector clip; the reference then slows).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.boed import carry as CA
from world_model.v2.testbeds.b_tray_ball.carry import G

DT = E.DT_WM
OMEGA = 4.0
E_LO, E_HI = 0.02, 0.06          # [m] error below which the reference plays at full pace / above which it is paused
R_DN, R_UP = 5.0, 2.0            # [1/s] slew of the reference rate (down fast, up slowly: no chatter at the pause edge)
TOL, VTOL = 0.015, 0.03          # arrival: within 1.5 cm of the goal and |v| <= 3 cm/s


def _mj(x):
    """min-jerk position fraction and its first two derivatives on [0, 1]."""
    x = min(max(x, 0.0), 1.0)
    return (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5, 30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4, 60 * x - 180 * x ** 2 + 120 * x ** 3)


class CLPlan:
    waypoint_mode = True

    def __init__(self, cr, a_scale, w_scale, I, home_xy=(0.0, 0.0), omega=OMEGA, e_lo=E_LO, e_hi=E_HI, r_dn=R_DN, r_up=R_UP,
                 closed=True, max_s=None):
        base = CA.Plan(cr, a_scale, w_scale, I, home_xy=home_xy, waypoint_mode=True)    # same path + timing as the open-loop plan
        self.cr = base.cr
        self.a_max = 5.0 * float(a_scale)
        self.w_clip = 2.0 * float(w_scale)
        self.segs = [(float(t0), float(T), np.asarray(D, float)[:2].copy()) for (t0, T, D) in base.segs]
        self.W = [np.asarray(w, float)[:2].copy() for w in base.W]
        self.goal = self.W[-1].copy()
        self.s_end = self.segs[-1][0] + self.segs[-1][1]
        self.T_ref = float(base.T_end)
        self.kp, self.kd = float(omega) ** 2, 2.0 * float(omega)
        self.e_lo, self.e_hi, self.r_dn, self.r_up = float(e_lo), float(e_hi), float(r_dn), float(r_up)
        self.closed = bool(closed)                   # closed=False: gains 0 and no pausing = the open-loop feed-forward (self-test)
        self.max_s = float(max_s) if max_s is not None else 2.5 * self.T_ref + 8.0
        self.s, self.r = 0.0, 1.0
        self.n_pause = self.n_slow = self.n_u = 0
        self.max_err = 0.0
        self.s_final = None

    @property
    def n_ticks(self):
        return int(self.max_s / DT)

    def ref(self, s):
        """(p, dp/ds, d2p/ds2) of the unit-rate reference at clock s."""
        if s >= self.s_end:
            return self.goal, np.zeros(2), np.zeros(2)
        k = -1
        for j, (t0, T, D) in enumerate(self.segs):
            if t0 <= s:
                k = j
        if k < 0:
            return self.W[0], np.zeros(2), np.zeros(2)
        t0, T, D = self.segs[k]
        if s >= t0 + T:
            return self.W[k + 1], np.zeros(2), np.zeros(2)
        m, m1, m2 = _mj((s - t0) / T)
        return self.W[k] + D * m, D * m1 / T, D * m2 / T ** 2

    def touchup(self, tp, pos_xy):
        return False

    def u(self, tp, axis, pos_xy, vel_xy):
        """command at this tick (stateful: advances the reference clock by one tick)."""
        pos, vel = np.asarray(pos_xy, float)[:2], np.asarray(vel_xy, float)[:2]
        p_r, dp, ddp = self.ref(self.s)
        e_vec = p_r - pos
        e = float(np.linalg.norm(e_vec))
        self.max_err = max(self.max_err, e)
        if self.closed and self.s < self.s_end:
            tgt = float(np.clip((self.e_hi - e) / (self.e_hi - self.e_lo), 0.0, 1.0))
            r_new = self.r + float(np.clip(tgt - self.r, -self.r_dn * DT, self.r_up * DT))
        else:
            r_new = 1.0 if self.s < self.s_end else self.r
        rdot = (r_new - self.r) / DT
        self.r = r_new
        v_r, a_r = dp * self.r, ddp * self.r ** 2 + dp * rdot
        a = a_r + (self.kp * e_vec + self.kd * (v_r - vel) if self.closed else 0.0)
        if self.s < self.s_end:
            self.n_pause += int(self.r < 1e-6)
            self.n_slow += int(self.r < 0.999)
        self.n_u += 1
        self.s += self.r * DT
        if self.s_final is None and self.s >= self.s_end:
            self.s_final = self.n_u * DT
        na = float(np.hypot(*a))
        if na > self.a_max:
            a = a * (self.a_max / na)
        a3 = np.array([a[0], a[1], 0.0])
        cr = self.cr
        ct = np.array([cr["kappa"] * a3[0], cr["kappa"] * a3[1], G]); ct /= np.linalg.norm(ct)
        w = np.cross(axis, ct) / cr["tau"]
        cr["ou"] = 0.9 * cr["ou"] + cr["noise"] * cr["rng"].normal(size=2) * np.sqrt(1 - 0.81)
        w[:2] = np.clip(w[:2] + cr["ou"], -self.w_clip, self.w_clip)
        w[2] = 0.0
        return np.concatenate([a3, w])

    def arrived(self, pos_xy, vel_xy):
        return bool(self.s >= self.s_end and np.linalg.norm(self.goal - np.asarray(pos_xy, float)[:2]) <= TOL
                    and np.linalg.norm(np.asarray(vel_xy, float)[:2]) <= VTOL)

    def stats(self):
        return dict(cl_s_final=float(self.s), cl_ref_done_s=(float(self.s_final) if self.s_final is not None else float("nan")),
                    cl_pause_ticks=int(self.n_pause), cl_slow_ticks=int(self.n_slow), cl_max_err=float(self.max_err),
                    cl_T_ref=float(self.T_ref))
