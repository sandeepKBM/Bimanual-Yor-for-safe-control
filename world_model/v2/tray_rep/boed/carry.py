"""boed/carry.py -- the carry plan with a speed knob ("intensity") and re-planning. SIMULATION / SAVED DATA ONLY.

The stored carry (gen_nominal, config piper: intensity 3.0, clip mode) is three straight min-jerk transports with an anti-roll tilt
loop. `Plan` rebuilds it from the episode seed with the SAME random draws as `bridge.episode.Nominal` / `gen_nominal.run_episode`
(mu_roll, ball start, carry plan), so the speed factor I = 1.0 reproduces the stored carry bitwise.

intensity I (speed factor): every duration of the stored plan is divided by I (transports, rests, tail). The path (the three
displacement vectors) is unchanged. I = 1 is the stored (nominal) carry; I < 1 is slower: peak acceleration falls as I^2.
The arm clips the acceleration at 5 * a_scale = 1.5 m/s^2. The stored plan is far above that, so at I = 1 the tray covers only about
30% of the nominal path (see the bridge path-progress metric). A transport whose peak acceleration fits under the clip is
completed. That happens for I <~ 0.4 (see `feasible_speed`).

Waypoints. Below I = 1 each transport re-aims at its waypoint from where the tray is when the transport starts (so a transport that
fell short is made up by the next). At I = 1 the stored displacement vectors are used unchanged (bridge-identical).
A transport's duration with a displacement D' instead of D is  T * sqrt(|D'| / |D|) / I  (same peak acceleration per I).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep import gen_nominal as GN
from world_model.v2.testbeds.b_tray_ball.carry import make_carry

DT = E.DT_WM
MJ_PEAK = 5.7735                          # peak of mj_acc(s) = 60 s - 180 s^2 + 120 s^3 on [0, 1]


def draw_episode(seed, intensity):
    """(mu, ball_xy, carry dict) with the generator's random-draw order."""
    rng = np.random.default_rng(int(seed))
    mu = E.sample_mu_roll(rng)
    ball_xy = rng.normal(0, 0.005, 2)
    cr = make_carry(rng, intensity)
    return mu, ball_xy, cr


class Plan:
    """Operator plan at speed factor I. u(tp, axis) gives the nominal command at plan time tp [s]."""

    def __init__(self, cr, a_scale, w_scale, I=1.0, home_xy=(0.0, 0.0), waypoint_mode=None):
        self.a_scale, self.w_scale = float(a_scale), float(w_scale)
        self.cr = dict(cr)                                                  # shares the tilt-loop rng / OU state with `cr`
        self.ref = [(float(t0), float(T), np.array(D, float)) for (t0, T, D) in cr["segs"]]      # stored plan (I = 1)
        self.rest = [self.ref[i + 1][0] - (self.ref[i][0] + self.ref[i][1]) for i in range(len(self.ref) - 1)]
        self.tail = float(cr["T_end"] - (self.ref[-1][0] + self.ref[-1][1]))
        self.I = float(I)
        self.waypoint_mode = (self.I < 1.0 - 1e-9) if waypoint_mode is None else bool(waypoint_mode)
        self.W = [np.array(home_xy, float)]
        for (_, _, D) in self.ref:
            self.W.append(self.W[-1] + D[:2])                               # nominal waypoints (xy)
        self.goal = self.W[-1].copy()
        self._aimed, self.n_touch = set(), 0
        self._build_from(0, self.W[0], origin=0.0)

    # ------------------------------------------------------------------ plan construction
    def _scaled_T(self, i, Dnorm):
        T0, D0 = self.ref[i][1], float(np.linalg.norm(self.ref[i][2]))
        return T0 * np.sqrt(max(Dnorm, 1e-9) / max(D0, 1e-9)) / self.I

    def _build_from(self, i0, p_xy, origin):
        """plan transports i0.. from tray position p_xy, plan time starting at `origin` (stationary start)."""
        segs, t = [], float(origin)
        for i in range(i0, len(self.ref)):
            if self.waypoint_mode or i0 > 0:
                D = np.zeros(3)
                D[:2] = (self.W[i + 1] - (p_xy if i == i0 else self.W[i]))
            else:
                D = self.ref[i][2].copy()
            T = self._scaled_T(i, float(np.linalg.norm(D[:2]))) if (self.waypoint_mode or i0 > 0) else self.ref[i][1] / self.I
            segs.append((t, T, D))
            t += T + (self.rest[i] / self.I if i < len(self.rest) else 0.0)
        self.segs = segs
        self._aimed = {i0} if (self.waypoint_mode or i0 > 0) else set()
        self.cr["segs"] = segs
        self.cr["T_end"] = t + self.tail / self.I
        self.T_end = self.cr["T_end"]
        self.i_first = i0

    def replan_from(self, tp, p_xy, I_new):
        """after a stop at plan time tp (tray stationary at p_xy): continue to the remaining waypoints at speed I_new.
        Returns the new plan duration [s] (plan time restarts at 0)."""
        m = self.active_segment(tp)
        t0, T, _ = self.segs[m - self.i_first]
        i0 = m + 1 if tp >= t0 + T else m
        self.I = float(I_new)
        self.waypoint_mode = True
        if i0 >= len(self.ref):                                             # all transports were planned and played
            self.segs, self.cr["segs"], self.cr["T_end"], self.T_end, self.i_first = [], [], 0.5, 0.5, i0
            return self.T_end
        self._build_from(i0, np.asarray(p_xy, float)[:2], origin=0.0)
        return self.T_end

    def active_segment(self, tp):
        """absolute index of the last transport with start <= tp (clamped to the first one in the current plan)."""
        k = 0
        for j, (t0, T, _) in enumerate(self.segs):
            if t0 <= tp + 1e-12:
                k = j
        return self.i_first + k

    # ------------------------------------------------------------------ command
    def u(self, tp, axis, pos_xy=None):
        """nominal command at plan time tp. In waypoint mode, pass the tray xy: a transport that is about to start is re-aimed at its
        waypoint from where the tray actually is (the filter may have moved the tray off the plan)."""
        if self.waypoint_mode and pos_xy is not None:
            self._reaim(tp, np.asarray(pos_xy, float)[:2])
        return GN.carry_u_clip(self.cr, tp, axis, self.a_scale, self.w_scale)

    def _reaim(self, tp, pos):
        for j, (t0, T, D) in enumerate(self.segs):
            i = self.i_first + j
            if i in self._aimed:
                continue
            if tp + 1e-12 >= t0:
                Dn = np.zeros(3)
                Dn[:2] = self.W[i + 1] - pos
                Tn = self._scaled_T(i, float(np.linalg.norm(Dn[:2])))
                dT = Tn - T
                self.segs[j] = (t0, Tn, Dn)
                for jj in range(j + 1, len(self.segs)):
                    a, b, c = self.segs[jj]
                    self.segs[jj] = (a + dT, b, c)
                self.T_end += dT
                self.cr["T_end"] = self.T_end
                self._aimed.add(i)
            return

    def arrived(self, tp, pos_xy, vel_xy, tol=0.015, vtol=0.03):
        """the last transport is over, the tray is within `tol` of the goal and (almost) stopped."""
        if not self.segs:
            return True
        t0, T, _ = self.segs[-1]
        return bool(tp >= t0 + T and np.linalg.norm(self.goal - np.asarray(pos_xy, float)[:2]) <= tol
                    and np.linalg.norm(np.asarray(vel_xy, float)[:2]) <= vtol)

    def touchup(self, tp, pos_xy, tol=0.012):
        """if the tray is more than `tol` from the goal when the plan ends, append one transport to the goal. -> True if appended."""
        pos = np.asarray(pos_xy, float)[:2]
        d = float(np.linalg.norm(self.goal - pos))
        if not self.waypoint_mode or d <= tol or self.n_touch >= 2:
            return False
        Dn = np.zeros(3)
        Dn[:2] = self.goal - pos
        T0 = max(self._scaled_T(len(self.ref) - 1, d), 0.5)
        t0 = self.T_end
        self.segs.append((t0, T0, Dn))
        self._aimed.add(self.i_first + len(self.segs) - 1)
        self.T_end = t0 + T0 + 0.3
        self.cr["T_end"] = self.T_end
        self.n_touch += 1
        return True

    @property
    def n_ticks(self):
        return int(self.T_end / DT)


def feasible_speed(cr, a_max):
    """largest speed factor I at which every transport's peak acceleration MJ_PEAK * D * I^2 / T^2 stays under a_max."""
    return float(min(T * np.sqrt(a_max / (MJ_PEAK * np.linalg.norm(D))) for (_, T, D) in cr["segs"]))
