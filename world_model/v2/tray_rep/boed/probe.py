"""boed/probe.py -- the identification probe: a closed-loop tilt-and-hold of the tray, and its certificate.
SIMULATION / SAVED DATA ONLY. No hardware, no network.

The tray stands still (a = 0) and tilts about its y axis by `theta` (0.12 rad). The ball rolls down the slope with an acceleration
that depends on mu_roll (analytic model: model.py). The tilt command is the operator's anti-roll law
    w = (c x c_des) / tau_p            c = realised axis, c_des = the desired axis for the current phase
and goes through the same box (|w| <= w_max) and slew limit (alpha) as every command, so the arm authority of configs/piper.json
shapes the ramp: about 0.22 s to reach 0.12 rad.

Phases (ticks of 20 ms)
  settle   (probe-on-trigger only) balance the ball to the tray centre until it is still
  pre      level, 8 ticks: ball samples before the tilt (they fix the start state)
  hold     +theta toward `direction`, until the tracked ball has moved `d_trig` or `n_hold_max` ticks. A ball that does not move
           (large mu_roll) is held the full time. A ball that moves is caught early.
  balance  a PD law on the tracked ball (tilt command = -(kp x + kd v)(1+k)/g, tilt <= theta_bal) brings it back to the centre and
           stops it. A free ball on a level tray with a small mu_roll would otherwise roll off the tray.
The likelihood window is pre + hold only (the model is checked there, not in the balance phase).

Certificate. Before the probe runs, the whole controller is run on a shadow env (restored from the true state, mu_roll set to the
worst case of the belief) and must (a) keep the ball on the tray with an edge distance of at least `d_margin` all the way, and (b) end
in a state that the oracle library (oracle.certify) labels hold_ok or savable. The shadow uses the true MuJoCo state: it is a
simulation-side certificate, not something a robot could compute without a model (documented in the README).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.boed import model as MD
from world_model.v2.tray_rep.boed.estimator import BallKF

DT = E.DT_WM
G = MD.G
K_IN = MD.K_INERTIA


@dataclass
class ProbeSpec:
    theta: float = 0.12            # [rad] tilt of the hold phase
    n_pre: int = 8                 # level ticks before the tilt
    n_hold_max: int = 28           # longest hold (0.56 s), counted from the command step; the ramp is inside it
    d_trig: float = 0.020          # [m] end the hold when the tracked ball has moved this far
    tau_p: float = 0.10            # [s] lag of the tilt law
    theta_bal: float = 0.12        # [rad] largest tilt of the balance law
    kp: float = 25.0               # [1/s^2]
    kd: float = 10.0               # [1/s]
    n_settle_max: int = 50         # longest settle phase before the probe (probe-on-trigger)
    n_bal_max: int = 90            # longest balance phase after the hold
    still_x: float = 0.015         # [m] settled: |x| below this ...

    still_v: float = 0.06          # [m/s] ... and |v| below this (the filtered speed has about 0.04 m/s of noise) ...
    still_n: int = 6               # ... for this many ticks in a row
    n_bal_min: int = 30            # the balance phase may end on a still ball that is off-centre (a sticky ball) after this many ticks
    theta_break: float = 0.09      # [rad] a slow ball off the centre gets at least this tilt toward it (breaks the stiction)
    lik_extra: int = 2             # the likelihood window ends this many ticks after the hold phase
    d_margin: float = 0.02         # [m] certificate: edge distance at least this all along the shadow probe

    def to_dict(self):
        return asdict(self)


def level_command(c, spec, w_max):
    w = np.cross(c, np.array([0.0, 0.0, 1.0])) / spec.tau_p
    w[2] = 0.0
    w[:2] = np.clip(w[:2], -w_max, w_max)
    return np.concatenate([np.zeros(3), w])


def tilt_to(c, c_des, spec, w_max):
    w = np.cross(c, c_des) / spec.tau_p
    w[2] = 0.0
    w[:2] = np.clip(w[:2], -w_max, w_max)
    return np.concatenate([np.zeros(3), w])


def axis_for_accel(a_xy, theta_max):
    """tray axis whose in-plane gravity gives the ball the in-plane acceleration a_xy (the ball accelerates by g c_xy / (1 + k))."""
    cxy = np.asarray(a_xy, float) * (1 + K_IN) / G
    n = float(np.hypot(cxy[0], cxy[1]))
    s = np.sin(theta_max)
    if n > s:
        cxy = cxy * (s / n)
    return np.array([cxy[0], cxy[1], np.sqrt(max(1.0 - float(cxy @ cxy), 0.0))])


class ProbeController:
    """stateful: call observe(obs_row, k) after every tick (feeds the ball filter), then u = step(c, k). `kf` is shared with the
    episode (probe-on-trigger) or a clone (shadow). direction: roll toward +1 / -1 along tray x (chosen from the ball position)."""

    def __init__(self, spec: ProbeSpec, kf: BallKF, w_max, balance_first=False):
        self.spec, self.kf, self.w_max = spec, kf, float(w_max)
        self.phase = "settle" if balance_first else "pre"
        self.n_in_phase = 0
        self.still = 0
        self.direction = 1.0
        self.t_pre = self.t_hold_end = -1
        self.x_pre = None
        self.prior = None
        self.n_hold = 0
        self.n_total = 0
        self.settled_end = False
        self.reason = ""

    # -------------------------------------------------------------- helpers
    def _balance(self, c):
        s = self.spec
        x, v = self.kf.now()
        a = -s.kp * x - s.kd * v
        cd = axis_for_accel(a, s.theta_bal)
        if float(np.hypot(*x)) > s.still_x and float(np.hypot(*v)) < 0.02:        # slow and off-centre: make sure the tilt can move it
            n = float(np.hypot(cd[0], cd[1]))
            if n < np.sin(s.theta_break):
                dxy = -x / max(float(np.hypot(*x)), 1e-9)
                cd = np.array([dxy[0] * np.sin(s.theta_break), dxy[1] * np.sin(s.theta_break), np.cos(s.theta_break)])
        return tilt_to(c, cd, s, self.w_max), x, v

    def _is_still(self, x, v, need_centre=True):
        s = self.spec
        ok_v = abs(v[0]) <= s.still_v and abs(v[1]) <= s.still_v
        ok = ok_v and ((abs(x[0]) <= s.still_x and abs(x[1]) <= s.still_x) or not need_centre)
        self.still = self.still + 1 if ok else 0
        return self.still >= s.still_n

    def _begin_pre(self, k):
        self.phase, self.n_in_phase = "pre", 0
        self.t_pre = int(k)
        x, v, P = self.kf.now(with_cov=True)
        self.x_pre = x.copy()
        self.prior = (x.copy(), v.copy(), P.copy())
        self.direction = -1.0 if x[0] > 0 else 1.0

    # -------------------------------------------------------------- step
    def step(self, c, k):
        s = self.spec
        self.n_total += 1
        if self.phase == "settle":
            u, x, v = self._balance(c)
            self.n_in_phase += 1
            if self._is_still(x, v) or self.n_in_phase >= s.n_settle_max:
                self._begin_pre(k)
            return u
        if self.phase == "pre":
            if self.t_pre < 0:
                self._begin_pre(k)
            u = level_command(c, s, self.w_max)
            self.n_in_phase += 1
            if self.n_in_phase >= s.n_pre:
                self.phase, self.n_in_phase = "hold", 0
            return u
        if self.phase == "hold":
            x, v = self.kf.now()
            disp = float((x - self.x_pre)[0]) * self.direction
            if (disp >= s.d_trig and self.n_in_phase >= 4) or self.n_in_phase >= s.n_hold_max:
                self.t_hold_end = int(k)
                self.n_hold = self.n_in_phase
                self.reason = "trigger" if disp >= s.d_trig else "max_hold"
                self.phase, self.n_in_phase, self.still = "balance", 0, 0
            else:
                th = s.theta * self.direction
                u = tilt_to(c, np.array([np.sin(th), 0.0, np.cos(th)]), s, self.w_max)
                self.n_in_phase += 1
                return u
        if self.phase == "balance":
            u, x, v = self._balance(c)
            self.n_in_phase += 1
            if self._is_still(x, v, need_centre=self.n_in_phase < s.n_bal_min) or self.n_in_phase >= s.n_bal_max:
                self.settled_end = self.still >= s.still_n
                self.phase = "done"
            return u
        return level_command(c, s, self.w_max)

    @property
    def done(self):
        return self.phase == "done"

    def window(self):
        """(t0, t1) rows of the likelihood window (state indices of the episode's observation array)."""
        return self.t_pre, self.t_hold_end + self.spec.lik_extra


KF_Q, KF_GRAV = 0.01, 1.0         # process noise and gravity weight of the ball filter (tuned in explore: estimate error vs truth)


def make_kf(trk, noisefree, sigma_floor=5e-4, q=None, grav=None):
    """ball filter for a tracker setting dict (model.TRK_REAL) or the noise-free setup."""
    q = KF_Q if q is None else q
    grav = KF_GRAV if grav is None else grav
    if noisefree:
        return BallKF(lat=0, sigma=sigma_floor, noisefree=True, q=q, grav=grav)
    return BallKF(lat=int(trk["obs_latency_ticks"]), sigma=float(trk["obs_noise_m"]), noisefree=False, q=q, grav=grav)


def run_shadow(shadow, snap, mu_worst, spec, kf, k_real, balance_first, w_max, max_ticks=None):
    """run the whole probe on the shadow env from `snap` with mu_roll = mu_worst. kf: a clone of the episode's ball filter.
    Returns dict(ok_ball, dmin, dropped, n_ticks, phase). The shadow's tracker noise differs from the real run's (restore reseeds it)."""
    shadow.restore(snap, mu_roll=mu_worst)
    ctrl = ProbeController(spec, kf, w_max, balance_first)
    dmin, dropped, k = 1e9, False, int(k_real)
    n_max = max_ticks or (spec.n_settle_max + spec.n_pre + spec.n_hold_max + spec.n_bal_max + 5)
    for i in range(n_max):
        u = ctrl.step(shadow.env.axis(), k)
        r = shadow.tick(u)
        k += 1
        kf.feed(r["obs"], k)
        dmin = min(dmin, r["d"])
        if r["failed"] or r["tunnel"]:
            dropped = True
            break
        if ctrl.done:
            break
    return dict(dropped=bool(dropped), dmin=float(dmin), n_ticks=i + 1, phase=ctrl.phase, settled=bool(ctrl.settled_end),
                reason=ctrl.reason, n_hold=int(ctrl.n_hold))


def run_probe(renv, spec, kf, balance_first=False, n_pre_rows=1):
    """run the probe on `renv` (already reset, state = row 0 of the history) with no filter and no carry. Feeds `kf` from row 0.
    Returns dict(obs (n,13), loc (n,3) true tray-frame ball position, act, window, dropped, dmin, ctrl)."""
    r0 = renv.current()
    obs, loc = [r0["obs"]], [renv.env.local().copy()]
    kf.feed(r0["obs"], 0)
    ctrl = ProbeController(spec, kf, renv.w_max, balance_first)
    dmin, dropped = r0["d"], False
    n_max = spec.n_settle_max + spec.n_pre + spec.n_hold_max + spec.n_bal_max + 5
    from world_model.v2.tray_rep.bridge.filters import project_box
    for i in range(n_max):
        u = project_box(ctrl.step(renv.env.axis(), len(obs) - 1), renv.a_max, renv.w_max)
        r = renv.tick(u)
        obs.append(r["obs"]); loc.append(renv.env.local().copy())
        kf.feed(r["obs"], len(obs) - 1)
        dmin = min(dmin, r["d"])
        if r["failed"] or r["tunnel"]:
            dropped = True
            break
        if ctrl.done:
            break
    return dict(obs=np.array(obs), loc=np.array(loc), dropped=bool(dropped), dmin=float(dmin), ctrl=ctrl, n=len(obs) - 1)
