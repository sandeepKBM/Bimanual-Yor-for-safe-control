"""sets/estimate.py -- ball state estimate (x, P) from the bridge observation stream. SIMULATION / SAVED DATA ONLY.

A Kalman filter on the tray-frame ball state [x, y, vx, vy] (4). Prediction = the QS ball model of reach.py (mid rolling resistance)
driven by the tray axis and tray acceleration read from obs (the tray comes from the joint encoders: exact), covariance by the
constant-velocity transition with white acceleration noise sigma_a (this absorbs the model error and the rolling-resistance
uncertainty). Measurement = the ball POSITION in obs (columns 9, 10). The obs ball velocity is a finite difference of the same
positions and is not used.

Tracker model (known sensor spec, `env.Tracker`): a new sample is delivered at a tick when the ball part of obs changes; it measures
the state `lat` ticks ago, with noise sigma_meas. Delayed measurements are handled by rewind-and-replay: the update is applied to the
stored posterior of that earlier tick and the filter is re-predicted to now. While samples are missing (dropout, 30 Hz hold) the
filter only predicts, so P grows: this is the "tracker uncertainty set that grows with the time since the last good measurement".
Noise-free setup (no tracker): every tick is a measurement with sigma_meas = 0.2 mm and lat = 0.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.sets import reach as RC

DT = RC.DT


class BallKF:
    def __init__(self, sigma_meas=0.003, lat=3, sigma_a=0.8, p0_pos=0.003, p0_vel=0.05, mu_eff=None, noise_free=False,
                 hist=64):
        self.noise_free = bool(noise_free)
        self.sm = 2e-4 if noise_free else float(sigma_meas)
        self.lat = 0 if noise_free else int(lat)
        self.sa = float(sigma_a)
        self.p0 = np.diag([p0_pos ** 2, p0_pos ** 2, p0_vel ** 2, p0_vel ** 2])
        self.mu_eff = RC.MU_EFF(np.sqrt(5e-4 * 6e-3)) if mu_eff is None else float(mu_eff)
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = DT
        q = self.sa ** 2
        Q2 = q * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
        self.Q = np.zeros((4, 4))
        for a in range(2):
            ix = [a, a + 2]
            self.Q[np.ix_(ix, ix)] = Q2
        self.R = (self.sm ** 2) * np.eye(2)
        self.hist = int(hist)
        self.reset()

    def reset(self):
        self.t = -1
        self.xs = {}           # tick -> posterior state
        self.Ps = {}
        self.n_meas = 0
        self.last_meas_tick = -10 ** 6

    # ------------------------------------------------------------------
    def _predict(self, x, P, obs, i):
        """predict state i -> i+1 using obs rows i and i+1 (tray axis, velocity)."""
        ca = obs[i, 0:3].astype(np.float64)
        cb = obs[i + 1, 0:3].astype(np.float64)
        a = (obs[i + 1, 3:6].astype(np.float64) - obs[i, 3:6].astype(np.float64)) / DT
        x2, _, _ = RC.ball_step(x, ca, cb, a, self.mu_eff, RC.K_IN, RC.R_BALL)
        return x2, self.F @ P @ self.F.T + self.Q

    def _update(self, x, P, z):
        S = P[:2, :2] + self.R
        Kg = P[:, :2] @ np.linalg.inv(S)
        x2 = x + Kg @ (z - x[:2])
        I_KH = np.eye(4) - Kg @ np.hstack([np.eye(2), np.zeros((2, 2))])
        P2 = I_KH @ P @ I_KH.T + Kg @ self.R @ Kg.T
        return x2, P2

    def step(self, obs):
        """obs (n, 13): the stream so far (row n-1 = now). Call once per tick, in order. Returns (x (4,), P (4,4)) for the state now."""
        n = len(obs)
        t = n - 1
        if t == 0 or self.t < 0:
            x = np.array([obs[0, 9], obs[0, 10], 0.0, 0.0], np.float64)
            self.reset()
            self.xs[0], self.Ps[0] = x, self.p0.copy()
            self.xs[0], self.Ps[0] = self._update(self.xs[0], self.Ps[0], x[:2]) if self.sm > 0 else (x, self.p0.copy())
            self.t = 0
            self.last_meas_tick = 0
            return self.xs[0].copy(), self.Ps[0].copy()
        assert t == self.t + 1, "BallKF.step must be called once per tick"
        x, P = self._predict(self.xs[t - 1], self.Ps[t - 1], obs, t - 1)
        self.xs[t], self.Ps[t] = x, P
        new = True if self.noise_free else bool(np.any(obs[t, 9:11] != obs[t - 1, 9:11]))
        if new:
            m = max(t - self.lat, 0)
            z = obs[t, 9:11].astype(np.float64)
            xm, Pm = self._update(self.xs[m], self.Ps[m], z)
            self.xs[m], self.Ps[m] = xm, Pm
            for i in range(m, t):
                self.xs[i + 1], self.Ps[i + 1] = self._predict(self.xs[i], self.Ps[i], obs, i)
            self.n_meas += 1
            self.last_meas_tick = t
        old = t - self.hist
        if old in self.xs:
            del self.xs[old], self.Ps[old]
        self.t = t
        return self.xs[t].copy(), self.Ps[t].copy()

    @property
    def age(self):
        """ticks since the last delivered sample."""
        return self.t - self.last_meas_tick
