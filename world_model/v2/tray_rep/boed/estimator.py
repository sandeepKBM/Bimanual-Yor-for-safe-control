"""boed/estimator.py -- a delayed-measurement Kalman filter for the ball in the tray frame, from the tracker stream.
SIMULATION / SAVED DATA ONLY. Pure numpy.

State per axis: (position, velocity) in the tray frame. Prediction uses the known in-plane gravity of the measured tray axis,
a = g c_xy / (1 + k) (the tray stands still during the probe), plus a process noise for what the model leaves out (rolling
resistance, model gap). A tracker sample arrives with `lat` ticks of delay (3 ticks = 60 ms): the filter runs on the delayed timeline
and `now()` extrapolates to the present with the same model. Samples are the rows where the ball part of obs changed (30 Hz
sample-and-hold, 2% dropout); with `noisefree` every tick is a sample.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.boed import model as MD

DT = E.DT_WM
G = MD.G
K_IN = MD.K_INERTIA


class BallKF:
    def __init__(self, lat, sigma, noisefree=False, q=0.15, sx0=0.02, sv0=0.3, grav=1.0):
        self.lat, self.sigma, self.noisefree, self.q = int(lat), float(sigma), bool(noisefree), float(q)
        self.grav = float(grav)                       # weight of the in-plane gravity term in the prediction (0 = constant velocity)
        self.sx0, self.sv0 = sx0, sv0
        self.x = np.zeros((2, 2))                    # [axis][pos, vel]
        self.P = np.diag([sx0 ** 2, sv0 ** 2])      # same covariance for both axes
        self.j = -1                                  # last state index the filter has been advanced to
        self.k = -1                                  # last obs row processed
        self.prev_ball = None
        self.n_samples = 0
        self.init = False
        self.c_hist = []                             # tray axes by state index (offset c_base)
        self.c_base = 0

    def reset_prior(self, x_mean=None, v_mean=None, sx=None, sv=None):
        if x_mean is not None:
            self.x[:, 0] = x_mean
        if v_mean is not None:
            self.x[:, 1] = v_mean
        self.P = np.diag([(sx if sx is not None else self.sx0) ** 2, (sv if sv is not None else self.sv0) ** 2])

    def _predict(self, c, n=1):
        for _ in range(n):
            a = self.grav * G * c[:2] / (1 + K_IN)
            F = np.array([[1.0, DT], [0.0, 1.0]])
            Q = self.q * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
            self.x[:, 0] += self.x[:, 1] * DT + 0.5 * a * DT * DT
            self.x[:, 1] += a * DT
            self.P = F @ self.P @ F.T + Q

    def _update(self, z):
        R = max(self.sigma, 3e-4) ** 2
        H = np.array([1.0, 0.0])
        S = self.P[0, 0] + R
        Kg = self.P @ H / S
        for ax in range(2):
            self.x[ax] += Kg * (z[ax] - self.x[ax, 0])
        self.P = (np.eye(2) - np.outer(Kg, H)) @ self.P

    def feed(self, obs_row, k):
        """process obs row k (13,). Rows must come in order, one per call (k = 0, 1, 2 ...)."""
        c = np.asarray(obs_row[0:3], float)
        ball = np.asarray(obs_row[9:11], float)
        self.c_hist.append(c)
        new = (k == 0) or self.noisefree or (self.prev_ball is not None and np.any(ball != self.prev_ball))
        self.prev_ball = ball.copy()
        self.k = k
        if not new:
            return
        j = k - self.lat
        if j < 0:
            j = 0
        if not self.init:
            if not np.any(self.x):                   # no prior mean given: start at the first sample
                self.x[:, 0] = ball
            self.j = j
            self.init = True
        else:
            self._predict_to(j)
        self._update(ball)
        self.n_samples += 1

    def _predict_to(self, j):
        while self.j < j:
            self._predict(self.c_hist[self._ci(self.j)])
            self.j += 1

    def _ci(self, idx):
        return int(min(max(idx - self.c_base, 0), len(self.c_hist) - 1))

    def now(self, with_cov=False):
        """(pos (2,), vel (2,)) extrapolated to the current tick k from the filter's delayed state (and the 2x2 covariance
        of (pos, vel) per axis when with_cov)."""
        x, P = self.x.copy(), self.P.copy()
        F = np.array([[1.0, DT], [0.0, 1.0]])
        Q = self.q * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
        for jj in range(self.j, self.k):
            a = self.grav * G * self.c_hist[self._ci(jj)][:2] / (1 + K_IN)
            x[:, 0] += x[:, 1] * DT + 0.5 * a * DT * DT
            x[:, 1] += a * DT
            P = F @ P @ F.T + Q
        if with_cov:
            return x[:, 0].copy(), x[:, 1].copy(), P
        return x[:, 0].copy(), x[:, 1].copy()

    def clone(self):
        """copy for a shadow run; keeps only the c history the filter can still need (the last lat + 3 rows)."""
        o = BallKF(self.lat, self.sigma, self.noisefree, self.q, self.sx0, self.sv0, self.grav)
        o.x, o.P, o.j, o.k, o.n_samples, o.init = self.x.copy(), self.P.copy(), self.j, self.k, self.n_samples, self.init
        o.prev_ball = None if self.prev_ball is None else self.prev_ball.copy()
        keep = max(0, len(self.c_hist) - (self.lat + 4))
        o.c_hist = [c.copy() for c in self.c_hist[keep:]]
        o.c_base = self.c_base + keep
        return o
