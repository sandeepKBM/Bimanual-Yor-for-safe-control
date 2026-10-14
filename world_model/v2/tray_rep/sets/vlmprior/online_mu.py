"""sets/vlmprior/online_mu.py -- ONLINE belief over l = ln(mu_roll) from the carry's own tracked ball motion (no probe).
SIMULATION / SAVED DATA ONLY. CPU. numba. No hardware, no network. Never reads the hidden mu_roll.

Method: a bank of N = 40 Kalman filters, one per log-spaced node of the population support [5e-4, 6e-3]. Node j runs the analytic
ball model of sets/reach.py (ball_step, with mu_eff = the calibrated map of boed/calib.json applied to node j) driven by the tray axis
and tray acceleration read from obs (as sets/estimate.BallKF does). The likelihood of node j is the product of the Kalman innovation
densities of the tracked ball position samples, so a node is rewarded when its model predicts the ball's motion.
  * tracker latency (3 ticks): a sample that appears in obs at tick t measures the state at tick t-3; the update is applied to the stored
    posterior of that tick and replayed to now (ring buffer of lat+2 states per node);
  * 30 Hz sample-and-hold, 2% dropout: a sample is "new" when the ball position part of obs changes; otherwise only predict;
  * noise-free setup (no tracker): every tick is a sample, lat = 0, small measurement noise.
Model error (analytic vs MuJoCo, plus the nodes' own wrong mu) is absorbed by the process noise sigma_a and the measurement-noise floor
sigma_model, and then by a TEMPERING exponent lam in (0, 1]: posterior_j  ∝  prior_j * exp(lam * LL_j). lam is chosen on VAL
(calibrate_online.py) so that the central 90% credible interval covers the true mu on about 90% of (carry, checkpoint) pairs.
The prior is uniform in l: prior mass of a node = the width of its cell (the end cells are half cells), so with no information the
belief is exactly uniform in l on the support and its quantiles are the population quantiles.

What the data can and cannot say: the ball model only depends on mu through the calibrated map, which is FLAT above mu_roll ~ 3.7e-3
(MuJoCo's soft rolling friction creeps; boed/calib.json). Nodes above that are not identifiable: they share one likelihood. And the
resistance only shows when the ball moves or when it would otherwise start to move, so early in the carry the belief stays near the prior.

Interface (one call per tick, in order, starting with tick 0):
    feed(obs_row, act_row=None)   obs_row (13,) = row t of the bridge obs; act_row = realised action of the previous tick (unused, kept for the interface)
    quantile(q)                   q-quantile of the posterior in l   (piecewise-linear cdf, mass uniform inside a cell)
    mean_sd_log()                 (mean, sd) of the posterior in l   (the Gaussian moment approximation)
    interval(level)               central credible interval in mu_roll units (lo, hi)
    interval_log(level)           the same in l
    loglik()                      raw (untempered) cumulative log-likelihood per node (N,), for calibration
"""
from __future__ import annotations

import numba as nb
import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401  thread pins
from world_model.v2.tray_rep.sets import reach as RC

DT = RC.DT
MU_LO, MU_HI = 5e-4, 6e-3
N_NODES = 40
L_NODES = np.linspace(np.log(MU_LO), np.log(MU_HI), N_NODES)
MU_NODES = np.exp(L_NODES)
ME_NODES = np.array([RC.MU_EFF(m) for m in MU_NODES])          # calibrated effective resistance per node
EDGES = np.concatenate([[L_NODES[0]], 0.5 * (L_NODES[1:] + L_NODES[:-1]), [L_NODES[-1]]])
WIDTH = np.diff(EDGES)
PRIOR = WIDTH / WIDTH.sum()
CENTRE = 0.5 * (EDGES[1:] + EDGES[:-1])
SIGMA_MODEL = dict(noise_free=2.0e-3, trk=1.0e-3)              # boed/model.py: analytic-vs-MuJoCo gap, added to the tracker noise


@nb.njit(cache=True)
def _predict_node(x, P, ca, cb, a, me, Q, k_in, r):
    """x (4,), P (4,4) -> state/covariance one tick later."""
    x2, _, _ = RC.ball_step(x, ca, cb, a, me, k_in, r)
    # F = [[I, dt I], [0, I]]  ->  P2 = F P F' + Q  (explicit: 4x4)
    T = P.copy()
    for i in range(2):                      # rows: pos_i += dt * vel_i
        for j in range(4):
            T[i, j] = P[i, j] + DT * P[i + 2, j]
    P2 = T.copy()
    for j in range(2):                      # columns: pos_j += dt * vel_j
        for i in range(4):
            P2[i, j] = T[i, j] + DT * T[i, j + 2]
    for i in range(4):
        for j in range(4):
            P2[i, j] += Q[i, j]
    return x2, P2


@nb.njit(cache=True)
def _tick(X, P, LL, C, V, t, R, lat, new, z, ME, Q, rvar, k_in, r):
    """advance every node from tick t-1 to tick t (t >= 1); if `new`, apply the sample z (2,) to the state of tick max(t-lat, 0) and replay."""
    N = X.shape[0]
    it, ip = t % R, (t - 1) % R
    a = np.empty(3)
    for q in range(3):
        a[q] = (V[it, q] - V[ip, q]) / DT
    ca, cb = C[ip], C[it]
    for n in range(N):
        x2, P2 = _predict_node(X[n, ip], P[n, ip], ca, cb, a, ME[n], Q, k_in, r)
        X[n, it] = x2
        P[n, it] = P2
    if new:
        m = t - lat
        if m < 0:
            m = 0
        im = m % R
        for n in range(N):
            x = X[n, im]
            Pm = P[n, im]
            s00, s01, s11 = Pm[0, 0] + rvar, Pm[0, 1], Pm[1, 1] + rvar
            s01 = Pm[0, 1]
            # the 2x2 innovation covariance S = Pm[:2,:2] + rvar I   (Pm[1,0] = Pm[0,1] holds: independent axes, but keep general)
            s10 = Pm[1, 0]
            det = s00 * s11 - s01 * s10
            if det < 1e-30:
                det = 1e-30
            n0, n1 = z[0] - x[0], z[1] - x[1]
            i00, i01, i10, i11 = s11 / det, -s01 / det, -s10 / det, s00 / det
            quad = n0 * (i00 * n0 + i01 * n1) + n1 * (i10 * n0 + i11 * n1)
            LL[n] += -0.5 * (quad + np.log(det))
            # Kalman update: K = P[:, :2] S^-1
            Kg = np.empty((4, 2))
            for i in range(4):
                Kg[i, 0] = Pm[i, 0] * i00 + Pm[i, 1] * i10
                Kg[i, 1] = Pm[i, 0] * i01 + Pm[i, 1] * i11
            xn = x.copy()
            for i in range(4):
                xn[i] = x[i] + Kg[i, 0] * n0 + Kg[i, 1] * n1
            Pn = Pm.copy()                                   # (I - K H) P   (H picks the first two states)
            for i in range(4):
                for j in range(4):
                    Pn[i, j] = Pm[i, j] - Kg[i, 0] * Pm[0, j] - Kg[i, 1] * Pm[1, j]
            X[n, im] = xn
            P[n, im] = Pn
            for k in range(m, t):                            # replay m -> t
                ik, ik1 = k % R, (k + 1) % R
                a2 = np.empty(3)
                for q in range(3):
                    a2[q] = (V[ik1, q] - V[ik, q]) / DT
                x3, P3 = _predict_node(X[n, ik], P[n, ik], C[ik], C[ik1], a2, ME[n], Q, k_in, r)
                X[n, ik1] = x3
                P[n, ik1] = P3


class OnlineMu:
    def __init__(self, noise_free=False, lam=1.0, sigma_a=None, sigma_meas=0.003, lat=3, sigma_model=None, p0_pos=None, p0_vel=0.05):
        self.noise_free = bool(noise_free)
        self.lat = 0 if noise_free else int(lat)
        self.lam = float(lam)
        sa = (0.05 if noise_free else 0.10) if sigma_a is None else float(sigma_a)
        self.sigma_a = sa
        sm = 2e-4 if noise_free else float(sigma_meas)
        smod = (SIGMA_MODEL["noise_free"] if noise_free else SIGMA_MODEL["trk"]) if sigma_model is None else float(sigma_model)
        self.rvar = sm ** 2 + smod ** 2
        Q2 = sa ** 2 * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
        self.Q = np.zeros((4, 4))
        for ax in range(2):
            ix = [ax, ax + 2]
            self.Q[np.ix_(ix, ix)] = Q2
        self.p0 = np.diag([(sm if p0_pos is None else p0_pos) ** 2] * 2 + [p0_vel ** 2] * 2)
        self.R = self.lat + 2
        self.reset()

    def reset(self):
        N, R = N_NODES, self.R
        self.X = np.zeros((N, R, 4))
        self.P = np.zeros((N, R, 4, 4))
        self.C = np.zeros((R, 3))
        self.V = np.zeros((R, 3))
        self.LL = np.zeros(N)
        self.t = -1
        self.prev_ball = None
        self.n_samples = 0
        self._w = None

    # ------------------------------------------------------------------ stream
    def feed(self, obs_row, act_row=None):
        o = np.asarray(obs_row)
        t = self.t + 1
        it = t % self.R
        self.C[it] = o[0:3]
        self.V[it] = o[3:6]
        ball = o[9:11].astype(np.float64)
        if t == 0:
            for n in range(N_NODES):
                self.X[n, :, :2] = ball
                self.X[n, :, 2:] = 0.0
                self.P[n, :] = self.p0
            self.prev_ball = ball.copy()
            self.t = 0
            self._w = None
            return
        new = True if self.noise_free else bool(np.any(ball != self.prev_ball))
        self.prev_ball = ball.copy()
        _tick(self.X, self.P, self.LL, self.C, self.V, t, self.R, self.lat, new, ball, ME_NODES, self.Q, self.rvar, RC.K_IN, RC.R_BALL)
        if new:
            self.n_samples += 1
        self.t = t
        self._w = None

    # ------------------------------------------------------------------ belief
    def loglik(self):
        return self.LL.copy()

    def weights(self):
        if self._w is None:
            self._w = posterior(self.LL, self.lam)
        return self._w

    def quantile(self, q):
        return cell_quantile(self.weights(), q)

    def mean_sd_log(self):
        return moments(self.weights())

    def interval_log(self, level=0.9):
        a = 0.5 * (1.0 - float(level))
        w = self.weights()
        return cell_quantile(w, a), cell_quantile(w, 1.0 - a)

    def interval(self, level=0.9):
        lo, hi = self.interval_log(level)
        return float(np.exp(lo)), float(np.exp(hi))


# ====================================================================== belief helpers (shared with calibrate_online / tests)
def posterior(LL, lam):
    """node-cell masses (N,) from raw cumulative log-likelihoods LL (N,) and tempering exponent lam."""
    lp = np.log(PRIOR) + float(lam) * (np.asarray(LL, float) - np.max(LL))
    p = np.exp(lp - lp.max())
    return p / p.sum()


def cell_quantile(w, q):
    """q-quantile in l of the piecewise-uniform density with cell masses w on EDGES."""
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cw /= cw[-1]
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.interp(q, cw, EDGES))             # linear cdf inside a cell; flat parts of the cdf (zero-mass cells) are skipped


def cell_cdf(w, l):
    cw = np.concatenate([[0.0], np.cumsum(w)])
    return float(np.interp(l, EDGES, cw / cw[-1]))


def moments(w):
    m = float((w * CENTRE).sum())
    v = float((w * (WIDTH ** 2 / 12.0 + (CENTRE - m) ** 2)).sum())
    return m, float(np.sqrt(v))
