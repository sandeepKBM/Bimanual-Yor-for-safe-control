"""boed/model.py -- the analytic rolling-ball model and the belief over the hidden rolling resistance mu_roll.
SIMULATION / SAVED DATA ONLY. No hardware, no network. Pure numpy.

Analytic model (b_tray_ball/hidden.py): a ball that rolls without slipping on the tilted tray, with rolling resistance mu_roll:
    (1 + k) a = g_in - (mu_roll / r) g cos(theta) * v_hat          (moving)
    stays still while |g_in| <= (mu_roll / r) g cos(theta)          (at rest)
with g_in = g * (c_x, c_y) the in-plane gravity (c = tray axis, exact from the joint encoders), k = 0.4 (solid ball), r = 0.03 m.
The model is integrated at 4 ms (5 sub-steps per 50 Hz tick) over the measured tray-axis history, for every grid value of mu_roll
at once (vectorised). The tray is assumed to stand still (a = 0): true for the probe.

Belief: a 40-point grid, log-uniform in [5e-4, 6e-3], uniform prior in log (so the prior weights are equal).
Likelihood: Gaussian on the ball samples the tracker delivers during the probe. A sample is "new" when the ball part of obs changes.
It is the true tray-frame ball position `lat` ticks earlier plus noise (3 mm). Total sigma^2 = sigma_trk^2 + sigma_model^2, where
sigma_model is the measured analytic-vs-MuJoCo gap (see `check_model.py`). The initial ball position and velocity are nuisance
parameters with a Gaussian prior; they are integrated out by a linearised (Laplace) marginal, two Gauss-Newton steps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401  thread pins
from world_model.v2.tray_rep import env as E

G = 9.81
DT = E.DT_WM
SUB = 5                                   # sub-steps per tick (4 ms)
R_BALL = E.RADIUS
K_INERTIA = 0.4
N_GRID = 40
MU_GRID = np.exp(np.linspace(np.log(E.MU_LO), np.log(E.MU_HI), N_GRID))
LOG_MU = np.log(MU_GRID)
TRK_REAL = dict(obs_noise_m=0.003, obs_latency_ticks=3, obs_rate_hz=30.0, obs_dropout_p=0.02)
CALIB_PATH = Path(__file__).resolve().parent / "calib.json"


def load_calibration():
    """mu_roll (MuJoCo parameter) -> the effective Coulomb rolling resistance the ball shows on the probe (check_model.py).
    MuJoCo's soft rolling friction resists less than mu_roll * N (effective 0.6-0.87 of it, and it creeps above 0.0036), so the
    ideal closed form is off by 3-12 mm over a 0.3 s hold. The table is a fitted monotone map; identity if the file is missing."""
    if CALIB_PATH.exists():
        d = json.loads(CALIB_PATH.read_text())
        return np.interp(LOG_MU, np.log(np.array(d["mu"])), np.array(d["mu_eff"])), d
    return MU_GRID.copy(), None


MU_EFF, CALIB = load_calibration()


# sigma_model: the analytic-vs-MuJoCo gap that the likelihood has to absorb (added to the tracker noise in quadrature). Set from
# check_model gap / posterior: 1 mm is enough with the 3 mm tracker; the noise-free setup has no noise to hide the (systematic) gap
# behind, so it needs 2 mm to be calibrated (coverage 90% interval: 0.975 at 2 mm, 0.57 at 1 mm).
SIGMA_MODEL = dict(piper=2.0e-3, piper_trk=1.0e-3)


def sigma_total(setup):
    base = 0.0 if setup == "piper" else TRK_REAL["obs_noise_m"]
    return float(np.sqrt(base ** 2 + SIGMA_MODEL[setup] ** 2))


# ================================================================================ analytic model
def integrate(mu, c_hist, x0, v0, sub=SUB):
    """mu (M,), c_hist (n, 3) tray axis at tick starts, x0 (M, 2) or (2,), v0 (M, 2) or (2,) tray-frame ball position / velocity at tick 0.
    Returns positions (n, M, 2) at the tick starts (row 0 = x0): position at row i is the state after i ticks."""
    mu = np.asarray(mu, float)
    M = len(mu)
    x = np.broadcast_to(np.asarray(x0, float), (M, 2)).copy()
    v = np.broadcast_to(np.asarray(v0, float), (M, 2)).copy()
    n = len(c_hist)
    out = np.empty((n, M, 2))
    out[0] = x
    h = DT / sub
    for i in range(n - 1):
        # linear interpolation of the axis across the tick (the tray rotates by a few mrad per tick)
        for s in range(sub):
            f = (s + 0.5) / sub
            c = c_hist[i] * (1 - f) + c_hist[i + 1] * f
            gin = G * c[:2]
            res = (mu / R_BALL) * G * max(c[2], 0.0)                         # (M,) rolling-resistance "gravity" threshold
            sp = np.hypot(v[:, 0], v[:, 1])
            gn = np.hypot(gin[0], gin[1])
            moving = sp > 1e-4
            vh = v / np.maximum(sp, 1e-9)[:, None]
            a = np.where(moving[:, None], (gin[None, :] - res[:, None] * vh) / (1 + K_INERTIA), 0.0)
            start = (~moving) & (gn > res)                                    # at rest, gravity beats the resistance
            gh = gin / max(gn, 1e-12)
            a = np.where(start[:, None], ((gn - res)[:, None] * gh[None, :]) / (1 + K_INERTIA), a)
            vn = v + a * h
            # a decelerating ball that would reverse stops instead (then re-tested for sticking next sub-step)
            dot = (vn * v).sum(1)
            stop = moving & (dot <= 0)
            vn[stop] = 0.0
            x = x + 0.5 * (v + vn) * h
            v = vn
        out[i + 1] = x
    return out


def tilt_axis_hist(obs):
    """tray axis history (n, 3) from an observation array (n, 13)."""
    return np.asarray(obs[:, 0:3], np.float64)


# ================================================================================ belief
class Belief:
    """posterior over MU_GRID. Prior uniform in log mu (= equal weights on the log grid)."""

    def __init__(self):
        self.logp = np.full(N_GRID, -np.log(N_GRID))
        self.n_updates = 0

    def update(self, loglik):
        lp = self.logp + np.asarray(loglik, float)
        lp -= lp.max()
        p = np.exp(lp)
        self.logp = np.log(p / p.sum() + 1e-300)
        self.n_updates += 1
        return self

    @property
    def w(self):
        p = np.exp(self.logp - self.logp.max())
        return p / p.sum()

    def mean_log(self):
        return float((self.w * LOG_MU).sum())

    def sd_log(self):
        m = self.mean_log()
        return float(np.sqrt((self.w * (LOG_MU - m) ** 2).sum()))

    def cdf(self, mu_true):
        """posterior mass strictly below mu_true plus half the cell that contains it (a randomised-PIT-free midpoint cdf)."""
        w = self.w
        j = int(np.argmin(np.abs(LOG_MU - np.log(mu_true))))
        return float(w[:j].sum() + 0.5 * w[j])

    def interval(self, level=0.9):
        """central credible interval in mu: (lo, hi) at the grid points that bracket the tails."""
        w = self.w
        cw = np.cumsum(w)
        lo = MU_GRID[int(np.searchsorted(cw, (1 - level) / 2))]
        hi = MU_GRID[min(int(np.searchsorted(cw, 1 - (1 - level) / 2)), N_GRID - 1)]
        return float(lo), float(hi)

    def covers(self, mu_true, level=0.9):
        lo, hi = self.interval(level)
        step = float(np.exp(np.diff(LOG_MU).mean()))
        return bool(lo / step ** 0.5 <= mu_true <= hi * step ** 0.5)           # half a grid cell of slack on each side

    def lower_mu(self, q=0.01):
        """worst-case mu of the belief for a DROP: the lower q-quantile (a small mu rolls fastest)."""
        cw = np.cumsum(self.w)
        return float(MU_GRID[min(int(np.searchsorted(cw, q)), N_GRID - 1)])


# ================================================================================ likelihood from a probe
def new_samples(obs_ball, lat):
    """indices k (rows of obs_ball, shape (n, 2) = obs[:, 9:11]) where the tracker delivered a NEW sample. Row k holds a measurement of
    the true ball position at state index k - lat. Row 0 is the sample taken at reset (true position lat ticks 'ago' = start)."""
    ch = np.any(np.diff(obs_ball, axis=0) != 0.0, axis=1)
    return np.flatnonzero(ch) + 1


def probe_loglik(obs, t0, t1, sigma, lat, prior, noisefree=False, mus=None, return_fit=False):
    """log-likelihood over the grid (mus = effective resistances, default MU_EFF) of the ball samples of the probe window.
    obs (n, 13): observation history, state index = row; the tray axis is obs[:, 0:3]. The model starts at state t0 (the first tick of
    the pre phase) from a Gaussian prior (x_mean (2), v_mean (2), P (2, 2) = covariance of (pos, vel) per axis, from the ball filter).
    Samples are rows k in [t0 + lat, t1] where the tracker delivered a new sample; row k measures the state at k - lat >= t0.
    sigma = total std per coordinate [m] (tracker noise and the model gap, in quadrature).
    noisefree: every tick is a sample, lat 0 (the piper setup, no tracker)."""
    mus = MU_EFF if mus is None else mus
    ball = obs[:t1 + 1, 9:11].astype(np.float64)
    ts = int(t0)
    if noisefree:
        ks = np.arange(max(ts, 0) + 1, t1 + 1)
        idx_state = ks
    else:
        ks = new_samples(ball, lat)
        ks = ks[(ks >= ts + lat) & (ks <= t1)]
        idx_state = ks - lat
    if len(ks) < 3:
        out = np.zeros(len(mus))
        return (out, dict(n=len(ks))) if return_fit else out
    z = ball[ks]                                                            # (m, 2)
    c = tilt_axis_hist(obs[ts:t1 + 1])
    rows = idx_state - ts                                                   # model row of each sample
    x_mean, v_mean, P2 = prior
    n_p = np.concatenate([np.asarray(x_mean, float), np.asarray(v_mean, float)])      # prior mean of (x0, y0, vx0, vy0)
    M = len(mus)
    P = np.zeros((4, 4))
    P[0, 0] = P[1, 1] = P2[0, 0]
    P[2, 2] = P[3, 3] = P2[1, 1]
    P[0, 2] = P[2, 0] = P[1, 3] = P[3, 1] = P2[0, 1]
    eps = np.array([1e-3, 1e-3, 1e-2, 1e-2])
    nk = np.tile(n_p, (M, 1))                                               # current linearisation point per mu

    def lin(nk):
        base = integrate(mus, c, nk[:, 0:2], nk[:, 2:4])[rows]               # (m, M, 2)
        J = np.empty((M, len(ks), 2, 4))
        for q in range(4):
            n2 = nk.copy()
            n2[:, q] += eps[q]
            J[..., q] = ((integrate(mus, c, n2[:, 0:2], n2[:, 2:4])[rows] - base) / eps[q]).transpose(1, 0, 2)
        return base, J.reshape(M, -1, 4)
    for it in range(3):                                                     # Gauss-Newton to the MAP nuisance
        base, Jm = lin(nk)
        S = sigma ** 2 * np.eye(Jm.shape[1])[None] + Jm @ P @ Jm.transpose(0, 2, 1)
        rho = (z[:, None, :] - base).transpose(1, 0, 2).reshape(M, -1) - (Jm @ (n_p[None] - nk)[..., None])[..., 0]
        sol = np.linalg.solve(S, rho[..., None])[..., 0]
        nk = n_p[None] + (P @ Jm.transpose(0, 2, 1) @ sol[..., None])[..., 0]
    base, Jm = lin(nk)
    S = sigma ** 2 * np.eye(Jm.shape[1])[None] + Jm @ P @ Jm.transpose(0, 2, 1)
    rho = (z[:, None, :] - base).transpose(1, 0, 2).reshape(M, -1) - (Jm @ (n_p[None] - nk)[..., None])[..., 0]
    sol = np.linalg.solve(S, rho[..., None])[..., 0]
    ll = -0.5 * (rho * sol).sum(1) - 0.5 * np.linalg.slogdet(S)[1]
    if return_fit:
        res = (z[:, None, :] - base).transpose(1, 0, 2).reshape(M, -1)
        return ll, dict(n=len(ks), resid_rms=np.sqrt((res ** 2).mean(1)), nuis=nk)
    return ll
