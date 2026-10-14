"""al/belief_v2.py -- beliefs over l = ln(mu_roll) for the active-learning policies, on the FIXED-physics (v2) mu_eff map. SIMULATION ONLY. CPU.

Everything lives on the 40 log-spaced nodes of sets/vlmprior/online_mu.py (OM.L_NODES, [5e-4, 6e-3]); a belief is a vector of RAW log-likelihoods LL (40,)
and a tempering exponent: posterior = softmax(log prior + lam * LL)  (OM.posterior; prior mass = cell width, i.e. uniform in l).
  * ProbeLL        probe-window likelihood: boed.model.probe_loglik evaluated with mus = OM.ME_NODES = the mu_eff map in force
                   (env BIMANUAL_MU_EFF_FILE = calib_v2). boed.model.MU_EFF reads the OLD boed/calib.json, which probe_loglik uses by default -> wrong map.
  * OnlineLL       sets/vlmprior/online_mu.OnlineMu fed from the carry's own motion; its raw LL is exposed so lam is applied at READ time
                   (one run serves any lam: calibrate_lam.py).
  * Posterior      combine: LL_total = lam_p * LL_probe + lam_o * LL_online on the shared prior; 95% interval, quantiles, cdf, set width.
Nothing here can see the hidden mu_roll (the only function that takes the true l is `cover_stats`, used for LOGGING after the carry).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep.boed import model as MD
from world_model.v2.tray_rep.sets.vlmprior import online_mu as OM

N = OM.N_NODES
L_POP = (float(OM.L_NODES[0]), float(OM.L_NODES[-1]))        # population support in l
L_PHYS = (float(np.log(1e-4)), float(np.log(3e-2)))
assert np.allclose(MD.MU_GRID, OM.MU_NODES), "probe grid and online grid differ"


def probe_ll_v2(obs, t0, t1, sigma, lat, prior, noisefree):
    """probe-window log-likelihood over the 40 nodes, evaluated with the v2 mu_eff map (RC.MU_EFF via OM.ME_NODES)."""
    return np.asarray(MD.probe_loglik(obs, t0, t1, sigma, lat, prior, noisefree, mus=OM.ME_NODES), float)


def posterior(parts, prior=None):
    """parts: list of (LL (N,), lam). -> cell masses (N,). prior: cell prior masses (default OM.PRIOR = uniform in l); zeros stay zero."""
    lp = np.log(np.maximum(np.asarray(OM.PRIOR if prior is None else prior, float), 1e-300))
    for ll, lam in parts:
        ll = np.asarray(ll, float)
        lp = lp + float(lam) * (ll - ll.max())
    lp = lp - lp.max()
    p = np.exp(lp)
    return p / p.sum()


def set_prior(lo, hi):
    """cell prior masses: uniform in l restricted to the interval [lo, hi] (overlap fraction of each cell; a point interval -> the cell containing it)."""
    ed = OM.EDGES
    if hi - lo < 1e-9:
        j = int(np.clip(np.searchsorted(ed, 0.5 * (lo + hi)) - 1, 0, N - 1))
        w = np.zeros(N); w[j] = 1.0
        return w
    ov = np.clip(np.minimum(ed[1:], hi) - np.maximum(ed[:-1], lo), 0.0, None)
    w = OM.PRIOR * ov / OM.WIDTH
    return w / w.sum() if w.sum() > 0 else OM.PRIOR.copy()


def interval_l(w, level=0.95):
    a = 0.5 * (1.0 - float(level))
    return OM.cell_quantile(w, a), OM.cell_quantile(w, 1.0 - a)


def interval(w, level=0.95, mode="hpd"):
    """mode 'central': equal-tail quantiles (what F5P uses). mode 'hpd': the HULL of the highest-density cells holding `level` of the mass (ties included): a flat
    (uninformed) posterior gives the WHOLE support instead of cutting 2.5% off each end with no data (the central interval of a flat belief excludes the outer
    (1-level)/2 of the range, so a cumulative intersection would exclude the extreme mu for good)."""
    if mode == "central":
        return interval_l(w, level)
    w = np.asarray(w, float)
    dens = w / OM.WIDTH
    order = np.argsort(-dens, kind="stable")
    cum = np.cumsum(w[order])
    k = min(int(np.searchsorted(cum, level * cum[-1] - 1e-12)), N - 1)
    inc = np.flatnonzero(dens >= dens[order[k]] * (1.0 - 1e-9))
    return float(OM.EDGES[inc.min()]), float(OM.EDGES[inc.max() + 1])


def width_l(w, level=0.95):
    lo, hi = interval_l(w, level)
    return hi - lo


def clip_phys(lo, hi):
    lo, hi = float(np.clip(lo, *L_PHYS)), float(np.clip(hi, *L_PHYS))
    return (lo, hi) if lo <= hi else (hi, hi)


def intersect(a, b):
    """intersection of two intervals in l; None if empty."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo <= hi else None


def to_mu(l):
    """exp with the population edges snapped to the exact constants (bit-identical to F5P's to_mu)."""
    if abs(l - L_POP[0]) < 1e-12:
        return OM.MU_LO
    if abs(l - L_POP[1]) < 1e-12:
        return OM.MU_HI
    return float(np.exp(l))


class OnlineLL:
    """OnlineMu with lam applied at read time. `lam` is the default tempering of the read-out methods."""

    def __init__(self, renv, lam, sigma_a):
        nf = renv.tracker is None
        okw = dict(noise_free=nf, lam=1.0, sigma_a=sigma_a)
        if not nf:
            okw.update(sigma_meas=renv.tracker.noise, lat=renv.tracker.lat)
        self.okw, self.lam, self.sa = okw, float(lam), float(sigma_a)
        self.om = OM.OnlineMu(**okw)

    def reset(self):
        self.om.reset()

    def feed(self, obs_row, act_row=None):
        self.om.feed(obs_row, act_row)

    @property
    def LL(self):
        return self.om.LL

    @property
    def t(self):
        return self.om.t

    @property
    def n_samples(self):
        return int(self.om.n_samples)

    def weights(self, lam=None):
        return OM.posterior(self.om.LL, self.lam if lam is None else lam)

    def interval_l(self, level=0.95, lam=None):
        return interval_l(self.weights(lam), level)

    def mean_sd_log(self, lam=None):
        return OM.moments(self.weights(lam))


def cover_stats(w, l_true):
    """LOGGING ONLY (takes the true l): coverage of the 50/90/95% central intervals of cell masses w, the cdf of the truth, the log-error of the mean."""
    out = {}
    for lev in (0.5, 0.9, 0.95):
        lo, hi = interval_l(w, lev)
        out[f"cov{int(lev * 100)}"] = bool(lo <= l_true <= hi)
        out[f"w{int(lev * 100)}"] = float(hi - lo)
    lo, hi = interval(w, 0.95, "hpd")
    out.update(hcov95=bool(lo <= l_true <= hi), hw95=float(hi - lo))
    m, sd = OM.moments(w)
    out.update(cdf=float(OM.cell_cdf(w, l_true)), mean_l=float(m), sd_l=float(sd), err_l=float(m - l_true))
    return out
