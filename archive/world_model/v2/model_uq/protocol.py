"""model_uq/protocol.py -- shared evaluation protocol: splits, conformal helpers, the global baseline, metrics. SIMULATION / SAVED DATA ONLY.

Splits are by EPISODE (VAL-split carries): ep -> {pool 75%, test 25%} with a fixed hash. A "budget n" draws n windows at random from the
pool and splits them 50/50 into FIT and CAL (so a method with n=50 has 25 + 25 labelled windows). Test = every window of the held-out episodes.

A method is any object with
    fit_calibrate(Xf, yf, Xc, yc, alpha, rng) -> self        Xf/Xc (m, d) regime features (columns chosen by `cols`), yf/yc (m,) scalar target y >= 0
    bound(X) -> (m,) upper bound b(x) such that P(y <= b(x)) >= 1 - alpha (marginally, split-conformal)
Per-regime coverage is measured on the test set in fixed bins of lrho (true gn/f).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.model_uq import residual as R

ALPHA = 0.05
RHO_BINS = (-9.0, -0.5, -0.1, 0.1, 0.5, 9.0)           # lrho = log10 gn/f; bins: deep-stuck | stuck | marginal | rolling | driven
RHO_NAMES = ("deep-stuck(<0.32)", "stuck(0.32-0.8)", "marginal(0.8-1.26)", "rolling(1.26-3.2)", "driven(>3.2)")
EPS_TODAY = 0.08


def ep_group(ep, test_frac=0.25):
    """deterministic: ep -> True if in the held-out TEST group."""
    h = (np.asarray(ep, np.int64) * 2654435761) % 1000
    return h < int(1000 * test_frac)


def load(path, k=5, feats=None):
    t = R.load_table(path, k)
    t["istest"] = ep_group(t["ep"])
    return t


def design(t, cols):
    return np.stack([t[c] for c in cols], 1).astype(np.float64)


def split_pool_test(t):
    return np.flatnonzero(~t["istest"]), np.flatnonzero(t["istest"])


def draw_budget(pool_idx, n, rng):
    """n random windows from the pool, split 50/50 into FIT/CAL index arrays."""
    idx = rng.choice(pool_idx, size=n, replace=False)
    h = n // 2
    return idx[:h], idx[h:]


def conformal_quantile(scores, alpha=ALPHA):
    """finite-sample split-conformal upper quantile of scores (higher = worse): the ceil((n+1)(1-alpha))-th smallest; +inf if n too small."""
    s = np.sort(np.asarray(scores, np.float64))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= n else float("inf")


class GlobalConformal:
    """today's approach made calibrated: ONE box eps for every state, eps = conformal 95% quantile of y on CAL (FIT unused)."""
    name = "global"

    def fit_calibrate(self, Xf, yf, Xc, yc, alpha=ALPHA, rng=None):
        self.eps = conformal_quantile(np.concatenate([yc]), alpha)
        return self

    def bound(self, X):
        return np.full(len(X), self.eps)


def evaluate(method, t, cols, test_idx, bins=RHO_BINS, cap=None):
    """-> dict(cov, width (mean of the finite bound, capped at `cap` if given), cov_by_regime (list), width_by_regime, n_by_regime, frac_inf)."""
    X = design(t, cols)[test_idx]
    y = t["y"][test_idx]
    b = np.asarray(method.bound(X), np.float64)
    fin = np.isfinite(b)
    bb = np.where(fin, b, cap if cap is not None else np.nan)
    ok = y <= b
    out = dict(cov=float(ok.mean()), width=float(np.nanmean(bb)), frac_inf=float(1 - fin.mean()))
    lr = t["lrho"][test_idx]
    cb, wb, nb = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (lr >= lo) & (lr < hi)
        nb.append(int(m.sum()))
        cb.append(float(ok[m].mean()) if m.sum() else float("nan"))
        wb.append(float(np.nanmean(bb[m])) if m.sum() else float("nan"))
    out.update(cov_by_regime=cb, width_by_regime=wb, n_by_regime=nb)
    return out
