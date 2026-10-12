"""sets/vlmprior/prior.py -- SYNTHETIC "VLM physics prior" on the log rolling resistance l = ln(mu_roll). SIMULATION ONLY. Pure numpy.

THE SPEC (Experiment B). A VLM looks at the scene and says "the ball rolls with resistance about m, give or take sigma".
We do not have a VLM here; we draw what it might say. The draw is a function of the TRUE l (the generator is a simulation of an
outside source, so it may read the truth; nothing downstream of the draw may) and is deterministic per (carry seed, rho, prior_seed).

    with prob 1 - rho   HONEST          m = l_true + 0.30 * T3        T3 = Student-t with 3 dof       reported sigma = 0.40
    with prob rho       CONFIDENTLY WRONG  m = l_true + s * U(1.0, 2.0)   s = random sign (+-1)          reported sigma = 0.15

Returns (m, sigma, confident_flag). Nothing is clipped here (m can fall outside the population range [ln 5e-4, ln 6e-3]).

Common random numbers: all five underlying draws (flag uniform, t3, sign, magnitude) come from ONE generator seeded by
(carry seed, prior_seed) and rho only thresholds the flag. So for a fixed carry and prior_seed, the confidently-wrong carries at
rho = 0.1 are a subset of those at rho = 0.2, and a carry's honest draw does not change with rho. The spec is unchanged by this.
"""
from __future__ import annotations

import numpy as np

SIGMA_HONEST = 0.40
SIGMA_CONF = 0.15
HONEST_SCALE = 0.30          # honest error = 0.30 * T3
T_DOF = 3
WRONG_LO, WRONG_HI = 1.0, 2.0   # |error| of the confidently wrong prior, in l (e-folds of mu_roll)
_TAG = 0x5650               # "VP": keeps this stream apart from every other use of the carry seed


def _draws(seed, pseed):
    rng = np.random.default_rng([_TAG, int(seed), int(pseed)])
    u_flag = rng.random()
    t3 = rng.standard_t(T_DOF)
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    mag = rng.uniform(WRONG_LO, WRONG_HI)
    return u_flag, t3, sgn, mag


def draw_prior(l_true, seed, rho, pseed=0):
    """(m, sigma, confident) for a carry with true log resistance `l_true`. Deterministic per (seed, rho, pseed)."""
    u_flag, t3, sgn, mag = _draws(seed, pseed)
    if u_flag < float(rho):
        return float(l_true + sgn * mag), SIGMA_CONF, True
    return float(l_true + HONEST_SCALE * t3), SIGMA_HONEST, False


def draw_many(l_true, seeds, rho, pseed=0):
    """vectorised over carries: l_true (n,), seeds (n,) -> m (n,), sigma (n,), confident (n,) bool."""
    out = [draw_prior(l, s, rho, pseed) for l, s in zip(l_true, seeds)]
    m, sg, cf = (np.array(c) for c in zip(*out))
    return m, sg, cf.astype(bool)
