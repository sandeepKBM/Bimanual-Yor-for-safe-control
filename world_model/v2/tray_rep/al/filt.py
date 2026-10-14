"""al/filt.py -- F5 with a PER-CARRY mu set built from the carry's hidden mu (for the value-of-information map). SIMULATION ONLY. CPU.

F5SetAL IS F5 (sets/filter.py F5SetFilter, unchanged). Only the rolling-resistance set handed to F5 by set_mu(lo, hi) at carry start differs. The set is
FIXED for the whole carry (no online belief, no refresh), defined from the true mu_roll m (read from ctx['hid'] -- this filter is a map-making tool,
not a deployable controller). All sets are clipped to the population / calibration support [5e-4, 6e-3] (the calibrated mu_eff map np.interp-clamps
outside it, so a bound outside would silently be the edge anyway).

  mode=oracle   [m, m]
  mode=narrow   [m/1.5, 1.5 m]            log-width 2 ln 1.5 = 0.81
  mode=medium   [m/3,   3 m]              log-width 2 ln 3   = 2.20
  mode=mis      c = m e^1.5 (confidently over-estimated); [c/1.5, 1.5 c]  (does not cover m unless clipped to the edge)
  mode=full     [5e-4, 6e-3]              the population range (F5 default)
  mode=popr     [pop_lo, pop_hi]          the regime's own population range (differs from full only for the low-mu regime)
Spec keys: mode, pop_lo, pop_hi, narrow (1.5), medium (3), shift (1.5), plus any F5 key.
Record extras (stats): x_al_mode, x_al_lo, x_al_hi, x_al_lw (clipped log-width ln(hi/lo)), x_al_lw_raw (unclipped), x_al_cov (m in clipped set),
x_al_cov_raw (m in unclipped set), x_al_lerr (centre of the clipped set minus ln m), x_al_mu_lo/hi_eff (calibrated mu_eff of the bounds).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep.sets.filter import F5SetFilter

POP_LO, POP_HI = 5e-4, 6e-3
MODES = ("oracle", "narrow", "medium", "mis", "full", "popr")


def raw_set(mode, m, narrow=1.5, medium=3.0, shift=1.5, pop=(POP_LO, POP_HI)):
    """unclipped (lo, hi) of the set for true mu m."""
    if mode == "oracle":
        return m, m
    if mode == "narrow":
        return m / narrow, m * narrow
    if mode == "medium":
        return m / medium, m * medium
    if mode == "mis":
        c = m * float(np.exp(shift))
        return c / narrow, c * narrow
    if mode == "full":
        return POP_LO, POP_HI
    if mode == "popr":
        return pop
    raise ValueError(mode)


def clip_set(lo, hi):
    lo, hi = max(lo, POP_LO), min(hi, POP_HI)
    if lo > hi:                                   # set entirely above (below) the support: collapse onto the nearest edge
        lo = hi = POP_HI if lo > POP_HI else POP_LO
    return float(lo), float(hi)


class F5SetAL(F5SetFilter):
    name = "F5AL"

    def __init__(self, renv, mode="oracle", pop_lo=POP_LO, pop_hi=POP_HI, narrow=1.5, medium=3.0, shift=1.5, **f5kw):
        super().__init__(renv, **f5kw)
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.kw = dict(narrow=float(narrow), medium=float(medium), shift=float(shift), pop=(float(pop_lo), float(pop_hi)))
        self._al = {}

    def begin(self, ctx):
        super().begin(ctx)
        m = float(ctx["hid"])
        rlo, rhi = raw_set(self.mode, m, **self.kw)
        lo, hi = clip_set(rlo, rhi)
        self.set_mu(lo, hi)
        eps = 1e-12
        self._al = dict(x_al_mode=self.mode, x_al_lo=lo, x_al_hi=hi, x_al_lw=float(np.log(hi / lo)), x_al_lw_raw=float(np.log(rhi / rlo)),
                        x_al_cov=bool(lo * (1 - eps) <= m <= hi * (1 + eps)), x_al_cov_raw=bool(rlo * (1 - eps) <= m <= rhi * (1 + eps)),
                        x_al_lerr=float(0.5 * (np.log(lo) + np.log(hi)) - np.log(m)),
                        x_al_mu_lo_eff=float(self.model.me_lo), x_al_mu_hi_eff=float(self.model.me_hi))

    def stats(self):
        s = super().stats()
        s.update(self._al)
        return s
