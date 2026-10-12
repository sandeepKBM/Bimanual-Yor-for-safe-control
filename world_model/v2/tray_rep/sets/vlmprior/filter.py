"""sets/vlmprior/filter.py -- F5P: F5 (analytic set-reachability filter) whose rolling-resistance set comes from a synthetic VLM prior and/or the
online belief. SIMULATION / SAVED DATA ONLY. CPU. No hardware, no network. Experiment B.

F5P IS F5 (sets/filter.py F5SetFilter, unchanged). Only the resistance set handed to F5 by set_mu(mu_lo, mu_hi) differs between arms, and it is
refreshed every `upd` ticks (default 5; at tick 0 too). Work in l = ln(mu_roll). The prior (m, sigma) comes from prior.draw_prior (synthetic).
The online belief (online_mu.OnlineMu, 40-node grid, tempered by lam) is fed once per tick from the obs stream; it never sees the hidden mu.
All sets are clipped to the physical range [1e-4, 3e-2].

  arm=point   Phys2Real-style inverse-variance fusion of N(m, sigma^2) and the online Gaussian approx (mean, sd in l) -> (l_f, sd_f); set = [e^l_f, e^l_f]
  arm=band    [l_f - 2 sd_f, l_f + 2 sd_f]
  arm=conf    split-conformal prior interval [m - q sigma, m + q sigma] (q from conformal_q.json, VAL only, same rho)  INTERSECTED with the online
              95% credible interval; empty intersection -> the online 95% interval
  arm=unif    population range [5e-4, 6e-3]  INTERSECTED with the online 95% credible interval
  arm=oracle  [mu_true, mu_true]   (reads ctx['hid']; the only arm allowed to)
  arm=calib   (calibration, not an experimental arm) plain F5 with the population range; logs the raw cumulative log-likelihood of banks of
              online estimators at checkpoint ticks (calibrate_online.py).

Spec string:  "F5P:arm=conf,rho=0.2,pseed=0"   keys: arm, rho (default 0), pseed (0), upd (5), lam (default: online_calib.json for the setup, else 1),
              sa (online sigma_a override), online (1; 0 = the online interval is the whole population range and the belief is uninformative:
              arm=unif,online=0 is record-identical to plain F5), plus any F5 key (H, k_sigma, eps_scale, kd, ... ) passed through.
Record extras (stats): x_p_m, x_p_sigma, x_p_conf (prior), x_p_lo/hi_t{0,50,100,200,end} (set bounds, mu units, null = tick not reached), x_p_cov_t* (true mu inside the
set; computed from the hidden mu for LOGGING only), x_p_lerr_t* (centre of the set minus l_true), x_p_logw_mean (mean over ticks of ln(hi/lo)),
x_p_ms (estimator + set update time), plus F5's x_f5_*.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.sets.filter import F5SetFilter
from world_model.v2.tray_rep.sets.vlmprior import conformal as CF
from world_model.v2.tray_rep.sets.vlmprior import online_mu as OM
from world_model.v2.tray_rep.sets.vlmprior import prior as PR

L_PHYS = (float(np.log(1e-4)), float(np.log(3e-2)))          # physical range of mu_roll
L_POP = (float(np.log(OM.MU_LO)), float(np.log(OM.MU_HI)))   # population range (env.py prior support)
CHECK_TICKS = {"t0": 0, "t1s": 50, "t2s": 100, "t4s": 200}   # 50 Hz
ARMS = ("point", "band", "conf", "unif", "oracle", "calib")
CALIB_FILE = K.ROOT / "vlmprior" / "online_calib.json"


def load_online_defaults(setup):
    """(lam, sigma_a) chosen on VAL for this setup (online_calib.json), else (1.0, None)."""
    try:
        d = json.loads(CALIB_FILE.read_text())["chosen"][setup]
        return float(d["lam"]), (None if d.get("sigma_a") is None else float(d["sigma_a"]))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return 1.0, None


def fuse(m, sigma, mo, so):
    """Phys2Real-style inverse-variance fusion of N(m, sigma^2) and N(mo, so^2) -> (mean, sd)."""
    pa, pb = 1.0 / sigma ** 2, 1.0 / so ** 2
    return (m * pa + mo * pb) / (pa + pb), float((pa + pb) ** -0.5)


def clip_phys(lo, hi):
    lo, hi = float(np.clip(lo, *L_PHYS)), float(np.clip(hi, *L_PHYS))
    return (lo, hi) if lo <= hi else (hi, hi)


def to_mu(l):
    """exp with the population edges snapped to the exact constants (so an uninformative unif arm hands F5 bit-identical bounds)."""
    if abs(l - L_POP[0]) < 1e-12:
        return OM.MU_LO
    if abs(l - L_POP[1]) < 1e-12:
        return OM.MU_HI
    return float(np.exp(l))


class F5PFilter(F5SetFilter):
    name = "F5P"

    def __init__(self, renv, arm="point", rho=0.0, pseed=0, upd=5, lam=None, sa=None, online=1, qfile=None, sas=None, **f5kw):
        super().__init__(renv, **f5kw)
        if arm not in ARMS:
            raise ValueError(f"F5P arm={arm!r} not in {ARMS}")
        self.arm, self.rho, self.pseed, self.upd = arm, float(rho), int(float(pseed)), max(int(float(upd)), 1)
        self.online_on = bool(float(online))
        self.nf_est = renv.tracker is None
        lam_d, sa_d = load_online_defaults("piper" if self.nf_est else "piper_trk")
        self.lam = lam_d if lam is None else float(lam)
        self.sa = sa_d if sa is None else float(sa)
        okw = dict(noise_free=self.nf_est, lam=self.lam, sigma_a=self.sa)
        if not self.nf_est:
            okw.update(sigma_meas=renv.tracker.noise, lat=renv.tracker.lat)
        self._okw = okw
        self.om = OM.OnlineMu(**okw)
        self.bank = []
        if arm == "calib":                                   # a bank of estimators over sigma_a, raw log-likelihood logged at checkpoints
            sl = [float(x) for x in str(sas).split("x")] if sas is not None else [0.03, 0.1, 0.3]
            self.bank = [OM.OnlineMu(**{**okw, "lam": 1.0, "sigma_a": s}) for s in sl]
            self.bank_sa = sl
        self.q = CF.load_q(self.rho, qfile or CF.QFILE) if arm == "conf" else None
        self._pstats_reset()

    # ------------------------------------------------------------------ per carry
    def _pstats_reset(self):
        self._p = dict(x_p_m=None, x_p_sigma=None, x_p_conf=None, x_p_ms=0.0, x_p_logw_sum=0.0, x_p_n=0, x_p_q=self.q, x_p_lam=self.lam)
        self._ck = {}                                         # name -> (lo_mu, hi_mu) in force at that tick
        self._cur = (OM.MU_LO, OM.MU_HI)
        self._ll_ck = {}
        self._l_true = None
        self._hid = None
        self._pr = None

    def begin(self, ctx):
        super().begin(ctx)
        self._pstats_reset()
        self.om = OM.OnlineMu(**self._okw)
        for b in self.bank:
            b.reset()
        self._hid = float(ctx["hid"])             # hidden truth: read by arm=oracle, by the prior SIMULATOR and for LOGGING only
        self._l_true = float(np.log(self._hid))
        m, sg, cf = PR.draw_prior(self._l_true, ctx["seed"], self.rho, self.pseed)
        self._pr = (m, sg)
        self._p.update(x_p_m=m, x_p_sigma=sg, x_p_conf=bool(cf))
        self._apply(*self._cur)

    def _apply(self, lo_mu, hi_mu):
        self._cur = (float(lo_mu), float(hi_mu))
        self.set_mu(lo_mu, hi_mu)

    # ------------------------------------------------------------------ the arms
    def _online(self):
        """(mean, sd of l, 95% interval in l) of the online belief; uninformative (population) if online=0."""
        if not self.online_on:
            w = OM.posterior(np.zeros(OM.N_NODES), 1.0)
            return (*OM.moments(w), L_POP)
        return (*self.om.mean_sd_log(), self.om.interval_log(0.95))

    def _set_l(self):
        """(lo, hi) in l for the current tick."""
        arm = self.arm
        if arm == "oracle":
            return self._l_true, self._l_true
        if arm in ("unif", "calib"):
            mo, so, (olo, ohi) = self._online()
            return (L_POP[0], L_POP[1]) if arm == "calib" else (max(L_POP[0], olo), min(L_POP[1], ohi))
        m, sg = self._pr
        mo, so, (olo, ohi) = self._online()
        if arm in ("point", "band"):
            lf, sdf = fuse(m, sg, mo, so)
            return (lf, lf) if arm == "point" else (lf - 2.0 * sdf, lf + 2.0 * sdf)
        lo, hi = m - self.q * sg, m + self.q * sg                    # conf
        lo, hi = max(lo, olo), min(hi, ohi)
        return (lo, hi) if lo <= hi else (olo, ohi)

    # ------------------------------------------------------------------ main
    def __call__(self, obs_hist, act_hist, u_nom, t):
        t0 = time.perf_counter()
        if self.arm in ("calib",):
            for b in self.bank:
                b.feed(obs_hist[-1])
        elif self.online_on and self.arm != "oracle":
            self.om.feed(obs_hist[-1], act_hist[-1] if len(act_hist) else None)
        if self.arm != "calib" and t % self.upd == 0:
            lo, hi = clip_phys(*self._set_l())
            self._apply(to_mu(lo), to_mu(hi))
        for name, tk in CHECK_TICKS.items():
            if t == tk:
                self._ck[name] = self._cur
                if self.arm == "calib":
                    self._ll_ck[name] = [b.loglik().tolist() for b in self.bank]
        self._p["x_p_logw_sum"] += float(np.log(self._cur[1] / self._cur[0]))
        self._p["x_p_n"] += 1
        self._p["x_p_ms"] += 1e3 * (time.perf_counter() - t0)
        return super().__call__(obs_hist, act_hist, u_nom, t)

    def stats(self):
        s = super().stats()
        p = self._p
        n = max(p["x_p_n"], 1)
        s.update(x_p_m=p["x_p_m"], x_p_sigma=p["x_p_sigma"], x_p_conf=p["x_p_conf"], x_p_q=p["x_p_q"], x_p_lam=p["x_p_lam"],
                 x_p_ms=p["x_p_ms"], x_p_ms_mean=p["x_p_ms"] / n, x_p_logw_mean=p["x_p_logw_sum"] / n, x_p_n=p["x_p_n"])
        ck = dict(self._ck)
        ck["end"] = self._cur if p["x_p_n"] else None
        lt = self._l_true
        for name in list(CHECK_TICKS) + ["end"]:
            b = ck.get(name)
            s[f"x_p_lo_{name}"] = None if b is None else b[0]
            s[f"x_p_hi_{name}"] = None if b is None else b[1]
            # LOGGING ONLY: is the true mu inside the set (small relative tolerance for the point arm's float round trip)
            s[f"x_p_cov_{name}"] = None if b is None else bool(b[0] * (1 - 1e-9) <= self._hid <= b[1] * (1 + 1e-9))
            s[f"x_p_lerr_{name}"] = None if b is None else float(0.5 * np.log(b[0] * b[1]) - lt)
        if self.arm == "calib":
            ll = dict(self._ll_ck)
            if p["x_p_n"]:
                ll["end"] = [b.loglik().tolist() for b in self.bank]
            s["x_p_ll"] = ll
            s["x_p_sas"] = self.bank_sa
            s["x_p_tend"] = p["x_p_n"] - 1
        return s
