"""model_uq/f5_uq.py -- F5 with a STATE-DEPENDENT model-error bound instead of one global eps_a. SIMULATION / SAVED DATA ONLY. CPU.

ADDITIVE: nothing in sets/ is edited. `UQReachModel` is `reach.ReachModel` with the per-tick acceleration box eps_a replaced by a vector
eps_k (k = 0..H-1), computed from regime features of the model's own centre rollout (what a filter can compute without the hidden mu):
    lrho_m  log10( mean|g_in| / mean f )  over the next `kwin` ticks, f at the MID of the mu prior      speed |v_rel| at tick k
    edge    distance to the nearest tray edge at tick k      tilt arccos(c_z) at tick k      acc  mean |a_xy| over the window
and a fitted bound b(features) (a pickled model from experiment.py, see `load_eps_model`). `F5UQFilter` = F5SetFilter with that model.
Spec string for the bridge (see run_f5uq.py):  F5UQ:eps=<name>   |   F5UQ:const=0.08   (const reproduces F5 exactly; tested)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import numba as nb

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.tray_rep.sets.filter import F5SetFilter
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY

DT = RC.DT
EPS_DIR = K.ROOT.parent / "model_uq" / "eps_models"
FEAT_ORDER = ("lrho_m", "speed", "edge", "tilt", "acc")


@nb.njit(cache=True)
def _axis_gens_eps(Pa, ksig, poly, H, eps_k, eps_p0, damp, kpos, delay, collapse_k, slide):
    """== reach._axis_gens with the model-error box eps_a replaced by the per-tick vector eps_k (H,)."""
    m0 = poly.shape[0]
    M = m0 + 2 + H
    gp = np.zeros(M)
    gv = np.zeros(M)
    l00 = np.sqrt(max(Pa[0, 0], 1e-18))
    l10 = Pa[1, 0] / l00
    l11 = np.sqrt(max(Pa[1, 1] - l10 * l10, 1e-18))
    for j in range(m0):
        gp[j] = ksig * l00 * poly[j, 0]
        gv[j] = ksig * (l10 * poly[j, 0] + l11 * poly[j, 1])
    gp[m0] = eps_p0
    n = m0 + 1
    rp = np.zeros(H + 1)
    rv = np.zeros(H + 1)
    rp[0] = np.sum(np.abs(gp[:n]))
    rv[0] = np.sum(np.abs(gv[:n]))
    frozen = False
    hw = 0.0
    for k in range(H):
        if frozen:
            rp[k + 1] = hw
            continue
        d = damp if k >= delay else 0.0
        kq = kpos if k >= delay else 0.0
        for j in range(n):
            p = gp[j]
            v = gv[j]
            gp[j] = p + DT * v
            gv[j] = v - DT * (d * v + kq * p)
        gp[n] = eps_k[k] * 0.5 * DT * DT
        gv[n] = eps_k[k] * DT
        n += 1
        if collapse_k > 0 and k + 1 == collapse_k:
            vm = np.sum(np.abs(gv[:n]))
            hw = np.sum(np.abs(gp[:n])) + slide * vm * vm
            frozen = True
            rp[k + 1] = hw
            rv[k + 1] = 0.0
            continue
        rp[k + 1] = np.sum(np.abs(gp[:n]))
        rv[k + 1] = np.sum(np.abs(gv[:n]))
    return rp, rv


class EpsModel:
    """wraps a fitted bound: per-tick features (H, 5 in FEAT_ORDER) -> eps_k (H,). `cols` = the columns the fitted method uses."""

    def __init__(self, method, cols, kwin, cap, floor=0.0):
        self.method, self.cols, self.kwin, self.cap, self.floor = method, tuple(cols), int(kwin), float(cap), float(floor)
        self.idx = [FEAT_ORDER.index(c) for c in self.cols]

    def __call__(self, F):
        b = np.asarray(self.method.bound(F[:, self.idx]), np.float64)
        b = np.where(np.isfinite(b), b, self.cap)
        return np.minimum(np.maximum(b, self.floor), self.cap)


class ConstEps:
    def __init__(self, e):
        self.e, self.kwin = float(e), 1

    def __call__(self, F):
        return np.full(len(F), self.e)


def load_eps_model(name):
    if name.startswith("const"):
        return ConstEps(float(name[5:]))
    with open(EPS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


class UQReachModel(RC.ReachModel):
    def __init__(self, p, eps_model):
        super().__init__(p)
        self.eps_model = eps_model
        self.last_eps = None

    def tick_features(self, S_mid, C_mid, GN, FT, AT, kwin):
        """(H, 5) features at ticks 0..H-1 from the mid-mu centre rollout."""
        H = self.p.H
        cg, cf, ca = np.concatenate([[0], np.cumsum(GN)]), np.concatenate([[0], np.cumsum(FT)]), \
            np.concatenate([[0], np.cumsum(np.hypot(AT[:, 0], AT[:, 1]))])
        j = np.arange(H)
        e = np.minimum(j + kwin, H)
        n = (e - j).astype(np.float64)
        gm, fm, am = (cg[e] - cg[j]) / n, (cf[e] - cf[j]) / n, (ca[e] - ca[j]) / n
        lrho = np.log10(np.maximum(gm / np.maximum(fm, 1e-9), 1e-6))
        sp = np.hypot(S_mid[:H, 2], S_mid[:H, 3])
        edge = np.minimum(HX - np.abs(S_mid[:H, 0]), HY - np.abs(S_mid[:H, 1]))
        tilt = np.arccos(np.clip(C_mid[:H, 2], -1, 1))
        return np.stack([lrho, sp, edge, tilt, am], 1)

    def predict(self, x, P, c0, vt0, w0, u0, jac=True, d_u=None):
        p, H = self.p, self.p.H
        u0 = np.asarray(u0, np.float64)
        Sm, Cm = self.roll(x, c0, vt0, w0, u0, self.me_mid, 0)
        S_mid, C_mid = Sm.copy(), Cm.copy()
        GN0, FT0, AT0 = self._GN[0].copy(), self._FT[0].copy(), self._AT[0].copy()
        Sl, _ = self.roll(x, c0, vt0, w0, u0, self.me_lo, 1)
        Sh, _ = self.roll(x, c0, vt0, w0, u0, self.me_hi, 2)
        FTl, GNl = self._FT[1].copy(), self._GN[1].copy()
        pos = np.stack([S_mid[:, :2], Sl[:, :2], Sh[:, :2]])
        lo, hi = pos.min(0), pos.max(0)
        shift = 0.5 * (lo + hi) - S_mid[:, :2]
        band_hw = 0.5 * (hi - lo)
        eps_k = np.ascontiguousarray(self.eps_model(self.tick_features(S_mid, C_mid, GN0, FT0, AT0, self.eps_model.kwin)),
                                     dtype=np.float64)
        self.last_eps = eps_k
        collapse_k, slide = 0, 0.0
        if p.kd == 0.0 and p.kp == 0.0:
            gn = np.maximum.reduce([GN0, GNl, self._GN[2]])
            ok = gn <= p.gamma_stuck * FTl
            spd = np.maximum.reduce([np.hypot(S_mid[1:, 2], S_mid[1:, 3]), np.hypot(Sl[1:, 2], Sl[1:, 3]),
                                     np.hypot(Sh[1:, 2], Sh[1:, 3])])
            ok &= spd < 2e-3
            if ok[-1]:
                bad = np.flatnonzero(~ok)
                kk = (bad[-1] + 1) if len(bad) else 0
                kk = max(kk, 1)
                a_dec = float(np.min(FTl[kk:] - gn[kk:])) / (1.0 + RC.K_IN)
                if a_dec > 1e-6:
                    collapse_k, slide = kk + 1, 1.0 / (2.0 * a_dec)
        hw = np.zeros((H + 1, 2))
        for a in range(2):
            Pa = np.array([[P[a, a], P[a, a + 2]], [P[a + 2, a], P[a + 2, a + 2]]])
            rp, rv = _axis_gens_eps(Pa, p.k_sigma, self._poly, H, eps_k, p.eps_p0, p.kd, p.kp, p.fb_delay, collapse_k, slide)
            hw[:, a] = rp
        hw = hw + band_hw
        J = Jc = None
        if jac:
            d_u = np.array([0.05, 0.05, 0.05, 0.03, 0.03, 0.03]) if d_u is None else d_u
            J = np.zeros((H + 1, 2, 6))
            Jc = np.zeros((H + 1, 2, 6))
            for j in range(6):
                up = u0.copy()
                up[j] += d_u[j]
                Sp, Cp = self.roll(x, c0, vt0, w0, up, self.me_mid, 0)
                J[:, :, j] = (Sp[:, :2] - S_mid[:, :2]) / d_u[j]
                Jc[:, :, j] = (Cp[:, :2] - C_mid[:, :2]) / d_u[j]
        return RC.Prediction(centre=S_mid[:, :2] + shift, hw=hw, S_mid=S_mid, C=C_mid, J=J, Jc=Jc, u0=u0, k_stuck=collapse_k,
                             speed_end=float(np.hypot(S_mid[-1, 2], S_mid[-1, 3])),
                             info=dict(band_hw_max=float(band_hw.max()), collapse_k=collapse_k, eps_mean=float(eps_k.mean())))


class F5UQFilter(F5SetFilter):
    name = "F5UQ"

    def __init__(self, renv, eps="const0.08", const=None, **kw):
        super().__init__(renv, **kw)
        em = ConstEps(float(const)) if const is not None else load_eps_model(str(eps))
        self.model = UQReachModel(self.rp, em)
        self.eps_model = em

    def reset_stats(self):
        super().reset_stats()
        self._x["x_f5_eps_mean"] = 0.0

    def __call__(self, obs_hist, act_hist, u_nom, t):
        u = super().__call__(obs_hist, act_hist, u_nom, t)
        le = self.model.last_eps
        if le is not None:
            self._x["x_f5_eps_mean"] += float(le.mean())
        return u

    def stats(self):
        s = super().stats()
        s["x_f5_eps_mean"] = s["x_f5_eps_mean"] / max(s["x_f5_n"], 1)
        return s
