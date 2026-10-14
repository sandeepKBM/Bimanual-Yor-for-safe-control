"""sets/reach.py -- analytic reachable sets of the rolling ball on the tray. SIMULATION / SAVED DATA ONLY. CPU, numba, no GPU.

Model (the quasi-static "5/7" ball of testbeds/b_tray_ball/analytic.py, QS, written in the tray frame; same physics as
boed/model.integrate, which was validated against MuJoCo):
    state s = [x, y, vx, vy]  (ball centre and velocity RELATIVE to the tray surface frame)
    (1 + k) a = g_in - f * v_hat           while moving         k = 0.4 (solid ball)
    a = (|g_in| - f) g_in_hat / (1 + k)    if at rest and |g_in| > f,   a = 0 if |g_in| <= f   (sticks)
    g_in = (e_x . (g - a_tray), e_y . (g - a_tray))     tray-frame components of the effective gravity
    f = (mu_eff / r) * N/m,   N/m = c . (a_tray + g z)  rolling-resistance threshold [m/s^2];  mu_eff = calibrated map of the MuJoCo mu_roll.
The tray frame is the minimal rotation taking z to the tray axis c (yaw ignored, as in bridge/filters.rot_from_axis).
50 Hz tick, 4 sub-steps (the same split as MuJoCo's physics rate).

Reachable set (what this file builds):
    S_k = centre_k  +  Z_k      k = 0 .. H  ticks ahead, in the TRAY frame, position part (x, y)
  centre_k   the nonlinear rollout at the MIDDLE rolling resistance, under  "tick 0: command u0; ticks >= 1: the BACKUP law".
  Z_k        a zonotope (an axis-aligned box in x, y; the two axes are independent) that adds
               1. the initial uncertainty: the tracker estimate's covariance (4 x 4, Kalman) at k_sigma, propagated through the
                  linearised error dynamics  delta_s' = A delta_s  (frictionless double integrator; with ball feedback in the backup,
                  a damped version after `fb_delay` ticks);
               2. the model error: an acceleration box +-eps_a per axis every tick (calibrated on VAL replays, see README);
               3. the rolling-resistance set: the hull of the rollouts at mu_lo, mu_mid, mu_hi (a bounded disturbance whose sign
                  follows the velocity, taken exactly by simulation instead of by a box);
               4. the stopping inflation: if the backup brings the ball to a certified stick (|g_in| <= gamma * f_lo from some tick
                  to the end) the velocity error is converted into a bounded extra slide  v^2 / (2 a_dec)  and then removed.
  Dependence on u0: only the CENTRE depends on u0. Its Jacobian J_k = d centre_k / d u0 (6 columns) is a forward difference of the
  nonlinear rollout (so saturations and sticking are in it). The rest of S_k is held at the linearisation point.

BACKUP law ("level the tray and brake", within the Piper authority and slew):
    a_b   = clip_norm(-v_tray / tau_b, a_brk)        (a_brk = a_brake_frac * a_max)
    c_des = (a_b + gin_des) / g,   gin_des = -(1 + k)(kd v_ball + kp x_ball)         [kd = kp = 0: pure level-and-brake with tilt compensation]
    w_cmd = kappa_t * (c_des - c)  (rotated to rates), clipped to w_max, then slew-limited by the arm (alpha) exactly as the env does.
With kd = kp = 0 the backup depends on the tray state only, so the tray part of the prediction is exact; with kd > 0 it also uses the
ball estimate (stronger: it brakes a fast ball by tilting against it) and the perturbation set is damped heuristically.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import numba as nb

from world_model.v2.tray_rep import common as K          # noqa: F401  thread pins, sys.path
from world_model.v2.tray_rep.sets.zono import Zonotope, ellipsoid_outer

G = 9.81
DT = 0.02
SUB = 4
K_IN = 0.4
R_BALL = 0.03


# ====================================================================================== calibrated rolling resistance
_MU_EFF_CACHE = {}


def load_mu_eff(path=None):
    """mu_roll (MuJoCo parameter) -> effective Coulomb rolling resistance the ball shows (boed/calib.json). Returns f(mu).
    path=None: the file named by env BIMANUAL_MU_EFF_FILE if set (rebaseline_v2 runs), else boed/calib.json (the original behaviour).
    A path may be given as a short name: "v2" = <data root>/calib_v2/mu_eff_calib.json (map refit on the fixed pin_cup physics)."""
    import json
    import os
    from pathlib import Path
    if not path:
        path = os.environ.get("BIMANUAL_MU_EFF_FILE", "")
    if path == "v2":
        from world_model.v2.tray_rep import common as _K
        path = str(_K.ROOT / "calib_v2" / "mu_eff_calib.json")
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "boed" / "calib.json"
    key = str(p)
    if key in _MU_EFF_CACHE:
        return _MU_EFF_CACHE[key]
    if not p.exists():
        if path:
            raise FileNotFoundError(p)
        fn = lambda mu: float(mu)
    else:
        d = json.loads(p.read_text())
        lm, me = np.log(np.array(d["mu"])), np.array(d["mu_eff"])
        fn = lambda mu: float(np.interp(np.log(mu), lm, me))
    _MU_EFF_CACHE[key] = fn
    return fn


MU_EFF = load_mu_eff()


# ====================================================================================== parameters
PAR_NAMES = ("a_max", "w_max", "alpha", "tau_b", "a_brake_frac", "kd", "kp", "kappa_t", "tilt_max", "k_in", "r_ball")


@dataclass
class ReachParams:
    H: int = 60                       # horizon [ticks] = 1.2 s. See README: the backup needs ~0.3 s to level, a ball at the
    #                                   fastest carry speeds is stopped by friction or tilt in <= ~0.6-0.9 s.
    mu_lo: float = 5e-4               # MuJoCo mu_roll set (the prior range of env.py); mapped through the calibration
    mu_hi: float = 6e-3
    k_sigma: float = 2.5              # initial set = estimate +- k_sigma Kalman sigma (per axis ellipse -> zonotope)
    eps_a: float = 0.08               # model error: acceleration box [m/s^2] per axis (calibrated on VAL)
    eps_p0: float = 0.0015            # model error: extra initial position half-width [m] (frame yaw, sensor floor)
    gamma_stuck: float = 0.8
    calib_file: str = ""              # '' = the default map (boed/calib.json, or env BIMANUAL_MU_EFF_FILE); 'v2' or a path = another map
    calib: bool = True                # False: raw mu_roll as the resistance (ablation: MuJoCo shows only 0.6-0.87 of it, boed/calib.json)
    # backup law
    tau_b: float = 0.5
    a_brake_frac: float = 0.5
    kd: float = 3.0                   # ball feedback of the backup (1/s): brakes the ball by tilting against its velocity
    kp: float = 2.0                   # ... and pulls it toward the tray centre (1/s^2)
    kappa_t: float = 6.0
    tilt_max: float = 0.2
    fb_delay: int = 5                 # ticks before the backup's ball feedback is assumed to act on the perturbations (tilt build-up: slew 10 rad/s^2)
    # authority (set from the env)
    a_max: float = 1.5
    w_max: float = 1.0
    alpha: float = 10.0

    def par(self):
        return np.array([self.a_max, self.w_max, self.alpha, self.tau_b, self.a_brake_frac, self.kd, self.kp, self.kappa_t,
                         self.tilt_max, K_IN, R_BALL], np.float64)


# ====================================================================================== numba kernels
@nb.njit(cache=True, inline="always")
def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


@nb.njit(cache=True)
def _frame(c):
    """tray-frame x and y axes (world components) for the minimal rotation z -> c."""
    cz1 = 1.0 + c[2]
    ex = np.array([1.0 - c[0] * c[0] / cz1, -c[0] * c[1] / cz1, -c[0]])
    ey = np.array([-c[0] * c[1] / cz1, 1.0 - c[1] * c[1] / cz1, -c[1]])
    return ex, ey


@nb.njit(cache=True)
def gin_of(c, a):
    """in-plane effective gravity (2,) and N/m for tray axis c and tray acceleration a (3,)."""
    ex, ey = _frame(c)
    gx = -ex[0] * a[0] - ex[1] * a[1] - ex[2] * (G + a[2])
    gy = -ey[0] * a[0] - ey[1] * a[1] - ey[2] * (G + a[2])
    nm = c[0] * a[0] + c[1] * a[1] + c[2] * (G + a[2])
    return gx, gy, nm


@nb.njit(cache=True)
def ball_step(s, ca, cb, a, mu_eff, k_in, r):
    """one 50 Hz tick of the QS ball (4 sub-steps, axis interpolated from ca to cb). Returns s2 (4,), and the mean in-plane
    effective gravity magnitude and the mean threshold f of the tick (for the stuck test)."""
    h = DT / SUB
    x, y, vx, vy = s[0], s[1], s[2], s[3]
    gsum = 0.0
    fsum = 0.0
    for i in range(SUB):
        fr = (i + 0.5) / SUB
        c = ca * (1.0 - fr) + cb * fr
        c = c / np.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
        gx, gy, nm = gin_of(c, a)
        if nm < 0.0:
            nm = 0.0
        f = (mu_eff / r) * nm
        gn = np.sqrt(gx * gx + gy * gy)
        gsum += gn
        fsum += f
        sp = np.sqrt(vx * vx + vy * vy)
        if sp > 1e-4:
            ax = (gx - f * vx / sp) / (1.0 + k_in)
            ay = (gy - f * vy / sp) / (1.0 + k_in)
        elif gn > f:
            ax = (gn - f) * gx / gn / (1.0 + k_in)
            ay = (gn - f) * gy / gn / (1.0 + k_in)
        else:
            ax = 0.0
            ay = 0.0
        vnx = vx + ax * h
        vny = vy + ay * h
        if sp > 1e-4 and (vnx * vx + vny * vy) <= 0.0:
            vnx = 0.0
            vny = 0.0
        x += 0.5 * (vx + vnx) * h
        y += 0.5 * (vy + vny) * h
        vx, vy = vnx, vny
    out = np.empty(4)
    out[0], out[1], out[2], out[3] = x, y, vx, vy
    return out, gsum / SUB, fsum / SUB


@nb.njit(cache=True)
def tray_step(c, vt, w, a, wcmd, alpha):
    """the env's tray kinematics for one tick: slew-limit the tilt rate, 4 sub-steps of c += h (w x c), v += a dt."""
    lim = alpha * DT
    w2 = np.empty(3)
    for i in range(3):
        w2[i] = w[i] + _clip(wcmd[i] - w[i], -lim, lim)
    h = DT / SUB
    cc = c.copy()
    for _ in range(SUB):
        cc = cc + h * np.cross(w2, cc)
        cc = cc / np.sqrt(cc[0] * cc[0] + cc[1] * cc[1] + cc[2] * cc[2])
    return cc, vt + a * DT, w2


@nb.njit(cache=True)
def backup_cmd(c, vt, s, par):
    """the BACKUP law. Returns (a (3,), wcmd (3,))."""
    a_max, w_max, tau_b, afrac, kd, kp, kap, tmax, k_in = par[0], par[1], par[3], par[4], par[5], par[6], par[7], par[8], par[9]
    cap = afrac * a_max
    ab = np.empty(3)
    ab[0] = -vt[0] / tau_b
    ab[1] = -vt[1] / tau_b
    na = np.sqrt(ab[0] * ab[0] + ab[1] * ab[1])
    if na > cap:
        ab[0] *= cap / na
        ab[1] *= cap / na
    ab[2] = _clip(-vt[2] / tau_b, -cap, cap)
    cdx = (ab[0] - (1.0 + k_in) * (kd * s[2] + kp * s[0])) / G
    cdy = (ab[1] - (1.0 + k_in) * (kd * s[3] + kp * s[1])) / G
    nd = np.sqrt(cdx * cdx + cdy * cdy)
    ts = np.sin(tmax)
    if nd > ts:
        cdx *= ts / nd
        cdy *= ts / nd
    w = np.zeros(3)
    w[1] = _clip(kap * (cdx - c[0]), -w_max, w_max)
    w[0] = _clip(-kap * (cdy - c[1]), -w_max, w_max)
    return ab, w


@nb.njit(cache=True)
def rollout(s0, c0, vt0, w0, u0, H, mu_eff, par, S, C, GN, FT, AT):
    """centre rollout: tick 0 under u0 = (a, wcmd), ticks 1..H-1 under the backup law. Fills S (H+1,4), C (H+1,3) tray axis,
    GN (H,) |g_in|, FT (H,) threshold f, AT (H,3) tray accelerations."""
    k_in, r, alpha = par[9], par[10], par[2]
    s = s0.copy()
    c = c0.copy()
    vt = vt0.copy()
    w = w0.copy()
    S[0] = s
    C[0] = c
    for k in range(H):
        if k == 0:
            a = u0[:3].copy()
            wc = u0[3:].copy()
        else:
            a, wc = backup_cmd(c, vt, s, par)
        c2, vt2, w2 = tray_step(c, vt, w, a, wc, alpha)
        s, g, f = ball_step(s, c, c2, a, mu_eff, k_in, r)
        c, vt, w = c2, vt2, w2
        S[k + 1] = s
        C[k + 1] = c
        GN[k] = g
        FT[k] = f
        AT[k] = a


@nb.njit(cache=True)
def _axis_gens(Pa, ksig, poly, H, eps_a, eps_p0, damp, kpos, delay, collapse_k, slide):
    """propagate the generator set of ONE axis (pos, vel) for H ticks -> half-widths rp (H+1,), rv (H+1,).
    Pa (2,2) covariance of (pos, vel); poly (m0,2) generators of the unit-circle outer polygon.
    Error dynamics per tick: p += dt v;  v -= dt (damp v + kpos p) once k >= delay (the backup's ball feedback has built its tilt); plus the model-error box.
    At tick `collapse_k` (> 0; 0 = never) the velocity error is turned into the bounded slide  slide * vm^2  and removed."""
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
        gp[n] = eps_a * 0.5 * DT * DT
        gv[n] = eps_a * DT
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


# ====================================================================================== python layer
_POLY = {}


def _poly(m=8):
    """generators (m, 2) of the zonotope that circumscribes the unit disc (2m-gon)."""
    if m not in _POLY:
        _POLY[m] = np.ascontiguousarray(ellipsoid_outer(np.zeros(2), np.eye(2), 1.0, m).G.T)
    return _POLY[m]


@dataclass
class Prediction:
    """reachable-set prediction for ticks 0..H in the TRAY frame."""
    centre: np.ndarray            # (H+1, 2) centre of the set (mid-rho rollout + band centre shift)
    hw: np.ndarray                # (H+1, 2) position half-widths of the zonotope around `centre` (initial + model error + rho band)
    S_mid: np.ndarray             # (H+1, 4) the mid-rho rollout (x, y, vx, vy)
    C: np.ndarray                 # (H+1, 3) tray axis along the plan
    J: np.ndarray | None          # (H+1, 2, 6) d centre / d u0
    Jc: np.ndarray | None         # (H+1, 2, 6) d (axis x, y) / d u0
    u0: np.ndarray
    k_stuck: int = 0
    speed_end: float = 0.0
    info: dict = field(default_factory=dict)

    def zonotopes(self):
        """batch Zonotope (H+1) in R^2: centre_k + diag(hw_k)."""
        H1 = len(self.centre)
        Gm = np.zeros((H1, 2, 2))
        Gm[:, 0, 0], Gm[:, 1, 1] = self.hw[:, 0], self.hw[:, 1]
        return Zonotope(self.centre, Gm)


class ReachModel:
    def __init__(self, p: ReachParams | None = None):
        self.p = p or ReachParams()
        self.par = self.p.par()
        self._map = (load_mu_eff(self.p.calib_file) if self.p.calib_file else MU_EFF) if self.p.calib else float
        self.me_lo, self.me_hi = self._map(self.p.mu_lo), self._map(self.p.mu_hi)
        self.me_mid = self._map(float(np.sqrt(self.p.mu_lo * self.p.mu_hi)))
        H = self.p.H
        self._S = [np.zeros((H + 1, 4)) for _ in range(3)]
        self._C = [np.zeros((H + 1, 3)) for _ in range(3)]
        self._GN = [np.zeros(H) for _ in range(3)]
        self._FT = [np.zeros(H) for _ in range(3)]
        self._AT = [np.zeros((H, 3)) for _ in range(3)]
        self._poly = _poly(8)

    def set_mu(self, mu_lo, mu_hi):
        """narrow (or widen) the rolling-resistance set, e.g. after a probe (BOED): mu_roll in [mu_lo, mu_hi]."""
        self.p.mu_lo, self.p.mu_hi = float(mu_lo), float(mu_hi)
        self.me_lo, self.me_hi = self._map(mu_lo), self._map(mu_hi)
        self.me_mid = self._map(float(np.sqrt(mu_lo * mu_hi)))

    def roll(self, s0, c0, vt0, w0, u0, mu_eff, idx=0):
        H = self.p.H
        rollout(np.asarray(s0, np.float64), np.asarray(c0, np.float64), np.asarray(vt0, np.float64),
                np.asarray(w0, np.float64), np.asarray(u0, np.float64), H, float(mu_eff), self.par,
                self._S[idx], self._C[idx], self._GN[idx], self._FT[idx], self._AT[idx])
        return self._S[idx], self._C[idx]

    def predict(self, x, P, c0, vt0, w0, u0, jac=True, d_u=None):
        """x (4,) estimate [x, y, vx, vy] in the tray frame, P (4, 4) its covariance; tray state c0 (axis), vt0 (tray velocity),
        w0 (realised tilt rate = slew state); u0 (6,) first-tick command (w part = commanded rate)."""
        p, H = self.p, self.p.H
        u0 = np.asarray(u0, np.float64)
        Sm, Cm = self.roll(x, c0, vt0, w0, u0, self.me_mid, 0)
        S_mid, C_mid = Sm.copy(), Cm.copy()
        GN0, FT0 = self._GN[0].copy(), self._FT[0].copy()
        Sl, _ = self.roll(x, c0, vt0, w0, u0, self.me_lo, 1)
        Sh, _ = self.roll(x, c0, vt0, w0, u0, self.me_hi, 2)
        FTl, GNl = self._FT[1].copy(), self._GN[1].copy()
        # rho band: hull of the three rollouts (position); centre shift and half-width
        pos = np.stack([S_mid[:, :2], Sl[:, :2], Sh[:, :2]])
        lo, hi = pos.min(0), pos.max(0)
        shift = 0.5 * (lo + hi) - S_mid[:, :2]
        band_hw = 0.5 * (hi - lo)
        # stopping certificate (only without ball feedback): earliest k such that for every j >= k the lowest-resistance ball
        # is held by friction (|g_in| <= gamma f_lo), and all three centres are at rest
        collapse_k, slide = 0, 0.0
        if p.kd == 0.0 and p.kp == 0.0:
            gn = np.maximum.reduce([GN0, GNl, self._GN[2]])
            ok = gn <= p.gamma_stuck * FTl
            spd = np.maximum.reduce([np.hypot(S_mid[1:, 2], S_mid[1:, 3]), np.hypot(Sl[1:, 2], Sl[1:, 3]),
                                     np.hypot(Sh[1:, 2], Sh[1:, 3])])
            ok &= spd < 2e-3
            if ok[-1]:
                bad = np.flatnonzero(~ok)
                kk = (bad[-1] + 1) if len(bad) else 0           # first index j such that ok[j:] all True
                kk = max(kk, 1)
                a_dec = float(np.min(FTl[kk:] - gn[kk:])) / (1.0 + K_IN)
                if a_dec > 1e-6:
                    collapse_k, slide = kk + 1, 1.0 / (2.0 * a_dec)       # set index of the first stuck state is kk+1... (state after tick kk)
        hw = np.zeros((H + 1, 2))
        damp = p.kd
        for a in range(2):
            Pa = np.array([[P[a, a], P[a, a + 2]], [P[a + 2, a], P[a + 2, a + 2]]])
            rp, rv = _axis_gens(Pa, p.k_sigma, self._poly, H, p.eps_a, p.eps_p0, damp, p.kp, p.fb_delay, collapse_k, slide)
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
        return Prediction(centre=S_mid[:, :2] + shift, hw=hw, S_mid=S_mid, C=C_mid, J=J, Jc=Jc, u0=u0, k_stuck=collapse_k,
                          speed_end=float(np.hypot(S_mid[-1, 2], S_mid[-1, 3])),
                          info=dict(band_hw_max=float(band_hw.max()), collapse_k=collapse_k))
