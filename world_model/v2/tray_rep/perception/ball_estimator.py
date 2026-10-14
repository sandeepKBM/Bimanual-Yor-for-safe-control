"""ball_estimator.py -- classical ball tracker for the tray testbed: point cloud -> sphere fit -> constant-velocity Kalman filter.
SIMULATION / SAVED DATA ONLY. Pure numpy, CPU. No learned parts. The estimator never reads geometry ids.

Pipeline
  1. `select_points`: put the cloud in the TRAY frame (the tray pose is known from the joint encoders), keep points above the
     tray plane and inside the footprint (grown by the ball radius so a ball at the edge is still seen).
     "Above" = z > 2 mm + 3 sigma_z * |ray z-component|: the tray's own noisy points must not leak in.
  2. `fit_sphere`: least squares for the centre with the KNOWN radius (Gauss-Newton / Levenberg-Marquardt, weighted),
     started from the median point pushed away from the camera, then one trim pass on residuals. A cap seen from one side is
     biased toward the camera if you take its centroid (about 14 mm at 0.85 m); the fit with the known radius removes that.
     Optional `fix_z`: the ball rests on the tray, so z = radius; this helps when only a few points are visible.
  3. `BallKF`: constant-velocity Kalman filter in the tray frame (x, y, vx, vy).
       * measurement noise adapts to quality (see below)
       * delayed measurements: a frame captured at tick t_c arrives at tick t_c + L. The filter keeps a short history of its
         states, applies the update at t_c and re-predicts to now (so latency costs accuracy only through the model).
       * track loss: with no good measurement the filter keeps predicting and its covariance grows; `since` counts ticks since
         the last good measurement; a loss EVENT is counted when `since` first passes T_LOSS; a measurement after a very long
         gap re-initialises the track.
Payload (10 floats) from one frame:  [valid, x, y, z, n_points, rms, Rxx, Rxy, Ryy, n_candidates]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_model.v2.tray_rep import common  # noqa: F401
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY, DT_WM

RADIUS = 0.03
PAYLOAD_LEN = 10
I_VALID, I_X, I_N, I_RMS, I_RXX, I_RXY, I_RYY, I_CAND = 0, 1, 4, 5, 6, 7, 8, 9


@dataclass
class EstCfg:
    radius: float = RADIUS
    z_above: float = 0.002          # keep points more than this above the tray plane [m]
    k_noise: float = 3.0            # ... plus k_noise * sigma_z * |ray_z| (so noisy tray points do not leak in)
    foot_margin: float = 0.02       # footprint grown by radius + this
    z_top: float = 0.05             # ... and points higher than 2*radius + z_top are dropped
    n_min: int = 6                  # fewer inlier points: no measurement
    rms_max: float = 0.006          # fit rms above this [m]: no measurement
    fix_z: bool = False             # z = radius (ball on the tray)
    trim_k: float = 3.0             # trim residuals above max(trim_floor, trim_k * rms)
    trim_floor: float = 0.005
    n_plane: int = 30               # fewer inliers than this: refit with z = radius (0 = never)
    s_floor: float = 0.002          # per-point residual scale floor for the covariance [m]
    sig_floor: float = 0.001        # measurement sigma floor [m]
    cap_sig8: float = 0.025         # extra sigma for small caps [m]: cap_sig8 * (8 / n_points)^3 (25 mm at 8 points, 1 mm at 25)
    max_shift: float = 0.06         # fit centre farther than this from the start guess: reject
    gate_r: float = 1.7             # coarse gate: points farther than gate_r * radius from the median point are dropped (a visible cap spans <= 1.42 r)


def sigma_z_default(z):
    """generic z noise used for the 'above the plane' gate when the caller gives no sensor model (3 mm at 1 m)."""
    return 0.003 * np.asarray(z, float) ** 2


def select_points(pts_t, origin_t, cfg: EstCfg, sigma_z=None, z_depth=None):
    """pts_t (n,3) in the tray frame; origin_t (3,) camera centre in the tray frame. -> index array of kept points."""
    if len(pts_t) == 0:
        return np.zeros(0, np.int64)
    r = cfg.radius
    d = pts_t - origin_t
    rng = np.linalg.norm(d, axis=1)
    uz = np.abs(d[:, 2]) / np.maximum(rng, 1e-9)
    if sigma_z is None:
        sz = sigma_z_default(rng)
    else:
        sz = np.asarray(sigma_z(rng if z_depth is None else z_depth))
    thr = cfg.z_above + cfg.k_noise * sz * uz
    z = pts_t[:, 2]
    keep = ((z > thr) & (z < 2 * r + cfg.z_top) & (np.abs(pts_t[:, 0]) < HX + r + cfg.foot_margin)
            & (np.abs(pts_t[:, 1]) < HY + r + cfg.foot_margin))
    return np.flatnonzero(keep)


def _gn(P, w, r, c, fix_z, iters=12):
    """weighted LM for min sum w^2 (|p - c| - r)^2. Returns c, residuals."""
    lam = 1e-4
    ax = [0, 1] if fix_z else [0, 1, 2]
    if fix_z:
        c = c.copy(); c[2] = r
    prev = None
    for _ in range(iters):
        dv = c - P
        n = np.linalg.norm(dv, axis=1)
        n = np.maximum(n, 1e-9)
        res = (n - r) * w
        J = (dv / n[:, None])[:, ax] * w[:, None]
        A = J.T @ J
        g = J.T @ res
        step = np.linalg.solve(A + lam * (np.trace(A) / len(ax) + 1e-12) * np.eye(len(ax)), -g)
        c2 = c.copy(); c2[ax] += step
        n2 = np.maximum(np.linalg.norm(c2 - P, axis=1), 1e-9)
        cost2, cost = np.sum(((n2 - r) * w) ** 2), np.sum(res ** 2)
        if cost2 <= cost:
            c, lam = c2, max(lam * 0.3, 1e-9)
            if np.linalg.norm(step) < 1e-6:
                break
        else:
            lam *= 10.0
    return c, np.linalg.norm(c - P, axis=1) - r


def fit_sphere(P, origin, r=RADIUS, w=None, fix_z=False, cfg: EstCfg | None = None):
    """known-radius sphere fit, robust (Tukey IRLS, then a MAD trim). P (n,3), origin (3,) camera centre (same frame).
    -> (centre (3,), rms of inliers, inlier mask) or None."""
    cfg = cfg or EstCfg()
    n = len(P)
    if n < 3:
        return None
    w = np.ones(n) if w is None else np.asarray(w, float)
    med = np.median(P, axis=0)
    u = med - origin
    u /= max(np.linalg.norm(u), 1e-9)
    c0 = med + 0.8 * r * u
    c, ww = c0, w.copy()
    for _ in range(4):
        c, res = _gn(P, ww, r, c, fix_z, iters=8)
        s = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 1e-3)
        ww = w * np.clip(1.0 - (res / (4.685 * s)) ** 2, 0.0, 1.0)          # Tukey biweight (sqrt of the squared-residual weight)
    keep = np.abs(res) <= max(cfg.trim_floor, cfg.trim_k * s)
    if keep.sum() >= 3 and keep.sum() < n:
        c, _ = _gn(P[keep], w[keep], r, c, fix_z, iters=8)
        res = np.linalg.norm(c - P, axis=1) - r
        keep = np.abs(res) <= max(cfg.trim_floor, cfg.trim_k * s)
    if keep.sum() < 3:
        return None
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    if np.linalg.norm(c - c0) > cfg.max_shift + r:
        return None
    return c, rms, keep


def fit_cov(P, c, r, fix_z, s):
    """3x3 covariance of the fitted centre: s^2 (J^T J)^-1 at the solution (unit weights); fix_z -> z entry 0."""
    dv = c - P
    nn = np.maximum(np.linalg.norm(dv, axis=1), 1e-9)
    J = dv / nn[:, None]
    ax = [0, 1] if fix_z else [0, 1, 2]
    A = J[:, ax].T @ J[:, ax]
    C = np.zeros((3, 3))
    C[np.ix_(ax, ax)] = (s ** 2) * np.linalg.pinv(A + 1e-9 * np.eye(len(ax)))
    return C


def measure(frames, cfg: EstCfg | None = None, specs=None, ret_pts=False):
    """Frames (list of depth_sim.Frame, one per camera, same tick) -> payload (PAYLOAD_LEN,). `specs` = matching CamSpec list,
    used only for the noise gate and the point weights (the estimator knows the sensor calibration; it never sees geometry ids).
    Payload: [valid, x, y, z (tray frame), n_inliers, rms, Rxx, Rxy, Ryy (xy covariance of the centre), n_candidates]."""
    cfg = cfg or EstCfg()
    P_all, W_all, ORG = [], [], []
    n_cand = 0
    for i, fr in enumerate(frames):
        if len(fr.pts_w) == 0:
            continue
        pt, org = fr.to_tray()
        sz_fn = None if specs is None else specs[i].sigma_z
        idx = select_points(pt, org, cfg, sz_fn, fr.z_depth)
        if len(idx) == 0:
            continue
        sig = (np.full(len(idx), 0.003) if sz_fn is None else np.asarray(sz_fn(fr.z_depth[idx])))
        P_all.append(pt[idx]); W_all.append(1.0 / np.maximum(sig, 5e-4)); n_cand += len(idx)
        ORG.append((org, len(idx)))
    out = np.zeros(PAYLOAD_LEN)
    out[I_CAND] = n_cand
    cand = np.concatenate(P_all) if P_all else np.zeros((0, 3))
    ret = (lambda o: (o, cand)) if ret_pts else (lambda o: o)
    if n_cand < 3:
        out[I_N] = n_cand
        return ret(out)
    P = np.concatenate(P_all); W = np.concatenate(W_all)
    W = W / W.mean()
    med = np.median(P, axis=0)                                   # coarse gate: isolated gross outliers do not enter the fit
    # view direction for the start guess: point-count weighted mean of the unit vectors camera -> median point
    vd = sum(n_i * (med - o) / max(np.linalg.norm(med - o), 1e-9) for o, n_i in ORG)
    vd /= max(np.linalg.norm(vd), 1e-9)
    origin = med - vd                                            # (only the direction med - origin is used by fit_sphere)
    g = np.linalg.norm(P - med, axis=1) < cfg.gate_r * cfg.radius
    P, W = P[g], W[g]
    f = fit_sphere(P, origin, cfg.radius, W, cfg.fix_z, cfg)
    if f is None:
        out[I_N] = len(P)
        return ret(out)
    c, rms, keep = f
    n_in = int(keep.sum())
    fz = cfg.fix_z
    if (not fz) and n_in < cfg.n_plane:                          # few points: use the tray plane (z = radius) to pin the fit
        f2 = fit_sphere(P, origin, cfg.radius, W, True, cfg)
        if f2 is not None:
            c, rms, keep = f2
            n_in, fz = int(keep.sum()), True
    ok = n_in >= cfg.n_min and rms <= cfg.rms_max
    C = fit_cov(P[keep], c, cfg.radius, fz, max(rms, cfg.s_floor))
    R = C[:2, :2] + (cfg.sig_floor ** 2 + (cfg.cap_sig8 * (8.0 / max(n_in, 1)) ** 3) ** 2) * np.eye(2)
    out[I_VALID], out[I_X:I_X + 3], out[I_N], out[I_RMS] = float(ok), c, n_in, rms
    out[I_RXX], out[I_RXY], out[I_RYY] = R[0, 0], R[0, 1], R[1, 1]
    return ret(out)


# ----------------------------------------------------------------------------------------------------------------
# Kalman filter
# ----------------------------------------------------------------------------------------------------------------
T_LOSS = 10           # ticks without a good measurement: count a track-loss event
T_REINIT = 60         # a measurement after this many ticks re-initialises the track
HIST = 16             # ticks of state history kept for delayed updates
COAST = 4             # ticks after the last good arrival before the coasting rules start
V_DECAY = 0.93        # velocity decay per tick while coasting (time constant 0.27 s)
FOOT = 0.04           # coasting estimate is clipped to the footprint grown by this [m]


class BallKF:
    """Constant-velocity KF on the tray-frame ball position. State x = [px, py, vx, vy]."""

    def __init__(self, sigma_a=1.5, dt=DT_WM, p0_pos=0.03, p0_vel=0.5, gate=25.0):
        self.sa, self.dt, self.gate = float(sigma_a), float(dt), float(gate)
        self.p0 = np.diag([p0_pos ** 2, p0_pos ** 2, p0_vel ** 2, p0_vel ** 2])
        self.F = np.eye(4); self.F[0, 2] = self.F[1, 3] = self.dt
        q = self.sa ** 2
        Q2 = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.Q = np.zeros((4, 4))
        for a in range(2):
            ix = [a, a + 2]
            self.Q[np.ix_(ix, ix)] = Q2
        self.width = 4 + 16 + 5 + HIST * 20 + 1
        self.reset_state(0, np.zeros(2), init=False)

    # ---- state
    def reset_state(self, tick, xy, init=True):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = self.p0.copy() if init else self.p0 * 4.0
        self.last_good = tick if init else -10_000          # tick of the last good ARRIVAL
        self.n_loss = 0
        self.lost = False
        self.n_upd = 0
        self.hx = np.zeros((HIST, 4)); self.hP = np.zeros((HIST, 4, 4)); self.hv = np.zeros(HIST, bool); self.htick = np.full(HIST, -1)
        self._store(tick)

    def _store(self, tick):
        k = tick % HIST
        self.hx[k], self.hP[k], self.htick[k] = self.x, self.P, tick

    def since(self, tick):
        return float(tick - self.last_good)

    def predict(self, tick):
        """advance one tick (call once per tick, before the deliveries of that tick)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if tick - self.last_good > COAST:                       # no measurement for a while: the ball is probably slowing, and it
            self.x[2:] *= V_DECAY                               # cannot be far outside the tray (a ball that leaves is lost anyway)
            self.x[0] = float(np.clip(self.x[0], -(HX + FOOT), HX + FOOT))
            self.x[1] = float(np.clip(self.x[1], -(HY + FOOT), HY + FOOT))
        self._store(tick)

    def _upd(self, x, P, z, Rm):
        S = P[:2, :2] + Rm
        y = z - x[:2]
        m2 = float(y @ np.linalg.solve(S, y))
        if m2 > self.gate:
            return x, P, False
        K = P[:, :2] @ np.linalg.inv(S)
        x2 = x + K @ y
        I_KH = np.eye(4) - K @ np.hstack([np.eye(2), np.zeros((2, 2))])
        P2 = I_KH @ P @ I_KH.T + K @ Rm @ K.T                               # Joseph form
        return x2, P2, True

    def deliver(self, tick, t_cap, payload):
        """a measurement captured at t_cap arrives at `tick` (call after predict(tick)). Returns True if it was used."""
        valid = payload[0] > 0.5
        if not valid:
            return False
        z = payload[1:3]
        Rm = np.array([[payload[I_RXX], payload[I_RXY]], [payload[I_RXY], payload[I_RYY]]])
        gap = tick - self.last_good
        lag = tick - t_cap
        if gap > T_REINIT or self.n_upd == 0:
            self.x = np.array([z[0], z[1], 0.0, 0.0]); self.P = self.p0.copy()
            self.P[:2, :2] = Rm
            self.n_upd += 1
            self._finish_good(tick)
            for tt in range(max(tick - HIST + 1, 0), tick + 1):               # history after a re-init is the new state
                k = tt % HIST; self.hx[k], self.hP[k], self.htick[k] = self.x, self.P, tt
            return True
        if 0 < lag < HIST - 1 and self.htick[t_cap % HIST] == t_cap:
            k = t_cap % HIST
            x, P = self.hx[k].copy(), self.hP[k].copy()
            x, P, ok = self._upd(x, P, z, Rm)
            if not ok:
                return False
            self.hx[k], self.hP[k] = x, P
            for tt in range(t_cap + 1, tick + 1):                            # re-predict through the delay
                x = self.F @ x
                P = self.F @ P @ self.F.T + self.Q
                kk = tt % HIST
                self.hx[kk], self.hP[kk], self.htick[kk] = x, P, tt
            self.x, self.P = x, P
        else:
            x, P, ok = self._upd(self.x, self.P, z, Rm)
            if not ok:
                return False
            self.x, self.P = x, P
            self._store(tick)
        self.n_upd += 1
        self._finish_good(tick)
        return True

    def _finish_good(self, tick):
        self.last_good = tick
        self.lost = False

    def end_tick(self, tick):
        """after the deliveries: update the loss flag / event count."""
        if self.since(tick) > T_LOSS and not self.lost:
            self.lost = True
            self.n_loss += 1

    # ---- snapshot
    def pack(self):
        return np.concatenate([self.x, self.P.ravel(), [self.last_good, self.n_loss, float(self.lost), self.n_upd, 0.0],
                               np.concatenate([self.hx, self.hP.reshape(HIST, 16)], 1).ravel(), [0.0]])

    def unpack(self, a):
        a = np.asarray(a, np.float64)
        self.x, self.P = a[:4].copy(), a[4:20].reshape(4, 4).copy()
        self.last_good, self.n_loss, self.lost, self.n_upd = int(a[20]), int(a[21]), bool(a[22]), int(a[23])
        h = a[25:25 + HIST * 20].reshape(HIST, 20)
        self.hx, self.hP = h[:, :4].copy(), h[:, 4:].reshape(HIST, 4, 4).copy()
        # tick labels of the history are implied by the ring index: recover them from the state tick
        self.htick = np.full(HIST, -1)


def unpack_with_tick(kf: BallKF, a, tick):
    """unpack and rebuild the history tick labels for the last HIST ticks (ring buffer: slot = tick % HIST)."""
    kf.unpack(a)
    for tt in range(max(tick - HIST + 1, 0), tick + 1):
        kf.htick[tt % HIST] = tt
