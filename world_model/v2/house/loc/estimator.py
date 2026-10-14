"""loc/estimator.py -- base pose estimate from simulated wheel odometry + ZED VIO.  SIMULATION ONLY.

Real interfaces this mimics (read, never run): YOR/robot/zed_pub_node.py publishes `zed/pose` (positional tracking, `enable_imu_fusion`, area memory);
the swerve base has wheel/steer encoders (odometry).  We do NOT know their error statistics; every number below is an ASSUMPTION (LocCfg).

Sensor models (truth enters ONLY here, as the source the sensors measure):
  wheel odometry  per-tick body-frame increment of the true motion, times a per-episode per-axis scale error (calibration), a heading-rate bias
                  (steer/wheel misalignment), white noise, and random SLIP events (a fraction of the translation is not seen).
  ZED VIO         an independent dead-reckoned pose in its own frame: the true body-frame increment + random-walk drift growing with the travelled
                  distance / rotation (so heading drift turns into position drift), sampled at 30 Hz with white measurement noise and delivered
                  `latency_s` late.  RELOCALIZATION (area memory recognises a place): at Poisson rate per metre the VIO state snaps back to the true
                  pose (+ small residual) = a jump in the VIO output.  TRACKING DROPOUTS: Poisson in time; no VIO output, internal drift x5.
Fusion          complementary filter: the estimate integrates the wheel odometry at tick rate; each delivered VIO sample z (taken at t_m) pulls the
                estimate with fixed gains K on the innovation  z - est(t_m)  (the estimate AT the sample time is looked up in a short history, so
                the latency is handled exactly in the linear approximation).
Outputs         `.pose` = (x, y, yaw) estimate after the latest update; `.sigma_xy` = a model-based 1-sigma position proxy (grows with distance since
                the last VIO sample / relocalization; used for a localization-error margin); error stats vs truth (ATE / RPE).
Own RNG (SeedSequence([30000 + seed])): never touches the sim's or the operator's streams.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(p, inc):
    """pose p=(x,y,yaw) composed with a body-frame increment inc=(dx,dy,dyaw) (midpoint heading)."""
    ym = p[2] + 0.5 * inc[2]
    c, s = math.cos(ym), math.sin(ym)
    return np.array([p[0] + c * inc[0] - s * inc[1], p[1] + s * inc[0] + c * inc[1], wrap(p[2] + inc[2])])


def body_increment(p0, p1):
    """body-frame increment taking pose p0 to p1 (inverse of compose, midpoint heading)."""
    dyaw = wrap(p1[2] - p0[2])
    ym = p0[2] + 0.5 * dyaw
    c, s = math.cos(ym), math.sin(ym)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, dyaw])


@dataclass
class LocCfg:
    # ---- wheel odometry (ASSUMED)
    odo_scale_sigma: float = 0.015           # per-episode relative scale error per body axis (x, y), 1 sigma
    odo_yaw_scale_sigma: float = 0.02        # relative scale error on the measured yaw increment
    odo_yaw_bias: float = 0.003              # rad/s, 1 sigma of the per-episode heading-rate bias
    odo_noise_xy: float = 0.003              # m / sqrt(s) white noise on the translation increment  (sigma = this * sqrt(dt))
    odo_noise_yaw: float = 0.004             # rad / sqrt(s)
    slip_rate: float = 0.03                  # 1/s Poisson rate of slip events
    slip_dur: tuple = (0.2, 0.8)             # s
    slip_frac: tuple = (0.2, 0.6)            # fraction of the translation (and yaw) increment NOT measured while slipping
    # ---- ZED VIO (ASSUMED)
    vio_rate_hz: float = 30.0
    vio_latency_s: float = 0.06              # 3 ticks of 20 ms (as depth_sim.CamSpec.latency_ticks)
    vio_meas_xy: float = 0.005               # m   white measurement noise
    vio_meas_yaw: float = 0.002              # rad
    vio_drift_xy: float = 0.02               # m per sqrt(m travelled)   random-walk drift of the VIO frame  (~1 % of path length after ~25 m)
    vio_drift_yaw: float = 0.003             # rad per sqrt(m travelled)  (0.17 deg/sqrt(m))
    vio_drift_rot: float = 0.006             # rad per sqrt(rad rotated)
    reloc_per_m: float = 0.10                # relocalization events per metre travelled (area memory recognises a place)
    reloc_xy: float = 0.02                   # m   residual after a relocalization
    reloc_yaw: float = 0.01                  # rad
    dropout_rate: float = 0.02               # 1/s tracking-loss events
    dropout_dur: tuple = (0.5, 2.0)          # s
    init_xy: float = 0.01                    # m   error of the initial pose (aligned to the map at start)
    init_yaw: float = 0.005                  # rad
    # ---- fusion
    gain_xy: float = 0.08                    # per VIO sample (30 Hz) -> time constant ~0.4 s
    gain_yaw: float = 0.08
    # ---- uncertainty proxy for a localization-error margin
    sig_drift_xy: float = 0.02               # m per sqrt(m) since the last relocalization (matches vio_drift_xy)
    sig_floor_xy: float = 0.01               # m

    def asdict(self):
        return asdict(self)


class PoseEstimator:
    """update(t, true_pose, dt) once per sim tick; read .pose.   `reset(true_pose)` re-initialises.  All times in sim seconds."""

    def __init__(self, cfg: LocCfg | None = None, seed: int = 0, true_pose0=(0.0, 0.0, 0.0)):
        self.cfg = cfg or LocCfg()
        self.rng = np.random.default_rng(np.random.SeedSequence([30000 + int(seed)]))
        self.reset(true_pose0)

    # ------------------------------------------------------------------------------------------------
    def reset(self, true_pose0):
        c, r = self.cfg, self.rng
        p0 = np.asarray(true_pose0, float)
        self.odo_scale = 1.0 + r.normal(0, c.odo_scale_sigma, 2)
        self.odo_yaw_scale = 1.0 + r.normal(0, c.odo_yaw_scale_sigma)
        self.odo_yaw_bias = r.normal(0, c.odo_yaw_bias)
        self.slip_until, self.slip_f = -1.0, 0.0
        init = np.array([r.normal(0, c.init_xy), r.normal(0, c.init_xy), r.normal(0, c.init_yaw)])
        self.pose = compose(p0, init)
        self.vio = compose(p0, np.array([r.normal(0, c.init_xy), r.normal(0, c.init_xy), r.normal(0, c.init_yaw)]))
        self.vio_dropout_until = -1.0
        self.prev_true = p0.copy()
        self.t = 0.0
        self.next_vio_t = 0.0
        self.queue = deque()                      # (deliver_t, t_meas, z)
        self.hist = deque(maxlen=64)              # (t, est pose)  after each update
        self.hist.append((0.0, self.pose.copy()))
        self.v_odo = np.zeros(3)
        self.dist_since_reloc = 0.0
        self.dist_total = 0.0
        self.n_reloc = self.n_dropout = self.n_slip = self.n_vio = 0
        self.log_t, self.log_true, self.log_est = [0.0], [p0.copy()], [self.pose.copy()]
        self.events = []                          # (t, kind)

    # ------------------------------------------------------------------------------------------------
    def update(self, t, true_pose, dt):
        c, r = self.cfg, self.rng
        tp = np.asarray(true_pose, float)
        inc = body_increment(self.prev_true, tp)
        self.prev_true = tp.copy()
        ds = float(math.hypot(inc[0], inc[1]))
        self.dist_total += ds
        self.dist_since_reloc += ds
        # -- wheel odometry (slip events, scale, bias, noise)
        if t >= self.slip_until and r.random() < c.slip_rate * dt:
            self.slip_until = t + r.uniform(*c.slip_dur)
            self.slip_f = r.uniform(*c.slip_frac)
            self.n_slip += 1
            self.events.append((t, "slip"))
        keep = 1.0 - (self.slip_f if t < self.slip_until else 0.0)
        sq = math.sqrt(dt)
        odo = np.array([inc[0] * self.odo_scale[0] * keep + r.normal(0, c.odo_noise_xy * sq),
                        inc[1] * self.odo_scale[1] * keep + r.normal(0, c.odo_noise_xy * sq),
                        inc[2] * self.odo_yaw_scale * keep + self.odo_yaw_bias * dt + r.normal(0, c.odo_noise_yaw * sq)])
        self.pose = compose(self.pose, odo)
        self.v_odo = odo / dt                      # measured body twist (vx, vy, wz): what arm-row feed-forward / base-acceleration code may read
        # -- VIO internal dead reckoning (own frame, drifting)
        in_drop = t < self.vio_dropout_until
        if not in_drop and r.random() < c.dropout_rate * dt:
            self.vio_dropout_until = t + r.uniform(*c.dropout_dur)
            in_drop = True
            self.n_dropout += 1
            self.events.append((t, "vio_dropout"))
        k = 5.0 if in_drop else 1.0
        sd = math.sqrt(max(ds, 0.0))
        sr = math.sqrt(abs(inc[2]))
        vinc = np.array([inc[0] + r.normal(0, k * c.vio_drift_xy * sd), inc[1] + r.normal(0, k * c.vio_drift_xy * sd),
                         inc[2] + r.normal(0, k * (c.vio_drift_yaw * sd + c.vio_drift_rot * sr))])
        self.vio = compose(self.vio, vinc)
        # -- relocalization (distance-driven Poisson)
        if not in_drop and ds > 0 and r.random() < c.reloc_per_m * ds:
            self.vio = compose(tp, np.array([r.normal(0, c.reloc_xy), r.normal(0, c.reloc_xy), r.normal(0, c.reloc_yaw)]))
            self.dist_since_reloc = 0.0
            self.n_reloc += 1
            self.events.append((t, "reloc"))
        # -- VIO sampling at vio_rate_hz, queued with latency
        tn = t + dt
        self.t = tn
        while self.next_vio_t <= tn + 1e-12:
            tm = tn                                  # the VIO state is the end-of-tick state: stamp it with the tick end (exact history lookup)
            self.next_vio_t += 1.0 / c.vio_rate_hz
            if in_drop:
                continue
            z = self.vio + np.array([r.normal(0, c.vio_meas_xy), r.normal(0, c.vio_meas_xy), r.normal(0, c.vio_meas_yaw)])
            z[2] = wrap(z[2])
            self.queue.append((tm + c.vio_latency_s, tm, z))
        # -- deliver + fuse
        while self.queue and self.queue[0][0] <= tn + 1e-12:
            _, tm, z = self.queue.popleft()
            est_m = self._hist_at(tm)
            innov = np.array([z[0] - est_m[0], z[1] - est_m[1], wrap(z[2] - est_m[2])])
            self.pose = np.array([self.pose[0] + c.gain_xy * innov[0], self.pose[1] + c.gain_xy * innov[1], wrap(self.pose[2] + c.gain_yaw * innov[2])])
            self.n_vio += 1
        self.hist.append((tn, self.pose.copy()))
        self.log_t.append(tn); self.log_true.append(tp.copy()); self.log_est.append(self.pose.copy())
        return self.pose

    def _hist_at(self, tm):
        h = self.hist
        if tm <= h[0][0]:
            return h[0][1]
        for i in range(len(h) - 1, -1, -1):
            if h[i][0] <= tm + 1e-12:
                return h[i][1]
        return h[0][1]

    # ------------------------------------------------------------------------------------------------
    @property
    def sigma_xy(self):
        """1-sigma position-error proxy [m] (model-based, not computed from truth): floor + drift since the last relocalization, capped."""
        c = self.cfg
        return min(c.sig_floor_xy + c.sig_drift_xy * math.sqrt(self.dist_since_reloc), 0.25)

    def errors(self):
        """ATE / RPE of the estimate vs truth over the logged run (position m, heading rad).  RPE over 1 m of travel and over 1 s."""
        T, E, t = np.array(self.log_true), np.array(self.log_est), np.array(self.log_t)
        e = E[:, :2] - T[:, :2]
        pe = np.hypot(e[:, 0], e[:, 1])
        ye = np.abs(np.array([wrap(a) for a in (E[:, 2] - T[:, 2])]))
        out = dict(ate_rmse_m=float(np.sqrt((pe ** 2).mean())), ate_max_m=float(pe.max()), ate_final_m=float(pe[-1]),
                   yaw_err_rms_rad=float(np.sqrt((ye ** 2).mean())), yaw_err_max_rad=float(ye.max()),
                   dist_m=float(self.dist_total), n_reloc=self.n_reloc, n_dropout=self.n_dropout, n_slip=self.n_slip, n_vio=self.n_vio)
        # RPE: relative-pose error between pairs of poses ~1 m of path apart / ~1 s apart
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(T[:, :2], axis=0).T))])

        def rpe(idx_pairs):
            v = []
            for i, j in idx_pairs:
                dt_ = body_increment(T[i], T[j]); de = body_increment(E[i], E[j])
                v.append(math.hypot(de[0] - dt_[0], de[1] - dt_[1]))
            return (float(np.sqrt(np.mean(np.square(v)))), float(np.max(v))) if v else (float("nan"), float("nan"))
        n = len(T)
        if s[-1] > 1.0:
            j = np.searchsorted(s, s + 1.0)
            pairs = [(i, jj) for i, jj in enumerate(j) if jj < n][::5]
            out["rpe_1m_rmse_m"], out["rpe_1m_max_m"] = rpe(pairs)
        k1 = int(round(1.0 / max(np.median(np.diff(t)), 1e-6)))
        pairs = [(i, i + k1) for i in range(0, n - k1, 5)]
        out["rpe_1s_rmse_m"], out["rpe_1s_max_m"] = rpe(pairs)
        return out


_SCALE_FIELDS = ("odo_scale_sigma", "odo_yaw_scale_sigma", "odo_yaw_bias", "odo_noise_xy", "odo_noise_yaw", "slip_rate", "vio_meas_xy", "vio_meas_yaw",
                 "vio_drift_xy", "vio_drift_yaw", "vio_drift_rot", "dropout_rate", "reloc_xy", "reloc_yaw", "init_xy", "init_yaw")


def scaled(cfg: LocCfg, k: float) -> LocCfg:
    """cfg with every error magnitude (sigmas, drift, slip / dropout rates) multiplied by k (k=0 -> perfect sensors, k=2 -> pessimistic).  The
    relocalization rate is divided by k (fewer snaps when worse)."""
    from dataclasses import replace
    if k == 1.0:
        return cfg
    kw = {f: getattr(cfg, f) * k for f in _SCALE_FIELDS}
    kw["reloc_per_m"] = cfg.reloc_per_m / k if k > 0 else cfg.reloc_per_m
    return replace(cfg, **kw)
