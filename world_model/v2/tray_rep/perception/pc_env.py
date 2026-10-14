"""pc_env.py -- TrayRepEnv with POINT-CLOUD ball tracking instead of privileged ball state. obs_mode 'pc_est'.
SIMULATION ONLY (MuJoCo ray casting on CPU; no GL, no GPU, no camera driver, no hardware, no network).

`PCRepEnv` subclasses `tray_rep.env.TrayRepEnv` (nothing in tray_rep is edited). The physics, the carry, the action, `priv`,
`l`, `n`, `nd` are the parent's, bit for bit. Only the ball part of `obs` changes.

    obs (16) = [axis c (3), tray velocity v (3), realised axis rate w (3)]          exact (the arms' joint encoders -> FK)
             + [ball x, y, vx, vy]  from the Kalman filter on the sphere-fit centres   (tray frame)
             + [n_points/100 (<=1), fit rms/4 mm (<=1; 1 when no fit), ticks since last good measurement/25 (<=1)]

Cameras (`camera=`): head | head_low | head_aim | wrist_l | wrist_r | wrist_both | head+wrist | head_aim+wrist | head_low+wrist
| zed_nav. See depth_sim.RIGS. With several cameras the clouds are merged (tray frame) and fitted together; the frame rate is the
slowest camera's, the latency the largest.

Compact object-centric cloud (`want_cloud=True`): K = 128 points in the tray frame (above-plane candidates of the LATEST DELIVERED
frame; subsampled if more, zero-padded, with a mask). Read it with `cloud()`; the generator can save it per tick.

State / snapshots. `snapshot()` returns the parent's RepSnap with `trk` = sensor clock + delivery queue + Kalman state + last
quality features + noise seed (+ the held cloud when `pack_cloud=True`). `restore()` rebuilds them. A snapshot with no sensor
state (e.g. from the data_piper_full dataset) restores to a steady start: the first measurement is taken from the restored scene.
Noise of frame at tick t comes from the generator seeded by (seed, t): the same snapshot gives the same noise.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep import common  # noqa: F401
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.perception import depth_sim as D
from world_model.v2.tray_rep.perception import ball_estimator as BE
from world_model.v2.testbeds.b_tray_ball.tray_env import STRIDE, DT_WM

PC_OBS_DIM = 16
K_CLOUD = 128
CLOUD_LEN = K_CLOUD * 4            # K points x 3 + K mask
N_CT = 6                           # capture-tick clouds kept in a snapshot when pack_cloud
N_REF, RMS_REF, T_REF = 100.0, 0.004, 25.0
POS_CLIP, VEL_CLIP = E.POS_CLIP, E.VEL_CLIP


# ----------------------------------------------------------------------------------------------------------------
class Pipeline:
    """sensor clock + delivery queue + Kalman filter. Driven by a capture function (live env) or by logged payloads (replay)."""

    def __init__(self, rate_hz, latency_ticks, sigma_a=1.5):
        self.P = BE.PAYLOAD_LEN
        self.sensor = D.DepthSensor(rate_hz, latency_ticks, BE.PAYLOAD_LEN)
        self.kf = BE.BallKF(sigma_a=sigma_a)
        self.feat = np.array([0.0, 0.0, 0.0, 0.0])          # last delivered: [n_points, rms, valid, n_candidates]
        self.cloud = None                                    # held compact cloud of the last delivered frame (set by the env)
        self.tick = 0

    # ---- lifecycle
    def init(self, tick, payload):
        """steady start at `tick` from the first measurement (taken immediately, no delay)."""
        self.sensor.reset()
        self.tick = tick
        p = np.asarray(payload, np.float64)
        if p[0] > 0.5:
            self.kf.reset_state(tick, p[1:3], init=True)
        else:
            self.kf.reset_state(tick, np.zeros(2), init=False)
        self.feat = np.array([p[BE.I_N], p[BE.I_RMS], p[BE.I_VALID], p[BE.I_CAND]])

    def step(self, tick, capture_fn):
        """one tick. Returns the capture ticks of the frames delivered now."""
        self.tick = tick
        self.kf.predict(tick)
        got = []
        for tc, p in self.sensor.step(tick, capture_fn):
            self.kf.deliver(tick, tc, p)
            self.feat = np.array([p[BE.I_N], p[BE.I_RMS], p[BE.I_VALID], p[BE.I_CAND]])
            got.append(tc)
        self.kf.end_tick(tick)
        return got

    # ---- outputs
    def ball(self):
        x = self.kf.x
        return np.concatenate([np.clip(x[:2], -POS_CLIP, POS_CLIP), np.clip(x[2:], -VEL_CLIP, VEL_CLIP)])

    def quality(self, tick):
        n, rms, valid, _ = self.feat
        fit = n >= 3 and rms > 0 or valid > 0.5
        return np.array([min(n / N_REF, 1.0), min(rms / RMS_REF, 1.0) if fit else 1.0,
                         min(self.kf.since(tick) / T_REF, 1.0)])

    # ---- snapshot
    def pack(self, with_cloud=False):
        parts = [[float(self.tick)], self.sensor.pack(), self.kf.pack(), self.feat]
        if with_cloud:
            parts.append(self.cloud if self.cloud is not None else np.zeros(CLOUD_LEN))
        return np.concatenate(parts)

    def width(self, with_cloud=False):
        return 1 + self.sensor.width + self.kf.width + 4 + (CLOUD_LEN if with_cloud else 0)

    def unpack(self, a, with_cloud=False):
        a = np.asarray(a, np.float64)
        self.tick = int(round(a[0]))
        o = 1
        self.sensor.unpack(a[o:o + self.sensor.width]); o += self.sensor.width
        BE.unpack_with_tick(self.kf, a[o:o + self.kf.width], self.tick); o += self.kf.width
        self.feat = a[o:o + 4].copy(); o += 4
        if with_cloud:
            self.cloud = a[o:o + CLOUD_LEN].copy()


def replay(payloads, rate_hz, latency_ticks, sigma_a=1.5):
    """Run the pipeline offline on logged per-tick payloads (captured every tick, shape (T, 8)). Returns
    est (T,4) [x, y, vx, vy], qual (T,3), since (T,), n_loss_events, delivered_valid (T,) bool."""
    T = len(payloads)
    pl = Pipeline(rate_hz, latency_ticks, sigma_a)
    pl.init(0, payloads[0])
    est, qual, since = np.zeros((T, 4)), np.zeros((T, 3)), np.zeros(T)
    est[0], qual[0], since[0] = pl.ball(), pl.quality(0), pl.kf.since(0)
    for t in range(1, T):
        pl.step(t, lambda t=t: payloads[t])
        est[t], qual[t], since[t] = pl.ball(), pl.quality(t), pl.kf.since(t)
    return est, qual, since, pl.kf.n_loss


# ----------------------------------------------------------------------------------------------------------------
class PCRepEnv(E.TrayRepEnv):
    def __init__(self, camera="head", a_scale: float = 1.0, lip: bool = False, mu_roll: float = 0.002, w_scale=None,
                 alpha=None, rate_hz=None, latency_ticks=None, cam_overrides=None, est_cfg: BE.EstCfg | None = None,
                 sigma_a: float = 1.5, want_cloud: bool = False, pack_cloud: bool = False, use_roi: bool = True,
                 log_meas: bool = False):
        super().__init__("full", a_scale, lip, mu_roll, w_scale, alpha)       # parent obs_mode 'full', tracker off
        self.obs_mode, self.obs_dim = "pc_est", PC_OBS_DIM
        self.camera = camera
        self.specs = D.rig_specs(camera, **(cam_overrides or {}))
        self.cams = [D.DepthCamera(s) for s in self.specs]
        self.scene = D.Scene(E.RADIUS, lip)
        self.cfg = est_cfg or BE.EstCfg()
        self.rate_hz = float(min(s.rate_hz for s in self.specs) if rate_hz is None else rate_hz)
        self.latency = int(max(s.latency_ticks for s in self.specs) if latency_ticks is None else latency_ticks)
        self.want_cloud, self.pack_cloud, self.use_roi = bool(want_cloud), bool(want_cloud and pack_cloud), use_roi
        self.sigma_a = float(sigma_a)
        self.pipe = Pipeline(self.rate_hz, self.latency, sigma_a)
        self._ctail = {}                                       # capture tick -> compact cloud tail (K*3 pts + K mask)
        self._seed = 0
        self._phase = None
        self.log = [] if log_meas else None                    # (tick, payload, n_ball_hits (eval), n_rays) per capture
        self.cloud_log = None                                  # list of (pts, mask) per recorded state, if enabled by the caller
        self.eval = dict(n_ball_hits=0, n_hits=0, n_rays=0)    # of the last capture (eval only: uses geometry ids)
        self.cost = dict(frames=0, rays=0, t_cast=0.0, t_fit=0.0)

    # ------------------------------------------------------------------ sensing
    def _tick_index(self):
        return int(self.env.k // STRIDE)

    def _capture(self, tick):
        """cast every camera at the current scene, fit, return the payload (+ cloud tail)."""
        import time
        e = self.env
        R = D.quat_to_R(e.quat)
        ball_w = e.d.qpos[e.bq:e.bq + 3]
        self.scene.set_pose(e.pos, e.quat, ball_w)
        t0 = time.perf_counter()
        frames = [c.capture(self.scene, e.pos, R, tick, self._seed, self.use_roi) for c in self.cams]
        t1 = time.perf_counter()
        pay, cand = BE.measure(frames, self.cfg, self.specs, ret_pts=True)
        t2 = time.perf_counter()
        self.cost["frames"] += 1; self.cost["rays"] += sum(f.n_rays for f in frames)
        self.cost["t_cast"] += t1 - t0; self.cost["t_fit"] += t2 - t1
        nb = int(sum((f.geom == self.scene.gid_ball).sum() for f in frames))
        self.eval = dict(n_ball_hits=nb, n_hits=int(sum(len(f.geom) for f in frames)), n_rays=int(sum(f.n_rays for f in frames)))
        if self.log is not None:
            self.log.append((tick, pay.copy(), nb, self.eval["n_rays"]))
        if self.want_cloud:
            tail = np.zeros(K_CLOUD * 4)
            n = len(cand)
            if n > 0:
                if n > K_CLOUD:
                    rng = np.random.default_rng(np.random.SeedSequence([self._seed & 0x7FFFFFFF, tick, 7]))
                    cand = cand[np.sort(rng.choice(n, K_CLOUD, replace=False))]
                    n = K_CLOUD
                tail[:n * 3] = cand.ravel()
                tail[K_CLOUD * 3:K_CLOUD * 3 + n] = 1.0
            self._ctail[tick] = tail
            for k in [k for k in self._ctail if k < tick - 12]:
                del self._ctail[k]
        return pay

    def _sense_init(self):
        t = self._tick_index()
        self._ctail = {}
        self.pipe.init(t, self._capture(t))
        self.pipe.cloud = self._ctail.get(t)

    def _sense_step(self):
        t = self._tick_index()
        got = self.pipe.step(t, lambda: self._capture(t))
        if got and self.want_cloud:
            tail = self._ctail.get(max(got))
            if tail is not None:
                self.pipe.cloud = tail

    # ------------------------------------------------------------------ parent hooks
    def reset(self, ball_xy=(0.0, 0.0), pos0=None, quat0=None, mu_roll=None, track_seed=0):
        self._seed = int(track_seed)
        self._phase = "init"
        try:
            return super().reset(ball_xy, pos0, quat0, mu_roll, track_seed)
        finally:
            self._phase = None

    def tick(self, u):
        self._phase = "step"
        try:
            return super().tick(u)
        finally:
            self._phase = None

    def record(self, w_obs, loc_prev):
        rec = super().record(w_obs, loc_prev)                 # parent: tracker is None -> privileged ball (replaced below)
        if self._phase == "init":
            self._sense_init()
        elif self._phase == "step":
            self._sense_step()
        t = self._tick_index()
        obs = np.concatenate([rec["obs"][:9], self.pipe.ball(), self.pipe.quality(t)])
        rec["obs"] = obs.astype(np.float32)
        if self.cloud_log is not None and self._phase is not None:
            self.cloud_log.append(self.cloud())
        return rec

    # ------------------------------------------------------------------ outputs
    def cloud(self):
        """(pts (K,3) float32 in the tray frame, mask (K,) bool) of the latest delivered frame; zeros if clouds are off."""
        pts, mask = np.zeros((K_CLOUD, 3), np.float32), np.zeros(K_CLOUD, bool)
        c = self.pipe.cloud
        if self.want_cloud and c is not None and len(c) == CLOUD_LEN:
            pts = c[:K_CLOUD * 3].reshape(K_CLOUD, 3).astype(np.float32)
            mask = c[K_CLOUD * 3:] > 0.5
        return pts, mask

    def track_info(self):
        t = self._tick_index()
        return dict(since=self.pipe.kf.since(t), lost=self.pipe.kf.lost, n_loss=self.pipe.kf.n_loss, feat=self.pipe.feat.copy())

    # ------------------------------------------------------------------ snapshot
    def snapshot(self):
        s = super().snapshot()
        parts = [[float(self._seed)], self.pipe.pack(self.pack_cloud)]
        if self.pack_cloud:                                    # clouds of frames still in the delivery queue
            ks = sorted(self._ctail)[-N_CT:]
            ct = np.zeros((N_CT, 1 + CLOUD_LEN)); ct[:, 0] = -1.0
            for i, k in enumerate(ks):
                ct[i, 0], ct[i, 1:] = k, self._ctail[k]
            parts.append(ct.ravel())
        s.trk = np.concatenate(parts)
        return s

    def restore(self, s, mu_roll=None):
        super().restore(s, mu_roll)
        w0 = 1 + self.pipe.width(self.pack_cloud)
        w = w0 + (N_CT * (1 + CLOUD_LEN) if self.pack_cloud else 0)
        if len(s.trk) == w:
            self._seed = int(round(s.trk[0]))
            self.pipe.unpack(s.trk[1:w0], self.pack_cloud)
            self._ctail = {}
            if self.pack_cloud:
                ct = s.trk[w0:].reshape(N_CT, 1 + CLOUD_LEN)
                self._ctail = {int(round(r[0])): r[1:].copy() for r in ct if r[0] >= 0}
        else:                                                  # snapshot without sensor state: steady start from the scene
            self._sense_init()


def env_from_meta_pc(meta: dict, **kw) -> PCRepEnv:
    """the PCRepEnv that matches a gen_pc dataset (meta.json)."""
    return PCRepEnv(meta.get("camera", "head"), meta.get("a_scale", 1.0), bool(meta.get("lip", False)),
                    w_scale=meta.get("w_scale"), alpha=meta.get("alpha"), rate_hz=meta.get("rate_hz"),
                    latency_ticks=meta.get("latency_ticks"), sigma_a=meta.get("sigma_a", 1.5),
                    cam_overrides=meta.get("cam_overrides"), **kw)
