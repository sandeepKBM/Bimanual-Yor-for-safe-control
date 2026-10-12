"""env.py -- thin wrapper around b_tray_ball's TrayEnv for the tray_rep replication.
SIMULATION ONLY (MuJoCo, CPU, no rendering). Nothing in b_tray_ball is edited: it is imported.

What the wrapper adds
  * hidden per-episode rolling resistance `mu_roll` (log-uniform [5e-4, 6e-3] m). It lives in the MuJoCo model
    (geom friction), is recorded in the dataset (`hid`) and is NEVER part of `obs`.
  * `a_scale`: multiplies the linear-accel columns of the 85-element fallback set and the accel limit of the brake
    controller. `w_scale` (default = `a_scale`, which is the phase-1a behaviour) does the same for the tilt-rate columns
    and the brake's rate limit. `alpha` (default None = off) is a tilt-rate SLEW limit [rad/s^2]: the tilt rate that
    reaches the tray is `w_slew + clip(w_cmd - w_slew, +-alpha*dt)` (the arm cannot step its angular rate; the
    authority agent's model, `authority/scaled_probe.py`). `w_slew` is part of the snapshot. The Piper-feasible values
    (a 0.3, w 0.5, alpha 10) are in `configs/piper.json`.
  * ball tracker model (`full` mode only, default OFF = today's obs bitwise): `obs_noise_m` (std of Gaussian noise on ball
    x, y [m]), `obs_latency_ticks` (the ball part is k ticks old), `obs_rate_hz` (sample-and-hold; 50 = every tick),
    `obs_dropout_p` (per-tick probability that a due sample is lost and the last value is held). The ball velocity in obs is
    the finite difference of the noisy positions between consecutive accepted samples. The tray part of obs (first 9
    numbers) and `priv`, `l`, `n`, `nd` stay exact (the tray comes from joint encoders; priv is privileged truth).
    The tracker state (a few numbers) is part of the snapshot. Its noise stream is separate from every other RNG.
  * `obs_mode`
        full       obs (13) = [axis c (3), tray-origin velocity v (3), realised axis rate w (3), ball x, y, vx, vy (4)]
        tray_only  obs (9)  = [axis c (3), v (3), w (3)]        (the ZED does not see the ball)
    The ball entries are in the TRAY frame. vx, vy = backward difference of the tray-frame ball position over one
    tick (what a tracker gives; noise-free here). The first 9 numbers are the same formulas as the pendulum obs.
  * `priv` (11, privileged: aux heads and labels only, never a WM input):
        [0:3] ball centre in the tray frame, [3:6] its backward-difference velocity in the tray frame,
        [6:9] ball angular velocity in the tray frame, [9] signed distance d to the open tray edge [m],
        [10] lift = z - radius [m]
  * `act` = [a (3), w_eff (3)] with a = the commanded tray-origin acceleration, w_eff = the realised axis rate, i.e. the
    same meaning as the pendulum `act`. After a tick, obs[6:9] == act[3:6].

State / time convention (== the pendulum data): arrays have one row per tick t; row t holds the state at tick t
(before the action) and the action `act[t]` applied from t to t+1.

Out-of-range values after the ball has left the tray (it falls forever, there is no floor) are clipped:
position +-0.5 m, velocity +-3 m/s, ball angular velocity +-30 rad/s, d in [-1, 1] m.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world_model.v2.tray_rep import common  # noqa: F401  (thread pins, sys.path)
from world_model.v2.testbeds.b_tray_ball import tray_env as TE
from world_model.v2.testbeds.b_tray_ball.tray_env import (TrayEnv, BallParams, B, DT_WM, STRIDE, ST_OK, ST_OFF,
                                                          ST_TUNNEL, HX, HY, M_LO, M_HI)

MU_LO, MU_HI = 5e-4, 6e-3
OBS_DIMS = {"full": 13, "tray_only": 9}
PRIV_DIM = 11
FEAT_DIM = 10
POS_CLIP, VEL_CLIP, OMEGA_CLIP = 0.5, 3.0, 30.0
RADIUS = BallParams().radius


def sample_mu_roll(rng) -> float:
    return float(np.exp(rng.uniform(np.log(MU_LO), np.log(MU_HI))))


def feats_tray(priv):
    """privileged features for the aux head (the analogue of train_kinwm.feats_t). priv (..., 11) -> (..., 10) float32,
    scaled to O(1) and clamped to +-4:  [x/HX, y/HY, lift/0.01, vx/0.5, vy/0.5, vz/0.5, wx/10, wy/10, wz/10, d/0.08]."""
    p = np.asarray(priv, np.float32)
    s = np.array([HX, HY, 0.01, 0.5, 0.5, 0.5, 10.0, 10.0, 10.0, M_HI], np.float32)
    idx = [0, 1, 10, 3, 4, 5, 6, 7, 8, 9]
    return np.clip(p[..., idx] / s, -4.0, 4.0).astype(np.float32)


class Tracker:
    """camera-tracker model for the ball part of obs. State = history of true tray-frame xy (latency + 1 ticks), the held
    measurement (pos, vel), the position of the last accepted sample, ticks since it, the rate accumulator."""

    def __init__(self, noise_m=0.0, latency=0, rate_hz=50.0, dropout_p=0.0):
        self.noise, self.lat, self.rate, self.drop = float(noise_m), int(latency), float(rate_hz), float(dropout_p)
        self.width = 2 * (self.lat + 1) + 2 + 2 + 2 + 1 + 1
        self.rng = np.random.default_rng(0)

    @property
    def active(self):
        return self.noise > 0 or self.lat > 0 or self.rate < 50.0 or self.drop > 0

    def reset(self, xy, seed=0):
        self.rng = np.random.default_rng(int(seed))
        self.hist = np.tile(np.asarray(xy, np.float64), (self.lat + 1, 1))
        self.pos = self.hist[0] + self.noise * self.rng.normal(size=2)
        self.vel = np.zeros(2); self.last = self.pos.copy(); self.n_since = 0; self.acc = 0.0
        return self.out()

    def update(self, xy):
        """one tick: push the true xy, maybe take a sample. Fixed RNG use (uniform + 2 normals every tick)."""
        self.hist = np.concatenate([self.hist[1:], np.asarray(xy, np.float64)[None]], 0)
        u, z = self.rng.random(), self.rng.normal(size=2)
        self.n_since += 1
        self.acc += self.rate / 50.0
        if self.acc >= 1.0 - 1e-9:
            self.acc -= 1.0
            if u >= self.drop:
                p = self.hist[0] + self.noise * z
                self.vel = (p - self.last) / (self.n_since * DT_WM)
                self.pos, self.last, self.n_since = p, p.copy(), 0
        return self.out()

    def out(self):
        """(4,) clipped [x, y, vx, vy]."""
        return np.concatenate([np.clip(self.pos, -POS_CLIP, POS_CLIP), np.clip(self.vel, -VEL_CLIP, VEL_CLIP)])

    def pack(self):
        return np.concatenate([self.hist.ravel(), self.pos, self.vel, self.last, [self.n_since, self.acc]])

    def unpack(self, a, seed):
        a = np.asarray(a, np.float64)
        n = 2 * (self.lat + 1)
        self.hist = a[:n].reshape(-1, 2).copy()
        self.pos, self.vel, self.last = a[n:n + 2].copy(), a[n + 2:n + 4].copy(), a[n + 4:n + 6].copy()
        self.n_since, self.acc = int(a[n + 6]), float(a[n + 7])
        self.rng = np.random.default_rng(int(seed))


@dataclass
class RepSnap:
    """flat, serialisable copy of TrayEnv.Snap (state, prev, has_prev, kin, status, k) + the tilt-rate slew state."""
    snap: TE.Snap
    w_slew: np.ndarray = field(default_factory=lambda: np.zeros(3))      # last tilt rate that reached the tray
    trk: np.ndarray = field(default_factory=lambda: np.zeros(0))         # tracker state (empty = tracker off / unknown)

    def arrays(self):
        s = self.snap
        return dict(state=s.state, prev=s.prev, has_prev=np.int8(s.has_prev), kin=s.kin, status=np.int8(s.status),
                    k=np.int32(s.k), w_slew=np.asarray(self.w_slew, np.float64),
                    trk=np.asarray(self.trk, np.float64))

    @staticmethod
    def from_arrays(state, prev, has_prev, kin, status, k, w_slew=None, trk=None):
        return RepSnap(TE.Snap(np.array(state, np.float64), np.array(prev, np.float64), int(has_prev),
                               np.array(kin, np.float64), int(status), int(k)),
                       np.zeros(3) if w_slew is None else np.array(w_slew, np.float64),
                       np.zeros(0) if trk is None else np.array(trk, np.float64))


class TrayRepEnv:
    def __init__(self, obs_mode: str = "full", a_scale: float = 1.0, lip: bool = False, mu_roll: float = 0.002,
                 w_scale=None, alpha=None, obs_noise_m=0.0, obs_latency_ticks=0, obs_rate_hz=50.0, obs_dropout_p=0.0):
        assert obs_mode in OBS_DIMS
        self.obs_mode, self.a_scale, self.lip = obs_mode, float(a_scale), bool(lip)
        self.w_scale = self.a_scale if w_scale is None else float(w_scale)
        self.alpha = None if alpha is None else float(alpha)
        self.w_slew = np.zeros(3)
        self.tracker = Tracker(obs_noise_m, obs_latency_ticks, obs_rate_hz, obs_dropout_p) if obs_mode == "full" else None
        if self.tracker is not None and not self.tracker.active:
            self.tracker = None                                   # defaults: today's obs, bitwise
        self._ball = None
        self.obs_dim = OBS_DIMS[obs_mode]
        self.env = TrayEnv(BallParams(mu_roll=mu_roll), lip=lip)
        m = self.env.m
        names = ("tray_top", "ball_g") + (("lip",) if lip else ())
        self._gids = [m.geom(n).id for n in names]
        U = np.array(B.fallback_U(), np.float64)                                          # (85, 6)
        U[:, :3] *= self.a_scale; U[:, 3:] *= self.w_scale
        self.U = np.ascontiguousarray(U)
        self.a_max, self.w_max = 5.0 * self.a_scale, 2.0 * self.w_scale
        self.set_mu_roll(mu_roll)
        self._loc = None

    # ------------------------------------------------------------------ hidden parameter
    def set_mu_roll(self, mu: float):
        """rolling friction of ball and tray (same value on both geoms, so the contact value is mu)."""
        for g in self._gids:
            self.env.m.geom_friction[g, 2] = mu
        self.mu_roll = float(mu)
        self.env.p.mu_roll = float(mu)

    # ------------------------------------------------------------------ state
    def reset(self, ball_xy=(0.0, 0.0), pos0=None, quat0=None, mu_roll=None, track_seed=0):
        if mu_roll is not None:
            self.set_mu_roll(mu_roll)
        e = self.env
        e.reset(e.home if pos0 is None else pos0, np.array([1.0, 0, 0, 0]) if quat0 is None else quat0, ball_xy=ball_xy)
        self._loc = e.local().copy()
        self.w_slew = np.zeros(3)
        if self.tracker is not None:
            self._ball = self.tracker.reset(self._loc[:2], track_seed)
        return self.record(w_obs=np.zeros(3), loc_prev=self._loc)

    def snapshot(self) -> RepSnap:
        return RepSnap(self.env.snapshot(), self.w_slew.copy(),
                       self.tracker.pack() if self.tracker is not None else np.zeros(0))

    def restore(self, s: RepSnap, mu_roll=None):
        if mu_roll is not None:
            self.set_mu_roll(mu_roll)
        self.env.restore(s.snap)
        self.w_slew = np.array(s.w_slew, np.float64)
        self._loc = self.env.local().copy()
        if self.tracker is not None:
            seed = (int(s.snap.k) * 1_000_003 + 12345) & 0x7FFFFFFF           # same snapshot -> same noise stream
            if len(s.trk) == self.tracker.width:
                self.tracker.unpack(s.trk, seed)
            else:                                                         # snapshot without tracker state: start steady
                self.tracker.reset(self._loc[:2], seed)
            self._ball = self.tracker.out()

    @property
    def status(self):
        return self.env.status

    # ------------------------------------------------------------------ observation / privileged state
    def record(self, w_obs, loc_prev):
        """dict(obs, priv, l, d) for the CURRENT state. w_obs = realised axis rate of the step that led here."""
        e = self.env
        loc = e.local()
        d = e.dist(loc)
        R = e._R(e.quat)
        qd = e.d.qvel
        Rb = e._R(e.d.qpos[e.bq + 3:e.bq + 7])
        om_t = R.T @ (Rb @ qd[e.bv + 3:e.bv + 6])
        vrel = (loc - loc_prev) / DT_WM
        c = e.axis()
        obs9 = np.concatenate([c, np.clip(e.v, -VEL_CLIP, VEL_CLIP), w_obs])
        locc = np.clip(loc, -POS_CLIP, POS_CLIP)
        vc = np.clip(vrel, -VEL_CLIP, VEL_CLIP)
        ball = [locc[0], locc[1], vc[0], vc[1]] if self.tracker is None else self._ball
        obs = np.concatenate([obs9, ball]) if self.obs_mode == "full" else obs9
        priv = np.concatenate([locc, vc, np.clip(om_t, -OMEGA_CLIP, OMEGA_CLIP), [np.clip(d, -1.0, 1.0)],
                               [np.clip(loc[2] - RADIUS, -POS_CLIP, POS_CLIP)]])
        return dict(obs=obs.astype(np.float32), priv=priv.astype(np.float32), l=float(e.margin(loc)), d=float(d))

    def current(self):
        """record of the current state when the previous-tick quantities are unknown (w_obs = 0, vrel = 0)."""
        return self.record(np.zeros(3), self.env.local())

    def tick(self, u):
        """one 50 Hz tick under the held u = (a[3], w[3]). Returns the record of the NEW state plus act, failed, tunnel.
        With `alpha` set, u[3:] is the COMMANDED tilt rate; the rate that reaches the tray is slew-limited."""
        e = self.env
        u = np.asarray(u, np.float64)
        if self.alpha is not None:
            self.w_slew = self.w_slew + np.clip(u[3:] - self.w_slew, -self.alpha * DT_WM, self.alpha * DT_WM)
            u = np.concatenate([u[:3], self.w_slew])
        loc_prev = self._loc
        r = e.tick(u)
        w_obs = r["obs"][6:9]
        self._loc = e.local().copy()
        if self.tracker is not None:
            self._ball = self.tracker.update(self._loc[:2])
        rec = self.record(w_obs, loc_prev)
        rec["act"] = np.concatenate([u[:3], w_obs]).astype(np.float32)
        rec["failed"] = e.status == ST_OFF
        rec["tunnel"] = e.status == ST_TUNNEL
        return rec

    # ------------------------------------------------------------------ misc
    def tray_kin(self):
        return self.env.pos.copy(), self.env.v.copy(), self.env.quat.copy()


def env_from_meta(meta: dict, a_scale=None, obs_mode=None) -> TrayRepEnv:
    return TrayRepEnv(obs_mode or meta["obs_mode"], meta.get("a_scale", 1.0) if a_scale is None else a_scale,
                      bool(meta.get("lip", False)), w_scale=meta.get("w_scale"), alpha=meta.get("alpha"),
                      obs_noise_m=meta.get("obs_noise_m", 0.0), obs_latency_ticks=meta.get("obs_latency_ticks", 0),
                      obs_rate_hz=meta.get("obs_rate_hz", 50.0), obs_dropout_p=meta.get("obs_dropout_p", 0.0))


def make_env(meta: dict, a_scale=None):
    """The env that matches a dataset's meta.json. `pc_est` (point-cloud ball tracker, perception/) -> PCRepEnv; every other
    obs mode -> `env_from_meta` exactly as before. Use this wherever a dataset's snapshots are restored (branches, closed
    loop, oracle): the PCRepEnv restore brings back the sensor and estimator state, so the observations stay on-distribution."""
    if meta.get("obs_mode") == "pc_est":
        from world_model.v2.tray_rep.perception.pc_env import env_from_meta_pc
        return env_from_meta_pc(meta if a_scale is None else dict(meta, a_scale=a_scale))
    return env_from_meta(meta, a_scale=a_scale)
