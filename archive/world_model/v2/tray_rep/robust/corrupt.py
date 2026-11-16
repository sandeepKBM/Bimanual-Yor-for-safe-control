"""robust/corrupt.py -- observation corruption for the ball part of obs (columns 9-12) and the tracker uncertainty set.
SIMULATION / SAVED DATA ONLY (numpy, CPU). No hardware, no network.

Corruptor   a camera-tracker model with the same maths and the same random draws as env.Tracker (so `trk` with no burst
            reproduces env.Tracker bitwise), plus
              * dropout BURSTS: a window [start, start+len) in which every due sample is lost (the last value is held,
                exactly as for the 2% random dropouts the models were trained on);
              * a per-tick `info` dict: accepted?, age (ticks since the last accepted sample), phase.
            A condition with no corruption passes the exact ball [x, y, vx, vy] through.
TrackerSet  the tracker-style uncertainty set. A constant-velocity Kalman covariance per axis (white-noise acceleration
            sigma_a), updated on every accepted sample and propagated by (latency + age) ticks. Its position std GROWS while
            samples are missing, and a Minkowski term lam * v_max * dt * delay_eff covers the latency / staleness bias.
            radius = scale * (kappa * sigma_p + extra). `scale` is the one number calibrated on VAL for coverage.
            `support(ball)` returns the centre plus 8 boundary points (45 degree steps) of the set, each with the
            conditional velocity shift (Sigma_vp / Sigma_pp) * dp.
Phases      steady    : the last accepted sample is younger than AGE_DROP ticks and no dropout ended in the last REACQ ticks
            dropout   : the last accepted sample is AGE_DROP (5 ticks = 100 ms) old or older
            reacq     : a dropout ended (a sample was accepted after a gap >= AGE_DROP) less than REACQ (10) ticks ago, and no new dropout
Conditions  `parse_cond(name)`: off | trk | trkn6 (6 mm) | lat<k> | trklat<k> | trk[B<L>(s|f|r)] | custom:key=val,...
            B<L>s burst of L ticks that started U ticks before the decision tick, U ~ {0..24}  (Test 1: phases at the state)
            B<L>f burst of L ticks that starts U ticks before the NOMINAL failure, U ~ {0..25}  (carry onset / t + hold_ftk)
            B<L>r burst of L ticks at a random time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401  (thread pins, sys.path)
from world_model.v2.tray_rep import env as E

DT = E.DT_WM
POS_CLIP, VEL_CLIP = E.POS_CLIP, E.VEL_CLIP
AGE_DROP, REACQ = 5, 10
PH_STEADY, PH_DROP, PH_REACQ = 0, 1, 2
PHASES = ("steady", "dropout", "reacq")
BURST_WINDOW = 25


# ===================================================================================== conditions
@dataclass(frozen=True)
class Cond:
    name: str = "off"
    noise: float = 0.0           # m, Gaussian, on x and y
    lat: int = 0                 # ticks
    rate: float = 50.0           # Hz (sample and hold)
    drop: float = 0.0            # random dropout probability per due sample
    blen: int = 0                # burst length [ticks]; 0 = none
    bmode: str = "none"          # none | state | fail | random

    @property
    def active(self):
        return self.noise > 0 or self.lat > 0 or self.rate < 50.0 or self.drop > 0 or self.blen > 0

    def with_lat(self, k):
        return replace(self, lat=int(k), name=f"{self.name}@lat{int(k)}")


TRK = dict(noise=0.003, lat=3, rate=30.0, drop=0.02)           # configs/piper.json tracker_realistic


def parse_cond(name: str) -> Cond:
    if name in ("off", "exact", ""):
        return Cond("off")
    if name.startswith("custom:"):
        kv = dict(x.split("=") for x in name[7:].split(",") if x)
        return Cond(name, noise=float(kv.get("noise", 0)), lat=int(kv.get("lat", 0)), rate=float(kv.get("rate", 50)),
                    drop=float(kv.get("drop", 0)), blen=int(kv.get("blen", 0)), bmode=kv.get("bmode", "none"))
    m = re.fullmatch(r"lat(\d+)", name)
    if m:
        return Cond(name, lat=int(m.group(1)))
    m = re.fullmatch(r"trklat(\d+)", name)
    if m:
        return Cond(name, **(TRK | dict(lat=int(m.group(1)))))
    m = re.fullmatch(r"trkn(\d+)", name)
    if m:
        return Cond(name, **(TRK | dict(noise=int(m.group(1)) * 1e-3)))
    m = re.fullmatch(r"trkB(\d+)([sfr])", name)
    if m:
        return Cond(name, **TRK, blen=int(m.group(1)), bmode={"s": "state", "f": "fail", "r": "random"}[m.group(2)])
    if name == "trk":
        return Cond(name, **TRK)
    raise ValueError(f"unknown corruption condition {name!r}")


def burst_start(cond: Cond, key: int, t_state: int = -1, t_fail: int = -1, T: int = 200):
    """absolute start tick of the burst for this (condition, run key), or None. Deterministic in `key`."""
    if cond.blen <= 0 or cond.bmode == "none":
        return None
    rng = np.random.default_rng([int(key) & 0x7FFFFFFF, 0xB005])
    U = int(rng.integers(0, BURST_WINDOW + (1 if cond.bmode == "fail" else 0)))
    hi = max(11, int(T) - cond.blen)
    if cond.bmode == "state":
        return max(1, int(t_state) - U)
    if cond.bmode == "fail" and t_fail >= 0:
        return max(1, int(t_fail) - U)
    return int(rng.integers(10, hi))


# ===================================================================================== the corruptor
class Corruptor:
    """Same random stream as env.Tracker: reset draws normal(2); each update draws random() then normal(2)."""

    def __init__(self, cond: Cond, seed: int = 0, burst=None):
        self.c = cond
        self.seed = int(seed)
        self.burst = burst            # (start_tick, length) or None
        self.k = 0                    # current tick (0 after reset)

    # --- helpers
    def in_burst(self, k):
        return self.burst is not None and self.burst[0] <= k < self.burst[0] + self.burst[1]

    def reset(self, xy, vel=(0.0, 0.0)):
        c = self.c
        self.k = 0
        self.rng = np.random.default_rng(self.seed)
        xy = np.asarray(xy, np.float64)
        self.hist = np.tile(xy, (c.lat + 1, 1))
        if c.active:
            self.pos = self.hist[0] + c.noise * self.rng.normal(size=2)
            self.vel = np.zeros(2)
        else:
            self.pos, self.vel = xy.copy(), np.asarray(vel, np.float64)
        self.last = self.pos.copy()
        self.n_since, self.acc, self.gap = 0, 0.0, 0
        self.reacq_k = -10 ** 6                                   # tick at which the last dropout (age >= AGE_DROP) ended
        return self._info(True)

    def update(self, xy, vel):
        """one tick: push the TRUE tray-frame ball xy (and its true backward-difference velocity), maybe take a sample."""
        c = self.c
        self.k += 1
        xy = np.asarray(xy, np.float64)
        if not c.active:
            self.pos, self.vel = xy.copy(), np.asarray(vel, np.float64)
            return self._info(True)
        self.hist = np.concatenate([self.hist[1:], xy[None]], 0)
        u, z = self.rng.random(), self.rng.normal(size=2)
        self.n_since += 1
        self.acc += c.rate / 50.0
        accepted = False
        if self.acc >= 1.0 - 1e-9:
            self.acc -= 1.0
            if u >= c.drop and not self.in_burst(self.k):
                p = self.hist[0] + c.noise * z
                self.vel = (p - self.last) / (self.n_since * DT)
                self.pos, self.last = p, p.copy()
                self.gap = self.n_since                       # age at the moment of acceptance (how long the gap was)
                if self.gap >= AGE_DROP:
                    self.reacq_k = self.k
                self.n_since = 0
                accepted = True
        return self._info(accepted)

    def _info(self, accepted):
        age = self.n_since
        if not self.c.active:
            ph = PH_STEADY
        elif age >= AGE_DROP:
            ph = PH_DROP
        elif self.k - self.reacq_k < REACQ:
            ph = PH_REACQ
        else:
            ph = PH_STEADY
        ball = np.concatenate([np.clip(self.pos, -POS_CLIP, POS_CLIP), np.clip(self.vel, -VEL_CLIP, VEL_CLIP)])
        return dict(ball=ball, acc=bool(accepted), age=int(age), gap=int(self.gap), phase=int(ph), tick=int(self.k),
                    in_burst=bool(self.in_burst(self.k)))


# ===================================================================================== tracker uncertainty set
ANGLES = np.arange(8) * (np.pi / 4)
DIRS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1)            # (8, 2)


def _fq(T, sa):
    """constant-velocity transition and white-noise-acceleration process noise for a step of T seconds (one axis)."""
    F = np.array([[1.0, T], [0.0, 1.0]])
    Q = sa * sa * np.array([[T ** 3 / 3.0, T * T / 2.0], [T * T / 2.0, T]])
    return F, Q


class TrackerSet:
    """P = per-axis covariance (pos, vel) at the CAPTURE time of the last accepted sample. Sigma_now = P propagated by
    (lat + age) ticks. x and y are independent with the same covariance."""

    def __init__(self, cond: Cond, sigma_a=1.0, vmax=0.6, lam=0.5, kappa=1.0, scale=1.0, p0_vel=0.5, sig_floor=1e-3):
        self.c = cond
        self.sa, self.vmax, self.lam, self.kappa, self.p0v, self.sf = sigma_a, vmax, lam, kappa, p0_vel, sig_floor
        self.scale = scale                                     # one overall factor, calibrated for coverage on VAL (tune.calibrate_scale)
        self.sm = max(cond.noise, sig_floor)
        self.period = int(np.ceil(50.0 / cond.rate - 1e-9))
        self.reset()

    def reset(self):
        self.P = np.diag([self.sm ** 2, self.p0v ** 2])
        return self

    def step(self, info):
        """feed the Corruptor's info of this tick. On an accepted sample: propagate P across the gap, then update."""
        if info["acc"] and info["tick"] > 0 and self.c.active:
            F, Q = _fq(max(info["gap"], 1) * DT, self.sa)
            P = F @ self.P @ F.T + Q
            S = P[0, 0] + self.sm ** 2
            Kg = P[:, 0] / S
            self.P = P - np.outer(Kg, P[0])
        return self

    def sigma_now(self, age):
        """(pos std, pos-vel cross ratio Sigma_vp / Sigma_pp) of the set at this tick."""
        delay = self.c.lat + int(age)
        F, Q = _fq(delay * DT, self.sa)
        S = F @ self.P @ F.T + Q
        return float(np.sqrt(max(S[0, 0], self.sm ** 2))), float(S[0, 1] / max(S[0, 0], 1e-12))

    def extra(self, age):
        """Minkowski term for latency and staleness: lam * v_max * dt * (latency + age beyond the normal sample period)."""
        return self.lam * self.vmax * DT * (self.c.lat + max(int(age) - self.period, 0))

    def radius(self, age, kappa=None):
        sp, _ = self.sigma_now(age)
        return self.scale * ((self.kappa if kappa is None else kappa) * sp + self.extra(age))

    def support(self, ball, age, kappa=None):
        """(9, 4): centre + 8 boundary points [x, y, vx, vy]; velocity shifted by the conditional ratio."""
        sp, ratio = self.sigma_now(age)
        r = self.radius(age, kappa)
        d = DIRS * r                                                       # (8, 2) position offsets
        out = np.tile(np.asarray(ball, np.float64), (9, 1))
        out[1:, :2] += d
        out[1:, 2:] += ratio * d
        out[:, :2] = np.clip(out[:, :2], -POS_CLIP, POS_CLIP)
        out[:, 2:] = np.clip(out[:, 2:], -VEL_CLIP, VEL_CLIP)
        return out


# ===================================================================================== stream helpers
def corrupt_prefix(cond: Cond, seed: int, burst, priv_prefix):
    """run a Corruptor over the TRUE ball of ticks 0..n-1 (priv rows: [x, y, z, vx, vy, vz, ...]).
    -> (ball (n, 4) float32, infos list, corruptor)."""
    cr = Corruptor(cond, seed, burst)
    p = np.asarray(priv_prefix, np.float64)
    infos = [cr.reset(p[0, 0:2], p[0, 3:5])]
    for k in range(1, len(p)):
        infos.append(cr.update(p[k, 0:2], p[k, 3:5]))
    return np.stack([i["ball"] for i in infos]).astype(np.float32), infos, cr
