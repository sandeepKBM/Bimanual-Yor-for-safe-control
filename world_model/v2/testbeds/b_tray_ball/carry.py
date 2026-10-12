"""carry.py -- documented SYNTHETIC nominal carry distribution (the spill_v2 cup trajectories cannot be replayed: they pour,
tilt past 90 deg, and 10% of ticks have tilt > 98 deg; see report). Everything stays inside the pipeline's action domain
(|a_h| <= 5 m/s^2, |w_xy| <= 2 rad/s, tilt <= 1.5 rad).

Carry = n_seg min-jerk transports of the tray centre (random direction, distance D ~ U(d_lo, d_hi), duration T ~ U(t_lo, t_hi),
rests of 0.1-0.3 s). A noisy "anti-roll" tilt controller tries to align the tray normal c with the effective-gravity normal
c* = normalize([kappa a_x, kappa a_y, g]) (kappa = 1 is ideal compensation, ball at rest on the tray):
    w = clip( (c x c*) / tau + OU noise , +-W_MAX )  per axis.
kappa ~ U(kappa_lo, kappa_hi) models imperfect compensation, tau the lag of the arm tilt loop, OU noise = tracking jitter.
"intensity" scales (d_lo, d_hi) and shortens T. Parameters are ASSUMED (a stand-in for a teleoperator / planner), not fitted.
"""
from __future__ import annotations

import numpy as np

G = 9.81
W_MAX = 2.0


def mj_acc(s):
    return 60 * s - 180 * s ** 2 + 120 * s ** 3


def make_carry(rng, intensity=1.0, n_seg=3):
    segs, t = [], 0.0
    for _ in range(n_seg):
        ang = rng.uniform(0, 2 * np.pi)
        D = rng.uniform(0.15, 0.45) * intensity
        T = rng.uniform(1.0, 2.0) / np.sqrt(intensity)
        segs.append((t, T, D * np.array([np.cos(ang), np.sin(ang), 0.0])))
        t += T + rng.uniform(0.1, 0.3)
    return dict(segs=segs, T_end=t + 0.5, kappa=rng.uniform(0.6, 1.1), tau=rng.uniform(0.12, 0.25),
                noise=rng.uniform(0.0, 0.25), ou=np.zeros(2), rng=np.random.default_rng(rng.integers(1 << 31)))


def carry_u(carry, t, c):
    """action u (6,) at time t for the current tray axis c."""
    a = np.zeros(3)
    for (t0, T, Dv) in carry["segs"]:
        if t0 <= t < t0 + T:
            a = Dv / T ** 2 * mj_acc((t - t0) / T)
    a = np.clip(a, -5.0, 5.0)
    ct = np.array([carry["kappa"] * a[0], carry["kappa"] * a[1], G])
    ct /= np.linalg.norm(ct)
    w = np.cross(c, ct) / carry["tau"]
    carry["ou"] = 0.9 * carry["ou"] + carry["noise"] * carry["rng"].normal(size=2) * np.sqrt(1 - 0.81)
    w[:2] = np.clip(w[:2] + carry["ou"], -W_MAX, W_MAX)
    w[2] = 0.0
    return np.concatenate([a, w])
