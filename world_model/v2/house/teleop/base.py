"""teleop/base.py -- mocap-base kinematics for the teleoperated YOR.  SIMULATION ONLY.

The base is kinematic (phase2): a commanded body twist u = (vx, vy, wz) is realised through a hardware slew limit
(|dv/dt| <= a_hw per axis), integrated at the physics rate.  Base acceleration reaches the arms/tray ONLY through the two tray
welds (same limitation as phase2: the mocap base gives the arms no fictitious force).  The cup's inertial acceleration for a cup at
base-frame position r_c is computed analytically (`cup_accel`) for the shield and for logging; the sim's own tray motion is the truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Assumed sim base limits (the real YOR base limits are NOT known to us -> ASSUMPTIONS, state them in every report)
V_STICK = np.array([0.9, 0.45, 1.2])      # full-stick twist (vx [m/s], vy [m/s], wz [rad/s])
A_HW = np.array([4.0, 4.0, 8.0])          # hardware slew (m/s^2, m/s^2, rad/s^2)


@dataclass
class BaseSim:
    x: float
    y: float
    yaw: float
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))       # body twist (vx, vy, wz) now

    def R(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

    def advance(self, u, dt, a_hw=A_HW, n_sub=4):
        """slew-limited step toward target twist u over dt; returns (poses[n_sub,3] after each substep, v_next, accel_body[3])."""
        u = np.asarray(u, float)
        v1 = self.v + np.clip(u - self.v, -a_hw * dt, a_hw * dt)
        h = dt / n_sub
        poses = np.zeros((n_sub, 3))
        for s in range(n_sub):
            vs = self.v + (v1 - self.v) * (s + 1) / n_sub
            vm = self.v + (v1 - self.v) * (s + 0.5) / n_sub
            ym = self.yaw + 0.5 * vm[2] * h
            c, sn = math.cos(ym), math.sin(ym)
            self.x += (c * vm[0] - sn * vm[1]) * h
            self.y += (sn * vm[0] + c * vm[1]) * h
            self.yaw += vm[2] * h
            poses[s] = (self.x, self.y, self.yaw)
        acc = (v1 - self.v) / dt
        self.v = v1
        return poses, v1, acc


def cup_accel_body(v, a, r_c):
    """acceleration of a point fixed at base-frame position r_c (2,) of a base with body twist v=(vx,vy,wz) and body accel
    a=(ax,ay,alpha), expressed in the (instantaneous) BASE frame:   a_b + alpha z x r + w z x (v_b + w z x r)."""
    w = v[2]
    wv = np.array([v[0] - w * r_c[1], v[1] + w * r_c[0]])            # v_b + w x r
    return np.array([a[0] - a[2] * r_c[1] - w * wv[1], a[1] + a[2] * r_c[0] + w * wv[0]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
