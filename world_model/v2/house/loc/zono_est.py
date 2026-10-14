"""loc/zono_est.py -- run the zonotope BASE filter (zono_rows/base_filter.py) on the ESTIMATED base pose.  SIMULATION ONLY.

The filter's points are read from MjData (true world).  A robot that localizes with an error places its body points in the map frame at
    p_map = T_est T_true^{-1} p_true     (rigid in the xy plane: relative geometry of the robot is exact, only its placement in the map is off)
and the perceived zonotopes / obstacles live in the map frame.  `attach(bf, get_true_pose)` wraps BaseFilter.points so the arm / tray points
(and their arm-induced velocities) are expressed in the estimated frame; the footprint points are already built from the (estimated) origin/yaw the
caller passes to bf.step.  Arm zonotope ROWS (rows.py, FL-MPC) are NOT re-framed (they keep the true pose): only the base CBF sees the pose error.
"""
from __future__ import annotations

import math

import numpy as np

from .estimator import wrap


def attach(bf, get_true_pose):
    orig = bf.points

    def points(d, origin, yaw):
        P, r, V, nf = orig(d, origin, yaw)
        tp = get_true_pose()
        dyaw = wrap(yaw - tp[2])
        c, s = math.cos(dyaw), math.sin(dyaw)
        K = len(P) - nf
        rel = P[:K, :2] - np.asarray(tp[:2])
        P[:K, 0] = origin[0] + c * rel[:, 0] - s * rel[:, 1]
        P[:K, 1] = origin[1] + s * rel[:, 0] + c * rel[:, 1]
        vx, vy = V[:K, 0].copy(), V[:K, 1].copy()
        V[:K, 0] = c * vx - s * vy
        V[:K, 1] = s * vx + c * vy
        return P, r, V, nf
    bf.points = points
    return bf
