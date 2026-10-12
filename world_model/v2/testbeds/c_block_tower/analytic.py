"""analytic.py -- reduced analytic model of an N-block stack on the carried plate (rigid-column, quasi-static).
Inputs: plate kinematics (pos, v, quat) + OBSERVED block centres in the plate frame (what a ZED would give) + u sequence.
For each tick and each sub-stack j (blocks j..N-1, equal mass, COM r_j in plate frame):
  inertial accel of the COM  a_c = a + w x (w x rho)            (alpha ~ 0: u is piecewise constant)
  support force per unit mass f = a_c + g e_z  (world), f_n = f . n_plate, f_t = tangential part
  slide margin  s_j = (mu f_n - |f_t|) / (mu g)                  (Coulomb, interface j)
  tip margin    t_j = (hs - ||zmp_j - c_{j-1}||_inf) / hs,  zmp_j = r_j,xy - (f_t/f_n) h_j   (ZMP-like, h_j = COM height above interface)
Quasi-static: there are NO block states -- it cannot represent slip history, rocking dynamics or post-failure motion.
Predicted failure = margin < 0 for >= K consecutive ticks (K tuned on MuJoCo; K=1 is the pure static test).
State dim (given observed offsets): plate (pos 3, v 3, axis 2 = 8) + frozen offsets 2N. Parameters: mu (hidden), N, geometry.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.testbeds.c_block_tower import tower_env as T

G = 9.81


def _R(q):
    return T.quat_mat(q)


def branch_margins(pos, v, quat, P, cfg: T.TowerCfg, mu, seq=None, tail="zero", horizon=100):
    """returns (slide (H,), tip (H,)) = min over sub-stacks of the margins at each tick."""
    n = cfg.n_blocks
    hz, hs_b = cfg.half[2], min(cfg.half[0], cfg.half[1])
    ph = cfg.plate_half
    slide = np.empty(horizon)
    tip = np.empty(horizon)
    pos, v, quat = np.array(pos, float), np.array(v, float), np.array(quat, float)
    z_if = [ph[2] + 2 * hz * j for j in range(n)]               # nominal interface heights (plate frame)
    for t in range(horizon):
        if seq is not None and t < len(seq):
            u = seq[t]
        elif tail == "brake":
            u = T.B.brake_action(v, T.B.quat_axis(quat))
        else:
            u = np.zeros(6)
        a, w = u[:3], u[3:]
        R = _R(quat)
        s_m, t_m = 9.0, 9.0
        for j in range(n):
            r = P[j:].mean(0)                                   # sub-stack COM, plate frame
            rho = R @ r
            a_c = a + np.cross(w, np.cross(w, rho))
            f = R.T @ (a_c + np.array([0, 0, G]))
            fn, ft = f[2], f[:2]
            if fn <= 1e-6:
                s_m, t_m = -1.0, -1.0
                break
            s_m = min(s_m, (mu * fn - np.linalg.norm(ft)) / (mu * G))
            h = r[2] - z_if[j]
            zmp = r[:2] - ft / fn * h
            c0 = np.zeros(2) if j == 0 else P[j - 1, :2]
            hs = ph[1] if j == 0 else hs_b
            t_m = min(t_m, (hs - np.abs(zmp - c0).max()) / hs)
        slide[t], tip[t] = s_m, t_m
        for _ in range(T.STRIDE):
            pos, v, quat = T._kin_step_nb(pos, v, quat, a, w, T.B.TILT_LIM, T.DT)
    return slide, tip


def predict_fail(slide, tip, K):
    """first tick at which min(slide, tip) < 0 held for K consecutive ticks, else -1."""
    bad = (np.minimum(slide, tip) < 0).astype(int)
    if K <= 1:
        i = np.where(bad)[0]
        return int(i[0]) if len(i) else -1
    run = np.convolve(bad, np.ones(K, int), "full")[:len(bad)]
    i = np.where(run >= K)[0]
    return int(i[0] - K + 1) if len(i) else -1
