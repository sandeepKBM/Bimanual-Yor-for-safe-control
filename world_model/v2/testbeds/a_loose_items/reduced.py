"""a_loose_items/reduced.py -- reduced analytic model of the dish + items (numpy, no MuJoCo).

State  : per item i, centre p_i (3) and velocity v_i (3) in the TRAY frame; tray: pos(3), v(3), quat(4) driven by the same
         kinematic integrator as the pipeline (bead_env.kin_step). Dimension 6N + 10 (N=10 -> 70).
Input  : u = [a(3), w(3)] held for one 50 Hz tick (4 sub-steps of 5 ms), the same as the 85-element fallback set.
Dynamics (items independent, NO item-item contact):
   a_eff  = R^T (g - a_o) - 2 w x v - w x (w x p)           (tray frame; w-dot = 0 within a tick)
   free flight: p'' = a_eff.
   floor contact (z <= r): v_z -> 0, z -> r (inelastic).
   wall contact  (s <= r, z < H): remove the inward-normal velocity component, push the centre back to s = r (inelastic).
   rolling without slip: while in contact, the tangential acceleration is scaled by kappa = 5/7 (solid sphere).
   Coulomb limit mu, rolling resistance and bounce are NOT modelled (hidden in MuJoCo: friction, dampratio, rolling).
Exit  : l_i < 0 with the same margin as tray_env (s < 0 above the rim height).
Parameters: geometry (RF, BETA, H, r) known from CAD; kappa, mu, restitution = hidden / assumed.
"""
from __future__ import annotations
import numpy as np
from world_model.v2.testbeds.a_loose_items.tray_env import kin_step, quat_axis, DT, STRIDE, TrayCfg

G = np.array([0.0, 0.0, -9.81])
KAPPA = 5.0 / 7.0


def rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Reduced:
    def __init__(self, cfg: TrayCfg, kappa=KAPPA):
        self.c = cfg; self.kappa = kappa
        b = np.deg2rad(cfg.beta_deg); self.tb, self.cb, self.sb = np.tan(b), np.cos(b), np.sin(b)

    def s_of(self, p):
        rad = np.hypot(p[:, 0], p[:, 1])
        return (self.c.rf + p[:, 2] * self.tb - rad) * self.cb, rad

    def margin_i(self, p):
        s, _ = self.s_of(p)
        return np.where(s < 0, s, np.maximum(self.c.h - p[:, 2], s))

    def step(self, p, v, quat, vt, a, w):
        """one 5 ms step; tray kinematics advanced by kin_step; p, v in tray frame (n,3)."""
        c = self.c
        R = rotmat(quat)
        wt = R.T @ w                                           # world-frame axis rate -> tray frame
        a_eff = (R.T @ (G - a))[None, :] - 2 * np.cross(wt, v) - np.cross(wt, np.cross(wt, p))
        # contact detection (pre-step): floor / wall
        s, rad = self.s_of(p)
        on_floor = (p[:, 2] <= c.item_r + 1e-4) & (rad < c.rf)
        on_wall = (s <= c.item_r + 1e-4) & (p[:, 2] < c.h + c.item_r) & (rad >= c.rf - 1e-9)
        ex = p[:, :2] / (rad[:, None] + 1e-12)
        n_w = np.stack([-self.cb * ex[:, 0], -self.cb * ex[:, 1], np.full(len(p), self.sb)], 1)
        n = np.where(on_wall[:, None], n_w, np.array([0, 0, 1.0])[None, :])
        contact = on_floor | on_wall
        an = (a_eff * n).sum(1)
        pressed = contact & (an < 0)                       # gravity-like push into the surface
        a_t = a_eff - an[:, None] * n
        acc = np.where(pressed[:, None], self.kappa * a_t, a_eff)
        v = v + acc * DT
        p = p + v * DT
        # constraint projection (inelastic)
        s, rad = self.s_of(p)
        fl = (p[:, 2] < c.item_r) & (rad < c.rf + c.item_r)
        p[fl, 2] = c.item_r; v[fl, 2] = np.maximum(v[fl, 2], 0.0)
        s, rad = self.s_of(p)
        ex = p[:, :2] / (rad[:, None] + 1e-12)
        n_w = np.stack([-self.cb * ex[:, 0], -self.cb * ex[:, 1], np.full(len(p), self.sb)], 1)
        wl = (s < c.item_r) & (s > -c.wall_t) & (p[:, 2] < c.h) & (rad >= c.rf - c.item_r)
        if wl.any():
            vn = (v[wl] * n_w[wl]).sum(1)
            v[wl] -= np.minimum(vn, 0.0)[:, None] * n_w[wl]
            p[wl] += ((c.item_r - s[wl]) / self.cb)[:, None] * np.stack([-ex[wl, 0], -ex[wl, 1], np.zeros(wl.sum())], 1) * 1.0
        return p, v

    def rollout(self, p0, v0, pos, quat, vt, acts, tmax=None):
        """p0, v0 (n,3) tray frame; tray (pos, quat, v); acts (T,6). Returns (spilled bool, tick of first exit or -1, min margin)."""
        p, v = p0.copy(), v0.copy(); pos = pos.copy(); vt = vt.copy(); quat = quat.copy()
        mn = np.inf
        for t, u in enumerate(acts if tmax is None else acts[:tmax]):
            for _ in range(STRIDE):
                p, v = self.step(p, v, quat, vt, u[:3], u[3:])
                pos, vt, quat, _ = kin_step(pos, vt, quat, u[:3], u[3:])
            li = self.margin_i(p)
            mn = min(mn, float(li.min()))
            if (li < 0).any():
                return True, t + 1, mn
        return False, -1, mn
