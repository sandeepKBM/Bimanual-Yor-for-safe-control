"""tray_env.py -- object-only MuJoCo rig: a round payload (ball / tomato proxy) on a rim-less flat tray or a dustpan
(back lip). SIMULATION ONLY, CPU only, no rendering. Mirrors world_model/v2/bead_rep/bead_env.py:
  * the tray is moved KINEMATICALLY with bead_env.kin_step (same 6-D action u = [a(3), w(3)], same 85-element set),
  * the margin is continuous (signed distance of the ball centre to the tray edge, scaled),
  * snapshot()/restore() cover the full state (bitwise replay).
Nothing in the repo is edited; everything is imported.

Frame: tray frame has its origin on the tray top surface at the tray centre, z = tray normal (= `axis`).
Ball centre in the tray frame: rho = (x, y, z); at rest on the tray z = radius.
Dustpan: a back lip wall at x = -HX (height LIP_H). Only the other three sides are open.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.bead_rep import bead_env as B  # noqa: E402  (kin_step, quat_axis, fallback_U, brake_action)
from world_model.v2.data import contents as C  # noqa: E402  (pin_cup)

DT, STRIDE, DT_WM = B.DT, B.STRIDE, B.DT_WM
G = 9.81
HX, HY = 0.15, 0.10            # tray half sizes [m]  (30 x 20 cm; ASSUMED size of a bimanual tray)
TRAY_T = 0.003                 # tray half thickness
LIP_H, LIP_T = 0.02, 0.004     # dustpan back lip height / half thickness
M_LO, M_HI = -0.04, 0.08       # margin clip [m]; l = clip(d, M_LO, M_HI) / M_HI in [-0.5, 1]
ST_OK, ST_OFF, ST_TUNNEL = 0, 1, 2
SOLREF = "0.01 1"              # contact stiffness: 2*dt time constant (default 0.02 sank the ball ~5 mm)


# ---- fast numba twin of bead_env.kin_step (reference = bead_env.kin_step; equality tested in selftest.py) -------------
import numba as nb  # noqa: E402


@nb.njit(cache=True)
def _kin_step_nb(pos, v, q, a, w, tilt_lim, dt):
    # axis (z column of R(q))
    c = np.array([2.0 * (q[1] * q[3] + q[0] * q[2]), 2.0 * (q[2] * q[3] - q[0] * q[1]),
                  1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])])
    cn = c + dt * np.cross(w, c)
    cn = cn / np.sqrt(cn @ cn)
    cz = min(max(cn[2], -1 + 1e-7), 1 - 1e-7)
    if np.arccos(cz) > tilt_lim:
        hn = np.sqrt(cn[0] * cn[0] + cn[1] * cn[1]) + 1e-12
        cn = np.array([cn[0] / hn * np.sin(tilt_lim), cn[1] / hn * np.sin(tilt_lim), np.cos(tilt_lim)])
    d = c @ cn
    if d < -1.0 + 1e-12:
        ax = np.cross(c, np.array([1.0, 0.0, 0.0]))
        if np.sqrt(ax @ ax) < 1e-6:
            ax = np.cross(c, np.array([0.0, 1.0, 0.0]))
        ax = ax / np.sqrt(ax @ ax)
        qr = np.array([0.0, ax[0], ax[1], ax[2]])
    else:
        cr = np.cross(c, cn)
        qr = np.array([1.0 + d, cr[0], cr[1], cr[2]])
        qr = qr / np.sqrt(qr @ qr)
    w1, x1, y1, z1 = qr
    w2, x2, y2, z2 = q
    qn = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    qn = qn / np.sqrt(qn @ qn)
    vn = v + a * dt
    return pos + vn * dt, vn, qn


def kin_step_fast(pos, v, quat, a, w, tilt_lim=B.TILT_LIM, dt=DT):
    return _kin_step_nb(pos, v, quat, np.ascontiguousarray(a, np.float64), np.ascontiguousarray(w, np.float64),
                        tilt_lim, dt)


@dataclass
class BallParams:
    """Ball (tomato proxy). `inertia_k` = I/(m r^2): 0.4 solid sphere, 0.667 thin shell."""
    radius: float = 0.03
    mass: float = 0.10
    inertia_k: float = 0.4
    mu: float = 0.4            # sliding friction
    mu_roll: float = 0.002     # rolling friction coefficient [m]  (torque = mu_roll * N)
    mu_tors: float = 0.002     # torsional


def tray_xml(p: BallParams, lip: bool, home=(0.0, 0.0, 1.0)):
    r = p.radius
    inertia = p.inertia_k * p.mass * r * r
    fr = f"{p.mu:.5f} {p.mu_tors:.5f} {p.mu_roll:.5f}"
    lip_xml = (f'<geom name="lip" type="box" size="{LIP_T} {HY} {LIP_H/2}" pos="{-HX+LIP_T:.5f} 0 {LIP_H/2:.5f}" '
               f'condim="6" friction="{fr}" solref="{SOLREF}"/>') if lip else ""
    return f"""<mujoco model="tray_ball">
  <option integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81" timestep="0.005"/>
  <size memory="16M"/>
  <worldbody>
    <body name="tray" pos="{home[0]} {home[1]} {home[2]}">
      <freejoint name="tray_free"/>
      <inertial pos="0 0 0" mass="0.3" diaginertia="2e-3 2e-3 3e-3"/>
      <geom name="tray_top" type="box" size="{HX} {HY} {TRAY_T}" pos="0 0 {-TRAY_T}" condim="6" friction="{fr}" solref="{SOLREF}"/>
      {lip_xml}
    </body>
    <body name="ball" pos="{home[0]} {home[1]} {home[2] + r + 0.0005:.5f}">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="{p.mass}" diaginertia="{inertia:.6e} {inertia:.6e} {inertia:.6e}"/>
      <geom name="ball_g" type="sphere" size="{r}" condim="6" friction="{fr}" solref="{SOLREF}"/>
    </body>
  </worldbody>
</mujoco>"""


def sdf_rect(x, y, lip: bool):
    """signed distance of the ball centre (tray frame) to the open tray edge, inside > 0.
    Dustpan: the back side (x = -HX) is closed by the lip and is ignored."""
    ex = HX - x if lip else HX - abs(x)          # distance to the open x edge(s)
    ey = HY - abs(y)
    if lip:
        ox, oy = max(x - HX, 0.0), max(abs(y) - HY, 0.0)
    else:
        ox, oy = max(abs(x) - HX, 0.0), max(abs(y) - HY, 0.0)
    if ex >= 0 and ey >= 0:
        return min(ex, ey)
    return -float(np.hypot(ox, oy))


def margin_from_d(d):
    return float(np.clip(d, M_LO, M_HI) / M_HI)


@dataclass
class Snap:
    state: np.ndarray
    prev: np.ndarray
    has_prev: int
    kin: np.ndarray
    status: int
    k: int


class TrayEnv:
    SPEC = mujoco.mjtState.mjSTATE_INTEGRATION | mujoco.mjtState.mjSTATE_WARMSTART

    def __init__(self, p: BallParams | None = None, lip: bool = False, home=(0.0, 0.0, 1.0)):
        self.p = p or BallParams()
        self.lip = lip
        self.home = np.asarray(home, float)
        self.m = mujoco.MjModel.from_xml_string(tray_xml(self.p, lip, home))
        self.d = mujoco.MjData(self.m)
        tb, bb = self.m.body("tray").id, self.m.body("ball").id
        j = self.m.body_jntadr[tb]
        self.tq, self.tv = int(self.m.jnt_qposadr[j]), int(self.m.jnt_dofadr[j])
        j = self.m.body_jntadr[bb]
        self.bq, self.bv = int(self.m.jnt_qposadr[j]), int(self.m.jnt_dofadr[j])
        self.n_state = int(mujoco.mj_stateSize(self.m, self.SPEC))
        self.reset(self.home, np.array([1.0, 0, 0, 0]))

    # ------------------------------------------------------------------ state
    def reset(self, pos0, quat0, v0=None, ball_xy=(0.0, 0.0)):
        """ball placed at tray-frame (x, y) resting on the (possibly tilted) tray, zero relative velocity."""
        mujoco.mj_resetData(self.m, self.d)
        self.pos, self.quat = np.array(pos0, float), np.array(quat0, float)
        self.v = np.zeros(3) if v0 is None else np.array(v0, float)
        R = self._R(self.quat)
        loc = np.array([ball_xy[0], ball_xy[1], self.p.radius + 0.0005])
        self.d.qpos[self.bq:self.bq + 3] = self.pos + R @ loc
        self.d.qpos[self.bq + 3:self.bq + 7] = [1, 0, 0, 0]
        self.d.qvel[self.bv:self.bv + 3] = self.v                # ball co-moves with the tray
        self.prev, self.k, self.status = None, 0, ST_OK
        self.fail_tick = -1

    @staticmethod
    def _R(q):
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, np.asarray(q, float))
        return R.reshape(3, 3)

    def snapshot(self) -> Snap:
        sv = np.zeros(self.n_state)
        mujoco.mj_getState(self.m, self.d, sv, self.SPEC)
        prev = self.prev.copy() if self.prev is not None else np.zeros(self.m.nq)
        return Snap(sv, prev, int(self.prev is not None), np.concatenate([self.pos, self.quat, self.v]),
                    self.status, self.k)

    def restore(self, s: Snap):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, s.state, self.SPEC)
        self.prev = s.prev.copy() if s.has_prev else None
        self.pos, self.quat, self.v = s.kin[:3].copy(), s.kin[3:7].copy(), s.kin[7:10].copy()
        self.status, self.k, self.fail_tick = s.status, s.k, -1

    # ------------------------------------------------------------------ observation
    def axis(self):
        return B.quat_axis(self.quat)

    def local(self):
        R = self._R(self.quat)
        return (self.d.qpos[self.bq:self.bq + 3] - self.pos) @ R

    def ball_state_tray(self):
        """ball (x, y, vx, vy) in the TRAY frame at the current tray pose (relative velocity incl. the tray's rotation
        is NOT subtracted: this is the world velocity expressed in tray axes minus tray lin. velocity v)."""
        R = self._R(self.quat)
        vb = self.d.qvel[self.bv:self.bv + 3]
        loc = (self.d.qpos[self.bq:self.bq + 3] - self.pos) @ R
        return loc, (vb - self.v) @ R

    def dist(self, loc=None):
        loc = self.local() if loc is None else loc
        return sdf_rect(loc[0], loc[1], self.lip)

    def margin(self, loc=None):
        return margin_from_d(self.dist(loc))

    def _check(self, loc):
        if self.status != ST_OK:
            return
        d = sdf_rect(loc[0], loc[1], self.lip)
        if d < 0.0:
            self.status, self.fail_tick = ST_OFF, self.k
        elif loc[2] < self.p.radius - 0.02:                   # sank through the tray: artifact
            self.status = ST_TUNNEL

    # ------------------------------------------------------------------ dynamics
    def phys_step(self):
        C.pin_cup(self.m, self.d, self.tq, self.tv, self.pos, self.quat, self.prev)
        self.prev = self.d.qpos.copy()
        self._check(self.local())
        mujoco.mj_step(self.m, self.d)
        self.k += 1

    def tick(self, u):
        """One 50 Hz WM tick (4 physics steps) under the held u = (a[3], w[3]). Returns dict."""
        u = np.asarray(u, np.float64)
        c0 = self.axis()
        for _ in range(STRIDE):
            self.phys_step()
            self.pos, self.v, self.quat = kin_step_fast(self.pos, self.v, self.quat, u[:3], u[3:])
        c1 = self.axis()
        w_obs = np.cross(c0, c1) / DT_WM
        loc = self.local()
        self._check(loc)
        return dict(obs=np.concatenate([c1, self.v, w_obs]), l=self.margin(loc), d=self.dist(loc),
                    failed=self.status == ST_OFF, tunnel=self.status == ST_TUNNEL, loc=loc)
