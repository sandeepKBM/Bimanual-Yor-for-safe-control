"""tower_env.py -- object-only MuJoCo rig: a stack of N blocks on a kinematically carried plate.
SIMULATION ONLY (offline, CPU). Mirrors world_model/v2/bead_rep/bead_env.py (no existing file is edited):
  * the plate is pinned kinematically each physics step (inertial carry, as contents.pin_cup);
  * u = [a (3), w (3)] from the 85-element fallback set is held for one WM tick = 4 physics steps (5 ms);
  * the plate update is bead_env.kin_step (imported, same action meaning, tilt limit 1.5 rad);
  * snapshot()/restore() of the full state (mj_getState INTEGRATION|WARMSTART + plate kinematics + bookkeeping).
Failure (sticky, checked once per tick) = any block (i) tilts more than TH_FAIL from the plate normal, or
(ii) has its centre outside the support face below it (plate for block 0, block below for the others),
(iii) is below the plate. Margin l = clip(min_i min(l_tilt, l_sup), -0.4, 1); l < 0 <=> failure event.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco
from numba import njit

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from world_model.v2.bead_rep import bead_env as B  # noqa: E402  (kin_step, fallback_U, brake_action; imported, not edited)

DT, STRIDE, DT_WM = B.DT, B.STRIDE, B.DT_WM
G = 9.81
TH_FAIL = 0.6          # rad, block tilt vs plate normal that counts as "toppled"  (ASSUMED threshold)
MARGIN_LO, MARGIN_HI = -0.4, 1.0
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION | mujoco.mjtState.mjSTATE_WARMSTART


# ---- numba twin of bead_env.kin_step (same action meaning; checked to ~1e-15 by selftest()) -------------
@njit(cache=True)
def _kin_step_nb(pos, v, quat, a, w, tilt_lim, dt):
    wq, x, y, z = quat[0], quat[1], quat[2], quat[3]
    c = np.array([2.0 * (x * z + wq * y), 2.0 * (y * z - wq * x), 1.0 - 2.0 * (x * x + y * y)])
    cn = c + dt * np.array([w[1] * c[2] - w[2] * c[1], w[2] * c[0] - w[0] * c[2], w[0] * c[1] - w[1] * c[0]])
    cn = cn / np.sqrt(cn @ cn)
    cz = min(max(cn[2], -1 + 1e-7), 1 - 1e-7)
    if np.arccos(cz) > tilt_lim:
        n = np.sqrt(cn[0] ** 2 + cn[1] ** 2) + 1e-12
        cn = np.array([cn[0] / n * np.sin(tilt_lim), cn[1] / n * np.sin(tilt_lim), np.cos(tilt_lim)])
    d = c @ cn
    cr = np.array([c[1] * cn[2] - c[2] * cn[1], c[2] * cn[0] - c[0] * cn[2], c[0] * cn[1] - c[1] * cn[0]])
    q = np.array([1.0 + d, cr[0], cr[1], cr[2]])
    q = q / np.sqrt(q @ q)
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = quat
    qn = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    qn = qn / np.sqrt(qn @ qn)
    vn = v + a * dt
    return pos + vn * dt, vn, qn


def selftest(n=200, seed=0):
    """max |fast - bead_env.kin_step| over random states/actions incl. tilt-clamp region."""
    r = np.random.default_rng(seed)
    err = 0.0
    for _ in range(n):
        q = r.normal(size=4); q /= np.linalg.norm(q)
        # keep tilt <~1.6 so the clamp branch is exercised near 1.5
        pos, v = r.normal(size=3), r.normal(size=3)
        a, w = r.normal(size=3) * 3, r.normal(size=3)
        p1, v1, q1, _ = B.kin_step(pos, v, q, a, w)
        p2, v2, q2 = _kin_step_nb(pos, v, q, a, w, B.TILT_LIM, DT)
        err = max(err, np.abs(p1 - p2).max(), np.abs(v1 - v2).max(), np.abs(q1 - q2).max())
    return err


@dataclass
class TowerCfg:
    n_blocks: int = 3
    half: tuple = (0.025, 0.025, 0.02)        # block half-extents [m] (5 x 5 x 4 cm)       ASSUMED
    block_mass: float = 0.06                  # kg                                          ASSUMED
    mu: float = 0.5                           # block-block and block-plate sliding friction ASSUMED, hidden
    plate_half: tuple = (0.15, 0.10, 0.005)   # 30 x 20 cm plate                            ASSUMED
    offset_std: float = 0.003                 # initial lateral block offset [m] (alignment error)
    solref: tuple = (0.01, 1.0)               # 2*dt, as the pipeline's cup scene (0.004 clamps to 0.01)
    noslip: int = 0
    timestep: float = DT
    plate_mass: float = 5.0            # heavy + gravcomp: the pinned plate must not react to contact or fall inside a step
    plate_gravcomp: float = 1.0


def build_model(c: TowerCfg, home=(0.0, 0.0, 1.0)):
    hx, hy, hz = c.half
    ph = c.plate_half
    bl = []
    for i in range(c.n_blocks):
        z = home[2] + ph[2] + hz + i * 2 * hz + 1e-4 * (i + 1)
        bl.append(f'<body name="blk{i}" pos="{home[0]} {home[1]} {z}"><freejoint/>'
                  f'<geom type="box" size="{hx - 3e-4 * i} {hy - 2.5e-4 * i} {hz}" mass="{c.block_mass}" rgba="0.8 0.5 0.2 1"/></body>')
    xml = f"""<mujoco><option integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81"
 timestep="{c.timestep}" noslip_iterations="{c.noslip}"/>
<default><geom friction="{c.mu} 0.005 0.0001" condim="3" solref="{c.solref[0]} {c.solref[1]}"/></default>
<worldbody><body name="plate" pos="{home[0]} {home[1]} {home[2]}" gravcomp="{c.plate_gravcomp}"><freejoint/>
<geom type="box" size="{ph[0]} {ph[1]} {ph[2]}" mass="{c.plate_mass}" rgba="0.5 0.5 0.5 1"/></body>
{''.join(bl)}</worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def quat_mat(q):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def tilt_between(Rp, Rb):
    """angle between plate z axis and block z axis."""
    return float(np.arccos(np.clip(Rp[:, 2] @ Rb[:, 2], -1.0, 1.0)))


@dataclass
class Snap:
    state: np.ndarray
    prev: np.ndarray
    has_prev: int
    kin: np.ndarray
    k: int
    failed: int
    fail_tick: int
    tick: int


class TowerEnv:
    def __init__(self, cfg: TowerCfg | None = None, home=(0.0, 0.0, 1.0), seed=0):
        self.cfg = cfg or TowerCfg()
        self.home = np.asarray(home, float)
        self.m = build_model(self.cfg, home)
        self.d = mujoco.MjData(self.m)
        self.pq = int(self.m.jnt_qposadr[self.m.body_jntadr[self.m.body("plate").id]])
        self.pv = int(self.m.jnt_dofadr[self.m.body_jntadr[self.m.body("plate").id]])
        self.bq = [int(self.m.jnt_qposadr[self.m.body_jntadr[self.m.body(f"blk{i}").id]]) for i in range(self.cfg.n_blocks)]
        self.n_state = int(mujoco.mj_stateSize(self.m, SPEC))
        self.rng = np.random.default_rng(seed)
        self.reset(self.home, np.array([1.0, 0, 0, 0]))

    # ---- state ---------------------------------------------------------------------------
    def set_kin(self, pos, quat, v):
        self.pos, self.quat, self.v = np.array(pos, float), np.array(quat, float), np.array(v, float)

    def reset(self, pos0, quat0, v0=None, offsets=None):
        c = self.cfg
        mujoco.mj_resetData(self.m, self.d)
        Rp = quat_mat(quat0)
        if offsets is None:
            offsets = self.rng.normal(0, c.offset_std, (c.n_blocks, 2))
        self.offsets0 = np.asarray(offsets)
        self.d.qpos[self.pq:self.pq + 3] = pos0
        self.d.qpos[self.pq + 3:self.pq + 7] = quat0
        hz = c.half[2]
        for i in range(c.n_blocks):
            loc = np.array([offsets[i, 0], offsets[i, 1], c.plate_half[2] + hz + i * 2 * hz + 1e-4 * (i + 1)])
            self.d.qpos[self.bq[i]:self.bq[i] + 3] = np.asarray(pos0) + Rp @ loc
            self.d.qpos[self.bq[i] + 3:self.bq[i] + 7] = quat0
        self.prev = None
        self.k = 0
        self.tick_n = 0
        self.failed = 0
        self.fail_tick = -1
        self.set_kin(pos0, quat0, np.zeros(3) if v0 is None else v0)

    def snapshot(self) -> Snap:
        sv = np.zeros(self.n_state)
        mujoco.mj_getState(self.m, self.d, sv, SPEC)
        prev = self.prev.copy() if self.prev is not None else np.zeros(self.m.nq)
        return Snap(sv, prev, int(self.prev is not None), np.concatenate([self.pos, self.quat, self.v]),
                    self.k, self.failed, self.fail_tick, self.tick_n)

    def restore(self, s: Snap):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, s.state, SPEC)
        self.prev = s.prev.copy() if s.has_prev else None
        self.pos, self.quat, self.v = s.kin[:3].copy(), s.kin[3:7].copy(), s.kin[7:10].copy()
        self.k, self.failed, self.fail_tick, self.tick_n = s.k, s.failed, s.fail_tick, s.tick

    # ---- observation / margin --------------------------------------------------------------
    def block_pose_plate_frame(self):
        """block centres in the plate frame (n,3) and tilts vs plate normal (n,)."""
        q = self.d.qpos
        Rp = quat_mat(self.quat)
        idx = np.array(self.bq)
        pos = np.stack([q[idx], q[idx + 1], q[idx + 2]], 1)
        qb = np.stack([q[idx + 3], q[idx + 4], q[idx + 5], q[idx + 6]], 1)
        w, x, y, z = qb.T
        zax = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
        T = np.arccos(np.clip(zax @ Rp[:, 2], -1.0, 1.0))
        return (pos - self.pos) @ Rp, T

    def margin(self):
        """continuous signed margin, > 0 safe. Returns (l, parts dict)."""
        c = self.cfg
        P, T = self.block_pose_plate_frame()
        ph, hx, hy = c.plate_half, c.half[0], c.half[1]
        l_tilt = (TH_FAIL - T) / TH_FAIL
        l_sup = np.empty(c.n_blocks)
        for i in range(c.n_blocks):
            if i == 0:   # support = plate top face (block centre vs plate edge)
                e = np.minimum(ph[0] - abs(P[0, 0]), ph[1] - abs(P[0, 1])) / ph[1]
            else:        # support = block below (centre of i within face of i-1)
                dxy = P[i, :2] - P[i - 1, :2]
                e = min(hx - abs(dxy[0]), hy - abs(dxy[1])) / min(hx, hy)
            l_sup[i] = e
        l_below = (P[:, 2] + 0.02) / 0.05      # below the plate
        l = float(np.clip(min(l_tilt.min(), l_sup.min(), l_below.min()), MARGIN_LO, MARGIN_HI))
        return l, dict(tilt=T, sup=l_sup, P=P)

    # ---- dynamics --------------------------------------------------------------------------
    def _pin(self):
        d = self.d
        d.qpos[self.pq:self.pq + 3] = self.pos
        d.qpos[self.pq + 3:self.pq + 7] = self.quat
        if self.prev is None:
            d.qvel[self.pv:self.pv + 6] = 0.0
        else:
            qv = np.zeros(self.m.nv)
            mujoco.mj_differentiatePos(self.m, qv, DT, self.prev, d.qpos)
            d.qvel[self.pv:self.pv + 6] = qv[self.pv:self.pv + 6]

    def phys_step(self):
        self._pin()
        self.prev = self.d.qpos.copy()
        mujoco.mj_step(self.m, self.d)
        self.k += 1

    def tick(self, u, check=True):
        """one WM tick under held u=(a[3], w[3]); returns (l, failed)."""
        u = np.asarray(u, float)
        for _ in range(STRIDE):
            self.phys_step()
            self.pos, self.v, self.quat = _kin_step_nb(self.pos, self.v, self.quat, u[:3], u[3:], B.TILT_LIM, DT)
        self.tick_n += 1
        l, _ = self.margin() if check else (1.0, None)
        if check and l < 0 and not self.failed:
            self.failed, self.fail_tick = 1, self.tick_n
        return l, self.failed

    def axis(self):
        return B.quat_axis(self.quat)


def fallback_U():
    return B.fallback_U()


def brake(env: TowerEnv):
    return B.brake_action(env.v, env.axis())
