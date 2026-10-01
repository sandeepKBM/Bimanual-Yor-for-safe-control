"""bead_rep/bead_env.py -- Phase 0: cup-with-beads MuJoCo rig wrapper for the latent-HJ replication.
SIMULATION ONLY (MuJoCo, offline). Edits NO existing file; reuses world_model/v2/data/contents.py
(rig_model, Oracle, pin_cup) and wm_audit/common.py (fallback_U, pendulum labeller semantics).

What this file provides
  * `kin_step`      one 5 ms kinematic cup update with EXACTLY the pendulum labeller's action meaning
                    (wm_audit/common.pend_rollout_torch): u = (a[3], w[3]) = (cup-ORIGIN linear accel,
                    commanded axis angular velocity); axis c <- normalize(c + dt w x c), then the tilt projection
                    (arccos(c_z) > tilt_lim -> c = [h_xy sin(lim), cos(lim)]); a is the cup-origin accel
                    (semi-implicit Euler: v += a dt; p += v dt  =>  backward 2nd difference of p == a exactly).
                    The quaternion follows the axis by the MINIMAL rotation c -> c_new (no spin about the axis,
                    like the labeller's realized rate w_real = c x c_new / dt).
  * `bead_margin`   continuous bead margin of DESIGN Q2 (clip(min_i l_i, -0.02, 0.05)/0.05 in [-0.4, 1]).
  * `BeadEnv`       cup-only rig (8 nominal beads), kinematically pinned cup + inertial carry, tick(u) at the
                    50 Hz WM rate (4 physics steps = 20 ms), snapshot()/restore() of the FULL state, tunnel guard.
  * `Snapshot`      flat-array (de)serialization of that state (for the sparse snapshots saved by gen_nominal).

State / label timing convention (== contents.run_rig, so replay labels match the existing oracle):
  "state at sim step k" = cup kinematic pose K_k (pos, quat, v) known, beads after k mj_steps, K_k NOT yet pinned.
  one physics step = pin K_k -> exit check (cup frame, pose k vs beads after k steps) -> mj_step -> K_{k+1} = kin_step.
  The margin / spill label at step k is computed in exactly that state.

Tunnel guard (DESIGN risks 3): the cup is kinematic, so above ~2.4 m/s beads can pass THROUGH the 1.5 mm wall /
3 mm floor (rig.py artifact). Each ball's FIRST exit is classified from its cup-frame z at detection:
  z > H - RIM_EXIT_TOL (0.068)  -> 'rim' exit  (a genuine spill, status 1, sticky)
  otherwise (side wall / floor)  -> 'tunnel'    (artifact, status 2, sticky): the ball is EXCLUDED from the
                                    margin and the spill count, and `tunnel` is reported so callers can drop the
                                    episode / branch. Margin==oracle consistency (l<0 <=> a ball out) holds for
                                    the non-tunnelled set.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.data import contents as C  # noqa: E402  (also puts eval/ slosh/ on sys.path)
from world_model.v2.wm_audit import common as WC  # noqa: E402

# ---- constants shared with the pendulum labeller -------------------------------------------------
DT = C.DT                       # 0.005 s physics step
STRIDE = WC.STRIDE              # 4 physics steps per WM tick
DT_WM = WC.DT_WM                # 0.02 s
P0 = WC.P0                      # PendParams(): A_max 5, W_max 2, tilt_lim 1.5, h = 0.04 (only the limits are used)
A_MAX, W_MAX, TILT_LIM = P0.A_max, P0.W_max, P0.tilt_lim
H, R = C.CUP_HEIGHT, C.CUP_RADIUS
RIM_TOL = C.RIM_EXIT_TOL
V_WALL_MAX = 1.6                # experiments/slosh/rig.py: speed above which kinematic-wall tunnelling appears
MARGIN_LO, MARGIN_HI = -0.02, 0.05      # [m]; margin = clip(min_i l_i, LO, HI) / HI in [-0.4, 1]
T_BRANCH = 100                  # ticks per counterfactual branch (2 s)

ST_IN, ST_RIM, ST_TUNNEL = 0, 1, 2      # per-ball first-exit status


def fallback_U():
    """The 85-element fallback set (17 horizontal accels x 5 tilt rates) == wm_audit.common.fallback_U()."""
    return WC.fallback_U()


# ============================================================================================ kinematics
def quat_axis(q):
    """cup axis (z column of R(q)), analytic (== mju_quat2Mat[:, 2])."""
    w, x, y, z = q
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _min_rot(c, c_new):
    """quaternion of the minimal rotation taking unit c to unit c_new (world frame)."""
    d = float(c @ c_new)
    if d < -1.0 + 1e-12:                                  # antiparallel: any perpendicular axis (never hit: steps are tiny)
        ax = np.cross(c, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(c, [0.0, 1.0, 0.0])
        ax /= np.linalg.norm(ax)
        return np.concatenate([[0.0], ax])
    q = np.concatenate([[1.0 + d], np.cross(c, c_new)])
    return q / np.linalg.norm(q)


def project_tilt(c, tilt_lim=TILT_LIM):
    """same projection as common.pend_rollout_torch (project=True)."""
    cz = min(max(c[2], -1 + 1e-7), 1 - 1e-7)
    if np.arccos(cz) > tilt_lim:
        hxy = c[:2] / (np.linalg.norm(c[:2]) + 1e-12)
        return np.array([hxy[0] * np.sin(tilt_lim), hxy[1] * np.sin(tilt_lim), np.cos(tilt_lim)])
    return c


def kin_step(pos, v, quat, a, w, project=True, tilt_lim=TILT_LIM, dt=DT):
    """One 5 ms cup update under the held WM-rate action (a, w) (see module doc). Returns pos', v', quat', c_new."""
    c = quat_axis(quat)
    c_new = c + dt * np.cross(w, c)
    c_new = c_new / np.linalg.norm(c_new)
    if project:
        c_new = project_tilt(c_new, tilt_lim)
    q_new = _qmul(_min_rot(c, c_new), quat)
    q_new /= np.linalg.norm(q_new)
    v_new = v + a * dt
    return pos + v_new * dt, v_new, q_new, c_new


def kin_tick(pos, v, quat, u, **kw):
    """4 x kin_step with the action held (one WM tick). Returns pos, v, quat, c_start, c_end, w_eff (mean axis rate)."""
    c0 = quat_axis(quat)
    for _ in range(STRIDE):
        pos, v, quat, _c = kin_step(pos, v, quat, u[:3], u[3:], **kw)
    c1 = quat_axis(quat)
    return pos, v, quat, c0, c1, np.cross(c0, c1) / DT_WM


# ============================================================================================ margin
def bead_margin_i(loc):
    """per-ball l_i in the cup frame (DESIGN Q2): H - z inside the radius, min(H - z, 0) - (rad - R) outside."""
    rad = np.hypot(loc[:, 0], loc[:, 1])
    z = loc[:, 2]
    return np.where(rad <= R, H - z, np.minimum(H - z, 0.0) - (rad - R))


def bead_margin(loc, excl=None):
    """l = clip(min_i l_i, -0.02, 0.05)/0.05 in [-0.4, 1], over balls not excluded (tunnelled)."""
    li = bead_margin_i(loc)
    if excl is not None:
        li = li[~excl]
        if li.size == 0:
            return 1.0
    return float(np.clip(li.min(), MARGIN_LO, MARGIN_HI) / MARGIN_HI)


# ============================================================================================ snapshot
@dataclass
class Snapshot:
    state: np.ndarray       # mj_getState(INTEGRATION | WARMSTART)
    prev: np.ndarray        # qpos right after the last pin (inertial-carry finite difference); zeros if has_prev == 0
    has_prev: int
    kin: np.ndarray         # pos(3) quat(4) v(3)
    status: np.ndarray      # (n,) int8 per-ball first-exit status
    exit_step: np.ndarray   # (n,) int32
    k: int                  # sim step index since reset

    def arrays(self):
        return dict(state=self.state, prev=self.prev, has_prev=np.int8(self.has_prev), kin=self.kin,
                    status=self.status, exit_step=self.exit_step, k=np.int32(self.k))

    @staticmethod
    def from_arrays(state, prev, has_prev, kin, status, exit_step, k):
        return Snapshot(np.array(state, np.float64), np.array(prev, np.float64), int(has_prev), np.array(kin, np.float64),
                        np.array(status, np.int8), np.array(exit_step, np.int32), int(k))


# ============================================================================================ environment
class BeadEnv:
    """Cup-only rig with `contents` (default: the 8 nominal beads) and a kinematically driven cup."""

    SPEC = mujoco.mjtState.mjSTATE_INTEGRATION | mujoco.mjtState.mjSTATE_WARMSTART

    def __init__(self, contents: "C.Contents | None" = None, cup_home=(0.0, 0.0, 1.0)):
        self.c = contents if contents is not None else C.Contents(**C.NOMINAL)
        self.cup_home = np.asarray(cup_home, float)
        self.m = C.rig_model(self.c, cup_home=tuple(cup_home))
        self.d = mujoco.MjData(self.m)
        cup = self.m.body("cup").id
        cj = self.m.body_jntadr[cup]
        self.cq, self.cv = int(self.m.jnt_qposadr[cj]), int(self.m.jnt_dofadr[cj])
        self.orc = C.Oracle(self.m, self.c.n_balls)
        self.n = self.c.n_balls
        self.n_state = int(mujoco.mj_stateSize(self.m, self.SPEC))
        self.set_kin(self.cup_home, np.array([1.0, 0, 0, 0]), np.zeros(3))
        self.reset(self.cup_home, np.array([1.0, 0, 0, 0]))

    # ---------------------------------------------------------------- state
    def set_kin(self, pos, quat, v):
        self.pos = np.array(pos, np.float64)
        self.quat = np.array(quat, np.float64)
        self.v = np.array(v, np.float64)

    def reset(self, pos0, quat0, v0=None):
        """fresh rig; beads translated (not rotated) with the cup's first pose, as contents.run_rig."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.orc.idx] += np.tile(np.asarray(pos0, float) - self.cup_home, self.n)
        self.prev = None
        self.k = 0
        self.status = np.zeros(self.n, np.int8)
        self.exit_step = -np.ones(self.n, np.int32)
        self.exit_z = np.full(self.n, np.nan)
        self.max_speed = 0.0
        self.set_kin(pos0, quat0, np.zeros(3) if v0 is None else v0)

    def snapshot(self) -> Snapshot:
        sv = np.zeros(self.n_state)
        mujoco.mj_getState(self.m, self.d, sv, self.SPEC)
        prev = self.prev.copy() if self.prev is not None else np.zeros(self.m.nq)
        return Snapshot(sv, prev, int(self.prev is not None), np.concatenate([self.pos, self.quat, self.v]),
                        self.status.copy(), self.exit_step.copy(), self.k)

    def restore(self, s: Snapshot):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, s.state, self.SPEC)
        self.prev = s.prev.copy() if s.has_prev else None
        self.pos, self.quat, self.v = s.kin[:3].copy(), s.kin[3:7].copy(), s.kin[7:10].copy()
        self.status = s.status.copy()
        self.exit_step = s.exit_step.copy()
        self.exit_z = np.full(self.n, np.nan)
        self.k = s.k
        self.max_speed = 0.0

    # ---------------------------------------------------------------- observation helpers
    def local(self):
        """bead centres in the cup frame at the CURRENT kinematic pose (n,3)."""
        return self.orc.local(self.d, self.pos, self.quat)

    def _exit_check(self, loc):
        out = C.Oracle.out_mask(loc)
        new = out & (self.status == ST_IN)
        if new.any():
            self.exit_step[new] = self.k
            self.exit_z[new] = loc[new, 2]
            self.status[new] = np.where(loc[new, 2] > H - RIM_TOL, ST_RIM, ST_TUNNEL)
        return out

    @property
    def n_spilled(self):
        """sticky count of RIM exits (genuine spills); tunnelled balls excluded."""
        return int((self.status == ST_RIM).sum())

    @property
    def tunnel(self):
        return bool((self.status == ST_TUNNEL).any())

    def margin(self, loc=None):
        loc = self.local() if loc is None else loc
        return bead_margin(loc, self.status == ST_TUNNEL)

    def axis(self):
        return quat_axis(self.quat)

    def bead_features(self, loc=None, clip=0.3):
        """privileged aux features: bead centres in the cup frame (n,3, clipped to +-clip m so spilled balls that fall
        away stay finite), centroid of the balls still inside the cup (3; zeros if none), #inside."""
        loc = self.local() if loc is None else loc
        inside = ~C.Oracle.out_mask(loc)
        cen = loc[inside].mean(0) if inside.any() else np.zeros(3)
        return np.clip(loc, -clip, clip), cen, int(inside.sum())

    def wm_obs(self, w_obs):
        """o = [axis (3), cup lin vel (3), axis ang vel (3)] (closed_loop.py semantics: v is the post-tick kinematic
        velocity == v + DT_WM*a, w_obs is the realized mean axis rate over the tick)."""
        return np.concatenate([self.axis(), self.v, w_obs])

    # ---------------------------------------------------------------- dynamics
    def phys_step(self):
        """pin the kinematic pose K_k, run the exit check (pose k vs beads after k steps), mj_step. -> loc (pre-step)."""
        C.pin_cup(self.m, self.d, self.cq, self.cv, self.pos, self.quat, self.prev)
        self.prev = self.d.qpos.copy()
        loc = self.orc.local(self.d, self.pos, self.quat)
        self._exit_check(loc)
        mujoco.mj_step(self.m, self.d)
        self.k += 1
        return loc

    def tick(self, u):
        """One WM tick under the held action u = (a[3], w[3]) from the 85-element set (any |a|,|w| is accepted).
        Returns dict(obs (9,), act (6,) = [a, w_eff], l, n_spilled, tunnel, speed)."""
        u = np.asarray(u, np.float64)
        c0 = self.axis()
        for _ in range(STRIDE):
            self.phys_step()
            self.pos, self.v, self.quat, _ = kin_step(self.pos, self.v, self.quat, u[:3], u[3:])
            self.max_speed = max(self.max_speed, float(np.linalg.norm(self.v[:2])))
        c1 = self.axis()
        w_obs = np.cross(c0, c1) / DT_WM
        loc = self.local()
        self._exit_check(loc)                       # state at step k (pose k, beads after k steps) == replay convention
        return dict(obs=self.wm_obs(w_obs), act=np.concatenate([u[:3], w_obs]), l=self.margin(loc),
                    n_spilled=self.n_spilled, tunnel=self.tunnel, speed=float(np.linalg.norm(self.v[:2])))


# ============================================================================================ non-fallback-set helper
def brake_action(v, c, a_max=A_MAX, w_max=W_MAX):
    """'retreat/decel' action (continuous, within the same |a| <= A_MAX, |w| <= W_MAX limits and a_z = w_z = 0):
    decelerate the horizontal velocity to 0 in <= 1 tick without overshoot and rotate the axis back to vertical."""
    sp = float(np.hypot(v[0], v[1]))
    a = np.zeros(3)
    if sp > 1e-9:
        a[:2] = -v[:2] / sp * min(a_max, sp / DT_WM)
    w = np.zeros(3)
    wh = np.cross(c, [0.0, 0.0, 1.0])
    nh = float(np.linalg.norm(wh))
    if nh > 1e-9:
        tilt = float(np.arcsin(min(1.0, nh)))
        w = wh / nh * min(w_max, tilt / DT_WM)
    return np.concatenate([a, w])

