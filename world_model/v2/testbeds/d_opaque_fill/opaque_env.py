"""opaque_env.py -- object-only MuJoCo rig for the "opaque container with hidden fill" testbed.
SIMULATION ONLY. Mirrors world_model/v2/bead_rep/bead_env.py (read-only import: kin_step, kin_tick, quat_axis,
fallback_U, brake_action) but with its own container model, margin, fill generator and wrench logger.

Container: open cylinder, inner radius R_IN=0.030, rim height H=0.120 (floor top at FLOOR_Z=0.020, thick to stop grain tunnelling), shell 0.15 kg.
Fill: n grains of radius 6 mm (hex-packed layers, then settled). Hidden: n, friction mu, drag (liquid proxy).
The container is pinned kinematically (contents.pin_cup, inertial carry), like the bead rig.
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.bead_rep import bead_env as B          # noqa: E402  (read-only reuse)
from world_model.v2.data import contents as C              # noqa: E402
from world_model.v2.latent_hj import slosh_pend as sp      # noqa: E402

DT, STRIDE, DT_WM = B.DT, B.STRIDE, B.DT_WM
R_IN, WALL, H, FLOOR_Z = 0.030, 0.004, 0.120, 0.020
R_OUT = R_IN + WALL
SHELL_MASS = 0.15
GR_R, GR_RHO = 0.006, 1400.0           # grain radius, grain density (bulk ~ 0.6*1400 = 840 kg/m3, rice-like)
G = 9.81
RIM_TOL = 0.012
MARGIN_LO, MARGIN_HI = -0.02, 0.05
ST_IN, ST_RIM, ST_TUNNEL = 0, 1, 2
H_TALL = 0.32                            # walls are compiled tall so the loose hex layout can settle; shrunk to H afterwards
FREE_H = H - FLOOR_Z                    # 0.100 m of inner height


@dataclass
class Fill:
    n: int = 60                         # number of grains (hidden)
    mu: float = 0.5                     # grain friction (hidden): 0.5 rice-like, 0.03 liquid proxy
    drag: float = 0.0                   # [1/s] viscous drag rel. to container (liquid proxy), hidden

    @property
    def mass(self):
        return self.n * GR_RHO * 4.0 / 3.0 * math.pi * GR_R ** 3


def layout(n):
    """hex-packed layers (grain centres in the container frame), filled bottom-up."""
    s = 2 * GR_R + 0.0008
    rc = R_IN - GR_R - 0.0005
    cand = [(i * s + (j % 2) * s / 2, j * s * math.sqrt(3) / 2) for j in range(-8, 9) for i in range(-8, 9)]
    disk = [p for p in cand if math.hypot(*p) <= rc]
    pts, z, layer = [], FLOOR_Z + GR_R + 0.0015, 0
    while len(pts) < n:
        off = (s / 2, s * math.sqrt(3) / 6) if layer % 2 else (0.0, 0.0)
        lp = [(x + off[0], y + off[1]) for x, y in disk if math.hypot(x + off[0], y + off[1]) <= rc]
        for x, y in lp[:n - len(pts)]:
            pts.append((x, y, z))
        layer += 1
        z += 2 * GR_R * 0.92 + 0.0008
    return np.round(np.array(pts), 5)


def fill_height(n):
    """settled-fill top surface height above the inner floor (approx from layout; settling compacts it slightly)."""
    p = layout(n)
    return float(p[:, 2].max() + GR_R - FLOOR_Z)


PACK = 0.57                              # random-close packing of 6 mm grains in a 60 mm bore (measured by settling, see README)


def n_for_fraction(f):
    """number of grains whose SETTLED height is f of the inner height (volume balance, packing fraction PACK)."""
    A = math.pi * (R_IN - 0.0005) ** 2
    return int(round(f * FREE_H * A * PACK / (4.0 / 3.0 * math.pi * GR_R ** 3)))


def build_model(fill: Fill):
    opt = re.search(r"<option[^>]*/>", C.ROBOT.read_text()).group(0)
    segs = 12
    rw = R_IN + WALL
    hw = rw * math.tan(math.pi / segs) + 0.0012
    walls = []
    for k in range(segs):
        a = 2 * math.pi * k / segs
        walls.append(f'<geom type="box" size="{WALL:.4f} {hw:.5f} {H_TALL / 2:.4f}" pos="{rw * math.cos(a):.5f} {rw * math.sin(a):.5f} {H_TALL / 2:.4f}" '
                     f'euler="0 0 {math.degrees(a):.4f}" friction="0.6 0.01 0.0001" condim="3" solref="0.01 1"/>')
    s = SHELL_MASS
    shell = (f'<body name="cup" pos="0 0 1.0"><freejoint name="cup_free"/>'
             f'<inertial pos="0 0 0.05" mass="{s}" diaginertia="{s * (3 * R_OUT**2 + H**2) / 12:.3e} {s * (3 * R_OUT**2 + H**2) / 12:.3e} {s * R_OUT**2 / 2:.3e}"/>'
             f'<geom type="cylinder" size="{R_OUT:.4f} {FLOOR_Z / 2:.4f}" pos="0 0 {FLOOR_Z / 2:.4f}" friction="0.6 0.01 0.0001" condim="3" solref="0.01 1"/>'
             + "".join(walls) + '</body>')
    lay = layout(fill.n)
    grains = "\n".join(
        f'<body name="ball_{i}" pos="{p[0]:.5f} {p[1]:.5f} {1.0 + p[2]:.5f}"><freejoint name="ball_{i}_free"/>'
        f'<geom name="ball_{i}" type="sphere" size="{GR_R}" density="{GR_RHO}" friction="{fill.mu:.4f} 0.001 0.0001" '
        f'condim="3" priority="1" solref="0.01 1"/></body>' for i, p in enumerate(lay))
    xml = f'<mujoco model="opaque_rig">{opt}<size memory="64M"/><worldbody>{shell}{grains}</worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml)
    # speed: dense Newton costs O(nv^3) with ~6 dof per grain (3.1 ms/step at 47 grains vs 1.3 ms with CG)
    m.opt.solver, m.opt.iterations, m.opt.cone, m.opt.impratio = 1, 50, 1, 1.0     # CG, elliptic (pyramidal CG is unstable at mu<0.2)
    return m


def margin_i(loc):
    rad = np.hypot(loc[:, 0], loc[:, 1])
    z = loc[:, 2]
    return np.where(rad <= R_OUT, H - z, np.minimum(H - z, 0.0) - (rad - R_OUT))


def margin(loc, excl=None):
    li = margin_i(loc)
    if excl is not None:
        li = li[~excl]
        if li.size == 0:
            return 1.0
    return float(np.clip(li.min(), MARGIN_LO, MARGIN_HI) / MARGIN_HI)


def out_mask(loc):
    return (np.hypot(loc[:, 0], loc[:, 1]) > R_OUT) | (loc[:, 2] > H) | (loc[:, 2] < -0.002)


class OpaqueEnv:
    SPEC = mujoco.mjtState.mjSTATE_INTEGRATION | mujoco.mjtState.mjSTATE_WARMSTART

    def __init__(self, fill: Fill, settle_steps=200):
        self.fill = fill
        self.m = build_model(fill)
        self.d = mujoco.MjData(self.m)
        cj = self.m.body_jntadr[self.m.body("cup").id]
        self.cq, self.cv = int(self.m.jnt_qposadr[cj]), int(self.m.jnt_dofadr[cj])
        self.n = fill.n
        bids = [self.m.body(f"ball_{i}").id for i in range(self.n)]
        self.bids = np.array(bids)
        self.idx = np.array([self.m.jnt_qposadr[self.m.body_jntadr[b]] for b in bids])[:, None] + np.arange(3)[None]
        self.idx = self.idx.ravel()
        self.vidx = (np.array([self.m.jnt_dofadr[self.m.body_jntadr[b]] for b in bids])[:, None] + np.arange(3)[None]).ravel()
        self.bmass = self.m.body_mass[self.bids]
        self.n_state = int(mujoco.mj_stateSize(self.m, self.SPEC))
        self.home = np.array([0.0, 0.0, 1.0])
        self.wall_g = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.m.body("cup").id and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        self._walls(H_TALL)
        self.reset(self.home, np.array([1.0, 0, 0, 0]))
        for _ in range(settle_steps):              # static settle inside the tall walls
            self.phys_step(log=False)
        self._walls(H)
        mujoco.mj_forward(self.m, self.d)
        self.status[:] = 0                          # exits flagged while the loose layout fell in are not spills
        self.exit_step[:] = -1
        self.settled_out = int(out_mask(self.local()).sum())   # must be 0 (sanity)
        self._settled = self.snapshot()
        self.top = float(self.local()[:, 2].max() + GR_R - FLOOR_Z)     # settled fill height above the inner floor [m]

    # -------------------------------------------------------------- state
    def reset(self, pos0, quat0, v0=None):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.idx] += np.tile(np.asarray(pos0, float) - self.home, self.n)
        self.prev = None
        self.k = 0
        self.status = np.zeros(self.n, np.int8)
        self.exit_step = -np.ones(self.n, np.int32)
        self.pos, self.quat = np.array(pos0, float), np.array(quat0, float)
        self.v = np.zeros(3) if v0 is None else np.array(v0, float)
        self.max_speed = 0.0

    def _walls(self, h):
        for g in self.wall_g:
            self.m.geom_size[g, 2] = h / 2
            self.m.geom_pos[g, 2] = h / 2

    def reset_settled(self, pos0=None, quat0=(1.0, 0, 0, 0)):
        """restore the settled fill; optionally translate container+grains to pos0 (upright)."""
        self.restore(self._settled)
        if pos0 is not None:
            self.d.qpos[self.idx] += np.tile(np.asarray(pos0, float) - self.home, self.n)
            self.set_kin(pos0, quat0, np.zeros(3))
            self.prev = None

    def snapshot(self):
        sv = np.zeros(self.n_state)
        mujoco.mj_getState(self.m, self.d, sv, self.SPEC)
        prev = self.prev.copy() if self.prev is not None else np.zeros(self.m.nq)
        return (sv, prev, int(self.prev is not None), np.concatenate([self.pos, self.quat, self.v]),
                self.status.copy(), self.exit_step.copy(), self.k)

    def restore(self, s):
        sv, prev, hp, kin, st, ex, k = s
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, sv, self.SPEC)
        self.prev = prev.copy() if hp else None
        self.pos, self.quat, self.v = kin[:3].copy(), kin[3:7].copy(), kin[7:10].copy()
        self.status, self.exit_step, self.k = st.copy(), ex.copy(), k
        self.max_speed = 0.0

    def set_kin(self, pos, quat, v):
        self.pos, self.quat, self.v = np.array(pos, float), np.array(quat, float), np.array(v, float)

    # -------------------------------------------------------------- helpers
    def local(self):
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, self.quat)
        return (self.d.qpos[self.idx].reshape(self.n, 3) - self.pos) @ R.reshape(3, 3)

    def axis(self):
        return B.quat_axis(self.quat)

    @property
    def n_spilled(self):
        return int((self.status == ST_RIM).sum())

    @property
    def tunnel(self):
        return bool((self.status == ST_TUNNEL).any())

    def _exit_check(self, loc):
        out = out_mask(loc)
        new = out & (self.status == ST_IN)
        if new.any():
            self.exit_step[new] = self.k
            self.status[new] = np.where(loc[new, 2] > H - RIM_TOL, ST_RIM, ST_TUNNEL)

    def phys_step(self, log=False):
        """pin pose K_k -> exit check -> mj_step.  If log: returns (F, M) exerted BY the grains ON the container
        (world frame, M about the container origin), from momentum balance on the grains."""
        C.pin_cup(self.m, self.d, self.cq, self.cv, self.pos, self.quat, self.prev)
        self.prev = self.d.qpos.copy()
        loc = self.local()
        self._exit_check(loc)
        fext = None
        if self.fill.drag > 0:
            vrel = self.d.qvel[self.vidx].reshape(-1, 3) - self.d.qvel[self.cv:self.cv + 3]
            fext = -self.fill.drag * self.bmass[:, None] * vrel
            self.d.xfrc_applied[self.bids, :3] = fext
        v0 = self.d.qvel[self.vidx].reshape(-1, 3).copy() if log else None
        mujoco.mj_step(self.m, self.d)
        self.k += 1
        if not log:
            return loc, None
        v1 = self.d.qvel[self.vidx].reshape(-1, 3)
        acc = (v1 - v0) / DT
        gv = np.array([0, 0, -G])
        N = self.bmass[:, None] * (acc - gv)                 # contact force ON grains from container ...
        if fext is not None:
            N = N - fext                                      # ... after removing the (relative) drag force
        Fc = -N                                               # force ON container
        rel = self.d.qpos[self.idx].reshape(self.n, 3) - self.pos
        return loc, (Fc.sum(0), np.cross(rel, Fc).sum(0))

    def tick(self, u, log=False):
        """one WM tick under u = (a[3], w[3]). Returns dict(l, n_spilled, tunnel, speed, [F, M])."""
        u = np.asarray(u, float)
        FM = []
        for _ in range(STRIDE):
            _, w = self.phys_step(log)
            if log:
                FM.append(w)
            self.pos, self.v, self.quat, _ = B.kin_step(self.pos, self.v, self.quat, u[:3], u[3:])
            self.max_speed = max(self.max_speed, float(np.linalg.norm(self.v[:2])))
        loc = self.local()
        self._exit_check(loc)
        out = dict(l=margin(loc, self.status == ST_TUNNEL), n_spilled=self.n_spilled, tunnel=self.tunnel,
                   speed=float(np.linalg.norm(self.v[:2])))
        if log:
            out["F"] = np.mean([f for f, _ in FM], 0)
            out["M"] = np.mean([m for _, m in FM], 0)
        return out


# ====================================================================== analytic reduced model (liquid / pendulum)
class PendModel:
    """equivalent-mechanical first slosh mode of a cylindrical tank (Ibrahim), same equations as slosh_pend.py /
    wm_audit.common.pend_rollout_torch, but with THIS container's R, H and an arbitrary fill height h (hidden).
    state (n, nd, c): bob direction, rate, container axis.  spill <=> phi >= phi* = atan((H_free - h)/R)."""

    def __init__(self, h, zeta=0.02, R=R_IN, Hfree=FREE_H, tilt_lim=B.TILT_LIM):
        self.h, self.zeta, self.R, self.Hf, self.tilt_lim = h, zeta, R, Hfree, tilt_lim
        self.w1 = math.sqrt(G * sp.XI / R * math.tanh(sp.XI * h / R))
        self.L = G / self.w1 ** 2
        self.phi_star = math.atan((Hfree - h) / R)

    def margin(self, n, c):
        phi = math.acos(float(np.clip(-n @ c, -1, 1)))
        return float(np.clip((self.phi_star - phi) / self.phi_star, -1.0, 1.0))

    def tick(self, n, nd, c, u):
        """one WM tick (4 x 5 ms; 4 sub-steps each) with the pendulum-labeller conventions."""
        a, w = u[:3], u[3:]
        c2 = 2 * self.zeta * self.w1
        hs = DT / 4
        g = np.array([0, 0, -G])
        for _ in range(STRIDE):
            c_new = c + DT * np.cross(w, c)
            c_new /= np.linalg.norm(c_new)
            c_new = B.project_tilt(c_new, self.tilt_lim)
            w_real = np.cross(c, c_new) / DT
            a_piv = a + self.h * np.cross(w_real, np.cross(w_real, c_new))
            for _ in range(4):
                f = g - a_piv
                f = f - n * (n @ f)
                ndd = f / self.L - (nd @ nd) * n - c2 * (nd - np.cross(w_real, n))
                nd = nd + hs * ndd
                n = n + hs * nd
                n /= np.linalg.norm(n)
                nd = nd - n * (n @ nd)
            c = c_new
        return n, nd, c
