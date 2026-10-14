"""a_loose_items/tray_env.py -- object-only MuJoCo rig: a kinematic DISH (low-rim open container) with N loose
candy-like spheres. SIMULATION ONLY, CPU only. Mirrors world_model/v2/bead_rep/bead_env.py (imports its kinematic
integrator, Snapshot, brake_action and fallback_U unchanged; edits nothing).

Container = flat floor disc (radius RF) + N_SEG inclined wall boxes. Wall angle BETA is measured from vertical:
BETA=0 is a straight-walled TRAY, BETA=45 deg is a shallow DISH/BOWL. Rim lip is at height H above the floor top.

Margin (per item i, tray frame, centre p_i):  l_i = s_i if s_i < 0 else max(H - z_i, s_i),
  s_i = (RF + z_i tan(BETA) - rad_i) cos(BETA)   (signed horizontal-ish distance to the wall plane; >0 inside)
  l_i < 0  <=>  centre above the rim height AND beyond the (extended) wall plane  == the item has crossed the lip.
Rig margin  l = clip(min_i l_i, MARGIN_LO, MARGIN_HI) / MARGIN_HI  (hard min); `softmin_margin` is the LSE version.

Contact settings copied from contents.Contents defaults (condim 6, torsional 0.005, rolling 3e-4, solref 0.004/1,
solimp .9/.95/.001), option copied from yor_robot_derived.mjcf (implicitfast, elliptic cone, impratio 10, dt 5 ms).
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.bead_rep import bead_env as BE          # noqa: E402  (imported, never edited)
from world_model.v2.data import contents as C                # noqa: E402

DT, STRIDE, DT_WM = BE.DT, BE.STRIDE, BE.DT_WM
kin_step, quat_axis, brake_action, fallback_U, Snapshot = BE.kin_step, BE.quat_axis, BE.brake_action, BE.fallback_U, BE.Snapshot
ST_IN, ST_RIM, ST_TUNNEL = 0, 1, 2
OPT = '<option integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81" timestep="0.005" />'


@dataclass
class TrayCfg:
    n_items: int = 10
    item_r: float = 0.007          # Skittle ~ 14 mm x 7 mm lens; sphere of r=7 mm (ASSUMED analogue)
    density: float = 900.0
    friction: float = 0.4          # hidden
    dampratio: float = 1.0         # hidden (bounce)
    rf: float = 0.06               # floor radius
    beta_deg: float = 45.0         # wall angle from vertical (0 = straight tray)
    h: float = 0.025               # rim lip height above floor top
    wall_t: float = 0.008          # wall thickness (thick = no tunnelling at dt 5 ms)
    n_seg: int = 16
    rolling: float = 3e-4
    torsional: float = 0.005
    margin_hi: float = 0.025       # margin scale [m]
    margin_lo: float = -0.02


def tray_xml(c: TrayCfg, home=(0, 0, 1.0), start_xy=None):
    b = np.deg2rad(c.beta_deg)
    L = c.h / np.cos(b)
    w = (c.rf + c.h * np.tan(b) + c.wall_t) * np.tan(np.pi / c.n_seg) * 1.15
    geoms = [f'<geom name="floor" type="cylinder" size="{c.rf + c.wall_t:.5f} 0.004" pos="0 0 -0.004" '
             f'friction="0.5 0.005 0.0001" solref="0.004 1" solimp="0.9 0.95 0.001" rgba="0.8 0.2 0.2 1"/>']
    for i in range(c.n_seg):
        phi = 2 * np.pi * i / c.n_seg
        xdir = np.array([np.cos(b), 0.0, -np.sin(b)])          # outward normal of the wall (local frame, before phi)
        mid = np.array([c.rf + 0.5 * c.h * np.tan(b), 0.0, 0.5 * c.h]) + 0.5 * c.wall_t * xdir
        # quaternion Rz(phi) Ry(b)
        qy = np.array([np.cos(b / 2), 0, np.sin(b / 2), 0])
        qz = np.array([np.cos(phi / 2), 0, 0, np.sin(phi / 2)])
        q = BE._qmul(qz, qy)
        cp, sp = np.cos(phi), np.sin(phi)
        pos = np.array([cp * mid[0] - sp * mid[1], sp * mid[0] + cp * mid[1], mid[2]])
        geoms.append(f'<geom name="wall{i}" type="box" size="{c.wall_t / 2:.5f} {w:.5f} {L / 2:.5f}" '
                     f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
                     f'friction="0.5 0.005 0.0001" solref="0.004 1" solimp="0.9 0.95 0.001"/>')
    items = []
    for i in range(c.n_items):
        items.append(f'<body name="item_{i}" pos="{home[0]:.4f} {home[1]:.4f} {home[2] + 0.02 + 0.02 * i:.4f}">'
                     f'<freejoint name="item_{i}_free"/>'
                     f'<geom name="item_{i}" type="sphere" size="{c.item_r}" density="{c.density}" '
                     f'friction="{c.friction} {c.torsional} {c.rolling}" condim="6" priority="1" '
                     f'solref="0.004 {c.dampratio}" solimp="0.9 0.95 0.001" rgba="0.9 0.5 0.1 1"/></body>')
    xml = f"""<mujoco model="tray_rig">
  {OPT}
  <size memory="64M"/>
  <worldbody>
    <body name="tray" pos="{home[0]} {home[1]} {home[2]}">
      <freejoint name="tray_free"/>
      <inertial pos="0 0 0.01" mass="0.2" diaginertia="1e-3 1e-3 1e-3"/>
      {chr(10).join(geoms)}
    </body>
    {chr(10).join(items)}
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


class TrayEnv:
    SPEC = BE.BeadEnv.SPEC

    def __init__(self, cfg: TrayCfg | None = None, home=(0.0, 0.0, 1.0)):
        self.cfg = cfg or TrayCfg()
        c = self.cfg
        self.home = np.asarray(home, float)
        self.m = tray_xml(c, tuple(home))
        self.d = mujoco.MjData(self.m)
        tb = self.m.body("tray").id
        tj = self.m.body_jntadr[tb]
        self.cq, self.cv = int(self.m.jnt_qposadr[tj]), int(self.m.jnt_dofadr[tj])
        self.n = c.n_items
        self.bq = np.array([int(self.m.jnt_qposadr[self.m.body_jntadr[self.m.body(f"item_{i}").id]]) for i in range(self.n)])
        self.idx = (self.bq[:, None] + np.arange(3)[None]).ravel()
        self.n_state = int(mujoco.mj_stateSize(self.m, self.SPEC))
        self.tb = np.tan(np.deg2rad(c.beta_deg)); self.cb = np.cos(np.deg2rad(c.beta_deg))
        self.reset(0)

    # ------------------------------------------------------------ margin / geometry (cup-frame analogue)
    def local(self):
        R = np.empty(9); mujoco.mju_quat2Mat(R, self.quat); R = R.reshape(3, 3)
        return (self.d.qpos[self.idx].reshape(self.n, 3) - self.pos) @ R

    def li(self, loc):
        c = self.cfg
        rad = np.hypot(loc[:, 0], loc[:, 1]); z = loc[:, 2]
        s = (c.rf + z * self.tb - rad) * self.cb
        return np.where(s < 0, s, np.maximum(c.h - z, s))

    def _scale(self, v):
        c = self.cfg
        return float(np.clip(v, c.margin_lo, c.margin_hi) / c.margin_hi)

    def margin(self, loc=None, excl=None):
        loc = self.local() if loc is None else loc
        li = self.li(loc)
        if excl is not None:
            li = li[~excl]
            if li.size == 0:
                return 1.0
        return self._scale(li.min())

    def softmin_margin(self, loc=None, tau=0.004):
        """log-sum-exp soft minimum over items (tau in metres), then the same clip/scale."""
        li = self.li(self.local() if loc is None else loc)
        mn = li.min()
        return self._scale(mn - tau * np.log(np.exp(-(li - mn) / tau).sum()))

    # ------------------------------------------------------------ state
    def set_kin(self, pos, quat, v):
        self.pos, self.quat, self.v = np.array(pos, float), np.array(quat, float), np.array(v, float)

    def reset(self, seed, settle=150):
        """random non-overlapping drop layout inside the dish, then settle with the tray static."""
        c = self.cfg
        mujoco.mj_resetData(self.m, self.d)
        rng = np.random.default_rng(seed)
        pts = []
        while len(pts) < self.n:
            r = (c.rf - 1.5 * c.item_r) * np.sqrt(rng.uniform()); a = rng.uniform(0, 2 * np.pi)
            p = np.array([r * np.cos(a), r * np.sin(a)])
            if all(np.hypot(*(p - q)) > 2.2 * c.item_r for q in pts) or rng.uniform() < 0.01 * len(pts):
                pts.append(p)
        for i, p in enumerate(pts):
            self.d.qpos[self.bq[i]:self.bq[i] + 3] = self.home + np.array([p[0], p[1], c.item_r + 0.003 + 0.0 * i])
        self.prev, self.k = None, 0
        self.status = np.zeros(self.n, np.int8)
        self.exit_step = -np.ones(self.n, np.int32)
        self.set_kin(self.home, [1.0, 0, 0, 0], np.zeros(3))
        for _ in range(settle):
            self.phys_step()
        self.status[:] = 0; self.exit_step[:] = -1
        self.k = 0

    def snapshot(self) -> Snapshot:
        sv = np.zeros(self.n_state); mujoco.mj_getState(self.m, self.d, sv, self.SPEC)
        prev = self.prev.copy() if self.prev is not None else np.zeros(self.m.nq)
        return Snapshot(sv, prev, int(self.prev is not None), np.concatenate([self.pos, self.quat, self.v]),
                        self.status.copy(), self.exit_step.copy(), self.k)

    def restore(self, s: Snapshot):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, s.state, self.SPEC)
        self.prev = s.prev.copy() if s.has_prev else None
        self.pos, self.quat, self.v = s.kin[:3].copy(), s.kin[3:7].copy(), s.kin[7:10].copy()
        self.status, self.exit_step, self.k = s.status.copy(), s.exit_step.copy(), s.k

    # ------------------------------------------------------------ dynamics
    def _exit_check(self, loc):
        """first exit: s<0 with the centre within 12 mm of the rim height or above -> RIM (genuine spill, sticky);
        s<0 lower down (through the wall) or below the floor -> TUNNEL (artifact, excluded from margin and count)."""
        c = self.cfg
        li = self.li(loc)
        new = ((li < 0) | (loc[:, 2] < -0.012)) & (self.status == ST_IN)
        if new.any():
            self.exit_step[new] = self.k
            self.status[new] = np.where((loc[new, 2] > c.h - 0.012) & (li[new] < 0), ST_RIM, ST_TUNNEL)

    def phys_step(self):
        C.pin_cup(self.m, self.d, self.cq, self.cv, self.pos, self.quat, self.prev)
        self.prev = self.d.qpos.copy()
        self._exit_check(self.local())
        mujoco.mj_step(self.m, self.d)
        self.k += 1

    @property
    def n_spilled(self): return int((self.status == ST_RIM).sum())
    @property
    def tunnel(self): return bool((self.status == ST_TUNNEL).any())
    def axis(self): return quat_axis(self.quat)

    def tick(self, u):
        u = np.asarray(u, np.float64)
        c0 = self.axis()
        for _ in range(STRIDE):
            self.phys_step()
            self.pos, self.v, self.quat, _ = kin_step(self.pos, self.v, self.quat, u[:3], u[3:])
        c1 = self.axis()
        w_obs = np.cross(c0, c1) / DT_WM
        loc = self.local(); self._exit_check(loc)
        return dict(obs=np.concatenate([c1, self.v, w_obs]), act=np.concatenate([u[:3], w_obs]),
                    l=self.margin(loc, self.status == ST_TUNNEL), n_spilled=self.n_spilled, tunnel=self.tunnel)

    def tilt(self): return float(np.arccos(np.clip(self.axis()[2], -1, 1)))
