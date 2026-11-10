"""nero/base/monitors.py -- base U flags of the table-behind tasks (SIMULATION ONLY).  Moved from smoke.G0Monitor (11-09); the semantics are unchanged.

  wheel_liftoff      MODEL: the two wheels of one support EDGE (same axle or same side) have no floor contact for LIFTOFF_STEPS consecutive steps:
                     the base pivots about the opposite edge (tip onset).  Not flagged: one wheel off (a rigid 4-wheel base on a plane often stands on 3
                     wheels) and a diagonal pair off (rocking / hop; seen 11-09 with 2 wheels off at 0.01-0.04 deg base tilt).
  base_world_contact MODEL: a base/torso geom (robot, not an arm, not a wheel) penetrates a world geom other than the floor
  load_world_contact MODEL: the carried object / a mug / the contents penetrate a world geom.  `load_ok` = world geoms the load may touch right now
                     (the episode passes the support tops while the base stands still; the G0 smoke passes none)
  route_unfinished   (drive.py, post-route) ASSUMED tolerance ROUTE_TOL_M / ROUTE_TOL_DEG
BaseMonitors.step(t, flag, load_ok) is read-only; `flag(kind, t, **kw)` emits a flag (once per kind until reset()).
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import swerve as SW

EDGES = ((0, 2), (1, 3), (0, 1), (2, 3))   # MODEL: swerve.MODULES = (front_left, back_left, front_right, back_right): front axle, back axle, left side, right side
LIFTOFF_STEPS = 10        # ASSUMED: 0.05 s debounce of single-step contact flicker
ROUTE_TOL_M = 0.02        # ASSUMED m (raised to the user 11-09; the place step has no base-error correction yet)
ROUTE_TOL_DEG = 2.0       # ASSUMED deg


class BaseMonitors:
    def __init__(self, m, sb, arm_g, load_b):
        """m: model; sb: SwerveBase; arm_g: set of arm geom ids; load_b: set of body ids of the carried load (object, mugs, contents, lid)."""
        self.m, self.sb, self.arm_g = m, sb, set(arm_g)
        root = m.body_rootid[m.body("base").id]
        self.base_id = m.body("base").id
        self.R0 = sb.d.xmat[self.base_id].reshape(3, 3).copy()     # base orientation at the start (its body frame is not the world frame)
        self.robot_g = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root}
        self.load_g = {g for g in range(m.ngeom) if m.geom_bodyid[g] in set(load_b)}
        self.world_g = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        self.floor_g = {g for g in self.world_g if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
        self.wheel_g = {g: i for i, h in enumerate(sb.hub) for g in range(m.ngeom) if m.geom_bodyid[g] == h and (m.geom_contype[g] | m.geom_conaffinity[g])}
        self.off_floor = np.zeros(4, int)          # consecutive steps without floor contact, per wheel
        self.flagged = set()
        self.max_off = 0

    def reset(self):
        """new window (CarryEpisode.reset_track): flags may be emitted again; the wheel debounce counters keep running."""
        self.flagged = set()

    def state(self):
        return dict(off_floor=self.off_floor.copy(), flagged=set(self.flagged), max_off=self.max_off)

    def set_state(self, s):
        self.off_floor, self.flagged, self.max_off = s["off_floor"].copy(), set(s["flagged"]), s["max_off"]

    def step(self, d, t, flag, load_ok=frozenset(), base_tilt_deg=float("nan")):
        m = self.m
        on_floor = np.zeros(4, bool)
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.wheel_g and b in self.floor_g:
                    on_floor[self.wheel_g[a]] = True
            if c.dist < 0.0:
                for a, b in ((g1, g2), (g2, g1)):
                    if b in self.world_g and b not in self.floor_g and a in self.robot_g and a not in self.arm_g and a not in self.wheel_g:
                        flag("base_world_contact", t, geom=m.geom(a).name or m.body(m.geom_bodyid[a]).name, world=m.geom(b).name)
                    elif b in self.world_g and a in self.load_g and b not in load_ok:
                        flag("load_world_contact", t, geom=m.geom(a).name or m.body(m.geom_bodyid[a]).name, world=m.geom(b).name)
        self.off_floor = np.where(on_floor, 0, self.off_floor + 1)
        off = self.off_floor >= LIFTOFF_STEPS
        n_off = int(off.sum())
        self.max_off = max(self.max_off, n_off)
        edge = [e for e in EDGES if off[list(e)].all()]
        if edge:
            if base_tilt_deg != base_tilt_deg:                                  # NaN = not given: tilt of the base z axis since the start (as the G0 smoke)
                base_tilt_deg = float(np.degrees(np.arccos(np.clip((d.xmat[self.base_id].reshape(3, 3) @ self.R0.T)[2, 2], -1.0, 1.0))))
            flag("wheel_liftoff", t, wheels_off=n_off, edge="+".join(SW.MODULES[i] for i in edge[0]), base_tilt_deg=base_tilt_deg)
