"""weldcontact.py -- welded-contact rule (USER-approved 11-03): while a gripper is rigidly welded to an object, the contacts between THAT gripper's
geoms (palm / pole / jaws) and THAT object are off; contacts with the table, shelf, counter, floor, the other arm and the other gripper stay on.
When the weld turns off (release) the contacts come back.

Why: the NYU jaws are pose / visual / collision only under the weld (USER 11-03).  A hold pose in which the jaws overlap the object by a few mm
would otherwise let contact forces carry part of the load (hookup finding: weld Fz 2.95 N against a weight of 13.06 N).

Mechanism (MuJoCo pair rule: geoms collide if (ct1 & ca2) | (ct2 & ca1); geom_contype / geom_conaffinity are read every step, so editing them in
the compiled model takes effect immediately; `<exclude>` and `<pair>` are compile-time and cannot be switched per side at runtime):
  carried-object geoms (the welded body and its children):   contype 1 -> OBJ_BIT (2),            conaffinity |= OBJ_BIT   (stay collidable with everything that had bit 1)
  every robot geom that had conaffinity bit 1:                 conaffinity |= OBJ_BIT           (arm links keep colliding with the object, ct_obj & ca_arm = 2)
  weld ON for side s:   that side's gripper geoms             conaffinity &= ~OBJ_BIT          (ct_obj & ca_grip = 0 and ct_grip & ca_obj = 0 -> no pair with the object;
                                                                                               table / shelf (contype 1) still pair through bit 1 and the other side's gripper through OBJ_BIT)
  weld OFF:             conaffinity |= OBJ_BIT again.
Composition with the arm-arm bits (config.ARM_ARM_BIT = 64 on the arm geoms' contype / conaffinity): this module touches only bit 2 (OBJ_BIT); it checks
at attach time that bit 2 is unused by every other geom (a collision-bit audit) and raises otherwise.
Snapshot / restore: the bits live in the MODEL, not in MjData, and a deferred release (`pending`) lives here.  Save `state()` with the snapshot and
call `sync(d, state)` after mj_setState: that restores the bits and the pending releases exactly.  `sync(d)` without a state re-derives the bits from
eq_active (welded side -> pairs off, released side -> pairs on) and drops pending releases.
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import config as C
from . import gripper as G

RESTORE_MARGIN_M = 0.005   # ASSUMED: clearance a deferred pair needs before it is switched back on
OBJ_BIT = 2          # MODEL: unused by every other geom (audited at attach); sim_common arm bits are 4/8/16/32, ARM_ARM_BIT 64


class WeldContacts:
    def __init__(self, m, obj_body: str, weld_ids: dict, sides=("left", "right"), active: bool | None = None):
        self.m, self.weld_ids, self.sides = m, dict(weld_ids), tuple(sides)
        self.active = True if active is None else bool(active)
        self.obj_geoms, self.grip_geoms, self.robot_geoms = [], {s: [] for s in self.sides}, []
        self.n_toggles = 0
        self.pending, self.deferred = {}, {}
        if not self.active:
            return
        root = m.body(obj_body).id
        for g in range(m.ngeom):
            ct, ca = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
            if (ct | ca) & OBJ_BIT:
                raise RuntimeError(f"weldcontact: bit {OBJ_BIT} already used by geom {m.geom(g).name!r} (ct {ct}, ca {ca})")
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            ct, ca = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
            name = m.body(b).name or ""
            if m.body_rootid[b] == m.body_rootid[root] and (b == root or self._is_descendant(b, root)) and (ct | ca):
                self.obj_geoms.append(g)
                m.geom_contype[g] = (ct & ~1) | OBJ_BIT if ct & 1 else ct
                m.geom_conaffinity[g] = ca | OBJ_BIT
            elif name.startswith(("left_arm", "right_arm")) and ca & 1:
                self.robot_geoms.append(g)
                m.geom_conaffinity[g] = ca | OBJ_BIT
        for s in self.sides:
            for part in ("gripper_base", "finger1", "finger2"):
                self.grip_geoms[s] += [g for g in G.col_geoms(m, s, part) if int(m.geom_conaffinity[g]) & OBJ_BIT]

    def _is_descendant(self, b, root):
        m = self.m
        while b != 0:
            if b == root:
                return True
            b = int(m.body_parentid[b])
        return False

    def _write(self, side: str, off: bool) -> None:
        for g in self.grip_geoms[side]:
            ca = int(self.m.geom_conaffinity[g])
            self.m.geom_conaffinity[g] = (ca & ~OBJ_BIT) if off else (ca | OBJ_BIT)

    def set(self, side: str, on: bool, d=None) -> None:
        """weld state of `side`: on -> the gripper's pairs with the object are off.  Weld OFF with `d` given: the pairs come back at once when the
        gripper does not overlap the object; if it does (hold poses can leave the jaws 10-20 mm inside the object's rim: they only pose / look, the weld
        carries), the pairs stay off until the gripper base has moved `depth + RESTORE_MARGIN_M` from its release position (`tick`), because switching a
        contact on inside a 20 mm overlap gives an artificial impulse (11-03: release at 127 N / 48 Nm, cup tipped).  The deviation from 'restore at
        release' is only that delay; it is recorded in `deferred` (side -> (travel [m], depth [m]))."""
        if not self.active:
            return
        self.n_toggles += 1
        self.pending.pop(side, None)
        if on or d is None:
            self._write(side, bool(on))
            return
        from . import clearance as CL
        mujoco.mj_forward(self.m, d)
        depth = min([CL._dist(self.m, d, a, b) for a in self.grip_geoms[side] for b in self.obj_geoms] + [CL._DMAX])
        if depth >= RESTORE_MARGIN_M:
            self._write(side, False)
            return
        gb = self.m.body(f"{side}_arm_gripper_base").id
        axis = d.xmat[gb].reshape(3, 3)[:, 2].copy()                 # approach axis: the release retreat is along -axis
        x0 = d.geom_xpos[self.grip_geoms[side]].copy()
        travel = None
        for t in np.arange(0.0, 0.121, 0.005):                       # first retreat distance at which the jaws are clear of the object by the margin
            d.geom_xpos[self.grip_geoms[side]] = x0 - axis * t
            if min(CL._dist(self.m, d, a, b) for a in self.grip_geoms[side] for b in self.obj_geoms) >= RESTORE_MARGIN_M:
                travel = float(t)
                break
        d.geom_xpos[self.grip_geoms[side]] = x0
        travel = 0.15 if travel is None else travel
        self.pending[side] = (d.xpos[gb].copy(), travel)
        self.deferred[side] = (travel, float(depth))

    def state(self) -> dict:
        """what a snapshot must keep: the gripper conaffinity bits and the deferred releases (not part of MjData)."""
        if not self.active:
            return {}
        return dict(ca={s: [int(self.m.geom_conaffinity[g]) for g in self.grip_geoms[s]] for s in self.sides},
                    pending={s: (p0.copy(), float(t)) for s, (p0, t) in self.pending.items()},
                    deferred=dict(self.deferred), n_toggles=self.n_toggles)

    def sync(self, d, state: dict | None = None) -> None:
        """after mj_setState: restore the bits + pending releases from `state` (exact), or, without one, re-derive the bits from d.eq_active."""
        if not self.active:
            return
        if state:
            for s in self.sides:
                for g, ca in zip(self.grip_geoms[s], state["ca"][s]):
                    self.m.geom_conaffinity[g] = ca
            self.pending = {s: (p0.copy(), float(t)) for s, (p0, t) in state["pending"].items()}
            self.deferred = dict(state["deferred"])
            self.n_toggles = int(state["n_toggles"])
            return
        self.pending = {}
        for s in self.sides:
            self._write(s, bool(d.eq_active[self.weld_ids[s]]))

    def tick(self, d) -> None:
        """per control step: re-enable the deferred pairs once the gripper base has moved far enough."""
        if not self.pending:
            return
        for side in list(self.pending):
            p0, travel = self.pending[side]
            if float(np.linalg.norm(d.xpos[self.m.body(f"{side}_arm_gripper_base").id] - p0)) >= travel:
                self._write(side, False)
                del self.pending[side]

    def pairs_off(self, d) -> dict:
        """diagnostic: per side, number of (object geom, gripper geom) pairs the collision rule currently lets collide (0 while welded)."""
        m, out = self.m, {}
        for s in self.sides:
            n = 0
            for a in self.obj_geoms:
                for b in self.grip_geoms[s]:
                    n += int((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
            out[s] = n
        return out
