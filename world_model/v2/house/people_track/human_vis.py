"""people_track/human_vis.py -- VISUAL-ONLY articulated human (Microsoft Rocketbox, MIT) on the scripted people.  SIMULATION ONLY.  Default OFF.

Asset: MolmoSpaces `humans_rocketbox_articulated` / `Male_Adult_03` (Rocketbox, MIT; fetched to /data/local/ss5772/models/humans, see DOWNLOADS.log).
How it is attached (the people physics and the shield's person geometry are UNCHANGED):
  * every mesh geom is contype = conaffinity = 0; the asset's own collision capsule + 'fallback' dummies are dropped; the free joint and all 19 BALL joints
    are removed => ZERO added DOFs (nv unchanged), nothing the dynamics integrates.  The human hangs off the person's mocap body, which keeps following the
    walker's track exactly as before; the old capsule (torso) + head sphere stay in the model (contype 0, as before) but are made transparent.
  * KINEMATIC gait: the limb bodies are joint-free children, so a pose is written as m.body_quat (hip / knee / shoulder / elbow flexion) + the pelvis
    height bob in m.body_pos once per control tick (set_people); MuJoCo's kinematics reads them in every substep.  Phase is locked to the walker's speed
    (stride length STRIDE_M), amplitude blends to a standing pose when the speed -> 0 (pause / yield).  The mocap quat carries the heading.
  * the depth camera must ray-cast the mesh (not the capsule): HeadCameraHuman re-maps `person_geoms` to the human's mesh geoms.
Enable:  sim = TeleopSimHuman(...)   (people_track.sim_fix.TeleopSimWalking subclass; nothing else changes unless this class is used).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np

from world_model.v2.house.teleop import scene as SC
from . import camera as CAM
from .sim_fix import TeleopSimWalking
from world_model.v2.house.teleop.sim import TICK

HUMANS = Path("/data/local/ss5772/models/humans")
ASSET = "Male_Adult_03"
STRIDE_M = 1.40            # metres per full gait cycle (2 steps)
ANG = dict(hip=math.radians(26), knee=math.radians(42), arm=math.radians(16), elbow=math.radians(14))


def _asset_xml(variant="articulated", asset=ASSET):
    d = HUMANS / f"human_{variant}" / asset
    xml = (d / f"{asset}.xml").read_text()
    xml = re.sub(r'file="([^"/]+)"', lambda m: f'file="{d / m.group(1)}"', xml)           # absolute mesh / texture paths
    xml = re.sub(r"<joint [^>]*/>", "", xml)                                              # free + ball joints -> joint-free bodies (kinematic)
    xml = re.sub(r'<geom name="[^"]*_(?:fallback|collision)"[^>]*/>', "", xml)           # dummies + the asset's own collision capsule
    return xml


def attach_human(body, name, variant="articulated", asset=ASSET, scale=1.0):
    """attach a visual-only human under the MjSpec body `body` (person mocap body).  Returns the child body name prefix."""
    spec = mujoco.MjSpec.from_string(_asset_xml(variant, asset))
    root = spec.body(f"{asset}_Pelvis")
    # standing: the asset root (pelvis) is at z=0.896 of the model frame with feet at z~0 ; human forward is -y_h, left is +x_h -> rotate so forward = +x
    fr = body.add_frame(pos=[0, 0, 0], quat=[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    fr.attach_body(root, f"{name}_h_", "")
    return f"{name}_h_{asset}"


def hide_capsule(m, name):
    """transparent torso capsule + head sphere of person `name` (they remain, contype 0 as before)."""
    for g in range(m.ngeom):
        if m.geom(g).name in (f"{name}_torso", f"{name}_head"):
            m.geom_rgba[g, 3] = 0.0


class HeadCameraHuman(CAM.HeadCamera):
    """HeadCamera whose person geoms are the human MESH geoms (ray-cast groups switched exactly as for the capsules); the transparent capsule is not cast."""

    def __init__(self, sim, spec=None):
        super().__init__(sim, spec)
        m = sim.m
        names = [p.name for p in sim.scene.people]
        pg = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b > 0:
                bn = m.body(b).name
                if bn.startswith(tuple(f"{n}_h_" for n in names)):
                    pi = [i for i, n in enumerate(names) if bn.startswith(f"{n}_h_")][0]
                    pg[g] = pi
                    break
                b = int(m.body_parentid[b])
        if pg:
            self.person_geoms = pg
            self._pg = np.array(sorted(pg), int)
            self._pg_group = m.geom_group[self._pg].copy()


class HumanMixin:
    """mixin for any TeleopSim class: visual articulated humans on every scripted person, animated kinematically (see human_sim_class)."""
    human_variant = "articulated"

    def __init__(self, *a, human_variant=None, **kw):
        self.human_mesh = human_variant or self.human_variant
        orig = SC._add_person
        var = human_variant or self.human_variant

        def add_person(house, name):
            orig(house, name)
            b = house.body(name)
            attach_human(b, name, var)

        SC._add_person = add_person
        try:
            super().__init__(*a, **kw)
        finally:
            SC._add_person = orig
        m = self.m
        self._hb = {}
        for p in self.scene.people:
            hide_capsule(m, p.name)
            pre = f"{p.name}_h_{ASSET}_"
            ids = {}
            for key in ("Pelvis", "L_Thigh", "R_Thigh", "L_Calf", "R_Calf", "L_UpperArm", "R_UpperArm", "L_Forearm", "R_Forearm"):
                try:
                    ids[key] = m.body(f"{p.name}_h_{ASSET}_{key}").id
                except KeyError:
                    pass
            if "Pelvis" not in ids:
                raise RuntimeError(f"human not attached for {p.name} (body names: {[m.body(i).name for i in range(m.nbody) if p.name in m.body(i).name][:5]})")
            self._hb[p.name] = dict(ids=ids, q0={k: m.body_quat[v].copy() for k, v in ids.items()}, p0=m.body_pos[ids["Pelvis"]].copy(),
                                    phase=0.0, amp=0.0, yaw=0.0)
        self._gait_on = True
        self.set_people(advance=False)

    # -------------------------------------------------------------------------------------------- gait
    def set_people(self, advance=False):
        pts = super().set_people(advance)
        if not getattr(self, "_gait_on", False):
            return pts
        m = self.m
        mocap_of = {p.name: mi for p, mi in zip(self._dyn(), self.rig.person_mocap)}
        for p, (xy, v) in zip(self.scene.people if self.with_people else [], pts):
            st = self._hb.get(p.name)
            if st is None:
                continue
            sp = float(np.hypot(*v))
            if advance:
                st["amp"] += (min(sp / 0.25, 1.0) - st["amp"]) * min(TICK / 0.25, 1.0)           # blend to / from the standing pose (tau 0.25 s)
                st["phase"] += 2 * math.pi * sp * TICK / STRIDE_M
                if sp > 0.05:
                    st["yaw"] = math.atan2(v[1], v[0])
            self._pose(st)
            if p.name in mocap_of:
                self.d.mocap_quat[mocap_of[p.name]] = (math.cos(st["yaw"] / 2), 0.0, 0.0, math.sin(st["yaw"] / 2))
        return pts

    @staticmethod
    def _rotx(a):
        return np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])

    def _pose(self, st):
        m, ids, q0 = self.m, st["ids"], st["q0"]
        a, ph = st["amp"], st["phase"]

        def mul(q, r):
            w1, x1, y1, z1 = q; w2, x2, y2, z2 = r
            return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        # human frame: +x left, -y forward, +z up.  Flexion (limb swings forward) = rotation about +x by -angle; knee / elbow flex = (-)/(+) as below.
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            ph_s = ph + (0.0 if side == "L" else math.pi)
            hip = ANG["hip"] * a * math.sin(ph_s)                                    # + = forward
            knee = a * ANG["knee"] * max(0.0, math.cos(ph_s)) ** 1.5 + 0.1 * a        # swing-phase knee flexion (heel goes back = +x rotation)
            arm = -ANG["arm"] * a * math.sin(ph_s)                                   # arm swings opposite to the same-side leg
            elbow = ANG["elbow"] * a * (1.0 + 0.5 * math.sin(ph_s)) + 0.12            # + = forearm forward
            for key, ang in ((f"{side}_Thigh", -hip), (f"{side}_Calf", knee), (f"{side}_UpperArm", -arm), (f"{side}_Forearm", -elbow)):
                if key in ids:
                    m.body_quat[ids[key]] = mul(q0[key], self._rotx(ang))
        pel = ids["Pelvis"]
        m.body_pos[pel] = st["p0"] + np.array([0.0, 0.0, 0.025 * a * math.cos(2 * ph) - 0.01 * a])


def render_ms_and_cost(sim_cls, n=200, **kw):
    """helper for the cost table: nv / ngeom / mj_step ms (n control ticks, no render)."""
    import time
    s = sim_cls("h5", grasped=True, seed=0, with_people=True, counter=False, desk=False, **kw)
    return dict(nv=int(s.m.nv), nbody=int(s.m.nbody), ngeom=int(s.m.ngeom), nmesh=int(s.m.nmesh))


def human_sim_class(Base=TeleopSimWalking):
    """Base TeleopSim class + visual articulated humans (flag: use this class instead of Base; Base itself is untouched)."""
    return type("Human" + Base.__name__, (HumanMixin, Base), {})


TeleopSimHuman = human_sim_class(TeleopSimWalking)
