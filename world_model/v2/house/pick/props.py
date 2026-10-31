"""pick/props.py -- OPT-IN pickable props + a pick table for the h5 teleop scene.  SIMULATION ONLY.

Inventory (measured 2026-10-15, TeleopSim("h5")): the ProcTHOR house has NO free bodies (no freejoint other than the tray / cup / beads): every house object is a
static mesh/primitive collision geom.  The nearest static surfaces at a reachable height (chest of drawers top 0.70-0.75 m, bed 0.71 m) are 2.3 m from the spawn.
So the scene option `pick_props` (scene.build_rig / TeleopSim(pick_props=...)) adds ONE table and a few primitive props in front of the spawn pose.  Default scene unchanged.

Frame: ROBOT frame (root at the spawn pose; +x forward, +y left; the robot's centre line is y = -0.21, see teleop.scene.TRAY_LOCAL).
Table top z = 0.68 m (low table), centred 0.66 m ahead of the base origin, 0.70 m deep x 1.30 m wide, arms+base collide with it (it is NOT an 'own' body: a
touch counts as furniture contact).  Props are free bodies named `pp_*` (excluded from the sim's furniture-bump count, but arms collide with them).

Prop table  (size = full extents [m] along robot x (forward), y (lateral), z;  cylinders: diameter x diameter x height;  pos = (fwd, lateral) in the robot frame)
  pp_bottle   cylinder 0.048 x 0.048 x 0.22   0.35 kg  (0.52, -0.64)  single-arm (right)
  pp_mug      cylinder 0.055 x 0.055 x 0.095  0.25 kg  (0.40, +0.30)  single-arm (left)
  pp_cube     box      0.050 x 0.050 x 0.060  0.20 kg  (0.55, +0.28)  single-arm (left)
  pp_crate    box      0.20  x 0.30  x 0.20   1.20 kg  (0.55, -0.21)  bimanual (wider than the 0.07 m jaw opening AND heavier than the per-arm payload assumption)
  pp_basket   box      0.22  x 0.34  x 0.18   0.55 kg  (0.55, -0.21)  bimanual (width only; replaces the crate in the "basket" set)
Table top 0.68 m (0.70 m deep x 1.30 m wide, 0.31..1.01 m ahead, y -0.91..+0.39).  The task="pick" tray + cup are stowed on the table at (0.80, +0.22).
Contact: all prop geoms friction (0.8, 0.005, 0.0001) -> max-combined with the pad's 2.0 (YOR derived model, left_arm_link7_pad friction) gives mu = 2.0 at the pads.
Masses/sizes are assumptions of this scene (no source), chosen so that they sit on both sides of the arm assignment rule in pick/assign.py.
"""
from __future__ import annotations

import math

import numpy as np
import mujoco

TABLE_NAME = "pickt_table"
TABLE_TOP_Z = 0.68
TABLE_FWD = 0.66                 # table centre ahead of the base origin [m]
TABLE_CY = -0.26                 # table centre (y) [m] (robot centre line is -0.21)
TABLE_SIZE = (0.70, 1.30)        # (forward depth, lateral width) [m]
ARM_BITS = 0b1100                # sim_common _LEFT_BIT | _RIGHT_BIT (arms collide with the prop)
PROP_FRICTION = (0.8, 0.005, 0.0001)
STOW_UP = 10.0

# name -> dict(kind, size (full extents), mass, pos=(fwd, lateral) in the ROBOT frame, rgba)
PROPS = {
    # reach (pick/ik.py, antiparallel-safe residual; robot frame, site z 0.9, top-down jaws along y): left arm x=0.5 y in [-0.40, 0.20], x=0.6 y in [-0.25, 0.20];
    # right arm x=0.4..0.5 y in [-0.70, -0.10], x=0.6 only y in [-0.40, -0.25]; nothing reaches x >= 0.7.  Side (squeeze) approach x=0.5: left y in [-0.10, 0.20], right y in [-0.55, -0.25].
    "pp_bottle": dict(kind="cylinder", size=(0.048, 0.048, 0.22), mass=0.35, pos=(0.52, -0.64), rgba=(0.20, 0.55, 0.95, 1)),
    "pp_mug": dict(kind="cylinder", size=(0.055, 0.055, 0.095), mass=0.25, pos=(0.40, 0.30), rgba=(0.95, 0.65, 0.15, 1)),
    "pp_cube": dict(kind="box", size=(0.050, 0.050, 0.060), mass=0.20, pos=(0.55, 0.28), rgba=(0.25, 0.80, 0.35, 1)),
    "pp_crate": dict(kind="box", size=(0.20, 0.30, 0.20), mass=1.20, pos=(0.55, -0.21), rgba=(0.70, 0.45, 0.25, 1)),
    "pp_basket": dict(kind="box", size=(0.22, 0.34, 0.18), mass=0.55, pos=(0.55, -0.21), rgba=(0.85, 0.30, 0.30, 1)),
}
# prop sets: crate and basket share a spot (never in the same scene); `all` == `default`
SETS = {
    "default": ("pp_bottle", "pp_mug", "pp_cube", "pp_crate"),
    "all": ("pp_bottle", "pp_mug", "pp_cube", "pp_crate"),
    "single": ("pp_bottle", "pp_mug", "pp_cube"),
    "bimanual": ("pp_crate",),
    "basket": ("pp_basket",),
}


def prop_names(spec):
    """pick_props option -> tuple of prop names.  spec: True | 'default' | set name | iterable of names | dict name -> dict(overrides)."""
    if spec is True:
        spec = "default"
    if isinstance(spec, str):
        return SETS[spec]
    return tuple(spec)


def prop_cfg(name, overrides=None):
    c = dict(PROPS[name])
    if isinstance(overrides, dict) and name in overrides:
        c.update(overrides[name])
    return c


def robot_to_world(spawn_xy, yaw, fwd, lat):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([spawn_xy[0] + c * fwd - s * lat, spawn_xy[1] + s * fwd + c * lat])


def add_pick_props(house, spawn_xy, yaw, spec="default"):
    """add the pick table + props to the (uncompiled) house MjSpec.  Returns the list of prop names.  The caller (scene.build_rig) must NOT put the table into its `own` list."""
    names = prop_names(spec)
    overrides = spec if isinstance(spec, dict) else None
    q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    tc = robot_to_world(spawn_xy, yaw, TABLE_FWD, TABLE_CY)
    tb = house.worldbody.add_body(name=TABLE_NAME, pos=[tc[0], tc[1], 0.0], quat=q)
    th = 0.03
    tb.add_geom(name=f"{TABLE_NAME}_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, th / 2], pos=[0, 0, TABLE_TOP_Z - th / 2],
                rgba=[0.55, 0.42, 0.30, 1], contype=1, conaffinity=1, friction=[0.6, 0.005, 0.0001])
    for sx in (-1, 1):
        for sy in (-1, 1):
            tb.add_geom(name=f"{TABLE_NAME}_leg{sx}{sy}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (TABLE_TOP_Z - th) / 2],
                        pos=[sx * (TABLE_SIZE[0] / 2 - 0.03), sy * (TABLE_SIZE[1] / 2 - 0.03), (TABLE_TOP_Z - th) / 2], rgba=[0.55, 0.42, 0.30, 1], contype=1, conaffinity=1)
    for nm in names:
        c = prop_cfg(nm, overrides)
        sx, sy, sz = c["size"]
        wp = robot_to_world(spawn_xy, yaw, c["pos"][0], c["pos"][1])
        # compiled 10 m up (out of the way of the hold-pose IK contact check in scene.set_hold); place_props() puts them on the table
        b = house.worldbody.add_body(name=nm, pos=[wp[0], wp[1], TABLE_TOP_Z + sz / 2 + 0.0005 + STOW_UP], quat=q)
        b.add_freejoint(name=f"{nm}_free")
        kw = dict(rgba=list(c["rgba"]), contype=1, conaffinity=1 | ARM_BITS, friction=list(PROP_FRICTION), condim=4)
        if c["kind"] == "cylinder":
            b.add_geom(name=f"{nm}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[sx / 2, sz / 2, 0], mass=c["mass"], **kw)
        else:
            b.add_geom(name=f"{nm}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, sz / 2], mass=c["mass"], **kw)
    return list(names)


# arm start pose of task="pick" (free arms): both hands above the table, jaws pointing DOWN (approach -z, jaw axis along y), site at x=0.36, y = centre line -0.21 +- 0.30, z = 0.95 (wide stance: keeps the arms out of the head camera's view of the table).
# Solved with pick/ik.py (residual < 1e-4), contact-free on the pick rig (checked in tests).  [rad]
READY_Q = dict(left=[-0.4448, 1.2689, -1.1066, -1.4116, -0.4535, -0.9615], right=[0.5541, 1.3165, -1.0451, 1.283, -0.5781, 1.1258])
FINGER_OPEN = 0.035


def set_ready(rig):
    """task="pick": put the arms in READY_Q with open fingers (and make it the controller's hold/reference pose).  Called by TeleopSim before the controller is built."""
    from world_model.v2.house import build as B
    m, d = rig.m, rig.d
    for side, names in (("left", B.LEFT_J), ("right", B.RIGHT_J)):
        qa, _, _ = B._jaddr(m, names)
        d.qpos[qa] = READY_Q[side]
        for k, sg in ((7, 1.0), (8, -1.0)):
            d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_joint{k}").id]] = sg * FINGER_OPEN
            d.ctrl[m.actuator(f"{side}_arm_joint{k}_pos").id] = sg * FINGER_OPEN
    rig.info["hold_q"] = {k: list(v) for k, v in READY_Q.items()}
    mujoco.mj_forward(m, d)


def place_props(m, d, spawn_xy, yaw, overrides=None):
    """move every compiled prop from its +STOW_UP parking spot onto the table (zero velocity).  Call once after scene.new_rig, before the first mj_forward of the sim."""
    for nm in PROPS:
        try:
            b = m.body(nm).id
        except KeyError:
            continue
        c = prop_cfg(nm, overrides)
        q = int(m.jnt_qposadr[m.body_jntadr[b]])
        wp = robot_to_world(spawn_xy, yaw, c["pos"][0], c["pos"][1])
        d.qpos[q:q + 3] = (wp[0], wp[1], TABLE_TOP_Z + c["size"][2] / 2 + 0.0005)
        d.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qvel[int(m.jnt_dofadr[m.body_jntadr[b]]):int(m.jnt_dofadr[m.body_jntadr[b]]) + 6] = 0.0
    mujoco.mj_forward(m, d)


def truth(sim, name):
    """TRUE pose/size of a prop for EVALUATION and the privileged stand-in only: dict(pos (3,), R (3,3), size, mass, kind)."""
    m, d = sim.m, sim.d
    b = m.body(name).id
    c = PROPS[name]
    return dict(pos=d.xpos[b].copy(), R=d.xmat[b].reshape(3, 3).copy(), size=np.array(c["size"]), mass=float(m.body_mass[b]), kind=c["kind"])
