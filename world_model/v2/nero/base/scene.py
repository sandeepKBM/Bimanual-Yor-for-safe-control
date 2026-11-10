"""nero/base/scene.py -- two-table contents scene on the swerve base (SIMULATION ONLY).

The existing contents scene (contents/scenes.build_spec: table + counter at robot-local (+x), tray on the counter) with
  * the swerve variant (build_variant.variant_xml(base="swerve")) instead of the fixed one,
  * the floor plane raised to the lowest wheel vertex (MODEL: see floor_z) so the robot stands exactly where the fixed-base scene has it
    (all table / counter / object coordinates are unchanged, the table legs are visual only),
  * a SECOND table + counter BEHIND the robot: the 180-deg point mirror of the first one about the rotation centre of the base (the mean of the
    four steer pivots) -- here about the mid-point of the base route (retreat RETREAT_M, turn 180 deg in place, advance ADVANCE_M), so that at the end of the route
    the place pose of the tray relative to the robot equals the pick pose.
Existing scenes are not touched; the swerve variant is injected with a temporary patch of contents.scenes.BV.variant_xml.
"""
from __future__ import annotations

import contextlib
import functools

import numpy as np
import mujoco

from .. import build_variant as BV
from .. import config as C
from ..contents import scenes as SC
from ..pan import scene as PS


@functools.lru_cache(maxsize=1)
def base_geometry():
    """(rotation centre xy, floor z) of the swerve variant at its XML pose: MODEL, read from the compiled model."""
    m = mujoco.MjModel.from_xml_string(BV.variant_xml(base="swerve"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    P = np.array([d.xanchor[m.joint(f"{n}_steer").id] for n in ("front_left", "back_left", "front_right", "back_right")])
    zs = []
    for h in ("MK4_Colson_Hub_4_X_1.5", "MK4_Colson_Hub_4_X_1.5_2", "MK4_Colson_Hub_4_X_1.5_3", "MK4_Colson_Hub_4_X_1.5_4"):
        g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == h and m.geom_contype[g] | m.geom_conaffinity[g]][0]
        me = m.geom_dataid[g]
        vs = m.mesh_vert[m.mesh_vertadr[me]:m.mesh_vertadr[me] + m.mesh_vertnum[me]]
        zs.append((vs @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
    return P[:, :2].mean(0), float(np.mean(zs))


@contextlib.contextmanager
def _swerve_variant():
    orig = SC.BV.variant_xml
    SC.BV.variant_xml = lambda *a, **k: orig(*a, **dict(k, base="swerve"))
    try:
        yield
    finally:
        SC.BV.variant_xml = orig


# Base route pick -> place (ASSUMED): at the pick pose the arms' shoulder links (z ~ 0.8, 0.13 m in front of the base centre) touch the counter corner, so the base cannot turn
# there (seen 11-09: the turn stalls at yaw ~0.4 rad against the counter).  Route: drive back RETREAT_M (-x), turn 180 deg in place, drive forward ADVANCE_M (now -x).
RETREAT_M = 0.50       # ASSUMED m: elbows (link3/4, z ~ counter height) sweep ~0.45-0.5 m from the base centre; 0.30 stalled against the counters at yaw ~1.0-1.45 rad (11-09)
ADVANCE_M = 0.50       # ASSUMED m: the same clearance from counter 2 at the turn


def mirror_centre(retreat=RETREAT_M, advance=ADVANCE_M):
    """point about which counter 2 is the 180-deg mirror of counter 1: the mid-point of the robot's start and end positions (start = rotation centre c, end = c - (retreat+advance) x)."""
    c, _ = base_geometry()
    return c + np.array([-(retreat + advance) / 2, 0.0])


_ORIG_BUILD = SC.build_spec          # contents.scenes.build_spec as shipped (captured at import)


def mirror(p_xy, centre):
    """180-deg point mirror about the rotation centre: robot-local position after the in-place turn."""
    return 2 * np.asarray(centre, float) - np.asarray(p_xy, float)


def build_spec(world, second_table=True, base="swerve", retreat=RETREAT_M, advance=ADVANCE_M):
    """-> MjSpec: contents scene `world` on the swerve base (+ the mirrored second table/counter).  base="fixed" = same scene on the fixed-base variant."""
    c = world
    if base == "swerve":
        with _swerve_variant():
            spec = _ORIG_BUILD(world)
        ctr, fz = base_geometry()
        spec.geom("floor").pos = [0, 0, fz]                 # MODEL: lowest wheel vertex at the XML pose
    else:
        spec = _ORIG_BUILD(world)
        ctr, _ = base_geometry()
    if second_table:
        W = spec.worldbody
        tab = np.array(c.table_half, float)
        ctr = mirror_centre(retreat, advance)
        t2 = mirror(c.table_centre_xy, ctr)
        W.add_geom(name="table2_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(tab), pos=[t2[0], t2[1], c.table_top_z - tab[2]],
                   contype=1, conaffinity=1, friction=[c.table_mu, *SC.FRIC], rgba=[0.55, 0.45, 0.35, 1])
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            W.add_geom(name=f"table2_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (c.table_top_z - 2 * tab[2]) / 2],
                       pos=[t2[0] + sx * (tab[0] - 0.03), t2[1] + sy * (tab[1] - 0.03), (c.table_top_z - 2 * tab[2]) / 2],
                       contype=0, conaffinity=0, rgba=[0.4, 0.35, 0.3, 1])
        cc = mirror(c.counter_centre[:2], ctr)
        W.add_geom(name="counter2", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(c.counter_half), pos=[cc[0], cc[1], c.counter_centre[2]],
                   contype=1, conaffinity=1, friction=[c.counter_mu, *SC.FRIC], rgba=[0.5, 0.5, 0.5, 1])
    return spec


def place_pose_xy(world):
    """world xy of the obj frame origin on counter2 (= the pick pose mirrored about mirror_centre())."""
    return mirror(world.obj_xy, mirror_centre())


def support_tops_behind(world):
    """support tops and xy extents of the table-behind scene: {'counter': dict(top, x, y), 'counter2': dict(top, x, y), ('shelf': ...)}.
    MODEL: counter2 = the point mirror of the counter about mirror_centre(), same size and height (build_spec)."""
    out = dict(SC.support_tops(world))
    cx, cy = mirror(world.counter_centre[:2], mirror_centre())
    hx, hy, _ = world.counter_half
    out["counter2"] = dict(top=float(world.counter_top), x=(float(cx - hx), float(cx + hx)), y=(float(cy - hy), float(cy + hy)))   # box faces are axis-aligned: the 180-deg mirror keeps the half sizes
    return out


def compose(world, **kw):
    return PS.apply_sim_defaults(build_spec(world, **kw).compile())


@contextlib.contextmanager
def use_swerve_scene(second_table=True, base="swerve"):
    """inside the block contents.scenes.build_spec (hence carry.CarryEpisode(world), which calls it) builds THIS scene.  Restored on exit."""
    SC.build_spec = lambda world: build_spec(world, second_table=second_table, base=base)
    try:
        yield
    finally:
        SC.build_spec = _ORIG_BUILD
