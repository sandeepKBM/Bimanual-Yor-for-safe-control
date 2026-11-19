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


# ---- bookcase variant (task box_bookcase, USER 11-16).  Existing counter2 scene above is untouched; a BoxWorld(bookcase=True) gets a 2-level bookcase INSTEAD OF table2 / counter2.
# Sizes: USER-approved defaults 11-16 ("go ahead with the bookcase sizes you proposed as long as you know a solution is possible"), all ASSUMED.
BC_INNER_W = 1.10      # ASSUMED m: inner width of the compartments (y).  USER range 0.55-0.70 is INFEASIBLE for any insertion deeper than ~0.04 m: at the hold pose the wrists / elbows of the two arms
                       # stand at |dy| = 0.44-0.52 m from the box centre (links 3-7, MODEL, measured on the compiled scene 11-16), at x from -0.35 to +0.02 m of the tab line, so the arms must fit
                       # between the side boards: half width >= 0.52 + 0.005 clearance (CLEARANCE_MIN) -> 1.10 here (0.60 gave -50 mm: right gripper base vs bookcase_side_left).
BC_INNER_H = 0.35      # ASSUMED m (range 0.30-0.40): inner height of each level
BC_FLOOR_W = 0.30      # ASSUMED m (= the pick counter width; 0.40 gave -13 mm: left finger vs the floor board): width (y) of the l1 floor board, centred; None = full inner width.  A full-width floor is INFEASIBLE with the two-sided tab grasp: at the hold pose arm links 3-5
                       # dip to 0.06-0.07 m BELOW the tab line / support top at |dy| 0.25-0.50 m (MODEL, compiled scene 11-16; the pick counter is only 0.30 wide for the same reason), so once the
                       # links are inside the compartment they hit the floor (-10 mm at the route end, worse at the place).  So the floor is a central board and the side voids hold the arms.
BC_PLINTH_H = 0.15     # ASSUMED m: the compartment floor board (bookcase_l1_base) top is this far BELOW the counter top; the goal support bookcase_l1_floor is a BC_FLOOR_W wide plinth (bookcase_plinth) on it.
                       # Arm links dip 0.06-0.07 m below the support top (MODEL), so 0.15 leaves >= 0.08 m; raise in 0.05 steps (max 0.25) if arm_world to the base floor < 20 mm.
BC_DEPTH = 0.45        # ASSUMED m (range 0.35-0.45): inner depth (x), opening to back board
BC_T = 0.02            # ASSUMED m: board / side / back thickness
BC_GAP_M = -0.25       # ASSUMED m: after the base route the box FRONT face (-x, 0.10 m ahead of the obj origin) stands this far in front of the opening plane (negative = INSIDE: the base route
                       # itself drives the box 0.25 m into the compartment, the arms 0.05 m more)
BC_INSERT_M = 0.05     # ASSUMED m: the arm's straight insert move (base frame +x after the turn).  USER range 0.10-0.20 is INFEASIBLE for the arms: SRS IK scan 11-16, the hold pose reaches
                       # 0.05 m (joint margin 0.032 rad) but not 0.07-0.095 (left/right arm outside a joint limit).  Final: front face 0.30 m inside the opening, box centre 0.20 m inside, back face 0.10 m inside, 0.15 m to the back board (depth 0.45).
BC_RETREAT_M = 0.50    # ASSUMED m: route "line -0.50"  (stack/tasks.py _BOOKCASE_ROUTE; keep in sync, tests/test_tasks_match_nero.py)
BC_ADVANCE_M = 1.25    # ASSUMED m: route "line 1.0" + "line 0.25"  (sum of the segments after the 180 turn)
BC_SUPPORT = "bookcase_l1"            # TaskDef.goal_support
GOAL_GEOMS = {"counter2": "counter2", "bookcase_l1": "bookcase_l1_floor"}      # goal_support name -> its MuJoCo geom


def goal_geom(goal_support):
    return GOAL_GEOMS[goal_support]


def is_bookcase(world):
    return bool(getattr(world, "bookcase", False))


def bookcase_geometry(world):
    """MODEL, all from the world + the BC_* constants: world frame, the bookcase opens toward +x (the robot, after the turn it faces -x and stands at larger x).
    -> dict(x_open, x_back_in, y0, y1 (inner), z_floor (top of the l1 floor = counter top), z_mid_bot, z_mid_top, z_top_bot (underside of the top board), z_top_top,
            box_end_xy (obj frame origin at the end of the route), goal_xy (after the insert))."""
    c = world
    mc = mirror_centre(BC_RETREAT_M, BC_ADVANCE_M)
    xr, yr = mirror(c.obj_xy, mc)                                   # obj origin (box centre) after the route
    xr, yr = xr + c.bc_dxy[0], yr + c.bc_dxy[1]                     # bookcase offset (geom_prior; 0 = nominal)
    h1 = BC_INNER_H if c.bc_l1_h is None else float(c.bc_l1_h)      # inner height of l1
    hx = c.box_outer[0] / 2
    x_open = float(xr - hx - BC_GAP_M)                               # opening plane; the box front face (-x) is BC_GAP_M in front of it
    zf = float(c.counter_top)                                        # MODEL: the l1 floor top = the counter top (same arm pose as at the pick)
    return dict(x_open=x_open, x_back_in=x_open - BC_DEPTH, y0=float(yr - BC_INNER_W / 2), y1=float(yr + BC_INNER_W / 2), z_floor=zf,
                yaw=float(c.bc_yaw), centre=(float(x_open - BC_DEPTH / 2), float(yr)),                  # bookcase yaw about its centre (see _add_bookcase)
                z_mid_bot=zf + h1, z_mid_top=zf + h1 + BC_T, z_top_bot=zf + h1 + BC_INNER_H + BC_T, z_top_top=zf + h1 + BC_INNER_H + 2 * BC_T,
                box_end_xy=(float(xr), float(yr)), goal_xy=(float(xr - BC_INSERT_M), float(yr)))


def _add_bookcase(spec, world, floor_z):
    g = bookcase_geometry(world)
    W = spec.worldbody
    t = BC_T
    xo, xb = g["x_open"], g["x_back_in"]
    xc, hxs = (xo + xb) / 2, (xo - xb) / 2
    yc, hyi = (g["y0"] + g["y1"]) / 2, BC_INNER_W / 2

    yaw = float(world.bc_yaw)

    def box(name, hs, pos, vis=False):
        kw = {}
        if yaw != 0.0:                                                                                           # yaw the whole bookcase about its centre (xc, yc)
            dx, dy = pos[0] - xc, pos[1] - yc
            pos = (xc + np.cos(yaw) * dx - np.sin(yaw) * dy, yc + np.sin(yaw) * dx + np.cos(yaw) * dy, pos[2])
            kw["quat"] = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
        W.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(hs), pos=list(pos), contype=0 if vis else 1, conaffinity=0 if vis else 1,
                   friction=[world.counter_mu, *SC.FRIC], rgba=[0.62, 0.45, 0.28, 1], **kw)
    zf = g["z_floor"]
    fh = hyi if BC_FLOOR_W is None else BC_FLOOR_W / 2
    box("bookcase_l1_floor", (hxs, fh, t / 2), (xc, yc, zf - t / 2))                                             # goal support (top = counter top), top of the plinth
    zb0 = zf - BC_PLINTH_H                                                                                      # top of the full-width compartment floor
    box("bookcase_l1_base", (hxs, hyi, t / 2), (xc, yc, zb0 - t / 2))
    if zf - t - zb0 > 1e-9:
        box("bookcase_plinth", (hxs, fh, (zf - t - zb0) / 2), (xc, yc, (zf - t + zb0) / 2))                                            # goal support: top = counter top
    box("bookcase_mid", (hxs, hyi, t / 2), (xc, yc, (g["z_mid_bot"] + g["z_mid_top"]) / 2))                      # roof of level 1 = floor of level 2
    box("bookcase_top", (hxs, hyi, t / 2), (xc, yc, (g["z_top_bot"] + g["z_top_top"]) / 2))
    zb, zt = floor_z, g["z_top_top"]                                                                            # sides and back run from the floor to the top board
    for nm, sy in (("bookcase_side_left", 1), ("bookcase_side_right", -1)):
        box(nm, (hxs, t / 2, (zt - zb) / 2), (xc, yc + sy * (hyi + t / 2), (zt + zb) / 2))
    box("bookcase_back", (t / 2, hyi + t, (zt - zb) / 2), (xb - t / 2, yc, (zt + zb) / 2))
    return g


def build_spec(world, second_table=True, base="swerve", retreat=RETREAT_M, advance=ADVANCE_M):
    """-> MjSpec: contents scene `world` on the swerve base (+ the mirrored second table/counter).  base="fixed" = same scene on the fixed-base variant.
    A BoxWorld(bookcase=True) gets the bookcase instead of the second table (no counter2)."""
    c = world
    if is_bookcase(world):
        second_table = False
    if base == "swerve":
        with _swerve_variant():
            spec = _ORIG_BUILD(world)
        ctr, fz = base_geometry()
        spec.geom("floor").pos = [0, 0, fz]                 # MODEL: lowest wheel vertex at the XML pose
    else:
        spec = _ORIG_BUILD(world)
        ctr, _ = base_geometry()
    if is_bookcase(world):
        _add_bookcase(spec, world, float(spec.geom("floor").pos[2]))
    if second_table:
        W = spec.worldbody
        tab = np.array(c.table_half, float)
        ctr = mirror_centre(retreat, advance)
        t2 = mirror(c.table_centre_xy, ctr) + np.asarray(c.c2_dxy, float)
        W.add_geom(name="table2_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(tab), pos=[t2[0], t2[1], c.table_top_z - tab[2]],
                   contype=1, conaffinity=1, friction=[c.table_mu, *SC.FRIC], rgba=[0.55, 0.45, 0.35, 1])
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            W.add_geom(name=f"table2_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (c.table_top_z - 2 * tab[2]) / 2],
                       pos=[t2[0] + sx * (tab[0] - 0.03), t2[1] + sy * (tab[1] - 0.03), (c.table_top_z - 2 * tab[2]) / 2],
                       contype=0, conaffinity=0, rgba=[0.4, 0.35, 0.3, 1])
        cc = mirror(c.counter_centre[:2], ctr) + np.asarray(c.c2_dxy, float)
        W.add_geom(name="counter2", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(c.counter_half), pos=[cc[0], cc[1], c.counter_centre[2]],
                   contype=1, conaffinity=1, friction=[c.counter_mu, *SC.FRIC], rgba=[0.5, 0.5, 0.5, 1])
    return spec


def place_pose_xy(world):
    """world xy of the obj frame origin on counter2 (= the pick pose mirrored about mirror_centre()); bookcase: the end of the insert (bookcase_geometry goal_xy)."""
    if is_bookcase(world):
        return np.array(bookcase_geometry(world)["goal_xy"])
    return mirror(world.obj_xy, mirror_centre()) + np.asarray(world.c2_dxy, float)


def support_tops_behind(world):
    """support tops and xy extents of the table-behind scene: {'counter': dict(top, x, y), 'counter2': dict(top, x, y), ('shelf': ...)}.
    MODEL: counter2 = the point mirror of the counter about mirror_centre(), same size and height (build_spec)."""
    out = dict(SC.support_tops(world))
    if is_bookcase(world):
        g = bookcase_geometry(world)
        fh = BC_INNER_W / 2 if BC_FLOOR_W is None else BC_FLOOR_W / 2
        yc = (g["y0"] + g["y1"]) / 2
        xc, yc = g["centre"]
        hxs = (g["x_open"] - g["x_back_in"]) / 2
        yaw = g["yaw"]
        ex, ey = abs(np.cos(yaw)) * hxs + abs(np.sin(yaw)) * fh, abs(np.sin(yaw)) * hxs + abs(np.cos(yaw)) * fh      # AABB of the oriented plinth rectangle
        out["bookcase_l1"] = dict(top=g["z_floor"], x=(float(xc - ex), float(xc + ex)), y=(float(yc - ey), float(yc + ey)), z_roof=g["z_mid_bot"],
                                  yaw=yaw, centre=(float(xc), float(yc)), half=(float(hxs), float(fh)))       # x / y = AABB; yaw / centre / half = the oriented rectangle (MODEL)
        return out
    cx, cy = mirror(world.counter_centre[:2], mirror_centre()) + np.asarray(world.c2_dxy, float)
    hx, hy, _ = world.counter_half
    out["counter2"] = dict(top=float(world.counter_top), x=(float(cx - hx), float(cx + hx)), y=(float(cy - hy), float(cy + hy)),
                         yaw=0.0, centre=(float(cx), float(cy)), half=(float(hx), float(hy)))   # box faces are axis-aligned: the 180-deg mirror keeps the half sizes
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
