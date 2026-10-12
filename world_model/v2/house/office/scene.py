"""scene.py -- custom single-floor "robotics lab -> corridor -> cafeteria" MuJoCo scene for the YOR coffee-run demo.  SIMULATION only.

    load_office_spec() -> mujoco.MjSpec     static scene (no joints anywhere; every furniture/wall geom is world-fixed)
    office_meta()      -> dict              room polygons, doors, named waypoints (+headings), persons, counter height, ...

Conventions (same as the MolmoSpaces houses so build/lift/nav/phase* code runs unchanged):
  * collision geoms: walls group 3, furniture/floor group 4, contype 8 / conaffinity 15 (house `__STRUCTURAL_MJT__` class);
    visual geoms group 0/1, contype=conaffinity=0.  nav.build_occupancy-style rays use groups 3+4.
  * floor = one infinite plane at z=0 (group 4, like the house) + per-room coloured visual slabs.
  * ceiling-free; walls 2.8 m.  World frame: lab SW corner = (0, 0); +x runs lab -> corridor -> cafeteria.
  * THOR objects come from the local pack (office/thor.py): textures stripped, collision = the pack's primitive colliders.
Layout (interior, free space):  LAB x[0,8] y[0,6] | door1 x[8,8.2] y-centre 3.0 | CORRIDOR x[8.2,17.8] y[2,4] | door2 x[17.8,18]
y-centre 3.0 | CAFETERIA x[18,28] y[-1,7].  Doorways are 1.10 m clear (hold footprint 0.774 m + 2x0.10 -> 0.974 m).
"""
from __future__ import annotations

import copy
import math
from functools import lru_cache

import numpy as np
import mujoco

from . import thor as T
from ..nav import DEFAULT_FP

# --------------------------------------------------------------------------------------------------------- layout constants
WALL_H = 2.8
WALL_T = 0.20
LAB = (0.0, 8.0, 0.0, 6.0)               # x0, x1, y0, y1  (free interior)
CORR = (8.2, 17.8, 2.0, 4.0)
CAFE = (18.0, 28.0, -1.0, 7.0)
DOOR_W = 1.10                             # clear width of both doorways
DOOR_Y = 3.0
DOOR1_X = (8.0, 8.2)
DOOR2_X = (17.8, 18.0)
LINTEL_Z = (2.10, WALL_H)
COUNTER_TOP = 0.90
STANDOFF_CLEAR = 0.07                     # footprint-to-obstacle clearance at the manipulation stand-offs [m]

# THOR assets used (category, model)
DESK = ("Living Room and Bedroom Objects/Desk", "Desk_204_1")
CHAIR = ("Common Objects/Chair", "Chair_204_1")
MONITOR = ("Bedroom Objects/Desktop", "Desktop_316_1")
LAMP = ("Bedroom Objects/DeskLamp", "Desk_Lamp_1")
LAPTOP = ("Living Room and Bedroom Objects/Laptop_Full", "Laptop_1")
SHELF_TALL = ("Common Objects/ShelvingUnit", "Shelving_Unit_206_1")
SHELF_WIDE = ("Common Objects/ShelvingUnit", "Shelving_Unit_001_1")
STOOL = ("Common Objects/Stool", "Stool_1_1")
COUNTER = ("ManipulaTHOR Objects/Countertops", "Countertop_I_10x2")
COFFEE = ("Kitchen Objects/CoffeeMachine", "coffee_machine_12")
FRIDGE = ("Kitchen Objects/Fridge", "Fridge_1")
BIN = ("Kitchen Objects/GarbageCan", "bin_1")
MUG = ("Kitchen Objects/Mug", "Mug_1")
TABLE_ROUND = ("Common Objects/DiningTable", "Dining_Table_16_1")
TABLE_RECT = ("Common Objects/DiningTable", "Dining_Table_201_1")

# colours
C_WALL = (0.90, 0.89, 0.85, 1)
C_LINTEL = (0.55, 0.55, 0.58, 1)
C_FLOOR_LAB = (0.62, 0.68, 0.74, 1)
C_FLOOR_CORR = (0.58, 0.54, 0.44, 1)
C_FLOOR_CAFE = (0.72, 0.56, 0.40, 1)
C_OAK = (0.76, 0.62, 0.44, 1)
C_STEEL = (0.45, 0.47, 0.50, 1)
C_WOOD = (0.66, 0.50, 0.33, 1)
TINT_FURN = (0.62, 0.50, 0.38, 1)          # replaces pure-white (texture-only) materials of THOR assets
TINT_METAL = (0.55, 0.58, 0.62, 1)
TINT_DARK = (0.22, 0.24, 0.27, 1)


# --------------------------------------------------------------------------------------------------------- primitives
def _friction():
    return [0.9, 0.9, 0.001]


def add_box(body, name, centre, half, rgba, collide=True, group_col=4, yaw=0.0, visible=True):
    """Box with a visible geom (group 1) and, if collide, an invisible collision twin (group 3/4, house structural bits)."""
    q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    if visible:
        body.add_geom(name=f"{name}_vis", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(centre), quat=q,
                      rgba=list(rgba), group=1, contype=0, conaffinity=0)
    if collide:
        body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(centre), quat=q,
                      rgba=[0, 1, 0, 0.3], group=group_col, contype=8, conaffinity=15, friction=_friction())


def add_rect(body, name, x0, x1, y0, y1, z0, z1, rgba, **kw):
    add_box(body, name, [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], [(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2], rgba, **kw)


def add_disc(body, name, xy, r, rgba, z=0.004):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, 0.002, 0], pos=[xy[0], xy[1], z], rgba=list(rgba),
                  group=1, contype=0, conaffinity=0)


def _mannequin(spec, idx, xy, shirt, yaw=0.0, height=1.70):
    """Static person: capsule torso/legs + sphere head (+ thin visual arms). Collision = the torso capsule + head sphere."""
    r_body, r_head = 0.22, 0.11
    b = spec.worldbody.add_body(name=f"person_{idx}", pos=[xy[0], xy[1], 0.0],
                                quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    z_top_body = height - 2 * r_head + 0.02
    hl = (z_top_body - 2 * r_body) / 2
    zc = r_body + hl
    CAP = mujoco.mjtGeom.mjGEOM_CAPSULE
    SPH = mujoco.mjtGeom.mjGEOM_SPHERE
    b.add_geom(name=f"person_{idx}_body_vis", type=CAP, size=[r_body, hl, 0], pos=[0, 0, zc], rgba=list(shirt), group=1,
               contype=0, conaffinity=0)
    b.add_geom(name=f"person_{idx}_head_vis", type=SPH, size=[r_head, 0, 0], pos=[0, 0, height - r_head],
               rgba=[0.87, 0.70, 0.58, 1], group=1, contype=0, conaffinity=0)
    for s in (-1, 1):
        b.add_geom(name=f"person_{idx}_arm{s}_vis", type=CAP, size=[0.045, 0.28, 0], pos=[0.02, s * (r_body + 0.03), 1.05],
                   rgba=list(shirt), group=1, contype=0, conaffinity=0)
    b.add_geom(name=f"person_{idx}_body_col", type=CAP, size=[r_body, hl, 0], pos=[0, 0, zc], rgba=[0, 1, 0, 0.3], group=4,
               contype=8, conaffinity=15, friction=_friction())
    b.add_geom(name=f"person_{idx}_head_col", type=SPH, size=[r_head, 0, 0], pos=[0, 0, height - r_head], rgba=[0, 1, 0, 0.3],
               group=4, contype=8, conaffinity=15, friction=_friction())
    return dict(name=f"person_{idx}", xy=[float(xy[0]), float(xy[1])], radius=r_body, height=height, yaw=float(yaw))


def _cam_quat(forward, up=(0, 0, 1)):
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    z = -f
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).ravel())
    return q


def standoff_y(edge, clear=STANDOFF_CLEAR, fp=DEFAULT_FP):
    """Robot-origin offset so the hold footprint's forward edge (fp x1) is `clear` short of `edge` (facing +y)."""
    return edge - clear - fp["x"][1]


# --------------------------------------------------------------------------------------------------------- the build
def _build():
    sp = mujoco.MjSpec()
    sp.modelname = "office"
    sp.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST          # same options as the house XMLs
    sp.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    sp.option.impratio = 10.0
    sp.option.noslip_iterations = 4
    sp.option.gravity = [0.0, 0.0, -9.81]
    sp.visual.global_.offwidth = 1920
    sp.visual.global_.offheight = 1080
    sp.visual.quality.shadowsize = 8192
    sp.visual.headlight.ambient = [0.30, 0.30, 0.30]
    sp.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    sp.visual.map.znear = 0.01
    sp.add_texture(name="office_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.62, 0.74, 0.90], rgb2=[0.22, 0.28, 0.38], width=512, height=512)
    W = sp.worldbody
    meta = dict(furniture={})

    # ---- floor: infinite collision plane (house convention) + visual slabs
    W.add_geom(name="office_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01], pos=[0, 0, 0], group=4,
               contype=8, conaffinity=15, friction=_friction(), rgba=[0.5, 0.5, 0.5, 1])
    for nm, (x0, x1, y0, y1), col in (("lab", LAB, C_FLOOR_LAB), ("corridor", (8.0, 18.0, 2.0, 4.0), C_FLOOR_CORR),
                                      ("cafe", CAFE, C_FLOOR_CAFE)):
        add_rect(W, f"floor_{nm}", x0, x1, y0, y1, -0.02, 0.0, col, collide=False)
    # door thresholds (visual)
    for i, (xa, xb) in enumerate((DOOR1_X, DOOR2_X), 1):
        add_rect(W, f"threshold_door{i}", xa, xb, DOOR_Y - DOOR_W / 2, DOOR_Y + DOOR_W / 2, -0.02, 0.002, (0.30, 0.30, 0.32, 1),
                 collide=False)

    # ---- walls (group 3, house convention)
    t = WALL_T
    g_lo, g_hi = DOOR_Y - DOOR_W / 2, DOOR_Y + DOOR_W / 2
    walls = {
        "lab_west": (-t, 0.0, -t, 6.0 + t), "lab_south": (-t, 8.0, -t, 0.0), "lab_north": (-t, 8.2, 6.0, 6.0 + t),
        "d1_south": (DOOR1_X[0], DOOR1_X[1], -t, g_lo), "d1_north": (DOOR1_X[0], DOOR1_X[1], g_hi, 6.0),
        "corr_south": (8.2, 17.8, 2.0 - t, 2.0), "corr_north": (8.2, 17.8, 4.0, 4.0 + t),
        "d2_south": (DOOR2_X[0], DOOR2_X[1], -1.0 - t, g_lo), "d2_north": (DOOR2_X[0], DOOR2_X[1], g_hi, 7.0 + t),
        "cafe_south": (18.0, 28.0 + t, -1.0 - t, -1.0), "cafe_north": (18.0, 28.0 + t, 7.0, 7.0 + t),
        "cafe_east": (28.0, 28.0 + t, -1.0, 7.0),
    }
    for nm, (x0, x1, y0, y1) in walls.items():
        add_rect(W, f"wall_{nm}", x0, x1, y0, y1, 0.0, WALL_H, C_WALL, group_col=3)
    for i, (xr) in enumerate((DOOR1_X, DOOR2_X), 1):                                   # lintels above the doorways
        add_rect(W, f"lintel_door{i}", xr[0], xr[1], g_lo, g_hi, LINTEL_Z[0], LINTEL_Z[1], C_LINTEL, group_col=3)

    # ================================================================ LAB
    desk_y = 5.55
    desks = {}
    for i, x in enumerate((2.0, 4.2, 6.4)):                                          # desks along the north wall, front = -y
        _, inf = T.place(sp, *DESK, f"desk_{i}", (x, desk_y), 0.0, tint_all=C_OAK)
        desks[f"desk_{i}"] = inf
    _, inf = T.place(sp, *DESK, "desk_3", (7.55, 1.2), T.facing((-1, 0)), tint_all=C_OAK)      # east wall, front = -x
    desks["desk_3"] = inf
    top = desks["desk_0"]["hi"][2]
    # chairs (desk_1 = the delivery desk: chair pulled out to the side, slightly angled)
    for i, (xy, face) in {"chair_0": ((2.0, 4.85), (0, 1)), "chair_2": ((6.4, 4.85), (0, 1)), "chair_3": ((6.55, 1.2), (1, 0))}.items():
        T.place(sp, *CHAIR, i, xy, T.facing(face), tint=TINT_DARK)
    T.place(sp, *CHAIR, "chair_1", (3.0, 4.3), T.facing((0.6, 0.8)), tint=TINT_DARK)
    # desk items
    T.place(sp, *MONITOR, "monitor_0", (1.7, 5.80), 0.0, z=top, tint=TINT_DARK)
    T.place(sp, *MONITOR, "monitor_1", (3.9, 5.80), 0.0, z=top, tint=TINT_DARK)
    T.place(sp, *LAPTOP, "laptop_2", (6.3, 5.45), 0.0, z=top, tint=TINT_METAL)
    T.place(sp, *LAPTOP, "laptop_3", (7.65, 1.7), T.facing((-1, 0)), z=top, tint=TINT_METAL)
    T.place(sp, *MONITOR, "monitor_3", (7.80, 0.75), T.facing((-1, 0)), z=top, tint=TINT_DARK)
    T.place(sp, *LAMP, "lamp_0", (2.75, 5.78), 0.0, z=top, tint=TINT_METAL)
    T.place(sp, *LAMP, "lamp_1", (4.85, 5.78), 0.0, z=top, tint=TINT_METAL)
    T.place(sp, *LAMP, "lamp_2", (7.0, 5.78), 0.0, z=top, tint=TINT_METAL)
    T.place(sp, *MUG, "mug_lab2", (5.8, 5.4), 0.0, z=top)
    # workbench (south wall) built from boxes: wood top, steel legs + lower shelf; instruments on top
    wb = dict(x0=3.4, x1=5.8, y0=0.05, y1=0.85)
    add_rect(W, "workbench_top", wb["x0"], wb["x1"], wb["y0"], wb["y1"], 0.86, 0.92, C_WOOD)
    for k, (lx, ly) in enumerate(((wb["x0"] + 0.04, wb["y0"] + 0.04), (wb["x1"] - 0.04, wb["y0"] + 0.04),
                                  (wb["x0"] + 0.04, wb["y1"] - 0.04), (wb["x1"] - 0.04, wb["y1"] - 0.04))):
        add_rect(W, f"workbench_post{k}", lx - 0.03, lx + 0.03, ly - 0.03, ly + 0.03, 0.0, 0.86, C_STEEL)
    add_rect(W, "workbench_shelf", wb["x0"], wb["x1"], wb["y0"], wb["y1"], 0.18, 0.21, C_STEEL)
    for k, (cx, hx, hy, hz, col) in enumerate(((3.8, 0.18, 0.15, 0.09, (0.20, 0.45, 0.70, 1)), (4.4, 0.25, 0.12, 0.05, (0.75, 0.20, 0.18, 1)),
                                                (5.0, 0.12, 0.20, 0.12, (0.25, 0.25, 0.28, 1)), (5.5, 0.14, 0.14, 0.07, (0.85, 0.75, 0.2, 1)))):
        add_rect(W, f"bench_item{k}", cx - hx, cx + hx, 0.45 - hy, 0.45 + hy, 0.92, 0.92 + 2 * hz, col)
    for k, x in enumerate((3.95, 5.25)):
        T.place(sp, *STOOL, f"stool_{k}", (x, 1.15), 0.0, tint=TINT_DARK)
    # shelving along the west wall (front = +x)
    for k, y in enumerate((1.3, 2.2, 3.0, 3.8)):
        T.place(sp, *SHELF_TALL, f"shelf_{k}", (0.21, y), T.facing((1, 0)), tint=TINT_METAL)
    T.place(sp, *SHELF_WIDE, "shelf_wide", (0.27, 4.95), T.facing((1, 0)), tint=TINT_METAL)
    # floor crates
    add_rect(W, "crate_0", 6.5, 7.1, 0.15, 0.6, 0.0, 0.32, (0.55, 0.40, 0.22, 1))
    add_rect(W, "crate_1", 6.6, 7.0, 0.62, 0.95, 0.0, 0.25, (0.45, 0.33, 0.18, 1))

    # ================================================================ CORRIDOR (visual wall details only; no collision)
    for k, x in enumerate((10.5, 13.0, 15.5)):
        add_rect(W, f"corr_panel_s{k}", x - 0.5, x + 0.5, 2.0, 2.02, 1.2, 1.9, (0.30, 0.55, 0.45, 1), collide=False)
        add_rect(W, f"corr_panel_n{k}", x - 0.5, x + 0.5, 3.98, 4.0, 1.2, 1.9, (0.55, 0.40, 0.30, 1), collide=False)
    add_rect(W, "corr_stripe", 8.2, 17.8, 2.95, 3.05, 0.0, 0.003, (0.95, 0.85, 0.15, 1), collide=False)
    W.add_geom(name="corr_extinguisher_vis", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.07, 0.25, 0], pos=[12.0, 2.08, 0.8],
               rgba=[0.85, 0.1, 0.1, 1], group=1, contype=0, conaffinity=0)

    # ================================================================ CAFETERIA
    cx, cy = 23.5, 6.65
    clo, chi = T.footprint_box(*COUNTER, 0.0)
    sink = float(chi[2] - clo[2]) - COUNTER_TOP                  # pack counter is 0.94 m: sink it so the top is exactly 0.90
    _, cinf = T.place(sp, *COUNTER, "counter", (cx, cy), 0.0, z_off=-sink, tint=TINT_FURN)
    meta["furniture"]["counter_sink"] = sink
    _, minf = T.place(sp, *COFFEE, "coffee_machine", (24.4, 6.62), 0.0, z=COUNTER_TOP, tint=TINT_METAL)
    T.place(sp, *MUG, "mug_counter", (23.55, 6.55), 0.0, z=COUNTER_TOP)
    T.place(sp, *FRIDGE, "fridge", (26.3, 6.62), 0.0, tint=TINT_METAL)
    T.place(sp, *BIN, "bin", (27.5, 6.70), 0.0, tint=TINT_DARK)
    pad_xy = (22.9, 6.50)                                                            # marked tray pick spot on the counter
    add_rect(W, "tray_pick_pad", pad_xy[0] - 0.19, pad_xy[0] + 0.19, pad_xy[1] - 0.13, pad_xy[1] + 0.13, COUNTER_TOP,
             COUNTER_TOP + 0.004, (0.1, 0.8, 0.3, 1), collide=False)
    # dining tables + chairs
    tables = {"table_0": (TABLE_ROUND, (21.8, 0.8)), "table_1": (TABLE_ROUND, (25.2, 0.8)),
              "table_2": (TABLE_ROUND, (26.6, 3.4)), "table_3": (TABLE_RECT, (23.6, 2.9))}
    tabs = {}
    for nm, (asset, xy) in tables.items():
        _, inf = T.place(sp, *asset, nm, xy, 0.0, tint=TINT_FURN)
        tabs[nm] = inf
    chairs = []
    for nm in ("table_0", "table_1", "table_2"):
        c = np.array(tables[nm][1])
        for a_deg in (90, 210, 330):
            a = math.radians(a_deg)
            chairs.append((c + 0.95 * np.array([math.cos(a), math.sin(a)]), -np.array([math.cos(a), math.sin(a)])))
    for xy, face in (((23.0, 3.7), (0, -1)), ((24.2, 3.7), (0, -1)), ((23.0, 2.1), (0, 1)), ((24.2, 2.1), (0, 1))):
        chairs.append((np.array(xy), np.array(face, float)))
    for k, (xy, face) in enumerate(chairs):
        T.place(sp, *CHAIR, f"cafe_chair_{k}", xy, T.facing(face), tint=TINT_DARK)
    # people: static mannequins
    persons = [_mannequin(sp, 0, (24.1, 5.15), (0.80, 0.25, 0.25, 1), yaw=math.pi / 2),
               _mannequin(sp, 1, (21.0, 4.55), (0.25, 0.45, 0.80, 1), yaw=-0.5),
               _mannequin(sp, 2, (22.0, 2.6), (0.30, 0.65, 0.35, 1), yaw=1.2)]

    # ================================================================ waypoints / markers / cameras / lights
    fp = DEFAULT_FP
    pick_rx = pad_xy[0] - 0.21                                                      # tray centre (robot-local (0.45,-0.21)) at the pad x
    pick_ry = standoff_y(cinf["lo"][1])
    place_rx = (desks["desk_1"]["centre"][0]) - 0.21
    place_ry = standoff_y(desks["desk_1"]["lo"][1])
    wps = {
        "lab_start": dict(xy=[1.8, 2.0], heading=0.0, room="lab"),
        "door1": dict(xy=[sum(DOOR1_X) / 2, DOOR_Y], heading=0.0, room="door", note="travel heading lab->corridor; reverse = pi"),
        "door2": dict(xy=[sum(DOOR2_X) / 2, DOOR_Y], heading=0.0, room="door", note="travel heading corridor->cafe; reverse = pi"),
        "cafe_counter_pick": dict(xy=[pick_rx, pick_ry], heading=math.pi / 2, room="cafe",
                                  target_xyz=[pad_xy[0], pad_xy[1], COUNTER_TOP], note="robot origin; tray centre sits 0.45 m ahead, 0.21 m right"),
        "lab_desk_place": dict(xy=[place_rx, place_ry], heading=math.pi / 2, room="lab",
                               target_xyz=[desks["desk_1"]["centre"][0], desk_y - 0.1, top], note="delivery desk = desk_1 (no chair in front)"),
    }
    cols = {"lab_start": (0.2, 0.4, 0.9, 1), "door1": (0.95, 0.85, 0.1, 1), "door2": (0.95, 0.85, 0.1, 1),
            "cafe_counter_pick": (0.1, 0.8, 0.3, 1), "lab_desk_place": (0.95, 0.5, 0.1, 1)}
    for nm, w in wps.items():
        add_disc(W, f"marker_{nm}", w["xy"], 0.16, cols[nm])
        h = w["heading"]
        W.add_geom(name=f"marker_{nm}_arrow", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.14, 0.025, 0.002],
                   pos=[w["xy"][0] + 0.14 * math.cos(h), w["xy"][1] + 0.14 * math.sin(h), 0.006],
                   quat=[math.cos(h / 2), 0, 0, math.sin(h / 2)], rgba=[1, 1, 1, 1], group=1, contype=0, conaffinity=0)
        W.add_site(name=f"wp_{nm}", pos=[w["xy"][0], w["xy"][1], 0.01], size=[0.05, 0, 0], group=5, rgba=list(cols[nm]))
    for key in ("cafe_counter_pick", "lab_desk_place"):
        tx = wps[key]["target_xyz"]
        W.add_site(name=f"{key}_target", pos=tx, size=[0.04, 0, 0], group=5, rgba=list(cols[key]))
    W.add_site(name="lab_desk_place", pos=wps["lab_desk_place"]["target_xyz"], size=[0.04, 0, 0], group=5, rgba=[0.95, 0.5, 0.1, 1])
    W.add_site(name="cafe_counter_pick", pos=wps["cafe_counter_pick"]["target_xyz"], size=[0.04, 0, 0], group=5, rgba=[0.1, 0.8, 0.3, 1])

    W.add_light(name="office_sun", pos=[14, 3, 14], dir=[-0.15, 0.1, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                castshadow=1, diffuse=[0.55, 0.55, 0.52], specular=[0.1, 0.1, 0.1])
    for nm, p in (("lab", (4, 3, 4.5)), ("corr", (13, 3, 4.0)), ("cafe", (23, 3, 5.0))):
        W.add_light(name=f"office_fill_{nm}", pos=list(p), dir=[0, 0, -1], castshadow=0, diffuse=[0.10, 0.10, 0.10],
                    specular=[0.0, 0.0, 0.0])
    ct = W.add_camera(name="overview_top", pos=[14.0, 3.0, 16.0], quat=_cam_quat([0, 0, -1], [0, 1, 0]).tolist(), fovy=11.0)
    ct.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC                              # fovy = view height [m] when orthographic
    W.add_camera(name="overview_lab", pos=[4.0, 3.0, 9.5], quat=_cam_quat([0, 0, -1], [0, 1, 0]).tolist(), fovy=50.0)
    W.add_camera(name="overview_cafe", pos=[23.0, 3.0, 11.5], quat=_cam_quat([0, 0, -1], [0, 1, 0]).tolist(), fovy=50.0)
    W.add_camera(name="overview_oblique", pos=[-3.0, -5.0, 8.0], quat=_cam_quat([14.0, 6.0, -6.5], [0, 0, 1]).tolist(), fovy=60.0)

    meta.update(
        rooms=dict(lab=_rect_poly(LAB), corridor=_rect_poly(CORR), cafeteria=_rect_poly(CAFE)),
        bounds=dict(x=[-t, 28.0 + t], y=[-1.0 - t, 7.0 + t]),
        wall_height=WALL_H, wall_thickness=WALL_T, counter_height=COUNTER_TOP, desk_top_height=float(top),
        doors=[dict(id="door1", centre=[sum(DOOR1_X) / 2, DOOR_Y], wall_dir=[0.0, 1.0], normal=[1.0, 0.0], clear_w=DOOR_W,
                    design_w=DOOR_W, side_pos="corridor", side_neg="lab", thickness=DOOR1_X[1] - DOOR1_X[0]),
               dict(id="door2", centre=[sum(DOOR2_X) / 2, DOOR_Y], wall_dir=[0.0, 1.0], normal=[1.0, 0.0], clear_w=DOOR_W,
                    design_w=DOOR_W, side_pos="cafeteria", side_neg="corridor", thickness=DOOR2_X[1] - DOOR2_X[0])],
        waypoints=wps, persons=persons,
        desks={k: dict(centre=v["centre"], size=v["size"], top=v["hi"][2]) for k, v in desks.items()},
        tables={k: dict(centre=v["centre"], size=v["size"], top=v["hi"][2]) for k, v in tabs.items()},
        counter=dict(centre=cinf["centre"][:2], lo=cinf["lo"], hi=cinf["hi"], top=COUNTER_TOP),
        coffee_machine=dict(centre=minf["centre"], size=minf["size"]),
        tray_pick_pad=dict(centre=[pad_xy[0], pad_xy[1], COUNTER_TOP], half=[0.19, 0.13]),
        cameras=["overview_top", "overview_lab", "overview_cafe", "overview_oblique"],
        footprint=DEFAULT_FP, standoff_clear=STANDOFF_CLEAR,
    )
    return sp, meta


def _rect_poly(r):
    x0, x1, y0, y1 = r
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


@lru_cache(maxsize=1)
def _meta_cached():
    return _build()[1]


def load_office_spec() -> mujoco.MjSpec:
    """Fresh static office spec (no joints, no free bodies; like build.load_house_spec(static=True))."""
    sp, _ = _build()
    for j in list(sp.joints):
        sp.delete(j)
    return sp


def office_meta() -> dict:
    """JSON-friendly metadata (deep copy): rooms, doors, waypoints (xy + heading [rad] + optional target_xyz), persons, heights."""
    return copy.deepcopy(_meta_cached())
