"""nero/contents/scenes.py -- MuJoCo scenes for objects with HIDDEN contents: a tray with mugs, and a closed box with a hidden block.
SIMULATION ONLY.  Scene geometry only: the owner adds the grasp weld equalities to `build_spec(world)` before compiling.

Robot, floor and table are those of pan/scene.build_spec (same table pose).  No pan/food/plate/spatula/stove pad.  A fixed
"counter" block (top z = 0.82, same height as the pan scene's stove pad) supports the carried object `obj` at (0.38, -0.30).  The
claws grasp tabs at y = -0.30 +/- 0.186, outside the counter's y half-width, so the lower finger passes beside it.

Frames: the `obj` body frame has its origin at `obj_ref` = centre of the base's bottom face, axes = world axes at the start.

Hidden contents:
  * tray: mug `cup{i}` (free body, collision cylinder `cup{i}_col`) + hidden fill geom `cup{i}_fill` (contype=conaffinity=0, mass only).
    The fill is a solid cylinder resting on the mug's inner bottom: right total mass and COM.  LIQUID SLOSH IS NOT MODELLED
    (the fill is rigid and rides with the mug); a mug is a solid cylinder (its COM sits at H/2, no hollow).
  * box: free body `contents` (box of half size contents_half) on the box floor.
object_truth() convention: "empty object" = tray alone / box shell; "contents" = mugs (incl. their fills) / the block.
Labels per AGENTS.md: MEASURED (source), MODEL (derived), ASSUMED (a number we chose).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from .. import config as C
from .. import build_variant as BV
from ..pan import scene as PS
from .objects import TRAY, CUP

G = 9.81
_P = PS.PanParams()                                    # table pose / height defaults of the pan scene (MODEL: same table)
FRIC = 0.005, 0.0001                                   # torsional, rolling (ASSUMED, as in pan/scene)

GRASP_Y = 0.186        # MODEL: |y| of the grasp sites in the obj frame (matches the pot grasp points the arms reach)
# (11-03 nyu: tried 0.220 so the closed jaw tips (66 mm past the TCP) stop at the tray base edge: the right arm cannot reach it, 0 of 4 IK solutions clear; kept 0.186)
# 11-03 nyu jaw/body fix: the ee_grasp site sits nyu_hold.SHIFT_M (0.055 ASSUMED) outside the tab so the closed jaw tips stop short of the body.  The right arm
# (far tab) cannot reach |y| = 0.186 + 0.055 at the old position; the whole contents scene (object, counter, shelf) moves +y by this (ASSUMED).  Scan 11-03: 0.06 -> 3/3 IK solutions, joint margin 0.04; 0.08 -> 5/5, 0.088.
SCENE_DY = 0.08
GRASP_Z = 0.0355       # MODEL: tab centre height above the support top (pot handle centre height)
TAB_X, TAB_Z = 0.024, C.HANDLE_T   # MODEL: tab cross-section (x, z) = pot handle section (20 mm ASSUMED, config.HANDLE_T)
WATER = 1000.0         # kg/m^3, MODEL
CUP_RIN = 0.035        # ASSUMED: mug inner radius (wall 5 mm)
CUP_BOT = 0.006        # ASSUMED: mug inner bottom 6 mm above the base


@dataclass
class Common:
    obj_xy: tuple = (0.38, -0.30 + SCENE_DY)                      # MODEL: pot start centre (pan_centre_xy)
    counter_half: tuple = (0.20, 0.15, 0.03)           # ASSUMED (11-02: y half 0.12 -> 0.15 so the tray COM set fits; claws at 0.186 still clear)
    counter_centre: tuple = (0.33, -0.30 + SCENE_DY, 0.79)        # ASSUMED; top z = 0.82
    counter_mu: float = 0.8                            # ASSUMED (= table_mu of pan scene)
    table_top_z: float = _P.table_top_z                # MODEL pan scene
    table_half: tuple = _P.table_half                  # MODEL pan scene
    table_centre_xy: tuple = _P.table_centre_xy        # MODEL pan scene
    table_mu: float = _P.table_mu
    gravcomp: bool = False                             # never gravity-compensate the held object
    tau_preset: str | None = None
    drop_eps: float = 1e-4                             # initial gap above support [m]

    @property
    def counter_top(self):
        return self.counter_centre[2] + self.counter_half[2]


@dataclass
class TrayWorld(Common):
    n_cups: int = 2
    cup_xy: list | None = None        # cup centres (x, y) in the tray frame (origin = obj_ref xy); None = centred default layout
    fills_kg: list | None = None      # per-cup fill [kg] in 0..0.25; None = empty
    cup_mu: float = 0.5               # ASSUMED ceramic on ceramic glaze
    tray_mu: float = 0.6              # ASSUMED
    tray_lx: float = TRAY["lx"]       # MODEL objects.TRAY
    tray_ly: float = TRAY["ly"]
    tray_mass: float = TRAY["mass"]
    base_t: float = 0.008             # MODEL build.py
    rim_t: float = 0.008              # MODEL build.py
    rim_h: float = TRAY["rim_h"]      # MODEL: rim height above the base top
    tab_inner_y: float = TRAY["ly"] / 2   # tab runs from the end wall out to |y| = 0.21 (11-02: follows the base edge)
    tab_outer_y: float = 0.21
    cup_D: float = CUP["D"]           # ASSUMED mug
    cup_H: float = CUP["H"]
    cup_mass: float = CUP["empty"]    # ASSUMED ceramic mug 0.30 kg

    def layout(self):
        n = self.n_cups
        if self.cup_xy is not None:
            xy = [tuple(map(float, q)) for q in self.cup_xy]
        else:   # default: cups in a row along y, equally spaced
            ys = (np.arange(n) - (n - 1) / 2) * (self.tray_ly - 0.02) / max(n, 1) if n > 1 else [0.0]
            xy = [(0.0, float(y)) for y in ys]
        fills = [0.0] * n if self.fills_kg is None else [float(f) for f in self.fills_kg]
        assert len(xy) == n and len(fills) == n
        return xy, fills


@dataclass
class BoxWorld(Common):
    contents_kg: float = 1.0
    contents_xy: tuple = (0.0, 0.0)   # contents centre xy in the box frame
    contents_half: tuple = (0.04, 0.06, 0.05)
    contents_mu: float = 0.4          # ASSUMED
    box_mu: float = 0.5               # ASSUMED cardboard
    box_outer: tuple = (0.20, 0.30, 0.15)   # ASSUMED crate (x, y, z)
    wall_t: float = 0.005             # ASSUMED
    box_mass: float = 0.25            # ASSUMED cardboard crate
    tab_inner_y: float = 0.15
    tab_outer_y: float = 0.21
    shelf: bool = True
    shelf_x: tuple = (0.06, 0.27)     # ASSUMED, wall-mounted (no post modelled)
    shelf_yc: float = -0.30 + SCENE_DY   # ASSUMED
    shelf_yhalf: float = 0.14         # ASSUMED (11-02: 0.09 -> 0.14; < box y half 0.15, claws at |dy| = 0.186 clear it)
    shelf_top: float = 0.97           # ASSUMED (11-02: counter top + 0.15; a 20 cm lift misses the right arm by 18 mm, 18 cm reaches)
    shelf_t: float = 0.02             # ASSUMED
    shelf_mu: float = 0.6             # ASSUMED


def sample_tray_world(n, rng, fill_max=0.25, **kw):
    """n mugs, random non-overlapping centres (edge >= 2 mm from the inner rim, centres >= D + 2 mm apart), random fills in [0, fill_max]."""
    w0 = TrayWorld(n_cups=n, **kw)
    D = w0.cup_D
    hx = w0.tray_lx / 2 - w0.rim_t - D / 2 - 0.002
    hy = w0.tray_ly / 2 - w0.rim_t - D / 2 - 0.002
    for _ in range(2000):
        pts = []
        for _ in range(n):
            for _ in range(500):
                q = np.array([rng.uniform(-hx, hx), rng.uniform(-hy, hy)])
                if all(np.linalg.norm(q - p) >= D + 0.002 for p in pts):
                    pts.append(q); break
            else:
                break
        if len(pts) == n:
            fills = rng.uniform(0.0, fill_max, n)
            return TrayWorld(n_cups=n, cup_xy=[tuple(p) for p in pts], fills_kg=[float(f) for f in fills], **kw)
    raise RuntimeError("could not place %d cups" % n)


def _base(spec_, c: Common):
    spec = mujoco.MjSpec.from_string(BV.variant_xml(lift=False, gravcomp=c.gravcomp, tau=C.tau_limits(c.tau_preset)))
    W = spec.worldbody
    W.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=1, conaffinity=1)
    tab = np.array(c.table_half, float); cx, cy = c.table_centre_xy
    W.add_geom(name="pan_table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(tab), pos=[cx, cy, c.table_top_z - tab[2]],
               contype=1, conaffinity=1, friction=[c.table_mu, *FRIC], rgba=[0.55, 0.45, 0.35, 1])
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        W.add_geom(name=f"pan_table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (c.table_top_z - 2 * tab[2]) / 2],
                   pos=[cx + sx * (tab[0] - 0.03), cy + sy * (tab[1] - 0.03), (c.table_top_z - 2 * tab[2]) / 2],
                   contype=0, conaffinity=0, rgba=[0.4, 0.35, 0.3, 1])
    W.add_geom(name="counter", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(c.counter_half), pos=list(c.counter_centre),
               contype=1, conaffinity=1, friction=[c.counter_mu, *FRIC], rgba=[0.5, 0.5, 0.5, 1])
    return spec


def _box_geom(body, name, half, pos, mass, mu, rgba):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos), mass=float(mass),
                  friction=[mu, *FRIC], contype=1, conaffinity=1, rgba=rgba)


def _distribute(parts, total):
    """parts: list of (name, half, pos). Explicit masses proportional to volume, summing to total."""
    vol = np.array([8 * np.prod(h) for _, h, _ in parts])
    return total * vol / vol.sum()


def _obj_body(W, c, name_sites_z):
    b = W.add_body(name="obj", pos=[c.obj_xy[0], c.obj_xy[1], c.counter_top + c.drop_eps])
    b.add_freejoint(name="obj_free")
    b.add_site(name="obj_ref", pos=[0, 0, 0], size=[0.004, 0, 0], rgba=[0.2, 1, 0.2, 1])
    b.add_site(name="obj_grasp_left", pos=[0, GRASP_Y, GRASP_Z], size=[0.006, 0, 0], rgba=[1, 0.2, 0.2, 1])
    b.add_site(name="obj_grasp_right", pos=[0, -GRASP_Y, GRASP_Z], size=[0.006, 0, 0], rgba=[1, 0.2, 0.2, 1])
    return b


def _tabs(w):
    yc, yh = (w.tab_inner_y + w.tab_outer_y) / 2, (w.tab_outer_y - w.tab_inner_y) / 2
    return [("tab_left", [TAB_X / 2, yh, TAB_Z / 2], [0, yc, GRASP_Z]),
            ("tab_right", [TAB_X / 2, yh, TAB_Z / 2], [0, -yc, GRASP_Z])]


def build_spec(world):
    c = world
    spec = _base(None, c)
    W = spec.worldbody
    b = _obj_body(W, c, None)
    if isinstance(world, TrayWorld):
        w = world
        xy, fills = w.layout()
        hx, hy, bt, rt, rh = w.tray_lx / 2, w.tray_ly / 2, w.base_t, w.rim_t, w.rim_h
        parts = [("tray_base", [hx, hy, bt / 2], [0, 0, bt / 2]),
                 ("tray_rim_n", [hx, rt / 2, rh / 2], [0, hy - rt / 2, bt + rh / 2]),
                 ("tray_rim_s", [hx, rt / 2, rh / 2], [0, -hy + rt / 2, bt + rh / 2]),
                 ("tray_rim_e", [rt / 2, hy - rt, rh / 2], [hx - rt / 2, 0, bt + rh / 2]),
                 ("tray_rim_w", [rt / 2, hy - rt, rh / 2], [-hx + rt / 2, 0, bt + rh / 2])]
        parts += [(n, h, p) for n, h, p in _tabs(w)]
        ms = _distribute(parts, w.tray_mass)
        for (n, h, p), m in zip(parts, ms):
            _box_geom(b, n, h, p, m, w.tray_mu, [0.8, 0.75, 0.6, 1])
        top = bt
        for i, ((x, y), f) in enumerate(zip(xy, fills)):
            cb = W.add_body(name=f"cup{i}", pos=[c.obj_xy[0] + x, c.obj_xy[1] + y, c.counter_top + c.drop_eps + top + 1e-4])
            cb.add_freejoint(name=f"cup{i}_free")
            cb.add_geom(name=f"cup{i}_col", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[w.cup_D / 2, w.cup_H / 2, 0],
                        pos=[0, 0, w.cup_H / 2], mass=w.cup_mass, friction=[w.cup_mu, *FRIC], contype=1, conaffinity=1,
                        rgba=[0.95, 0.95, 0.95, 1])
            if f > 0:
                h = f / (WATER * np.pi * CUP_RIN ** 2)          # MODEL: fill height of a rigid cylinder of water
                cb.add_geom(name=f"cup{i}_fill", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RIN, h / 2, 0],
                            pos=[0, 0, CUP_BOT + h / 2], mass=f, contype=0, conaffinity=0, rgba=[0.4, 0.25, 0.1, 0.6])
    else:
        w = world
        X, Y, Z = w.box_outer; t = w.wall_t
        parts = [("box_floor", [X / 2, Y / 2, t / 2], [0, 0, t / 2]),
                 ("box_lid", [X / 2, Y / 2, t / 2], [0, 0, Z - t / 2]),
                 ("box_wall_e", [t / 2, Y / 2, (Z - 2 * t) / 2], [X / 2 - t / 2, 0, Z / 2]),
                 ("box_wall_w", [t / 2, Y / 2, (Z - 2 * t) / 2], [-X / 2 + t / 2, 0, Z / 2]),
                 ("box_wall_n", [X / 2 - t, t / 2, (Z - 2 * t) / 2], [0, Y / 2 - t / 2, Z / 2]),
                 ("box_wall_s", [X / 2 - t, t / 2, (Z - 2 * t) / 2], [0, -Y / 2 + t / 2, Z / 2])]
        parts += _tabs(w)
        ms = _distribute(parts, w.box_mass)
        for (n, h, p), m in zip(parts, ms):
            _box_geom(b, n, h, p, m, w.box_mu, [0.7, 0.55, 0.35, 1])
        ch = np.array(w.contents_half, float)
        cb = W.add_body(name="contents", pos=[c.obj_xy[0] + w.contents_xy[0], c.obj_xy[1] + w.contents_xy[1],
                                              c.counter_top + c.drop_eps + t + ch[2] + 1e-4])
        cb.add_freejoint(name="contents_free")
        cb.add_geom(name="contents_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(ch), mass=w.contents_kg,
                    friction=[w.contents_mu, *FRIC], contype=1, conaffinity=1, rgba=[0.3, 0.3, 0.8, 1])
        if w.shelf:
            x0, x1 = w.shelf_x
            W.add_geom(name="shelf", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, w.shelf_yhalf, w.shelf_t / 2],
                       pos=[(x0 + x1) / 2, w.shelf_yc, w.shelf_top - w.shelf_t / 2], contype=1, conaffinity=1,
                       friction=[w.shelf_mu, *FRIC], rgba=[0.6, 0.5, 0.4, 1])
    return spec


def compose(world):
    return PS.apply_sim_defaults(build_spec(world).compile())


def settle(m, d, seconds=1.0):
    """step with the arms' position servos as in pan.scene.settle; returns the displacement of `obj` [m]."""
    mujoco.mj_forward(m, d)
    b = m.body("obj").id
    p0 = d.xpos[b].copy()
    for _ in range(int(seconds / m.opt.timestep)):
        PS.step(m, d)
    return float(np.linalg.norm(d.xpos[b] - p0))


def _free_bodies(m):
    return [m.body("obj").id] + [b for b in range(m.nbody) if (m.body(b).name.startswith("cup") or m.body(b).name == "contents")]


def object_truth(m, d):
    """Truth in the obj frame at obj_ref.  empty = tray / box shell alone; contents = mugs incl. fills / the block."""
    ob = m.body("obj").id
    R = d.xmat[ob].reshape(3, 3); o = d.site_xpos[m.site("obj_ref").id]
    to_obj = lambda p: R.T @ (np.asarray(p) - o)
    m_e, c_e = float(m.body_mass[ob]), to_obj(d.xipos[ob])
    cb = [b for b in range(m.nbody) if m.body(b).name.startswith("cup") or m.body(b).name == "contents"]
    ms = np.array([m.body_mass[b] for b in cb]); cs = np.array([to_obj(d.xipos[b]) for b in cb]).reshape(-1, 3)
    m_c = float(ms.sum()); c_c = (ms[:, None] * cs).sum(0) / m_c if m_c > 0 else np.zeros(3)
    tot = m_e + m_c
    return dict(empty_mass=m_e, empty_com=c_e, contents_mass=m_c, contents_com=c_c, total_mass=tot,
                total_com=(m_e * c_e + m_c * c_c) / tot)


def support_tops(world=None):
    """support top heights and xy extents: {'counter': dict(top, x, y), 'shelf': ...} (shelf only for a BoxWorld with shelf)."""
    c = world or Common()
    cx, cy, _ = c.counter_centre; hx, hy, _ = c.counter_half
    out = {"counter": dict(top=c.counter_top, x=(cx - hx, cx + hx), y=(cy - hy, cy + hy))}
    if isinstance(c, BoxWorld) and c.shelf:
        out["shelf"] = dict(top=c.shelf_top, x=tuple(c.shelf_x), y=(c.shelf_yc - c.shelf_yhalf, c.shelf_yc + c.shelf_yhalf))
    return out
