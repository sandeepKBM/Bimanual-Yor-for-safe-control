"""adapters/base_tip_model.py -- inputs of stack/checks/base_tip for the table-behind tasks, read from the compiled swerve scene.  SIMULATION ONLY.

  mass_model(task)        MODEL: robot bodies (the base's free-joint tree, arms at the hold pose) as point masses at their COMs + the EMPTY
                          object from the ObjectSpec (empty_mass, empty_com), all in the base frame B (rotation centre on the floor, x forward).
                          The contents (mugs + fill, the box block) are NOT in it: the check adds them per contents vertex.
  polygons()              MODEL: inner wheel contact rectangle per segment kind.  Contact point = steer pivot +- off * axle direction (both
                          wheel reversals), steer = 0 / pi for lines, the in-place-turn tangent angle for turns.
  accels(level, source)   "command": nero/base/smoke.LEVELS (ASSUMED caps).  "measured": achieved peaks of the base card (MEASURED, CARD file).
  true_contents(task, w)  the contents of one sim world as a point set (for comparing the row with a sim run).
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import mujoco

from world_model.v2.adapters import contents_specs as CS
from world_model.v2.nero.base import scene as BS
from world_model.v2.nero.base import swerve as SW
from world_model.v2.nero.contents import carry as CA
from world_model.v2.nero.contents import objects as OB
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks.base_tip import Accel, MassModel, Polygon

CARD = Path(__file__).resolve().parents[1] / "nero/results/base_card_G0b_2026-11-09/CARD.json"
KINDS = ("line", "turn")


def _kind(task):
    return TK.task(task).kind


def _episode(world):
    with BS.use_swerve_scene(second_table=True, base="swerve"):
        return CA.CarryEpisode(world)


def _frame(m, d):
    """(R0 world->B rotation (B axes in world), origin in world) of the base frame at the XML pose; and the SwerveBase."""
    sb = SW.SwerveBase(m, d)
    c, fz = BS.base_geometry()
    yaw = sb._yaw0
    R0 = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return R0, np.array([c[0], c[1], fz]), sb


def _build(task: str, world):
    w = CS.build_world(task) if world is None else world
    ep = _episode(w)
    m, d = ep.m, ep.d
    mujoco.mj_forward(m, d)
    R0, o, sb = _frame(m, d)
    toB = lambda p: R0.T @ (np.asarray(p) - o)
    root = m.body_rootid[m.body("base").id]
    bodies = [b for b in range(1, m.nbody) if m.body_rootid[b] == root and m.body_mass[b] > 0]
    ms = [float(m.body_mass[b]) for b in bodies]
    ps = [toB(d.xipos[b]) for b in bodies]
    ob = m.body("obj").id
    Ro = R0.T @ d.xmat[ob].reshape(3, 3)
    po = toB(d.site_xpos[m.site("obj_ref").id])
    spec = CS.spec_from_cell(task, *TK.prior_box(task))
    ms.append(float(spec.empty_mass))
    ps.append(Ro @ np.asarray(spec.empty_com, float) + po)
    labels = dict(mass_model=f"MODEL: {len(bodies)} robot bodies of the base tree at the hold pose (scene of task {task}) + the empty object from the "
                             f"ObjectSpec ({spec.empty_mass} kg); contents per vertex", robot_mass=round(sum(ms[:-1]), 3))
    return MassModel(np.array(ms), np.array(ps), Ro, po, labels), sb


@functools.lru_cache(maxsize=8)
def _model_cached(task: str):
    return _build(task, None)


def mass_model(task: str, world=None) -> MassModel:
    """task: a table-behind task (tray6_behind | box_behind) or its fixed-base twin; world None = the task's default world."""
    return _model_cached(task)[0] if world is None else _build(task, world)[0]


def polygons(task: str = "box_behind") -> dict:
    sb = _model_cached(task)[1]
    out = {}
    for kind in KINDS:
        pts = []
        for i in range(4):
            rx, ry = sb.r[i]
            th = 0.0 if kind == "line" else float(np.arctan2(rx, -ry))     # turn: wheel rolls along the tangent of the pivot circle
            ax = np.array([-np.sin(th), np.cos(th)])                       # axle direction at this steer angle
            pts.append([np.array([rx, ry]) + s * sb.off[i] * ax for s in (1.0, -1.0)])     # both reversals (steer th / th + pi)
        x_hi = min(min(p[0] for p in pp) for pp, (rx, _) in zip(pts, sb.r) if rx > 0)
        x_lo = max(max(p[0] for p in pp) for pp, (rx, _) in zip(pts, sb.r) if rx < 0)
        y_hi = min(min(p[1] for p in pp) for pp, (_, ry) in zip(pts, sb.r) if ry > 0)
        y_lo = max(max(p[1] for p in pp) for pp, (_, ry) in zip(pts, sb.r) if ry < 0)
        out[kind] = Polygon(float(x_lo), float(x_hi), float(y_lo), float(y_hi))
    return out


@functools.lru_cache(maxsize=1)
def card():
    return json.loads(CARD.read_text())


def accels(level: str, source: str = "measured") -> dict:
    from world_model.v2.nero.base import smoke as SM
    v, a, w, al = SM.LEVELS[level]
    if source == "command":
        src = f"nero/base/smoke.LEVELS[{level}] (ASSUMED caps)"
        return dict(line=Accel(a_lin=a, source="command", src=src), turn=Accel(alpha=al, w=w, source="command", src=src))
    c = card()["levels"][level]
    src = f"MEASURED {CARD.name}: max over n={c['n']} G0b runs ({c['filter']})"
    return dict(line=Accel(a_lin=c["a_lin"], source="measured", src=src, n=c["n"]),
                turn=Accel(alpha=c["alpha"], w=w, source="measured", src=src + "; w = command", n=c["n"]))


def pick_level(task: str, spec, levels=("L1", "L2", "L3"), source: str = "measured", world=None):
    """admission's base level for a table-behind task: the highest level whose tip row passes on spec.contents (Theta_t); None = refuse."""
    from world_model.v2.stack.checks import base_tip as T
    return T.pick_level(mass_model(task, world), spec.contents, polygons(task), {L: accels(L, source) for L in levels})


def true_contents(task: str, world):
    """the contents of `world` as a one-point Contents (true mass, true COM in the object frame): sim truth, for comparisons only."""
    from world_model.v2.nero.contents import scenes as SC
    ep = _episode(world)
    t = SC.object_truth(ep.m, ep.d)
    return OB.Contents(t["contents_mass"], t["contents_mass"], np.asarray(t["contents_com"], float)[None, :], labels=dict(m="sim truth", P="sim truth"))
