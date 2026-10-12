"""thor.py -- load AI2-THOR (MolmoSpaces) object assets from the local pack into an mjSpec, STATIC, texture-free. SIMULATION only.

Each asset dir `<HOUSE_DATA>/objects/thor/<category>/Prefabs/<model>/<model>.xml` is a standalone MJCF (Y-up Unity convention,
visual mesh geoms in class __VISUAL_MJT__ (contype 0, group 0), primitive/mesh colliders in group 4, one free joint).  The houses
place them with quat = Rz(yaw) * Rx(90 deg) (verified on val_0 chairs), pos = object origin.  We do the same:
    place(spec, category, model, name, xy, yaw, z=0, tint=None) -> attaches a copy with every joint deleted (static furniture),
    geom names prefixed `<name>__`, textures stripped (material rgba kept; white/untextured materials get `tint`).
`yaw` is the rotation about world z applied AFTER the Y-up -> Z-up conversion (so yaw=0 means the asset's local +z (Unity
"front") points to world -y, local +x to world +x).  Use `facing(...)` to pick yaw from a desired front direction.
Offline only; no network, no robot code.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import mujoco

from .. import assets as A

THOR_DIR = A.THOR_DIR
RX90 = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def yaw_q(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def thor_quat(yaw):
    """Rz(yaw) * Rx(90): the house placement convention."""
    return qmul(yaw_q(yaw), RX90)


def facing(dir_xy):
    """yaw such that the asset's local +z (Unity front) points along dir_xy in the world (front = Rz(yaw)*(0,-1,0))."""
    return math.atan2(dir_xy[1], dir_xy[0]) + math.pi / 2


def asset_dir(category: str, model: str) -> Path:
    return THOR_DIR / category / "Prefabs" / model


@lru_cache(maxsize=None)
def _xml_text(category: str, model: str) -> str:
    d = asset_dir(category, model)
    s = (d / f"{model}.xml").read_text()
    s = re.sub(r"<texture [^>]*/>\s*", "", s)
    s = re.sub(r' texture="[^"]*"', "", s)
    # absolute mesh paths: independent of the compiler meshdir of whichever spec we attach into
    s = re.sub(r'(<mesh [^>]*file=")([^"]+)(")', lambda mo: f"{mo.group(1)}{d}/{mo.group(2)}{mo.group(3)}", s)
    return s


def load_asset_spec(category: str, model: str, tint=None, tint_all=None) -> mujoco.MjSpec:
    sp = mujoco.MjSpec.from_string(_xml_text(category, model))
    for j in list(sp.joints):
        sp.delete(j)
    for st in list(sp.sites):                       # red ReceptacleCollider markers
        sp.delete(st)
    for mat in sp.materials:
        rgba = np.array(mat.rgba)
        if tint_all is not None:
            mat.rgba = list(tint_all)
        elif tint is not None and np.allclose(rgba, 1.0, atol=1e-3):
            mat.rgba = list(tint)
    return sp


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@lru_cache(maxsize=None)
def local_bbox(category: str, model: str):
    """(lo, hi) of the asset's collision geoms (group 4; visual group 0 if it has none) in its OWN Y-up frame."""
    sp = load_asset_spec(category, model)
    m = sp.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for grp in (4, 0):
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for g in range(m.ngeom):
            if m.geom_group[g] != grp or (grp == 0 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH):
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            e = np.abs(R) @ m.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
        if np.all(hi > lo):
            return lo, hi
    raise RuntimeError(f"no geoms in {category}/{model}")


def footprint_box(category: str, model: str, yaw=0.0):
    """World-frame (lo, hi) of the collision AABB when the origin sits at 0 and the asset has world yaw `yaw`
    (corners of the local box rotated by Rz(yaw)*Rx(90): conservative for non-box shapes)."""
    lo, hi = local_bbox(category, model)
    R = quat_mat(thor_quat(yaw))
    cs = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]) @ R.T
    return cs.min(0), cs.max(0)


def place(parent: mujoco.MjSpec, category: str, model: str, name: str, xy, yaw=0.0, z=0.0, tint=None, collide: bool = True,
          z_off: float = 0.0, tint_all=None):
    """Attach a static copy of the asset, aligned so that the collision AABB is centred on `xy`, its underside at world
    height `z + z_off` (z=0: on the floor). All names are prefixed `<name>__`.  collide=False: decor only.
    Returns (body, dict(lo, hi, size, centre) of the placed collision AABB in the world frame)."""
    lo, hi = footprint_box(category, model, yaw)
    org = np.array([xy[0] - 0.5 * (lo[0] + hi[0]), xy[1] - 0.5 * (lo[1] + hi[1]), z + z_off - lo[2]])
    child = load_asset_spec(category, model, tint, tint_all)
    fr = parent.worldbody.add_frame(pos=org.tolist(), quat=thor_quat(yaw).tolist())
    body = fr.attach_body(child.body(model), f"{name}__", "")
    if not collide:
        for b in _walk(body):
            for g in b.geoms:
                g.contype, g.conaffinity = 0, 0
    wl, wh = lo + org, hi + org
    return body, dict(lo=wl.tolist(), hi=wh.tolist(), size=(wh - wl).tolist(), centre=(0.5 * (wl + wh)).tolist(), origin=org.tolist())


def _walk(b):
    out = [b]
    for c in b.bodies:
        out += _walk(c)
    return out


def bbox(parent: mujoco.MjSpec, prefix: str, group=None):
    """World AABB (lo, hi) of the geoms whose name starts with `prefix` in the COMPILED spec (compiles a copy)."""
    m = parent.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo = np.full(3, 1e9)
    hi = np.full(3, -1e9)
    for g in range(m.ngeom):
        if not (m.geom(g).name or "").startswith(prefix):
            continue
        if group is not None and m.geom_group[g] != group:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        e = np.abs(R) @ m.geom_aabb[g, 3:]
        lo = np.minimum(lo, c - e)
        hi = np.maximum(hi, c + e)
    return lo, hi
