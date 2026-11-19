"""nero/base/geom_prior.py -- scene-geometry prior of the table-behind tasks (SIMULATION ONLY), so worlds differ in geometry, not only in hidden contents.

Every range is ASSUMED (USER 11-19, approved).  draw(task, i) uses its OWN rng, seeded from the world seed (probe._seed("e2e", base task, i)) and a fixed
tag, so the contents draw of world i (seed -> task_world) is unchanged.  apply(world, d) writes the drawn values into the world's geometry fields.
nominal = today's scene = the field defaults (apply(world, {}) is the identity).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

TAG = 0x6E0A                                      # fixed tag of the geometry rng stream

# (lo, hi) uniform ranges.  Every value ASSUMED (11-19).
_COMMON = dict(
    obj_dx=(-0.05, 0.05),                         # ASSUMED m: object (crate / tray) start x offset
    obj_dy=(-0.05, 0.05),                         # ASSUMED m: start y offset
    obj_yaw_deg=(-10.0, 10.0),                    # ASSUMED deg: start yaw
    counter_dz=(-0.03, 0.03),                     # ASSUMED m: counter top height
)
GEOM_PRIOR = {
    "box_bookcase": dict(
        _COMMON,
        box_x=(0.18, 0.26),                       # ASSUMED m: crate outer size x
        box_y=(0.26, 0.34),                       # ASSUMED m: y
        box_z=(0.12, 0.18),                       # ASSUMED m: z
        bc_dx=(-0.10, 0.10),                      # ASSUMED m: bookcase position offset, added to the mirrored crate pose
        bc_dy=(-0.10, 0.10),                      # ASSUMED m
        bc_yaw_deg=(-5.0, 5.0),                   # ASSUMED deg: bookcase yaw
        bc_l1_dh=(-0.03, 0.03),                   # ASSUMED m: l1 inner height offset from scene.BC_INNER_H
    ),
    # tray6_behind / box_behind: entries that apply: counter height, object xy/yaw; the second counter position (instead of a bookcase).  Object size: box_behind only.
    "tray6_behind": dict(_COMMON, c2_dx=(-0.10, 0.10), c2_dy=(-0.10, 0.10)),               # ASSUMED m: counter2 position offset (same range as the bookcase)
    "box_behind": dict(_COMMON, box_x=(0.18, 0.26), box_y=(0.26, 0.34), box_z=(0.12, 0.18), c2_dx=(-0.10, 0.10), c2_dy=(-0.10, 0.10)),
}


def draw(task, i):
    """-> {name: value} of world i of `task` (a table-behind task name); {} for a task without an entry."""
    from ..contents import probe as PR
    from ...stack import tasks as TK
    if task not in GEOM_PRIOR:
        return {}
    seed = PR._seed("e2e", TK.fixed_task(task), i)
    rng = np.random.default_rng(np.random.SeedSequence([TAG, seed]))
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in GEOM_PRIOR[task].items()}      # dict order is fixed -> deterministic


def apply(world, d):
    """world with the drawn geometry (absent entries = nominal)."""
    if not d:
        return world
    kw = {}
    if "obj_dx" in d:
        kw["obj_xy"] = (world.obj_xy[0] + d["obj_dx"], world.obj_xy[1] + d["obj_dy"])
        kw["obj_yaw"] = float(np.deg2rad(d["obj_yaw_deg"]))
    if "counter_dz" in d:
        cx, cy, cz = world.counter_centre
        kw["counter_centre"] = (cx, cy, cz + d["counter_dz"])       # counter xy stays: the start offset moves the object on the counter
    if "box_x" in d and hasattr(world, "box_outer"):
        kw["box_outer"] = (d["box_x"], d["box_y"], d["box_z"])
        from ..contents import scenes as SC
        b0 = SC.BoxWorld()                                          # USER 11-19: the contents xy draw (uniform over the NOMINAL crate's interior reach) is SCALED to the drawn crate's interior reach (same random number)
        cx, cy = world.contents_xy
        r0x, r0y = b0.box_outer[0] / 2 - b0.wall_t - b0.contents_half[0], b0.box_outer[1] / 2 - b0.wall_t - b0.contents_half[1]
        r1x, r1y = d["box_x"] / 2 - world.wall_t - world.contents_half[0], d["box_y"] / 2 - world.wall_t - world.contents_half[1]
        kw["contents_xy"] = (float(cx * r1x / r0x), float(cy * r1y / r0y))
    if "c2_dx" in d:
        kw["c2_dxy"] = (d["c2_dx"], d["c2_dy"])
    if "bc_dx" in d and getattr(world, "bookcase", False):
        from . import scene as BS
        kw["bc_dxy"] = (d["bc_dx"], d["bc_dy"])
        kw["bc_yaw"] = float(np.deg2rad(d["bc_yaw_deg"]))
        kw["bc_l1_h"] = BS.BC_INNER_H + d["bc_l1_dh"]
    return replace(world, **kw)
