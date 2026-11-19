"""checks/support.py -- the total COM must stay over the contact shape on the destination support (static, proof under MODEL).  SIMULATION ONLY.

The object's contact shape comes from TaskDef.shape: Rect(lx, ly) (box, tray) or Disc(r) (pot, pan: bottom disc, MODEL r = pan_radius + pan_wall_t).
For every vertex pi = (m, m c) of the contents set the total COM is  c_tot = dest + (M_empty com_empty + m c)_xy / (M_empty + m),  M = M_empty + m.
Rect:  c_tot inside (rect around dest) intersected with the support top.
Disc:  also || c_tot - dest || <= r (the test is convex in pi, so its worst case over the contents polytope is at a vertex: exact).
Guarantee: proof under MODEL (rigid object on a flat support, at rest iff the total COM is over the contact shape).  The placement error is NOT covered.
Margin = the smallest signed distance [m] from c_tot to the boundary (> 0 inside).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.stack.checks.row import Row
from world_model.v2.stack.tasks import Disc, Rect, TaskDef

GUARANTEE = ("proof under MODEL (rigid object on a flat support: at rest iff the total COM is over the contact shape; "
             "placement at the nominal destination, placement error NOT covered)")


def support_row(t: TaskDef, dest_xy, sup_x, sup_y, empty_mass, empty_com, contents, geo=None) -> Row:
    """dest_xy: the object reference point at the goal (world xy); sup_x / sup_y: (lo, hi) extents of the support top;
    contents: objects.Contents (pi_vertices).
    geo (optional dict, from the geometry backend): sup_yaw + sup_centre + sup_half (the support top as an ORIENTED rectangle), obj_yaw (the object's yaw in the world at the goal),
    footprint (lx, ly) of the object (the drawn crate).  Then the contact shape is the object rectangle rotated by obj_yaw intersected with the oriented support rectangle, and the margin is
    the smallest signed distance to the 8 half-planes (identical to the axis-aligned numbers when both yaws are 0 and the footprint is TaskDef.shape)."""
    dest = np.asarray(dest_xy, float)[:2]
    if isinstance(t.shape, Disc):
        hx = hy = t.shape.r                    # bounding square of the disc; the disc test below is the binding one
        disc = t.shape.r
    elif isinstance(t.shape, Rect):
        hx, hy, disc = t.shape.lx / 2, t.shape.ly / 2, None
        if geo and geo.get("footprint"):
            hx, hy = float(geo["footprint"][0]) / 2, float(geo["footprint"][1]) / 2
    else:
        raise ValueError(f"task {t.name} has no support shape")
    oriented = bool(geo and geo.get("sup_yaw") is not None and (abs(float(geo.get("sup_yaw", 0.0))) > 0.0 or abs(float(geo.get("obj_yaw", 0.0))) > 0.0))
    if oriented:
        planes = _planes(dest, hx, hy, float(geo.get("obj_yaw", 0.0)), geo.get("sup_centre"), geo.get("sup_half"), float(geo.get("sup_yaw", 0.0)), sup_x, sup_y)
        x0 = x1 = y0 = y1 = 0.0
    else:
        x0, x1 = max(dest[0] - hx, sup_x[0]), min(dest[0] + hx, sup_x[1])
        y0, y1 = max(dest[1] - hy, sup_y[0]), min(dest[1] + hy, sup_y[1])
    com_e = np.asarray(empty_com, float)
    worst, worst_v = np.inf, None
    for pi in contents.pi_vertices():
        M = empty_mass + pi[0]
        c = dest + (empty_mass * com_e[:2] + pi[1:3]) / M
        marg = min(_dist(planes, c)) if oriented else min(c[0] - x0, x1 - c[0], c[1] - y0, y1 - c[1])
        if disc is not None:
            marg = min(marg, disc - float(np.linalg.norm(c - dest)))
        if marg < worst:
            worst, worst_v = float(marg), dict(m=round(float(pi[0]), 3), c=(pi[1:4] / max(pi[0], 1e-9)).round(4).tolist())
    if oriented:                                # region = the AABB of the contact polygon's bounding half-planes is not defined: report the object rectangle's and the support's AABB intersection
        x0, x1 = max(dest[0] - hx, sup_x[0]), min(dest[0] + hx, sup_x[1])
        y0, y1 = max(dest[1] - hy, sup_y[0]), min(dest[1] + hy, sup_y[1])
    reg = dict(region=[[float(x0), float(x1)], [float(y0), float(y1)]], shape=dict(kind=type(t.shape).__name__, **vars(t.shape)))
    return Row("support_dest", bool(worst >= 0.0), margin=worst, unit="m", guarantee="proof", worst=dict(worst_v or {}, **reg),
               assumptions=dict(model=GUARANTEE, dest_xy=[float(dest[0]), float(dest[1])]),
               why="" if worst >= 0.0 else f"total COM leaves the contact shape by {-worst:.4f} m (contents m={worst_v['m']} kg)")


def _planes(dest, hx, hy, obj_yaw, sup_centre, sup_half, sup_yaw, sup_x, sup_y):
    """half-planes (point, unit inward normal) of the object rectangle (centre dest, yawed obj_yaw) and of the oriented support rectangle (axis-aligned from sup_x / sup_y when no centre / half given)."""
    out = []

    def rect(c, h, yaw):
        cs, sn = np.cos(yaw), np.sin(yaw)
        for ax, hh in ((np.array([cs, sn]), h[0]), (np.array([-sn, cs]), h[1])):
            out.append((np.asarray(c, float) + ax * hh, -ax))
            out.append((np.asarray(c, float) - ax * hh, ax))
    rect(dest, (hx, hy), obj_yaw)
    if sup_centre is not None and sup_half is not None:
        rect(sup_centre, sup_half, sup_yaw)
    else:
        rect(((sup_x[0] + sup_x[1]) / 2, (sup_y[0] + sup_y[1]) / 2), ((sup_x[1] - sup_x[0]) / 2, (sup_y[1] - sup_y[0]) / 2), 0.0)
    return out


def _dist(planes, c):
    return [float(np.dot(c - p, n)) for p, n in planes]
