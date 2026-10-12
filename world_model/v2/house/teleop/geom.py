"""teleop/geom.py -- footprint / clearance geometry shared by the shield (keep-out rows) and the metrics.  SIMULATION ONLY.

Footprint = phase1 hold-pose AABB of all YOR collision geoms + tray in the robot frame (nav.DEFAULT_FP; conservative).
Static obstacles: 1 cm occupancy grid + EDT (nav cache; read-only copy in scene.OccGrid).  People: discs.
"""
from __future__ import annotations

import math

import numpy as np

FP = dict(x=[-0.1645576, 0.5770901], y=[-0.6438942, 0.1303877])       # robot frame, m (== nav.DEFAULT_FP)
PERSON_CLEAR_R = 0.30            # disc radius used for a person (torso 0.22 + 0.08 tracking/arm swing allowance)


def boundary_pts(fp=FP, step=0.03, pad=0.0):
    x0, x1 = fp["x"][0] - pad, fp["x"][1] + pad
    y0, y1 = fp["y"][0] - pad, fp["y"][1] + pad
    bx = np.arange(x0, x1 + 1e-9, step)
    by = np.arange(y0, y1 + 1e-9, step)
    return np.concatenate([np.stack([bx, np.full_like(bx, y0)], 1), np.stack([bx, np.full_like(bx, y1)], 1),
                           np.stack([np.full_like(by, x0), by], 1), np.stack([np.full_like(by, x1), by], 1)])


def rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def edt_grad(grid):
    """unit gradient field of the EDT (points away from the nearest obstacle); cached on the grid object."""
    g = getattr(grid, "_grad", None)
    if g is None:
        gy, gx = np.gradient(grid.edt, grid.res)
        n = np.hypot(gx, gy) + 1e-9
        g = (gx / n, gy / n)
        grid._grad = g
    return g


def grid_lookup_grad(grid, wx, wy):
    gx, gy = edt_grad(grid)
    ix = np.floor((wx - grid.x0) / grid.res).astype(int)
    iy = np.floor((wy - grid.y0) / grid.res).astype(int)
    ny, nx = grid.occ.shape
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ixc, iyc = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
    d = np.where(ok, grid.edt[iyc, ixc], 0.0)
    return d, np.where(ok, gx[iyc, ixc], 0.0), np.where(ok, gy[iyc, ixc], 0.0)


def wall_clearance(grid, pose, pts=None):
    """footprint-to-static-obstacle clearance [m] at pose (x,y,yaw): min EDT over boundary points (<= 0: touching/inside)."""
    pts = boundary_pts(step=0.02) if pts is None else pts
    R = rot(pose[2])
    w = pts @ R.T + np.asarray(pose[:2])
    return float(grid.lookup(w[:, 0], w[:, 1]).min())


def rect_dist_point(p_world, pose, fp=FP):
    """distance (>=0 outside, <0 inside = -penetration) from a world point to the footprint rectangle, plus the nearest point on the
    rectangle in the robot frame and the unit vector (world) from the rectangle to the point."""
    R = rot(pose[2])
    q = R.T @ (np.asarray(p_world) - np.asarray(pose[:2]))
    lo = np.array([fp["x"][0], fp["y"][0]]); hi = np.array([fp["x"][1], fp["y"][1]])
    c = np.clip(q, lo, hi)
    dv = q - c
    dist = float(np.linalg.norm(dv))
    if dist < 1e-9:                                   # inside: penetration to the nearest face
        pen = np.minimum(q - lo, hi - q)
        k = int(np.argmin(pen))
        sgn = -1.0 if (q - lo)[k] < (hi - q)[k] else 1.0
        n = np.zeros(2); n[k] = sgn
        c = q.copy(); c[k] = lo[k] if sgn < 0 else hi[k]
        return -float(pen[k]), c, R @ n
    return dist, c, R @ (dv / dist)
