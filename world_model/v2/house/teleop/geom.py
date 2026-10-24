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


# ---------------------------------------------------------------------------------------------- live (configuration-aware) footprint
_DIRS26 = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)], float)
_DIRS26 /= np.linalg.norm(_DIRS26, axis=1, keepdims=True)
_SPHERE_OUT = 1.0 / math.cos(math.radians(22.5))        # the 26 directions are <= 45 deg apart: scaling by 1/cos(22.5 deg) makes the polyhedron circumscribe the sphere


def _local_points(m, g):
    """points whose convex hull contains geom g (MuJoCo geom frame): mesh -> hull vertices of the mesh; capsule / sphere / cylinder / box -> circumscribed samples."""
    import mujoco
    t, sz = int(m.geom_type[g]), m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        try:
            from scipy.spatial import ConvexHull
            v = v[ConvexHull(v).vertices]
        except Exception:                                           # degenerate / scipy missing: keep all vertices
            pass
        return v
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _DIRS26 * sz[0] * _SPHERE_OUT
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        ends = np.array([[0.0, 0.0, -sz[1]], [0.0, 0.0, sz[1]]])
        return (ends[:, None, :] + _DIRS26[None] * sz[0] * _SPHERE_OUT).reshape(-1, 3)
    c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]                  # box / cylinder / ellipsoid / other: the 8 corners of the local AABB (exact for a box)
    return c + np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * h


class LiveFootprint:
    """Footprint of the robot in ITS CURRENT CONFIGURATION (arms, tray, cup, grippers), in the base frame.

    Precomputed once: for every colliding geom of the robot subtree + tray + cup the convex-hull points in the geom frame (mesh hull vertices; circumscribed samples of
    capsules / spheres; box corners).  Per call: world points = R_geom p + t_geom (one gather + einsum over ~2500 points), rotate into the base frame, take the xy
    min / max (+ pad).  The result is the tight axis-aligned (base frame) box of the union of the geom hulls -- the same quantity as the fixed `FP` (which was this
    computation at the phase-1 hold pose from the geom AABB corners), but of the present arm / tray pose.  Uses the kinematics of d (geom_xpos / geom_xmat), so it
    is as fresh as the last mj_step (<= 5 ms old).
    """

    def __init__(self, m, bodies=None, pad=0.02):
        self.pad = float(pad)
        if bodies is None:
            from .. import build as B
            bodies = set(B._robot_tray_bodies(m)) | {int(m.body("cup").id)}
        pts, gi = [], []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in bodies or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            p = _local_points(m, g)
            pts.append(p)
            gi.append(np.full(len(p), g, np.int32))
        self.P = np.concatenate(pts)                                 # (N, 3) geom frame
        self.gi = np.concatenate(gi)                                 # (N,) geom index
        self.n_geoms = len(set(self.gi.tolist()))
        self.n_pts = len(self.P)

    def world_pts(self, d):
        R = d.geom_xmat[self.gi].reshape(-1, 3, 3)
        return np.einsum("nij,nj->ni", R, self.P) + d.geom_xpos[self.gi]

    def box(self, d, pose, pad=None):
        """{x: [lo, hi], y: [lo, hi]} in the base frame at base pose (x, y, yaw), padded by `pad` (default self.pad) on every side."""
        pad = self.pad if pad is None else pad
        w = self.world_pts(d)[:, :2] - np.asarray(pose[:2])
        c, s = math.cos(pose[2]), math.sin(pose[2])
        qx = w[:, 0] * c + w[:, 1] * s                                 # R(yaw)^T w
        qy = -w[:, 0] * s + w[:, 1] * c
        return dict(x=[float(qx.min()) - pad, float(qx.max()) + pad], y=[float(qy.min()) - pad, float(qy.max()) + pad])


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


def cup_attached(sim) -> bool:
    """True while the tray (and the cup on it) rides on the robot: weld grasp active (sim.grasped), or the friction grasp assist in ANY state but IDLE
    (closing, HOLD, placing, FAILED/ABORT all count as attached -- conservative: cup rows stay on whenever the hands may be carrying the tray).
    Unknown (no `grasped` attribute) -> attached.  Used by ShieldCfg.cup_row / GKConfig.cup_row = "attached"."""
    if bool(getattr(sim, "grasped", True)):
        return True
    a = getattr(sim, "assist", None)
    return a is not None and getattr(a, "state", "IDLE") != "IDLE"
