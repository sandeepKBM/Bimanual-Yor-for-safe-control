"""viz.py -- drawing helpers for the demo video: zonotope wireframes (image + MuJoCo scene), a small numpy point-cloud viewer, colour palettes.  SIMULATION ONLY."""
from __future__ import annotations

import math

import numpy as np
import cv2
import mujoco

from . import perceive as PC
from . import stereo as ST
from world_model.v2.perception_zono.fit import fit_box

PALETTE = [(255, 99, 71), (65, 190, 255), (120, 230, 90), (255, 200, 40), (210, 120, 255), (60, 235, 200), (255, 130, 190), (150, 180, 255),
           (255, 160, 60), (170, 255, 140), (255, 120, 120), (100, 220, 255), (230, 230, 90), (190, 150, 255)]


def color_of(i):
    return PALETTE[i % len(PALETTE)]


_EDGE_CACHE = {}


def edges_of(Z):
    key = id(Z)
    if key not in _EDGE_CACHE:
        _EDGE_CACHE[key] = (Z, PC.zono_edges(Z))     # keep Z alive so the id stays unique
    return _EDGE_CACHE[key][1]


# ------------------------------------------------------------------------------------------------ image overlays
def draw_zono_image(img, Z, K, origin, R, color, scale=1.0, thickness=1, alpha=1.0):
    """wireframe of zonotope Z in the (RGB uint8) image; K intrinsics of the FULL-res render, scale maps to the displayed image."""
    V, E = edges_of(Z)
    if not E:
        return
    uv, z = ST.project(V, K, origin, R)
    uv = uv * scale
    for a, b in E:
        if z[a] < 0.05 or z[b] < 0.05 or not np.isfinite(uv[[a, b]]).all():
            continue
        pa, pb = tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int))
        if max(abs(pa[0]), abs(pa[1]), abs(pb[0]), abs(pb[1])) > 8000:
            continue
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def label_image(img, text, Z, K, origin, R, color, scale=1.0):
    lo, hi = Z.interval_hull()
    top = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
    uv, z = ST.project(top[None], K, origin, R)
    if z[0] < 0.05 or not np.isfinite(uv).all():
        return
    p = (int(uv[0, 0] * scale) - 14, int(uv[0, 1] * scale) - 6)
    cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


# ------------------------------------------------------------------------------------------------ MuJoCo scene decor
def _add(scn, typ, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def add_segment(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def add_zono_scene(scn, Z, color, fill=0.16, width=0.0035):
    """inflated zonotope as hull edges (capsules) + a translucent oriented box (the box is the 'up'-frame bounding box of the zonotope: fill only,
    for visibility)."""
    rgb = np.array(color, float) / 255.0
    V, E = edges_of(Z)
    for a, b in E:
        add_segment(scn, V[a], V[b], width, [*rgb, 0.95])
    if fill > 0 and len(V) >= 4:
        B = fit_box(V, "up")
        Rm = B.G / np.maximum(np.linalg.norm(B.G, axis=0), 1e-9)
        half = np.linalg.norm(B.G, axis=0)
        _add(scn, mujoco.mjtGeom.mjGEOM_BOX, half, B.c, Rm, [*rgb, fill])


def add_polyline(scn, P, rgba, width=0.006, step=1):
    P = np.asarray(P)[::step]
    for a, b in zip(P[:-1], P[1:]):
        add_segment(scn, a, b, width, rgba)


def add_tray_ghost(scn, xy, z, yaw, rgba, half=(0.15, 0.10, 0.006)):
    c, s = math.cos(yaw), math.sin(yaw)
    Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    _add(scn, mujoco.mjtGeom.mjGEOM_BOX, half, [xy[0], xy[1], z], Rm, rgba)


# ------------------------------------------------------------------------------------------------ numpy point-cloud viewer
class CloudView:
    """orbit camera around `target`, pinhole projection with painter's (far-to-near) splatting.  Draws into a (H,W,3) uint8 canvas."""

    def __init__(self, W=960, H=540, target=(4.2, 5.35, 0.85), dist=2.3, fov_deg=48.0, elev=24.0):
        self.W, self.H = W, H
        self.target = np.asarray(target, float)
        self.dist, self.elev = dist, elev
        self.f = (H / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

    def pose(self, az_deg):
        az, el = math.radians(az_deg), math.radians(self.elev)
        eye = self.target + self.dist * np.array([math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)])
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, right, up, fwd

    def project(self, P, az_deg):
        eye, right, up, fwd = self.pose(az_deg)
        d = P - eye
        z = d @ fwd
        x, y = d @ right, d @ up
        with np.errstate(invalid="ignore", divide="ignore"):
            u = self.W / 2 + self.f * x / z
            v = self.H / 2 - self.f * y / z
        return np.stack([u, v], 1), z

    def canvas(self, bg=(14, 16, 22)):
        img = np.empty((self.H, self.W, 3), np.uint8)
        img[:] = bg
        return img

    def points(self, img, P, colors, az_deg, size=2):
        if len(P) == 0:
            return
        uv, z = self.project(P, az_deg)
        ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < self.W - size) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H - size)
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(-z[idx])]
        u = uv[idx, 0].astype(int)
        v = uv[idx, 1].astype(int)
        col = np.asarray(colors)
        col = col[idx] if col.ndim == 2 else np.broadcast_to(col, (len(idx), 3))
        for dy in range(size):
            for dx in range(size):
                img[v + dy, u + dx] = col

    def segments(self, img, A, B, az_deg, color, thickness=1):
        ua, za = self.project(np.asarray(A, float), az_deg)
        ub, zb = self.project(np.asarray(B, float), az_deg)
        for p, q, z1, z2 in zip(ua, ub, za, zb):
            if z1 < 0.1 or z2 < 0.1:
                continue
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, thickness, cv2.LINE_AA)

    def zono(self, img, Z, az_deg, color, thickness=1):
        V, E = edges_of(Z)
        if not E:
            return
        A = np.array([V[a] for a, b in E])
        Bp = np.array([V[b] for a, b in E])
        self.segments(img, A, Bp, az_deg, color, thickness)

    def polyline(self, img, P, az_deg, color, thickness=2):
        P = np.asarray(P, float)
        self.segments(img, P[:-1], P[1:], az_deg, color, thickness)

    def marker(self, img, p, az_deg, color, r=6, label=None):
        uv, z = self.project(np.asarray(p, float)[None], az_deg)
        if z[0] > 0.1:
            c = (int(uv[0, 0]), int(uv[0, 1]))
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            if label:
                cv2.putText(img, label, (c[0] + 9, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def axes_floor(self, img, az_deg, z0, x0, x1, y0, y1, color=(70, 76, 90)):
        """a faint grid at height z0 (the desk top)."""
        for x in np.arange(np.ceil(x0 * 5) / 5, x1 + 1e-6, 0.2):
            self.segments(img, [[x, y0, z0]], [[x, y1, z0]], az_deg, color, 1)
        for y in np.arange(np.ceil(y0 * 5) / 5, y1 + 1e-6, 0.2):
            self.segments(img, [[x0, y, z0]], [[x1, y, z0]], az_deg, color, 1)
