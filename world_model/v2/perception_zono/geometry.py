"""geometry.py -- analytic ground-truth geometry (box / cylinder / sphere primitives), an object library, and an MJCF scene builder.
SIMULATION ONLY. The same primitive list builds (a) the MuJoCo model that is ray-cast and (b) the truth used for scoring
(SDF, interior test, surface samples, volume), so truth and sensor can not disagree.

World frame = the tray_rep testbed world: tray home at (0,0,1), tray top surface at z = 1. Objects are defined with their base at z = 0
in a local yaw frame and placed with place(proto, x, y, yaw) (x, y in the tray frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRAY_HX, TRAY_HY, TRAY_T = 0.15, 0.10, 0.003
Z0 = 1.0   # tray top height in the world


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass
class Prim:
    kind: str                 # 'box' (size = half extents) | 'cyl' (size = (r, half height), axis local z) | 'sphere' (size = (r,))
    size: tuple
    pos: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))

    # ---- SDF (exact outside, a lower bound of the depth inside)
    def sdf(self, X):
        L = (np.asarray(X) - self.pos) @ self.R
        if self.kind == "sphere":
            return np.linalg.norm(L, axis=-1) - self.size[0]
        if self.kind == "box":
            q = np.abs(L) - np.asarray(self.size)
            return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)
        r, hh = self.size
        d = np.stack([np.linalg.norm(L[..., :2], axis=-1) - r, np.abs(L[..., 2]) - hh], -1)
        return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(-1), 0.0)

    def area(self):
        if self.kind == "sphere":
            return 4 * np.pi * self.size[0] ** 2
        if self.kind == "box":
            a, b, c = self.size
            return 8 * (a * b + b * c + a * c)
        r, hh = self.size
        return 2 * np.pi * r * (2 * hh) + 2 * np.pi * r * r

    def surface(self, n, rng):
        if self.kind == "sphere":
            v = rng.normal(size=(n, 3))
            return self.pos + (v / np.linalg.norm(v, axis=1, keepdims=True)) * self.size[0]
        if self.kind == "box":
            h = np.asarray(self.size)
            ar = np.array([h[1] * h[2], h[0] * h[2], h[0] * h[1]])
            ax = rng.choice(3, size=n, p=ar / ar.sum())
            L = rng.uniform(-1, 1, (n, 3)) * h
            sgn = np.where(rng.random(n) < 0.5, -1.0, 1.0)
            L[np.arange(n), ax] = sgn * h[ax]
            return self.pos + L @ self.R.T
        r, hh = self.size
        a_side, a_cap = 2 * np.pi * r * 2 * hh, np.pi * r * r
        u = rng.random(n) * (a_side + 2 * a_cap)
        L = np.zeros((n, 3))
        side = u < a_side
        th = rng.uniform(0, 2 * np.pi, n)
        rho = r * np.sqrt(rng.random(n))
        L[side] = np.stack([r * np.cos(th), r * np.sin(th), rng.uniform(-hh, hh, n)], 1)[side]
        cap = ~side
        zc = np.where(u[cap] < a_side + a_cap, hh, -hh)
        L[cap] = np.stack([rho * np.cos(th), rho * np.sin(th), np.zeros(n)], 1)[cap]
        L[cap, 2] = zc
        return self.pos + L @ self.R.T

    def aabb(self):
        if self.kind == "sphere":
            e = np.full(3, self.size[0])
        elif self.kind == "box":
            e = np.abs(self.R) @ np.asarray(self.size)
        else:
            r, hh = self.size
            e = np.abs(self.R) @ np.array([r, r, hh])      # conservative
        return self.pos - e, self.pos + e


class Obj:
    """a union of primitives with a name, a type label and an integer id."""

    def __init__(self, name, kind, prims, oid=0):
        self.name, self.kind, self.prims, self.oid = name, kind, list(prims), oid

    def sdf(self, X):
        return np.min([p.sdf(X) for p in self.prims], axis=0)

    def contains(self, X):
        return self.sdf(X) < 0

    def aabb(self):
        bb = [p.aabb() for p in self.prims]
        return np.min([b[0] for b in bb], 0), np.max([b[1] for b in bb], 0)

    def surface(self, n, rng):
        """n points on the boundary of the union (uniform per area; samples buried inside another primitive are discarded)."""
        ar = np.array([p.area() for p in self.prims])
        out = []
        need, tries = n, 0
        while need > 0 and tries < 8:
            cnt = rng.multinomial(int(need * 1.5) + 8, ar / ar.sum())
            pts = np.concatenate([p.surface(c, rng) for p, c in zip(self.prims, cnt) if c > 0])
            if len(self.prims) > 1:
                keep = np.ones(len(pts), bool)
                k = 0
                for p, c in zip(self.prims, cnt):
                    if c > 0:
                        for q in self.prims:
                            if q is not p:
                                keep[k:k + c] &= q.sdf(pts[k:k + c]) > -5e-5
                        k += c
                pts = pts[keep]
            out.append(pts)
            need -= len(pts)
            tries += 1
        pts = np.concatenate(out)
        return pts[rng.permutation(len(pts))[:n]]

    def volume(self, rng, n=200000):
        lo, hi = self.aabb()
        X = rng.uniform(lo, hi, (n, 3))
        return float(self.contains(X).mean() * np.prod(hi - lo))

    def moved(self, x, y, yaw):
        """place a prototype (base at z=0, local frame) in the world: yaw about z, then translate to (x, y, Z0)."""
        R = Rz(yaw)
        t = np.array([x, y, Z0])
        return Obj(self.name, self.kind, [Prim(p.kind, p.size, R @ p.pos + t, R @ p.R) for p in self.prims], self.oid)


# ====================================================================================== object library (prototypes; base at z=0)
def _ring(n, r, half_t, half_h, z, half_w=None):
    """n thin box walls on a circle of radius r (outward normal = wall x axis), centre height z."""
    half_w = half_w if half_w is not None else r * np.tan(np.pi / n) * 1.05
    return [Prim("box", (half_t, half_w, half_h), np.array([r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), z]),
                 Rz(2 * np.pi * i / n)) for i in range(n)]


def proto(kind):
    P = lambda *a, **k: Prim(*a, **k)
    v = lambda *a: np.array(a, float)
    if kind == "ball":
        return Obj("ball", kind, [P("sphere", (0.03,), v(0, 0, 0.03))])
    if kind == "item":      # a candy-like sphere (a_loose_items: r = 7 mm) lying on the tray
        return Obj("item", kind, [P("sphere", (0.007,), v(0, 0, 0.007))])
    if kind == "box":
        return Obj("box", kind, [P("box", (0.04, 0.025, 0.02), v(0, 0, 0.02))])
    if kind == "block":     # c_block_tower block, 5 x 5 x 4 cm
        return Obj("block", kind, [P("box", (0.025, 0.025, 0.02), v(0, 0, 0.02))])
    if kind == "bottle":    # stepped bottle: body, shoulder, neck (non-convex outline, convex pieces)
        return Obj("bottle", kind, [P("cyl", (0.03, 0.07), v(0, 0, 0.07)), P("cyl", (0.02, 0.01), v(0, 0, 0.15)),
                                    P("cyl", (0.011, 0.025), v(0, 0, 0.185))])
    if kind == "mug":       # 12-gon thin wall + floor + C handle
        pr = _ring(12, 0.04, 0.002, 0.045, 0.048) + [P("cyl", (0.042, 0.003), v(0, 0, 0.003))]
        pr += [P("box", (0.022, 0.006, 0.006), v(0.062, 0, 0.075)), P("box", (0.006, 0.006, 0.03), v(0.078, 0, 0.05)),
               P("box", (0.022, 0.006, 0.006), v(0.062, 0, 0.025))]
        return Obj("mug", kind, pr)
    if kind == "L":         # L-shaped bracket, 10 x 10 cm footprint, 3 cm bars, 4 cm high
        return Obj("L", kind, [P("box", (0.05, 0.015, 0.02), v(0.0, -0.035, 0.02)), P("box", (0.015, 0.035, 0.02), v(-0.035, 0.015, 0.02))])
    if kind == "cup":       # scene_yor_cup_collect cup: floor disc + 12 box walls, r = 30 mm, h = 80 mm
        return Obj("cup", kind, [P("cyl", (0.030, 0.0015), v(0, 0, 0.0015))] + _ring(12, 0.030, 0.0015, 0.040, 0.040, 0.00785))
    if kind == "dish":      # a_loose_items dish: floor disc + 16 inclined walls (beta = 45 deg, rim h = 25 mm)
        b, h, rf, wt, ns = np.deg2rad(45.0), 0.025, 0.06, 0.008, 16
        L = h / np.cos(b)
        w = (rf + h * np.tan(b) + wt) * np.tan(np.pi / ns) * 1.15
        z0 = 0.008
        pr = [P("cyl", (rf + wt, 0.004), v(0, 0, z0 - 0.004))]
        xdir = np.array([np.cos(b), 0.0, -np.sin(b)])
        mid = np.array([rf + 0.5 * h * np.tan(b), 0.0, 0.5 * h]) + 0.5 * wt * xdir
        for i in range(ns):
            phi = 2 * np.pi * i / ns
            pr.append(P("box", (wt / 2, w, L / 2), Rz(phi) @ mid + v(0, 0, z0), Rz(phi) @ Ry(b)))
        return Obj("dish", kind, pr)
    if kind == "tray":
        return Obj("tray", kind, [P("box", (TRAY_HX, TRAY_HY, TRAY_T), v(0, 0, -TRAY_T))])
    raise KeyError(kind)


def bead(i, layer_z=None):
    """the 8 nominal beads (r = 8 mm) of the cup in two layers of 4, cup-local frame."""
    layer, k = divmod(i, 4)
    a = np.pi / 2 * k + (np.pi / 4 if layer else 0.0)
    z = 0.003 + 0.008 + layer * 0.0125
    return Obj(f"bead{i}", "bead", [Prim("sphere", (0.008,), np.array([0.0125 * np.cos(a), 0.0125 * np.sin(a), z]))])


def dish_item(i):
    """6 items (r = 7 mm) on the dish floor, centre + ring of 5."""
    z = 0.008 + 0.007
    if i == 0:
        p = np.array([0.0, 0.0, z])
    else:
        a = 2 * np.pi * (i - 1) / 5
        p = np.array([0.025 * np.cos(a), 0.025 * np.sin(a), z])
    return Obj(f"item{i}", "item", [Prim("sphere", (0.007,), p)])


# ====================================================================================== scenes
class Scene:
    """tray + a list of placed objects. objs[k].oid = k + 1 (0 = tray). Builds the MuJoCo model used for ray casting."""

    def __init__(self, objs, with_tray=True):
        import mujoco
        self.tray = proto("tray").moved(0.0, 0.0, 0.0)
        self.tray.oid = 0
        self.objs = []
        for k, o in enumerate(objs):
            o.oid = k + 1
            self.objs.append(o)
        self.all = ([self.tray] if with_tray else []) + self.objs
        geoms, labels = [], []
        q = np.zeros(4)
        for o in self.all:
            for j, p in enumerate(o.prims):
                mujoco.mju_mat2Quat(q, p.R.reshape(9))
                sz = " ".join("%.6f" % x for x in p.size)
                typ = {"box": "box", "cyl": "cylinder", "sphere": "sphere"}[p.kind]
                geoms.append(f'<geom name="o{o.oid}_{j}" type="{typ}" size="{sz}" pos="%.6f %.6f %.6f" quat="%.8f %.8f %.8f %.8f" '
                             f'contype="0" conaffinity="0"/>' % (*p.pos, *q))
                labels.append(o.oid)
        xml = "<mujoco><worldbody>" + "".join(geoms) + "</worldbody></mujoco>"
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.geom_label = np.array(labels, np.int32)
        self.radius = 0.0                     # duck-typing for depth_sim.DepthCamera (unused here)

    def center_radius(self):
        lo = np.min([o.aabb()[0] for o in self.all], 0)
        hi = np.max([o.aabb()[1] for o in self.all], 0)
        return (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)


def pair_distances(objs, rng, n=4000):
    """true min distance between every pair of objects (sampled surface of A against the exact SDF of B; sampling error ~ mm)."""
    S = [o.surface(n, rng) for o in objs]
    D = {}
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            D[(i, j)] = float(min(objs[j].sdf(S[i]).min(), objs[i].sdf(S[j]).min()))
    return D
