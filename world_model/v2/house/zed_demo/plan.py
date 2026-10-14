"""plan.py -- use the perceived (inflated) zonotopes to choose a free spot on the desk and plan the carry path.  SIMULATION ONLY.

Task geometry.  The tray is held by both hands (hold pose, arms regulated by FL-MPC); the mocap base is holonomic and translates the whole payload
(tray + hands + arms + chassis).  With the yaw fixed, the payload is a rigid set P_i (a zonotope per link, fitted from the robot's OWN collision
geoms in the base frame) translated by t = (tx, ty).  For an obstacle zonotope O_j:
        dist(P_i + t, O_j) > 0   <=>   t not in  Z_ij = O_j (+) (-P_i)        (Minkowski sum of zonotopes = a zonotope)
and the signed distance of t to Z_ij is bounded below by the H-rep value  max_k (a_k . t - b_k)  (exact inside, a lower bound outside).  The clearance
map C(t) = min_ij of that is therefore a CONSERVATIVE zonotope clearance of the whole payload at base offset t; the spot and the path are chosen
on it (spot = argmax of C, capped), then the final path is verified with the exact zonotope-zonotope distance (zonoutil.dist_union_union, BVLS).
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np
import mujoco

from world_model.v2.perception_zono.fit import fit_hull_zono
from world_model.v2.perception_zono.zonoutil import halfspaces, ZUnion, dist_union_union
from world_model.v2.tray_rep.sets.zono import Zonotope
from . import perceive as PC


# ------------------------------------------------------------------------------------------------ payload zonotopes (robot's own kinematics)
def payload_zonotopes(C, cls, order=2, base_xy=None, yaw=None):
    """one zonotope per robot body that has collision geoms (chassis bodies merged), in the WORLD frame at the current (hold) pose."""
    m, d = C.m, C.d
    mujoco.mj_forward(m, d)
    groups = {}
    for g in range(m.ngeom):
        if cls[g] not in "LRBTP" or not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        b = int(m.geom_bodyid[g])
        nm = m.body(b).name or f"b{b}"
        key = nm if cls[g] in "LR" else ("chassis" if cls[g] == "B" else "tray")
        groups.setdefault(key, []).append(g)
    out = []
    for key, gs in groups.items():
        P = np.concatenate([PC.geom_pts(m, d, g) for g in gs])
        if len(P) < 8:
            continue
        out.append((key, fit_hull_zono(P, order=order)))
    return out


# ------------------------------------------------------------------------------------------------ clearance map
class ClearanceMap:
    def __init__(self, payload, obstacles, xs, ys, reach=0.6):
        self.xs, self.ys = np.asarray(xs), np.asarray(ys)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        self.shape = X.shape
        C = np.full(len(self.pts), 1.0)
        who = np.full(len(self.pts), -1, np.int32)
        self.pairs = []
        for i, (pn, P) in enumerate(payload):
            for j, (on, O) in enumerate(obstacles):
                Zij = O.minkowski(P.neg())
                lo, hi = Zij.interval_hull()
                sel = np.flatnonzero((self.pts[:, 0] > lo[0] - reach) & (self.pts[:, 0] < hi[0] + reach) & (self.pts[:, 1] > lo[1] - reach) &
                                     (self.pts[:, 1] < hi[1] + reach) & (lo[2] - reach < 0.0) & (0.0 < hi[2] + reach))
                if len(sel) == 0:
                    continue
                A, b = halfspaces(Zij)
                v = (self.pts[sel] @ A.T - b).max(1)
                better = v < C[sel]
                C[sel[better]] = v[better]
                who[sel[better]] = len(self.pairs)
                self.pairs.append((pn, on))
        self.C = C.reshape(self.shape)
        self.who = who.reshape(self.shape)

    def at(self, t):
        """bilinear interpolation of C at offsets t (n,2)."""
        t = np.atleast_2d(t)
        fx = np.interp(t[:, 0], self.xs, np.arange(len(self.xs)))
        fy = np.interp(t[:, 1], self.ys, np.arange(len(self.ys)))
        i0, j0 = np.clip(np.floor(fx).astype(int), 0, len(self.xs) - 2), np.clip(np.floor(fy).astype(int), 0, len(self.ys) - 2)
        ax, ay = fx - i0, fy - j0
        c = self.C
        return (c[i0, j0] * (1 - ax) * (1 - ay) + c[i0 + 1, j0] * ax * (1 - ay) + c[i0, j0 + 1] * (1 - ax) * ay + c[i0 + 1, j0 + 1] * ax * ay)


# ------------------------------------------------------------------------------------------------ search
def astar(Cm: ClearanceMap, start_t, goal_t, c_min, c_soft=0.08, w_soft=6.0):
    xs, ys = Cm.xs, Cm.ys
    si = (int(np.argmin(np.abs(xs - start_t[0]))), int(np.argmin(np.abs(ys - start_t[1]))))
    gi = (int(np.argmin(np.abs(xs - goal_t[0]))), int(np.argmin(np.abs(ys - goal_t[1]))))
    C = Cm.C
    nx, ny = C.shape
    res = float(xs[1] - xs[0])
    dist = {si: 0.0}
    prev = {}
    pq = [(0.0, si)]
    nbrs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    while pq:
        f, u = heapq.heappop(pq)
        if u == gi:
            break
        if f > dist[u] + math.hypot(u[0] - gi[0], u[1] - gi[1]) * res + 1e-9:
            continue
        for a, b in nbrs:
            v = (u[0] + a, u[1] + b)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny) or C[v] < c_min:
                continue
            step = math.hypot(a, b) * res
            cost = step * (1.0 + w_soft * max(0.0, c_soft - C[v]) / c_soft)
            nd = dist[u] + cost
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd + math.hypot(v[0] - gi[0], v[1] - gi[1]) * res, v))
    if gi not in prev and gi != si:
        return None
    path = [gi]
    while path[-1] != si:
        path.append(prev[path[-1]])
    path = path[::-1]
    return np.array([[xs[i], ys[j]] for i, j in path])


def reachable(Cm: ClearanceMap, start_t, c_min):
    """boolean map of cells connected to the start through cells with C >= c_min (8-connected flood fill)."""
    from scipy.ndimage import label
    ok = Cm.C >= c_min
    lab, _ = label(ok, structure=np.ones((3, 3)))
    si = (int(np.argmin(np.abs(Cm.xs - start_t[0]))), int(np.argmin(np.abs(Cm.ys - start_t[1]))))
    if lab[si] == 0:
        return np.zeros_like(ok)
    return lab == lab[si]


def shortcut(Cm, path, c_min, step=0.01):
    """greedy line-of-sight smoothing on the continuous clearance map."""
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            n = max(int(np.linalg.norm(path[j] - path[i]) / step), 2)
            seg = path[i] + (path[j] - path[i]) * np.linspace(0, 1, n + 1)[:, None]
            if (Cm.at(seg) >= c_min).all():
                break
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


def smooth_path(Cm, wp, c_min, n=400):
    """C^1 spline through the shortcut waypoints (chord-length parameter), falling back to the polyline if the spline leaves the free set."""
    from scipy.interpolate import PchipInterpolator
    if len(wp) < 3:
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
        u = np.linspace(0, s[-1], n)
        return np.stack([np.interp(u, s, wp[:, 0]), np.interp(u, s, wp[:, 1])], 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
    u = np.linspace(0, s[-1], n)
    Q = np.stack([PchipInterpolator(s, wp[:, 0])(u), PchipInterpolator(s, wp[:, 1])(u)], 1)
    if (Cm.at(Q) >= c_min).all():
        return Q
    return np.stack([np.interp(u, s, wp[:, 0]), np.interp(u, s, wp[:, 1])], 1)


def exact_clearance(payload, obstacles, t):
    """exact zonotope-zonotope distance of the whole payload translated by t=(tx,ty) to the obstacle union (0 if intersecting) + the limiting pair."""
    best, who = 1e9, ("", "")
    for pn, P in payload:
        Pt = Zonotope(P.c + np.array([t[0], t[1], 0.0]), P.G)
        lo, hi = Pt.interval_hull()
        for on, O in obstacles:
            lo2, hi2 = O.interval_hull()
            gap = np.maximum(np.maximum(lo - hi2, lo2 - hi), 0.0)
            if np.linalg.norm(gap) >= best:
                continue
            dd = dist_union_union(ZUnion([Pt]), ZUnion([O]))
            if dd < best:
                best, who = dd, (pn, on)
    return best, who


def plan_task(payload, obstacles, support, tray_xy0, tray_half=(0.15, 0.10), res=0.02, win=((-0.95, 0.95), (-0.12, 0.62)), c_min=0.02, c_cap=0.12,
              edge_margin=0.03):
    """returns dict(Cm, spot_t, path_t, ...).  support: the desk-top slab zonotope (perceived).  Tray stays 'on the desk' = its xy rectangle inside the
    perceived desk-top hull (convex, shrunk by edge_margin)."""
    xs = np.arange(win[0][0], win[0][1] + 1e-9, res)
    ys = np.arange(win[1][0], win[1][1] + 1e-9, res)
    t0 = time.perf_counter()
    Cm = ClearanceMap(payload, obstacles, xs, ys)
    t_map = time.perf_counter() - t0
    reach = reachable(Cm, (0.0, 0.0), c_min)
    # tray-on-desk test: all 4 corners (tray yaw: long axis along world -x/+x at hold) inside the perceived desk-top hull
    from scipy.spatial import ConvexHull
    V = PC.zono_vertices(support)
    sh = ConvexHull(V[:, :2])
    eq = sh.equations
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    on_desk = np.ones_like(X, bool)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx = tray_xy0[0] + X + sx * tray_half[0]
            cy = tray_xy0[1] + Y + sy * tray_half[1]
            on_desk &= (eq[:, 0, None, None] * cx + eq[:, 1, None, None] * cy + eq[:, 2, None, None] + edge_margin <= 0).all(0)
    score = np.where(reach & on_desk, np.minimum(Cm.C, c_cap), -1.0)
    # among cells within 5 mm of the best capped score, take the one closest to the start (shortest carry)
    best = score.max()
    cand = np.argwhere(score >= best - 0.005)
    d0 = np.hypot(xs[cand[:, 0]], ys[cand[:, 1]])
    ci = cand[int(np.argmin(d0))]
    spot_t = np.array([xs[ci[0]], ys[ci[1]]])
    path = astar(Cm, (0.0, 0.0), spot_t, c_min)
    info = dict(Cm=Cm, reach=reach, on_desk=on_desk, score=score, spot_t=spot_t, spot_xy=np.array(tray_xy0) + spot_t, best_score=float(best),
                c_at_spot=float(Cm.C[ci[0], ci[1]]), xs=xs, ys=ys, t_map_s=t_map, path_grid=path, n_feasible=int((reach & on_desk).sum()))
    if path is not None:
        wp = shortcut(Cm, path, c_min)
        info["path_wp"] = wp
        info["path_t"] = smooth_path(Cm, wp, c_min)
    return info
