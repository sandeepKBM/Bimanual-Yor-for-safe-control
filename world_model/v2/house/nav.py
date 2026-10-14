"""nav.py -- 2-D route planning for the mocap YOR base in a MolmoSpaces house (Phase 2, SIMULATION only).

Pipeline (docs/status/molmospaces_integration_plan.md section 3.4):
  1. occupancy   fine 1 cm grid from UPWARD rays (house collision groups 3+4, static furniture only).  Upward from below
                 the floor so that a start point inside a wall box cannot read as "free"; thin decals (top < 4 cm,
                 rugs) are skipped; anything whose underside is < 1.15 m (robot top 1.05 m) is an obstacle.
                 Coarse grid (5 cm, the plan's resolution) = 5x5 max-pool of the fine grid (conservative).
  2. footprint   the REAL off-centre hold-pose footprint (phase1.footprint: robot-frame x [-0.165, 0.577], y
                 [-0.644, 0.130], geom AABBs, conservative), heading-aware (8 headings, 45 deg).
  3. door        the doorway is crossed on a straight, door-normal segment whose lateral offset is chosen on the 1 cm
                 grid (max clearance); a 5 cm raster cannot resolve a 0.884 m gap for a 0.774 m footprint.
  4. A*          coarse, state (cell, heading), moves = forward along heading or +-45 deg in place, big turn penalty
                 (few turns).  start -> pre-door pose, post-door pose -> goal.
  5. smoothing   corners filleted by arcs (largest radius that stays valid on the 1 cm grid), then
                 trajectory = time parameterisation with speed, accel and jerk caps (trapezoid in arc length,
                 curvature speed limit, then two box filters in time => bounded jerk).  The EXECUTED samples are
                 re-validated on the fine grid.
Nothing here imports YOR/robot/base.py or touches hardware.
"""
from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass, field

import numpy as np
import mujoco
from scipy import ndimage

from . import assets as A
from . import build as B

CACHE = A.HOUSE_DATA.parent / "nav_cache"
Z_BASE = 0.04                         # geoms whose top is below this (rugs, decals) are not obstacles
ROBOT_TOP = 1.15                      # obstacle band upper edge (robot top 1.05 m)
NTH = 8                               # headings, 45 deg apart
DEFAULT_FP = dict(x=[-0.1645576, 0.5770901], y=[-0.6438942, 0.1303877])   # phase1 hold footprint (house 17)


# ------------------------------------------------------------------------------------------------- occupancy
@dataclass
class Grid:
    res: float
    x0: float
    y0: float
    occ: np.ndarray                   # [ny, nx] bool
    edt: np.ndarray | None = None     # [ny, nx] distance to nearest occupied cell centre [m]

    def cell(self, x, y):
        return (np.floor((np.asarray(x) - self.x0) / self.res).astype(int),
                np.floor((np.asarray(y) - self.y0) / self.res).astype(int))

    def with_edt(self):
        self.edt = ndimage.distance_transform_edt(~self.occ) * self.res
        return self


def _geom_top(m, d):
    top = np.zeros(m.ngeom)
    for g in range(m.ngeom):
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        top[g] = c[2] + (np.abs(R) @ m.geom_aabb[g, 3:])[2]
    return top


def house_bounds(idx, margin=0.3):
    P = np.concatenate([np.array(p) for _, _, p in B.room_polys(idx)])
    return P[:, 0].min() - margin, P[:, 1].min() - margin, P[:, 0].max() + margin, P[:, 1].max() + margin


def build_occupancy(idx, res=0.01, use_cache=True, m=None, d=None):
    """Fine occupancy grid of the (static) house. Cached in HOUSE_DATA/../nav_cache. ~1 min per house at 1 cm."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"occ_{idx}_{res:.3f}.npz"
    x0, y0, x1, y1 = house_bounds(idx)
    x0, y0 = math.floor(x0 * 20) / 20, math.floor(y0 * 20) / 20
    if use_cache and f.exists():
        z = np.load(f)
        return Grid(float(z["res"]), float(z["x0"]), float(z["y0"]), z["occ"]).with_edt()
    if m is None:
        m, d = B.load_house(idx, True)
        mujoco.mj_forward(m, d)
    top = _geom_top(m, d)
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    occ = np.zeros((ny, nx), bool)
    mask = np.array([0, 0, 0, 1, 1, 0], np.uint8)
    up = np.array([0.0, 0.0, 1.0])
    gid = np.zeros(1, np.int32)
    ray = mujoco.mj_ray
    for iy in range(ny):
        y = y0 + (iy + 0.5) * res
        for ix in range(nx):
            x = x0 + (ix + 0.5) * res
            z0 = -0.3
            for _ in range(6):                       # skip thin decals stacked under real obstacles
                dist = ray(m, d, np.array([x, y, z0]), up, mask, 1, -1, gid)
                if dist < 0:
                    break
                zh = z0 + dist
                if zh >= ROBOT_TOP:
                    break
                if top[int(gid[0])] > Z_BASE:
                    occ[iy, ix] = True
                    break
                z0 = zh + 0.002
    np.savez_compressed(f, res=res, x0=x0, y0=y0, occ=occ)
    return Grid(res, x0, y0, occ).with_edt()


def coarsen(F: Grid, k=5):
    """max-pool by k (conservative): coarse cell occupied if ANY fine cell inside is."""
    ny, nx = F.occ.shape
    py, px = (-ny) % k, (-nx) % k
    o = np.pad(F.occ, ((0, py), (0, px)))
    c = o.reshape(o.shape[0] // k, k, o.shape[1] // k, k).any(axis=(1, 3))
    return Grid(F.res * k, F.x0, F.y0, c)


# ------------------------------------------------------------------------------------------------- footprint
def rect_pts(fp, pad=0.0, step=0.01, interior=0.02):
    """Robot-frame sample points: boundary (step) and interior lattice (interior)."""
    x0, x1 = fp["x"][0] - pad, fp["x"][1] + pad
    y0, y1 = fp["y"][0] - pad, fp["y"][1] + pad
    bx = np.arange(x0, x1 + 1e-9, step)
    by = np.arange(y0, y1 + 1e-9, step)
    bnd = np.concatenate([np.stack([bx, np.full_like(bx, y0)], 1), np.stack([bx, np.full_like(bx, y1)], 1),
                          np.stack([np.full_like(by, x0), by], 1), np.stack([np.full_like(by, x1), by], 1)])
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1e-9, interior), np.arange(y0, y1 + 1e-9, interior))
    return bnd, np.stack([gx.ravel(), gy.ravel()], 1)


def poses_clearance(F: Grid, fp, poses, pad=0.0):
    """Fine-grid footprint test for poses (N,3)=(x,y,yaw). Returns (collide[N] bool, clearance[N] m): clearance =
    min over footprint sample points of the distance to the nearest obstacle sample (+-res accuracy; 0 if inside)."""
    poses = np.atleast_2d(np.asarray(poses, float))
    bnd, inn = rect_pts(fp, pad)
    pts = np.concatenate([bnd, inn])
    ny, nx = F.occ.shape
    out_c = np.zeros(len(poses), bool)
    out_d = np.zeros(len(poses))
    for k0 in range(0, len(poses), 64):
        P = poses[k0:k0 + 64]
        c, s = np.cos(P[:, 2])[:, None], np.sin(P[:, 2])[:, None]
        wx = P[:, 0:1] + c * pts[None, :, 0] - s * pts[None, :, 1]
        wy = P[:, 1:2] + s * pts[None, :, 0] + c * pts[None, :, 1]
        ix = np.floor((wx - F.x0) / F.res).astype(int)
        iy = np.floor((wy - F.y0) / F.res).astype(int)
        oob = (ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)
        dd = np.where(oob, 0.0, F.edt[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)])
        out_d[k0:k0 + 64] = dd.min(axis=1)
        out_c[k0:k0 + 64] = (dd <= 0.0).any(axis=1)
    return out_c, out_d


# ------------------------------------------------------------------------------------------------- coarse maps
def free_maps(C: Grid, fp, pad=0.0):
    """free[h] [ny,nx] bool: robot origin in this cell at heading h*45 deg has no occupied coarse cell under the
    footprint rectangle (dilated by the half-diagonal of a cell). Outside the grid counts as occupied."""
    res = C.res
    dil = 0.5 * res * math.sqrt(2) + pad
    x0, x1 = fp["x"][0] - dil, fp["x"][1] + dil
    y0, y1 = fp["y"][0] - dil, fp["y"][1] + dil
    R = int(math.ceil(1.2 / res))
    ii = np.arange(-R, R + 1)
    gx, gy = np.meshgrid(ii * res, ii * res)        # gx[j,i] = world x offset of the cell at (dy=ii[j], dx=ii[i])
    out = []
    for h in range(NTH):
        th = h * 2 * math.pi / NTH
        c, s = math.cos(th), math.sin(th)
        lx, ly = c * gx + s * gy, -s * gx + c * gy
        msk = (lx >= x0) & (lx <= x1) & (ly >= y0) & (ly <= y1)       # msk[dy+R, dx+R]: offset cell under footprint
        # collide(p) = OR_{o in msk} occ(p + o)  ==  dilation of occ by the reflected mask
        struct = msk[::-1, ::-1]
        col = ndimage.binary_dilation(C.occ, structure=struct, border_value=1)
        out.append(~col)
    return out


# ------------------------------------------------------------------------------------------------- A*
_STEP = [(int(round(math.cos(h * 2 * math.pi / NTH))), int(round(math.sin(h * 2 * math.pi / NTH)))) for h in range(NTH)]


def astar(C: Grid, free, start, goal, turn_pen=1.5):
    """start=(x,y,h|None), goal=(x,y,h|None).  Returns [(x,y,h), ...] cell centres or None.  h=None: free heading."""
    ny, nx = C.occ.shape
    sx, sy = (int(v) for v in C.cell(start[0], start[1]))
    gx, gy = (int(v) for v in C.cell(goal[0], goal[1]))
    hs = [start[2]] if start[2] is not None else list(range(NTH))
    heap, best, parent = [], {}, {}
    for h in hs:
        if free[h][sy, sx]:
            heapq.heappush(heap, (0.0, 0.0, (sx, sy, h)))
            best[(sx, sy, h)] = 0.0
    res = C.res
    while heap:
        f, g, st = heapq.heappop(heap)
        if g > best.get(st, 1e18) + 1e-12:
            continue
        x, y, h = st
        if (x, y) == (gx, gy) and (goal[2] is None or h == goal[2]):
            path = [st]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return [(float(C.x0 + (a + .5) * res), float(C.y0 + (b + .5) * res), hh) for a, b, hh in path[::-1]]
        dx, dy = _STEP[h]
        nbrs = []
        if 0 <= x + dx < nx and 0 <= y + dy < ny and free[h][y + dy, x + dx]:
            nbrs.append(((x + dx, y + dy, h), res * math.hypot(dx, dy)))
        for dh in (-1, 1):
            h2 = (h + dh) % NTH
            if free[h2][y, x]:
                nbrs.append(((x, y, h2), turn_pen))
        for s2, c in nbrs:
            g2 = g + c
            if g2 < best.get(s2, 1e18) - 1e-12:
                best[s2] = g2
                parent[s2] = st
                heapq.heappush(heap, (g2 + math.hypot(s2[0] - gx, s2[1] - gy) * res, g2, s2))
    return None


def corners(nodes):
    """Polyline vertices of an A* node list: start, every cell where the heading changes, goal."""
    pts = [nodes[0][:2]]
    for a, b in zip(nodes[:-1], nodes[1:]):
        if a[2] != b[2] and pts[-1] != a[:2]:
            pts.append(a[:2])
    if pts[-1] != nodes[-1][:2]:
        pts.append(nodes[-1][:2])
    return [tuple(p) for p in pts]


# ------------------------------------------------------------------------------------------------- doors
def door_list(idx, m, d):
    """Doors: centre xy, wall direction, normal, measured clear width (selftest.measure_doors), rooms on each side."""
    from . import selftest as S
    meas = {r["id"]: r for r in S.measure_doors(idx, m, d)}
    sj = {dr["id"]: dr for dr in A.spec_json(idx)["doors"]}
    polys = B.room_polys(idx)
    out = []
    for k, v in A.metadata(idx).items():
        if v["category"] not in ("Doorway", "Doorframe") or v["object_id"] not in sj:
            continue
        dr = sj[v["object_id"]]
        mm = re.match(r"wall\|\d+\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)", dr["wall0"])
        x0, z0, x1, z1 = (float(g) for g in mm.groups())
        wd = np.array([x1 - x0, z1 - z0]); wd /= np.linalg.norm(wd)
        c = d.xpos[m.body(next(iter(v["name_map"]["bodies"]))).id][:2].copy()
        n = np.array([-wd[1], wd[0]])
        sides = []
        for sgn in (+1, -1):
            p = c + sgn * 0.9 * n
            sides.append(next(((rid, rt) for rid, rt, poly in polys if B._in_poly(p, poly)), (None, None)))
        out.append(dict(id=v["object_id"], asset=dr["assetId"], centre=c, wall_dir=wd, normal=n,
                        side_pos=sides[0], side_neg=sides[1], bodies=list(v["name_map"]["bodies"]),
                        design_w=meas.get(v["object_id"], {}).get("design"),
                        clear_w=meas.get(v["object_id"], {}).get("measured")))
    return out


def door_passage(F: Grid, fp, door, travel_sign=+1, span=1.6, du_max=0.45, du_step=0.005, ds=0.01, pad=0.0):
    """Best straight crossing of one door along its normal (travel = travel_sign * normal).  Sweeps the lateral offset
    u of the robot ORIGIN; for each u the whole straight sweep s in [-span, span] (heading along travel) must be
    collision-free.  Returns dict(ok, u, clearance, heading, ...) -- clearance is the grid (EDT) value at the best u."""
    c, wd, n = door["centre"], door["wall_dir"], door["normal"] * travel_sign
    yaw = math.atan2(n[1], n[0])
    s = np.arange(-span, span + 1e-9, ds)
    best = dict(ok=False, u=None, clearance=-1.0)
    curve = []
    for u in np.arange(-du_max, du_max + 1e-9, du_step):
        P = np.stack([c[0] + wd[0] * u + n[0] * s, c[1] + wd[1] * u + n[1] * s, np.full_like(s, yaw)], 1)
        col, cl = poses_clearance(F, fp, P, pad)
        ok = not col.any()
        mc = float(cl.min())
        curve.append((float(u), mc if ok else 0.0))
        if ok and mc > best["clearance"]:
            best = dict(ok=True, u=float(u), clearance=mc, s_at_min=float(s[cl.argmin()]))
    best.update(heading=yaw, normal=n.tolist(), wall_dir=wd.tolist(), centre=c.tolist(), curve=curve)
    return best


# ------------------------------------------------------------------------------------------------- smoothing
def fillet(verts, R):
    """Polyline with circular arcs of radius R at interior vertices (tangent length capped at 45% of the adjacent
    segments, radius reduced to match). Returns (Q[N,2] resampled at 1 cm, s[N], corner info list)."""
    V = [np.array(v, float) for v in verts]
    pts = [V[0]]
    info = []
    for i in range(1, len(V) - 1):
        a, b, c = V[i - 1], V[i], V[i + 1]
        u1, u2 = b - a, c - b
        l1, l2 = np.linalg.norm(u1), np.linalg.norm(u2)
        u1, u2 = u1 / l1, u2 / l2
        dth = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], u1 @ u2)
        if abs(dth) < math.radians(1.0):
            pts.append(b)
            continue
        T = R * math.tan(abs(dth) / 2)
        Re = R
        cap = 0.45 * min(l1, l2)
        if T > cap:
            T = cap
            Re = T / math.tan(abs(dth) / 2)
        p0 = b - u1 * T
        cen = p0 + Re * np.array([-u1[1], u1[0]]) * np.sign(dth)
        a0 = math.atan2(p0[1] - cen[1], p0[0] - cen[0])
        na = max(int(abs(dth) * Re / 0.01), 4)
        for k in range(na + 1):
            ang = a0 + dth * k / na
            pts.append(cen + Re * np.array([math.cos(ang), math.sin(ang)]))
        info.append(dict(vertex=b.tolist(), turn_deg=math.degrees(dth), radius=float(Re)))
    pts.append(V[-1])
    P = np.array(pts)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9])
    P = P[keep]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    sn = np.append(np.arange(0, s[-1], 0.01), s[-1])
    return np.stack([np.interp(sn, s, P[:, 0]), np.interp(sn, s, P[:, 1])], 1), sn, info


def clean_polyline(verts, keep, tol_deg=3.0, dup=0.03):
    """Drop duplicate vertices (< dup apart) and (near-)collinear ones (< tol_deg turn); vertices whose index is in
    `keep` (door-axis points, endpoints) are never dropped."""
    V = [np.array(v, float) for v in verts]
    K = [i in keep for i in range(len(V))]
    changed = True
    while changed:
        changed = False
        for i in range(1, len(V) - 1):
            if K[i]:
                continue
            a, b, c = V[i - 1], V[i], V[i + 1]
            l1, l2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
            if l1 < dup or l2 < dup:
                drop = True
            else:
                u1, u2 = (b - a) / l1, (c - b) / l2
                drop = abs(math.degrees(math.atan2(u1[0] * u2[1] - u1[1] * u2[0], u1 @ u2))) < tol_deg
            if drop:
                del V[i]; del K[i]
                changed = True
                break
    return [tuple(v) for v in V]


def _box(x, w):
    xp = np.concatenate([np.full(w, x[0]), x, np.full(w, x[-1])])
    return np.convolve(xp, np.ones(w) / w, mode="same")[w:-w]


@dataclass
class Traj:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    caps: dict
    meta: dict = field(default_factory=dict)

    def derivs(self):
        dt = float(self.t[1] - self.t[0])
        vx, vy = np.gradient(self.x, dt), np.gradient(self.y, dt)
        ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)
        jx, jy = np.gradient(ax, dt), np.gradient(ay, dt)
        return dict(vx=vx, vy=vy, ax=ax, ay=ay, jx=jx, jy=jy, wz=np.gradient(np.unwrap(self.yaw), dt),
                    speed=np.hypot(vx, vy), acc=np.hypot(ax, ay), jerk=np.hypot(jx, jy))


def time_parameterize(Q, s, vmax=0.3, amax=0.25, jmax=None, lat_frac=0.85, dt=0.005):
    """Arc-length speed profile: v <= vmax, v <= sqrt(lat_frac*amax/kappa) (centripetal), tangential accel/decel
    0.9*amax; integrated in time; then two box filters of width Tf = amax/jmax on x(t), y(t) => bounded jerk.
    Achieved peak speed / |a| / jerk are MEASURED from the output (Traj.derivs) and reported by plan_route."""
    jmax = jmax if jmax is not None else amax / 0.6
    Tf = amax / jmax
    n = len(s)
    th = np.unwrap(np.arctan2(np.gradient(Q[:, 1], s), np.gradient(Q[:, 0], s)))
    kap = np.abs(np.gradient(th, s))
    kap = np.convolve(np.pad(kap, 5, mode="edge"), np.ones(11) / 11, mode="same")[5:-5]
    vlim = np.minimum(vmax, np.sqrt(lat_frac * amax / np.maximum(kap, 1e-6)))
    at = 0.9 * amax
    v = vlim.copy(); v[0] = 0.0; v[-1] = 0.0
    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * at * (s[i] - s[i - 1])))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * at * (s[i + 1] - s[i])))
    vm = np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)
    tn = np.concatenate([[0], np.cumsum(np.diff(s) / vm)])
    tt = np.arange(0, tn[-1] + 1e-9, dt)
    sx = np.interp(tt, tn, s)
    x, y = np.interp(sx, s, Q[:, 0]), np.interp(sx, s, Q[:, 1])
    w = max(int(round(Tf / dt)), 1)
    pre = int(round(0.3 / dt)) + 2 * w
    x = np.concatenate([np.full(pre, x[0]), x, np.full(pre, x[-1])])
    y = np.concatenate([np.full(pre, y[0]), y, np.full(pre, y[-1])])
    for _ in range(2):
        x, y = _box(x, w), _box(y, w)
    t = np.arange(len(x)) * dt
    sf = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    sf *= s[-1] / max(sf[-1], 1e-9)
    yaw = _box(np.interp(sf, s, th), max(w // 2, 1))
    return Traj(t, x, y, yaw, dict(vmax=vmax, amax=amax, jmax=jmax, lat_frac=lat_frac, t_filter=Tf),
                meta=dict(length_m=float(s[-1]), duration_s=float(t[-1]), dt=dt, lead_in_s=pre * dt))


def validate_traj(F: Grid, fp, traj: Traj, every=2, pad=0.0):
    idx = np.arange(0, len(traj.t), every)
    col, cl = poses_clearance(F, fp, np.stack([traj.x[idx], traj.y[idx], traj.yaw[idx]], 1), pad)
    return col, cl, idx


def _snap_to_line(p, p_line, n, tol=0.12):
    """project p onto the line through p_line with direction n -- only if p already lies within `tol` m of that line
    (i.e. the A* leg into p_line was the straight door run-in); otherwise keep p (the turn happens at p_line)."""
    n = np.asarray(n) / np.linalg.norm(n)
    d = np.asarray(p) - np.asarray(p_line)
    if abs(d[0] * n[1] - d[1] * n[0]) > tol:
        return tuple(p)
    return tuple(np.asarray(p_line) + n * (d @ n))


# ------------------------------------------------------------------------------------------------- top level
def plan_route(idx, start_xy, goal_xy, door, fp=None, vmax=0.3, amax=0.25, jmax=None, dt=0.005, clear_min=0.05,
               F: Grid | None = None, C: Grid | None = None, travel_sign=None, verbose=True,
               fillet_radii=(0.6, 0.45, 0.3, 0.2), approach=1.2, turn_pen=1.5, min_radius=0.2, tray_off=(0.45, -0.21), extras=(0.8, 0.6, 0.4, 0.2, 0.0), pads=(0.0, 0.05, 0.10)):
    """Plan start -> door -> goal.  `door` = an entry of door_list().  Always returns a dict; out['route'] is None with
    out['reason'] when there is no route (a valid result)."""
    fp = fp or DEFAULT_FP
    F = F or build_occupancy(idx)
    C = C or coarsen(F, 5)
    out = dict(house=idx, door=door["id"], start=list(map(float, start_xy)), goal=list(map(float, goal_xy)), fp=fp,
               res_coarse=C.res, res_fine=F.res, route=None, door_clear_w_m=door["clear_w"],
               door_design_w_m=door["design_w"], door_centre=door["centre"].tolist(),
               fp_width_m=fp["y"][1] - fp["y"][0], fp_depth_m=fp["x"][1] - fp["x"][0])
    if travel_sign is None:
        travel_sign = -1 if float((np.array(start_xy) - door["centre"]) @ door["normal"]) > 0 else +1
    out["travel_sign"] = travel_sign
    free = free_maps(C, fp)
    dp = door_passage(F, fp, door, travel_sign)
    out["door_passage"] = {k: v for k, v in dp.items() if k != "curve"}
    out["door_u_curve"] = dp["curve"]
    h_door = int(round(dp["heading"] / (2 * math.pi / NTH))) % NTH
    n, wd, c = np.array(dp["normal"]), np.array(dp["wall_dir"]), np.array(dp["centre"])
    # 5 cm raster check of the same straight crossing (ignoring the u optimum: centre line + best u)
    for tag, u in (("centre", 0.0), ("best", dp["u"] if dp["ok"] else 0.0)):
        cc = np.array([c + wd * u + n * s for s in np.arange(-1.0, 1.0001, 0.01)])
        ix, iy = C.cell(cc[:, 0], cc[:, 1])
        out[f"door_coarse_5cm_free_{tag}"] = bool(all(free[h_door][b, a] for a, b in zip(ix, iy)))
    if not dp["ok"]:
        out["reason"] = "no route: footprint does not fit through the door on the 1 cm grid"
        return out
    if dp["clearance"] < clear_min:
        out["reason"] = f"no route: best door clearance {dp['clearance']:.3f} m < required {clear_min} m"
        return out
    u = dp["u"]
    cands, tried = [], []
    fm = {}
    for pad, extra in [(pp, ee) for pp in pads for ee in extras]:   # coarse footprint pad x run-in length
        if pad not in fm:
            fm[pad] = free_maps(C, fp, pad)
        free = fm[pad]
        gs = abs(float((np.array(start_xy) - c) @ n))      # start/goal distance from the door plane along the axis
        ge = abs(float((np.array(goal_xy) - c) @ n))
        lat_s = abs(float((np.array(start_xy) - c) @ wd) - u)
        lat_g = abs(float((np.array(goal_xy) - c) @ wd) - u)
        di = min(approach + extra, max(gs - 0.5, 0.6)) if lat_s < 0.5 else approach + extra
        de = min(approach + extra, max(ge - 0.5, 0.6)) if lat_g < 0.5 else approach + extra
        P1 = c + wd * u - n * di
        P3 = c + wd * u + n * de
        n1 = astar(C, free, (start_xy[0], start_xy[1], None), (P1[0], P1[1], h_door), turn_pen)
        n2 = astar(C, free, (P3[0], P3[1], h_door), (goal_xy[0], goal_xy[1], None), turn_pen)
        if n1 is None or n2 is None:
            tried.append(dict(pad=pad, extra=extra, astar=False))
            continue
        v1, v2 = corners(n1), corners(n2)
        a1 = v1[:-1]
        if len(a1) >= 2:
            a1[-1] = _snap_to_line(a1[-1], P1, n)
        b2 = v2[1:]
        if len(b2) >= 2:
            b2[0] = _snap_to_line(b2[0], P3, n)
        verts = [tuple(start_xy)] + a1[1:] + [tuple(P1), tuple(P3)] + b2[:-1] + [tuple(goal_xy)]
        i1 = verts.index(tuple(P1)); i3 = verts.index(tuple(P3))
        verts = clean_polyline(verts, keep={0, i1, i3, len(verts) - 1})
        out["polyline"] = [list(map(float, v)) for v in verts]       # last attempt (best replaces it below)
        for R in fillet_radii:
            Q, s_, info = fillet(verts, R)
            traj = time_parameterize(Q, s_, vmax, amax, jmax, dt=dt)
            col, cl, ix = validate_traj(F, fp, traj)
            rmin = min([i["radius"] for i in info] or [9.0])
            ok = (not col.any()) and rmin >= 0.99 * min_radius
            tried.append(dict(pad=pad, extra=extra, R=R, colliding=int(col.sum()), min_clear=float(cl.min()), min_corner_r=rmin,
                              length=traj.meta["length_m"], turns=len([i for i in info if abs(i["turn_deg"]) > 10]), ok=ok))
            if verbose:
                print(f"  pad={pad} extra={extra} R={R}: colliding {int(col.sum())}/{len(col)} min clear {cl.min():.3f} "
                      f"min corner r {rmin:.2f} len {traj.meta['length_m']:.2f} ok={ok}", flush=True)
            if ok:
                cands.append(dict(pad=pad, extra=extra, R=R, info=info, traj=traj, col=col, cl=cl, ix=ix, verts=verts,
                                  turns=len([i for i in info if abs(i["turn_deg"]) > 10])))
    out["tried"] = tried
    if not cands:
        out["reason"] = ("no route: no collision-free trajectory with corner radius >= %.2f m on the 1 cm grid "
                         "(A* ok for %d/%d run-in lengths)" % (min_radius, len({t['extra'] for t in tried if t.get('astar', True)}), len(extras)))
        return out
    best = min(cands, key=lambda q: (q["turns"], -min(float(q["cl"].min()), 0.15), q["traj"].meta["length_m"]))
    out["polyline"] = [list(map(float, v)) for v in best["verts"]]
    out["chosen"] = dict(pad=best["pad"], extra=best["extra"], R=best["R"])
    R, info, traj, col, cl, ix = best["R"], best["info"], best["traj"], best["col"], best["cl"], best["ix"]
    dv = traj.derivs()
    # accel of the carried tray (hold pose: tray centre at (0.45, -0.21) in the robot frame; long axis = robot y): the
    # base-origin accel understates it because yaw rate / yaw accel act on a 0.5 m lever
    cy, sy_ = np.cos(traj.yaw), np.sin(traj.yaw)
    tx = traj.x + cy * tray_off[0] - sy_ * tray_off[1]
    ty = traj.y + sy_ * tray_off[0] + cy * tray_off[1]
    dts = float(traj.t[1] - traj.t[0])
    at_ = np.gradient(np.gradient(np.stack([tx, ty], 1), dts, axis=0), dts, axis=0)
    a_long = -sy_ * at_[:, 0] + cy * at_[:, 1]
    a_short = cy * at_[:, 0] + sy_ * at_[:, 1]
    near = np.hypot(traj.x[ix] - c[0], traj.y[ix] - c[1]) < 1.3
    out["route"] = dict(traj=traj, fillet_radius=R, corners=info, n_turns=len([i for i in info if abs(i["turn_deg"]) > 10]),
                        min_clearance_all_m=float(cl.min()),
                        min_clearance_door_zone_m=float(cl[near].min()) if near.any() else None,
                        length_m=traj.meta["length_m"], duration_s=traj.meta["duration_s"],
                        peak_speed=float(dv["speed"].max()), peak_acc=float(dv["acc"].max()),
                        peak_jerk=float(dv["jerk"].max()), peak_wz=float(np.abs(dv["wz"]).max()),
                        peak_tray_acc_long=float(np.abs(a_long).max()), peak_tray_acc_short=float(np.abs(a_short).max()),
                        peak_tray_acc=float(np.hypot(*at_.T).max()),
                        caps=traj.caps, door_pass_u=u, travel_heading_deg=math.degrees(dp["heading"]),
                        end_err_m=float(math.hypot(traj.x[-1] - goal_xy[0], traj.y[-1] - goal_xy[1])))
    dz = out["route"]["min_clearance_door_zone_m"]
    out["reason"] = "ok" if (dz is None or dz >= clear_min) else "route exists but door-zone grid clearance below requirement"
    return out


def best_free_pose(F: Grid, C: Grid, fp, poly, heading=0, step=0.10, pad=0.0):
    """Pose in a room polygon with the largest fine-grid clearance for the footprint at 45deg*heading."""
    free = free_maps(C, fp, pad)
    P = np.array(poly)
    best = (-1.0, None)
    for x in np.arange(P[:, 0].min() + 0.2, P[:, 0].max() - 0.2, step):
        for y in np.arange(P[:, 1].min() + 0.2, P[:, 1].max() - 0.2, step):
            if not B._in_poly((x, y), poly):
                continue
            ix, iy = C.cell(x, y)
            if not free[heading][int(iy), int(ix)]:
                continue
            col, cl = poses_clearance(F, fp, [[x, y, heading * 2 * math.pi / NTH]])
            if not col[0] and cl[0] > best[0]:
                best = (float(cl[0]), (float(x), float(y)))
    return best
