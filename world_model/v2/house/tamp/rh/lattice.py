"""tamp/rh/lattice.py -- LatticeProposer: motion-primitive lattice for the receding-horizon base planner (tamp/rh/api.py).  SIMULATION ONLY.

Candidates are gatekeeper Trajs  (chained nominal pieces + traj.backup_from at the end; Traj.t_sw = backup start, ends at rest).  The proposer NEVER certifies: api.admit does
(person_model="pred", walls, cup, hw).  The only checks here are a loose wall-footprint prune (EDT > prune_wall_m, much looser than the certificate's d_wall + slack) so the
beam does not waste its width on wall-crossing sequences, and the cup/hw rows that hold BY CONSTRUCTION (traj._step with L = gk.plan_frac * a_cup_max, as plan_candidate).

Primitive = constant body-twist target held PIECE_S = 0.5 s, reached with the cup-safe step of the gatekeeper (traj.cup_safe_target + traj._step).  Translation primitives are
expressed in the GUIDE frame  d = -grad(cost-to-go) at the piece start (unit, world frame), n = d rotated +90 deg:  u_world = a d + b n,  u_body = R(-yaw) u_world:
    G   : a in (0.3, 0.2, 0.12, 0.06) m/s, b = 0, wz = 0                         forward along the guide (0.06 = creep for the final approach)
    L   : a = 0, b = +-0.1, +-0.2                                               holonomic sidestep
    D   : (a, b) = (0.2, +-0.1), (0.12, +-0.1)                                  diagonals
    R   : a = -0.12                                                             reverse
    Y   : a = b = 0, wz = +-0.3 rad/s                                           turn in place
    A   : a = 0.2, wz = +-0.3 rad/s                                             arcs
    T   : a in (0.3, 0.2, 0.12), wz = clip(1.5 * (guide heading - yaw), +-0.3)  forward while turning the body (+x axis) toward the guide heading
    H   : hold still
All translation speeds are capped at cfg.v_max (scaled down if a primitive would exceed it).  Search: beam over sequences of depth <= horizon_s / PIECE_S; every node (depth >= 1) is a
candidate (shorter sequences are the fall-backs when the long ones are not admitted).  Ranking = cost-to-go at the END OF THE NOMINAL (+ a weak heading term, + an overshoot term
that is zero unless the rest pose is farther from the goal than the nominal end).  Cost-to-go = Dijkstra distance (8-connected, octile) to the goal on the scene occupancy grid,
inflated by the footprint half-width (INFL), computed once per (grid, goal); bilinear + exact at the goal (the lattice nodes are aligned to the goal).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ...gatekeeper.traj import Traj, _step, backup_from, cup_safe_target
from ...teleop import geom as GM
from ..segments import _integrate, wrap
from .api import Candidate, RHConfig, State

PIECE_S = 0.5
INFL = 0.40                  # m, cost-field obstacle inflation (footprint half-width ~0.39 m; the field is for the base ORIGIN, the certificate does the exact footprint)
FIELD_RES = 0.025            # m, Dijkstra lattice resolution
W_YAW = 0.30                 # m/rad, penalty on |wrap(yaw - guide heading)| (the footprint is a 0.74 x 0.77 m rectangle hanging 0.64 m to one side of the origin: a robot that
                             # drives the corridor at a skew heading sweeps a much wider strip, which the person / wall rows then refuse)
YAW_TRACK_K, YAW_TRACK_MAX = 1.5, 0.3
W_OVERSHOOT = 1.0            # weight of max(0, cost(rest pose) - cost(nominal end))
PRUNE_WALL_M = 0.04          # m, loose EDT prune on footprint boundary points along the nominal (the certificate uses d_wall + slack)
YAW_FREE_R = 0.30            # m, no heading term within this distance of the goal (the guide direction degenerates there)
BEAM = 40
N_OUT = 30
GOAL_STOP_M = 0.04           # m: if the cup-safe stop from the state ends within this of the goal, "stop now" is ranked FIRST (otherwise it is LAST); the loop's goal tolerance is 0.05


@dataclass(frozen=True)
class Prim:
    name: str
    a: float                 # speed along the guide direction [m/s] (negative = reverse)
    b: float                 # speed along the guide normal (left +) [m/s]
    wz: float                # yaw rate [rad/s]
    k_yaw: float = 0.0       # > 0: heading-tracking primitive, wz = clip(k_yaw * wrap(guide heading - yaw), +-YAW_TRACK_MAX) (wz is ignored)


def default_primitives() -> list:
    P = []
    for s in (0.3, 0.2, 0.12, 0.06):
        P.append(Prim(f"G{s:g}", s, 0.0, 0.0))
    for b in (0.1, -0.1, 0.2, -0.2):
        P.append(Prim(f"L{b:+g}", 0.0, b, 0.0))
    for a in (0.2, 0.12):
        for b in (0.1, -0.1):
            P.append(Prim(f"D{a:g}{b:+g}", a, b, 0.0))
    P.append(Prim("R0.12", -0.12, 0.0, 0.0))
    for w in (0.3, -0.3):
        P.append(Prim(f"Y{w:+g}", 0.0, 0.0, w))
    for w in (0.3, -0.3):
        P.append(Prim(f"A0.2{w:+g}", 0.2, 0.0, w))
    for s in (0.3, 0.2, 0.12):
        P.append(Prim(f"T{s:g}", s, 0.0, 0.0, YAW_TRACK_K))
    P.append(Prim("H", 0.0, 0.0, 0.0))
    return P


# ---------------------------------------------------------------------------------------------------------------- cost-to-go
class CostField:
    """distance-to-goal field of the BASE ORIGIN on the occupancy grid (obstacles inflated by `infl`), Dijkstra on an 8-connected lattice aligned to the goal."""

    def __init__(self, grid, goal_xy, infl: float = INFL, res: float = FIELD_RES, w_blocked: float = 3.0):
        from scipy import ndimage
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        self.res, self.goal = float(res), np.asarray(goal_xy, float)[:2].copy()
        gx0, gy0 = float(grid.x0), float(grid.y0)
        ny, nx = grid.occ.shape
        gx1, gy1 = gx0 + nx * grid.res, gy0 + ny * grid.res
        kx0, kx1 = math.ceil((gx0 - self.goal[0]) / res), math.floor((gx1 - self.goal[0]) / res)
        ky0, ky1 = math.ceil((gy0 - self.goal[1]) / res), math.floor((gy1 - self.goal[1]) / res)
        self.x_c = self.goal[0] + np.arange(kx0, kx1 + 1) * res                     # node coordinates, goal exactly on a node
        self.y_c = self.goal[1] + np.arange(ky0, ky1 + 1) * res
        self.ix_g, self.iy_g = -kx0, -ky0
        X, Y = np.meshgrid(self.x_c, self.y_c)
        free = grid.lookup(X, Y) >= infl
        ny_c, nx_c = free.shape
        if not free[self.iy_g, self.ix_g]:
            raise ValueError(f"goal {self.goal} is inside the inflated obstacles (infl={infl})")
        idx = np.arange(ny_c * nx_c).reshape(ny_c, nx_c)
        rows, cols, w = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ys0, ys1 = 0, ny_c - dy
            xs0, xs1 = (0, nx_c - dx) if dx >= 0 else (-dx, nx_c)
            a = idx[ys0:ys1, xs0:xs1]
            b = idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            ok = free[ys0:ys1, xs0:xs1] & free[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            rows.append(a[ok]); cols.append(b[ok]); w.append(np.full(int(ok.sum()), res * math.hypot(dx, dy)))
        g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(ny_c * nx_c,) * 2).tocsr()
        d = dijkstra(g, directed=False, indices=int(idx[self.iy_g, self.ix_g]))
        C = d.reshape(ny_c, nx_c)
        C[~np.isfinite(C)] = 1e3
        C[~free] = np.nan
        dist, (iy, ix) = ndimage.distance_transform_edt(~free, return_indices=True)       # blocked cells: nearest free cell's cost + w * distance
        Cb = C[iy, ix] + w_blocked * dist * res
        self.C = np.where(free, C, Cb)
        self.free = free
        self._ndi = ndimage
        gy, gx = np.gradient(self.C, res)                                                   # descent direction = -grad (central differences on the lattice)
        self.GX, self.GY = -gx, -gy
        self._x0, self._y0 = float(self.x_c[0]), float(self.y_c[0])
        self._nx, self._ny = self.C.shape[1], self.C.shape[0]

    def _bil(self, A, fx, fy):
        i, j = int(fx), int(fy)
        i = 0 if i < 0 else (self._nx - 2 if i > self._nx - 2 else i)
        j = 0 if j < 0 else (self._ny - 2 if j > self._ny - 2 else j)
        tx, ty = min(max(fx - i, 0.0), 1.0), min(max(fy - j, 0.0), 1.0)
        return (A[j, i] * (1 - tx) * (1 - ty) + A[j, i + 1] * tx * (1 - ty) + A[j + 1, i] * (1 - tx) * ty + A[j + 1, i + 1] * tx * ty)

    def cost_guide(self, x: float, y: float):
        """-> (cost-to-go, unit descent direction (gx, gy)) at (x, y); direction (0, 0) where the field is flat.  Pure-python bilinear (hot path of the beam search)."""
        fx, fy = (x - self._x0) / self.res, (y - self._y0) / self.res
        c = float(self._bil(self.C, fx, fy))
        gx, gy = float(self._bil(self.GX, fx, fy)), float(self._bil(self.GY, fx, fy))
        n = math.hypot(gx, gy)
        return c, ((gx / n, gy / n) if n > 1e-9 else (0.0, 0.0))

    def value(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        cx = (x - self.x_c[0]) / self.res
        cy = (y - self.y_c[0]) / self.res
        return self._ndi.map_coordinates(self.C, [np.atleast_1d(cy), np.atleast_1d(cx)], order=1, mode="nearest").reshape(np.shape(x))

    def v(self, x: float, y: float) -> float:
        return self.cost_guide(x, y)[0]

    def guide(self, x: float, y: float):
        """unit descent direction (world) of the field at (x, y); zero vector where the field is flat."""
        return np.array(self.cost_guide(x, y)[1])


_FIELD_CACHE: dict = {}


def cost_field(grid, goal_xy, infl: float = INFL, res: float = FIELD_RES) -> CostField:
    key = (id(grid), round(float(goal_xy[0]), 4), round(float(goal_xy[1]), 4), infl, res)
    f = _FIELD_CACHE.get(key)
    if f is None:
        f = _FIELD_CACHE[key] = CostField(grid, goal_xy, infl, res)
        f._grid_ref = grid                                                                  # keep the grid alive so id() stays unique
    return f


# ---------------------------------------------------------------------------------------------------------------- proposer
@dataclass
class _Node:
    pose: tuple
    v: tuple
    V: list                      # twist samples from the state (list of 3-lists), length 1 + 10 depth
    names: tuple
    cost: float                  # ranking cost at the end of the nominal
    cost_pure: float             # field value only
    guide: tuple = (0.0, 0.0)    # unit descent direction at `pose` (world)


class LatticeProposer:
    name = "lattice"

    def __init__(self, beam: int = BEAM, n_out: int = N_OUT, prims=None, prune_wall_m: float = PRUNE_WALL_M, w_yaw: float = W_YAW, infl: float = INFL):
        self.beam, self.n_out = int(beam), int(n_out)
        self.prims = list(prims) if prims is not None else default_primitives()
        self.prune_wall_m, self.w_yaw, self.infl = float(prune_wall_m), float(w_yaw), float(infl)
        self._fp = GM.boundary_pts(step=0.10)
        self.last_stats: dict = {}

    def params(self) -> dict:
        return dict(beam=self.beam, n_out=self.n_out, piece_s=PIECE_S, prims=[p.name for p in self.prims], prune_wall_m=self.prune_wall_m, w_yaw=self.w_yaw,
                    infl=self.infl, field_res=FIELD_RES, w_overshoot=W_OVERSHOOT)

    # ---------------------------------------------------------------- helpers
    def _target(self, prim: Prim, pose, d, cfg: RHConfig, fld: CostField):
        ux, uy = prim.a * d[0] - prim.b * d[1], prim.a * d[1] + prim.b * d[0]            # a d + b n,  n = (-d_y, d_x)
        sp = math.hypot(ux, uy)
        if sp > cfg.v_max:
            ux, uy = ux * cfg.v_max / sp, uy * cfg.v_max / sp
        c, s = math.cos(pose[2]), math.sin(pose[2])
        wz = prim.wz
        if prim.k_yaw > 0.0:
            near = math.hypot(pose[0] - fld.goal[0], pose[1] - fld.goal[1]) < YAW_FREE_R
            wz = 0.0 if (near or (d[0] == 0.0 and d[1] == 0.0)) else float(min(max(prim.k_yaw * wrap(math.atan2(d[1], d[0]) - pose[2]), -YAW_TRACK_MAX), YAW_TRACK_MAX))
        return np.array([c * ux + s * uy, -s * ux + c * uy, wz])

    def _yaw_cost(self, pose, d, fld: CostField) -> float:
        if self.w_yaw <= 0.0 or math.hypot(pose[0] - fld.goal[0], pose[1] - fld.goal[1]) < YAW_FREE_R or (d[0] == 0.0 and d[1] == 0.0):
            return 0.0
        return self.w_yaw * abs(wrap(pose[2] - math.atan2(d[1], d[0])))

    def _wall_ok(self, grid, poses) -> bool:
        P = np.asarray(poses)                                                               # (S, 3)
        c, s = np.cos(P[:, 2])[:, None], np.sin(P[:, 2])[:, None]
        X = P[:, 0:1] + c * self._fp[None, :, 0] - s * self._fp[None, :, 1]
        Y = P[:, 1:2] + s * self._fp[None, :, 0] + c * self._fp[None, :, 1]
        return float(grid.lookup(X, Y).min()) > self.prune_wall_m

    def _extend(self, node: _Node, prim: Prim, ctx):
        fld, grid, L, r, a_hw, v_cap_steady, h, cfg = ctx["fld"], ctx["grid"], ctx["L"], ctx["r"], ctx["a_hw"], ctx["steady"], ctx["h"], ctx["cfg"]
        u = self._target(prim, node.pose, node.guide, cfg, fld)
        tgt = cup_safe_target(u, r, v_cap_steady, ctx["v_cap"])
        v = list(node.v)
        pose = tuple(node.pose)
        V = list(node.V)
        poses = [pose]
        for _ in range(ctx["n_piece"]):
            v1 = _step(v, tgt, h, L, r, a_hw)
            pose = _integrate(pose, v, v1, h)
            v = list(v1)
            V.append(v)
            poses.append(pose)
        if not self._wall_ok(grid, poses[::2] + ([poses[-1]] if len(poses) % 2 == 0 else [])):
            return None
        cp, gd = fld.cost_guide(pose[0], pose[1])
        return _Node(pose, tuple(v), V, node.names + (prim.name,), cp + self._yaw_cost(pose, gd, fld), cp, gd)

    @staticmethod
    def _key(n: _Node):
        return (round(n.pose[0] / 0.03), round(n.pose[1] / 0.03), round(n.pose[2] / 0.1), round(n.v[0] / 0.05), round(n.v[1] / 0.05), len(n.names))

    # ---------------------------------------------------------------- Proposer.propose
    def propose(self, state: State, world, goal_xy, route_xy, cfg: RHConfig):
        t_start = time.perf_counter()
        gk = world.gk
        h = float(gk.h)
        n_piece = int(round(PIECE_S / h))
        assert abs(n_piece * h - PIECE_S) < 1e-9, (PIECE_S, h)
        pose0 = tuple(float(x) for x in state.pose)
        bel = world.belief_at(pose0, True, "tray")                                          # payload belief: cup lever arm, a_cup_max, hw slew, speed caps
        grid = world.bel0.grid
        fld = cost_field(grid, goal_xy, self.infl)
        ctx = dict(fld=fld, grid=grid, L=gk.plan_frac * bel.a_cup_max, r=(float(bel.r_cup[0]), float(bel.r_cup[1])), a_hw=[float(x) for x in bel.a_hw],
                   steady=gk.steady_frac * gk.plan_frac * bel.a_cup_max, v_cap=np.asarray(bel.v_cap, float), h=h, n_piece=n_piece, cfg=cfg)
        depth_max = max(1, int(round(cfg.horizon_s / PIECE_S)))
        c_root, g_root = fld.cost_guide(pose0[0], pose0[1])
        root = _Node(pose0, tuple(float(x) for x in state.v), [[float(x) for x in state.v]], (), c_root, c_root, g_root)
        c0 = root.cost_pure
        beam, pool = [root], []
        n_ext = 0
        for _ in range(depth_max):
            nxt, seen = [], set()
            for node in beam:
                for pr in self.prims:
                    ch = self._extend(node, pr, ctx)
                    n_ext += 1
                    if ch is not None:
                        nxt.append(ch)
            nxt.sort(key=lambda n: n.cost)
            beam = []
            for n in nxt:
                k = self._key(n)
                if k in seen:
                    continue
                seen.add(k)
                beam.append(n)
                if len(beam) >= self.beam:
                    break
            pool.extend(beam)
        t_search = time.perf_counter()
        # attach the cup-safe backup; final ranking key = cost at the end of the nominal (+ overshoot of the rest pose)
        rows = []
        for n in pool:
            if max(abs(x) for row in n.V for x in row) <= 1e-9:
                continue                                                                    # never leaves rest: that is "stop / wait", not a motion candidate
            vend = np.array(n.v)
            bk = backup_from(vend, bel, gk)
            nn = len(n.V) - 1
            Vf = np.concatenate([np.array(n.V), bk.V[1:]]) if len(bk.V) > 1 else np.array(n.V)
            p_rest = n.pose
            for k in range(1, len(bk.V)):
                p_rest = _integrate(p_rest, bk.V[k - 1], bk.V[k], h)
            c_rest = fld.v(p_rest[0], p_rest[1])
            key = n.cost + W_OVERSHOOT * max(0.0, c_rest - n.cost_pure)
            rows.append((key, n, Vf, nn, c_rest))
        rows.sort(key=lambda r_: r_[0])
        out, seen = [], set()
        for key, n, Vf, nn, c_rest in rows:
            kk = self._key(n)[:3] + (len(n.names),)
            if kk in seen:
                continue
            seen.add(kk)
            tr = Traj(Vf, h, float(state.t), nn * h, nn * h)
            out.append(Candidate(tr, c0 - n.cost_pure, "lattice:" + "/".join(n.names),
                                 dict(rank_cost=key, cost_end=n.cost_pure, cost_rest=c_rest, depth=len(n.names), end_pose=[float(x) for x in n.pose])))
            if len(out) >= self.n_out:
                break
        bk0 = backup_from(np.array(state.v, float), bel, gk, t0=float(state.t))             # "stop now": the cup-safe stop from where we are (LAST, or FIRST if it ends at the goal)
        p_rest = pose0
        for k in range(1, len(bk0.V)):
            p_rest = _integrate(p_rest, bk0.V[k - 1], bk0.V[k], h)
        at_goal = math.hypot(p_rest[0] - fld.goal[0], p_rest[1] - fld.goal[1]) <= GOAL_STOP_M
        stop = Candidate(bk0, c0 - fld.v(p_rest[0], p_rest[1]), "lattice:stop", dict(depth=0, rank_cost=-math.inf if at_goal else math.inf, end_pose=[float(x) for x in pose0]))
        if at_goal:
            out.insert(0, stop)                                                              # converged: the best plan is to stop here (the loop then reports done)
        else:
            out.append(stop)
        t_end = time.perf_counter()
        self.last_stats = dict(n_ext=n_ext, n_pool=len(pool), n_out=len(out), search_ms=(t_search - t_start) * 1e3, total_ms=(t_end - t_start) * 1e3)
        return out
