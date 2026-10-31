"""tamp/rh/mppi_prop.py -- MPPI as a RECEDING-HORIZON PROPOSER (tamp/rh/api.py).  SIMULATION ONLY.

    MPPIProposer(world, cfg).propose(state, world, goal_xy, route_xy, cfg) -> [Candidate, ...] best-first, "stop now" last.

MPPI never decides safety: it proposes, tamp/rh/api.admit certifies (cup + hw + walls + people, person_model="pred").  What it does:

  1. PLAN.  A GPU SamplingPlanner (mppi/planner.py, MPPI update, CUDA graph, warm-started by shifting the previous mean) searches a BODY-twist sequence over
     horizon_s on a grid of dt = 0.1 s (H = horizon_s / dt).  The rollout is mppi/dynamics.rollout (body frame, slew-limited, midpoint integration) with a
     slew box of 0.8 * L per axis (L = gk.plan_frac * a_cup_max), i.e. roughly what the cup-safe step below will allow, so the search optimises the motion that
     is actually executed.  The COST is minimal and lives in this file (RHProblem.cost; mppi/cost.py's spill/CBF/intent terms do not apply):
        progress     V(x_t, y_t): distance-to-goal [m] on the scene occupancy grid (GoalField: 8-connected Dijkstra at 5 cm over cells whose EDT >= the
                     footprint half-width ~0.37 m; smaller inflation if the start is cut off; computed once per (scene, goal)), weighted
                     w_prog * (f_end V_H + (1 - f_end) mean_t V_t);
        walls        hinge on the footprint-boundary clearance to the occupancy EDT (soft + a linear penetration term);
        people       hinge on clearance to the constant-velocity PREDICTION disc  c + v_hat t,  radius 0.30 + d_person + e_v t + a_p t^2 / 2  (soft only);
        smoothness   sum ((u_t - u_{t-1}) / v_max)^2 (u_{-1} = current twist);   yaw  weak pull to the current yaw (goal yaw within 1 m of the goal).
     Speed cap RHConfig.v_max on vx, vy (wz cap `wz_max`).  Safety is NOT the cost's job: the soft terms only steer the search towards admissible motions.
  2. CONVERT.  For the updated mean and up to `n_extra` DIVERSE low-cost samples (taken from the top n_top of the K samples; greedy by RMS twist distance) every
     twist sequence becomes a gatekeeper Traj: each gk.h step moves toward the (cup_safe_target of the) sampled twist with traj._step at L = plan_frac * a_cup_max,
     so the Traj is cup- and hw-feasible BY CONSTRUCTION; the nominal part is cut at min(cfg.horizon_s, planned horizon); traj.backup_from appends the cup-safe
     stop (Traj ends at rest, t_sw = backup start).  Candidates are ranked by V at the END of the nominal part (ascending); the pure-backup "stop now" candidate
     is appended LAST.  Candidate.progress = V(start) - V(end of nominal).
  3. Latency is reported separately: `last_stats` (and Candidate.meta) hold plan_ms (GPU plan incl. state copy + readback), convert_ms (CPU), total_ms.

ASSUMPTIONS: holding = True (the tray with the cup rides on the robot; the payload belief of the world); route_xy is not used (the distance field replaces it); the
distance field is a geometric guide (octile Dijkstra, <= 8 % metric error), not a certified quantity; wz_max = 0.5 rad/s is a sampling bound only (the cup-safe
step and the checker limit the real rotation).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ...gatekeeper.traj import Traj, _step, backup_from, cup_safe_target
from ...mppi import cost as CO, dynamics as DY, fields as FL, planner as PL
from ...teleop import geom as GM
from .. import plan_cert as PC
from .api import Candidate, RHConfig, State

# ---------------------------------------------------------------------------------------------------------------- distance-to-goal field
FOOTPRINT_HALF = 0.5 * min(GM.FP["x"][1] - GM.FP["x"][0], GM.FP["y"][1] - GM.FP["y"][0])      # ~0.371 m
INFLATE_LEVELS = (FOOTPRINT_HALF, 0.30, 0.22, 0.13)                                           # tried in order until the start is connected to the goal


class GoalField:
    """V(x, y) [m] = geodesic distance to `goal_xy` over the free cells of a coarse copy (res) of an OccGrid; cells that are blocked (EDT < inflate) or cut off from the
    goal get V = V(nearest reachable cell) + out_slope * distance, so the field still points back into free space.  Numpy bilinear lookup (`v`), and a torch twin
    (`torch_vol`) for the planner.  Cell centres at x0 + (i + 0.5) res (same convention as mppi.fields.Grid2DField)."""

    def __init__(self, grid, goal_xy, inflate: float = FOOTPRINT_HALF, res: float = 0.05, out_slope: float = 3.0):
        from scipy import ndimage
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        f = max(1, int(round(res / grid.res)))
        self.res, self.x0, self.y0, self.inflate, self.out_slope = f * float(grid.res), float(grid.x0), float(grid.y0), float(inflate), float(out_slope)
        ny, nx = grid.edt.shape
        nyc, nxc = ny // f, nx // f
        edt = grid.edt[np.ix_(np.arange(nyc) * f + f // 2, np.arange(nxc) * f + f // 2)]
        free = edt >= inflate
        self.shape = (nyc, nxc)
        idx = np.arange(nyc * nxc).reshape(nyc, nxc)
        rows, cols, w = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):             # edge i,j -> i+di, j+dj
            j0, j1 = (0, nxc - dj) if dj >= 0 else (-dj, nxc)
            A = (slice(0, nyc - di), slice(j0, j1))
            Bn = (slice(di, nyc), slice(j0 + dj, j1 + dj))
            ok = free[A] & free[Bn]
            if di and dj:                                           # no corner cutting between two blocked cells
                ok &= free[slice(0, nyc - di), slice(j0 + dj, j1 + dj)] & free[slice(di, nyc), slice(j0, j1)]
            a, b = idx[A][ok], idx[Bn][ok]
            rows.append(a); cols.append(b); w.append(np.full(len(a), self.res * math.hypot(di, dj)))
        rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
        G = coo_matrix((w, (rows, cols)), shape=(nyc * nxc, nyc * nxc)).tocsr()
        gi = int(np.clip((goal_xy[1] - self.y0) / self.res, 0, nyc - 1)); gj = int(np.clip((goal_xy[0] - self.x0) / self.res, 0, nxc - 1))
        if not free[gi, gj]:                                        # goal (a dock) inside the inflated band: snap to the nearest free cell
            _, (si, sj) = ndimage.distance_transform_edt(~free, return_indices=True)
            gi, gj = int(si[gi, gj]), int(sj[gi, gj])
        d = dijkstra(G, directed=False, indices=idx[gi, gj]).reshape(nyc, nxc)
        fin = np.isfinite(d)
        self.reach = fin
        dist, (ii, jj) = ndimage.distance_transform_edt(~fin, sampling=self.res, return_indices=True)
        self.V = np.where(fin, d, d[ii, jj] + out_slope * dist).astype(np.float32)
        self.goal_xy = np.asarray(goal_xy, float)[:2].copy()

    def v(self, xy):
        """bilinear V at world points (..., 2) -> (...)."""
        from scipy import ndimage
        xy = np.asarray(xy, float)
        sh = xy.shape[:-1]
        q = xy.reshape(-1, 2)
        out = ndimage.map_coordinates(self.V, [(q[:, 1] - self.y0) / self.res - 0.5, (q[:, 0] - self.x0) / self.res - 0.5], order=1, mode="nearest")
        return out.reshape(sh)

    def grad(self, xy, eps: float = 0.05):
        xy = np.asarray(xy, float)[:2]
        gx = (self.v(xy + (eps, 0)) - self.v(xy - (eps, 0))) / (2 * eps)
        gy = (self.v(xy + (0, eps)) - self.v(xy - (0, eps))) / (2 * eps)
        return np.array([gx, gy])

    def reachable(self, xy) -> bool:
        i = int(np.clip((xy[1] - self.y0) / self.res, 0, self.shape[0] - 1)); j = int(np.clip((xy[0] - self.x0) / self.res, 0, self.shape[1] - 1))
        return bool(self.reach[i, j])


class _TorchField:
    """bilinear lookup of a (ny, nx) field; static device buffer so a CUDA graph can capture it (update with `set`)."""

    def __init__(self, shape, x0, y0, res, device):
        self.vol = torch.zeros(1, 1, *shape, dtype=torch.float32, device=device)
        self.size = torch.as_tensor([shape[1], shape[0]], dtype=torch.float32, device=device)
        self.org = torch.as_tensor([x0, y0], dtype=torch.float32, device=device)
        self.res = res

    def set(self, V):
        self.vol.copy_(torch.as_tensor(V, dtype=torch.float32)[None, None])

    def query(self, xy):
        sh = xy.shape[:-1]
        u = (xy.reshape(1, 1, -1, 2) - self.org) / self.res
        gn = 2.0 * u / self.size - 1.0
        return F.grid_sample(self.vol, gn, mode="bilinear", padding_mode="border", align_corners=False).reshape(sh)


# ---------------------------------------------------------------------------------------------------------------- MPPI problem (cost in this file)
@dataclass
class RHCost:
    w_prog: float = 10.0          # per metre of distance-to-goal
    f_end: float = 0.5            # weight of the END value vs the time-mean value of V along the rollout
    d_wall: float = 0.10          # m, wall buffer (== GKConfig/Belief d_wall)
    d_soft_wall: float = 0.15     # m, soft zone beyond d_wall
    w_wall: float = 20.0
    w_pen_wall: float = 400.0
    r_person: float = GM.PERSON_CLEAR_R
    d_person: float = 0.10
    d_soft_person: float = 0.25   # m, soft zone beyond the prediction disc
    w_person: float = 20.0
    w_pen_person: float = 200.0
    w_approach: float = 400.0      # per metre of distance lost to a predicted person centre while inside its (soft) keep-out
    d_exit_zone: float = 0.05      # m, 'inside' for the approach term: clearance to the prediction disc below this
    w_du: float = 0.5
    w_speed: float = 2000.0       # hinge on |(vx, vy)| - v_max (the planner's own clip is a per-axis BOX; the cap is on the NORM)
    w_yaw: float = 3.0
    foot_step: float = 0.08       # m, footprint boundary sampling for the clearance terms


class RHProblem(CO.CarryProblem):
    """CarryProblem state machinery (flat device buffer, set_state, resample_mu) with the receding-horizon cost; the parent's cost is replaced."""

    def __init__(self, model, H, grid, goal_t: _TorchField, ccfg: RHCost, e_v, a_p, r_cup, device="cuda", n_people=4):
        pts = GM.boundary_pts(step=ccfg.foot_step)
        super().__init__(model, H, pts, np.zeros(len(pts)), np.zeros(len(pts)), FL.Grid2DField(grid, device=device), CO.CostCfg(use_cup=False), mode="intent", device=device,
                         r_cup=tuple(r_cup), n_people=n_people, v_scale=tuple(model.v_max))
        self.cc, self.goal, self.e_v, self.a_p = ccfg, goal_t, float(e_v), float(a_p)
        self.v_trans = float(model.v_max[0])

    def cost(self, U, return_parts=False):
        c, S, m = self.cc, self.S, self.model
        ro = DY.rollout(m, S["pose0"], S["v0"], U, S["r_cup"])
        pose = ro["pose"]                                                              # (B,H,3)
        t = self.t_axis                                                                # (H,) time at the end of each step
        V = self.goal.query(pose[..., :2])                                             # (B,H)
        c_prog = c.w_prog * (c.f_end * V[:, -1] + (1.0 - c.f_end) * V.mean(1))
        pw = self.world_points(pose)                                                   # (B,H,P,2)
        d = self.field.query(pw)                                                       # (B,H,P)
        h = d.amin(-1) - c.d_wall                                                      # (B,H)
        c_wall = (c.w_wall * torch.relu(c.d_soft_wall - h) ** 2 + c.w_pen_wall * torch.relu(-h)).sum(1)
        pp = S["people"]
        cen = pp[:, :2][None] + pp[:, 2:][None] * t[:, None, None]                     # (H,Np,2)
        dd = torch.linalg.norm(pw[:, :, :, None, :] - cen[None, :, None, :, :], dim=-1).amin(2)       # (B,H,Np)
        rad = c.r_person + c.d_person + self.e_v * t + 0.5 * self.a_p * t * t           # (H,)
        hpp = dd - rad[None, :, None]                                                  # (B,H,Np)
        hp = hpp.amin(-1)                                                              # (B,H)
        c_ppl = (c.w_person * torch.relu(c.d_soft_person - hp) ** 2 + c.w_pen_person * torch.relu(-hp)).sum(1)
        # separating-motion rule of the certifier (checks.py): a start inside a keep-out is admissible iff the distance to the PREDICTED centre never decreases
        # until the footprint is clear -> penalise a decrease of that distance while (nearly) inside
        pw0 = self.world_points(S["pose0"].reshape(1, 1, 3))                           # (1,1,P,2)
        d0 = torch.linalg.norm(pw0[:, :, :, None, :] - pp[:, :2][None, None, None], dim=-1).amin(2)   # (1,1,Np)
        dall = torch.cat([d0.expand(dd.shape[0], 1, -1), dd], 1)                       # (B,H+1,Np)
        ins = torch.cat([(d0 - rad[0]).expand(dd.shape[0], 1, -1), hpp], 1)[:, :-1] < c.d_exit_zone
        c_app = c.w_approach * (torch.relu(dall[:, :-1] - dall[:, 1:]) * ins.float()).sum((1, 2))
        c_ppl = c_ppl + c_app
        du = torch.cat([U[:, :1, :3] - S["u_prev"][:3], U[:, 1:, :3] - U[:, :-1, :3]], 1) / self.vs
        c_s = c.w_du * (du ** 2).sum((1, 2))
        c_s = c_s + c.w_speed * (torch.relu(torch.linalg.norm(U[..., :2], dim=-1) - self.v_trans) ** 2).sum(1)
        dy = pose[..., 2] - S["yaw_ref"]
        c_yaw = c.w_yaw * (torch.atan2(torch.sin(dy), torch.cos(dy)) ** 2).mean(1)
        tot = c_prog + c_wall + c_ppl + c_s + c_yaw
        if return_parts:
            return tot, dict(total=tot, rollout=ro, h_min=h.amin(1), terms=dict(prog=c_prog, wall=c_wall, person=c_ppl, du=c_s, yaw=c_yaw), hp_min=hp.amin(1))
        return tot


class _TopKPlanner(PL.SamplingPlanner):
    """SamplingPlanner that also keeps the n_top cheapest of the K evaluated sequences in static buffers (written INSIDE the CUDA graph)."""

    def __init__(self, *a, n_top: int = 256, **kw):
        super().__init__(*a, **kw)
        self.n_top = n_top
        self._inner = self._cost
        self._cost = self._record
        f32 = dict(dtype=torch.float32, device=self.dev)
        self.top_U = torch.zeros(n_top, self.H, self.nu, **f32)
        self.top_c = torch.zeros(n_top, **f32)

    def _record(self, U):
        c = self._inner(U)
        self._U_last, self._c_last = U, c
        return c

    def _step(self):
        super()._step()
        idx = torch.topk(self._c_last, self.n_top, largest=False).indices
        self.top_U.copy_(self._U_last.index_select(0, idx))
        self.top_c.copy_(self._c_last.index_select(0, idx))


# ---------------------------------------------------------------------------------------------------------------- conversion (CPU, no torch)
def convert_twists(v0, U, dt_u: float, n_nom: int, bel, gk, t0: float = 0.0, v_trans_max: float | None = None) -> Traj:
    """twist sequence U (H, 3) (step i spans [i dt_u, (i+1) dt_u)) -> cup-feasible Traj: n_nom gk.h steps toward cup_safe_target(U[i]) with traj._step at
    L = gk.plan_frac * a_cup_max, then traj.backup_from (cup-safe stop).  Ends at rest; t_sw = n_nom h."""
    h = gk.h
    L = gk.plan_frac * bel.a_cup_max
    r = (float(bel.r_cup[0]), float(bel.r_cup[1]))
    a_hw = [float(x) for x in bel.a_hw]
    U = np.array(U, float)[:, :3]
    if v_trans_max is not None:
        nrm = np.hypot(U[:, 0], U[:, 1])
        U[:, :2] *= np.where(nrm > v_trans_max, v_trans_max / np.maximum(nrm, 1e-12), 1.0)[:, None]
    tg = [cup_safe_target(u, r, gk.steady_frac * L, bel.v_cap) for u in U]
    v = [float(x) for x in v0]
    V = [list(v)]
    for k in range(n_nom):
        v = _step(v, tg[min(int((k * h + 1e-9) / dt_u), len(tg) - 1)], h, L, r, a_hw)
        V.append(list(v))
    bk = backup_from(v, bel, gk, t0)
    V.extend(bk.V[1:].tolist())
    return Traj(np.array(V, float), h, t0, n_nom * h, n_nom * h)


def end_of_nominal_xy(traj: Traj, pose) -> np.ndarray:
    """map xy of the sample at the end of the nominal part (t_sw), or sample 0 if already in the backup."""
    n = 0 if traj.t_sw <= 0 else min(traj.N, int(round(traj.t_sw / traj.h)))
    p = traj.poses()[n]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([pose[0] + c * p[0] - s * p[1], pose[1] + s * p[0] + c * p[1]])


def candidates_from_twists(Us, state: State, bel, gk, goal: GoalField, dt_u: float, horizon_s: float, source="mppi", metas=None, v_trans_max: float | None = None) -> list:
    """CPU part: Us -> Candidates ranked by V at the end of the nominal part (ascending), pure-backup 'stop now' LAST.  Pure functions of (Us, state, bel, gk, goal)."""
    n_nom = int(round(horizon_s / gk.h))
    v0 = np.asarray(state.v, float)
    v_start = float(goal.v(np.asarray(state.pose[:2], float)))
    out = []
    for i, U in enumerate(Us):
        tr = convert_twists(v0, U, dt_u, n_nom, bel, gk, state.t, v_trans_max)
        if np.abs(tr.V[:n_nom + 1]).max() < 1e-6:                       # an all-zero nominal part == stop now
            continue
        v_end = float(goal.v(end_of_nominal_xy(tr, state.pose)))
        out.append((v_end, i, Candidate(tr, v_start - v_end, f"{source}:{'mean' if i == 0 else f's{i}'}", dict(v_end=v_end, **(metas[i] if metas else {})))))
    out.sort(key=lambda a: (a[0], a[1]))
    cands = [c for _, _, c in out]
    stop = backup_from(v0, bel, gk, state.t)
    stop.t_sw = 0.0
    cands.append(Candidate(stop, 0.0, f"{source}:stop", dict(v_end=v_start)))
    return cands


def diverse(top_U: np.ndarray, n: int, thr: float, scale) -> list:
    """greedy indices into top_U (sorted by cost ascending): the first is always taken; the next must differ from every taken one by RMS(U - U_j)/scale >= thr."""
    keep = [0]
    sc = np.asarray(scale, float)
    for i in range(1, len(top_U)):
        if len(keep) >= n:
            break
        if all(np.sqrt(np.mean(((top_U[i, :, :3] - top_U[j, :, :3]) / sc) ** 2)) >= thr for j in keep):
            keep.append(i)
    return keep


# ---------------------------------------------------------------------------------------------------------------- the proposer
class MPPIProposer:
    name = "mppi"

    def __init__(self, world, cfg: RHConfig | None = None, gk=None, K: int = 4096, dt: float = 0.1, horizon_s: float | None = None, n_extra: int = 10, n_top: int = 256,
                 div_thr: float = 0.08, n_cold: int = 3, wz_max: float = 0.5, sigma=None, method: str = "mppi", seed: int = 0, ccfg: RHCost | None = None, device: str = "cuda",
                 use_graph: bool = True, goal_xy=None, verbose: bool = False):
        cfg = cfg or RHConfig()
        self.cfg, self.world, self.gk = cfg, world, gk if gk is not None else world.gk
        grid = world.bel0.grid
        assert grid is not None, "world.bel0.grid (OccGrid) is required"
        self.grid, self.dt, self.n_extra, self.n_top, self.div_thr, self.n_cold, self.verbose = grid, dt, n_extra, n_top, div_thr, n_cold, verbose
        self.horizon_s = float(horizon_s if horizon_s is not None else cfg.horizon_s)
        H = int(round(self.horizon_s / dt))
        self.H = H
        belh = PC.plan_belief(world, world.q0.xyth(), True, "tray", [], 0.0, self.gk)
        L = self.gk.plan_frac * belh.a_cup_max
        rn = max(0.2, float(np.hypot(*belh.r_cup)))
        self.v_cap = np.array([cfg.v_max, cfg.v_max, wz_max])
        a_box = (0.8 * L, 0.8 * L, min(8.0, 0.8 * L / rn))
        model = DY.BaseModel(dt=dt, frame="body", a_hw=a_box, v_max=tuple(self.v_cap))
        gf = GoalField(grid, np.asarray(goal_xy, float) if goal_xy is not None else np.asarray(world.q0.xyth()[:2]))
        self._fields = {}
        self._goal_key = None
        self.goal_t = _TorchField(gf.V.shape, gf.x0, gf.y0, gf.res, device)
        self.ccfg = ccfg or RHCost()
        self.prob = RHProblem(model, H, grid, self.goal_t, self.ccfg, cfg.e_v, cfg.a_p, belh.r_cup, device=device)
        sg = sigma if sigma is not None else (0.10, 0.10, 0.15)
        self.pl = _TopKPlanner(self.prob, K=K, method=method, sigma=sg, plan_dt=dt, use_graph=use_graph, seed=seed, prim_speed=(0.9, 0.9), n_top=n_top)
        self._put_field(gf)                                         # real field in the (static) device buffer, before the warm-up captures the graph
        self.warmup_ms = self.pl.warmup()
        self._t_last = None
        self.last_stats: dict = {}
        self.history: list = []

    # ------------------------------------------------------------------ goal field cache
    def _put_field(self, gf: GoalField):
        self.gf = gf
        self.goal_t.set(gf.V)

    def _ensure_goal(self, goal_xy, start_xy):
        g = np.asarray(goal_xy, float)[:2]
        key = (round(float(g[0]), 2), round(float(g[1]), 2))
        gf = self._fields.get(key)
        if gf is None:
            for lvl in INFLATE_LEVELS:                              # largest inflation for which the start is connected to the goal
                gf = GoalField(self.grid, g, inflate=lvl)
                if gf.reachable(np.asarray(start_xy, float)):
                    break
            self._fields[key] = gf
        if key != self._goal_key:
            self._goal_key = key
            self._put_field(gf)
            self.pl.reset()
            self._t_last = None
        return gf

    # ------------------------------------------------------------------ API
    def propose(self, state: State, world, goal_xy, route_xy, cfg: RHConfig):
        t_all = time.perf_counter()
        gxy = np.asarray(goal_xy, float)
        gf = self._ensure_goal(gxy, state.pose[:2])
        horizon = min(float(cfg.horizon_s), self.H * self.dt)
        if cfg.horizon_s > self.H * self.dt + 1e-9 and not getattr(self, "_warned", False):
            print(f"[mppi] WARNING: cfg.horizon_s={cfg.horizon_s} > planned horizon {self.H * self.dt}; nominal cut at the planned horizon", flush=True)
            self._warned = True
        pose = np.asarray(state.pose, float)
        v0 = np.asarray(state.v, float)
        # people rows [x, y, vx, vy] (nearest n_people to the robot origin)
        pl = sorted(state.people, key=lambda p: float(np.hypot(*(np.asarray(p[0], float)[:2] - pose[:2]))))[:self.prob.S["people"].shape[0]]
        rows = [[float(c[0]), float(c[1]), float(v[0]), float(v[1])] for c, v in pl]
        # intent: full-speed descent of V in the body frame; yaw reference
        g = gf.grad(pose[:2])
        ng = float(np.hypot(*g))
        uw = -g / ng * cfg.v_max if ng > 1e-6 and gf.v(pose[:2]) > 0.05 else np.zeros(2)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        u_int = np.array([c * uw[0] + s * uw[1], -s * uw[0] + c * uw[1], 0.0])
        yaw_ref = float(gxy[2]) if len(gxy) >= 3 and gf.v(pose[:2]) < 1.0 else float(pose[2])
        self.prob.e_v, self.prob.a_p = float(cfg.e_v), float(cfg.a_p)
        self.prob.set_state(pose=pose, v=v0, u_prev=v0, u_int=u_int, people=rows, yaw_ref=yaw_ref)
        cold = self._t_last is None or not (0.0 <= state.t - self._t_last <= self.H * self.dt)
        if cold:
            self.pl.reset(); el = 0.0
        else:
            el = state.t - self._t_last
        self._t_last = float(state.t)
        t0 = time.perf_counter()
        for _ in range(self.n_cold if cold else 0):                 # cold start: a few extra passes refine the (zero) mean before the first proposal
            self.pl.plan(elapsed=0.0)
        U, info = self.pl.plan(elapsed=el)
        tops = self.pl.top_U.cpu().numpy()
        plan_ms = (time.perf_counter() - t0) * 1e3
        # candidates: the updated mean first, then diverse cheap samples
        idx = diverse(tops, self.n_extra, self.div_thr, self.v_cap)
        Us = [U[:, :3]] + [tops[i, :, :3] for i in idx]
        t1 = time.perf_counter()
        bel = PC.plan_belief(world, pose, True, "tray", [], 0.0, self.gk)
        cands = candidates_from_twists(Us, state, bel, self.gk, gf, self.dt, horizon, "mppi", v_trans_max=cfg.v_max)
        convert_ms = (time.perf_counter() - t1) * 1e3
        total_ms = (time.perf_counter() - t_all) * 1e3
        self.last_stats = dict(plan_ms=plan_ms, convert_ms=convert_ms, total_ms=total_ms, n_cand=len(cands), plan_cost=info["cost"], plan_hmin=info["h_min"], elapsed_shift=el, cold=bool(cold))
        self.history.append(self.last_stats)
        for cd in cands:
            cd.meta.update(plan_ms=plan_ms, convert_ms=convert_ms)
        return cands
