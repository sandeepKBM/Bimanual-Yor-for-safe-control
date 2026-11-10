"""stack/planners/closed_chain.py -- ClosedChainPlanner for the contents carry tasks (box, tray2/4/6).  SIMULATION ONLY.   (stack plan step 4, 11-05)

Plan once, then track (planner decisions D1-D7, 11-03):  object path from the TASK GOAL -> both claws by analytic SRS IK (stack/kin/srs_ik.py), the arm
angle psi and the IK branch chosen per path sample by a DP for joint-limit margin and continuity -> TOPP-RA timing -> the joint references (q, qd, qdd at
the 5 ms control period) that FL-MPC tracks.  No continuation IK at execution time: an unreachable goal is REFUSED (PlanRefused: sample, arm, miss).

Object path (translation only, orientation fixed = level; the claws are rigidly welded to the object, so each claw TCP pose = start pose + the object offset):
  lift   S -> C1 = S + (0, dy, z_c)       straight up to the clearance height, with a lateral offset dy chosen by the planner (below)
  carry  C1 -> C2 = (goal xy, z_c)        above the goal, straight line (this also removes dy)
  place  C2 -> G  = (goal xy, goal z)     straight down onto the goal support
  z_c  = (highest support top the object passes over + CLEAR_M[task]) - start support top;   G.z = goal support top - start support top.
  dy   = the smallest |dy| on the DY_GRID whose bottleneck joint margin (screening poses along the lift, both arms, best psi/branch) reaches MARGIN_WANT;
         if none does, the dy with the largest bottleneck margin.  Why a lateral offset exists at all: with the NYU hold pose the right arm cannot reach
         z_c = 0.18 m straight above the start with the object level (best psi, all 8 branches: no solution; 11-05 map), the lift leans toward +y where it can.
  Candidates whose object volume would hit the shelf slab (AABB, bloated) are dropped.
Timing: TOPP-RA on the 17-D path [q_left(7), q_right(7), object offset(3)], rest to rest per segment:  joint speed <= datasheet (config.MANUAL, 180/225 deg/s),
  object acceleration |a_i| <= a_plan per axis (the box the static admission rows are certified on); NO joint acceleration limit (config.py has none: open),
  the resulting max |qdd| is reported.
Backups (same structure as carry.py today): stop, retrace to the segment start (and on through the earlier segments) along the PLANNED joint path in reverse
  (rest to rest per segment, a = carry.A_BACKUP), let go in place, retreat.  Retreat = a planned 5 cm claw translation, smoothstep 0.8 s (legacy), stretched if
  a joint speed limit binds.

Labels (AGENTS.md bounds rule):  ASSUMED = CLEAR_M, TK.GOAL_DXY (user-confirmed 11-02 plans; stack/tasks.py), LAT_CLEAR, DY_GRID, MARGIN_WANT (= carry.PATH_MARGIN_MIN), M_GOOD, DS, PSI_STEP_DEG,
DQ_MAX, N_GRID, FK_TOL;  MODEL = SRS kinematics from the compiled MJCF, FK validation by MuJoCo, TOPP-RA constraints from config.MANUAL (datasheet speeds).
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import pathlib

import numpy as np

from world_model.v2.nero import config as C
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.kin.srs_ik import SRSArm
from world_model.v2.stack.planners import base_route as BR
from world_model.v2.stack.types import SIDES, BimanualPlan, Segment, SkillCall, TaskSpec

NAME = "closed_chain"
DT = 0.005                                  # control period [s] (= carry.DT)
SEGS = ("lift", "carry", "place", "release")
MOVES = ("lift", "carry", "place")

# ---- ASSUMED planner numbers (each listed in plan.assumptions)
CLEAR_M = {"tray": 0.05, "box": 0.03, "pot": 0.05}       # ASSUMED [m]: object underside clearance above the highest support it passes over (the clearances implied by the user-confirmed 11-02 plans: tray lift 5 cm, box 18 cm = shelf 15 cm + 3 cm)
LAT_CLEAR = 0.005                           # ASSUMED [m]: lateral bloat of the object volume in the shelf collision test
DY_GRID = tuple(np.round(np.arange(-0.04, 0.0601, 0.005), 4))   # ASSUMED search grid of the lateral lift offset [m]
MARGIN_WANT = 0.05                          # ASSUMED [rad] (= carry.PATH_MARGIN_MIN): joint-limit distance wanted at every path sample
M_GOOD = 0.15                               # ASSUMED [rad]: above this a joint margin is not traded against smoothness in the DP
DS = 0.01                                   # ASSUMED [m]: path sample spacing
PSI_STEP_DEG = 3.0                          # ASSUMED: arm-angle grid step
DQ_MAX = 0.12                               # ASSUMED [rad]: largest joint change (inf-norm) between consecutive path samples (continuity)
SMOOTH_SIGMAS = (4.0, 3.0, 2.0, 1.0)        # ASSUMED [samples]: Gaussian widths tried when smoothing psi(s) after the DP
SMOOTH_MARGIN_LOSS = 0.003                  # ASSUMED [rad]: how much joint margin below the DP bottleneck the smoothed chain may lose
N_GRID = 160                                # TOPP-RA grid points per piece
FK_TOL_M, FK_TOL_RAD = 1e-6, 1e-6           # MODEL checks of the chosen configurations against MuJoCo FK
RETREAT_T = 0.8                             # s, legacy release retreat time (carry.run_phases)
RETREAT_M = 0.05                            # m, = carry.RETREAT
BIG_ACC = 1e4                               # sentinel for "no joint acceleration limit" in TOPP-RA (NOT a robot limit; config.py has none)
BIG_VEL = 1e3                               # sentinel for the object-offset dimensions' speed (no object speed limit)
CACHE_DIR = pathlib.Path(os.environ.get("BIMANUAL_PLAN_CACHE", pathlib.Path(__file__).resolve().parent / "plan_cache"))


class PlanRefused(RuntimeError):
    """the goal is not reachable under the planner's checks; .info says where."""

    def __init__(self, msg, **info):
        super().__init__(msg)
        self.info = info


@dataclasses.dataclass(frozen=True)
class WaypointPlan:
    """11-05 entry point: the OBJECT-LEVEL plan written by the proposer (stack/proposer/compile.py), in place of the fixed GOAL_DXY / lift-carry-place shape.
    One move per step; each move is a straight object translation, orientation fixed (level).  Everything else (IK, psi / branch DP, the lateral lift offset
    dy, TOPP-RA timing, backups) is still this module's.
      names    segment name per move (the plan's segments follow the steps)
      deltas   per move (dx, dy, dz) [m] of the object, world frame
      backups  per move: the backup kind run from inside that segment: set_down | retrace | hold | let_go  (backup_phases_for)
      lands    the last move lands ON the goal support (the shelf collision test then counts penetration only on it, as for the default plan)
    The lateral lift offset dy (see object_path) is applied to the end of move 0 when move 0 is a pure rise; every later node is the proposer's own, so a
    standard lift / carry / place request reproduces the default path."""
    names: tuple
    deltas: tuple
    backups: tuple
    lands: bool = True
    base: tuple = ()                 # table-behind (11-09): ((after, route, level),) -- ONE base_move after `after` arm moves (1 <= after <= len(names));
                                     # route = (("line", m) | ("turn", deg), ...); level "L1".. (the base speed level, does not enter the arm geometry key).
                                     # deltas are ARM-SPACE offsets in the BASE frame (start-aligned): the object keeps its base-frame offset across the base move.

    def __post_init__(self):
        if not (len(self.names) == len(self.deltas) == len(self.backups) >= 1):
            raise ValueError("WaypointPlan: names, deltas, backups must have the same length >= 1")
        for a, r, lv in self.base:
            if not (isinstance(a, int) and 1 <= a <= len(self.names)):
                raise ValueError(f"WaypointPlan.base: 'after' {a!r} must be an int in 1..{len(self.names)}")

    def nodes(self):
        return np.vstack([np.zeros(3), np.cumsum(np.asarray(self.deltas, float), axis=0)])

    @property
    def pure_rise_first(self):
        d = self.deltas[0]
        return abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9 and d[2] > 0

    def key_data(self, level=False):
        """cache-key / JSON data.  The base route enters the geometry key (it is checked in geometry_for); the level only with level=True (JSON round trip)."""
        d = dict(names=list(self.names), deltas=[[round(float(x), 9) for x in d] for d in self.deltas], backups=list(self.backups), lands=bool(self.lands))
        if self.base:                                                   # fixed-base keys (and cache files) are unchanged
            d["base"] = [[int(a), [[k, float(v)] for k, v in r]] + ([lv] if level else []) for a, r, lv in self.base]
        return d

    @classmethod
    def from_data(cls, w):
        base = tuple((int(b[0]), tuple((str(k), float(v)) for k, v in b[1]), str(b[2]) if len(b) > 2 else "L1") for b in w.get("base", ()))
        return cls(tuple(w["names"]), tuple(tuple(x) for x in w["deltas"]), tuple(w["backups"]), bool(w["lands"]), base)


def kind_of(task):
    t = str(task)
    if getattr(TK.TASKS.get(t), "base", ""):       # a *_behind task: its kind from the TaskDef; its goal is TaskDef.goal_dxy on goal_support, never the fixed-base GOAL_DXY (the plan is a base WaypointPlan)
        return TK.task(t).kind
    return "tray" if t.startswith("tray") else ("pot" if t == "pot" else "box")


def is_base_task(task):
    return bool(getattr(TK.TASKS.get(str(task)), "base", ""))


def vlim():
    return np.radians(np.asarray(C.MANUAL["joint_speed_deg_s"], float))          # datasheet speeds [rad/s] (SPEC)


# ----------------------------------------------------------------------------------------------------------------------- inputs
class PlanInputs:
    """everything the path planner needs, independent of the episode object: model (for the SRS constants), hold q per arm, the world."""

    def __init__(self, m, world, q0, task, wp=None):
        self.m, self.world, self.task = m, world, kind_of(task)
        self.task_name = str(task)
        if wp is None and is_base_task(task):                                          # a table-behind task has no GOAL_DXY plan: its default is lift -> base_move R1 -> place on counter 2
            wp = default_waypoint(task, world)
        self.wp = wp                                                                   # WaypointPlan | None (None = the default GOAL_DXY lift / carry / place)
        self.q0 = {s: np.asarray(q0[s], float) for s in SIDES}
        self.arms = {s: SRSArm(m, s) for s in SIDES}
        self.T0 = {s: self.arms[s].fk(self.q0[s]) for s in SIDES}                      # claw TCP pose at the hold = rigidly fixed relative to the object

    def key(self):
        from world_model.v2.nero.contents import scenes as SC
        w = self.world
        tops = SC.support_tops(w)
        d = dict(task=self.task, obj_xy=list(w.obj_xy), tops={k: dict(top=v["top"], x=list(v["x"]), y=list(v["y"])) for k, v in tops.items()},
                 box=list(w.box_outer) if self.task == "box" else None, shelf_t=getattr(w, "shelf_t", None),
                 q0={s: [round(float(x), 9) for x in self.q0[s]] for s in SIDES}, lim=[SRSArm_limits(self.arms[s]) for s in SIDES],
                 params=dict(CLEAR_M=CLEAR_M[self.task], GOAL=TK.GOAL_DXY[self.task], LAT=LAT_CLEAR, DY=list(map(float, DY_GRID)), MW=MARGIN_WANT, MG=M_GOOD,
                             DS=DS, PSI=PSI_STEP_DEG, DQ=DQ_MAX, SS=list(SMOOTH_SIGMAS), SL=SMOOTH_MARGIN_LOSS), v=3)
        if self.wp is not None:                                                        # only a waypoint plan adds a key: default-plan keys (and cache files) are unchanged
            d["wp"] = self.wp.key_data()
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=float).encode()).hexdigest()[:20]


def SRSArm_limits(arm):
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(arm.lo, arm.hi)]


def inputs_from_episode(ep):
    return PlanInputs(ep.m, ep.world, ep.q0, ep.task, wp=getattr(ep, "wp", None))


BASE_LEVEL_DEFAULT = "L1"      # ASSUMED: base speed level written into a default plan (= today's swerve Limits); the admission / caller passes the level it picked (plan(level=...))
_BASE_BACKEND = None           # tests may set a backend object here (base_route.check / sweep interface); None = base_route.default_backend()


def default_waypoint(task, world, backend=None):
    """the default plan of a table-behind task (ASSUMED shape, design section 2): lift CLEAR_M -> base_move (TaskDef route R1) -> straight down onto the goal support
    (the route puts the place pose of the object at the pick pose in the robot frame: scene.place_pose_xy), then release.  Arm-space deltas, base frame."""
    td = TK.task(str(task))
    be = backend or _BASE_BACKEND or BR.default_backend()
    sup = be.supports(world)
    zc = CLEAR_M[td.kind]
    dz = float(sup[td.goal_support]["top"]) - (float(sup["counter"]["top"]) + zc)
    return WaypointPlan(("s0_lift", "s2_place"), ((0.0, 0.0, zc), (0.0, 0.0, round(dz, 9))), ("set_down", "hold"), True,
                        ((1, BR.default_route(td), BASE_LEVEL_DEFAULT),))


@functools.lru_cache(maxsize=None)
def _nominal_inputs(task, wp=None):
    """default world of the task + the hold pose of contents/capture_nyu.json (what the static admission uses)."""
    from world_model.v2.nero.contents import envelope as E, scenes as SC
    k = kind_of(task)
    if is_base_task(task):                                                         # table-behind: the task's own world (n_cups, world_kw) and the two-table swerve scene
        be = BR.default_backend()
        w = be.world_default(task)
        m = be.compose(w)
    else:
        w = (SC.TrayWorld(n_cups=int(task[4:]) if task.startswith("tray") and task[4:].isdigit() else 2) if k == "tray"
             else (SC.PotWorld() if k == "pot" else SC.BoxWorld()))
        m = SC.compose(w)
    cap = json.loads(E.CAP_FILE.read_text())["pot"]
    return PlanInputs(m, w, {s: cap["q"]["W0_liftoff"][s] for s in SIDES}, task, wp=wp)


# ----------------------------------------------------------------------------------------------------------------------- object path
def object_path(inp: PlanInputs, dy):
    """-> (nodes (4,3) object offsets S, C1, C2, G; info).  Offsets from the start pose of obj_ref."""
    from world_model.v2.nero.contents import scenes as SC
    w, task = inp.world, inp.task
    tops = SC.support_tops(w)
    t0 = tops["counter"]["top"]
    if inp.wp is not None:                                  # proposer's waypoints (WaypointPlan): its own nodes; dy only on the end of a pure rise
        nodes = inp.wp.nodes()
        if inp.wp.pure_rise_first:
            nodes[1, 1] += dy
        return nodes, dict(z_c=float(nodes[:, 2].max()), goal_dz=float(nodes[-1, 2]), goal_xy=[float(w.obj_xy[0] + nodes[-1, 0]), float(w.obj_xy[1] + nodes[-1, 1])])
    goal_top = tops["shelf"]["top"] if (task == "box" and "shelf" in tops) else t0
    z_c = max(goal_top, t0) + CLEAR_M[task] - t0
    gx, gy = TK.GOAL_DXY[task]
    nodes = np.array([[0, 0, 0], [0, dy, z_c], [gx, gy, z_c], [gx, gy, goal_top - t0]], float)
    return nodes, dict(z_c=float(z_c), goal_dz=float(goal_top - t0), goal_xy=[float(w.obj_xy[0] + gx), float(w.obj_xy[1] + gy)])


def _collides(inp: PlanInputs, nodes, n=400):
    """object volume (AABB of the box, bloated LAT_CLEAR laterally; CLEAR_M[task] above the slab except on the landing move) vs the shelf slab along the polyline.  -> first colliding offset or None."""
    from world_model.v2.nero.contents import scenes as SC
    w = inp.world
    tops = SC.support_tops(w)
    if inp.task != "box" or "shelf" not in tops:
        return None
    hx, hy, H = w.box_outer[0] / 2, w.box_outer[1] / 2, w.box_outer[2]
    sx, sy, top, t = tops["shelf"]["x"], tops["shelf"]["y"], tops["shelf"]["top"], w.shelf_t
    z0 = tops["counter"]["top"]
    for k, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        for f in np.linspace(0, 1, n):
            o = a + (b - a) * f
            cx, cy, cz = w.obj_xy[0] + o[0], w.obj_xy[1] + o[1], z0 + o[2]
            xo = (cx + hx + LAT_CLEAR > sx[0]) and (cx - hx - LAT_CLEAR < sx[1])
            yo = (cy + hy + LAT_CLEAR > sy[0]) and (cy - hy - LAT_CLEAR < sy[1])
            landing = (k == len(nodes) - 2) if (inp.wp is None or inp.wp.lands) else False      # default plan: k == 2 of 3 moves
            zmin = top - 1e-6 if landing else top + CLEAR_M[inp.task] - 1e-9        # the last move lands ON the support: only penetration counts
            zo = (cz < zmin) and (cz + H > top - t)
            if xo and yo and zo:
                return o.tolist()
    return None


def _samples(nodes, ds=DS, names=MOVES):
    """polyline -> (offsets (N,3), seg index per sample (N,), {seg: (i0, i1)}); consecutive segments share their boundary sample."""
    pts, seg, bounds = [nodes[0]], [0], {}
    for k, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        n = max(3, int(np.ceil(np.linalg.norm(b - a) / ds - 1e-9)))
        i0 = len(pts) - 1
        for f in np.linspace(0, 1, n + 1)[1:]:
            pts.append(a + (b - a) * f)
            seg.append(k)
        bounds[names[k]] = (i0, len(pts) - 1)
    return np.array(pts), np.array(seg), bounds


# ----------------------------------------------------------------------------------------------------------------------- SRS candidate nodes + DP
def _margin(arm, q):
    return float(np.min(np.minimum(q - arm.lo, arm.hi - q)))


def _psi_grid(arm, q0):
    psi0 = arm.arm_angle(q0)
    if not np.isfinite(psi0):
        psi0 = 0.0
    n = int(round(360.0 / PSI_STEP_DEG))
    return psi0 + np.arange(n) * (2 * np.pi / n), float(psi0)


def _nodes(arm, T, psis):
    """all in-limit SRS solutions of pose T over the psi grid and both elbow roots: Q (n,7), PSI (n,), M (n,) margin."""
    pr = arm._prep(T)
    Q, P, M = [], [], []
    if pr is None:
        return np.zeros((0, 7)), np.zeros(0), np.zeros(0)
    for s4 in (0, 1):
        for psi in psis:
            for q in arm._solve(pr, psi, s4).values():
                if arm.in_limits(q):
                    Q.append(q)
                    P.append(psi)
                    M.append(_margin(arm, q))
    return (np.array(Q).reshape(-1, 7), np.array(P), np.array(M))


def _limit_violation(arm, T, psis):
    """the smallest joint-limit excess [rad] over all (psi, branch) solutions of pose T (inf if the pose is out of the arm's reach): how far outside its joint limits the closest solution is."""
    pr = arm._prep(T)
    if pr is None:
        return float("inf")
    best = float("inf")
    for s4 in (0, 1):
        for psi in psis:
            for q in arm._solve(pr, psi, s4).values():
                best = min(best, float(np.max(np.maximum(arm.lo - q, q - arm.hi))))
    return best


def _best_margin(arm, T, psis):
    """largest joint margin over all (psi, branch) solutions of pose T (-inf if none)."""
    q, p, m = _nodes(arm, T, psis)
    return float(m.max()) if m.size else -np.inf


def _tcp_pose(T0, off):
    T = T0.copy()
    T[:3, 3] = T0[:3, 3] + off
    return T


def _reach_forward(Q, M, q0, mu, dq_max):
    """boolean reachability through the samples using only nodes with margin >= mu (sample 0 = the pinned hold q0).  -> list of boolean arrays or None."""
    alive = [np.array([True])]
    prev = q0[None, :]
    for i in range(1, len(Q)):
        ok = M[i] >= mu
        if not ok.any():
            return None
        d = np.max(np.abs(Q[i][None, :, :] - prev[:, None, :]), axis=2)               # (n_prev, n_i)
        a = ((d <= dq_max) & alive[-1][:, None]).any(axis=0) & ok
        if not a.any():
            return None
        alive.append(a)
        prev = Q[i]
    return alive


def _dp_chain(arm, T0, offs, q0, psis):
    """-> dict(q (N,7), psi (N,), margin (N,), mu_star, mu_used, ...) or raises PlanRefused with the first sample that has no continuation."""
    N = len(offs)
    nodes = [(_nodes(arm, _tcp_pose(T0, offs[i]), psis)) for i in range(N)]
    Q = [np.asarray(q0, float)[None, :]] + [n[0] for n in nodes[1:]]
    PSI = [np.array([arm.arm_angle(q0)])] + [n[1] for n in nodes[1:]]
    M = [np.array([_margin(arm, q0)])] + [n[2] for n in nodes[1:]]
    for i in range(1, N):
        if len(Q[i]) == 0:
            miss = _limit_violation(arm, _tcp_pose(T0, offs[i]), psis[::2])
            raise PlanRefused(f"{arm.side}: no SRS solution inside the joint limits at path sample {i}/{N - 1} (object offset {offs[i].round(4).tolist()}); the closest "
                              f"configuration (best arm angle, all branches) is {miss:.3f} rad outside a joint limit (inf = pose out of reach)",
                              arm=arm.side, sample=i, offset=offs[i].tolist(), miss_rad=miss, reason="no in-limit solution at the sample")
    if _reach_forward(Q, M, Q[0][0], 0.0, DQ_MAX) is None:                           # which sample breaks the chain (for the message)
        alive_last, prev = np.array([True]), Q[0]
        for i in range(1, N):
            d = np.max(np.abs(Q[i][None, :, :] - prev[:, None, :]), axis=2)
            a = ((d <= DQ_MAX) & alive_last[:, None]).any(axis=0)
            if not a.any():
                raise PlanRefused(f"{arm.side}: no joint-continuous chain (|dq|_inf <= {DQ_MAX} rad per {DS * 100:.0f} cm) reaches path sample {i}/{N - 1} "
                                  f"(object offset {offs[i].round(4).tolist()}); the in-limit solutions there are disconnected from the hold pose",
                                  arm=arm.side, sample=i, offset=offs[i].tolist(), reason="disconnected")
            alive_last, prev = a, Q[i]
    lo, hi = 0.0, float(max(m.max() for m in M[1:]))
    for _ in range(18):                                                              # bottleneck margin: largest mu with a feasible chain
        mu = 0.5 * (lo + hi)
        if _reach_forward(Q, M, Q[0][0], mu, DQ_MAX) is None:
            hi = mu
        else:
            lo = mu
    mu_star = lo
    mu_use = min(mu_star, M_GOOD)
    alive = _reach_forward(Q, M, Q[0][0], mu_use, DQ_MAX)
    # min-cost chain (sum of squared joint steps = smoothest) among nodes with margin >= mu_use
    cost = [np.zeros(1)]
    par = [None]
    for i in range(1, N):
        d = Q[i][None, :, :] - Q[i - 1][:, None, :]
        dd = np.max(np.abs(d), axis=2)
        c = np.sum(d * d, axis=2) + np.where((dd <= DQ_MAX) & alive[i - 1][:, None], 0.0, np.inf) + np.where(alive[i][None, :], 0.0, np.inf)
        c = c + cost[-1][:, None]
        par.append(np.argmin(c, axis=0))
        cost.append(np.min(c, axis=0))
    j = int(np.argmin(cost[-1]))
    idx = [j]
    for i in range(N - 1, 0, -1):
        j = int(par[i][j])
        idx.append(j)
    idx = idx[::-1]
    q = np.array([Q[i][idx[i]] for i in range(N)])
    psi = np.array([PSI[i][idx[i]] for i in range(N)])
    mg = np.array([M[i][idx[i]] for i in range(N)])
    sm = _smooth_chain(arm, T0, offs, q, psi, mg, mu_use)
    return dict(q=sm["q"], psi=sm["psi"], margin=sm["margin"], mu_star=float(mu_star), mu_used=float(mu_use), n_nodes=[int(len(x)) for x in Q],
                smooth=dict(sigma=sm["sigma"], max_d2_raw=sm["d2_raw"], max_d2=sm["d2"]))


def _max_d2(q):
    return float(np.abs(q[2:] - 2 * q[1:-1] + q[:-2]).max()) if len(q) > 2 else 0.0


def _smooth_chain(arm, T0, offs, q, psi, mg, mu_use):
    """the DP chain sits on a psi GRID (PSI_STEP_DEG), which makes q(s) a staircase.  Smooth psi(s) (Gaussian over SMOOTH_SIGMAS samples, first sample pinned = the hold pose),
    re-solve the IK exactly at the smoothed psi on the branch nearest the previous sample, and keep the smoothest candidate (smallest max second difference of q) that
    stays inside the joint limits with margin >= mu_use - SMOOTH_MARGIN_LOSS and |dq| <= DQ_MAX.  Falls back to the DP chain."""
    from scipy.ndimage import gaussian_filter1d
    best = dict(q=q, psi=psi, margin=mg, sigma=0.0, d2_raw=_max_d2(q), d2=_max_d2(q))
    pu = np.unwrap(psi)
    for sig in SMOOTH_SIGMAS:
        ps = gaussian_filter1d(pu, sig, mode="nearest")
        ps[0] = pu[0]
        qn, mn = [q[0]], [mg[0]]
        ok = True
        for i in range(1, len(offs)):
            pr = arm._prep(_tcp_pose(T0, offs[i]))
            if pr is None:
                ok = False
                break
            cands = [x for s4 in (0, 1) for x in arm._solve(pr, ps[i], s4).values() if arm.in_limits(x)]
            if not cands:
                ok = False
                break
            d = [float(np.max(np.abs(x - qn[-1]))) for x in cands]
            j = int(np.argmin(d))
            m_ = _margin(arm, cands[j])
            if d[j] > DQ_MAX or m_ < mu_use - SMOOTH_MARGIN_LOSS:
                ok = False
                break
            qn.append(cands[j])
            mn.append(m_)
        if not ok:
            continue
        qn = np.array(qn)
        d2 = _max_d2(qn)
        if d2 < best["d2"]:
            best = dict(q=qn, psi=ps.copy(), margin=np.array(mn), sigma=float(sig), d2_raw=best["d2_raw"], d2=d2)
    return best


def _fk_check(arm, T0, offs, q):
    """MuJoCo FK of every chosen configuration vs the target pose (position miss [m], rotation miss [rad])."""
    pe, re = np.zeros(len(offs)), np.zeros(len(offs))
    for i in range(len(offs)):
        T = arm.fk(q[i])
        Tt = _tcp_pose(T0, offs[i])
        pe[i] = np.linalg.norm(T[:3, 3] - Tt[:3, 3])
        dR = Tt[:3, :3].T @ T[:3, :3]
        re[i] = np.arctan2(np.linalg.norm([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2, (np.trace(dR) - 1) / 2)
    return pe, re


# ----------------------------------------------------------------------------------------------------------------------- the geometry (a_plan independent)
class Geometry:
    """planned object path + both arms' joint configurations at the samples.  Pure data (JSON-able via to_json)."""

    def __init__(self, task, nodes, offs, seg, bounds, q, margin, psi, fk_pos, fk_rot, info):
        self.task, self.nodes, self.offs, self.seg, self.bounds = task, nodes, offs, seg, bounds
        self.q, self.margin, self.psi, self.fk_pos, self.fk_rot, self.info = q, margin, psi, fk_pos, fk_rot, info

    def to_json(self):
        L = lambda x: np.asarray(x).tolist()
        return dict(task=self.task, nodes=L(self.nodes), offs=L(self.offs), seg=L(self.seg), bounds={k: list(v) for k, v in self.bounds.items()},
                    q={s: L(v) for s, v in self.q.items()}, margin={s: L(v) for s, v in self.margin.items()}, psi={s: L(v) for s, v in self.psi.items()},
                    fk_pos={s: L(v) for s, v in self.fk_pos.items()}, fk_rot={s: L(v) for s, v in self.fk_rot.items()}, info=self.info)

    @classmethod
    def from_json(cls, d):
        A = lambda x: np.asarray(x, float)
        return cls(d["task"], A(d["nodes"]), A(d["offs"]), np.asarray(d["seg"], int), {k: tuple(v) for k, v in d["bounds"].items()},
                   {s: A(v) for s, v in d["q"].items()}, {s: A(v) for s, v in d["margin"].items()}, {s: A(v) for s, v in d["psi"].items()},
                   {s: A(v) for s, v in d["fk_pos"].items()}, {s: A(v) for s, v in d["fk_rot"].items()}, d["info"])


def _dys(inp: PlanInputs):
    """lateral lift offsets tried: DY_GRID (default plan, or a waypoint plan whose first move is a pure rise), else only 0 (dy would change nothing)."""
    return DY_GRID if (inp.wp is None or inp.wp.pure_rise_first) else (0.0,)


def _screen_dy(inp: PlanInputs, psis):
    """-> list of (dy, bottleneck margin) over DY_GRID: best (psi, branch) margin of both arms at screening poses along the lift and the start of the carry."""
    out = []
    for dy in _dys(inp):
        nodes, _ = object_path(inp, float(dy))
        if _collides(inp, nodes) is not None:
            continue
        pts = [nodes[1] * f for f in (0.5, 0.75, 1.0)] + [nodes[1] + (nodes[2] - nodes[1]) * 0.25 if len(nodes) > 2 else nodes[1]]
        b = np.inf
        for s in SIDES:
            for o in pts:
                b = min(b, _best_margin(inp.arms[s], _tcp_pose(inp.T0[s], o), psis[s][0][::2]))
            if b <= 0:
                break
        out.append((float(dy), float(b)))
    return out


def plan_geometry(inp: PlanInputs, use_cache=True):
    key = inp.key()
    f = CACHE_DIR / f"geo_{inp.task}_{key}.json"
    if use_cache and f.exists():
        try:
            return Geometry.from_json(json.loads(f.read_text()))
        except Exception:
            pass
    psis = {s: _psi_grid(inp.arms[s], inp.q0[s]) for s in SIDES}
    scr = _screen_dy(inp, psis)
    if not scr:
        raise PlanRefused("every lateral lift offset collides with the shelf", reason="collision")
    good = sorted([x for x in scr if x[1] >= MARGIN_WANT], key=lambda x: (abs(x[0]), -x[1]))
    order = good + sorted([x for x in scr if x[1] < MARGIN_WANT], key=lambda x: -x[1])
    last_err = None
    for dy, bscr in order:
        if bscr <= 0:
            continue
        nodes, oi = object_path(inp, dy)
        offs, seg, bounds = _samples(nodes, names=(inp.wp.names if inp.wp is not None else MOVES))
        try:
            ch = {s: _dp_chain(inp.arms[s], inp.T0[s], offs, inp.q0[s], psis[s][0]) for s in SIDES}
        except PlanRefused as e:
            last_err = e
            continue
        fk = {s: _fk_check(inp.arms[s], inp.T0[s], offs, ch[s]["q"]) for s in SIDES}
        bad = [(s, i, float(fk[s][0][i])) for s in SIDES for i in range(len(offs)) if fk[s][0][i] > FK_TOL_M or fk[s][1][i] > FK_TOL_RAD]
        if bad:
            last_err = PlanRefused(f"FK validation failed (arm, sample, miss m): {bad[:3]}", reason="fk")
            continue
        info = dict(dy=float(dy), screen=scr, screen_bottleneck=float(bscr), **oi, n_samples=int(len(offs)), mu_star={s: ch[s]["mu_star"] for s in SIDES},
                    mu_used={s: ch[s]["mu_used"] for s in SIDES}, n_nodes_max={s: max(ch[s]["n_nodes"]) for s in SIDES},
                    margin_min={s: float(ch[s]["margin"].min()) for s in SIDES}, psi0={s: psis[s][1] for s in SIDES},
                    smooth={s: ch[s]["smooth"] for s in SIDES})
        g = Geometry(inp.task, nodes, offs, seg, bounds, {s: ch[s]["q"] for s in SIDES}, {s: ch[s]["margin"] for s in SIDES}, {s: ch[s]["psi"] for s in SIDES},
                     {s: fk[s][0] for s in SIDES}, {s: fk[s][1] for s in SIDES}, info)
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(g.to_json(), default=float))
                tmp.replace(f)
            except OSError:
                pass
        return g
    if last_err is not None:
        raise last_err
    nodes0, _ = object_path(inp, 0.0)                                             # name the arm and the miss at the top of the lift (dy = 0)
    miss = {s: _limit_violation(inp.arms[s], _tcp_pose(inp.T0[s], nodes0[1]), psis[s][0][::2]) for s in SIDES}
    arm_w = max(miss, key=miss.get)
    raise PlanRefused(f"no lateral lift offset on the grid has an in-limit configuration for both arms along the lift/carry start (bottleneck margin <= 0); at the lift top "
                      f"(object offset {nodes0[1].round(4).tolist()}, dy = 0) the {arm_w} arm's closest configuration is {miss[arm_w]:.3f} rad outside a joint limit "
                      f"(inf = out of reach)", reason="unreachable", arm=arm_w, offset=nodes0[1].tolist(), miss_rad=miss, screen=scr)


# ----------------------------------------------------------------------------------------------------------------------- TOPP-RA timing
class Piece:
    """one rest-to-rest piece: arrays at the control period (n+1 rows): q {side}, qd, qdd, object offset p (n+1,3), and the summary numbers."""

    def __init__(self, T, q, qd, qdd, p, pdd, qd_ratio, **meta):
        self.T, self.q, self.qd, self.qdd, self.p, self.pdd, self.qd_ratio, self.meta = T, q, qd, qdd, p, pdd, qd_ratio, meta

    @property
    def n(self):
        return self.p.shape[0] - 1

    def summary(self):
        return dict(T=round(float(self.T), 4), a_obj_max=np.abs(self.pdd).max(0).round(5).tolist(), qd_ratio_max=round(float(self.qd_ratio), 4),
                    qdd_max=round(float(max(np.abs(self.qdd[s]).max() for s in SIDES)), 3), **self.meta)


def _toppra(QL, QR, P, a_cap):
    """rest-to-rest TOPP-RA through the knots.  -> Piece."""
    import toppra as ta
    import toppra.algorithm as alg
    import toppra.constraint as tc
    P = np.asarray(P, float)
    Pstart = P[0].copy()
    P = P - Pstart
    Y = np.hstack([QL, QR, P])
    ds = np.sqrt(np.sum(np.diff(Y[:, :14], axis=0) ** 2, axis=1) + np.sum(np.diff(P, axis=0) ** 2, axis=1))
    keep = np.r_[True, ds > 1e-6]
    Y, P = Y[keep], P[keep]
    dp = np.linalg.norm(np.diff(Y[:, 14:], axis=0), axis=1)
    if dp.min() > 1e-9:                                  # the object moves between every pair of knots: parametrize by its arc length (the straight-line object offset is then exactly linear in s)
        ds = dp
    else:                                                # retrace start (the commanded q is off the path): combined joint + object metric
        ds = np.sqrt(np.sum(np.diff(Y[:, :14], axis=0) ** 2, axis=1) + dp ** 2)
    s = np.r_[0.0, np.cumsum(ds)]
    s /= s[-1]
    if len(s) < 3:                                                                   # spline needs >= 3 knots: add the midpoint
        Y = np.vstack([Y[0], 0.5 * (Y[0] + Y[-1]), Y[-1]])
        s = np.array([0.0, 0.5, 1.0])
    path = ta.SplineInterpolator(s, Y)
    v = vlim()
    vl = np.r_[v, v, np.full(3, BIG_VEL)]
    al = np.r_[np.full(14, BIG_ACC), np.full(3, float(a_cap))]
    cons = [tc.JointVelocityConstraint(np.vstack([-vl, vl]).T), tc.JointAccelerationConstraint(np.vstack([-al, al]).T)]
    inst = alg.TOPPRA(cons, path, gridpoints=np.linspace(0, 1, N_GRID), parametrizer="ParametrizeConstAccel")
    jc = inst.compute_trajectory(0, 0)
    if jc is None:
        raise PlanRefused("TOPP-RA found no feasible time parametrization", reason="toppra")
    T = float(jc.duration)
    n = max(int(np.ceil(T / DT - 1e-9)), 1)
    ts = np.minimum(np.arange(n + 1) * DT, T)
    y, yd, ydd = jc(ts, 0), jc(ts, 1), jc(ts, 2)
    q = {"left": y[:, :7], "right": y[:, 7:14]}
    qd = {"left": yd[:, :7], "right": yd[:, 7:14]}
    qdd = {"left": ydd[:, :7], "right": ydd[:, 7:14]}
    ratio = max(float(np.max(np.abs(qd[s]) / v)) for s in SIDES)
    p = y[:, 14:] + Pstart                                                           # object offset from the path start S (absolute)
    return Piece(n * DT, q, qd, qdd, p, ydd[:, 14:], ratio)


def _joint_only_piece(KL, KR, p0, t_min=0.1):
    """a tiny rest-to-rest joint move from the first to the last knot with the object not moving (smoothstep, duration >= t_min and long enough for the datasheet speeds)."""
    q0 = {"left": KL[0], "right": KR[0]}
    dq = {"left": KL[-1] - KL[0], "right": KR[-1] - KR[0]}
    v = vlim()
    T = max(t_min, 1.5 * max(float(np.max(np.abs(dq[s]) / v)) for s in SIDES) * 1.05)
    n = int(np.ceil(T / DT - 1e-9))
    T = n * DT
    a = np.arange(n + 1) * DT / T
    u, ud, udd = a * a * (3 - 2 * a), 6 * a * (1 - a) / T, (6 - 12 * a) / T ** 2
    q = {s: q0[s] + u[:, None] * dq[s] for s in SIDES}
    qd = {s: ud[:, None] * dq[s] for s in SIDES}
    qdd = {s: udd[:, None] * dq[s] for s in SIDES}
    p = np.repeat(np.asarray(p0, float)[None, :], n + 1, axis=0)
    return Piece(T, q, qd, qdd, p, np.zeros((n + 1, 3)), max(float(np.max(np.abs(qd[s]) / v)) for s in SIDES))


def _smooth_piece(Q, dur):
    """smoothstep timing (legacy retreat) through knots Q (k,7) of ONE arm over `dur` s, stretched if a joint speed limit binds.  -> (q (n+1,7), qd, T)."""
    from scipy.interpolate import CubicSpline
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    s /= s[-1]
    cs = CubicSpline(s, Q)
    v = vlim()
    T = dur
    for _ in range(30):
        n = int(np.ceil(T / DT - 1e-9))
        t = np.minimum(np.arange(n + 1) * DT, T)
        a = t / T
        u = a * a * (3 - 2 * a)
        du = 6 * a * (1 - a) / T
        q = cs(u)
        qd = cs(u, 1) * du[:, None]
        r = float(np.max(np.abs(qd) / v))
        if r <= 1.0:
            return q, qd, n * DT
        T *= r * 1.02
    raise PlanRefused("retreat cannot satisfy the joint speed limits", reason="retreat_speed")


# ----------------------------------------------------------------------------------------------------------------------- the plan object (what the episode tracks)
class ClosedChainPlan:
    """geometry + timing.  Immutable after construction except the lazily built retreat paths; `ep.cc` points to one of these."""

    def __init__(self, geo: Geometry, inp: PlanInputs, a_plan):
        self.geo, self.inp, self.a_plan = geo, inp, float(a_plan)
        self.pieces = {}
        for sg in geo.bounds:                                  # MOVES for the default plan, the waypoint plan's names otherwise
            i0, i1 = geo.bounds[sg]
            self.pieces[sg] = _toppra(geo.q["left"][i0:i1 + 1], geo.q["right"][i0:i1 + 1], geo.offs[i0:i1 + 1], self.a_plan)
        self._retreat = {}
        self._retrace_cache = {}

    # ---- nominal
    def segment(self, seg):
        return self.pieces[seg]

    def seg_margin_min(self, seg):
        i0, i1 = self.geo.bounds[seg]
        return float(min(self.geo.margin[s][i0:i1 + 1].min() for s in SIDES))

    def seg_fk_max(self, seg):
        i0, i1 = self.geo.bounds[seg]
        return float(max(self.geo.fk_pos[s][i0:i1 + 1].max() for s in SIDES))

    def accel_record(self, seg):
        pc = self.pieces[seg]
        a = float(np.abs(pc.pdd).max())
        return dict(T=round(pc.T, 4), a_plan=self.a_plan, a_cmd_peak=round(a, 5), clamped=False)

    # ---- retrace (backups): rest to rest per segment, planned joint path in reverse, a = a_backup
    def _nearest(self, seg, q_now):
        """continuous index u* in [0, n-1] on the sampled path of `seg` closest (sum of both arms, joint 2-norm) to the commanded q.  Uses the knots refined by linear interpolation."""
        i0, i1 = self.geo.bounds[seg]
        K = np.hstack([self.geo.q["left"][i0:i1 + 1], self.geo.q["right"][i0:i1 + 1]])
        qn = np.concatenate([q_now["left"], q_now["right"]])
        best = (np.inf, 0.0)
        for k in range(len(K) - 1):
            a, b = K[k], K[k + 1]
            ab = b - a
            f = float(np.clip((qn - a) @ ab / max(ab @ ab, 1e-18), 0, 1))
            d = float(np.linalg.norm(a + f * ab - qn))
            if d < best[0]:
                best = (d, k + f)
        return best[1], best[0]

    def retrace(self, q_now, segs, a_backup):
        """pieces (list of Piece) from the commanded q back to the start of segs[0], then each following segment of `segs` reversed.  Each piece rest to rest."""
        out = []
        qn = {s: np.asarray(q_now[s], float) for s in SIDES}
        for j, sg in enumerate(segs):
            i0, i1 = self.geo.bounds[sg]
            if j == 0:
                u, dist = self._nearest(sg, qn)
                k = int(np.floor(u))
                f = u - k
                if f < 1e-3:                                                          # snap to a knot (1e-3 of a 1 cm step) so TOPP-RA never sees two almost equal knots
                    f = 0.0
                elif f > 1 - 1e-3:
                    k, f = k + 1, 0.0
                # the piece STARTS at the commanded q (no reference jump); the object offset there is the projection u* on the planned path
                P0 = self.geo.offs[i0 + k] * (1 - f) + self.geo.offs[i0 + k + 1] * f if f > 0 else self.geo.offs[i0 + k]
                idx = list(range(k, -1, -1)) if (f > 0 or k == 0) else list(range(k - 1, -1, -1))     # f == 0: knot k is replaced by the commanded q
                knots_L = [qn["left"]] + [self.geo.q["left"][i0 + i] for i in idx]
                knots_R = [qn["right"]] + [self.geo.q["right"][i0 + i] for i in idx]
                P = [P0] + [self.geo.offs[i0 + i] for i in idx]
            else:
                knots_L = list(self.geo.q["left"][i0:i1 + 1][::-1])
                knots_R = list(self.geo.q["right"][i0:i1 + 1][::-1])
                P = list(self.geo.offs[i0:i1 + 1][::-1])
            KL, KR, PP = np.array(knots_L), np.array(knots_R), np.array(P)
            if np.max(np.linalg.norm(np.diff(np.hstack([KL, KR]), axis=0), axis=1), initial=0.0) < 1e-4 and len(KL) <= 2:      # already at the start (< 1e-4 rad)
                continue
            if np.linalg.norm(PP[-1] - PP[0]) < 1e-9:                                   # no object motion (the commanded q is already at the segment start): joint-only smoothstep
                out.append(_joint_only_piece(KL, KR, PP[0]))
            else:
                out.append(_toppra(KL, KR, PP, a_backup))
        return out

    # ---- retreat after release (weld off): claws move +-y by RETREAT_M, orientation fixed
    def retreat(self, where):
        """where in ("goal", "start").  Each claw translates outward (left +y, right -y) with the orientation fixed, as far as RETREAT_M allows: the largest distance
        (0.5 cm steps) whose SRS chain exists (checked by the best-psi margin, then the DP).  -> dict(q {side} (n+1,7), T, d {side} achieved retreat [m], qd_ratio)."""
        if where in self._retreat:
            return self._retreat[where]
        inp, geo = self.inp, self.geo
        i = -1 if where == "goal" else 0
        qs, dd = {}, {}
        for s in SIDES:
            sign = 1.0 if s == "left" else -1.0
            arm = inp.arms[s]
            psis, _ = _psi_grid(arm, geo.q[s][i])
            d = RETREAT_M
            while d > 1e-9 and _best_margin(arm, _tcp_pose(inp.T0[s], geo.offs[i] + np.array([0.0, sign * d, 0.0])), psis[::2]) <= 0:
                d = round(d - 0.005, 6)
            if d <= 1e-9:
                raise PlanRefused(f"{s}: no retreat of the claw is reachable from the {where} pose", arm=s, reason="retreat")
            while True:
                o = np.array([geo.offs[i] + np.array([0.0, sign * d * f, 0.0]) for f in np.linspace(0, 1, 7)])
                try:
                    ch = _dp_chain(arm, inp.T0[s], o, geo.q[s][i], psis)
                    break
                except PlanRefused:
                    d = round(d - 0.005, 6)
                    if d <= 1e-9:
                        raise
            pe, re = _fk_check(arm, inp.T0[s], o, ch["q"])
            if pe.max() > FK_TOL_M or re.max() > FK_TOL_RAD:
                raise PlanRefused(f"retreat FK validation failed on the {s} arm", arm=s, reason="fk")
            qs[s], dd[s] = ch["q"], d
        res = {s: _smooth_piece(qs[s], RETREAT_T) for s in SIDES}                      # legacy time, both claws; stretched if a joint speed limit binds
        T = max(r[2] for r in res.values())
        if abs(T - RETREAT_T) > 1e-9:
            res = {s: _smooth_piece(qs[s], T) for s in SIDES}
            T = max(r[2] for r in res.values())
        n = int(round(T / DT))
        rec = dict(q={s: _pad(res[s][0], n + 1) for s in SIDES}, T=n * DT, d=dd, qd_ratio=max(float(np.max(np.abs(res[s][1]) / vlim())) for s in SIDES))
        self._retreat[where] = rec
        return rec

    # ---- summary numbers per segment
    def summary(self):
        geo = self.geo
        out = {}
        for sg in geo.bounds:
            i0, i1 = geo.bounds[sg]
            pc = self.pieces[sg]
            out[sg] = dict(pc.summary(), fk_pos_max_m={s: float(geo.fk_pos[s][i0:i1 + 1].max()) for s in SIDES},
                           fk_rot_max_rad={s: float(geo.fk_rot[s][i0:i1 + 1].max()) for s in SIDES},
                           margin_min={s: round(float(geo.margin[s][i0:i1 + 1].min()), 4) for s in SIDES}, n_samples=int(i1 - i0 + 1))
        return out


def _pad(a, n):
    if len(a) >= n:
        return a[:n]
    return np.vstack([a, np.repeat(a[-1:], n - len(a), axis=0)])


# ----------------------------------------------------------------------------------------------------------------------- planner
_GEO = {}
_PLANS = {}


def geometry_for(inp: PlanInputs):
    k = inp.key()
    if k not in _GEO:
        g = plan_geometry(inp)
        if inp.wp is not None and inp.wp.base:                   # base route: swept clearance, height over counter 2, table-2 reach info (base_route.check; raises PlanRefused)
            g.info = dict(g.info, base=BR.check(inp.world, inp.task_name, inp.wp, g, CLEAR_M[inp.task], backend=_BASE_BACKEND))
        _GEO[k] = g
    return _GEO[k]


def plan_for(inp: PlanInputs, a_plan):
    k = (inp.key(), round(float(a_plan), 6))
    if k not in _PLANS:
        _PLANS[k] = ClosedChainPlan(geometry_for(inp), inp, a_plan)
    return _PLANS[k]


def plan_for_episode(ep):
    return plan_for(inputs_from_episode(ep), ep.a_plan)


def phases_for(seg):
    """legacy-shaped phase lists of the planned segments (executor handlers: track / retrace / jretreat)."""
    if seg == "release":
        return [("hold", 0.2), ("weld", False), ("jretreat", "goal"), ("hold", 0.5)]
    return [("track", seg), ("hold", 0.3)]


def backup_phases_for(seg, wp=None):
    """stored backup of a move.  Default plan: stop, retrace to the lift start, let go.  wp (WaypointPlan): the kind named for that move --
    set_down  = the default (retrace this move and all earlier ones in reverse to the start of the first, let go there)
    retrace   = stop, back along this move to its start, hold (claws keep the object)
    hold      = stop, hold
    let_go    = stop, let go in place (weld off, legacy 5 cm retreat)"""
    if wp is not None:
        if seg not in wp.names:
            return None
        j = wp.names.index(seg)
        kind = wp.backups[j]
        if kind == "retrace":
            return [("stop", 0.3), ("retrace", (seg,)), ("hold", 0.6)]
        if kind == "hold":
            return [("stop", 0.3), ("hold", 1.0)]
        if kind == "let_go":
            return [("stop", 0.3), ("hold", 0.3), ("weld", False), ("retreat", RETREAT_M, RETREAT_T), ("hold", 0.6)]
        if kind != "set_down":
            raise ValueError(f"unknown backup kind {kind!r}")
        k0 = wp.base[0][0] if (wp.base and j >= wp.base[0][0]) else 0    # v1: a retrace never runs back across the base move (no base retrace between arm pieces): after it, set_down
        back = list(wp.names[k0:j + 1])[::-1]                            # retraces to the start of the post-base moves and lets go THERE (the object is not back on counter 1)
    else:
        back = {"lift": ["lift"], "carry": ["carry", "lift"], "place": ["place", "carry", "lift"]}.get(seg)
    if back is None:
        return None
    return [("stop", 0.3), ("retrace", tuple(back)), ("hold", 0.3), ("weld", False), ("jretreat", "start"), ("hold", 0.6)]


def _duration(phases, plan):
    t = 0.0
    for ph in phases:
        if ph[0] == "track":
            t += plan.pieces[ph[1]].T
        elif ph[0] == "jretreat":
            t += plan.retreat(ph[1])["T"]
        elif ph[0] == "hold":
            t += float(ph[1])
    return t


class ClosedChainPlanner:
    """types.Planner for the contents carry tasks.  plan(ws, task, sim): task = SkillCall(args={"task": ..., "a_plan": ...}) or TaskSpec; sim.episode = a CarryEpisode
    (its model, world and hold pose are used).  Raises PlanRefused for an unreachable goal."""
    name = NAME

    def plan(self, ws, task, sim=None, theta=None, a_plan=None, inputs=None):
        ep = getattr(sim, "episode", None) if sim is not None else None
        args = task.args if isinstance(task, SkillCall) else {}
        name = args.get("task") or (task.name if isinstance(task, TaskSpec) else None) or getattr(ep, "task", None)
        if name is None:
            raise ValueError("ClosedChainPlanner: task name missing")
        a = args.get("a_plan", a_plan)
        a = float(a if a is not None else getattr(ep, "a_plan", 0.5))
        inp = inputs if inputs is not None else (inputs_from_episode(ep) if ep is not None else _nominal_inputs(name))
        cc = plan_for(inp, a)
        segs, t = [], 0.0
        base = inp.wp.base[0] if (inp.wp is not None and inp.wp.base) else None
        names = list(cc.geo.bounds)
        for ix, sg in enumerate(names + ["release"]):
            if base is not None and ix == base[0]:                     # the base move between arm move base[0]-1 and base[0]
                bseg = _base_segment(inp, cc, names[ix - 1], base, args.get("level"), t)
                segs.append(bseg)
                t = bseg.t1
            ph = phases_for(sg)
            dur = _duration(ph, cc)
            if sg == "release":
                meta = dict(phases=ph, backup_phases=None, accel=None, cc=cc)
                x_obj, q_ref = None, None
            else:
                pc = cc.pieces[sg]
                meta = dict(phases=ph, backup_phases=backup_phases_for(sg, inp.wp), accel=cc.accel_record(sg), cc=cc, summary=pc.summary(), piece=pc)
                x_obj = _x_obj(inp, cc, sg, base is not None and ix >= base[0])
                q_ref = _q_ref(pc)
            segs.append(Segment(mode=sg, t0=t, t1=t + dur, q_ref=q_ref, x_obj=x_obj, weld={s: sg != "release" for s in SIDES}, meta=meta))
            t += dur
        return BimanualPlan(segments=segs, planner=NAME,
                            provenance=dict(legacy=False, task=kind_of(name), a_plan=a, geometry=_geo_info(cc.geo), key=inp.key(), summary=cc.summary()),
                            assumptions=assumptions())


def _base_segment(inp, cc, prev, base, level, t0):
    """the Segment of a base move: mode "carry" (no new Mode literal), phases [("base_move", ref)], arms hold the last planned q of the previous move (ASSUMED A4).
    ref = dict(route=[[kind, value], ...], level).  meta: base_ref, accel = dict(source="base_card", level), backup_phases None (v1: no recovery run inside a base move)."""
    after, route, lv = base
    lv = level or lv or BASE_LEVEL_DEFAULT
    be = _BASE_BACKEND or BR.default_backend()
    ref = dict(route=[[k, float(v)] for k, v in route], level=lv)
    pc = cc.pieces[prev]
    q_end = {s: (pc.q[s][-1].copy(), np.zeros(7), np.zeros(7)) for s in SIDES}
    dur = BR.route_duration(route, lv, be.levels())
    meta = dict(phases=[("base_move", ref)], base_ref=ref, accel=dict(source="base_card", level=lv), backup_phases=None, cc=cc,
                end_pose=cc.geo.info.get("base", {}).get("end_pose"))
    return Segment(mode="carry", t0=t0, t1=t0 + dur, q_ref=lambda t, q=q_end: q, x_obj=None, weld={s: True for s in SIDES}, meta=meta)


def _q_ref(pc):
    def f(t):
        k = int(np.clip(round(t / DT), 0, pc.n))
        return {s: (pc.q[s][k], pc.qd[s][k], pc.qdd[s][k]) for s in SIDES}
    return f


def _x_obj(inp, cc, sg, after_base=False):
    pc = cc.pieces[sg]
    x0 = np.array([inp.world.obj_xy[0], inp.world.obj_xy[1], 0.0])
    if not after_base:
        def f(t):
            k = int(np.clip(round(t / DT), 0, pc.n))
            return np.r_[x0 + pc.p[k], 1.0, 0, 0, 0]          # z relative to the support start; orientation fixed (level)
        return f
    b = cc.geo.info["base"]                                   # after the base move: base-frame position -> world through the nominal end pose (pose of the rotation centre)
    c0, ce = np.array(b["start_pose"][:2]), np.array(b["end_pose"][:2])
    dl = b["end_pose"][2] - b["start_pose"][2]
    R = BR.rot(dl)
    qz = np.array([np.cos(dl / 2), 0.0, 0.0, np.sin(dl / 2)])

    def g(t):
        k = int(np.clip(round(t / DT), 0, pc.n))
        p = x0 + pc.p[k]
        return np.r_[np.r_[ce + R @ (p[:2] - c0), p[2]], qz]
    return g


def _geo_info(g):
    return {k: v for k, v in g.info.items() if k != "screen"} | dict(screen=g.info.get("screen"))


def assumptions():
    A = lambda v, src: dict(tag="ASSUMED", value=v, src=src)
    return dict(
        CLEAR_M=A(CLEAR_M, "object underside clearance over the highest support it passes over; implied by the user-confirmed 11-02 plans (tray 5 cm, box 3 cm over the shelf)"),
        GOAL_DXY=A(TK.GOAL_DXY, "goal xy of obj_ref relative to its start (user-confirmed 11-02; stack/tasks.py is the one copy)"),
        LAT_CLEAR=A(LAT_CLEAR, "lateral bloat of the object in the shelf collision test [m]"),
        DY_GRID=A([float(DY_GRID[0]), float(DY_GRID[-1]), 0.005], "lateral lift offset search grid [m] (min, max, step)"),
        MARGIN_WANT=A(MARGIN_WANT, "= carry.PATH_MARGIN_MIN [rad]"), M_GOOD=A(M_GOOD, "margin above which the DP prefers smoothness [rad]"),
        DS=A(DS, "path sample spacing [m]"), PSI_STEP_DEG=A(PSI_STEP_DEG, "arm angle grid step"), DQ_MAX=A(DQ_MAX, "joint continuity bound between samples [rad]"),
        object_acc=A("per-axis |a_i| <= a_plan", "the box the static rows are certified on; a_plan = admitted acceleration (envelope.A_NOM 0.5 ASSUMED, or a*)"),
        a_backup=A("carry.A_BACKUP", "ASSUMED 0.25 m/s^2"), retreat=A(dict(m=RETREAT_M, T=RETREAT_T), "legacy release retreat, smoothstep"),
        joint_speed=dict(tag="MEASURED-spec", value=list(C.MANUAL["joint_speed_deg_s"]), src="NERO datasheet, config.MANUAL (deg/s)"),
        joint_accel=dict(tag="OPEN", value=None, src="config.py has no joint acceleration limit; none imposed (TOPP-RA sentinel 1e4 rad/s^2 is not a limit); max |qdd| reported"),
        kinematics=dict(tag="MODEL", src="SRS constants read from the compiled MJCF; every chosen q FK-validated (MuJoCo)"),
        base_route=A(dict(MAX_SEGMENTS=BR.MAX_SEGMENTS, MAX_LINE_M=BR.MAX_LINE_M, MAX_TURN_DEG=BR.MAX_TURN_DEG, STEP_M=BR.STEP_M, STEP_DEG=BR.STEP_DEG, GAP_S=BR.GAP_S,
                         TRACK_MARGIN_M=BR.TRACK_MARGIN_M, level_default=BASE_LEVEL_DEFAULT),
                     "table-behind base_move ranges, sweep sampling (A7), rest gap, tracking margin 0 (no MEASURED base error wired), default level (ASSUMED; design section 8)"))


# ----------------------------------------------------------------------------------------------------------------------- static-admission hooks
def static_path(task, wp=None):
    """(offs {key: object offset}, per_seg {seg: [keys]}, cap) for carry_cert.static_cell: the planned path samples (N_STATIC per move, evenly in path
    parameter, both ends) with the planned joint configurations of both arms.  cap = capture_nyu 'pot' dict with q / tcp replaced (same layout as thresholds.capture_path)."""
    from world_model.v2.nero.contents import envelope as E
    inp = _nominal_inputs(task, wp)
    g = geometry_for(inp)
    caps = json.loads(E.CAP_FILE.read_text())["pot"]
    offs, per_seg, q, tcp = {}, {}, {}, {}
    prev_key = None
    k = 0
    for sg in g.bounds:
        i0, i1 = g.bounds[sg]
        idx = np.unique(np.round(np.linspace(i0, i1, N_STATIC + 1)).astype(int))
        keys = []
        for i in idx:
            if i == i0 and prev_key is not None:
                keys.append(prev_key)                                                # shared boundary sample
                continue
            key = f"p{k}"
            k += 1
            offs[key] = g.offs[i].copy()
            for s in SIDES:
                q.setdefault(key, {})[s] = g.q[s][i].tolist()
                tcp.setdefault(key, {})[s] = _tcp_pose(inp.T0[s], g.offs[i])[:3, 3].tolist()
            keys.append(key)
        per_seg[sg] = keys
        prev_key = keys[-1]
    return offs, per_seg, dict(caps, q=q, tcp=tcp, planner=NAME)


N_STATIC = 6                                    # path samples per move for the static rows (same count as carry_cert.N_PER_MOVE)


def goal_offset(task, wp=None):
    """total object offset S -> G of the planned path [m] (3,)."""
    return geometry_for(_nominal_inputs(task, wp)).offs[-1].copy()


def accel_records(task, a, wp=None):
    """per move: the planned object acceleration record for plan acceleration a (carry_cert.cell_accel analogue)."""
    inp = _nominal_inputs(task, wp)
    cc = plan_for(inp, a)
    return {sg: cc.accel_record(sg) for sg in cc.geo.bounds}


def main(argv=None):
    """python -m world_model.v2.stack.planners.closed_chain --tasks box,tray6 [--a 0.5]: plan (fills the on-disk cache) and print the per-segment numbers."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="box,tray6")
    ap.add_argument("--a", type=float, default=0.5)
    a = ap.parse_args(argv)
    for task in a.tasks.split(","):
        inp = _nominal_inputs(task)
        try:
            cc = plan_for(inp, a.a)
        except PlanRefused as e:
            print(task, "REFUSED:", e, e.info.get("reason"), flush=True)
            continue
        g = cc.geo
        print(f"{task}: dy {g.info['dy']:+.3f} m, z_c {g.info['z_c']:.3f} m, goal xy {g.info['goal_xy']}, samples {g.info['n_samples']}, "
              f"margin min {g.info['margin_min']}", flush=True)
        print(json.dumps(cc.summary(), default=float), flush=True)


if __name__ == "__main__":
    main()
