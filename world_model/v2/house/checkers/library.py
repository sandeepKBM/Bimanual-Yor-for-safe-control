"""checkers/library.py -- the CERTIFIED VIABILITY LIBRARY.  SIMULATION ONLY (offline, CPU, no hardware).

Idea (user): recovery should not be "go to a fixed home pose".  Every gatekeeper step already produces a CERTIFIED (nominal segment + backup-to-rest) under the
current set-valued belief.  Keep those certificates.  The union of their preconditions is a growing set of certified, viable states; recovery = get back INTO it.

Theory (docs/paper/theorems_T1_T2_draft_2026-10-16.md section 4):
  T2(a) admission of a FIXED trajectory is antitone in belief inclusion   -> reuse test is a CONTAINMENT test  b ⊑ X  (sound: lookup never returns an entry
        whose precondition X does not contain the belief, and shrinking the belief can only ADD applicable entries);
  T2(b) parameter sets only shrink by intersection (learn())             -> entries are never re-certified because of LEARNING;
  T2(c) STATE sets grow (occlusion, time)                                 -> containment can fail although the pose is right -> LOOK (re-observe) recovery;
  invalidation = FALSIFICATION only (falsified()/monitor_failure()).

Entry  = (precondition region X, certified twist profile + backup, end state, margins, progress, provenance).
X (class Ctx) is the GROWN belief the segment was admitted under: pose box (pose_wall, e_xy, e_th; pose_people), velocity box (v, v_tol), parameter
intervals (mu_lo..mu_hi, optional mass), person context (per person: centre, position radius, observation age, velocity), person-model constants, map key
and the scalar buffers (d_wall, d_person, a_cup_max, a_hw, v_cap, r_cup, footprint).  A belief view b is CONTAINED (`contained`) iff
  - the map / footprint / docks / r_cup / person model match, and b's constants are no larger than X's (v_p_max, a_p, d_person, d_wall) or no smaller (a_hw, v_cap,
    a_cup_max);  mu-interval and mass-interval of b ⊆ X's;
  - |v_b - v_X|_inf + e_v,b <= v_tol_X;
  - pose: |c_b - c_X| + rp*|dth| + e_xy,b <= e_xy,X  and  |dth| + e_th,b <= e_th,X   (rp = max path radius: a rigidly shifted path moves by <= |dxy| + rp |dth|);
  - every person j of b is either FAR (b's own reachable discs, grown by b's e_p/v_p/latency, never touch the entry's swept footprint) or COVERED by some person
    i of X:  |c_j - c_i| + s + r0_j <= r0_i  with s the path shift bound, plus latency/velocity/accel conditions.
Each condition is monotone in the set sizes => T2(a) applies.  tests/test_library.py also checks soundness against check1/check2 directly.

Hook for the gatekeeper (do NOT edit it here):  lib.hook()  ->  f(traj, bel, gk, *, t=0.0, step=None, T_S=None, dev=0.0, v0=None)
  call it from Gatekeeper._commit (see LIBRARY.md); `Library.add` re-checks check1/check2 itself, so a mis-call can never insert an uncertified entry.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..gatekeeper.belief import Belief, GKConfig
from ..gatekeeper.checks import check1, check2, world_poses
from ..gatekeeper.traj import Traj, backup_from, plan_candidate
from ..teleop import geom as GM

EPS = 1e-9
R_FAR_DEFAULT = None


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rfar(fp):
    return float(max(math.hypot(x, y) for x in fp["x"] for y in fp["y"]))


def _fp_key(fp):
    return (tuple(round(float(x), 6) for x in fp["x"]), tuple(round(float(y), 6) for y in fp["y"]))


_MAPKEYS = {}


def map_key(grid):
    """stable identity of an occupancy grid (hash of resolution, origin, shape and the EDT bytes); None -> "nomap"."""
    if grid is None:
        return "nomap"
    k = _MAPKEYS.get(id(grid))
    if k is not None and k[0] is grid:
        return k[1]
    h = hashlib.blake2b(digest_size=8)
    for nm in ("res", "x0", "y0"):
        h.update(repr(round(float(getattr(grid, nm, 0.0)), 9)).encode())
    edt = getattr(grid, "edt", None)
    if edt is not None:
        h.update(np.ascontiguousarray(edt, dtype=np.float32).tobytes())
    key = h.hexdigest()
    _MAPKEYS[id(grid)] = (grid, key)
    return key


# ----------------------------------------------------------------------------------------------------------------------------------------------------- context
@dataclass
class Ctx:
    """a set-valued state/parameter description: BOTH the live belief view (b) and an entry's precondition region (X)."""
    pose_w: np.ndarray                  # map-frame pose the static-obstacle check is anchored at (ctrl pose estimate)
    pose_p: np.ndarray                  # pose the relative people check is anchored at
    e_xy: float
    e_th: float
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_tol: float = 0.0                  # X: velocity box half-width (inf-norm); b: the velocity-estimate error e_v,state
    mu_lo: float = 0.18
    mu_hi: float = 0.35
    mass: tuple | None = None           # (lo, hi) kg of the payload; None = not modelled
    a_cup_max: float = 0.0
    r_cup: np.ndarray = field(default_factory=lambda: np.zeros(2))
    fp: dict = field(default_factory=lambda: dict(GM.FP))
    a_hw: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_cap: np.ndarray = field(default_factory=lambda: np.zeros(3))
    d_wall: float = 0.10
    d_wall_dock: float = 0.03
    dock_radius: float = 0.8
    docks: tuple = ()
    d_person: float = 0.10
    e_p: float = 0.0
    v_p_max: float = 1.5
    latency_s: float = 0.0
    a_p: float = 2.0
    e_v: float = 0.2
    person_model: str = "speed"
    pc: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))      # (n,2) person centres
    pe: np.ndarray = field(default_factory=lambda: np.zeros(0))          # (n,) total position error  e_p + per-person extra
    pl: np.ndarray = field(default_factory=lambda: np.zeros(0))          # (n,) total observation age latency + stale
    pev: np.ndarray = field(default_factory=lambda: np.zeros(0))         # (n,) total velocity-estimate error
    pvh: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))    # (n,2) velocity estimates
    map_key: str = "nomap"
    gk_key: tuple = ()

    def at(self, pose_w=None, pose_p=None, v=None, extra_lat=0.0, v_tol=None):
        """the same set seen from another pose / velocity / `extra_lat` seconds later (people discs grow; a track centre is already predicted so only the radius ages)."""
        c = replace(self)
        if pose_w is not None:
            c.pose_w = np.asarray(pose_w, float).copy()
        if pose_p is not None:
            c.pose_p = np.asarray(pose_p, float).copy()
        if v is not None:
            c.v = np.asarray(v, float).copy()
        if v_tol is not None:
            c.v_tol = float(v_tol)
        if extra_lat:
            c.pl = self.pl + float(extra_lat)
        return c


def gk_key(gk):
    return (round(float(gk.h), 9), round(float(gk.fp_step), 9), bool(gk.sweep_slack), bool(getattr(gk, "cup_chord", False)))


def ctx_from_belief(bel: Belief, gk, v=None, v_tol=0.0, mass=None) -> Ctx:
    """Ctx of a gatekeeper Belief (real dataclass or any object with the same attributes).  v: body twist estimate (default 0)."""
    n = len(bel.people)
    pe_x = list(getattr(bel, "people_e_p", ()) or ())
    ev_x = list(getattr(bel, "people_e_v", ()) or ())
    st_x = list(getattr(bel, "people_stale", ()) or ())
    vel = list(getattr(bel, "people_vel", ()) or ())
    pc = np.array([np.asarray(c, float)[:2] for c in bel.people]).reshape(n, 2)
    return Ctx(
        pose_w=np.asarray(bel.pose_wall, float).copy(), pose_p=np.asarray(bel.pose_people, float).copy(), e_xy=float(bel.e_xy), e_th=float(bel.e_th),
        v=np.zeros(3) if v is None else np.asarray(v, float).copy(), v_tol=float(v_tol), mu_lo=float(bel.mu_lo), mu_hi=float(bel.mu_hi),
        mass=None if mass is None else (float(mass[0]), float(mass[1])), a_cup_max=float(bel.a_cup_max), r_cup=np.asarray(bel.r_cup, float).copy(),
        fp={"x": list(bel.fp["x"]), "y": list(bel.fp["y"])}, a_hw=np.asarray(bel.a_hw, float).copy(), v_cap=np.asarray(bel.v_cap, float).copy(),
        d_wall=float(bel.d_wall), d_wall_dock=float(bel.d_wall_dock), dock_radius=float(bel.dock_radius), docks=tuple(sorted((round(float(d[0]), 6), round(float(d[1]), 6)) for d in bel.docks)),
        d_person=float(bel.d_person), e_p=float(bel.e_p), v_p_max=float(bel.v_p_max), latency_s=float(bel.latency_s), a_p=float(bel.a_p), e_v=float(bel.e_v),
        person_model=str(bel.person_model), pc=pc,
        pe=np.array([bel.e_p + (pe_x[i] if i < len(pe_x) else 0.0) for i in range(n)], float),
        pl=np.array([bel.latency_s + (st_x[i] if i < len(st_x) else 0.0) for i in range(n)], float),
        pev=np.array([bel.e_v + (ev_x[i] if i < len(ev_x) else 0.0) for i in range(n)], float),
        pvh=np.array([np.asarray(vel[i], float)[:2] if i < len(vel) else np.zeros(2) for i in range(n)], float).reshape(n, 2),
        map_key=map_key(bel.grid), gk_key=gk_key(gk))


# ----------------------------------------------------------------------------------------------------------------------------------------------------- path info
@dataclass
class PathInfo:
    Prel: np.ndarray        # (N+1,3) path in the start frame
    rp: float               # max |xy| of the path (rigid-shift lever arm)
    n_mov: int              # samples with v != 0 (< i_rest)
    i_rest: int
    t_rest: float
    vfar: np.ndarray        # (N+1,) sweep speed of the farthest footprint point
    rfar: float
    h: float


def path_info(traj: Traj, fp) -> PathInfo:
    P = traj.poses()
    rf = _rfar(fp)
    V = traj.V
    vpt = np.hypot(V[:, 0], V[:, 1]) + np.abs(V[:, 2]) * rf
    m = vpt.copy()
    m[1:] = np.maximum(m[1:], vpt[:-1])
    m[:-1] = np.maximum(m[:-1], vpt[1:])
    ir = traj.i_rest
    n_mov = len(V) if ir is None else ir
    return PathInfo(P, float(np.hypot(P[:, 0], P[:, 1]).max()), n_mov, len(V) if ir is None else ir, (len(V) if ir is None else ir) * traj.h, m, rf, traj.h)


# ----------------------------------------------------------------------------------------------------------------------------------------------------- containment
@dataclass
class Containment:
    ok: bool
    hard: list          # reasons looking cannot fix (map, parameters, centre outside the box, new person, ...)
    soft: dict          # name -> (current, required_max): state-set radii that re-observation could shrink
    slack: float = 0.0  # min over the pose-box slacks (diagnostic)


def _far(b: Ctx, j, P: PathInfo, extra_latency: float) -> bool:
    """person j of b never reaches the entry's swept footprint (speed disc, b's own radii; sound: the accel model only REFINES the speed disc)."""
    c = b.pc[j]
    r0 = GM.PERSON_CLEAR_R + b.pe[j]
    lat = b.pl[j] + extra_latency
    n = P.n_mov
    if n > 0:
        W = world_poses(P.Prel[:n], b.pose_p)
        tk = np.arange(n) * P.h
        d = np.hypot(W[:, 0] - c[0], W[:, 1] - c[1])
        m = d - P.rfar - (r0 + b.v_p_max * (tk + lat)) - b.d_person - 0.5 * P.h * (P.vfar[:n] + b.v_p_max)
        if float(m.min()) < 0.0:
            return False
    if P.i_rest > 0:
        W = world_poses(P.Prel[P.i_rest:P.i_rest + 1], b.pose_p)
        d = math.hypot(W[0, 0] - c[0], W[0, 1] - c[1])
        if b.v_p_max <= 0:
            return d - P.rfar >= r0
        if (d - P.rfar - r0) / b.v_p_max - lat - P.t_rest < 0.0:
            return False
    return True


def contained(b: Ctx, X: Ctx, P: PathInfo, extra_latency: float = 0.0) -> Containment:
    """b ⊑ X ?  (see module docstring).  Sound: ok=True => every margin of check1/check2 at b is >= the entry's margin at X."""
    hard, soft = [], {}
    if b.map_key != X.map_key:
        hard.append("map")
    if b.gk_key != X.gk_key:
        hard.append("gk_cfg")
    if _fp_key(b.fp) != _fp_key(X.fp):
        hard.append("footprint")
    if b.docks != X.docks or abs(b.dock_radius - X.dock_radius) > EPS:
        hard.append("docks")
    if float(np.abs(b.r_cup - X.r_cup).max()) > 1e-6:
        hard.append("r_cup")
    if b.person_model != X.person_model:
        hard.append("person_model")
    if b.mu_lo < X.mu_lo - EPS or b.mu_hi > X.mu_hi + EPS:
        hard.append("mu")
    if b.a_cup_max < X.a_cup_max - EPS:
        hard.append("cup")
    if X.mass is not None and (b.mass is None or b.mass[0] < X.mass[0] - EPS or b.mass[1] > X.mass[1] + EPS):
        hard.append("mass")
    if b.d_wall > X.d_wall + EPS or b.d_wall_dock > X.d_wall_dock + EPS or b.d_person > X.d_person + EPS:
        hard.append("buffers")
    if b.v_p_max > X.v_p_max + EPS or b.a_p > X.a_p + EPS:
        hard.append("person_bounds")
    if np.any(b.a_hw < X.a_hw - EPS) or np.any(b.v_cap < X.v_cap - EPS):
        hard.append("hw")
    # velocity box
    dv = float(np.abs(b.v - X.v).max())
    if dv > X.v_tol + 1e-7:
        hard.append("velocity")
    elif dv + b.v_tol > X.v_tol + 1e-7:
        soft["v_state"] = (b.v_tol, X.v_tol - dv)
    # pose box (path shifted rigidly by the centre offset)
    dxy = float(np.hypot(*(b.pose_w[:2] - X.pose_w[:2])))
    dth = abs(_wrap(b.pose_w[2] - X.pose_w[2]))
    tx = X.e_xy - dxy - P.rp * dth
    tt = X.e_th - dth
    slack = min(tx - b.e_xy, tt - b.e_th)
    if tx < -EPS or tt < -EPS:
        hard.append("pose_center")
    else:
        if b.e_xy > tx + EPS:
            soft["e_xy"] = (b.e_xy, tx)
        if b.e_th > tt + EPS:
            soft["e_th"] = (b.e_th, tt)
    # people: far OR covered
    if len(b.pc):
        dp = float(np.hypot(*(b.pose_p[:2] - X.pose_p[:2])))
        dpt = abs(_wrap(b.pose_p[2] - X.pose_p[2]))
        s = dp + (P.rp + P.rfar) * dpt
        for j in range(len(b.pc)):
            if _far(b, j, P, extra_latency):
                continue
            best = None                                                                    # (n_soft, -slack, i, soft dict)
            for i in range(0 if (extra_latency > 1e-12 and b.person_model == "accel") else len(X.pc)):      # aged accel centres are not modelled -> only the far test applies
                dc = float(np.hypot(*(b.pc[j] - X.pc[i])))
                need = {}
                tgt = X.pe[i] - dc - s
                if tgt < -EPS:
                    continue                                                               # centre outside the entry's disc: not fixable by shrinking radii
                if b.pe[j] > tgt + EPS:
                    need[f"person{j}:e_p"] = (b.pe[j], tgt)
                lat_t = X.pl[i] - extra_latency
                if b.pl[j] > lat_t + EPS:
                    if lat_t < -EPS:
                        continue
                    need[f"person{j}:age"] = (b.pl[j], lat_t)
                if b.person_model == "accel":
                    dvh = float(np.hypot(*(b.pvh[j] - X.pvh[i])))
                    evt = X.pev[i] - dvh
                    if evt < -EPS:
                        continue
                    if b.pev[j] > evt + EPS:
                        need[f"person{j}:e_v"] = (b.pev[j], evt)
                key = (len(need), -tgt, i)
                if best is None or key < best[0]:
                    best = (key, need)
            if best is None:
                hard.append(f"person{j}:uncovered")
            else:
                soft.update(best[1])
    return Containment(not hard and not soft, hard, soft, slack)


# ----------------------------------------------------------------------------------------------------------------------------------------------------- entries
@dataclass
class Entry:
    id: int
    X: Ctx
    V: np.ndarray                       # (N+1,3) certified twist profile (nominal + backup), start-frame poses derive from it
    h: float
    t_sw: float
    T_S: float
    P: PathInfo
    pose_end: np.ndarray                # world pose (pose_w frame) where the backup comes to rest
    ready: dict                         # task-ready flags at the end state (rest, cup level, ...)
    margins: dict                       # check1+check2 margins at the BASE (ungrown) belief
    margins_x: dict                     # ... at the grown precondition region X (tightest = the certified boundary)
    progress: float                     # task-route distance gained between start and end pose [m]
    rem_end: float                      # remaining task distance at the end pose [m]
    prov: dict                          # provenance: step, t, T_S, dev, belief snapshot
    tags: frozenset = frozenset({"mu", "mass", "v_p", "a_p", "e_p", "pose", "map"})
    hits: int = 0
    created: float = 0.0

    def traj(self, t0=0.0) -> Traj:
        return Traj(self.V.copy(), self.h, t0, self.t_sw, self.T_S)

    @property
    def at_rest_start(self):
        return float(np.abs(self.X.v).max()) <= 1e-6 and self.X.v_tol <= 1e-3

    def summary(self):
        return dict(id=self.id, pose=self.X.pose_w.round(3).tolist(), e_xy=round(self.X.e_xy, 3), v0=self.X.v.round(3).tolist(), T_S=self.T_S, n=len(self.V) - 1,
                    progress=round(self.progress, 3), end=self.pose_end.round(3).tolist(), hits=self.hits)


@dataclass
class Match:
    entry: Entry
    info: Containment


@dataclass
class Leg:
    kind: str                       # "stop" | "translate" | "rotate"
    traj: Traj
    bel: Belief                     # the belief this leg was certified under (re-anchored, latency-aged)
    pose_w0: np.ndarray
    pose_p0: np.ndarray
    pose_w1: np.ndarray
    pose_p1: np.ndarray
    t_off: float
    margins: dict


@dataclass
class Recovery:
    mode: str                       # "direct" | "move" | "look" | "none"
    entry: Entry | None = None
    matches: list = field(default_factory=list)
    legs: list = field(default_factory=list)
    t_connect: float = 0.0
    look: dict | None = None        # {entry_id, reduce: {name: (current, required_max)}, kind}
    reason: str = ""
    ms: float = 0.0


# ----------------------------------------------------------------------------------------------------------------------------------------------------- progress
class RouteProgress:
    """remaining arc length along an xy polyline from the projection of a point (task-route progress for ranking)."""

    def __init__(self, route):
        self.R = np.asarray(route, float)[:, :2]
        seg = np.diff(self.R, axis=0)
        self.L = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.L)])

    def remaining(self, xy):
        xy = np.asarray(xy, float)[:2]
        best, bs = np.inf, 0.0
        for i in range(len(self.L)):
            a, d = self.R[i], self.R[i + 1] - self.R[i]
            f = float(np.clip(np.dot(xy - a, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0))
            q = a + f * d
            dist = float(np.hypot(*(xy - q)))
            if dist < best:
                best, bs = dist, self.cum[i] + f * self.L[i]
        return float(self.cum[-1] - bs)


def _compose(pose, rel):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([pose[0] + c * rel[0] - s * rel[1], pose[1] + s * rel[0] + c * rel[1], pose[2] + rel[2]])


# ----------------------------------------------------------------------------------------------------------------------------------------------------- library
class Library:
    def __init__(self, gk: GKConfig | None = None, progress_fn=None, cell: float = 0.5, max_entries: int = 4000, merge_tol: float = 0.01,
                 grow: dict | None = None, v_nom: float = 0.3):
        """progress_fn: xy -> remaining task distance (e.g. RouteProgress(route).remaining); None -> progress 0.
        grow: default precondition growth {"xy": max extra pose half-width, "p": max extra person position error, "mu": min mu_lo}; None = no growth.
        v_nom: nominal speed for the time-to-go ranking of MOVE targets."""
        self.gk = gk or GKConfig()
        self.progress_fn = progress_fn
        self.max_entries = max_entries
        self.merge_tol = merge_tol
        self.grow = grow
        self.v_nom = v_nom
        self.entries: dict[int, Entry] = {}
        self._next = 0
        self._tree = None
        self._tree_ids = []
        self._dirty = True
        self._max_exy = 0.0
        self.invalidated: list = []
        self.theta = {"mu": None, "mass": None}                   # current parameter sets learned so far (intersection)
        self.n_added = self.n_rejected = self.n_merged = self.n_dominated = 0
        self.lookup_ms: list = []

    def __len__(self):
        return len(self.entries)

    # ---------------------------------------------------------------------------------------------------------------- add
    def _admit(self, traj, bel, gk):
        o1, m1, r1, mg1 = check1(traj, bel, gk)
        o2, m2, r2, mg2 = check2(traj, bel, gk)
        return o1 and o2, {**mg1, **mg2}

    def grow_region(self, traj, bel, gk=None, xy=0.0, p=0.0, mu_floor=None, iters=9):
        """largest precondition region (sequentially: pose box, then person position error, then mu_lo) under which the FIXED trajectory is still admitted.
        Bisection is sound because admission is antitone in each parameter (T2a); the final joint check is re-run.  -> Belief."""
        gk = gk or self.gk

        def ok(b):
            return self._admit(traj, b, gk)[0]

        cur = bel
        if xy > 0:
            lo, hi = 0.0, xy
            if ok(cur.with_(e_xy=bel.e_xy + hi, e_th=bel.e_th + gk.th_over_xy * hi)):
                lo = hi
            else:
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if ok(cur.with_(e_xy=bel.e_xy + mid, e_th=bel.e_th + gk.th_over_xy * mid)) else (lo, mid)
            cur = cur.with_(e_xy=bel.e_xy + lo, e_th=bel.e_th + gk.th_over_xy * lo)
        if p > 0 and len(bel.people):
            lo, hi = 0.0, p
            if ok(cur.with_(e_p=bel.e_p + hi)):
                lo = hi
            else:
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if ok(cur.with_(e_p=bel.e_p + mid)) else (lo, mid)
            cur = cur.with_(e_p=bel.e_p + lo)
        if mu_floor is not None and mu_floor < bel.mu_lo:
            lo, hi = mu_floor, bel.mu_lo                                     # lo = smallest tried (maybe infeasible), hi = known feasible
            if ok(cur.with_(mu_lo=lo)):
                hi = lo
            else:
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (lo, mid) if ok(cur.with_(mu_lo=mid)) else (mid, hi)
            cur = cur.with_(mu_lo=hi)
        assert ok(cur), "grow_region produced an inadmissible region (non-monotone checker?)"
        return cur

    def add(self, traj: Traj, bel: Belief, gk: GKConfig | None = None, *, v0=None, t: float = 0.0, step=None, T_S=None, dev: float = 0.0, v_tol: float = 0.0,
            mass=None, grow: dict | None = None, ready: dict | None = None, extra_prov: dict | None = None):
        """insert a certified segment.  Returns the Entry, the dominating existing entry (dedupe), or None if the trajectory is NOT admitted under `bel`."""
        gk = gk or self.gk
        if v0 is None:
            v0 = traj.V[0]
        ok, mg = self._admit(traj, bel, gk)
        if not ok:
            self.n_rejected += 1
            return None
        g = self.grow if grow is None else grow
        belx = bel
        if g:
            belx = self.grow_region(traj, bel, gk, xy=g.get("xy", 0.0), p=g.get("p", 0.0), mu_floor=g.get("mu"))
        okx, mgx = self._admit(traj, belx, gk)
        X = ctx_from_belief(belx, gk, v=v0, v_tol=v_tol, mass=mass)
        P = path_info(traj, X.fp)
        pend = _compose(X.pose_w, P.Prel[-1])
        pr0 = float(self.progress_fn(X.pose_w[:2])) if self.progress_fn else 0.0
        pr1 = float(self.progress_fn(pend[:2])) if self.progress_fn else 0.0
        e = Entry(self._next, X, traj.V.copy(), traj.h, traj.t_sw, float(traj.T_S if T_S is None else T_S), P, pend,
                  dict(ready or {}, rest=traj.i_rest is not None, cup_ok=mg.get("cup", 0.0) >= 0.0),
                  {k: round(float(v), 6) for k, v in mg.items()}, {k: round(float(v), 6) for k, v in mgx.items()}, pr0 - pr1, pr1,
                  dict(step=step, t=float(t), T_S=float(traj.T_S if T_S is None else T_S), dev=float(dev), e_xy0=float(bel.e_xy), e_xy_grown=float(belx.e_xy),
                       n_people=len(bel.people), mu_lo=float(bel.mu_lo), **(extra_prov or {})), created=time.time())
        return self._insert(e)

    def _equiv(self, a: Entry, b: Entry):
        return len(a.V) == len(b.V) and abs(a.h - b.h) < 1e-12 and float(np.abs(a.V - b.V).max()) < 1e-3

    def _insert(self, e: Entry):
        near = self._near_ids(e.X.pose_w[:2], max(self.merge_tol, 1e-9) + 1e-12) if self.entries else []
        drop = []
        for i in near:
            o = self.entries[i]
            if not self._equiv(o, e) or float(np.hypot(*(o.X.pose_w[:2] - e.X.pose_w[:2]))) > self.merge_tol:
                continue
            if contained(e.X, o.X, o.P).ok:                                                               # o dominates e (e's whole X lies inside o's)
                o.hits += 1
                self.n_dominated += 1
                return o
            if contained(o.X, e.X, e.P).ok:                                                               # e dominates o
                drop.append(i)
        for i in drop:
            del self.entries[i]
            self.n_merged += 1
        self._next += 1
        self.entries[e.id] = e
        self._dirty = True
        self._max_exy = max(self._max_exy, e.X.e_xy)
        self.n_added += 1
        if len(self.entries) > self.max_entries:                                                          # evict the least useful (fewest hits, oldest)
            victim = min(self.entries.values(), key=lambda q: (q.hits, q.created))
            del self.entries[victim.id]
        return e

    def hook(self, **defaults):
        """-> f(traj, bel, gk, *, t=0.0, step=None, T_S=None, dev=0.0, v0=None) the gatekeeper calls when it COMMITS a certified candidate."""
        def f(traj, bel, gk=None, *, t=0.0, step=None, T_S=None, dev=0.0, v0=None, **kw):
            return self.add(traj, bel, gk, v0=v0, t=t, step=step, T_S=T_S, dev=dev, **{**defaults, **kw})
        return f

    # ---------------------------------------------------------------------------------------------------------------- spatial index
    def _build(self):
        if not self._dirty:
            return
        ids = list(self.entries)
        self._tree_ids = ids
        if ids:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(np.array([self.entries[i].X.pose_w[:2] for i in ids]))
        else:
            self._tree = None
        self._dirty = False
        self._max_exy = max((self.entries[i].X.e_xy for i in ids), default=0.0)

    def _near_ids(self, xy, r):
        self._build()
        if self._tree is None:
            return []
        return [self._tree_ids[k] for k in self._tree.query_ball_point(np.asarray(xy, float)[:2], r)]

    def _knn(self, xy, k):
        self._build()
        if self._tree is None:
            return []
        k = min(k, len(self._tree_ids))
        d, idx = self._tree.query(np.asarray(xy, float)[:2], k=k)
        d, idx = np.atleast_1d(d), np.atleast_1d(idx)
        return [(float(dd), self._tree_ids[int(ii)]) for dd, ii in zip(d, idx)]

    # ---------------------------------------------------------------------------------------------------------------- lookup
    def _view(self, belief, gk, v, e_v_state, mass):
        if isinstance(belief, Ctx):
            return belief
        c = ctx_from_belief(belief, gk, v=v, v_tol=e_v_state, mass=mass)
        return c

    def lookup(self, belief, v=None, *, e_v_state: float = 0.0, mass=None, extra_latency: float = 0.0, gk=None, verify: bool = False) -> list:
        """applicable entries by CONTAINMENT (sound), ranked by progress (desc), then by the tightest certified margin.  belief: gatekeeper Belief (or Ctx).
        verify=True additionally re-runs check1/check2 on every returned entry under `belief` (belt and braces; costs ms)."""
        t0 = time.perf_counter()
        gk = gk or self.gk
        b = self._view(belief, gk, v, e_v_state, mass)
        self._build()
        out = []
        rq = self._max_exy + 1e-6                       # containment needs |c_b - c_X| <= e_xy,X - e_xy,b - rp |dth| <= e_xy,X
        for i in self._near_ids(b.pose_w[:2], rq):
            e = self.entries[i]
            c = contained(b, e.X, e.P, extra_latency)
            if c.ok:
                if verify and not isinstance(belief, Ctx):
                    ok, _ = self._admit(e.traj(), belief, gk)
                    if not ok:
                        raise AssertionError(f"containment accepted entry {i} but check1/check2 reject it (soundness bug)")
                out.append(Match(e, c))
        out.sort(key=lambda m: (-m.entry.progress, -min(m.entry.margins_x.values(), default=0.0), m.entry.id))
        for m in out:
            m.entry.hits += 1
        self.lookup_ms.append((time.perf_counter() - t0) * 1e3)
        return out

    # ---------------------------------------------------------------------------------------------------------------- parameter learning / falsification
    def learn(self, mu=None, mass=None) -> dict:
        """T2(b): intersect the stored parameter sets with new evidence.  Entries stay valid (their X ⊇ any subset).  Empty intersection = FALSIFICATION:
        entries that do not cover the observed set are dropped.  -> {mu: set or None, falsified: [names], dropped: n}"""
        out = {"falsified": [], "dropped": 0}
        for name, obs in (("mu", mu), ("mass", mass)):
            if obs is None:
                continue
            obs = (float(obs[0]), float(obs[1]))
            cur = self.theta[name]
            new = obs if cur is None else (max(cur[0], obs[0]), min(cur[1], obs[1]))
            if new[0] > new[1] + EPS:
                out["falsified"].append(name)
                out["dropped"] += self.falsified(name, obs)
                self.theta[name] = obs                                                    # restart from the evidence (regime change)
            else:
                self.theta[name] = new
        out["mu"], out["mass"] = self.theta["mu"], self.theta["mass"]
        return out

    def falsified(self, kind: str, observed) -> int:
        """an assumption was observed to be violated.  Drop every entry certified under an assumption that does NOT cover the observation.
        kind: "mu" (interval) | "mass" (interval) | "v_p" (observed person speed) | "a_p" (person accel) | "e_p" (observed track error) | "pose" (observed pose error)
        | "map" (new grid or its key)."""
        def bad(e: Entry):
            X = e.X
            if kind == "mu":
                return not (X.mu_lo <= observed[0] + EPS and X.mu_hi >= observed[1] - EPS)
            if kind == "mass":
                return X.mass is not None and not (X.mass[0] <= observed[0] + EPS and X.mass[1] >= observed[1] - EPS)
            if kind == "v_p":
                return X.v_p_max < float(observed) - EPS
            if kind == "a_p":
                return X.a_p < float(observed) - EPS
            if kind == "e_p":
                return (len(X.pe) and float(X.pe.max()) < float(observed) - EPS) or (not len(X.pe) and X.e_p < float(observed) - EPS)
            if kind == "pose":
                return X.e_xy < float(observed) - EPS
            if kind == "map":
                return X.map_key != (observed if isinstance(observed, str) else map_key(observed))
            raise ValueError(kind)
        return self.drop_where(bad, f"falsified:{kind}")

    def monitor_failure(self, tags=("mu", "v_p", "pose"), reason="monitor") -> int:
        """the gatekeeper's online monitor failed (n_falsified): the failing assumption is unknown -> drop every entry that carries one of the suspect tags."""
        return self.drop_where(lambda e: bool(set(tags) & set(e.tags)), f"monitor:{reason}")

    def drop_where(self, pred, reason="") -> int:
        ids = [i for i, e in self.entries.items() if pred(e)]
        for i in ids:
            self.invalidated.append((i, reason))
            del self.entries[i]
        if ids:
            self._dirty = True
        return len(ids)

    # ---------------------------------------------------------------------------------------------------------------- MOVE connector
    def _leg_bel(self, bel, pw, pp, t_off):
        """belief re-anchored at a leg start, `t_off` seconds after the observation: people discs aged (latency), speed model only (centre advance of the accel
        model is not defined for the aged observation -> the always-valid speed disc is used)."""
        return bel.with_(pose_wall=np.asarray(pw, float).copy(), pose_people=np.asarray(pp, float).copy(), latency_s=bel.latency_s + t_off, person_model="speed")

    def _certify_leg(self, kind, traj, bel, gk, pw, pp, t_off):
        lb = self._leg_bel(bel, pw, pp, t_off)
        ok, mg = self._admit(traj, lb, gk)
        P = traj.poses()[-1]
        return ok, Leg(kind, traj, lb, pw, pp, _compose(pw, P), _compose(pp, P), t_off, {k: round(float(v), 5) for k, v in mg.items()}), mg

    T_GRID = (0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0)

    def _fit_leg(self, bel, gk, u_dir, dist, kind):
        """rest -> rest leg: hold m*u_dir for T_S then cup-safe backup, with m in (0,1] bisected so the displacement (translate: |xy|, rotate: |yaw|) equals `dist`."""
        idx = 2 if kind == "rotate" else None

        def disp(tr):
            p = tr.poses()[-1]
            return abs(p[2]) if idx == 2 else float(np.hypot(p[0], p[1]))

        v0 = np.zeros(3)
        for T_S in self.T_GRID:
            if disp(plan_candidate(v0, u_dir, T_S, bel, gk)) >= dist:
                lo, hi = 0.0, 1.0
                for _ in range(26):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if disp(plan_candidate(v0, mid * u_dir, T_S, bel, gk)) < dist else (lo, mid)
                return plan_candidate(v0, hi * u_dir, T_S, bel, gk)
        return None

    def plan_connector(self, bel, v, target_w, gk=None, order="TR"):
        """certified rest-to-rest connector from the belief's pose (and twist v) to the world pose target_w=(x,y,th): optional stop leg, then a straight
        translation (heading fixed, holonomic base) and a pure rotation, each fitted by bisection and CERTIFIED with check1+check2 under the belief re-anchored at
        the leg start and aged by the time already elapsed.  -> (legs, reason); legs=None if any leg is inadmissible."""
        gk = gk or self.gk
        pw, pp = np.asarray(bel.pose_wall, float).copy(), np.asarray(bel.pose_people, float).copy()
        t_off = 0.0
        legs = []
        v = np.zeros(3) if v is None else np.asarray(v, float)
        if float(np.abs(v).max()) > 1e-6:
            tr = backup_from(v, bel, gk)
            ok, leg, mg = self._certify_leg("stop", tr, bel, gk, pw, pp, t_off)
            if not ok:
                return None, "stop_leg_inadmissible"
            legs.append(leg)
            pw, pp, t_off = leg.pose_w1, leg.pose_p1, t_off + tr.N * tr.h
        target_w = np.asarray(target_w, float)
        for step in order:
            if step == "T":
                d = target_w[:2] - pw[:2]
                dist = float(np.hypot(*d))
                if dist < 2e-3:
                    continue
                c, s = math.cos(pw[2]), math.sin(pw[2])
                db = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]) / dist
                vmax = min(bel.v_cap[0] / max(abs(db[0]), 1e-9), bel.v_cap[1] / max(abs(db[1]), 1e-9))
                u = np.array([db[0], db[1], 0.0]) * min(vmax, 1.0)
                kind = "translate"
                tr = self._fit_leg(self._leg_bel(bel, pw, pp, t_off), gk, u, dist, kind)
            else:
                dth = _wrap(target_w[2] - pw[2])
                if abs(dth) < 2e-3:
                    continue
                u = np.array([0.0, 0.0, math.copysign(bel.v_cap[2], dth)])
                kind = "rotate"
                tr = self._fit_leg(self._leg_bel(bel, pw, pp, t_off), gk, u, abs(dth), kind)
            if tr is None:
                return None, f"{kind}_too_far"
            ok, leg, mg = self._certify_leg(kind, tr, bel, gk, pw, pp, t_off)
            if not ok:
                bad = min(mg, key=mg.get)
                return None, f"{kind}_inadmissible:{bad}"
            legs.append(leg)
            pw, pp, t_off = leg.pose_w1, leg.pose_p1, t_off + tr.N * tr.h
        return legs, "ok"

    # ---------------------------------------------------------------------------------------------------------------- recover
    def recover(self, belief, v=None, *, e_v_state: float = 0.0, mass=None, gk=None, k_nearest: int = 8, max_connect: float = 3.0, allow=("direct", "move", "look")) -> Recovery:
        """a plan to get back INTO the library.  direct (an applicable entry exists) > move (short certified connector to some entry's precondition region)
        > look (state set too large for containment: re-observe) > none."""
        t0 = time.perf_counter()
        gk = gk or self.gk
        b = self._view(belief, gk, v, e_v_state, mass)
        vv = np.zeros(3) if v is None else np.asarray(v, float)
        if "direct" in allow:
            ms = self.lookup(b, gk=gk)
            if ms:
                return Recovery("direct", ms[0].entry, ms, reason="applicable entry (containment)", ms=(time.perf_counter() - t0) * 1e3)
        if self.entries and not isinstance(belief, Ctx):
            bel = belief
        else:
            bel = None
        cands = self._knn(b.pose_w[:2], k_nearest * 3) if self.entries else []
        move_note = "no_candidates"
        if "move" in allow and bel is not None:
            best = None
            tried = 0
            for dist, i in cands:
                e = self.entries[i]
                if dist > max_connect or tried >= k_nearest:
                    continue
                if not e.at_rest_start:
                    continue                                                               # a rest-to-rest connector can only land in a rest-start precondition
                # cheap prefilter: parameters/map/hw etc. and a radius that can ever fit
                c0 = contained(b.at(pose_w=e.X.pose_w, pose_p=b.pose_p, v=np.zeros(3), v_tol=0.0), e.X, e.P)
                if c0.hard and any(not h.startswith("person") and h != "pose_center" for h in c0.hard):
                    move_note = "preconditions:" + ",".join(c0.hard)
                    continue
                if b.e_xy > e.X.e_xy + EPS or b.e_th > e.X.e_th + EPS:
                    move_note = "state_set_too_large"
                    continue
                tried += 1
                for order in ("TR", "RT"):
                    legs, why = self.plan_connector(bel, vv, e.X.pose_w, gk, order)
                    if legs is None:
                        move_note = why
                        continue
                    T = sum(l.traj.N * l.traj.h for l in legs)
                    end_w = legs[-1].pose_w1 if legs else b.pose_w
                    end_p = legs[-1].pose_p1 if legs else b.pose_p
                    arr = b.at(pose_w=end_w, pose_p=end_p, v=np.zeros(3), v_tol=0.0)
                    c = contained(arr, e.X, e.P, extra_latency=T)
                    if not c.ok:
                        move_note = "arrival_not_contained:" + ",".join(c.hard + list(c.soft))
                        continue
                    cost = T + e.rem_end / self.v_nom
                    if best is None or cost < best[0]:
                        best = (cost, e, legs, T)
                    break
            if best is not None:
                return Recovery("move", best[1], legs=best[2], t_connect=best[3], reason=f"connector {best[3]:.2f}s then entry {best[1].id}", ms=(time.perf_counter() - t0) * 1e3)
        if "look" in allow:
            lk = self._look(b, cands)
            if lk is not None:
                return Recovery("look", self.entries[lk["entry_id"]], look=lk, reason=f"state set too large for entry {lk['entry_id']}", ms=(time.perf_counter() - t0) * 1e3)
        return Recovery("none", reason=f"no direct/move/look ({move_note})", ms=(time.perf_counter() - t0) * 1e3)

    def _look(self, b: Ctx, cands):
        """the best entry for which ONLY state-set radii (soft deficits) stand between b and containment: re-observing to reduce them would give a direct hit."""
        best = None
        for _, i in cands:
            e = self.entries[i]
            c = contained(b, e.X, e.P)
            if c.ok or c.hard or not c.soft:
                continue
            ratio = max(cur / max(req, 1e-6) for cur, req in c.soft.values())
            key = (ratio, -e.progress)
            if best is None or key < best[0]:
                best = (key, dict(entry_id=e.id, kind="re-observe", reduce={k: (float(a), float(r)) for k, (a, r) in c.soft.items()}, deficit_ratio=float(ratio)))
        return None if best is None else best[1]

    def stats(self):
        ms = np.array(self.lookup_ms) if self.lookup_ms else np.zeros(1)
        return dict(n=len(self.entries), added=self.n_added, rejected=self.n_rejected, dominated=self.n_dominated, merged=self.n_merged,
                    invalidated=len(self.invalidated), lookup_ms_mean=float(ms.mean()), lookup_ms_p95=float(np.percentile(ms, 95)), lookup_ms_max=float(ms.max()))


# ----------------------------------------------------------------------------------------------------------------------------------------------------- offline demo
def _yaw_along(route, wp, i):
    """heading used for synthetic rest poses: the waypoint yaw of the lab start for the first leg, then facing the counter (-pi/2)."""
    return wp["lab_start"][2] if i == 0 else wp["door1"][2]


def run_demo(out_dir="/data/local/ss5772/molmospaces/gatekeeper/library_demo", seed=0, episode_s=6.0, spacing=0.25):
    """OFFLINE (MuJoCo CPU, no GL, no hardware): (1) run a short gatekeeper episode on h5 and feed every COMMITTED certified segment to the library through
    a Gatekeeper subclass (the gatekeeper itself is untouched); (2) add rest-start entries synthesised every `spacing` m along the h5 carry route
    (plan_candidate on a truth belief of a TeleopSim); (3) perturb beliefs and report which recovery mode recover() chooses."""
    import json
    import os
    from ..gatekeeper.gate import Gatekeeper
    from ..teleop import run as RUN
    from ..teleop import shield as SH
    from ..teleop.sim import TeleopSim, TICK
    from ..gatekeeper.belief import build_belief

    os.makedirs(out_dir, exist_ok=True)
    gk = GKConfig()
    sim0 = TeleopSim("h5", grasped=True, seed=seed, with_people=True)
    route = sim0.scene.out_route
    wp = sim0.scene.waypoints
    prog = RouteProgress(route)
    lib = Library(gk, progress_fn=prog.remaining, grow=dict(xy=0.15, p=0.3, mu=0.10))
    t_all0 = time.perf_counter()

    class LibGK(Gatekeeper):
        n_cb = 0

        def _commit(self, tr, T_S, dev, *a, **k):                      # tolerant of the gatekeeper's evolving _commit signature (bel, t appended later)
            super()._commit(tr, T_S, dev, *a, **k)
            LibGK.n_cb += 1
            bel = a[0] if a and a[0] is not None else self.last_bel       # the belief the candidate was actually certified under
            lib.add(tr, bel, self.gk, v0=tr.V[0], t=self.n_calls * TICK, step=self.n_steps, T_S=T_S, dev=dev)

    g = LibGK(gk=gk)
    t0 = time.perf_counter()
    summ, _ = RUN.run_episode(True, seed, "carry", "h5", episode_s, with_people=True, verbose=False, filt=g)
    n_ep = len(lib)
    ep_info = dict(sim_s=episode_s, wall_s=round(time.perf_counter() - t0, 1), commits=LibGK.n_cb, gk_steps=g.n_steps, entries_after_episode=n_ep,
                   added=lib.n_added, dominated=lib.n_dominated, merged=lib.n_merged, rejected=lib.n_rejected)

    cfg = SH.ShieldCfg()
    base = build_belief(sim0, cfg, gk, fp=GM.FP, docks=[wp["cafe_counter_pick"][:2]])
    R = np.asarray(route, float)
    pts, i_seg = [], 0
    for i in range(len(R) - 1):
        d = R[i + 1] - R[i]
        L = float(np.hypot(*d))
        for s in np.arange(0.0, L - 1e-9, spacing):
            pts.append((R[i] + d / L * s, d / L, i))
    t0 = time.perf_counter()
    n_try = 0
    for xy, dirn, i in pts:
        yaw = _yaw_along(route, wp, i)
        pose = np.array([xy[0], xy[1], yaw])
        b = base.with_(pose_wall=pose.copy(), pose_people=pose.copy())
        c, s = math.cos(yaw), math.sin(yaw)
        ub = np.array([c * dirn[0] + s * dirn[1], -s * dirn[0] + c * dirn[1], 0.0]) * 0.3
        for T_S in (0.6, 0.1):
            n_try += 1
            lib.add(plan_candidate(np.zeros(3), ub, T_S, b, gk), b, gk, T_S=T_S, step=-1, extra_prov=dict(source="synthetic_route"))
    synth = dict(poses=len(pts), tried=n_try, wall_s=round(time.perf_counter() - t0, 1))

    # ---- perturbed beliefs
    rest = [e for e in lib.entries.values() if e.at_rest_start and e.prov.get("source") == "synthetic_route"]
    pp = [np.asarray(c, float)[:2] for c in base.people]
    # probe entry = the rest entry farthest from every person (so the connector cases are not dominated by the walker / standing person of h5)
    e0 = max(rest, key=lambda e: (min((float(np.hypot(*(e.X.pose_w[:2] - q))) for q in pp), default=9.0), -abs(e.rem_end - 2.0)))
    p0 = e0.X.pose_w.copy()
    fwd = np.array([-math.sin(p0[2]), math.cos(p0[2])])           # sideways in the world frame (perpendicular to the body x axis)

    def mk(dxy=(0.0, 0.0), dth=0.0, **kw):
        pose = np.array([p0[0] + dxy[0], p0[1] + dxy[1], p0[2] + dth])
        return base.with_(pose_wall=pose, pose_people=pose.copy(), **kw)

    e_cap = e0.X.e_xy
    cases = {
        "A exact pose, e_xy=0.02": mk(e_xy=0.02, e_th=0.003),
        "B 0.6 m sideways, e_xy=0.02": mk(dxy=tuple(0.6 * fwd), dth=0.1, e_xy=0.02, e_th=0.003),
        "C 1.2 m sideways + 0.4 rad, e_xy=0.02": mk(dxy=tuple(1.2 * fwd), dth=0.4, e_xy=0.02, e_th=0.003),
        "B2 as B, ASSUMED person speed 0.4 m/s": mk(dxy=tuple(0.6 * fwd), dth=0.1, e_xy=0.02, e_th=0.003, v_p_max=0.4),
        "C2 as C, ASSUMED person speed 0.4 m/s": mk(dxy=tuple(1.2 * fwd), dth=0.4, e_xy=0.02, e_th=0.003, v_p_max=0.4),
        "D exact pose, e_xy=%.2f (> box)" % (e_cap + 0.15): mk(e_xy=e_cap + 0.15, e_th=0.03),
        "E exact pose, mu_lo=0.05 (outside every entry)": mk(e_xy=0.02, e_th=0.003, mu_lo=0.05),
        "F exact pose, mu_lo=0.30 (learned, tighter)": mk(e_xy=0.02, e_th=0.003, mu_lo=0.30),
        "G exact pose, person 0.6 m ahead": mk(e_xy=0.02, e_th=0.003, people=(p0[:2] + 0.6 * np.array([math.cos(p0[2]), math.sin(p0[2])]),)),
    }
    res = {}
    for name, b in cases.items():
        t1 = time.perf_counter()
        rc = lib.recover(b)
        res[name] = dict(mode=rc.mode, entry=None if rc.entry is None else rc.entry.id, n_direct=len(rc.matches), connector_s=round(rc.t_connect, 2),
                         legs=[l.kind for l in rc.legs], look=rc.look, reason=rc.reason, ms=round((time.perf_counter() - t1) * 1e3, 2))
    # lookup timing on the full library
    b = cases["A exact pose, e_xy=0.02"]
    lib.lookup_ms.clear()
    for _ in range(200):
        lib.lookup(b)
    st = lib.stats()
    out = dict(library=st, episode=ep_info, synthetic=synth, cases=res, episode_summary={k: summ.get(k) for k in ("task_done", "min_wall_clearance_m", "min_person_distance_m", "spilled")},
               e_xy_box_of_probe_entry=e_cap, total_wall_s=round(time.perf_counter() - t_all0, 1))
    with open(os.path.join(out_dir, "library_demo_seed%d.json" % seed), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="certified viability library: offline demo (CPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-s", type=float, default=6.0)
    ap.add_argument("--out", default="/data/local/ss5772/molmospaces/gatekeeper/library_demo")
    a = ap.parse_args()
    print(json.dumps(run_demo(a.out, a.seed, a.episode_s), indent=1, default=str))
