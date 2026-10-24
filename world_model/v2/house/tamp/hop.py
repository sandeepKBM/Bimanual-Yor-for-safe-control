"""tamp/hop.py -- the HOP primitive (GAP-4 / T4(c) hop lemma): a very short, straight, pure-translation, rest-to-rest base move certified by the gatekeeper's
UNCHANGED check1 + check2.  SIMULATION ONLY.  Opt-in (run_tamp --hop ...); nothing here is imported by the default code path.

Why: with a person just outside the frozen radius every library move (T_S >= 0.1 s + a cup-safe backup) moves longer than the admissible hop, so the ladder waits/retreats.
The hop lemma (docs/paper/theorems_T3_T4_draft_2026-10-16.md, T4(c)): a rest-to-rest bang-bang translation of duration t passes C1-person iff
        rho_i0 + g_i(t) + sw(t) + a_r t^2 / 4  <=  d_i     for every person i  (worst case: straight at the person),
so for d_i > rho_i0 + sw_0 some t > 0 is admissible; hop -> re-observe at rest (the person sets reset) -> hop repeats.

The hop is the SAME object the gatekeeper certifies (gatekeeper.traj.Traj, body-twist samples on the grid h, piecewise-linear in time):
  V_k = a_r h min(k, n-k) * (db_x, db_y, 0)          k = 0..n,   n = t_h / h,   db = R(-yaw) d   (body-frame unit direction of the WORLD direction d; yaw constant, wz = 0)
  nominal  = the acceleration half   (t_sw = floor(n/2) h; ends at the peak velocity),
  backup   = the deceleration half   (cup-safe brake from the peak at a_r <= plan_frac * a_cup_max == what traj.backup_from does from that state; ends EXACTLY at rest).
check1/check2 (gatekeeper/checks.py) read only V, h and the belief, never t_sw: the split is labelling for the committed-trajectory machinery (predict_state, _chain), so the
checks apply unchanged to the whole rest-to-rest profile.  Odd n has a zero-acceleration middle segment (displacement a_r h^2 m (m+1) = a_r t^2/4 - a_r h^2/4, m = n // 2).

a_r (m/s^2) for body direction db and duration t_h:
    min( plan_frac * bel.a_cup_max            (cup row bound the gatekeeper's planner uses: pure translation => |a_cup| = a_r; with empty hands and cup_row="attached"
                                               a_cup_max = FREE_CUP_MARGIN-large, so only the hardware limit binds),
         A_HW_i / |db_i|                       (hardware slew box, per body axis),
         2 V_CAP_i / (|db_i| t_h) )            (peak speed a_r t_h / 2 inside the speed cap).
"""
from __future__ import annotations

import math

import numpy as np

from ..gatekeeper.checks import _ev, _pe, check1, check2
from ..gatekeeper.traj import Traj
from ..teleop import geom as GM
from . import plan_monitor as PMON
from . import segments as SEG

HOP_DURATIONS = (0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1)      # s, tried longest first (each rounded DOWN to a multiple of gk.h)
HOP_MIN_M = 0.02                                                       # m, a hop that moves less is not a hop
HOP_DIRS = ("prog", "clear", "back")                                   # priority order
COMPASS = tuple((math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8))
TUBE_HORIZON = 4.0                                                     # s, "clear" hops stop once the footprint is out of every tracked tube for this long


def hop_a_r(bel, gk, db, t_h):
    """acceleration magnitude of the hop along body direction db (unit 2-vector) of duration t_h; see the module docstring."""
    lim = gk.plan_frac * bel.a_cup_max
    for i in (0, 1):
        a = abs(float(db[i]))
        if a > 1e-12:
            lim = min(lim, float(bel.a_hw[i]) / a, 2.0 * float(bel.v_cap[i]) / (a * t_h) if t_h > 0 else math.inf)
    return float(lim)


def body_dir(d_world, yaw):
    d = np.asarray(d_world, float)
    d = d / max(float(np.hypot(*d)), 1e-12)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def hop_traj(d_world, yaw, t_h, a_r, h, t0=0.0):
    """rest-to-rest bang-bang translation (see the module docstring).  t_h is rounded DOWN to a multiple of h."""
    n = int(math.floor(t_h / h + 1e-9))
    assert n >= 2, "a hop needs at least two grid steps"
    db = body_dir(d_world, yaw)
    sp = a_r * h * np.minimum(np.arange(n + 1), n - np.arange(n + 1))
    V = np.stack([sp * db[0], sp * db[1], np.zeros(n + 1)], 1)
    ts = (n // 2) * h
    return Traj(V, h, t0, ts, ts)


def displacement(tr: Traj) -> float:
    P = tr.poses()
    return float(np.hypot(P[-1, 0], P[-1, 1]))


def certify(tr: Traj, bel, gk):
    """check1 + check2 (unchanged) -> (ok, info) with info = dict(why, c1=margins, c2=margins, margin=min over the informative rows)."""
    o1, m1, r1, g1 = check1(tr, bel, gk)
    if not o1:
        return False, dict(why=f"c1 {r1} {m1:.3f}", c1=g1, c2=None, margin=float(m1))
    o2, m2, r2, g2 = check2(tr, bel, gk)
    if not o2:
        return False, dict(why=f"c2 {r2} {m2:.3f}", c1=g1, c2=g2, margin=float(m2))
    rows = [float(x) for g in (g1, g2) for k, x in g.items() if k not in ("rest_speed", "hw_slew", "speed_cap")]
    return True, dict(why="", c1=g1, c2=g2, margin=min(rows) if rows else 0.0)


def durations_grid(durations, h):
    """descending, each rounded down to a multiple of h, duplicates and n < 2 removed -> list of step counts n."""
    out = []
    for t in sorted((float(x) for x in durations), reverse=True):
        n = int(math.floor(t / h + 1e-9))
        if n >= 2 and n not in out:
            out.append(n)
    return out


def longest_admissible(bel, gk, d_world, durations=HOP_DURATIONS, min_m=HOP_MIN_M, t0=0.0):
    """the LONGEST hop along the world direction d_world that passes check1 + check2 under `bel` and moves >= min_m.  -> dict(t_h, a_r, traj, disp, info) or None."""
    yaw = float(np.asarray(bel.pose_wall, float)[2])
    db = body_dir(d_world, yaw)
    for n in durations_grid(durations, gk.h):
        t_h = n * gk.h
        a_r = hop_a_r(bel, gk, db, t_h)
        tr = hop_traj(d_world, yaw, t_h, a_r, gk.h, t0)
        disp = displacement(tr)
        if disp < min_m:
            continue
        ok, info = certify(tr, bel, gk)
        if ok:
            return dict(t_h=t_h, a_r=a_r, traj=tr, disp=disp, info=info)
    return None


# ---------------------------------------------------------------------------------------------------------------- directions
def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return None if n < 1e-9 else np.array([v[0] / n, v[1] / n])


def dir_prog(path, pose, max_lat=0.6):
    """travel direction of the immediate route at the robot's projection (segments.Path), None if the robot is farther than max_lat from it or at its end."""
    if path is None or path.total < 1e-9:
        return None
    pose = np.asarray(pose, float)
    s = path.project(pose[:2], 0.5 * path.total, back=path.total, fwd=path.total)
    if float(np.hypot(*(pose[:2] - path.point(s)))) > max_lat or s >= path.total - 0.02:
        return None
    hd = path.heading(s)
    return np.array([math.cos(hd), math.sin(hd)])


def _vels(bel):
    n = len(bel.people)
    return [np.asarray(v, float)[:2] for v in bel.people_vel] if len(bel.people_vel) == n else [np.zeros(2) for _ in range(n)]


def tube_clearance(bel, pose, i, horizon=TUBE_HORIZON, dt=0.25):
    """min over t in [0, horizon] of dist(footprint at `pose`, c_i + v_hat_i t)."""
    c, v = np.asarray(bel.people[i], float)[:2], _vels(bel)[i]
    return min(GM.rect_dist_point(c + v * t, pose, bel.fp)[0] for t in np.arange(0.0, horizon + 1e-9, dt))


def nearest_person(bel, pose):
    if not len(bel.people):
        return None
    d = [GM.rect_dist_point(np.asarray(c, float)[:2], pose, bel.fp)[0] for c in bel.people]
    return int(np.argmin(d))


def dir_clear(bel, pose, probe=0.1, v_still=0.05):
    """away from the tracked tube of the nearest person: the candidate (both perpendiculars of v_hat; straight away if v_hat ~ 0) whose `probe`-step most increases the
    footprint-to-tube clearance over the next 4 s; None if none increases it."""
    i = nearest_person(bel, pose)
    if i is None:
        return None
    pose = np.asarray(pose, float)
    c, v = np.asarray(bel.people[i], float)[:2], _vels(bel)[i]
    if float(np.hypot(*v)) < v_still:
        cands = [_unit(pose[:2] - c)]
    else:
        u = _unit(v)
        cands = [np.array([-u[1], u[0]]), np.array([u[1], -u[0]])]
    cands = [x for x in cands if x is not None]
    base = tube_clearance(bel, pose, i)
    best, bd = 1e-6, None
    for d in cands:
        g = tube_clearance(bel, (pose[0] + probe * d[0], pose[1] + probe * d[1], pose[2]), i) - base
        if g > best:
            best, bd = g, d
    return bd


def dir_back(pose, target, min_m=0.02):
    """toward the last rest pose (the retreat direction)."""
    if target is None:
        return None
    d = np.asarray(target[:2], float) - np.asarray(pose[:2], float)
    return _unit(d) if float(np.hypot(*d)) >= min_m else None


def out_of_tubes(bel, pose, t_now=0.0, horizon=TUBE_HORIZON, dt=0.25):
    """True iff the (resting) footprint stays clear of every tracked tube for `horizon` s (plan_monitor.conflict_time, the c2-ready/plan-monitor tube)."""
    if len(bel.people) == 0:
        return True
    return PMON.rest_tube_conflict_time(bel, pose, t_now, horizon, dt) == math.inf


def candidate_dirs(kinds, bel, pose, route_path=None, back_target=None, compass=False):
    """[(kind, unit world direction)] in priority order for the requested kinds (kinds not available at this state are skipped)."""
    out = []
    for k in kinds:
        if k == "prog":
            d = dir_prog(route_path, pose)
        elif k == "clear":
            d = dir_clear(bel, pose)
        elif k == "back":
            d = dir_back(pose, back_target)
        else:
            raise ValueError(f"unknown hop direction kind {k!r}")
        if d is not None:
            out.append((k, d))
    if compass:
        out += [(f"compass{k * 45}", np.array(c)) for k, c in enumerate(COMPASS)]
    return out


def choose_hop(bel, gk, kinds=HOP_DIRS, durations=HOP_DURATIONS, min_m=HOP_MIN_M, route_path=None, back_target=None, compass=False, t0=0.0):
    """first direction (priority order) with an admissible hop, the LONGEST such hop.  -> dict(kind, d, t_h, a_r, traj, disp, info, tried) or None.
    tried = [(kind, "ok" | "none")] for the directions examined."""
    pose = np.asarray(bel.pose_wall, float)
    tried = []
    for kind, d in candidate_dirs(kinds, bel, pose, route_path, back_target, compass):
        r = longest_admissible(bel, gk, d, durations, min_m, t0)
        tried.append((kind, "ok" if r else "none"))
        if r:
            return dict(r, kind=kind, d=d, tried=tried)
    return None


# ---------------------------------------------------------------------------------------------------------------- E3: the lemma
def lemma_tbar(bel, gk, d, model="max", t_max=3.0, sweep=True, d_person=True):
    """largest t (continuous) with  rho_i0 [+ d_person] + g(t) [+ sw(t)] + a_r(t) t^2/4 <= d  for a STATIC person straight ahead at footprint distance d (accel model, v_hat = 0).
      rho_i0 = PERSON_CLEAR_R + e_p;  g_speed = v_p (t + lat);  g_accel = e_v (t + lat) + a_p (t + lat)^2 / 2;
      sw(t) = 0.5 h (v_peak + rate)   (the checker's inter-sample sweep allowance at the PEAK speed v_peak = a_r t / 2; rate = v_p, or e_v + a_p (t + lat + h));
      a_r(t) = hop_a_r along the body-x axis.
    model "max": the robot is safe if clear of EITHER disc (check1 takes the max margin) -> max(t_speed, t_accel).  Returns 0.0 if even t -> 0 fails (frozen).
    sweep/d_person False -> the lemma as WRITTEN in the draft (no sweep, no d_person)."""
    h, lat = gk.h, bel.latency_s
    r0 = GM.PERSON_CLEAR_R + bel.e_p
    dp = bel.d_person if d_person else 0.0
    db = np.array([1.0, 0.0])

    def lhs(t, m):
        a = hop_a_r(bel, gk, db, t)
        vpk = a * t / 2.0
        if m == "speed":
            g, rate = bel.v_p_max * (t + lat), bel.v_p_max
        else:
            g, rate = bel.e_v * (t + lat) + 0.5 * bel.a_p * (t + lat) ** 2, bel.e_v + bel.a_p * (t + lat + h)
        sw = 0.5 * h * (vpk + rate) if sweep else 0.0
        return r0 + dp + g + sw + a * t * t / 4.0

    models = ("speed", "accel") if model == "max" else (model,)
    best = 0.0
    for m in models:
        ts = np.linspace(0.0, t_max, 3001)
        ok = [lhs(t, m) <= d + 1e-12 for t in ts]
        # lhs is increasing in t -> the feasible set is [0, t*]
        last = 0.0
        for t, o in zip(ts, ok):
            if not o:
                break
            last = t
        best = max(best, last if ok[0] else 0.0)
    return float(best)
