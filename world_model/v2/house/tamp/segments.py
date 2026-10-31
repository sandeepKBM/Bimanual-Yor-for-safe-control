"""tamp/segments.py -- timed, certified route SEGMENTS in the gatekeeper's Traj format ("one certificate for both layers").  SIMULATION ONLY.

Why: the legacy route stream certifies each sample pose FROM REST at one fixed speed, but the gatekeeper at execution certifies the MOVING state, so the plan
admitted routes the gatekeeper then refused.  Here the planner builds the SAME object the gatekeeper certifies:

  * a speed cap v_max from SPEEDS (ASSUMPTION: candidate speeds [m/s], fastest first; 0.3 = the legacy cruise, 0.12 = a slow, short-braking crawl);
  * a path-following twist profile along the polyline (pure pursuit + heading law, the stick law of teleop.operator._drive with the cup-safe accel limits of
    gatekeeper.traj: `cup_safe_target` for the steady twist, `_step` for the slew/cup-limited acceleration, `backup_from` for the stop);
  * the profile is cut into segments whose nominal duration is one of the gatekeeper's switch times T_S in gk.t_switch (1.0, 0.6, 0.3, 0.1 s), tried LONGEST first
    and the first one that passes check1 + check2 is taken -- exactly the gatekeeper's own commit rule (a long commitment is refused near a person by the
    passive-safety / reachable-set rows, a short one is admitted).  So segments are ~0.3 m of travel at 0.3 m/s in free space and shrink to ~3 cm near people.  Segment k = Traj(V, h, t_sw): the nominal part covers its stretch and starts at the PREDICTED
    moving state (pose, twist) at the end of segment k-1's nominal; then the cup-safe backup to rest.  The route starts and ends at rest (final segment's
    nominal ends at rest).  Every segment is checked with check1 + check2 against the belief anchored at its predicted start pose.
  * opt-in C2-ready ranking (GKConfig.c2_ready="rank"): among the passing candidates prefer one whose rest pose leaves a certified next progress move under both
    people hypotheses (gatekeeper/ready.py); ranking only, never admits/rejects.  Opt-in lateral offsets (GKConfig.seg_offsets): at a boundary also try tracking the
    route offset to the left/right with the holonomic vy (taper to 0 at the route end).  Both default off = bit-identical.
  * the execution layer (tamp/seg_gate.py) rebuilds each segment with the SAME function from the ACTUAL state and re-runs check1 + check2.

Units: m, s, rad; grid h, accel limits (A_HW, a_cup_max), v_cap, plan_frac are the gatekeeper's (read from the belief / GKConfig, never copied).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..gatekeeper.checks import check1, check2
from ..gatekeeper.ready import HYPS, ready_score, rest_poses
from ..gatekeeper.traj import Traj, _step, backup_from, cup_safe_target
from ..teleop import operator as OP
from . import plan_monitor as PMON

SPEEDS = (0.3, 0.2, 0.12)       # ASSUMPTION: candidate route speed caps [m/s], tried fastest first
BRAKE_DEC = 0.35                # m/s^2: speed target falls as sqrt(2 BRAKE_DEC * remaining) near the end of the driven part (the cup-safe _step still limits the real decel)
STOP_EPS = 0.02                 # m: the driven part is done when this close to its end (and at rest)
V_FLOOR = 0.03                  # m/s: minimum approach speed while the end is > STOP_EPS away (avoids an asymptotic crawl)
LEAD_S = 0.1                    # s: the executor rebuilds the next segment this long before the committed nominal ends
_OPP = OP.OperatorParams()      # operator heading law (kw, turn_slow, vx_min_turn)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Path:
    """xy polyline with arclength; projection restricted to a window around a hint (routes may double back)."""

    def __init__(self, pts):
        P = [np.asarray(pts[0], float)]
        for p in pts[1:]:
            p = np.asarray(p, float)
            if float(np.hypot(*(p - P[-1]))) > 1e-9:
                P.append(p)
        self.P = np.array(P)
        self.seg = np.diff(self.P, axis=0) if len(P) > 1 else np.zeros((0, 2))
        self.L = np.hypot(self.seg[:, 0], self.seg[:, 1]) if len(P) > 1 else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(self.L)])
        self.total = float(self.cum[-1])

    def point(self, s):
        s = min(max(float(s), 0.0), self.total)
        if self.total < 1e-9:
            return self.P[0].copy()
        i = min(int(np.searchsorted(self.cum, s, side="right") - 1), len(self.L) - 1)
        return self.P[i] + self.seg[i] * ((s - self.cum[i]) / self.L[i])

    def heading(self, s):
        if self.total < 1e-9:
            return 0.0
        s = min(max(float(s), 0.0), self.total)
        i = min(int(np.searchsorted(self.cum, s, side="right") - 1), len(self.L) - 1)
        return math.atan2(self.seg[i, 1], self.seg[i, 0])

    def project(self, xy, s_hint, back=0.3, fwd=1.5):
        """arclength of the closest point on the polyline within [s_hint - back, s_hint + fwd]."""
        if self.total < 1e-9:
            return 0.0
        lo, hi = s_hint - back, s_hint + fwd
        best, bs = math.inf, max(s_hint, 0.0)
        xy = np.asarray(xy, float)
        for i in range(len(self.L)):
            if self.cum[i + 1] < lo or self.cum[i] > hi:
                continue
            d = self.seg[i]
            f = min(1.0, max(0.0, float((xy - self.P[i]) @ d) / float(d @ d)))
            lat = float(np.hypot(*(xy - (self.P[i] + f * d))))
            if lat < best - 1e-12:
                best, bs = lat, float(self.cum[i] + f * self.L[i])
        return bs


def _integrate(pose, v0, v1, h):
    """one midpoint-heading step (== Traj.poses / base.advance)."""
    vm = (0.5 * (v0[0] + v1[0]), 0.5 * (v0[1] + v1[1]), 0.5 * (v0[2] + v1[2]))
    ym = pose[2] + 0.5 * vm[2] * h
    c, s = math.cos(ym), math.sin(ym)
    return (pose[0] + (c * vm[0] - s * vm[1]) * h, pose[1] + (s * vm[0] + c * vm[1]) * h, pose[2] + vm[2] * h)


def make_segment(path: Path, reverse: bool, v_max: float, pose, v0, s0: float, bel, gk, seg_t: float, s_stop: float, max_steps: int = 4000,
                 offset: float = 0.0, gk_lat=None):
    """Build ONE segment from the state (pose, v0 body twist, arclength s0).  -> dict(V, t_sw, end_pose, end_v, s_end, final) or None (no convergence).
    Nominal: pure pursuit toward the polyline (lookahead 2 v_max in [0.25, 0.6] m), forward speed v_max scaled down by the heading error and by the braking
    envelope sqrt(2 BRAKE_DEC (s_stop - s)); each step is cup/slew limited toward the cup-safe version of that twist.  It ends after `seg_t` seconds
    (moving state; then the cup-safe backup is appended) or earlier if the base comes to rest at the end of the driven part (final).
    offset != 0 (opt-in, holonomic base): track the route laterally offset by delta(s) = offset * clip((s_stop - s)/gk.seg_offset_taper, 0, 1) (left normal +) with
    the body vy (gain gk.seg_k_lat, cap gk.seg_v_lat_max), heading still along the path; s stays the projection on the ORIGINAL path.  offset == 0 is exactly the
    original code path.  gk_lat: optional config overriding gk for the lateral gains only."""
    h = gk.h
    gl = gk if gk_lat is None else gk_lat
    offset = float(offset)
    L = gk.plan_frac * bel.a_cup_max
    r = (float(bel.r_cup[0]), float(bel.r_cup[1]))
    a_hw = [float(x) for x in bel.a_hw]
    v_cap = np.asarray(bel.v_cap, float)
    la = min(0.6, max(0.25, 2.0 * v_max))
    pose = (float(pose[0]), float(pose[1]), float(pose[2]))
    v = [float(x) for x in v0]
    s = float(s0)
    n_max = int(round(seg_t / h))
    nom = [list(v)]
    final = False
    for _ in range(max_steps):
        rem = s_stop - s
        at_rest = max(abs(x) for x in v) <= 1e-9
        if rem <= STOP_EPS and at_rest and len(nom) > 1:
            final = True
            break
        if len(nom) - 1 >= n_max:
            break
        aim = path.point(min(s + la, s_stop))
        if offset != 0.0:
            sa = min(s + la, s_stop)
            ha = path.heading(sa)
            da = offset * min(1.0, max(0.0, (s_stop - sa) / gl.seg_offset_taper))
            aim = aim + da * np.array([-math.sin(ha), math.cos(ha)])
        dx, dy = aim[0] - pose[0], aim[1] - pose[1]
        dist = math.hypot(dx, dy)
        if dist < 0.05 or rem < 0.15:
            e = 0.0                                                     # very close to the end of the driven part: don't spin
        else:
            e = wrap(math.atan2(dy, dx) + (math.pi if reverse else 0.0) - pose[2])
        stopping = rem <= STOP_EPS
        if stopping:
            vt = 0.0
        else:
            scale = max(_OPP.vx_min_turn, 1.0 - abs(e) / _OPP.turn_slow)
            vt = min(v_max * scale, max(V_FLOOR, math.sqrt(2.0 * BRAKE_DEC * max(rem, 0.0))))
        wz = float(np.clip(_OPP.kw * e, -v_cap[2], v_cap[2]))
        if offset == 0.0:
            u = np.array([-vt if reverse else vt, 0.0, 0.0 if stopping else wz])
        else:
            hs = path.heading(s)
            nx, ny = -math.sin(hs), math.cos(hs)
            ps = path.point(s)
            e_lat = (pose[0] - ps[0]) * nx + (pose[1] - ps[1]) * ny
            d_s = offset * min(1.0, max(0.0, (s_stop - s) / gl.seg_offset_taper))
            vy_w = 0.0 if stopping else float(np.clip(gl.seg_k_lat * (d_s - e_lat), -gl.seg_v_lat_max, gl.seg_v_lat_max))
            c_, s_ = math.cos(pose[2]), math.sin(pose[2])
            bx, by = vy_w * (nx * c_ + ny * s_), vy_w * (-nx * s_ + ny * c_)
            u = np.array([(-vt if reverse else vt) + bx, by, 0.0 if stopping else wz])
        tgt = cup_safe_target(u, r, gk.steady_frac * L, v_cap)
        v1 = _step(v, tgt, h, L, r, a_hw)
        if stopping and max(abs(x) for x in v1) <= 1e-3:
            v1 = [0.0, 0.0, 0.0]                                        # snap the asymptotic tail (accel jump <= 1e-3/h), as plan_candidate does
        pose = _integrate(pose, v, v1, h)
        v = list(v1)
        nom.append(list(v))
        s = path.project(pose[:2], s)
    else:
        return None
    n = len(nom) - 1
    if n < 1:
        return None
    bk = backup_from(np.array(v), bel, gk)
    V = np.concatenate([np.array(nom), bk.V[1:]]) if len(bk.V) > 1 else np.array(nom)
    return dict(V=V, t_sw=n * h, end_pose=pose, end_v=np.array(v), s_end=float(s), final=bool(final), n_nom=n, offset=offset)


def start_pose(path: Path, reverse: bool):
    p = path.P[0]
    return (float(p[0]), float(p[1]), wrap(path.heading(0.0) + (math.pi if reverse else 0.0)))


def check_segment(tr: Traj, bel, gk):
    """check1 + check2 -> (ok, margin, binding_row, reason).  margin/row over the informative rows (rows tight by construction excluded)."""
    o1, m1, r1, g1 = check1(tr, bel, gk)
    if not o1:
        return False, m1, r1, f"c1 {r1} {m1:.3f}"
    o2, m2, r2, g2 = check2(tr, bel, gk)
    if not o2:
        return False, m2, r2, f"c2 {r2} {m2:.3f}"
    rows = {}
    for mg in (g1, g2):
        for k, x in mg.items():
            if k not in ("rest_speed", "hw_slew"):
                rows[k] = min(rows.get(k, math.inf), float(x))
    if not rows:
        return True, 0.0, "none", ""
    k = min(rows, key=rows.get)
    return True, rows[k], k, ""


def _ready_moves(path: Path, reverse: bool, s_hint: float, s_stop: float, gk, offsets=(0.0,)):
    """-> moves_fn(b_r): from the rest belief b_r (rest pose, re-observed people) the first library move (offset, speed) that passes check1 + check2 and makes
    route progress >= gk.c2_ready_dmin (or reaches the end), as a description string; "at_goal" if the route end is already reached; else None.
    Depth 1: next_certified_segment with ready=False and gk.seg_offsets=() (no ranking, no offset expansion inside)."""
    gk0 = dataclasses.replace(gk, seg_offsets=())

    def moves_fn(b_r):
        pose = tuple(float(x) for x in b_r.pose_wall)
        s_r = path.project(pose[:2], s_hint)
        if s_stop - s_r <= STOP_EPS + 0.02:
            return "at_goal"
        for o in offsets:
            for v in gk.c2_ready_speeds:
                tr2, res2, _m, _row, T2, _why = next_certified_segment(path, reverse, v, pose, np.zeros(3), s_r, b_r, gk0, s_stop, offset=o, ready=False)
                if tr2 is not None and (res2["final"] or res2["s_end"] - s_r >= gk.c2_ready_dmin):
                    return f"route v={v} off={o} T_S={T2}"
        return None
    return moves_fn


def _site_on(gk, site: str) -> bool:
    """the C2-ready ranking runs at this call site?  scope "both" (default, today): everywhere; scope "gate": only at the executor's segment boundaries (site "gate")."""
    return gk.c2_ready_scope == "both" or site == "gate"


def tube_off(tr: Traj, bel, gk) -> bool:
    """tube term: True iff the candidate's REST pose keeps the footprint >= d_person + gk.c2_ready_tube_clear away from every person's TRACKED tube
    disc(c_i + v_i t, r0_i + e_v_i t)  (r0_i = PERSON_CLEAR_R + e_p_i, the plan monitor's geometry) for t in [t_rest, t_rest + gk.c2_ready_tube_h]; t is measured from the belief
    time (= the candidate's start).  No rest instant -> False.  No people -> True.  The rest pose is the people-frame one (the frame bel.people is in).  Liveness heuristic over tracked
    (not worst-case) sets: it only re-orders candidates that already passed check1 + check2."""
    if len(bel.people) == 0:
        return True
    rp = rest_poses(tr, bel)
    if rp is None:
        return False
    _pw, pp, t_rest = rp
    return PMON.rest_tube_conflict_time(bel, pp, t_rest, gk.c2_ready_tube_h, 0.25, t_f=0.0, clear=gk.c2_ready_tube_clear) == math.inf


def next_certified_segment(path: Path, reverse: bool, v_max: float, pose, v0, s0: float, bel, gk, s_stop: float, offset: float = 0.0, ready=None, site: str = "plan"):
    """the ONE function used by the planner and by the executor: from the state (pose, v0, s0) try T_S in gk.t_switch (longest first), build the segment
    (make_segment) and check1 + check2 it under `bel`; the first that passes wins.  -> (Traj, res, margin, row, T_S, "") or (None, None, margin, row, None, why).
    Opt-in (all default off => exactly the function above):
      * gk.seg_offsets: lateral offsets tried (outer loop) besides the current `offset`: [offset, 0, *seg_offsets] (current first);
      * ready / gk.c2_ready == "rank": among the passing candidates prefer one whose rest pose leaves a certified next move under both people hypotheses
        (gatekeeper.ready); the first fully ready candidate wins, else the best by (score desc, try order).  Ranking never admits or rejects.
      * site: "plan" (planner route streams, plan monitor's predicted old route, readiness sub-calls; default) | "gate" (the executor's segment boundary).  With
        gk.c2_ready_scope == "gate" the c2_ready ranking and the tube term below run only at site "gate"; with "both" (default) the ranking runs at every site (today).
      * gk.c2_ready_tube_h > 0 (site "gate" only): TUBE TERM.  Key = (rest pose off every tracked tube [tube_off], ready score [only if ranking is on], try order): the first
        off-tube candidate in try order wins (with ranking on: the first fully ready one among the off-tube candidates, else the best score).  If no passing candidate is
        off-tube the choice is exactly today's (first passing / today's ranking key).  Never admits or rejects.
    res["offset"], res["ready"] (None when off), res["ready_n"], res["ready_detail"], and with the tube term res["tube_off"], res["tube_changed"] are set on the returned res."""
    offs = [float(offset)]
    if gk.seg_offsets:
        offs += [o for o in (0.0,) + tuple(float(x) for x in gk.seg_offsets) if o not in offs]
    site_on = _site_on(gk, site)
    rank_on = (gk.c2_ready == "rank" and site_on) if ready is None else bool(ready)
    tube_on = gk.c2_ready_tube_h > 0.0 and site == "gate"
    whys, last = [], (None, -math.inf, "none")
    cands = []
    first_pass = None                                       # tube term: the first passing candidate in try order (= what today's rank-off loop returns)
    pend = []                                               # tube term + ranking: passing candidates that are ON a tube, readiness not yet computed
    for o in offs:
        for T in gk.t_switch:
            tag = f"off={o:g}: " if len(offs) > 1 else ""
            res = make_segment(path, reverse, v_max, pose, v0, s0, bel, gk, T, s_stop, offset=o)
            if res is None:
                whys.append(f"{tag}T_S={T}: no profile")
                continue
            tr = Traj(res["V"], gk.h, 0.0, res["t_sw"], res["t_sw"])
            ok, m, row, why = check_segment(tr, bel, gk)
            if not ok:
                whys.append(f"{tag}T_S={T}: {why}")
                continue
            if tube_on:
                off_t = tube_off(tr, bel, gk)
                res["tube_off"] = off_t
                if first_pass is None:
                    first_pass = (tr, res, m, row, T)
                if not off_t:
                    if rank_on:
                        pend.append((tr, res, m, row, T, o))
                    continue
                res["tube_changed"] = first_pass[1] is not res
            if not rank_on:
                res["ready"] = None
                return tr, res, m, row, T, ""
            _score_candidate(res, tr, bel, gk, path, reverse, s_stop, o)
            cands.append((tr, res, m, row, T))
            if res["ready"] == res["ready_n"]:
                return tr, res, m, row, T, ""
    if cands:
        best = max(range(len(cands)), key=lambda i: (cands[i][1]["ready"], -i))
        tr, res, m, row, T = cands[best]
        return tr, res, m, row, T, ""
    if tube_on and first_pass is not None:                  # no off-tube candidate: today's key exactly
        if not rank_on:
            tr, res, m, row, T = first_pass
            res["ready"], res["tube_changed"] = None, False
            return tr, res, m, row, T, ""
        cands = []
        for (tr, res, m, row, T, o) in pend:
            _score_candidate(res, tr, bel, gk, path, reverse, s_stop, o)
            res["tube_changed"] = False
            cands.append((tr, res, m, row, T))
            if res["ready"] == res["ready_n"]:
                return tr, res, m, row, T, ""
        best = max(range(len(cands)), key=lambda i: (cands[i][1]["ready"], -i))
        tr, res, m, row, T = cands[best]
        return tr, res, m, row, T, ""
    return None, None, last[1], last[2], None, "; ".join(whys)


def _score_candidate(res, tr, bel, gk, path, reverse, s_stop, o):
    """C2-ready readiness of one passing candidate -> sets res["ready"], res["ready_n"], res["ready_detail"]."""
    if res["final"]:
        score, n, detail = len(HYPS), len(HYPS), ["final"]
    else:
        mf = _ready_moves(path, reverse, res["s_end"], s_stop, gk, offsets=(o,) if o == 0 else (o, 0.0))
        score, n, detail = ready_score(tr, bel, gk, mf)
    res["ready"], res["ready_n"], res["ready_detail"] = score, n, detail


def plan_route_segments(path: Path, reverse: bool, v_max: float, gk, bel_fn, s_stop: float | None = None, start=None):
    """plan + certify the segments of a route at speed cap v_max from rest at the route start (T_S per segment chosen as the gatekeeper would).  bel_fn(pose, s_start, t_start) -> Belief anchored at that predicted pose (t_start = segment start time from the route start [s]; the caller may predict people to it)
    (the caller decides people / payload).  -> (list[RouteSegment], "") or (None, reason).
    start: optional (pose, body twist v0, s0, t0) -- plan from this (possibly MOVING) state instead of rest at the route start (the plan-prediction monitor's replan starts from the
    state the committed trajectory predicts at t_flag + L); t0 = time since the route start of that state (people prediction clock).  None = exactly the original code path."""
    from .types import RouteSegment
    s_stop = path.total if s_stop is None else s_stop
    if start is None:
        pose, v, s, t, off = start_pose(path, reverse), np.zeros(3), 0.0, 0.0, 0.0
    else:
        pose, v, s, t, off = tuple(float(x) for x in start[0]), np.asarray(start[1], float).copy(), float(start[2]), float(start[3]), 0.0
    segs = []
    for _ in range(3000):
        bel = bel_fn(pose, s, t)
        tr, res, m, row, T, why = next_certified_segment(path, reverse, v_max, pose, v, s, bel, gk, s_stop, offset=off)
        if tr is None:
            return None, f"segment {len(segs)} s={s:.2f} m pose=({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f}deg) v={v_max}: {why}"
        segs.append(RouteSegment(res["V"], gk.h, res["t_sw"], T, tuple(pose), float(s), res["s_end"], float(t), float(m), row,
                                float(res["offset"]), -1 if res.get("ready") is None else int(res["ready"])))
        t += res["t_sw"]
        pose, v, s, off = res["end_pose"], res["end_v"], res["s_end"], res["offset"]
        if res["final"]:
            return segs, ""
    return None, "segment count limit"
