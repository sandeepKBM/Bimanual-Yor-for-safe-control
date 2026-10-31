"""tamp/streams.py -- PDDLStream streams backed by OUR checkers (layer 3).  SIMULATION ONLY.

A stream never yields an output its checker rejects, so every fact the planner sees is certified over the set-valued belief:
  s-dock        C2 static: the robot at REST at the dock conf passes check1 + check2 (static clearance with the dock buffer) under the pose box.
  s-grasp       C3 (checkers/grasp3.check_grasp): the grasp's worst margin over ALL 128 (mass, CoM, a) corners at mu_lo is >= 0; filtered to the grasps the
                executor has a skill for (world.SKILLS).
  s-place       support test: the item footprint (placed at the dock's hold pose) lies inside the surface top; margin = smallest inset [m].
  t-reach       kinematics: from dock conf q, placement p equals the item's hold pose within the align-jog range (REACH_TOL).
  s-route-free  C1 + C2 static route certificate (T3 H0): at every sample pose along the route (ds), from REST, some forward candidate
  s-route-hold  (u = cruise twist along the route held T_S in gk.t_switch, then the cup-safe backup) is admitted with NO people.  -hold uses the payload belief
                (cup rows, payload mu_lo); -free disables the cup row (hands empty).
Every call is logged in Streams.log (stream, inputs, outcome, reason, margin, ms) -- the per-checker admission record of a plan.
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections import Counter

import numpy as np

from ..checkers import grasp3 as G3
from ..gatekeeper.checks import check1, check2
from ..gatekeeper.traj import Traj, plan_candidate
from . import plan_cert as PC
from . import segments as SEG
from .people_pred import predict_people
from .types import Cert, Grasp, Placement, Route
from .world import SKILLS, World, compose, rel_pose, runtime_belief_at, wrap

REACH_TOL = (0.05, 0.06)            # (m, rad) ASSUMPTION: what the ALIGN tray-frame jog absorbs (operator.py CREEP note: "a few cm / ~1 deg" -> 5 cm, 3.4 deg)
ROUTE_MATCH_R = 0.6                 # m: a scene route serves (q1, q2) if its endpoints are within this of the confs
LATERAL_OFFSETS = (0.15, -0.15, 0.30, -0.30)   # m: interior-vertex shifts (+ = left of the travel direction) offered as extra route candidates
SAME_PLACE = (0.05, 0.05)           # (m, rad): two confs at the same place -> trivial in-place route


TIGHT_BY_CONSTRUCTION = ("rest_speed", "hw_slew")       # rows that are 0 by construction (backup ends at rest; free-hand plan uses the full slew box)


def _short(x):
    return repr(x) if hasattr(x, "__dataclass_fields__") else str(x)


def _binding(*mgs):
    """(min margin, row name) over the informative rows of check1/check2 margin dicts (rows tight by construction excluded)."""
    rows = {}
    for mg in mgs:
        for k, v in mg.items():
            if k not in TIGHT_BY_CONSTRUCTION:
                rows[k] = min(rows.get(k, math.inf), float(v))
    if not rows:
        return 0.0, "none"
    k = min(rows, key=rows.get)
    return rows[k], k


class Streams:
    def __init__(self, world: World, ds: float = 0.2, v_cruise: float = 0.3, end_skip: float = 0.3, near_m: float = 2.0,
                 lateral_offsets: tuple = LATERAL_OFFSETS, cert_mode: str = "rest", speeds: tuple = SEG.SPEEDS,
                 people_pred: str = "frozen", t_offset: float = 0.0, seg_cache: dict | None = None, people_ev_scale: float = 1.0,
                 start_v=None, plan_cert: str | None = None):
        """plan_cert: "off" (default, unchanged) | "pred": route certification (cert_mode "segments") by the plan-time certifier tamp/plan_cert.py -- cup + hw + walls + ALL people over
        the WHOLE route (no near_m cut-off, no people-free tail), people predicted by constant velocity from the plan-time observation (person_model "pred": prediction disc only +
        separating-motion exit rule), each segment at its planned start time (t_offset + segment start).  None -> world.gk.plan_cert.  The planner ALWAYS checks walls and
        people: if the runtime gatekeeper config has runtime_rows="cup", the planner still certifies with runtime_rows="all".
        ds: route sample spacing [m]; v_cruise: candidate speed along the route [m/s] (ASSUMPTION, nominal phase2 route speed 0.3); end_skip: the last
        end_skip m before the goal conf are not required to admit a FORWARD candidate (the dock surface is ahead; the final approach is the docking creep).
        cert_mode: "rest" (default, legacy: each sample certified FROM REST at v_cruise) | "segments" (timed gatekeeper-format segments at a planner-chosen speed from
        `speeds`, fastest first; segment durations T_S in gk.t_switch, longest that passes; tamp/segments.py -- the executor re-certifies the same segments from the actual state).
        people_pred: "frozen" (default, bit-identical) = people at their plan-time position, applied to segments with s_start <= near_m only | "time" = the IMMEDIATE route
        (q1 == the world's current conf) is checked, EVERY segment, against people predicted to c + v_hat t_start (people_pred.py; constant velocity, ASSUMPTION); t_offset =
        extra start delay [s] (timed-wait probe: the robot holds at rest t_offset, then starts); later routes keep the people-free H0 rule.
        seg_cache: optional shared cache dict (valid only while the world is unchanged; the timed-wait probes share it).
        start_v: optional body twist (3,) of the robot at the world's q0 (plan-prediction replan from a MOVING predicted state); the immediate route's segments are then planned
        from (q0 pose, start_v) instead of rest.  None / people_pred "frozen" = unchanged."""
        assert cert_mode in ("rest", "segments"), cert_mode
        assert people_pred in ("frozen", "time"), people_pred
        self.cert_mode, self.speeds = cert_mode, tuple(float(x) for x in speeds)
        self.people_pred, self.t_offset, self.people_ev_scale = people_pred, float(t_offset), float(people_ev_scale)
        self._seg_cache = {} if seg_cache is None else seg_cache
        self.start_v = None if start_v is None else tuple(float(x) for x in start_v)
        self.w = world
        self.plan_cert = getattr(world.gk, "plan_cert", "off") if plan_cert is None else plan_cert
        assert self.plan_cert in ("off", "pred"), self.plan_cert
        assert self.plan_cert == "off" or cert_mode == "segments", "plan_cert='pred' needs cert_mode='segments'"
        # the planner's checkers: the world's config, but never in the runtime "cup" mode (planning must see walls and people).  Identity (bit-identical) when runtime_rows == "all".
        self._gk = world.gk if getattr(world.gk, "runtime_rows", "all") == "all" else dataclasses.replace(world.gk, runtime_rows="all")
        self.ds, self.v, self.end_skip = float(ds), float(v_cruise), float(end_skip)
        self.near_m = float(near_m)                       # route samples within this arclength of the route START are certified against the current people belief
        self.lateral_offsets = tuple(float(o) for o in lateral_offsets)     # () = no shifted variants
        self.log = []

    # ------------------------------------------------------------------------------------------ logging
    def _rec(self, stream, inputs, ok, reason="", margin=None, t0=None, out=None):
        self.w.count(stream, ok)
        self.log.append(dict(stream=stream, inputs=[_short(x) for x in inputs], ok=bool(ok), reason=reason,
                             margin=None if margin is None else round(float(margin), 4), out=None if out is None else _short(out),
                             ms=None if t0 is None else round((time.perf_counter() - t0) * 1e3, 2)))

    # ------------------------------------------------------------------------------------------ s-dock (C2 static at rest)
    def dock_gen(self, s):
        t0 = time.perf_counter()
        surf = self.w.surfaces.get(s)
        if surf is None:
            self._rec("s-dock", (s,), False, "unknown surface", t0=t0)
            return
        q = surf.dock
        bel = self.w.belief_at(q.xyth(), holding=False)
        rest = Traj(np.zeros((1, 3)), self.w.gk.h, 0.0, 0.0, 0.0)
        o1, m1, r1, g1 = check1(rest, bel, self._gk)
        o2, m2, r2, g2 = check2(rest, bel, self._gk)
        ok = o1 and o2
        mb, kb = _binding(g1, g2)
        self._rec("s-dock", (s,), ok, f"binding {kb}" if ok else (r1 if not o1 else r2), mb, t0, q if ok else None)
        if ok:
            yield (q,)

    # ------------------------------------------------------------------------------------------ s-grasp (C3 corners)
    def grasp_gen(self, o):
        t0 = time.perf_counter()
        it = self.w.items.get(o)
        if it is None:
            self._rec("s-grasp", (o,), False, "unknown item", t0=t0)
            return
        rep = G3.check_grasp(it.extent, it.mass_range, mu_range=it.mu_range, category=it.category)
        allowed = SKILLS.get(o)
        out = []
        for r in rep.candidates:
            if not r.ok:
                self._rec("s-grasp", (o, r.name), False, f"C3: {r.reason}", r.margin, t0)
                continue
            if allowed is not None and r.name not in allowed:
                self._rec("s-grasp", (o, r.name), False, "no executor skill for this grasp", r.margin, t0)
                continue
            g = Grasp(o, r.name, int(r.arms), tuple(map(float, it.mass_range)), tuple(map(float, it.mu_range)),
                      Cert("C3_grasp_corners", float(r.margin), (("reason", r.reason), ("n_corners", 128))))
            self._rec("s-grasp", (o, r.name), True, r.reason, r.margin, t0, g)
            out.append(g)
        for g in out:
            yield (g,)

    # ------------------------------------------------------------------------------------------ s-place (support footprint)
    def place_gen(self, o, s):
        t0 = time.perf_counter()
        it, surf = self.w.items.get(o), self.w.surfaces.get(s)
        if it is None or surf is None:
            self._rec("s-place", (o, s), False, "unknown item/surface", t0=t0)
            return
        x, y, yaw = compose(surf.dock.xyth(), self.w.hold_rel[o])
        margin = self._support_margin(it, surf, (x, y, yaw))
        ok = margin >= 0.0
        p = Placement(o, s, x, y, surf.top_z + it.extent[2] / 2, yaw, Cert("support_footprint", float(margin)))
        self._rec("s-place", (o, s), ok, "" if ok else "footprint outside the surface top", margin, t0, p if ok else None)
        if ok:
            yield (p,)

    @staticmethod
    def _support_margin(it, surf, pose):
        """smallest inset [m] of the item's footprint corners inside the surface rectangle (< 0: a corner overhangs)."""
        a, b = it.extent[0] / 2, it.extent[1] / 2
        c, s = math.cos(pose[2]), math.sin(pose[2])
        best = math.inf
        for sx, sy in ((a, b), (a, -b), (-a, b), (-a, -b)):
            px, py = pose[0] + c * sx - s * sy, pose[1] + s * sx + c * sy
            r = rel_pose((surf.x, surf.y, surf.yaw), (px, py, 0.0))
            best = min(best, surf.size[0] / 2 - abs(r[0]), surf.size[1] / 2 - abs(r[1]))
        return float(best)

    # ------------------------------------------------------------------------------------------ t-reach (kinematics)
    def reach_test(self, o, p, g, q):
        t0 = time.perf_counter()
        if q.surface is None or p.surface != q.surface:
            self._rec("t-reach", (o, p, g, q), False, "conf is not this placement's dock", t0=t0)
            return False
        r = rel_pose(q.xyth(), (p.x, p.y, p.yaw))
        h = self.w.hold_rel[o]
        exy, eth = math.hypot(r[0] - h[0], r[1] - h[1]), abs(wrap(r[2] - h[2]))
        ok = exy <= REACH_TOL[0] and eth <= REACH_TOL[1]
        self._rec("t-reach", (o, p, g, q), ok, f"hold-pose error {exy:.3f} m / {math.degrees(eth):.1f} deg", min(REACH_TOL[0] - exy, REACH_TOL[1] - eth), t0)
        return ok

    # ------------------------------------------------------------------------------------------ routes (C1 + C2 static, T3 H0)
    def _candidate_routes(self, q1, q2):
        """[(pts, reverse, source)]: the scene/suffix routes first (unshifted), then their lateral-shift variants (each certified like any other route)."""
        base = self._base_candidate_routes(q1, q2)
        out = list(base)
        if self.lateral_offsets:
            for pts, rev, src in base:
                if src == "same_place":
                    continue
                for off in self.lateral_offsets:
                    sh = self.shift_route(pts, off)
                    if sh is not None:
                        out.append((sh, rev, f"{src}:shift{off:+.2f}"))
        return out

    @staticmethod
    def shift_route(pts, off):
        """pts with every INTERIOR vertex moved `off` m along the left normal of the local direction (unit(P[i+1]-P[i-1])); first/last kept.  None if no interior vertex."""
        P = np.asarray(pts, float)
        if len(P) < 3:
            return None
        Q = P.copy()
        for i in range(1, len(P) - 1):
            t = P[i + 1] - P[i - 1]
            n = float(np.hypot(*t))
            if n < 1e-9:
                continue
            t = t / n
            Q[i] = P[i] + off * np.array([-t[1], t[0]])
        return tuple(map(tuple, Q))

    def _base_candidate_routes(self, q1, q2):
        """[(pts, reverse, source)] scene routes connecting q1 -> q2 (forward along a route, or back along it in the scene's back mode)."""
        a, b = np.array(q1.xyth()[:2]), np.array(q2.xyth()[:2])
        if np.hypot(*(a - b)) <= SAME_PLACE[0] and abs(wrap(q1.yaw - q2.yaw)) <= SAME_PLACE[1]:
            return [(((q1.x, q1.y), (q2.x, q2.y)), False, "same_place")]
        out = []
        for name, pts in self.w.routes:
            P = np.asarray(pts, float)
            if np.hypot(*(P[0] - a)) <= ROUTE_MATCH_R and np.hypot(*(P[-1] - b)) <= ROUTE_MATCH_R:
                out.append((tuple(map(tuple, P)), False, f"scene:{name}"))
            elif name == "out_route" and np.hypot(*(P[-1] - a)) <= ROUTE_MATCH_R and np.hypot(*(P[0] - b)) <= ROUTE_MATCH_R:
                out.append((tuple(map(tuple, P[::-1])), self.w.back_mode == "reverse", f"scene:{name}:back"))
        # partial routes (replanning from a mid-route pose): q1 on the polyline (lateral <= ROUTE_MATCH_R) but not at its start -> the SUFFIX from q1's
        # projection toward the end near q2.  Offered in addition to the whole routes; certify_route checks it like any other route (no special case).
        for name, pts in self.w.routes:
            P = np.asarray(pts, float)
            orients = [(P, False, f"scene:{name}:suffix")]
            if name == "out_route":
                orients.append((P[::-1], self.w.back_mode == "reverse", f"scene:{name}:back:suffix"))
            for Q, rev, src in orients:
                if np.hypot(*(Q[-1] - b)) > ROUTE_MATCH_R or np.hypot(*(Q[0] - a)) <= ROUTE_MATCH_R:
                    continue
                proj = self._project(Q, a)
                if proj is None or proj[1] > ROUTE_MATCH_R:
                    continue
                i, _lat, pt = proj
                suf = [(float(q1.x), float(q1.y))] + [tuple(map(float, v)) for v in Q[i + 1:]]
                if len(suf) >= 2 and math.hypot(suf[1][0] - suf[0][0], suf[1][1] - suf[0][1]) < 1e-6:
                    suf.pop(1)
                if len(suf) >= 2:
                    out.append((tuple(suf), rev, src))
        return out

    @staticmethod
    def _project(Q, a):
        """-> (index i of the segment Q[i]->Q[i+1] containing the projection of a, lateral distance, projected point) or None."""
        best = None
        for i in range(len(Q) - 1):
            d = Q[i + 1] - Q[i]
            L2 = float(d @ d)
            if L2 < 1e-12:
                continue
            f = min(1.0, max(0.0, float((a - Q[i]) @ d) / L2))
            pt = Q[i] + f * d
            lat = float(np.hypot(*(a - pt)))
            if best is None or lat < best[1] - 1e-9:
                best = (i, lat, pt)
        return best

    def certify_route(self, pts, reverse, holding, item=None):
        """-> (Cert or None, reason).  T3 H0 at sigma*: from rest at each sample pose (heading along the travel direction, +pi when driving in reverse),
        a cruise-twist candidate (any T_S in gk.t_switch) passes check1 + check2.  People: samples within self.near_m of the route START use the CURRENT
        set-valued people belief (world.bel_people); farther samples use the people-free belief (T3 H0).
        RATIONALE: far-route people predictions are meaningless at plan time (they move / are re-observed before we get there), so only the near part is
        checked against people.  The gatekeeper still enforces T1 at execution, so this changes ADMISSION at plan time only, not execution safety."""
        if self.cert_mode == "segments":
            c, why, _segs, _v = self._certify_segments(pts, reverse, holding, item)
            return c, why
        gk = self._gk
        P = np.asarray(pts, float)
        seg = np.diff(P, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(L)])
        total = float(cum[-1])
        if total < 1e-6:
            return Cert("C1C2_static_route", 0.0, (("n_samples", 0), ("note", "zero-length (in-place) route"))), ""
        u = np.array([-self.v if reverse else self.v, 0.0, 0.0])
        worst, worst_row, worst_s = math.inf, "none", 0.0
        n = 0
        n_near, worst_near, row_near = 0, math.inf, "none"
        n_ppl = len(self.w.bel_people.people) if self.w.bel_people is not None else 0
        s_end = max(0.0, total - self.end_skip)
        for s in np.arange(0.0, s_end + 1e-9, self.ds):
            i = min(int(np.searchsorted(cum, s, side="right") - 1), len(L) - 1)
            while i < len(L) - 1 and L[i] < 1e-9:
                i += 1
            f = 0.0 if L[i] < 1e-9 else (s - cum[i]) / L[i]
            xy = P[i] + f * seg[i]
            yaw = math.atan2(seg[i, 1], seg[i, 0]) + (math.pi if reverse else 0.0)
            pose = (float(xy[0]), float(xy[1]), wrap(yaw))
            near = bool(n_ppl) and self.near_m > 0 and s <= self.near_m + 1e-9
            bel = self.w.belief_at(pose, holding=holding, item=item, people=near)
            ok_any, best, row, why = False, -math.inf, "none", ""
            for T_S in gk.t_switch:
                tr = plan_candidate(np.zeros(3), u, T_S, bel, gk)
                o1, m1, r1, g1 = check1(tr, bel, gk)
                if not o1:
                    why = f"T_S={T_S}: c1 {r1} {m1:.3f}"
                    continue
                o2, m2, r2, g2 = check2(tr, bel, gk)
                if not o2:
                    why = f"T_S={T_S}: c2 {r2} {m2:.3f}"
                    continue
                ok_any = True
                best, row = _binding(g1, g2)
                break
            n += 1
            if not ok_any:
                return None, f"no admissible candidate at s={s:.2f} m pose=({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f}deg): {why}"
            if best < worst:
                worst, worst_row, worst_s = best, row, float(s)
            if near:
                n_near += 1
                if best < worst_near:
                    worst_near, row_near = best, row
        return Cert("C1C2_static_route", float(worst), (("binding_row", worst_row), ("binding_s_m", round(worst_s, 2)), ("n_samples", n), ("ds", self.ds),
                                                        ("sigma_star", self.w.sigma_star), ("holding", bool(holding)), ("v_cruise", self.v),
                                                        ("length_m", round(total, 3)),
                                                        ("near_people_applied", bool(n_ppl) and self.near_m > 0), ("near_m", self.near_m), ("n_people", n_ppl), ("n_near_samples", n_near),
                                                        ("near_binding_row", row_near), ("near_margin", None if n_near == 0 else round(float(worst_near), 4)))), ""

    # ------------------------------------------------------------------------------------------ segment-mode route certificate
    def is_immediate(self, q1):
        """the route starts NOW: q1 is the world's current conf (within SAME_PLACE).  ASSUMPTION: a route from the current conf that follows a pick/place at that
        conf is also treated as starting now (that action's duration is not modelled)."""
        q0 = self.w.q0
        return math.hypot(q1.x - q0.x, q1.y - q0.y) <= SAME_PLACE[0] and abs(wrap(q1.yaw - q0.yaw)) <= SAME_PLACE[1]

    def _certify_segments(self, pts, reverse, holding, item=None, immediate=False):
        """-> (Cert or None, reason, segments or None, v_max or None).  For each speed in self.speeds (fastest first): plan the timed segments (segments.py) and
        check1 + check2 each one at its PREDICTED moving start state.  People: segments whose start arclength is within self.near_m of the route start use the CURRENT
        set-valued people belief (as the rest mode); later segments the people-free belief (T3 H0).  The first speed whose segments ALL certify wins.  The driven part
        ends end_skip m before the goal (the executor's final approach / docking creep takes over from rest, as in the rest mode)."""
        timed = self.people_pred == "time" and bool(immediate)
        key = (tuple(map(tuple, pts)), bool(reverse), bool(holding), item) + (("time", self.t_offset) if timed else ()) + \
              (("start_v", self.start_v) if (timed and self.start_v is not None) else ()) + \
              ((("plan_cert", "pred", self.t_offset),) if self.plan_cert == "pred" else ())
        if key in self._seg_cache:
            return self._seg_cache[key]
        out = self._certify_segments_uncached(pts, reverse, holding, item, timed)
        self._seg_cache[key] = out
        return out

    def _certify_segments_uncached(self, pts, reverse, holding, item, timed=False):
        gk = self._gk
        path = SEG.Path(pts)
        total = path.total
        if total < 1e-6:
            return Cert("C1C2_segments", 0.0, (("n_segments", 0), ("note", "zero-length (in-place) route"))), "", (), None
        s_stop = max(0.0, total - self.end_skip)
        n_ppl = len(self.w.bel_people.people) if self.w.bel_people is not None else 0
        if s_stop <= 0.05:
            return Cert("C1C2_segments", 0.0, (("n_segments", 0), ("note", "route shorter than the final-approach hand-over"), ("length_m", round(total, 3)))), "", (), None

        def bel_fn(pose, s_start, t_start=0.0):
            if self.plan_cert == "pred":                    # plan-time certifier (plan_cert.py): ALL people, whole route, prediction disc only (no near_m cut-off, no people-free tail)
                return PC.plan_belief(self.w, pose, holding, item, None, self.t_offset + t_start)
            if timed and n_ppl:
                return predict_people(runtime_belief_at(self.w, pose, holding, item, people=True), self.t_offset + t_start, self.people_ev_scale)
            near = bool(n_ppl) and self.near_m > 0 and s_start <= self.near_m + 1e-9
            return runtime_belief_at(self.w, pose, holding, item, people=near)

        whys = []
        for v in self.speeds:
            start = None
            if timed and self.start_v is not None:
                q0 = self.w.q0
                start = (q0.xyth(), self.start_v, path.project(np.array([q0.x, q0.y]), 0.0), 0.0)
            segs, why = SEG.plan_route_segments(path, reverse, v, gk, bel_fn, s_stop) if start is None else \
                SEG.plan_route_segments(path, reverse, v, gk, bel_fn, s_stop, start=start)
            if segs is None:
                whys.append(why)
                continue
            ms = [sg.margin for sg in segs]
            iw = int(np.argmin(ms))
            near_i = [i for i, sg in enumerate(segs) if n_ppl and (timed or (self.near_m > 0 and sg.s_start <= self.near_m + 1e-9))]
            nm = min((ms[i] for i in near_i), default=None)
            detail = (("binding_row", segs[iw].binding), ("binding_s_m", round(segs[iw].s_start, 2)), ("speed", v), ("v_cruise", v), ("n_segments", len(segs)),
                      ("T_S_hist", tuple(sorted(Counter(round(sg.T_S, 3) for sg in segs).items(), reverse=True))), ("seg_margins", tuple(round(float(m), 4) for m in ms)), ("holding", bool(holding)), ("length_m", round(total, 3)),
                      ("driven_m", round(s_stop, 3)), ("sigma_star", self.w.sigma_star), ("near_people_applied", bool(n_ppl) and self.near_m > 0), ("near_m", self.near_m),
                      ("n_people", n_ppl), ("n_near_samples", len(near_i)), ("near_binding_row", segs[min(near_i, key=lambda i: ms[i])].binding if near_i else "none"),
                      ("near_margin", None if nm is None else round(float(nm), 4)), ("speeds_failed", len(whys)))
            if timed:
                detail += (("people_pred", "time"), ("t_offset", self.t_offset))
            if self.plan_cert == "pred":
                detail += (("plan_cert", "pred"), ("t_offset", self.t_offset), ("pred_e_v", self.w.gk.e_v), ("pred_a_p", self.w.gk.a_p))
            return Cert("C1C2_segments", float(min(ms)), detail), "", tuple(segs), float(v)
        return None, "no admissible segment profile at any speed " + str(self.speeds) + ": " + " | ".join(w[:160] for w in whys), None, None

    def build_route(self, q1, q2, pts, rev, holding, item, src, immediate=None):
        """-> (Route or None, reason).  Certifies (mode-dependent) and attaches the segments / chosen speed in segments mode."""
        if self.cert_mode == "segments":
            cert, why, segs, v = self._certify_segments(pts, rev, holding, item, self.is_immediate(q1) if immediate is None else bool(immediate))
            if cert is None:
                return None, why
            return Route(q1.name, q2.name, pts, rev, "holding" if holding else "free", cert, src, v_max=v,
                         s_stop=max(0.0, SEG.Path(pts).total - self.end_skip), segments=segs), ""
        cert, why = self.certify_route(pts, rev, holding, item)
        if cert is None:
            return None, why
        return Route(q1.name, q2.name, pts, rev, "holding" if holding else "free", cert, src), ""

    def _route_gen(self, stream, q1, q2, holding, item=None, g=None):
        t0 = time.perf_counter()
        ins = (q1, q2) if g is None else (item, g, q1, q2)
        if q1 == q2:
            self._rec(stream, ins, False, "same conf", t0=t0)
            return
        cands = self._candidate_routes(q1, q2)
        if not cands:
            self._rec(stream, ins, False, "no scene route connects these confs (nav fallback not implemented)", t0=t0)
            return
        for pts, rev, src in cands:
            r, why = self.build_route(q1, q2, pts, rev, holding, item, src)
            if r is None:
                self._rec(stream, ins, False, f"{src}: {why}", t0=t0)
                continue
            cert = r.cert
            dt = dict(cert.detail)
            self._rec(stream, ins, True, f"{src}; binding {dt.get('binding_row')} at s={dt.get('binding_s_m')} m", cert.margin, t0, r)
            yield (r,)

    def route_free_gen(self, q1, q2):
        yield from self._route_gen("s-route-free", q1, q2, holding=False)

    def route_hold_gen(self, o, g, q1, q2):
        yield from self._route_gen("s-route-hold", q1, q2, holding=True, item=o, g=g)

    # ------------------------------------------------------------------------------------------ PDDLStream map
    def stream_map(self):
        from pddlstream.language.generator import from_gen_fn, from_test
        return {
            "s-dock": from_gen_fn(self.dock_gen),
            "s-grasp": from_gen_fn(self.grasp_gen),
            "s-place": from_gen_fn(self.place_gen),
            "t-reach": from_test(self.reach_test),
            "s-route-free": from_gen_fn(self.route_free_gen),
            "s-route-hold": from_gen_fn(self.route_hold_gen),
        }
