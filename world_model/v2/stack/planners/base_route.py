"""stack/planners/base_route.py -- plan checks of a base route (table-behind D1, 11-09).  PURE: no physics import; the MuJoCo side is a backend
(world_model/v2/adapters/base_sweep.py, resolved by default_backend(); the stack core never imports adapters / nero).   SIMULATION ONLY.

A route = 1..MAX_SEGMENTS segments ("line", d metres along the robot's CURRENT +x) | ("turn", deg CCW) about the base rotation centre c (in place).
Internal only (USER 11-17, never from the proposer: the schema accepts line | turn objects only): ("shift", d metres along the robot's CURRENT +y = left, yaw held, holonomic swerve base),
produced by remaining_route / _finish_correction for the sideways end-pose error; speed / acceleration = the level's line limits (swerve.py has no separate lateral limit).
Arms hold their joint references during it (ASSUMED A4); the carried object is welded to the claws, so it moves with the base.

  normalize_route(route)            [{"line": d}, ["turn", deg], ("line", d)] -> (("line", d), ("turn", deg), ...)       (raises ValueError)
  route_poses(p0, route)            sampled base poses along the route (1 cm per line step, 2 deg per turn step: ASSUMED A7)
  route_end(p0, route)              nominal end pose (x, y, yaw)
  route_duration(route, level, levels, gap)   trapezoid-profile time incl. the rest gap after every segment (nero/base/swerve.trap_profile law)
  check(world, task, wp, geo, backend)        the checks of a planned geometry (what closed_chain.geometry_for calls):
        (a) swept clearance of base, arms, object, mugs vs the world (convex-hull distance) >= backend CLEARANCE_MIN + margin   -> PlanRefused naming segment + geom pair + distance
        (b) table-2 reach: the arm path after the base move is part of the planned geometry (SRS IK + DP in the base frame: the object keeps its base-frame
            offset across the base move; later moves are rotated into the base frame), so reach / joint margins there are the planner's own; reported
            here as info (post-base margin).  TODO: the end-error box corners (MEASURED base card not wired).
        (c) object underside above counter 2 during the advance >= CLEAR_M[kind]   (the planner's existing ASSUMED clearance)
Labels: A7 sweep sampling (ASSUMED), MAX_SEGMENTS 4 / MAX_LINE_M 1.0 / MAX_TURN_DEG 180 (ASSUMED outer ranges, design section 8 A3), GAP_S 0.5 (ASSUMED, smoke gap),
clearance 5 mm = nero/clearance.CLEARANCE_MIN (ASSUMED, read from the backend), tracking margin = 0 (no MEASURED base path-tracking error exists yet; stated in the result).
"""
from __future__ import annotations

import importlib
import math
import os

import numpy as np

MAX_SEGMENTS = 4            # ASSUMED (design A3)
MAX_LINE_M = 1.0            # ASSUMED (design A3) [m]
MAX_TURN_DEG = 180.0        # ASSUMED (design A3) [deg]
STEP_M = 0.01               # ASSUMED A7: sweep sampling along a line [m]
STEP_DEG = 2.0              # ASSUMED A7: sweep sampling in a turn [deg]
GAP_S = 0.5                 # ASSUMED: rest after each route segment (smoke gap) [s]
TRACK_MARGIN_M = 0.0        # no MEASURED base path-tracking error is wired yet -> 0, stated in the result (design section 2, item 2)
MIN_LINE_M = 1e-6


def default_backend():
    """the MuJoCo backend.  Resolved by a lazy import string, as stack/registry.py does for its components (the stack core must not import adapters)."""
    return importlib.import_module("world_model.v2.adapters.base_sweep")


def _refuse(msg, **info):
    from world_model.v2.stack.planners.closed_chain import PlanRefused
    raise PlanRefused(msg, **info)


# ---- routes -------------------------------------------------------------------------------------------------------------------------
def normalize_route(route):
    """-> tuple of ("line", float) | ("turn", float).  Accepts {"line": d} / {"turn": deg} objects and ("line", d) / ["turn", deg] pairs.  Ranges are NOT checked here."""
    out = []
    for s in route:
        if isinstance(s, dict):
            if len(s) != 1:
                raise ValueError(f"route segment must have exactly one key (line | turn): {s!r}")
            (k, v), = s.items()
        else:
            k, v = s
        if k not in ("line", "turn", "shift"):
            raise ValueError(f"route segment kind {k!r}; have line | turn")
        if k == "shift" and isinstance(s, dict):
            raise ValueError("route segment kind 'shift' is internal (the executor's sideways correction); not accepted as an object")
        out.append((str(k), float(v)))
    return tuple(out)


def check_ranges(route):
    """-> list of problems (empty = inside the ASSUMED outer ranges)."""
    errs = []
    if not 1 <= len(route) <= MAX_SEGMENTS:
        errs.append(f"route: {len(route)} segments, need 1..{MAX_SEGMENTS}")
    for i, (k, v) in enumerate(route):
        if not math.isfinite(v):
            errs.append(f"route[{i}].{k}: not finite")
        elif k in ("line", "shift") and abs(v) > MAX_LINE_M + 1e-12:
            errs.append(f"route[{i}].line: {v} m outside [-{MAX_LINE_M}, {MAX_LINE_M}] m")
        elif k == "turn" and abs(v) > MAX_TURN_DEG + 1e-12:
            errs.append(f"route[{i}].turn: {v} deg outside [-{MAX_TURN_DEG}, {MAX_TURN_DEG}] deg")
    return errs


def route_end(p0, route):
    x, y, yaw = (float(a) for a in p0)
    for k, v in route:
        if k == "line":
            x, y = x + v * math.cos(yaw), y + v * math.sin(yaw)
        elif k == "shift":                                  # internal (USER 11-17): sideways along the base's own +y (left), yaw held
            x, y = x - v * math.sin(yaw), y + v * math.cos(yaw)
        else:
            yaw += math.radians(v)
    return (x, y, yaw)


def route_poses(p0, route, step_m=STEP_M, step_deg=STEP_DEG, scrub=0.0):
    """-> [(seg index, x, y, yaw)]: the start pose (seg 0) and every sample; a segment's last sample is its exact end.
    scrub (rad, default 0): yaw corner of a shift piece (the steer scrub, SHIFT_YAW_SCRUB_RAD): the yaw ramps by `scrub` over each shift and the offset is kept by the later pieces (open-loop turns)."""
    x, y, yaw = (float(a) for a in p0)
    out = [(0, x, y, yaw)]
    for i, (k, v) in enumerate(route):
        if k in ("line", "shift"):
            ux, uy = (math.cos(yaw), math.sin(yaw)) if k == "line" else (-math.sin(yaw), math.cos(yaw))
            n = max(1, int(math.ceil(abs(v) / step_m - 1e-9)))
            sc = scrub if k == "shift" else 0.0
            for f in np.linspace(0.0, 1.0, n + 1)[1:]:
                out.append((i, x + f * v * ux, y + f * v * uy, yaw + f * sc))
            x, y, yaw = x + v * ux, y + v * uy, yaw + sc
        else:
            n = max(1, int(math.ceil(abs(v) / step_deg - 1e-9)))
            for f in np.linspace(0.0, 1.0, n + 1)[1:]:
                out.append((i, x, y, yaw + f * math.radians(v)))
            yaw += math.radians(v)
    return out


def trap_T(dist, vmax, amax):
    """duration of the rest-to-rest trapezoid / triangle profile over |dist| (same law as nero/base/swerve.trap_profile)."""
    s = abs(float(dist))
    if s <= 0:
        return 0.0
    if s >= vmax * vmax / amax:
        return vmax / amax + s / vmax
    return 2.0 * math.sqrt(s / amax)


from world_model.v2.stack.checks import base_stop as _BS          # pure (no physics): the stop rows own SHIFT_YAW_SCRUB_RAD (one definition) and the mirrored PRE_STEER_MAX_S

SHIFT_YAW_SCRUB_RAD = _BS.SHIFT_YAW_SCRUB_RAD       # MEASURED scratchpad/swerve_shift/b.py, n=4, max over n (not a bound); see base_stop.py
PRE_STEER_MAX_S = 1.0       # ASSUMED s: the driver's pre-steer phase before a shift = the base STANDS STILL while the steer modules turn (= nero/base/drive.PRE_STEER_MAX_S; the stack core cannot import nero: keep equal)


assert PRE_STEER_MAX_S == _BS.PRE_STEER_MAX_S, "PRE_STEER_MAX_S must be equal in base_route and base_stop (and nero/base/drive)"


def route_duration(route, level, levels, gap=GAP_S):
    """seconds: every segment's profile at the level's (v, a) or (w, alpha), plus the rest gap after each; a shift also adds the driver's pre-steer REST phase at its MAX (PRE_STEER_MAX_S, no motion), so the
    timeline is never shorter than what the driver runs (fail closed)."""
    v, a, w, al = levels[level]
    return float(sum((trap_T(x, v, a) if k in ("line", "shift") else trap_T(math.radians(x), w, al)) + gap + (PRE_STEER_MAX_S if k == "shift" else 0.0) for k, x in route))


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


# ---- continuation from a landed pose (11-13 F2) -----------------------------------------------------------------------------------------
MIN_LINE_M = 0.005          # ASSUMED m: a continuation line shorter than this is skipped (inside the ROUTE_TOL_M order)
MIN_TURN_DEG = 0.5          # ASSUMED deg: a continuation turn smaller than this is skipped


def continuation_route(p_now, p_end):
    """PURE.  turn -> line -> turn route (normalized) from the base pose p_now = (x, y, yaw) to p_end: face the target (driving backwards when that
    turn is > 90 deg), drive (chunks of <= MAX_LINE_M), turn to p_end's yaw.  [] when already there.  ValueError when it needs more than MAX_SEGMENTS."""
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    x, y, yaw = (float(a) for a in p_now)
    dx, dy = float(p_end[0]) - x, float(p_end[1]) - y
    dist = math.hypot(dx, dy)
    out = []
    if dist >= MIN_LINE_M:
        t1 = wrap(math.atan2(dy, dx) - yaw)
        sign = 1.0
        if abs(t1) > math.pi / 2:
            t1, sign = wrap(t1 + math.pi), -1.0
        if abs(math.degrees(t1)) >= MIN_TURN_DEG:
            out.append(("turn", math.degrees(t1)))
        yaw += t1
        n = int(math.ceil(dist / MAX_LINE_M - 1e-9))
        out += [("line", sign * dist / n)] * n
    t2 = wrap(float(p_end[2]) - yaw)
    if abs(math.degrees(t2)) >= MIN_TURN_DEG:
        out.append(("turn", math.degrees(t2)))
    if len(out) > MAX_SEGMENTS:
        raise ValueError(f"continuation route needs {len(out)} segments (max {MAX_SEGMENTS}): {out}")
    return tuple(out)


def route_start(p_end, route):
    """PURE.  the start pose (x, y, yaw) of `route` given its nominal end pose (inverse of route_end)."""
    x, y, yaw = (float(a) for a in p_end)
    for k, v in reversed(tuple(route)):
        if k == "line":
            x, y = x - v * math.cos(yaw), y - v * math.sin(yaw)
        elif k == "shift":
            x, y = x + v * math.sin(yaw), y - v * math.cos(yaw)
        else:
            yaw -= math.radians(v)
    return (x, y, yaw)


def _wrap_pi(a, tie_sign=1.0):
    """a wrapped to [-pi, pi]; a tie at exactly +-pi (within 1e-9) takes the sign tie_sign (the planned turn's sign), so +-180 is well defined."""
    t = (float(a) + math.pi) % (2 * math.pi) - math.pi
    if abs(abs(t) - math.pi) < 1e-9:
        return math.copysign(math.pi, tie_sign if tie_sign else 1.0)
    return t


def _finish_correction(p_from, p_end, tie_sign=1.0):
    """PURE (11-17).  The pieces that bring the base from p_from = (x, y, yaw) to the planned end pose p_end.  USER 11-17 "swerve shift": the forward / back error is a ("line", fwd), the
    sideways error (perpendicular to p_from's heading) a holonomic ("shift", lat) (yaw held, +lat = the base's own left), then the yaw turn (SHORTEST rotation; +-180 takes tie_sign).  No turn-in-place
    for the sideways error.  Pieces below MIN_LINE_M (line, shift) / MIN_TURN_DEG (turn) are dropped.  BIMANUAL_LEGACY_TURN_SHIFT=1: a sideways error > MIN_LINE_M -> turn-line-turn (continuation_route)."""
    x, y, yaw = (float(a) for a in p_from)
    ex, ey = float(p_end[0]) - x, float(p_end[1]) - y
    fwd = ex * math.cos(yaw) + ey * math.sin(yaw)
    lat = -ex * math.sin(yaw) + ey * math.cos(yaw)
    if abs(lat) > MIN_LINE_M and os.environ.get("BIMANUAL_LEGACY_TURN_SHIFT") == "1":
        try:
            return tuple(continuation_route(p_from, p_end))
        except ValueError:
            pass
    out = []
    if abs(fwd) >= MIN_LINE_M:
        out.append(("line", float(fwd)))
    if abs(lat) >= MIN_LINE_M:
        out.append(("shift", float(lat)))
    t = math.degrees(_wrap_pi(float(p_end[2]) - yaw, tie_sign))
    if abs(t) >= MIN_TURN_DEG:
        out.append(("turn", float(t)))
    return tuple(out)


def remaining_route(p0, route, i_seg, p_land):
    """PURE (USER 11-17: resume = FINISH THE PLANNED ROUTE).  The route from the landed pose p_land = (x, y, yaw) of a stop in segment `i_seg` of `route` (planned from the start pose p0):
    the rest of segment i_seg to ITS planned end pose, then segments i_seg+1.. unchanged, then a final correction so that the returned route ends at route_end(p0, route).
      line   signed distance from p_land to the segment's planned end pose, projected on the landed heading (negative = drive back: the stop overshot the end by more than MIN_LINE_M)
      turn   the SHORTEST rotation (USER 11-17) from the landed yaw to the segment's planned end yaw, in degrees; exactly +-180 takes the planned turn's sign
      A landing within MIN_LINE_M (line) / MIN_TURN_DEG (turn) of the segment end, before or PAST it, counts as arrived (the piece is dropped: no reverse drive).
      final correction  (_finish_correction) from the pose the returned route reaches to route_end(p0, route): line + shift (sideways, holonomic, USER 11-17) + shortest turn; BIMANUAL_LEGACY_TURN_SHIFT=1 = turn-line-turn for a sideways error.
    BIMANUAL_LEGACY_REMAINING_ROUTE=1: the 11-17 first version (wrap to [-pi, pi), no final correction).  i_seg >= len(route) (the stop came after the last segment) -> ()."""
    import os
    route = normalize_route(route)
    if i_seg >= len(route):
        return ()
    legacy = os.environ.get("BIMANUAL_LEGACY_REMAINING_ROUTE") == "1"
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    xe, ye, yawe = route_end(p0, route[:i_seg + 1])
    x, y, yaw = (float(a) for a in p_land)
    k, val = route[i_seg]
    sign = 1.0 if float(val) >= 0 else -1.0
    out = []
    if k in ("line", "shift"):
        d = ((xe - x) * math.cos(yaw) + (ye - y) * math.sin(yaw)) if k == "line" else (-(xe - x) * math.sin(yaw) + (ye - y) * math.cos(yaw))
        if abs(d) >= MIN_LINE_M:
            out.append((k, float(d)))
    else:
        t = math.degrees(wrap(yawe - yaw) if legacy else _wrap_pi(yawe - yaw, sign))
        if abs(t) >= MIN_TURN_DEG:
            out.append(("turn", float(t)))
    res = tuple(out) + tuple(route[i_seg + 1:])
    if legacy:
        return res
    p_end = route_end(p0, route)
    tie = 1.0
    for kk, vv in reversed(route):
        if kk == "turn" and vv != 0:
            tie = 1.0 if vv > 0 else -1.0
            break
    return res + _finish_correction(route_end(p_land, res), p_end, tie)


def check_continuation(world, geo, prev_name, p_now, route, backend=None, margin_m=TRACK_MARGIN_M):
    """sweep clearance (as `check` (a)) of `route` from p_now with the arms at their planned hold pose (end of move `prev_name`).  -> info dict with ok."""
    be = backend or default_backend()
    i_hold = geo.bounds[prev_name][1]
    off = np.asarray(geo.offs[i_hold], float)
    q_hold = {s: np.asarray(geo.q[s][i_hold], float) for s in geo.q}
    clr = float(getattr(be, "CLEARANCE_MIN", 0.005))
    poses = route_poses(p_now, route)
    if any(k == "shift" for k, _ in route):            # the steer scrub of a shift: sweep both yaw corners too (end-pose yaw error stays: the final turn is the planned one, open loop)
        poses = poses + route_poses(p_now, route, scrub=SHIFT_YAW_SCRUB_RAD) + route_poses(p_now, route, scrub=-SHIFT_YAW_SCRUB_RAD)
    r = be.sweep(world, poses, q_hold=q_hold, obj_off=tuple(float(x) for x in off), clearance_min=clr + margin_m)
    return dict(ok=bool(r["min_dist"] >= clr + margin_m), min_dist_m=float(r["min_dist"]), clearance_min_m=clr, worst=r["worst"],
                tag="sweep from the LANDED pose; arms at the planned hold pose (an interrupted arm pose can differ: not modelled); margin 0 ASSUMED")


# ---- default plan -------------------------------------------------------------------------------------------------------------------
def default_route(task_def):
    """the TaskDef's first named route (R1) as a normalized route."""
    if not task_def.routes:
        raise ValueError(f"{task_def.name}: no route in TaskDef.routes")
    return normalize_route(task_def.routes[0][1])


def _goal_support(sup):
    """the support the route ends at, from the backend's support dict (the episode path passes the KIND as `task`, not the TaskDef name): the bookcase compartment when the
    world has one, else counter2."""
    return "bookcase_l1" if "bookcase_l1" in sup else "counter2"


# ---- the checks ---------------------------------------------------------------------------------------------------------------------
def check(world, task, wp, geo, clear_m, backend=None, margin_m=TRACK_MARGIN_M):
    """geo = a closed_chain.Geometry of `wp` (arm space, base frame).  clear_m = closed_chain.CLEAR_M[kind].  Raises PlanRefused.  -> info dict (stored in geo.info['base'])."""
    be = backend or default_backend()
    if len(wp.base) != 1:
        _refuse(f"{task}: a base plan needs exactly one base_move (found {len(wp.base)})", reason="base_plan")
    after, route, level = wp.base[0]
    route = tuple(route)
    errs = check_ranges(route)
    if errs:
        _refuse("; ".join(errs), reason="base_route_range")
    sup = be.supports(world)
    gs = _goal_support(sup)
    top1, top2 = float(sup["counter"]["top"]), float(sup[gs]["top"])
    if after < 1 or after > len(wp.names):
        _refuse(f"{task}: base_move must come after the first arm move (the object is lifted before the base moves)", reason="base_plan")
    i_hold = geo.bounds[wp.names[after - 1]][1]                       # last sample of the move before the base move
    off = np.asarray(geo.offs[i_hold], float)
    q_hold = {s: np.asarray(geo.q[s][i_hold], float) for s in geo.q}
    z_under2 = top1 + float(off[2]) - top2                            # object underside above counter 2 (flat object: underside = start support top + offset)
    if z_under2 < clear_m - 1e-9:
        _refuse(f"{task}: during the base move the object underside is {z_under2:.3f} m above counter 2 (top {top2:.3f} m); needs >= {clear_m:.3f} m "
                f"(closed_chain.CLEAR_M, ASSUMED)", reason="base_height", z_under=float(z_under2), need=float(clear_m))
    p0 = be.start_pose(world)
    poses = route_poses(p0, route)
    clr = float(getattr(be, "CLEARANCE_MIN", 0.005))
    r = be.sweep(world, poses, q_hold=q_hold, obj_off=tuple(float(x) for x in off), clearance_min=clr + margin_m)
    end = route_end(p0, route)
    info = dict(route=[list(s) for s in route], level=level, start_pose=[float(a) for a in p0], end_pose=[float(a) for a in end], n_poses=len(poses),
                sweep_min_dist_m=float(r["min_dist"]), sweep_worst=r["worst"], sweep_by_segment=r["by_segment"], clearance_min_m=clr, tracking_margin_m=float(margin_m),
                tracking_margin_tag="no MEASURED base path-tracking error wired: 0 (ASSUMED)", z_under_counter2_m=float(z_under2),
                table2_margin_min=float(min(geo.margin[s][i_hold:].min() for s in geo.margin)),
                table2_end_error_box="TODO: not evaluated (no MEASURED base end-error box wired)",
                sampling=dict(step_m=STEP_M, step_deg=STEP_DEG, tag="ASSUMED A7"))
    if r["min_dist"] < clr + margin_m:
        w = r["worst"]
        k, v = route[w["seg"]]
        _refuse(f"base_move segment {w['seg']} ({k} {v:g}{' m' if k == 'line' else ' deg'}) collides: {w['group']} geom {w['pair'][0]} vs world geom {w['pair'][1]} at "
                f"{r['min_dist'] * 1e3:.1f} mm (need >= {(clr + margin_m) * 1e3:.1f} mm)"
                + (f" with the base at x {w['pose'][0]:.3f}, y {w['pose'][1]:.3f}, yaw {math.degrees(w['pose'][2]):.0f} deg" if os.environ.get("BIMANUAL_LEGACY_REFUSAL_POSE") == "1" else ""),
                reason="base_sweep", segment=int(w["seg"]), pair=list(w["pair"]), dist_m=float(r["min_dist"]), pose=list(w["pose"]), info=info)
    return info
