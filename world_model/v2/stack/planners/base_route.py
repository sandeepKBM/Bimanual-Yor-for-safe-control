"""stack/planners/base_route.py -- plan checks of a base route (table-behind D1, 11-09).  PURE: no physics import; the MuJoCo side is a backend
(world_model/v2/adapters/base_sweep.py, resolved by default_backend(); the stack core never imports adapters / nero).   SIMULATION ONLY.

A route = 1..MAX_SEGMENTS segments ("line", d metres along the robot's CURRENT +x) | ("turn", deg CCW) about the base rotation centre c (in place).
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
        if k not in ("line", "turn"):
            raise ValueError(f"route segment kind {k!r}; have line | turn")
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
        elif k == "line" and abs(v) > MAX_LINE_M + 1e-12:
            errs.append(f"route[{i}].line: {v} m outside [-{MAX_LINE_M}, {MAX_LINE_M}] m")
        elif k == "turn" and abs(v) > MAX_TURN_DEG + 1e-12:
            errs.append(f"route[{i}].turn: {v} deg outside [-{MAX_TURN_DEG}, {MAX_TURN_DEG}] deg")
    return errs


def route_end(p0, route):
    x, y, yaw = (float(a) for a in p0)
    for k, v in route:
        if k == "line":
            x, y = x + v * math.cos(yaw), y + v * math.sin(yaw)
        else:
            yaw += math.radians(v)
    return (x, y, yaw)


def route_poses(p0, route, step_m=STEP_M, step_deg=STEP_DEG):
    """-> [(seg index, x, y, yaw)]: the start pose (seg 0) and every sample; a segment's last sample is its exact end."""
    x, y, yaw = (float(a) for a in p0)
    out = [(0, x, y, yaw)]
    for i, (k, v) in enumerate(route):
        if k == "line":
            n = max(1, int(math.ceil(abs(v) / step_m - 1e-9)))
            for f in np.linspace(0.0, 1.0, n + 1)[1:]:
                out.append((i, x + f * v * math.cos(yaw), y + f * v * math.sin(yaw), yaw))
            x, y = x + v * math.cos(yaw), y + v * math.sin(yaw)
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


def route_duration(route, level, levels, gap=GAP_S):
    """seconds: every segment's profile at the level's (v, a) or (w, alpha), plus the rest gap after each."""
    v, a, w, al = levels[level]
    return float(sum((trap_T(x, v, a) if k == "line" else trap_T(math.radians(x), w, al)) + gap for k, x in route))


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


# ---- default plan -------------------------------------------------------------------------------------------------------------------
def default_route(task_def):
    """the TaskDef's first named route (R1) as a normalized route."""
    if not task_def.routes:
        raise ValueError(f"{task_def.name}: no route in TaskDef.routes")
    return normalize_route(task_def.routes[0][1])


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
    top1, top2 = float(sup["counter"]["top"]), float(sup["counter2"]["top"])
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
                f"{r['min_dist'] * 1e3:.1f} mm (need >= {(clr + margin_m) * 1e3:.1f} mm) with the base at x {w['pose'][0]:.3f}, y {w['pose'][1]:.3f}, yaw {math.degrees(w['pose'][2]):.0f} deg",
                reason="base_sweep", segment=int(w["seg"]), pair=list(w["pair"]), dist_m=float(r["min_dist"]), pose=list(w["pose"]), info=info)
    return info
