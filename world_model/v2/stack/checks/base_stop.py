"""checks/base_stop.py -- Rule 1 rows of the BASE-STOP recovery of a base_move step (11-16).  PURE: no physics / nero / adapters / mujoco import.   SIMULATION ONLY.

The recovery (nero/contents/carry.py CarryEpisode.base_stop, executor phase "base_stop"; stored backup [("base_stop", BASE_STOP_MAX_S), ("hold", 0.5)]):
the rest of the route is dropped, the twist is rate-limited to 0 at the admitted level's a / alpha, the arms hold, the base stops where it is (it does NOT
return), then holds; program "base_stop_finish" (USER 11-17): after the hold the base FINISHES THE PLANNED ROUTE from the landed pose (base_route.remaining_route) and
the plan's later steps run, landing on the goal support (an object held in the arms is not a safe landing).  The rows below say whether that is admissible at every branch time.

  trap_state(t, length, vmax, amax)               (s, v) of the rest-to-rest trapezoid / triangle profile (same law as nero/base/swerve.trap_profile, stack/planners/base_route.trap_T)
  segment_table(route, caps, gap)                 route + level caps -> [segment dict(index, kind, length, vmax, amax, t0, t1)] on the COMMANDED profile (+ rest gap after each)
  stop_state(segments, t)                         (segment index | None, s, v) at time t (None = at rest in a gap)
  pose_at(p0, route, segments, t, stop=False)     commanded base pose (x, y, yaw) at t; stop=True = the pose where a stop commanded at t comes to rest (s + v^2 / 2 a)
  recovery_row(skills)                            the step's recovery list -> base_stop.recovery
  in_route_row(segments, branch_times, extra, sweeps)   base_stop.in_route   the stop stays inside its segment, or the overshoot pose is clear (sweeps passed in)
  overshoot_cases / overshoot_pose / case_key     the branches that run past the segment end + the pose to sweep (adapter does the sweep); eval_times / decel_start_times add each moving segment's
                                                  decel start (analytic worst case) to the branch grid; drift_poses adds the MEASURED lateral / yaw drift (four corners)
  interrupted_index(segments, t)                  the segment a stop at t interrupts
  at_rest_row(segments, branch_times, settle, max_s)   base_stop.at_rest   the base is at rest within BASE_STOP_MAX_S
  base_rows_on_theta(rows)                        base_stop.tip / base_stop.torque   the admitted level's base rows on theta_t (renamed, guarantee and n kept)
  landing_hold_row(rest)                          base_stop.landing_hold   the payload rest row at the hold pose (a = 0)
  continuation_row(results)                       base_stop.continuation   the remaining route from the landed pose is collision-free (check_continuation at each branch time)

Labels (AGENTS.md "Bounds and assumptions"): the stop deceleration = the level's a / alpha (MODEL: driver.abort rate-limits the twist to 0 at them); the extra overshoot beyond
v^2 / 2a and the settle time are MEASURED on the stop card (nero/results/base_stop_card_2026-11-16/CARD.json: extra_overshoot_m / _rad, settle_s, n, src; a max over n, never a bound);
until that card exists the rows that need it are MISSING (never a pass).  BASE_STOP_MAX_S = 4 s is ASSUMED (carry.BASE_STOP_MAX_S; the planner stores the same number).
"""
from __future__ import annotations

import math

from world_model.v2.stack.checks.row import Row, missing

BASE_STOP_MAX_S = 4.0        # ASSUMED s: the recovery's timeout (= CarryEpisode.BASE_STOP_MAX_S = closed_chain.BASE_STOP_MAX_S)
PRE_STEER_MAX_S = 1.0        # ASSUMED s (= base_route.PRE_STEER_MAX_S = nero/base/drive.PRE_STEER_MAX_S)
SHIFT_YAW_SCRUB_RAD = math.radians(0.722)   # MEASURED scratchpad/swerve_shift/b.py (sim, base_move shift +0.02 / +0.10 m at L1 / L2, after the steer pre-align), n=4, max |yaw change| over n (not a bound): the yaw the
                                            # steer scrub + shift leave; the continuation sweep applies it as a yaw corner (+-, ramped over each shift piece, kept through the later pieces) on top of the card's extra_yaw_rad
TOL = 1e-9                   # numerical slack on the "stop ends inside the segment" comparison [m | rad]


# ---- the commanded profile --------------------------------------------------------------------------------------------------------------
def trap_state(t: float, length: float, vmax: float, amax: float):
    """(s, v) at time t of the rest-to-rest trapezoid (triangle when length < vmax^2 / amax) over |length|; unsigned; t clamped to [0, T]."""
    D = abs(float(length))
    if D <= 0.0:
        return 0.0, 0.0
    ta = vmax / amax
    if D < vmax * ta:
        ta = math.sqrt(D / amax)
        vp, tc = amax * ta, 0.0
    else:
        vp, tc = vmax, (D - vmax * ta) / vmax
    T = 2 * ta + tc
    t = min(max(float(t), 0.0), T)
    if t < ta:
        return 0.5 * amax * t * t, amax * t
    if t < ta + tc:
        return 0.5 * amax * ta * ta + vp * (t - ta), vp
    u = T - t
    return D - 0.5 * amax * u * u, amax * u


def _trap_T(D: float, vmax: float, amax: float) -> float:
    if D <= 0.0:
        return 0.0
    if D >= vmax * vmax / amax:
        return vmax / amax + D / vmax
    return 2.0 * math.sqrt(D / amax)


def _profile(D: float, vmax: float, amax: float):
    """(ta, tc, vp) of the rest-to-rest profile over D: accel time, cruise time, peak speed (same law as trap_state)."""
    ta = vmax / amax
    if D < vmax * ta:
        ta = math.sqrt(D / amax)
        return ta, 0.0, amax * ta
    return ta, (D - vmax * ta) / vmax, vmax


def decel_start_times(segments) -> list:
    """the time at which each MOVING segment's deceleration phase starts (t0 + ta + tc): the peak-speed instant of the profile.  A stop commanded there is the analytic worst case of
    the segment (stop distance = the remaining length exactly, so the slack is -extra), whatever the branch grid is (a segment shorter than the grid may have no branch time inside it)."""
    out = []
    for sg in segments:
        if sg["length"] > 0.0:
            ta, tc, _ = _profile(sg["length"], sg["vmax"], sg["amax"])
            out.append(round(float(sg["t0"] + ta + tc), 6))
    return out


def eval_times(segments, branch_times) -> list:
    """sorted unique (6 decimals) branch times PLUS every moving segment's decel-start time (5, USER 11-17: no moving segment may be skipped)."""
    return sorted({round(float(t), 6) for t in list(branch_times) + decel_start_times(segments)})


def segment_table(route, caps, gap: float) -> list:
    """route = ((kind, value m | deg), ...)  (kind line | turn | shift: shift = the internal holonomic sideways piece, stops like a line), caps = (v m/s, a m/s^2, w rad/s, alpha rad/s^2) of the level, gap = rest after each segment [s]
    -> [dict(index, kind, length (m | rad, abs), vmax, amax, t0, t1)]; t1 = end of the motion, the next segment starts at t1 + gap (as base_route.route_duration)."""
    v, a, w, al = caps
    out, t = [], 0.0
    for i, (k, x) in enumerate(route):
        if k == "shift":                                     # the driver's pre-steer phase (ASSUMED PRE_STEER_MAX_S, USER 11-17 "wait for wheels"): a REST phase at the MAX time (base still, stop distance 0); a stop in it interrupts the shift
            t += PRE_STEER_MAX_S
        D = abs(float(x)) if k in ("line", "shift") else math.radians(abs(float(x)))
        vm, am = (v, a) if k in ("line", "shift") else (w, al)
        T = _trap_T(D, vm, am)
        out.append(dict(index=i, kind=str(k), length=D, vmax=float(vm), amax=float(am), t0=t, t1=t + T))
        t += T + gap
    return out


def stop_state(segments, t: float):
    """-> (segment dict | None, s, v) at time t on the commanded profile; None = in a rest gap (or past the end): s = 0, v = 0."""
    for sg in segments:
        if sg["t0"] - 1e-12 <= t <= sg["t1"] + 1e-12:
            s, v = trap_state(t - sg["t0"], sg["length"], sg["vmax"], sg["amax"])
            return sg, s, v
    return None, 0.0, 0.0


def interrupted_index(segments, t: float) -> int:
    """index of the route segment a stop commanded at time t interrupts: the first segment whose motion has not finished (t <= t1); in a rest gap that is the NEXT segment;
    len(segments) when t is past the last motion."""
    for sg in segments:
        if t <= sg["t1"] + 1e-12:
            return sg["index"]
    return len(segments)


def pose_at(p0, route, segments, t: float, stop: bool = False, extra: dict | None = None):
    """commanded base pose (x, y, yaw) at time t (route poses of base_route.route_poses on the commanded profile).  stop=True: the pose where a stop commanded at t
    comes to rest = arc length s + v^2 / (2 a), capped at the segment end (MODEL: constant deceleration = the level's a / alpha), plus extra = dict(line=m, turn=rad)
    (the MEASURED overshoot beyond it, along the direction of motion) when given."""
    x, y, yaw = (float(a) for a in p0)
    for sg in segments:
        k, val = route[sg["index"]]
        sign = 1.0 if float(val) >= 0 else -1.0
        if t < sg["t0"] - 1e-12:
            break                                            # in the rest gap before this segment
        if t > sg["t1"] + 1e-12:
            x, y, yaw = _advance(x, y, yaw, k, sign * sg["length"])      # this segment is finished
            continue
        s, v = trap_state(t - sg["t0"], sg["length"], sg["vmax"], sg["amax"])
        if stop and v > 0.0:
            s = min(sg["length"], s + v * v / (2.0 * sg["amax"])) + (_ex(extra, k) if extra else 0.0)
        return _advance(x, y, yaw, k, sign * s)
    return (x, y, yaw)


def _advance(x, y, yaw, kind, ds):
    if kind == "line":
        return (x + ds * math.cos(yaw), y + ds * math.sin(yaw), yaw)
    if kind == "shift":                                      # internal sideways piece (USER 11-17): along the base's own +y, yaw held
        return (x - ds * math.sin(yaw), y + ds * math.cos(yaw), yaw)
    return (x, y, yaw + ds)


# ---- rows ------------------------------------------------------------------------------------------------------------------------------
def recovery_row(skills) -> Row:
    """the base step's recovery list (first entry = the stored one).  hold -> ok (program base_stop_finish); set_down | let_go -> refuted; retrace -> missing; empty -> the stored default (hold)."""
    sk = [str(s) for s in (skills or [])]
    first = sk[0] if sk else None
    if first is None or first == "hold":
        why = "stored default: base_stop then hold, then finish the planned route (program base_stop_finish, USER 11-17)" if first is None else "recovery skill hold: base_stop then hold, then finish the planned route (program base_stop_finish, USER 11-17)"
        return Row("base_stop.recovery", True, margin=None, unit="", guarantee="proof", worst=dict(skill=first or "hold", default=first is None),
                   assumptions=dict(model="CarryEpisode.base_stop (driver.abort, arms hold) then hold 0.5 s; MODEL"), why=why)
    if first in ("set_down", "let_go"):
        why = "no support under the object mid-route" if first == "set_down" else "drops it"
        return Row("base_stop.recovery", False, margin=None, unit="", guarantee="refuted", worst=dict(skill=first), assumptions={}, why=f"recovery {first}: {why}")
    if first == "retrace":
        return missing("base_stop.recovery", "base retrace not built")
    return missing("base_stop.recovery", f"recovery skill {first!r} has no base-step program")


def _ex(extra, kind) -> float:
    return max(0.0, float(extra["line" if kind in ("line", "shift") else "turn"]))       # clamp >= 0 (a negative "extra" must never shorten the stop)


def overshoot_cases(segments, branch_times, extra_overshoot: dict | None) -> list:
    """the evaluation times (the branch grid plus each moving segment's decel-start time, eval_times) where a stop commanded at t, with the MEASURED extra overshoot, runs past the end of
    its segment (USER 11-17: always add the extra; no shortcut): stop distance v^2 / (2 amax) + extra > remaining length.
    -> [dict(t, segment, kind, v, s, remaining, stop_dist, extra, excess)] (excess > 0 in m | rad).  [] when extra is None."""
    if extra_overshoot is None:
        return []
    out = []
    for t in eval_times(segments, branch_times):
        sg, s, v = stop_state(segments, t)
        if sg is None or v <= 0.0:
            continue
        rem = sg["length"] - s
        d0 = v * v / (2.0 * sg["amax"])
        ex = _ex(extra_overshoot, sg["kind"])
        if rem - (d0 + ex) < -TOL:
            out.append(dict(t=float(t), segment=sg["index"], kind=sg["kind"], v=float(v), s=float(s), remaining=float(rem), stop_dist=float(d0), extra=ex, excess=float(d0 + ex - rem)))
    return out


def drift_poses(pose, lateral_m: float, yaw_rad: float) -> list:
    """the poses of one landed / overshoot pose with the MEASURED drift of the stop (card extra_lateral_m / extra_yaw_rad): lateral perpendicular to the pose's heading, +- both sides,
    and the yaw +- (worst side not known a priori: all four corners are swept).  MODEL: the card's max over n is applied as a symmetric box."""
    x, y, yaw = (float(a) for a in pose)
    lat, dy = abs(float(lateral_m)), abs(float(yaw_rad))
    out = []
    for sl in ((-1.0, 1.0) if lat > 0.0 else (0.0,)):
        for sy in ((-1.0, 1.0) if dy > 0.0 else (0.0,)):
            out.append((x - math.sin(yaw) * sl * lat, y + math.cos(yaw) * sl * lat, yaw + sy * dy))
    return out


def overshoot_pose(p0, route, case: dict):
    """the base pose of an overshoot case: the planned END pose of the case's segment pushed on by the excess along the heading (line, in the direction of motion) / rotated by the excess (turn)."""
    x, y, yaw = (float(a) for a in p0)
    for i, (k, val) in enumerate(route[:case["segment"] + 1]):
        sign = 1.0 if float(val) >= 0 else -1.0
        L = abs(float(val)) if k in ("line", "shift") else math.radians(abs(float(val)))
        ds = sign * (L + (case["excess"] if i == case["segment"] else 0.0))
        x, y, yaw = _advance(x, y, yaw, k, ds)
    return (x, y, yaw)


def case_key(case: dict) -> float:
    return round(float(case["t"]), 6)


def in_route_row(segments, branch_times, extra_overshoot: dict | None, sweeps: dict | None = None) -> Row:
    """at every evaluation time t (the branch grid + each moving segment's decel start) inside a segment's motion with v(t) > 0: v^2 / (2 amax) + extra <= length - s(t)
    (the stop ends inside the segment it was commanded in).  extra_overshoot = dict(line=m, turn=rad) MEASURED (the card: extra beyond v^2 / 2a), ALWAYS added (USER 11-17), clamped >= 0;
    None -> MISSING.  Empty route / no moving evaluation time -> MISSING (never a vacuous pass).
    A branch where the stop runs past the segment end is NOT refuted outright: the adapter sweeps the robot at the overshoot pose (overshoot_cases / overshoot_pose, the
    base_sweep backend, same clearance as check_continuation) and passes the results in `sweeps` = {case_key(case): dict(ok, min_dist_m, clearance_min_m, worst)}; clear -> ok,
    colliding -> refuted (pair, distance); no sweep given for a case -> MISSING.
    Margins by kind: worst["margin_line_m"] = min over line within-segment slacks and over every overshoot sweep clearance margin [m]; worst["margin_turn_rad"] = min turn slack [rad];
    Row.margin = the line margin (unit "m"); None when no line case exists."""
    if extra_overshoot is None or any(extra_overshoot.get(k) is None for k in ("line", "turn")):
        return missing("base_stop.in_route", "no MEASURED stop overshoot (base_stop card CARD.json missing): not evaluated")
    if not segments:
        return missing("base_stop.in_route", "empty route: no segment to stop in")
    sweeps = sweeps or {}
    times = eval_times(segments, branch_times)
    cases = {case_key(c): c for c in overshoot_cases(segments, times, extra_overshoot)}
    worst, m_line, m_turn = None, math.inf, math.inf
    n_over, n_mov, bad, miss = 0, 0, None, None
    for t in times:
        sg, s, v = stop_state(segments, t)
        if sg is None or v <= 0.0:
            continue
        n_mov += 1
        c = cases.get(round(float(t), 6))
        if c is None:
            rem = sg["length"] - s
            slack = rem - (v * v / (2.0 * sg["amax"]) + _ex(extra_overshoot, sg["kind"]))
            if sg["kind"] in ("line", "shift"):
                m_line = min(m_line, slack)
            else:
                m_turn = min(m_turn, slack)
            if worst is None or slack < worst["slack"]:
                worst = dict(t=float(t), segment=sg["index"], kind=sg["kind"], v=float(v), remaining=float(rem), within_segment=True, slack=float(slack))
            continue
        n_over += 1
        r = sweeps.get(case_key(c))
        if r is None:
            miss = miss or c
            continue
        m = float(r["min_dist_m"] - r.get("clearance_min_m", 0.0))
        m_line = min(m_line, m)                                                      # a sweep clearance margin is a distance [m]
        if worst is None or m < worst["slack"]:
            worst = dict(t=c["t"], segment=c["segment"], kind=c["kind"], excess=c["excess"], sweep=dict(min_dist_m=float(r["min_dist_m"]), worst=r.get("worst")), within_segment=False, slack=float(m))
        if not r["ok"] and (bad is None or m < bad[0]):
            bad = (m, c, r)
    asm = dict(model="stop distance v^2 / 2a at the level's a / alpha (MODEL) + the MEASURED extra overshoot (max over the card's n); past the segment end: swept at the overshoot pose; "
                     "each moving segment also at its decel start (analytic worst case: stop distance = remaining length)",
               extra_overshoot=extra_overshoot, branch_grid="sampled branch times; triggers between them not checked (the decel-start case of each segment is analytic)")
    nb = len(times)
    ml = None if m_line == math.inf else float(m_line)
    mt = None if m_turn == math.inf else float(m_turn)
    wd = lambda w: dict(w, n_overshoot=n_over, margin_line_m=ml, margin_turn_rad=mt)
    if bad is not None:
        m, c, r = bad
        return Row("base_stop.in_route", False, margin=ml, unit="m", guarantee="refuted", worst=dict(t=c["t"], segment=c["segment"], excess=c["excess"], sweep=r.get("worst"), n_overshoot=n_over,
                   margin_line_m=ml, margin_turn_rad=mt), n=nb, assumptions=asm,
                   why=f"t = {c['t']:.2f} s, segment {c['segment']} ({c['kind']}): the stop runs {c['excess']:.4f} past the segment end and the robot there is {r['min_dist_m'] * 1e3:.1f} mm from the world "
                       f"(clearance {r.get('clearance_min_m', 0.0) * 1e3:.1f} mm; {r.get('worst')})")
    if miss is not None:
        return missing("base_stop.in_route", f"overshoot sweep not evaluated at t = {miss['t']:.2f} s (segment {miss['segment']}, excess {miss['excess']:.4f})")
    if worst is None:
        return missing("base_stop.in_route", "no evaluation time with the base moving: nothing was checked")
    return Row("base_stop.in_route", True, margin=ml, unit="m", guarantee="sampled", worst=wd(worst), n=nb, assumptions=asm, why="")


def at_rest_row(segments, branch_times, settle_s: float | None, max_s: float = BASE_STOP_MAX_S) -> Row:
    """max over the evaluation times (branch grid + each moving segment's decel start) of v(t) / amax + settle_s <= max_s (BASE_STOP_MAX_S, ASSUMED); settle_s MEASURED (the card); None -> MISSING.
    Separate margins per kind in worst: worst["margin_line_s"] / ["margin_turn_s"] = max_s - (the longest stop of that kind) [s]; Row.margin = the overall minimum.
    Empty route / no evaluation time inside a segment -> MISSING (never a vacuous pass)."""
    if settle_s is None:
        return missing("base_stop.at_rest", "no MEASURED settle time (base_stop card CARD.json missing): not evaluated")
    if not segments:
        return missing("base_stop.at_rest", "empty route: no segment to stop in")
    settle_s = max(0.0, float(settle_s))
    times = eval_times(segments, branch_times)
    worst, tmax, by = None, -1.0, {"line": -1.0, "turn": -1.0}
    for t in times:
        sg, s, v = stop_state(segments, t)
        if sg is None or v <= 0.0:
            continue
        T = v / sg["amax"] + settle_s
        kk = "line" if sg["kind"] in ("line", "shift") else "turn"          # a shift stops like a line (USER 11-17: same v^2 / 2a + card extra)
        by[kk] = max(by[kk], T)
        if T > tmax:
            tmax, worst = T, dict(t=float(t), segment=sg["index"], kind=sg["kind"], v=float(v), t_decel=float(v / sg["amax"]), settle_s=settle_s)
    if worst is None:
        return missing("base_stop.at_rest", "no evaluation time with the base moving: nothing was checked")
    worst = dict(worst, margin_line_s=(None if by["line"] < 0 else float(max_s - by["line"])), margin_turn_s=(None if by["turn"] < 0 else float(max_s - by["turn"])))
    ok = tmax <= max_s
    return Row("base_stop.at_rest", ok, margin=float(max_s - tmax), unit="s", guarantee="sampled" if ok else "refuted", worst=worst, n=len(times),
               assumptions=dict(model="decel time v / amax (MODEL) + the MEASURED settle time (max over the card's n)", BASE_STOP_MAX_S=dict(value=max_s, tag="ASSUMED", src="carry.BASE_STOP_MAX_S")),
               why="" if ok else f"v / a + settle = {tmax:.2f} s > BASE_STOP_MAX_S {max_s:.2f} s at t = {worst['t']:.2f}")


def _rename(r: Row | None, name: str) -> Row:
    if r is None:
        return missing(name, "the row is absent for the admitted level")
    return Row(name, r.ok, margin=r.margin, unit=r.unit, guarantee=r.guarantee, worst=r.worst, n=r.n, eps_upper=r.eps_upper, assumptions=dict(r.assumptions), why=r.why)


def base_rows_on_theta(rows: dict | None) -> list:
    """rows = {"base_tip": Row, "base_torque": Row, ["base_open_content": Row (tray tasks)]} of the admitted level on theta_t (adapters/base_envelope.pick_level) ->
    [base_stop.tip, base_stop.torque, (base_stop.open_content when the level has the row)] (guarantee and n kept)."""
    rows = rows or {}
    out = [_rename(rows.get("base_tip"), "base_stop.tip"), _rename(rows.get("base_torque"), "base_stop.torque")]
    if rows.get("base_open_content") is not None:
        out.append(_rename(rows["base_open_content"], "base_stop.open_content"))
    return out


def landing_hold_row(rest: Row | None) -> Row:
    """the payload rest row at the hold pose (a = 0) -> base_stop.landing_hold."""
    return _rename(rest, "base_stop.landing_hold")


def continuation_row(results) -> Row:
    """results = [dict(t, ok, min_dist_m, clearance_min_m, worst)] (base_route.check_continuation at the pose where a stop commanded at t lands).  ok iff all ok; sampled; MISSING when empty."""
    res = list(results or [])
    if not res:
        return missing("base_stop.continuation", "no continuation check evaluated")
    bad = [r for r in res if not r["ok"]]
    w = min(res, key=lambda r: r["min_dist_m"] - r.get("clearance_min_m", 0.0))
    margin = float(w["min_dist_m"] - w.get("clearance_min_m", 0.0))
    why = "" if not bad else (f"continuation from the stop at t = {w['t']:.2f} s: sweep distance {w['min_dist_m'] * 1e3:.1f} mm < clearance {w.get('clearance_min_m', 0.0) * 1e3:.1f} mm "
                              f"(worst {w.get('worst')}); {len(bad)} of {len(res)} branch times fail")
    return Row("base_stop.continuation", not bad, margin=margin, unit="m", guarantee="sampled" if not bad else "refuted", worst=dict(t=float(w["t"]), worst=w.get("worst"), n_fail=len(bad)), n=len(res),
               assumptions=dict(model="swept clearance of the rest of the PLANNED route (base_route.remaining_route, USER 11-17) from the landed pose (stop distance + MEASURED extra), arms at the planned hold pose; sampled branch times"), why=why)


def goal_support_row(r: Row | None) -> Row:
    """S of the program base_stop_finish: the support row at the goal on theta_t (the landing of the finished plan) -> base_stop.goal_support."""
    return _rename(r, "base_stop.goal_support")


_KIND_ORDER = {"UNTAGGED": 0, "FIXABLE": 1, "FIXABLE_GEOMETRY": 2, "NOT_FIXABLE": 3}


def next_step_row(steps) -> Row:
    """steps = [(name, admitted, why) | (name, admitted, why, tag)] of the plan's arm steps AFTER the base move (each is its own Rule 1 step); tag = dict(kind, knobs, key) of the refused
    step (fixability.step_tag), carried in worst["step_tags"] for the row's own fixability tag (the most fixable of them).  The base program finishes the route and then runs them, so it is
    not viable when Rule 1 refuses one.  Not yet evaluated (None) -> MISSING."""
    if steps is None:
        return missing("base_stop.next_step", "the steps after the base move are not evaluated")
    steps = [tuple(x) + (None,) * (4 - len(x)) for x in steps]
    bad = [(n, w, tg) for n, a, w, tg in steps if not a]
    if bad:
        tags = [dict(step=n, **{k: tg[k] for k in ("kind", "knobs", "key") if k in tg}) for n, _, tg in bad if isinstance(tg, dict)]
        hint = ""
        if tags:                                       # callers that tag by name / why only (no worst) read the most fixable kind from here
            hint = " [most fixable refused step: " + min((t["kind"] for t in tags), key=lambda k: _KIND_ORDER.get(k, 9)) + "]"
        return Row("base_stop.next_step", False, margin=None, unit="", guarantee="refuted", worst=dict(refused=[n for n, _, _ in bad], step_tags=tags), assumptions={},
                   why="the base program is not viable: Rule 1 refuses the next step(s) " + "; ".join(f"{n} ({w[:160]})" for n, w, _ in bad) + hint)
    return Row("base_stop.next_step", True, margin=None, unit="", guarantee="proof", worst=dict(steps=[n for n, _, _, _ in steps]), assumptions=dict(model="each following step is admitted by Rule 1 on its own"), why="")
