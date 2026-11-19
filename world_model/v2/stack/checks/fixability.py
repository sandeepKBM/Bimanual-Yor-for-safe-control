"""checks/fixability.py -- which failing checker rows can the PROPOSER'S PLAN fix?  (T4: Rule 1 refusal -> re-ask, USER 11-16.)  PURE: no physics, no nero import.

A row is tagged by what it DEPENDS ON in the checker code (adapters/rule1_contents motion_row / collision_row / landing_row, checks/admit, checks/collision, checks/motion ...):
  FIXABLE           the plan's own choices change the row: placement (move dx, dy, place support), lift height (lift dz), waypoints (the arm poses along the path: torque / payload,
                    open content, joint limits, reach, IK), or the recovery skill of the refused step (hold | retrace | set_down | let_go).
  FIXABLE_GEOMETRY  a collision row whose binding term is the SWEPT lower bound (distance minus a speed term between two 1 cm samples).  The plan does not set speeds, so only the
                    geometry (placement / lift height) can move it; the speed model is the checker's.
  NOT_FIXABLE       a MISSING row (no model), or a row that fails only because of an ASSUMED value / model the plan cannot change (stop time, sensor-noise bins, reading coverage,
                    base level / base speed, stop model, the executor's stored backups).
  UNTAGGED          not recognised.  The owner's gate fails on any UNTAGGED row name emitted today.
Torque / payload on the plan's path is FIXABLE (USER 11-16) although "a heavy bin may fail on every path": the knob exists, success is not promised.

Row shapes seen: leaf rows (`support_dest`, `torque_payload[seg]`, ...) and MERGED rows of rule1_contents (`<program>:<what>@t=<t>/<kind>[seg]`, `<program>:landing[bin<j>]/landing[...]`) whose
`worst` dict holds the sub-rows by name (`stop`, `hold_after_stop`, `retrace`, `advance`, `open_content[stop|move]`, `collision`, `dynamic[seg]`, `support_dest`, `hold_poses`, `R_K[...]`, `S`).
`leaves(row)` expands a merged row into its failing leaves; `tag(row)` tags a leaf, or a merged row by its failing leaves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FIXABLE, FIXABLE_GEOMETRY, NOT_FIXABLE, UNTAGGED = "FIXABLE", "FIXABLE_GEOMETRY", "NOT_FIXABLE", "UNTAGGED"
KINDS = (FIXABLE, FIXABLE_GEOMETRY, NOT_FIXABLE, UNTAGGED)

K_PLACE = "placement (move dx, dy before place)"
K_LIFT = "lift height (lift dz)"
K_WAYPTS = "waypoints (arm poses along the path)"
K_RECOV = "recovery skill for this step (hold | retrace | set_down | let_go)"


@dataclass(frozen=True)
class Tag:
    kind: str
    knobs: tuple = ()
    why: str = ""
    key: str = ""             # short machine key of the row family, e.g. "collision_sample:obj_world" (the failure signature uses it)
    pair: str = ""            # collision rows: the closest pair "geomA|geomB" parsed from the row's why text ("" otherwise)
    distance_m: float | None = None
    d_min_m: float | None = None
    swept_bound_m: float | None = None


_COLL = re.compile(r"(?P<cls>obj_world|arm_world|arm_arm)\s+(?P<pair>\S+)\s+distance\s+(?P<d>-?[\d.]+)\s*m,\s*swept bound\s+(?P<lb>-?[\d.]+)\s*m\s*<\s*d_min\s+(?P<dmin>[\d.]+)\s*m.*?\((?P<kind>sample|interval)\s+\d+\)")

# the families this module covers: (key, kind, how matched, knobs).  Matching is by the LEAF row name (the part after "/" of a merged row, or the sub-row name in its `worst`), base = name before "[".
KNOWN_ROW_FAMILIES = {
    "collision_sample:<class>":   "FIXABLE: leaf `collision*`, guarantee != missing, why text '(sample k)' (a sampled pose is below d_min); class obj_world | arm_world | arm_arm parsed from why; knobs placement, lift height",
    "collision_swept:<class>":    "FIXABLE_GEOMETRY: same row, why text '(interval k)' (the swept bound between samples is below d_min); knobs placement, lift height",
    "collision_missing":          "NOT_FIXABLE: `collision` with guarantee missing (scripted plan: no collision model)",
    "support_dest":               "FIXABLE: leaf `support_dest` (and `S` when not missing); knobs placement",
    "torque_payload":             "FIXABLE (USER 11-16): leaf base `torque_payload`, `rest_payload`, `hold_poses`; knobs waypoints; heavy bin may fail on every path",
    "open_content_path":          "FIXABLE (USER 11-16): leaf `open_content[<seg>]` of the static admission (seg is not stop / move); knobs waypoints",
    "recovery_torque:<which>":    "FIXABLE: leaf `stop` | `hold_after_stop` | `retrace` | `advance` and `open_content[stop]` | `open_content[move]` of ONE recovery program; knobs recovery skill",
    "bin_refuse":                 "FIXABLE: leaf base `refuse` (read program bin: no acceleration >= A_MIN admitted); knobs waypoints, recovery skill",
    "witness":                    "FIXABLE: leaf base `R_K` with guarantee != missing (the planner's remaining plan is refused on the landing's set); knobs waypoints, placement",
    "joint_reach_ik":             "FIXABLE: leaf base `joint_pos` | `joint_speed` | `reach` | `closed_chain` | any base containing `ik`; knobs waypoints",
    "plan_shape":                 "FIXABLE: leaf `plan_shape` (goal miss); knobs placement",
    "dynamic":                    "NOT_FIXABLE: leaf base `dynamic` (sampled worlds with the executor's stored backups)",
    "coverage":                   "NOT_FIXABLE: `<program>:coverage` / `coverage` (reading bins cover the reading: an ASSUMED bin width) and `<program>:theta_t` (empty set)",
    "tilt":                       "NOT_FIXABLE: leaf `obj_tilt` (measured table; plan does not set the acceleration)",
    "base":                       "NOT_FIXABLE: leaf base starting `base_` (base tip / torque / open content at the base level) and `recovery[base_move]` (no stored base recovery)",
    "missing:<family>":           "NOT_FIXABLE: any leaf with guarantee missing (no model built): stop overshoot past the last sample (`stop_back[seg]` ...), `S`, `R_K`, `plan`, `tilt`, `recovery[base_move]`",
    "base_stop.<fixable>":        "FIXABLE (also when MISSING / refuted): `base_stop.recovery` (the step's recovery skill), `base_stop.continuation` (the plan's route), `base_stop.landing_hold` (waypoints / hold pose); "
                                  "EXCEPT a row MISSING because the MEASURED stop card is absent / does not cover the level or task (why names the card) -> NOT_FIXABLE, key `missing:measured stop card`",
    "base_stop.goal_support":     "FIXABLE: the support row at the goal on theta_t (the landing of the finished plan); knobs placement (a MISSING one stays FIXABLE only when the why does not name the stop card)",
    "base_stop.next_step":        "copies the MOST FIXABLE tag among the refusing later steps (step_tag of each refused step's records; the row's worst['step_tags']); NOT_FIXABLE when every refusing step is; "
                                  "a caller that passes no worst gets the bare row's default (FIXABLE, waypoints)",
    "base_stop.open_content":     "NOT_FIXABLE (11-17): the open-content limit of the tray at the BASE level's acceleration (pick_level row base_open_content); the base level is not a plan choice",
    "landing.not_supported":      "FIXABLE: a `hold` landing (held in the arms, not on a support; USER 11-17); knobs recovery skill",
    "base_stop.<checker>":        "NOT_FIXABLE: `base_stop.in_route` | `at_rest` | `tip` | `torque` (the checker's speed / stop model; MISSING without the MEASURED stop card)",
    "not_executable":             "NOT_FIXABLE (11-17): a step admitted ONLY by a program the executor cannot run yet (run_cell refusal, `executable` False)",
    "rule1_a_exceeds":            "NOT_FIXABLE: run_cell refusal 'a exceeds a_admit' (no rows)",
}


def _parse_t(name: str):
    m = re.search(r"@t=([\d.]+)", name)
    return float(m.group(1)) if m else None


def parent_info(name: str) -> dict:
    """{program, branch_t, what} of a merged row name `<program>:<what>[@t=<t>]/<kind>[seg]`; leaf names give program None."""
    head = name.split("/")[0]
    if ":" not in name or "/" not in name:
        return dict(program=None, branch_t=_parse_t(name), what=None)
    prog, _, what = head.partition(":")
    return dict(program=prog, branch_t=_parse_t(head), what=what.split("@")[0])


def leaves(row: dict) -> list:
    """the failing leaves of `row` (a Row dict): the sub-rows of a merged row's `worst` with ok False, else the row itself.  -> [dict(name, guarantee, why, parent)]."""
    w = row.get("worst")
    subs = [(k, v) for k, v in (w or {}).items() if isinstance(v, dict) and "ok" in v and "guarantee" in v] if isinstance(w, dict) else []
    if not subs:
        return [dict(name=row.get("name", ""), guarantee=row.get("guarantee", ""), why=row.get("why") or "", parent=row.get("name", ""), worst=w if isinstance(w, dict) else None)]
    out = [dict(name=k, guarantee=v.get("guarantee", ""), why=v.get("why") or "", parent=row.get("name", ""), worst=v) for k, v in subs if not v.get("ok")]
    return out or [dict(name=row.get("name", ""), guarantee=row.get("guarantee", ""), why=row.get("why") or "", parent=row.get("name", ""))]


_RECOV = {"stop", "hold_after_stop", "retrace", "advance"}


def _next_step_tag(worst, why: str = "") -> Tag:
    """base_stop.next_step: the most fixable tag among the refusing later steps (worst["step_tags"] = [dict(step, kind, knobs, key)]); NOT_FIXABLE when all are; no info -> FIXABLE (waypoints)."""
    sts = [t for t in ((worst or {}).get("step_tags") or []) if isinstance(t, dict) and t.get("kind") in _ORDER]
    if not sts:
        m = re.search(r"\[most fixable refused step: (\w+)\]", why or "")        # callers that pass no worst: the row's why carries the kind
        if m and m.group(1) in _ORDER:
            return Tag(m.group(1), (K_WAYPTS,) if m.group(1) == FIXABLE else (), f"most fixable of the refusing later steps: {m.group(1)}", key="base_stop.next_step")
        return Tag(FIXABLE, (K_WAYPTS,), "the plan's later steps (each refused by its own rows)", key="base_stop.next_step")
    best = min(sts, key=lambda t: _ORDER[t["kind"]])
    return Tag(best["kind"], tuple(best.get("knobs") or ()), f"most fixable of the refusing later steps ({best.get('step', '?')}: {best['kind']})", key="base_stop.next_step")


def tag_leaf(name: str, guarantee: str = "", why: str = "", worst=None) -> Tag:
    leaf = name.split("/")[-1]
    base = leaf.split("[")[0]
    arg = leaf[len(base) + 1:].rstrip("]") if "[" in leaf else ""
    miss = guarantee == "missing"
    if base == "landing.not_supported":
        return Tag(FIXABLE, (K_RECOV,), "held in the arms is not on a support (USER 11-17): the plan's recovery choice", key=base)
    if base == "not_executable":
        return Tag(NOT_FIXABLE, (), "admitted only by a program the executor cannot run yet (read_then_retrace has no executor phase): not a plan choice", key="not_executable")
    if base == "base_stop.open_content":
        return Tag(NOT_FIXABLE, (), "open-content limit at the base level's acceleration (base level is not a plan choice)", key=base)
    if miss and base.startswith("base_stop.") and "card" in (why or "").lower():
        return Tag(NOT_FIXABLE, (), "missing: measured stop card", key="missing:measured stop card")
    if base == "base_stop.next_step" and not miss:
        return _next_step_tag(worst, why)
    if base in ("base_stop.recovery", "base_stop.continuation", "base_stop.landing_hold", "base_stop.goal_support", "base_stop.next_step"):
        kn = {"base_stop.recovery": (K_RECOV,), "base_stop.continuation": (K_WAYPTS,), "base_stop.landing_hold": (K_WAYPTS,), "base_stop.goal_support": (K_PLACE,), "base_stop.next_step": (K_WAYPTS,)}[base]
        return Tag(FIXABLE, kn, {"base_stop.recovery": "the base step's recovery skill", "base_stop.continuation": "the plan's route (sweep from the landed pose)",
                                 "base_stop.landing_hold": "the hold pose (waypoints) carrying the payload at rest (prefix row)",
                                 "base_stop.goal_support": "the goal placement (support at the goal)", "base_stop.next_step": "the plan's later steps (each refused by its own rows)"}[base], key=base)
    if miss:
        return Tag(NOT_FIXABLE, (), f"no model for this row ({why[:160]})" if why else "no model for this row", key=f"missing:{base}")
    if base == "collision":
        m = _COLL.search(why or "")
        if m:
            swept = m.group("kind") == "interval"
            return Tag(FIXABLE_GEOMETRY if swept else FIXABLE, (K_PLACE, K_LIFT),
                       "swept bound below d_min (the plan cannot set speed: only the geometry moves it)" if swept else "a sampled pose is below d_min",
                       key=f"collision_{'swept' if swept else 'sample'}:{m.group('cls')}", pair=m.group("pair"), distance_m=float(m.group("d")),
                       d_min_m=float(m.group("dmin")), swept_bound_m=float(m.group("lb")))
        swept = "swept bound" in (why or "")
        return Tag(FIXABLE_GEOMETRY if swept else FIXABLE, (K_PLACE, K_LIFT), "collision clearance below d_min", key=f"collision_{'swept' if swept else 'sample'}:any")
    if base in ("support_dest", "S"):
        return Tag(FIXABLE, (K_PLACE,), "the landing is not supported for every contents vertex", key="support_dest")
    if base in ("torque_payload", "rest_payload", "hold_poses"):
        return Tag(FIXABLE, (K_WAYPTS,), "torque / payload on the plan's path (a heavy bin may fail on every path)", key="torque_payload")
    if base == "open_content" and arg not in ("stop", "move"):
        return Tag(FIXABLE, (K_WAYPTS,), "open-content limit on the plan's path (a heavy bin may fail on every path)", key="open_content_path")
    if base in _RECOV or (base == "open_content" and arg in ("stop", "move")):
        which = leaf if base != "open_content" else f"open_content[{arg}]"
        return Tag(FIXABLE, (K_RECOV,), "a row of this step's recovery program (stop / hold / retrace / advance torque or open content)", key=f"recovery_torque:{which}")
    if base == "refuse":
        return Tag(FIXABLE, (K_WAYPTS, K_RECOV), "no acceleration above A_MIN is admitted on this reading bin (payload along the retrace path)", key="bin_refuse")
    if base == "R_K":
        return Tag(FIXABLE, (K_WAYPTS, K_PLACE), "the planner's remaining plan is refused on the landing's set", key="witness")
    if base in ("joint_pos", "joint_speed", "reach", "closed_chain") or "ik" in base.split("_"):
        return Tag(FIXABLE, (K_WAYPTS,), "joint limit / reach / IK row", key="joint_reach_ik")
    if base == "plan_shape":
        return Tag(FIXABLE, (K_PLACE,), "goal miss of the plan", key="plan_shape")
    if base == "dynamic":
        return Tag(NOT_FIXABLE, (), "sampled worlds run with the executor's stored backups: the plan does not change them", key="dynamic")
    if leaf.endswith(":coverage") or base == "coverage" or leaf.endswith(":theta_t"):
        return Tag(NOT_FIXABLE, (), "reading-bin coverage (ASSUMED bin width) or an empty theta set: not a plan choice", key="coverage")
    if base == "obj_tilt":
        return Tag(NOT_FIXABLE, (), "tilt table is measured at the checker's accelerations", key="tilt")
    if base in ("base_stop.in_route", "base_stop.at_rest", "base_stop.tip", "base_stop.torque"):
        return Tag(NOT_FIXABLE, (), "the checker's speed / stop model (base level, measured stop overshoot / settle, tip / torque at the level)", key=base)
    if base.startswith("base_") or leaf == "recovery[base_move]" or base == "recovery":
        return Tag(NOT_FIXABLE, (), "base level / base speed / no stored base recovery: not a plan choice", key="base")
    return Tag(UNTAGGED, (), f"row {name!r} not recognised", key=f"untagged:{base}")


_ORDER = {UNTAGGED: 0, FIXABLE: 1, FIXABLE_GEOMETRY: 2, NOT_FIXABLE: 3}


_SEV = {FIXABLE: 1, FIXABLE_GEOMETRY: 2, NOT_FIXABLE: 3}


def _severest(ts):
    """UNTAGGED if any leaf is (the gate), else the most severe kind (FIXABLE only when ALL are FIXABLE)."""
    for t in ts:
        if t.kind == UNTAGGED:
            return t
    return max(ts, key=lambda t: _SEV[t.kind])


def tag(row: dict) -> Tag:
    """tag one Row dict.  A merged row is tagged by its failing leaves: UNTAGGED if any leaf is, else the MOST SEVERE one (reviewer F7: a row is FIXABLE only when every failing
    leaf is; fixing one leaf does not pass a row whose other leaf is not fixable)."""
    return _severest([tag_leaf(lf["name"], lf["guarantee"], lf["why"], lf.get("worst")) for lf in leaves(row)])


def step_tag(rec: dict) -> Tag:
    """tag a refused STEP record of rule1_contents.run_cell (11-17, reviewer F7).  Per program: the failing leaves of its failing rows (rec["per_program"][p]["failing"]; a record with no
    per_program: rec["rows"] as one program) -> the program's kind = _severest of the leaves.  The step is FIXABLE only if at least one program's failing leaves are ALL fixable;
    otherwise the most severe program kind.  No failing rows at all -> NOT_FIXABLE (nothing the plan can change is named)."""
    progs = [pp.get("failing", []) for pp in (rec.get("per_program") or {}).values()]
    if not any(progs):
        progs = [rec.get("rows") or []]                    # no program failing row (e.g. admitted only by a non-executable program, or a record with no per_program): the record's own rows
    kinds = []
    for rows in progs:
        ts = [tag_leaf(lf["name"], lf["guarantee"], lf["why"], lf.get("worst")) for r in rows for lf in leaves(r)]
        if ts:
            kinds.append(_severest(ts))
    if not kinds:
        return Tag(NOT_FIXABLE, (), "no failing row named", key="step:no_rows")
    for t in kinds:
        if t.kind == FIXABLE:
            return t
    return _severest(kinds)


def row_names_untagged(rows) -> list:
    """names of the rows (leaves) in `rows` (Row dicts) that tag as UNTAGGED (the owner's gate)."""
    return [lf["name"] for r in rows for lf in leaves(r) if tag_leaf(lf["name"], lf["guarantee"], lf["why"], lf.get("worst")).kind == UNTAGGED]
