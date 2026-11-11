"""stack/proposer/compile.py -- a validated proposer plan -> what the pipeline executes.  SIMULATION ONLY.   (11-05)

    compile_plan(plan, task, world=None, build=True) -> CompiledPlan      (raises schema.PlanRejected stage "compile" | "planner")

The OBJECT waypoint path of the steps goes to the closed-chain planner as a WaypointPlan (stack/planners/closed_chain.py), in place of its fixed GOAL_DXY / lift-carry-place
shape.  Segment names follow the steps: "s<i>_<skill>" per motion step, plus "release".  The step's first recovery becomes that segment's stored backup
(closed_chain.backup_phases_for: set_down | retrace | hold | let_go); later alternatives are recorded in `unexecuted` (carry runs one backup per segment today).
The fallback becomes the phases run when the plan is refused (fallback_phases).

Checks done here (MODEL = scene geometry, scenes.support_tops; ASSUMED = a number we chose, listed in ASSUMPTIONS):
  * place(support): descent to the support top; the object centre must be over the support's xy extent
  * the object never goes below its start support top; horizontal motion needs MIN_CLEAR_M clearance above it at both ends (no dragging)  [MIN_CLEAR_M ASSUMED]
  * every motion has length >= MIN_MOVE_M  [ASSUMED]
  * the plan ends on the task's goal support within GOAL_TOL_M of the goal xy  (prompt.task_spec)
  * table-behind tasks (schema.BASE_TASKS, design 2026-11-09 section 2): lift -> optional move -> base_move -> optional move -> place(goal support) -> release.  base_move needs
    the object >= MIN_CLEAR_M above counter 1 (no dragging), a `level` (caller: adapters.base_tip_model.pick_level; None -> rejected "no base level passes the tip row").
    Arm numbers after the base move are WORLD translations; they are rotated into the base frame for the WaypointPlan (deltas are arm-space, base frame).  The goal check
    maps the object's base-frame position through the NOMINAL end pose of the route (sim truth end pose: no odometry error, ASSUMED A5).  The sweep / height / reach checks
    run in the planner (closed_chain.geometry_for -> base_route.check).
  * build=True: the planner builds it (SRS IK, joint limits, shelf collision, TOPP-RA); PlanRefused -> plan_rejected stage "planner" with the planner's reason
Not checked here (the checker's job, not built): torque / spill / support rows for the plan (admission today = carry_cert.static_cell over the compiled segments),
and whether a recovery's claimed `covers` holds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.stack.planners import base_route as BR
from world_model.v2.stack.planners import closed_chain as CCP
from world_model.v2.stack.proposer import prompt as PR
from world_model.v2.stack.proposer import schema as S

MIN_CLEAR_M = 0.02          # ASSUMED [m]: height of the object underside above its start support for any horizontal motion (the planner's own shelf clearance CLEAR_M is separate)
MIN_MOVE_M = 0.005          # ASSUMED [m]: shortest motion (a shorter one is degenerate for the timing)
Z_EPS = 1e-6
ASSUMPTIONS = dict(MIN_CLEAR_M=dict(tag="ASSUMED", value=MIN_CLEAR_M, src="compile.py: horizontal motion clearance above the start support"),
                   MIN_MOVE_M=dict(tag="ASSUMED", value=MIN_MOVE_M, src="compile.py: shortest motion"),
                   GOAL_TOL_M=dict(tag="ASSUMED", value=PR.GOAL_TOL_M, src="decision 11-03 goal tolerance 3 cm"),
                   ranges=dict(tag="ASSUMED", value=S.RANGE_TAG, src="schema.STEP_SKILLS"))


@dataclass
class CompiledPlan:
    task: str
    wp: CCP.WaypointPlan
    steps: list                                    # per motion step: dict(index, name, skill, args, delta, backup, alternatives, unexecuted)
    fallback: tuple                                # recovery skills (names) run when the plan is refused
    belief: dict                                   # B
    goal: dict                                     # task goal and where the plan ends
    plan_id: str = ""
    planner: dict = field(default_factory=dict)    # geometry summary (dy, margins) when built
    assumptions: dict = field(default_factory=lambda: dict(ASSUMPTIONS))

    @property
    def names(self):
        return list(self.wp.names)

    def job_plan(self, outside_b="refuse"):
        """the dict probe._e2e_job takes (picklable)."""
        return dict(wp=self.wp, fallback=tuple(self.fallback), belief=self.belief, outside_b=outside_b, plan_id=self.plan_id)

    def to_json(self):
        return dict(task=self.task, wp=self.wp.key_data(level=True), steps=self.steps, fallback=list(self.fallback), belief=self.belief, goal=self.goal,
                    plan_id=self.plan_id, planner=self.planner, assumptions=self.assumptions)

    @classmethod
    def from_json(cls, d):
        w = d["wp"]
        wp = CCP.WaypointPlan.from_data(w)
        return cls(d["task"], wp, d["steps"], tuple(d["fallback"]), d["belief"], d["goal"], d.get("plan_id", ""), d.get("planner", {}), d.get("assumptions", {}))


def _reject(reasons):
    raise S.PlanRejected("compile", reasons)


def compile_plan(plan, task, world=None, build=True, level=None):
    """plan: a status-ok plan dict (propose()) or a schema.TaskPlan.  -> CompiledPlan.
    level: the base speed level of a base_move step ("L1" ...), chosen by the caller (adapters.base_tip_model.pick_level(task, spec)); None with a base_move -> PlanRejected
    ("planner": no base level passes the tip row).  Ignored for fixed-base plans."""
    if isinstance(plan, dict):
        if plan.get("status") != "ok":
            raise S.PlanRejected(plan.get("stage", "schema"), plan.get("reasons", ["not an ok plan"]))
        spec0 = PR.task_spec(task, world)
        tp = S.validate(S.reply_of(plan), task, tuple(spec0["supports"]))
        pid = plan.get("plan_id") or S.plan_id(tp.to_dict())
    else:
        tp, pid = plan, S.plan_id(plan.to_dict())
    spec = PR.task_spec(task, world)
    if task in S.BASE_TASKS:
        return _compile_behind(tp, task, world, build, level, pid, spec)
    sup = spec["supports"]
    t0 = sup["counter"]["top"]                      # the object starts on the counter
    sx, sy = spec["start_xy"]
    pos = np.zeros(3)
    names, deltas, backups, steps = [], [], [], []
    errs = []
    n_motion = len(tp.steps) - 1                    # the last step is release (schema)
    for i, st in enumerate(tp.steps[:n_motion]):
        sk, a = st.call.skill, st.call.args
        if sk == "lift":
            d = np.array([0.0, 0.0, a["dz"]])
        elif sk == "move":
            d = np.array([a["dx"], a["dy"], a["dz"]])
        else:                                       # place
            top = sup[a["support"]]["top"]
            d = np.array([0.0, 0.0, top - (t0 + pos[2])])
            cx, cy = sx + pos[0], sy + pos[1]
            s = sup[a["support"]]
            if not (s["x"][0] <= cx <= s["x"][1] and s["y"][0] <= cy <= s["y"][1]):
                errs.append(f"step {i} place({a['support']}): the object centre ({cx:.3f}, {cy:.3f}) is not over the support (x {s['x']}, y {s['y']})")
            if d[2] > -MIN_MOVE_M:
                errs.append(f"step {i} place({a['support']}): the object underside is at {t0 + pos[2]:.3f} m, the support top at {top:.3f} m; nothing to lower (needs >= {MIN_MOVE_M} m)")
        new = pos + d
        if np.linalg.norm(d) < MIN_MOVE_M:
            errs.append(f"step {i} {sk}: motion {np.linalg.norm(d):.4f} m shorter than {MIN_MOVE_M} m")
        horiz = float(np.hypot(d[0], d[1])) > 1e-9
        if new[2] < -Z_EPS and not (sk == "place"):
            errs.append(f"step {i} {sk}: the object would go to {new[2]:.3f} m relative to its start support (below it)")
        if horiz and (pos[2] < MIN_CLEAR_M - Z_EPS or new[2] < MIN_CLEAR_M - Z_EPS):
            errs.append(f"step {i} {sk}: horizontal motion with the object {min(pos[2], new[2]):.3f} m above its start support; needs >= {MIN_CLEAR_M} m at both ends (no dragging)")
        name = f"s{i}_{sk}"
        rc = st.recoveries
        names.append(name)
        deltas.append(tuple(round(float(x), 9) for x in d))
        backups.append(rc[0].skill)
        steps.append(dict(index=i, name=name, skill=sk, args=st.call.args, delta=[float(x) for x in d], backup=rc[0].skill, why=st.why,
                          alternatives=[dict(skill=r.skill, covers=r.covers) for r in rc], unexecuted=[dict(skill=r.skill, covers=r.covers) for r in rc[1:]]))
        pos = new
    end = [float(sx + pos[0]), float(sy + pos[1])]
    goal = dict(goal_xy=spec["goal_xy"], goal_support=spec["goal_support"], goal_tol_m=spec["goal_tol_m"], end_xy=end,
                end_support=tp.steps[n_motion - 1].call.args.get("support") if n_motion else None,
                miss_m=float(np.hypot(end[0] - spec["goal_xy"][0], end[1] - spec["goal_xy"][1])))
    if goal["end_support"] != spec["goal_support"]:
        errs.append(f"the plan places on {goal['end_support']!r}, the task goal is on {spec['goal_support']!r}")
    if goal["miss_m"] > spec["goal_tol_m"]:
        errs.append(f"the plan ends at xy ({end[0]:.3f}, {end[1]:.3f}), {goal['miss_m']:.3f} m from the goal ({spec['goal_xy'][0]}, {spec['goal_xy'][1]}); tolerance {spec['goal_tol_m']} m")
    if errs:
        _reject(errs)
    wp = CCP.WaypointPlan(tuple(names), tuple(deltas), tuple(backups), True)
    cp = CompiledPlan(task, wp, steps, tuple(r.skill for r in tp.fallback), tp.belief, goal, pid)
    if build:
        build_geometry(cp, world)
    return cp


def _compile_behind(tp, task, world, build, level, pid, spec):
    be = CCP._BASE_BACKEND or BR.default_backend()
    w = world if world is not None else be.world_default(task)
    sup = spec["supports"]
    t0 = sup["counter"]["top"]
    sx, sy = spec["start_xy"]
    p0 = be.start_pose(w)                                       # (x, y, yaw) of the base rotation centre at the start (MODEL)
    c0, yaw0 = np.array(p0[:2], float), float(p0[2])
    pb0 = BR.rot(yaw0).T @ (np.array([sx, sy]) - c0)            # object reference in the base frame
    cur = tuple(p0)
    pos = np.zeros(3)                                           # arm-space offset (base frame, start-aligned) of the object
    names, deltas, backups, steps, bases = [], [], [], [], []
    errs = []
    wxy = lambda: cur_xy(cur, pb0, pos)
    n_motion = len(tp.steps) - 1
    for i, st in enumerate(tp.steps[:n_motion]):
        sk, a = st.call.skill, st.call.args
        rc = st.recoveries
        if sk == "base_move":
            route = BR.normalize_route(a["route"])
            if level is None:
                errs.append(f"step {i} base_move: no base level passes the tip row (level=None); the planner cannot time the route")
            if pos[2] < MIN_CLEAR_M - Z_EPS:
                errs.append(f"step {i} base_move: the object is {pos[2]:.3f} m above counter 1; the base must not move with the object closer than {MIN_CLEAR_M} m to the support (no dragging; lift first)")
            if bases:
                errs.append(f"step {i} base_move: a second base_move (one per plan)")
            if not names:
                errs.append(f"step {i} base_move: before any arm move (lift the object first)")
            bases.append((len(names), route, str(level)))
            cur = BR.route_end(cur, route)
            steps.append(dict(index=i, name=f"s{i}_base_move", skill=sk, args=dict(route=[[k, v] for k, v in route]), delta=[0.0, 0.0, 0.0], backup=rc[0].skill, why=st.why,
                              alternatives=[dict(skill=r.skill, covers=r.covers) for r in rc], unexecuted=[dict(skill=r.skill, covers=r.covers) for r in rc[1:]],
                              base_end_pose=[float(x) for x in cur], level=level))
            continue
        dyaw = cur[2] - yaw0                                    # body rotation since the start: world translations -> start-aligned base-frame translations
        if sk == "lift":
            dw = np.array([0.0, 0.0, a["dz"]])
        elif sk == "move":
            dw = np.array([a["dx"], a["dy"], a["dz"]])
        else:                                                   # place
            top = sup[a["support"]]["top"]
            dw = np.array([0.0, 0.0, top - (t0 + pos[2])])
            cx, cy = wxy()
            s_ = sup[a["support"]]
            if not (s_["x"][0] <= cx <= s_["x"][1] and s_["y"][0] <= cy <= s_["y"][1]):
                errs.append(f"step {i} place({a['support']}): the object centre ({cx:.3f}, {cy:.3f}) is not over the support (x {s_['x']}, y {s_['y']})"
                            + ("" if bases else "; the plan has no base_move: the goal is behind the robot"))
            if dw[2] > -MIN_MOVE_M:
                errs.append(f"step {i} place({a['support']}): the object underside is at {t0 + pos[2]:.3f} m, the support top at {top:.3f} m; nothing to lower (needs >= {MIN_MOVE_M} m)")
        d = np.array([*(BR.rot(dyaw).T @ dw[:2]), dw[2]])
        new = pos + d
        if np.linalg.norm(d) < MIN_MOVE_M:
            errs.append(f"step {i} {sk}: motion {np.linalg.norm(d):.4f} m shorter than {MIN_MOVE_M} m")
        horiz = float(np.hypot(d[0], d[1])) > 1e-9
        if new[2] < -Z_EPS and not (sk == "place"):
            errs.append(f"step {i} {sk}: the object would go to {new[2]:.3f} m relative to its start support (below it)")
        if horiz and (pos[2] < MIN_CLEAR_M - Z_EPS or new[2] < MIN_CLEAR_M - Z_EPS):
            errs.append(f"step {i} {sk}: horizontal motion with the object {min(pos[2], new[2]):.3f} m above its start support; needs >= {MIN_CLEAR_M} m at both ends (no dragging)")
        name = f"s{i}_{sk}"
        names.append(name)
        deltas.append(tuple(round(float(x), 9) for x in d))
        backups.append(rc[0].skill)
        steps.append(dict(index=i, name=name, skill=sk, args=st.call.args, delta=[float(x) for x in d], delta_world=[float(x) for x in dw], backup=rc[0].skill, why=st.why,
                          alternatives=[dict(skill=r.skill, covers=r.covers) for r in rc], unexecuted=[dict(skill=r.skill, covers=r.covers) for r in rc[1:]]))
        pos = new
    end = [float(x) for x in wxy()]
    goal = dict(goal_xy=spec["goal_xy"], goal_support=spec["goal_support"], goal_tol_m=spec["goal_tol_m"], end_xy=end,
                end_support=tp.steps[n_motion - 1].call.args.get("support") if n_motion else None,
                miss_m=float(np.hypot(end[0] - spec["goal_xy"][0], end[1] - spec["goal_xy"][1])), base_end_pose=[float(x) for x in cur],
                base_end_pose_tag="nominal route end (sim truth, no odometry error: ASSUMED A5)")
    if goal["end_support"] != spec["goal_support"]:
        errs.append(f"the plan places on {goal['end_support']!r}, the task goal is on {spec['goal_support']!r}")
    if goal["miss_m"] > spec["goal_tol_m"]:
        errs.append(f"the plan ends at world xy ({end[0]:.3f}, {end[1]:.3f}), {goal['miss_m']:.3f} m from the goal ({spec['goal_xy'][0]}, {spec['goal_xy'][1]}); tolerance {spec['goal_tol_m']} m")
    if errs:
        if level is None and bases:
            raise S.PlanRejected("planner", [e for e in errs if "level" in e] + ["reason=base_level"] + [e for e in errs if "level" not in e])
        _reject(errs)
    wp = CCP.WaypointPlan(tuple(names), tuple(deltas), tuple(backups), True, tuple(bases))
    cp = CompiledPlan(task, wp, steps, tuple(r.skill for r in tp.fallback), tp.belief, goal, pid)
    cp.assumptions = dict(cp.assumptions, base_route=dict(tag="ASSUMED", value=dict(MAX_SEGMENTS=BR.MAX_SEGMENTS, MAX_LINE_M=BR.MAX_LINE_M, MAX_TURN_DEG=BR.MAX_TURN_DEG,
                                                                                    STEP_M=BR.STEP_M, STEP_DEG=BR.STEP_DEG, GAP_S=BR.GAP_S, TRACK_MARGIN_M=BR.TRACK_MARGIN_M),
                                                          src="stack/planners/base_route.py (design 2026-11-09 A3, A7; tracking margin 0: no MEASURED base error wired)"))
    if build:
        build_geometry(cp, world)
    return cp


def cur_xy(cur, pb0, pos):
    """world xy of the object reference: base pose `cur` (x, y, yaw of the rotation centre) applied to its base-frame position."""
    return np.array(cur[:2], float) + BR.rot(cur[2]) @ (pb0 + pos[:2])


def build_geometry(cp, world=None):
    """run the closed-chain planner on the compiled waypoints (the geometry goes to the on-disk plan cache).  PlanRefused -> PlanRejected('planner', reason)."""
    inp = CCP._nominal_inputs(cp.task, cp.wp) if world is None else CCP.PlanInputs(*(_inputs(cp.task, world)), wp=cp.wp)
    try:
        g = CCP.geometry_for(inp)
    except CCP.PlanRefused as e:
        raise S.PlanRejected("planner", [str(e), f"reason={e.info.get('reason')}"] + ([f"arm={e.info['arm']}"] if "arm" in e.info else [])) from e
    cp.planner = dict(dy=g.info["dy"], z_c=g.info["z_c"], goal_xy=g.info["goal_xy"], margin_min=g.info["margin_min"], n_samples=g.info["n_samples"], key=inp.key())
    if "base" in g.info:
        cp.planner["base"] = g.info["base"]
    return cp


def _inputs(task, world):
    from world_model.v2.nero.contents import envelope as E, scenes as SC
    cap = json.loads(E.CAP_FILE.read_text())["pot"]
    if CCP.is_base_task(task):
        return (CCP._BASE_BACKEND or BR.default_backend()).compose(world), world, {s: cap["q"]["W0_liftoff"][s] for s in CCP.SIDES}, task
    return SC.compose(world), world, {s: cap["q"]["W0_liftoff"][s] for s in CCP.SIDES}, task


# ---- the fallback -------------------------------------------------------------------------------------------------------------------
def fallback_phases(kinds, ep):
    """phases run when the plan is refused (nothing has moved: the object is on its start support).  hold -> keep holding; let_go / set_down -> weld off + retreat of
    the claws (set_down = let_go here: already set down); retrace -> nothing to retrace.  [let_go] reproduces the default refuse phases of probe._e2e_job."""
    ph, released = [("hold", 0.3)], False
    for k in kinds:
        if k in ("let_go", "set_down") and not released:
            ph += [("weld", False), ep.retreat_phase()]
            released = True
        elif k == "hold" and not released:
            ph += [("hold", 0.5)]
    return ph + [("hold", 0.5)]


# ---- Rule 2: is the probe set inside B? -------------------------------------------------------------------------------------------------
def probe_in_belief(task, belief, c_lo, c_hi, tol=1e-9):
    """c_lo..c_hi = the probe's interval on the total contents mass [kg] (probe.py).  Only the probed belief parameter is checked (the others are not measured:
    listed in `not_probed`).  -> dict(param, probe=[lo, hi], B=[lo, hi], prior, B_eff, covers_prior, inside, not_probed).

    11-07 fix (clarity report 11-06 (f)1): B must lie inside the prior box Theta_0 (sets rule; bounds rule 4).  The model states B without
    seeing Theta_0 (prompt.py), and the schema checks B only against a wide sanity range, so the replies so far had B covering all of
    Theta_0.  The reply is never repaired: B is kept as written and the check uses B_eff = B intersected with Theta_0 (the prior box of the
    probed parameter, ASSUMED: probe.prior_contents minus the probe offset).  covers_prior = B_eff is all of Theta_0: then no sound probe set
    can fall outside B and Rule 2 cannot trigger; reports must say so instead of counting `inside`.  B disjoint from Theta_0 -> inside False."""
    if not S.legacy_v1():
        raise RuntimeError("probe_in_belief is the v1 Rule 2 (numeric belief); retired from the schema-v2 path (11-09): see nero/contents/probe_select.py (T1/T2/T3). "
                           "BIMANUAL_LEGACY_PROPOSER_V1=1 keeps it.")
    from world_model.v2.nero.contents import probe as PR
    name, off = S.probe_offset(task)
    lo, hi = float(c_lo) - off, float(c_hi) - off
    b = [float(x) for x in belief[name]]
    p0, p1 = (float(x) - off for x in PR.prior_contents(task))
    e0, e1 = max(b[0], p0), min(b[1], p1)
    disjoint = e0 > e1 + tol
    covers = (not disjoint) and e0 <= p0 + tol and e1 >= p1 - tol
    inside = (not disjoint) and lo >= e0 - tol and hi <= e1 + tol
    return dict(param=name, probe=[lo, hi], B=b, prior=[p0, p1], B_eff=None if disjoint else [e0, e1], covers_prior=bool(covers),
                inside=bool(inside), not_probed=[k for k in belief if k != name])
