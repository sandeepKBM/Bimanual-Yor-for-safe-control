"""stack/proposer/prompt.py -- the system and user prompt of the proposer, per task.  SIMULATION ONLY.   (11-05)

(Schema v2, 11-09: no hidden-parameter table at all; phenomenon cards + measuring-skill cards instead.  BIMANUAL_LEGACY_PROPOSER_V1=1 = the format described below.)
The prompt holds: the task instruction (destination = the TASK INPUT), the scene facts (supports, object size, grasp), the robot limits (config), the skill
vocabulary and the output format.  It holds NO hidden truth: no sampled world, no true contents mass / centre / fills, no prior box (bounds rule: the proposer
states its own belief B).  The text is deterministic (no times, no random ids), so the response cache replays offline.

Destinations: GOAL_DXY of stack/tasks.py (ASSUMED, user-confirmed 11-02) turned into a world xy -- the same destination the fixture plans reach, so
a model plan and the fixture plan are judged on one goal.  Goal tolerance GOAL_TOL_M = 0.03 m (decision 11-03 "goal tolerance 3 cm", ASSUMED).
"""
from __future__ import annotations

import json

import numpy as np

from world_model.v2.stack.proposer import schema as S

GOAL_TOL_M = 0.03                 # ASSUMED (11-03 handoff "goal tolerance 3 cm"): end xy of the plan vs the task goal xy


def _world(task, world=None):
    if world is not None:
        return world
    from world_model.v2.nero.contents import carry_cert as CC
    return CC.world_default(task)


def task_spec(task, world=None):
    """the task input: -> dict(instruction, start_xy, goal_xy, goal_support, goal_top, goal_tol_m, supports {name: dict(top, x, y)}, object {...}).  Frames: world."""
    from world_model.v2.nero.contents import objects as OB, scenes as SC
    from world_model.v2.stack.planners import closed_chain as CCP
    from world_model.v2.stack import tasks as TK
    if getattr(TK.TASKS.get(str(task)), "base", ""):
        return _task_spec_behind(task, world)
    w = _world(task, world)
    kind = CCP.kind_of(task)
    tops = SC.support_tops(w)
    gx, gy = TK.GOAL_DXY[kind]
    goal = [round(float(w.obj_xy[0] + gx), 4), round(float(w.obj_xy[1] + gy), 4)]
    goal_support = TK.of_kind(kind).goal_support if TK.of_kind(kind).goal_support in tops else "counter"
    sup = {k: dict(top=round(float(v["top"]), 4), x=[round(float(a), 4) for a in v["x"]], y=[round(float(a), 4) for a in v["y"]]) for k, v in tops.items()}
    if kind == "box":
        obj = dict(name="closed crate", footprint_xy_m=[w.box_outer[0], w.box_outer[1]], height_m=w.box_outer[2], empty_mass_kg=OB.BOX["mass"],
                   grasp="end tabs on the +y and -y sides, tab centre %.4f m above the support" % SC.GRASP_Z)
        text = (f"Carry the closed crate (it holds hidden contents) from where it stands on the counter and put it on the shelf at xy = ({goal[0]}, {goal[1]}) "
                f"(object centre), then let go.")
    elif kind == "tray":
        n = int(task[4:])
        obj = dict(name=f"serving tray with {n} mugs of coffee", footprint_xy_m=[OB.TRAY["lx"], OB.TRAY["ly"]], rim_height_m=OB.TRAY["rim_h"],
                   empty_mass_kg=OB.TRAY["mass"] + n * OB.CUP["empty"], mugs=dict(n=n, diameter_m=OB.CUP["D"], height_m=OB.CUP["H"], empty_kg_each=OB.CUP["empty"],
                                                                                 capacity_kg_each=OB.CUP["fill_max_kg"]),
                   grasp="end tabs on the +y and -y sides, tab centre %.4f m above the support" % SC.GRASP_Z)
        text = (f"Carry the serving tray with {n} mugs of coffee (the fill of each mug is hidden) from where it stands on the counter to the counter at "
                f"xy = ({goal[0]}, {goal[1]}) (object centre) and set it down there, then let go.  Do not spill.")
    else:
        r = SC.PS.ASSUMED["pan_radius"] + SC.PS.ASSUMED["pan_wall_t"]
        obj = dict(name="covered cooking pot with a loose lid", footprint_radius_m=round(float(r), 4), empty_mass_kg=round(float(SC.pot_empty_props()[0]), 3),
                   grasp="two side handles along the y axis (handle length %.2f m)" % SC.POT_HANDLE_LEN)
        text = (f"Carry the covered pot (it holds hidden contents) from where it stands on the counter to the counter at xy = ({goal[0]}, {goal[1]}) "
                f"(object centre) and set it down there, then let go.  The lid is loose: do not let it slide off.")
    obj["start_xy"] = [round(float(w.obj_xy[0]), 4), round(float(w.obj_xy[1]), 4)]
    obj["start_support"] = "counter"
    return dict(task=task, instruction=text, start_xy=obj["start_xy"], goal_xy=goal, goal_support=goal_support, goal_top=sup[goal_support]["top"],
                goal_tol_m=GOAL_TOL_M, supports=sup, object=obj)


def _task_spec_behind(task, world=None):
    """table-behind tasks (tray6_behind, box_behind; docs/status/table_behind_design_2026-11-09.md section 2 "Proposer prompt").  Goal = TaskDef.goal_dxy on TaskDef.goal_support
    (counter2), world frame.  Two counters with their world extents (MODEL: nero/base/scene mirror), the base start pose, route R1 as the known-good route, the one scene fact about turning at the
    pick pose, the planner's refusal rule.  NO contents prior and no sanity ranges (user 11-08: skills supply numbers, the model names phenomena)."""
    from world_model.v2.nero.contents import objects as OB, scenes as SC
    from world_model.v2.stack import tasks as TK
    from world_model.v2.stack.planners import base_route as BR
    td = TK.task(str(task))
    be = BR.default_backend()
    w = world if world is not None else be.world_default(task)
    sup = {k: dict(top=round(float(v["top"]), 4), x=[round(float(a), 4) for a in v["x"]], y=[round(float(a), 4) for a in v["y"]]) for k, v in be.supports(w).items()}
    gx, gy = td.goal_dxy
    goal = [round(float(w.obj_xy[0] + gx), 4), round(float(w.obj_xy[1] + gy), 4)]
    p0 = be.start_pose(w)
    r1 = [{k: v} for k, v in BR.default_route(td)]
    if td.kind == "box":
        obj = dict(name="closed crate", footprint_xy_m=[w.box_outer[0], w.box_outer[1]], height_m=w.box_outer[2], empty_mass_kg=OB.BOX["mass"],
                   grasp="end tabs on the +y and -y sides, tab centre %.4f m above the support" % SC.GRASP_Z)
        what = "the closed crate (it holds hidden contents)"
        tail = ""
    else:
        n = td.n_cups
        obj = dict(name=f"serving tray with {n} mugs of coffee", footprint_xy_m=[OB.TRAY["lx"], OB.TRAY["ly"]], rim_height_m=OB.TRAY["rim_h"],
                   empty_mass_kg=OB.TRAY["mass"] + n * OB.CUP["empty"], mugs=dict(n=n, diameter_m=OB.CUP["D"], height_m=OB.CUP["H"], empty_kg_each=OB.CUP["empty"],
                                                                                 capacity_kg_each=OB.CUP["fill_max_kg"]),
                   grasp="end tabs on the +y and -y sides, tab centre %.4f m above the support" % SC.GRASP_Z)
        what = f"the serving tray with {n} mugs of coffee (the fill of each mug is hidden)"
        tail = "  Do not spill."
    text = (f"Carry {what} from where it stands on counter 1 (xy = ({round(float(w.obj_xy[0]), 4)}, {round(float(w.obj_xy[1]), 4)}), object centre) to counter 2 BEHIND the robot, "
            f"at xy = ({goal[0]}, {goal[1]}) (object centre), set it down there and let go.{tail}  "
            f"The robot base can drive: the step skill base_move(route) moves the whole robot, the arms hold still and keep carrying the object during it; "
            f"the robot's +x axis turns with the base (so after a 180 degree turn the robot faces -x).  base_move is a motion step like lift and place: give it a recovery too.  "
            f"The planner refuses a base route that sweeps the base, arms or object into a counter or table.  "
            f"Known-good route R1 (it sweeps clear): {json.dumps(r1)}.  Turning in place at the pick pose hits counter 1.  Place only on counter 2 (support name \"counter2\").")
    obj["start_xy"] = [round(float(w.obj_xy[0]), 4), round(float(w.obj_xy[1]), 4)]
    obj["start_support"] = "counter"
    base = dict(start_pose_xy_yaw_deg=[round(float(p0[0]), 4), round(float(p0[1]), 4), round(float(np.degrees(p0[2])), 2)],
                note="pose of the base rotation centre; in-place turns turn about it; a line moves along the robot's CURRENT +x axis (negative = backwards)",
                routes_known_good={td.routes[0][0]: r1}, supports_world_frame_xy_m=dict(counter_1=sup["counter"], counter_2=sup["counter2"]))
    return dict(task=task, instruction=text, start_xy=obj["start_xy"], goal_xy=goal, goal_support=td.goal_support, goal_top=sup[td.goal_support]["top"],
                goal_tol_m=GOAL_TOL_M, supports=sup, object=obj, base=base)


def robot_limits():
    """datasheet / config numbers shown to the model (nero/config.py is the one source; nothing is copied into this file)."""
    from world_model.v2.nero import config as C
    tl = np.asarray(C.tau_limit(), float)
    return dict(arms=2, arm_dof=7, payload_kg_per_arm_datasheet=C.MANUAL["payload_kg"], joint_speed_deg_s=list(C.MANUAL["joint_speed_deg_s"]),
                joint_torque_limit_Nm=[round(float(x), 2) for x in tl], reach_m_datasheet=C.MANUAL["reach_m"])


def scene_facts(task, world=None):
    """everything the model may know about the scene (JSON-able, deterministic).  No hidden truth.  Schema v2 (default): no hidden-parameter table, no sanity ranges, no
    contents prior -- the phenomenon cards and the built-skill cards instead (skills/catalog.cards).  BIMANUAL_LEGACY_PROPOSER_V1=1: the 11-05..11-09 facts."""
    sp = task_spec(task, world)
    behind = task in S.BASE_TASKS
    if behind:
        frame = ("world frame, metres, z up.  The left arm holds the +y handle/tab, the right arm the -y one (at the start; they turn with the base).  Arm-step numbers "
                 "(lift, move, place) are world-frame translations of the object; the robot's +x axis is the world +x axis at the start.")
    else:
        frame = "world frame, metres, z up.  Decreasing x moves the object toward the robot.  The left arm holds the +y handle/tab, the right arm the -y one."
    common = dict(task=sp, robot=robot_limits(), frame=frame,
                  grasp="Both claws are rigidly welded to the object at its handles/tabs for the whole motion (ASSUMED: grasping is not part of this task).  The object stays level.",
                  vocabulary=S.vocabulary(tuple(sp["supports"]), base=behind))
    if not S.legacy_v1():
        from world_model.v2.stack.skills import catalog as CAT
        c = CAT.cards(task)
        return dict(common, phenomena=c["phenomena"], measuring_skills=c["skills"])
    P = S.params(task)
    if behind:                                    # 11-08: no sanity ranges / contents prior in the table-behind prompt (validation still rejects a belief outside them, with the reason)
        hp = {k: dict(unit=v["unit"], meaning=v["doc"]) for k, v in P.items()}
    else:
        hp = {k: dict(unit=v["unit"], sanity_range=[round(float(v["range"][0]), 4), round(float(v["range"][1]), 4)], meaning=v["doc"]) for k, v in P.items()}
    return dict(common, hidden_parameters=hp)


SYSTEM_V1 = """You are the high-level planner of a two-arm robot (two 7-DoF NERO arms) in a simulation. You write the plan; a separate low-level planner turns it into joint motions and a checker verifies it before anything moves. You never control joints.

You must return ONE JSON object and nothing else (no prose outside it, no code fence needed). Format:
{"belief": {"<hidden parameter>": [lo, hi], ...},
 "steps": [{"skill": "<name>", "args": {...}, "why": "<one short sentence>"}, ...],
 "recoveries": {"<step index>": [{"skill": "<recovery skill>", "args": {}, "covers": {"<hidden parameter>": [lo, hi]}}, ...], ...},
 "fallback": [{"skill": "<recovery skill>", "args": {}}, ...]}

Rules:
1. belief: give a range [lo, hi] for EVERY hidden parameter listed in the scene facts, wide enough that you believe the true value lies inside it. The robot will measure the load with a probe before moving; if the measurement falls outside your belief you will be asked again, so be honest about what you do not know.
2. steps: a sequence of SEGMENTS, one object-level motion each (lift, move, place), then release as the LAST step. Exactly one place, directly before release. Use only the skills and argument ranges listed in the scene facts. All numbers are metres in the world frame.
3. recoveries: for EVERY motion step (lift, move, place) give at least one recovery skill that is safe to run if the step has to be abandoned halfway (hold, retrace, set_down, let_go). Keys are the 0-based step indices as strings. "covers" is optional: the part of your belief the recovery is written for (a sub-range of the belief). Only the first recovery of a step is executed today; list the best one first. Letting go in the air drops the object.
4. fallback: 1 to 3 recovery skills to run if the whole plan is not admitted (nothing has moved yet; the object is still on its start support).
5. The object must clear everything it passes over, must end within the goal tolerance of the task goal on the named support, and must never be dragged along a support: horizontal motion needs height above the support.
6. You do not set speeds or accelerations; the checker admits or slows the plan.
7. Unknown skills, extra keys, missing keys or numbers outside the ranges make the whole reply invalid; it is then rejected, not repaired."""


SYSTEM_V2 = """You are the high-level planner of a two-arm robot (two 7-DoF NERO arms) in a simulation. You write the plan; a separate low-level planner turns it into joint motions and a checker verifies it before anything moves. You never control joints.

The object holds hidden contents. You do NOT estimate any number about them. Instead you name the physical effects (phenomena) of the hidden contents that matter for this task, and say how each one is handled: measured by one of the robot's measuring skills, or not measurable. The robot's measuring skills produce the numbers; the checker uses them.

You must return ONE JSON object and nothing else (no prose outside it, no code fence needed). Format:
{"phenomena": [{"id": "<phenomenon id from the cards>", "why": "<one short sentence>", "handle": {"measure": "<measuring skill id>"}},
               {"id": "<phenomenon id from the cards>", "why": "<one short sentence>", "handle": {"unmeasurable": "<one short sentence: why no skill measures it>"}},
               {"id": "NEW", "name": "<short name>", "why": "<one short sentence>", "handle": {"unmeasurable": "<one short sentence>"}}],
 "probes": [{"skill": "<measuring skill id>"}, ...],
 "steps": [{"skill": "<name>", "args": {...}, "why": "<one short sentence>"}, ...],
 "recoveries": {"<step index>": [{"skill": "<recovery skill>", "args": {}}, ...], ...},
 "fallback": [{"skill": "<recovery skill>", "args": {}}, ...]}

Rules:
1. phenomena: list every effect of the hidden contents that could make this carry fail, using the phenomenon ids on the cards. A phenomenon that is not on the cards gets the id "NEW" and a "name". Each phenomenon appears once. Its handle is EITHER {"measure": <id of a measuring skill whose card lists this phenomenon under "measures">} OR {"unmeasurable": <reason>}. Name a skill only if its card says it measures that phenomenon; a plan that names a skill for a phenomenon it does not measure is rejected.
2. probes: the measuring skills to run before the carry, in the order to run them, each once. Every skill in probes must be the handle of at least one phenomenon, and every skill used as a handle must be in probes. The robot runs only the probes you list, and the checker may skip a probe that its own limits do not allow.
3. No number about the contents anywhere (no masses, no ranges, no positions): numbers are allowed only in the arguments of steps. A reply with a number outside step arguments is rejected.
4. steps: a sequence of SEGMENTS, one object-level motion each (lift, move, place), then release as the LAST step. Exactly one place, directly before release. Use only the skills and argument ranges listed in the scene facts. All numbers are metres in the world frame.
5. recoveries: for EVERY motion step (lift, move, place, and base_move where offered) give at least one recovery skill that is safe to run if the step has to be abandoned halfway (hold, retrace, set_down, let_go). Keys are the 0-based step indices as strings. Only the first recovery of a step is executed today; list the best one first. Letting go in the air drops the object. Recoveries take no "covers" field.
6. fallback: 1 to 3 recovery skills to run if the whole plan is not admitted (nothing has moved yet; the object is still on its start support).
7. The object must clear everything it passes over, must end within the goal tolerance of the task goal on the named support, and must never be dragged along a support: horizontal motion needs height above the support.
8. You do not set speeds or accelerations; the checker admits or slows the plan.
9. Unknown skills, unknown phenomenon ids, extra keys, missing keys or numbers outside the ranges make the whole reply invalid; it is then rejected, not repaired."""


def system():
    """the system prompt of the current format (schema v2 unless BIMANUAL_LEGACY_PROPOSER_V1=1)."""
    return SYSTEM_V1 if S.legacy_v1() else SYSTEM_V2


def __getattr__(name):                        # PR.SYSTEM is read at call time (propose.py), so the env switch applies per call
    if name == "SYSTEM":
        return system()
    raise AttributeError(name)


def user_prompt(task, facts=None, world=None):
    f = facts if facts is not None else scene_facts(task, world)
    return (f"TASK_ID: {task}\n"
            f"TASK: {f['task']['instruction']}\n\n"
            f"SCENE FACTS (JSON):\n{json.dumps(f, sort_keys=True)}\n\n"
            "Return the JSON object now.")


def uncovered(belief, probe_set):
    """parts of the probe set outside B: {param: [[lo, hi], ...]}."""
    out = {}
    for k, (lo, hi) in probe_set.items():
        b = belief.get(k)
        if b is None:
            out[k] = [[lo, hi]]
            continue
        parts = []
        if lo < b[0] - 1e-9:
            parts.append([lo, min(hi, b[0])])
        if hi > b[1] + 1e-9:
            parts.append([max(lo, b[1]), hi])
        if parts:
            out[k] = parts
    return out


def reask_prompt(task, plan_dict, evidence, facts=None, world=None):
    """Rule 2 re-ask.  Schema v2 (default): `evidence` = the dict nero/contents/probe_select builds ({trigger: "T1"|"T2", readings, probes_skipped, failing_rows, worst,
    phenomena_shown, coverage}); see reask_evidence_block.  BIMANUAL_LEGACY_PROPOSER_V1=1: the v1 prompt, `evidence` = probe_set {param: [lo, hi]} of the probed parameter(s)
    (Rule 2 as of 11-05: the probe set is not inside B)."""
    if not S.legacy_v1():
        f = facts if facts is not None else scene_facts(task, world)
        return (f"TASK_ID: {task}\n"
                f"TASK: {f['task']['instruction']}\n\n"
                f"SCENE FACTS (JSON):\n{json.dumps(f, sort_keys=True)}\n\n"
                "YOUR PREVIOUS PLAN (JSON):\n" + json.dumps(S.reply_of(plan_dict), sort_keys=True) + "\n\n"
                + reask_evidence_block(evidence) + "\n"
                "Return a COMPLETE new plan in the same JSON format (phenomena, probes, steps, recoveries, fallback). Name the phenomena and the probes again; "
                "change the steps and recoveries to what the evidence above requires. Return the JSON object now.")
    probe_set = evidence
    f = facts if facts is not None else scene_facts(task, world)
    unc = uncovered(plan_dict["belief"], probe_set)
    return (f"TASK_ID: {task}\n"
            f"TASK: {f['task']['instruction']}\n\n"
            f"SCENE FACTS (JSON):\n{json.dumps(f, sort_keys=True)}\n\n"
            "YOUR PREVIOUS PLAN (JSON):\n" + json.dumps({k: plan_dict[k] for k in ("belief", "steps", "recoveries", "fallback")}, sort_keys=True) + "\n\n"
            "NEW MEASUREMENT: the robot's probe (arms push the object up and read the joint torques) shows the true value of the following parameters lies in these "
            "ranges (treat them as correct):\n" + json.dumps(probe_set, sort_keys=True) + "\n"
            "Part of this measured range lies OUTSIDE your previous belief: " + json.dumps(unc, sort_keys=True) + "\n"
            "Return a COMPLETE new plan in the same JSON format. Its belief must contain the measured range, and its steps and recoveries must be written for the part "
            "your previous plan did not cover (a recovery that was only safe inside the old belief may not be safe now; the object is heavier or lighter than you assumed). "
            "Return the JSON object now.")


REASK_MARK = "RULE 2 RE-ASK"          # the mock proposer keys on it (mock.py)


def reask_evidence_block(ev):
    """the schema-v2 Rule 2 evidence as prompt text (T1: a reading showed a phenomenon the plan did not name; T2: no recovery of the plan is viable on the current set).  Only what
    the robot measured and the checker computed: readings (sound intervals), the probes the checker skipped (with the failing own-limit rows), the failing admission rows, the worst
    vertex of the binding row.  No hidden truth, no world seed."""
    trig = ev.get("trigger", "T2")
    head = {"T1": f"{REASK_MARK} (T1): a probe reading shows a phenomenon your plan did not name.",
            "T2": f"{REASK_MARK} (T2): no recovery of your plan is viable on the parameter set the probes left (the checker refuses the plan on it)."}[trig]
    parts = [head]
    parts.append("READINGS (the robot's measurements; treat as correct; a one-sided reading has hi = null, i.e. only a lower bound was read):\n"
                 + json.dumps(ev.get("readings", []), sort_keys=True))
    if ev.get("probes_skipped"):
        parts.append("PROBES THE CHECKER DID NOT RUN (probe, why, failing limit rows):\n" + json.dumps(ev["probes_skipped"], sort_keys=True))
    if ev.get("phenomena_shown"):
        parts.append("PHENOMENA SHOWN BY A READING AND NOT NAMED IN YOUR PLAN:\n" + json.dumps(ev["phenomena_shown"], sort_keys=True))
    if ev.get("failing_rows"):
        parts.append("FAILING CHECKER ROWS ON THE CURRENT SET:\n" + json.dumps(ev["failing_rows"], sort_keys=True))
    if ev.get("worst"):
        parts.append("WORST VERTEX OF THE BINDING ROW (contents mass m [kg], contents COM c [m], waypoint wp):\n" + json.dumps(ev["worst"], sort_keys=True))
    if ev.get("coverage"):
        parts.append("COVERAGE (the checker's own phenomenon list: measured, on an approved fallback, or uncovered):\n" + json.dumps(ev["coverage"], sort_keys=True))
    return "\n\n".join(parts) + "\n"


def rejection_prompt(task, attempts, facts=None, world=None, prefix=None):
    """Rule 2 re-ask after a REJECTED plan (user 11-09: no cap on re-asks).  attempts = [{"plan": <validated plan dict or None>, "reply": <raw reply text>, "stage", "reasons"}, ...]
    oldest first.  The prompt = the original prompt + every rejected plan with the checker's structured rejection.  Only what the checker said: no truth, no world seed."""
    f = facts if facts is not None else scene_facts(task, world)
    parts = [user_prompt(task, f).rsplit("\n\nReturn the JSON object now.", 1)[0], ""]
    if prefix:                                               # a Rule 2 re-ask whose new plan was rejected: the evidence block stays in the request (it is part of the cache key)
        parts += [prefix, ""]
    for i, at in enumerate(attempts, 1):
        pl = at.get("plan")
        body = (json.dumps(S.reply_of(pl), sort_keys=True)
                if pl and pl.get("status") == "ok" else "(the reply did not parse or validate; raw reply follows)\n" + str(at.get("reply") or ""))
        parts += [f"REJECTED PLAN {i} (JSON):", body,
                  f"REJECTION {i}: stage={at['stage']}; reasons: " + json.dumps(list(at["reasons"])), ""]
    parts.append("The checker rejected your plan; it was not run. Propose a COMPLETE corrected plan in the same JSON format that avoids the reasons above "
                 "(stages: parse/schema = the reply format or ranges; compile = the plan's own consistency with the task; planner = the arms cannot build it, e.g. a joint limit "
                 "or a collision at the named step). Change only what the reasons require. Return the JSON object now.")
    return "\n".join(parts)
