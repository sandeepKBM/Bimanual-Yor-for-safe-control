"""stack/proposer/prompt.py -- the system and user prompt of the proposer, per task.  SIMULATION ONLY.   (11-05)

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
        raise NotImplementedError(f"{task}: a moving-base task; the proposer prompt / closed-chain plan have no base route yet (goal is TaskDef.goal_dxy on {TK.task(str(task)).goal_support})")
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


def robot_limits():
    """datasheet / config numbers shown to the model (nero/config.py is the one source; nothing is copied into this file)."""
    from world_model.v2.nero import config as C
    tl = np.asarray(C.tau_limit(), float)
    return dict(arms=2, arm_dof=7, payload_kg_per_arm_datasheet=C.MANUAL["payload_kg"], joint_speed_deg_s=list(C.MANUAL["joint_speed_deg_s"]),
                joint_torque_limit_Nm=[round(float(x), 2) for x in tl], reach_m_datasheet=C.MANUAL["reach_m"])


def scene_facts(task, world=None):
    """everything the model may know about the scene (JSON-able, deterministic).  No hidden truth."""
    sp = task_spec(task, world)
    P = S.params(task)
    return dict(task=sp, hidden_parameters={k: dict(unit=v["unit"], sanity_range=[round(float(v["range"][0]), 4), round(float(v["range"][1]), 4)], meaning=v["doc"])
                                           for k, v in P.items()},
                robot=robot_limits(),
                frame="world frame, metres, z up.  Decreasing x moves the object toward the robot.  The left arm holds the +y handle/tab, the right arm the -y one.",
                grasp="Both claws are rigidly welded to the object at its handles/tabs for the whole motion (ASSUMED: grasping is not part of this task).  The object stays level.",
                vocabulary=S.vocabulary(tuple(sp["supports"])))


SYSTEM = """You are the high-level planner of a two-arm robot (two 7-DoF NERO arms) in a simulation. You write the plan; a separate low-level planner turns it into joint motions and a checker verifies it before anything moves. You never control joints.

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


def reask_prompt(task, plan_dict, probe_set, facts=None, world=None):
    """Rule 2: the probe set is not inside B.  Tell the model the numbers (a measurement the robot made; no hidden truth) and ask for a complete new plan whose belief and
    recoveries cover the uncovered part.  probe_set = {param: [lo, hi]} of the probed parameter(s)."""
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


def rejection_prompt(task, attempts, facts=None, world=None):
    """Rule 2 re-ask after a REJECTED plan (user 11-09: no cap on re-asks).  attempts = [{"plan": <validated plan dict or None>, "reply": <raw reply text>, "stage", "reasons"}, ...]
    oldest first.  The prompt = the original prompt + every rejected plan with the checker's structured rejection.  Only what the checker said: no truth, no world seed."""
    f = facts if facts is not None else scene_facts(task, world)
    parts = [user_prompt(task, f).rsplit("\n\nReturn the JSON object now.", 1)[0], ""]
    for i, at in enumerate(attempts, 1):
        pl = at.get("plan")
        body = (json.dumps({k: pl[k] for k in ("belief", "steps", "recoveries", "fallback") if k in pl}, sort_keys=True)
                if pl and pl.get("status") == "ok" else "(the reply did not parse or validate; raw reply follows)\n" + str(at.get("reply") or ""))
        parts += [f"REJECTED PLAN {i} (JSON):", body,
                  f"REJECTION {i}: stage={at['stage']}; reasons: " + json.dumps(list(at["reasons"])), ""]
    parts.append("The checker rejected your plan; it was not run. Propose a COMPLETE corrected plan in the same JSON format that avoids the reasons above "
                 "(stages: parse/schema = the reply format or ranges; compile = the plan's own consistency with the task; planner = the arms cannot build it, e.g. a joint limit "
                 "or a collision at the named step). Change only what the reasons require. Return the JSON object now.")
    return "\n".join(parts)
