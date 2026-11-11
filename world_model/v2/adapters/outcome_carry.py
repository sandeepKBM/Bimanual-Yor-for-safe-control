"""adapters/outcome_carry.py -- a finished CarryEpisode -> the goal Row + the one Outcome (stack.goal.check_goal, stack.outcome.classify).  SIMULATION ONLY.
Every method that ends a carry episode (ours, AgentChord, the ungated baseline) writes its record through `measure`: one success rule (USER 11-09 R1)."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from world_model.v2.adapters.state_carry import from_carry
from world_model.v2.stack import goal as GL, outcome as OC
from world_model.v2.stack.checks.row import Row, missing


def _R_row(okR, whyR):
    if okR is None:
        return missing("R", "end_in_R not measured")
    return Row("R", bool(okR), guarantee="runtime", why=",".join(whyR or []) if not isinstance(whyR, str) else whyR)


def row_dict(r: Row) -> dict:
    return asdict(r)


def measure(ep, task, finished, unsafe, pre_hold_viol=None, refused=False, reason="", end_in_R=None) -> dict:
    """-> dict(goal = Row as dict, measured = raw goal quantities, result = Outcome.record()).
    finished: True | False | None.  unsafe: True | False | None = any U entry of the whole episode (the caller knows its track(s)); pre_hold_viol: the viol dict or None.
    end_in_R: (ok, why) as CarryEpisode.in_R() returns it; None -> read it from ep now.  The goal is the TASK's (stack.tasks) on ep.world."""
    es = from_carry(ep, finished=finished, pre_hold_viol=pre_hold_viol)
    goal = GL.goal_for(task, ep.world)
    grow = GL.check_goal(goal, es)
    okR, whyR = end_in_R if end_in_R is not None else ep.in_R()
    oc = OC.classify(finished, grow, unsafe, pre_hold_viol is not None, _R_row(okR, whyR), refused=refused, reason=reason)
    d = GL.derive(goal, es.now)
    measured = dict(xy_err=round(float(d["xy_err"]), 4), tilt_deg=round(float(es.now["obj_tilt_deg"]), 2), mug_tilt_deg=round(float(es.now["cup_tilt_deg"]), 2),
                    goal_support=goal_support(task))
    if "z_err" in d:
        measured["z_err"] = round(float(d["z_err"]), 4)
    measured.update({k: round(float(v), 4) for k, v in es.now.items() if k.startswith("cap_rel_")})     # coffee tray: capsule in the tray frame
    return dict(goal=row_dict(grow), measured=measured, result=oc.record())


def goal_support(task):
    from world_model.v2.stack import tasks as TK
    return TK.task(task).goal_support
