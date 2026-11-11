"""stack/goal.py -- the goal of a task as labelled terms, and the one goal check.  SIMULATION ONLY.

A Goal is a list of GoalTerm(quantity, Bound) plus the reference values the derived quantities need:
  xy_err  = hypot(obj_x - gx, obj_y - gy)             goal xy (world frame, or relative to the start for goal_frame "start")
  z_err   = |obj_z - obj_z0 - dz|                     dz = goal support top - start support top (= the old |z - (support top + z0 - counter top)|)
Every other term reads a state quantity directly (state.read).  Tolerances live in tasks.py (per task, tagged).
Comparators are the old code's: a term's Bound is strict where the old test used < (pan tilt / lid / support, coffee tray level), else inclusive.
A state that already carries `xy_err` (coffee tray: the sim's unrounded tray_xy_err) is not re-derived.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks.row import Row, bound_dict, bound_margin, inside
from world_model.v2.stack.state import StepState, has, read
from world_model.v2.stack.types import Bound


@dataclass(frozen=True)
class GoalTerm:
    quantity: str
    bound: Bound


@dataclass(frozen=True)
class Goal:
    task: str
    terms: tuple
    ref: dict = field(default_factory=dict)       # gx, gy, dz (those the task needs)


def goal_for(task: str, world=None, plan=None) -> Goal:
    """the goal of `task` in `world` (carry tasks: the world object, for its start xy and support tops; other backends: None).
    `plan` is accepted for the planner interface and not used: the goal is the TASK's, a plan does not change it."""
    t = TK.task(task)
    ref: dict = {}
    if t.goal_dxy is not None:
        gx, gy = t.goal_dxy
        if t.goal_frame == "world":
            gx, gy = world.obj_xy[0] + gx, world.obj_xy[1] + gy
        ref.update(gx=float(gx), gy=float(gy))
    if any(q == "z_err" for q, _ in t.goal_terms):
        ref["dz"] = float(getattr(world, f"{t.goal_support}_top") - getattr(world, f"{t.start_support}_top"))
    return Goal(task, tuple(GoalTerm(q, b) for q, b in t.goal_terms), ref)


def goal_xy(task: str, world) -> tuple:
    """the world xy of the task's goal (object reference), e.g. where a plan must place the object."""
    g = goal_for(task, world)
    return (g.ref["gx"], g.ref["gy"])


def derive(goal: Goal, values: dict) -> dict:
    """values + the derived quantities the goal's reference gives."""
    out = dict(values)
    if "gx" in goal.ref and "obj_x" in values and "xy_err" not in values:
        out["xy_err"] = math.hypot(values["obj_x"] - goal.ref["gx"], values["obj_y"] - goal.ref["gy"])
    if "dz" in goal.ref and "obj_z" in values:
        out["z_err"] = abs(values["obj_z"] - values["obj_z0"] - goal.ref["dz"])
    return out


def check_goal(goal: Goal, state: StepState) -> Row:
    """every term inside its Bound on state.now.  margin = the smallest term margin (terms have different units: see worst);
    a term whose quantity the state lacks is reported; with no other failing term the row is `missing` (unknown), else a plain failure."""
    vals = derive(goal, state.now)
    margins, absent, bad = {}, [], []
    for gt in goal.terms:
        if has(vals, gt.quantity):
            margins[gt.quantity] = bound_margin(read(vals, gt.quantity), gt.bound)
            if not inside(margins[gt.quantity], gt.bound):
                bad.append(gt.quantity)
        else:
            absent.append(gt.quantity)
    ok = bool(not absent and margins and not bad)
    why = ("; ".join([f"{q} outside ({margins[q]:.4g})" for q in bad] + [f"{q} not in the state" for q in absent]))
    guar = "missing" if (absent and not bad) or not goal.terms else "runtime"        # not measured and nothing else failed = unknown, not "not reached"
    return Row("goal", ok, margin=min(margins.values()) if margins else None, unit="mixed (see worst)", guarantee=guar, worst=margins,
               assumptions={gt.quantity: bound_dict(gt.bound) for gt in goal.terms}, why=why)
