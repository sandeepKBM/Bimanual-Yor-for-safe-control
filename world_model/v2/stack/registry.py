"""stack/registry.py -- name -> "module:attr" import strings per component kind, resolved lazily by get(kind, name).  SIMULATION ONLY.

None = placeholder (not built yet): get() raises NotImplementedError.  Add an entry here when a component lands; no component imports another
by path, so universes/skills/planners stay swappable from an EvalSpec.
"""
from __future__ import annotations

import importlib

UNIVERSES = {
    "contents": "world_model.v2.stack.universes.contents:ContentsUniverse",
    "democ": "world_model.v2.stack.universes.democ:DemoCUniverse",
    "pot": "world_model.v2.stack.universes.pot:PotUniverse",
    "house": None,    # placeholder: not built
}
SKILLS = {
    "pregrasp": "world_model.v2.stack.skills.pregrasp:PreGrasp",
}
MEASURE = {                        # measuring skills of the probe-selection build (11-09): id -> adapter class; None = design only (skills/catalog.py status "design")
    "weigh": "world_model.v2.adapters.skills_contents:Weigh",      # adapter layer (drives the carry sim); lands in build step 3
    "hold_moment": None,
    "shake_listen": None,
    "watch_item": None,
}
PLANNERS = {
    "scripted": "world_model.v2.stack.planners.scripted:ScriptedPlanner",
    "closed_chain": "world_model.v2.stack.planners.closed_chain:ClosedChainPlanner",
}
PRODUCERS = {
    "truth": "world_model.v2.stack.producers:ContentsTruthProducer",
}
CHECKERS = {                       # unified rows (11-09): each is a function returning checks.row.Row (see the module docstrings for the arguments)
    "payload": "world_model.v2.stack.checks.payload:torque_row",
    "payload_rest": "world_model.v2.stack.checks.payload:rest_row",
    "support": "world_model.v2.stack.checks.support:support_row",
    "open_content": "world_model.v2.stack.checks.open_content:open_content_row",
    "probe_rung": "world_model.v2.stack.checks.probe_rung:rung_row",
    "plan_shape": "world_model.v2.stack.checks.plan_shape:check",
    "dynamic": "world_model.v2.stack.checks.dynamic:family_row",
    "tilt": "world_model.v2.stack.checks.tilt:tilt_row",           # MISSING row (not built)
    "monitors_U": "world_model.v2.stack.checks.monitors:u_row",
    "monitors_R": "world_model.v2.stack.checks.monitors:recoverable",
    "admit": "world_model.v2.stack.checks.admit:admit",
    "goal": "world_model.v2.stack.goal:check_goal",
}

_KINDS = {"universes": UNIVERSES, "skills": SKILLS, "measure": MEASURE, "planners": PLANNERS, "producers": PRODUCERS, "checkers": CHECKERS}


def get(kind: str, name: str):
    """resolve registry[kind][name] to the object (class / function).  Imports lazily."""
    if kind not in _KINDS:
        raise KeyError(f"unknown registry kind {kind!r}; have {sorted(_KINDS)}")
    table = _KINDS[kind]
    if name not in table:
        raise KeyError(f"{kind}: unknown name {name!r}; have {sorted(table)}")
    spec = table[name]
    if spec is None:
        raise NotImplementedError(f"{kind}/{name} is a registered placeholder (not built yet)")
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


def names(kind: str) -> list[str]:
    return sorted(_KINDS[kind])
