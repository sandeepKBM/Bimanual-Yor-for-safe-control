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
PLANNERS = {
    "scripted": "world_model.v2.stack.planners.scripted:ScriptedPlanner",
    "closed_chain": "world_model.v2.stack.planners.closed_chain:ClosedChainPlanner",
}
PRODUCERS = {
    "truth": "world_model.v2.stack.producers:ContentsTruthProducer",
}
CHECKERS: dict = {}

_KINDS = {"universes": UNIVERSES, "skills": SKILLS, "planners": PLANNERS, "producers": PRODUCERS, "checkers": CHECKERS}


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
