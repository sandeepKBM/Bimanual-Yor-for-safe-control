"""adapters/contents_specs.py -- ObjectSpec (contents set) builders, world builders and the pot's empty properties, for the carry tasks.  SIMULATION ONLY.
Table keyed by TaskDef.kind; the spill formula is stack/checks/open_content.spill_limit (one copy)."""
from __future__ import annotations

import importlib

import numpy as np

from world_model.v2.nero.contents import objects as OB
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks.open_content import spill_limit


def resolve(ref: str):
    mod, attr = ref.split(":")
    return getattr(importlib.import_module(mod), attr)


def build_world(task: str):
    """the default world object of a carry task (the static admission's)."""
    t = TK.task(task)
    cls = resolve(t.world_ref)
    kw = dict(t.world_kw)
    return cls(n_cups=t.n_cups, **kw) if t.n_cups else cls(**kw)


def empty_props(task: str):
    """(mass, COM) of the empty object: TaskDef numbers, or for the pot the MODEL scene geometry (scenes.pot_empty_props)."""
    t = TK.task(task)
    if t.empty_mass is not None:
        return TK.empty_props(t)
    m, c = SC.pot_empty_props()
    return float(m), tuple(float(x) for x in c)


def _fill_h(f_per_mug):
    return SC.CUP_BOT + f_per_mug / (SC.WATER * np.pi * SC.CUP_RIN ** 2)      # MODEL: fill height of a rigid cylinder of water


def _tray_cell(t, lo, hi):
    sp = OB.coffee_tray(t.n_cups)
    sp.contents = OB.Contents(t.n_cups * (OB.CUP["empty"] + lo), t.n_cups * (OB.CUP["empty"] + hi), sp.contents.P, sp.contents.labels)
    sp.open_limits = [OB.OpenLimit("cup_spill", spill_limit(OB.CUP["D"], OB.CUP["H"], _fill_h(hi)), "a_max = g 2 (H - h) / D, fullest fill of the cell")]
    return sp


def _tray_total(t, c_lo, c_hi):
    sp = OB.coffee_tray(t.n_cups)
    sp.contents = OB.Contents(c_lo, c_hi, sp.contents.P, dict(sp.contents.labels, m=f"interval {c_lo:.3f}..{c_hi:.3f} kg"))
    f_hi = float(np.clip(c_hi - t.n_cups * OB.CUP["empty"], 0.0, OB.CUP["fill_max_kg"]))
    sp.open_limits = [OB.OpenLimit("cup_spill", spill_limit(OB.CUP["D"], OB.CUP["H"], _fill_h(f_hi)),
                                   "a_max = g 2 (H - h) / D, one mug holding the whole fill hi")]
    return sp


def _box(t, lo, hi):
    return OB.boxed(lo, hi)


def _pot(t, lo, hi):
    return SC.pot_spec(lo, hi)


_CELL = {"tray": _tray_cell, "box": _box, "pot": _pot}
_TOTAL = {"tray": _tray_total, "box": _box, "pot": _pot}


def spec_from_cell(task: str, lo: float, hi: float):
    """ObjectSpec for a belief cell in the prior's unit (tray: fill per mug [kg]; box / pot: contents kg)."""
    t = TK.task(task)
    return _CELL[t.kind](t, lo, hi)


def spec_from_total(task: str, c_lo: float, c_hi: float):
    """ObjectSpec for a TOTAL carried contents mass in [c_lo, c_hi] (tray: mugs + fill; one mug may hold all the fill: conservative spill limit)."""
    t = TK.task(task)
    return _TOTAL[t.kind](t, c_lo, c_hi)


def prior_total(task: str):
    """the contents prior as a total mass interval (tray: n mugs x (empty + fill))."""
    t = TK.task(task)
    if t.n_cups:
        return t.n_cups * OB.CUP["empty"] + t.n_cups * t.prior.lo, t.n_cups * (OB.CUP["empty"] + t.prior.hi)
    return t.prior.lo, t.prior.hi
