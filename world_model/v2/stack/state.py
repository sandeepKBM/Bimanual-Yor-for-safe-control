"""stack/state.py -- the quantities every check reads (dataclasses + a Reader protocol).  SIMULATION ONLY.
The ONE reader per SIM BACKEND lives in world_model/v2/adapters/ (state_carry.py, state_pan.py, state_embodichain.py): stack core does not import sims.

A state carries two dicts of floats with the same vocabulary:
  now    instantaneous values (R is evaluated on these; goals too)
  worst  worst value over the window since the episode's track was reset (U is evaluated on these).  A backend that only keeps the FIRST
         violation (carry: collision, cup_off_tray, drop) gives a flag / a just-past-the-bound value there; `first_viol` is the backend's own record.
Quantity names are the keys of tasks.py U / R / goal tables; "abs(q)" in a table reads |q|.  Flags are 0.0 / 1.0.
The readers call the backends' own measurement helpers (carry._tilt_deg, CarryEpisode._lid_rel, pan Episode.pan_state / contacts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol



@dataclass
class StepState:
    t: float
    now: dict
    worst: dict
    first_viol: dict | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class EndState(StepState):
    finished: bool | None = None           # the plan / graph ran to its end (None = not known)
    pre_hold_viol: dict | None = None      # a U entry during the load-transfer hold before the plan (carry)


def read(values: dict, name: str) -> float:
    """values[name]; "abs(q)" -> |values[q]|.  KeyError when the quantity is absent."""
    if name.startswith("abs(") and name.endswith(")"):
        return abs(values[name[4:-1]])
    return values[name]


def has(values: dict, name: str) -> bool:
    return (name[4:-1] if name.startswith("abs(") and name.endswith(")") else name) in values




class Reader(Protocol):
    """one per sim backend (adapters/state_*.py): a live episode / record -> EndState."""

    def __call__(self, source: Any, **kw) -> EndState: ...
