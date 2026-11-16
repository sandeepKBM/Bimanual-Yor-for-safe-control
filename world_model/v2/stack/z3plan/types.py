"""Shared interface between the Z3 planner (this package) and the oracle (adapters side).  Pure."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Domain:
    """one plan knob: a finite grid.  Order of `values` carries no preference."""
    name: str
    values: tuple
    unit: str = ""
    tag: str = "ASSUMED"        # where the grid comes from: MEASURED / MODEL / ASSUMED


PlanVals = dict                  # {knob name: value}; hashable key = plan_key(plan)


def plan_key(plan: dict) -> tuple:
    return tuple(sorted(plan.items()))


@dataclass(frozen=True)
class Theta:
    """contents interval [lo, hi] kg (carry_cert cell units): the set the plan must pass on (for all theta in it)."""
    lo: float
    hi: float

    def __post_init__(self):
        if not (self.lo <= self.hi):
            raise ValueError(f"empty theta [{self.lo}, {self.hi}]")

    def contains(self, o: "Theta") -> bool:
        """self is a superset of o."""
        return self.lo <= o.lo and self.hi >= o.hi


@dataclass(frozen=True)
class ReadingModel:
    """y = offset_kg + contents + n, |n| <= noise_kg (offset = empty object mass; the oracle side provides it)."""
    offset_kg: float
    noise_kg: float
    tag: str = "ASSUMED"


KINDS = ("block_exact", "block_theta_superset", "block_all_theta_a", "block_a_ge", "block_all_theta")


@dataclass
class Learned:
    """kind: block_exact (this plan on exactly this theta); block_theta_superset (this plan on any theta containing this one);
    block_all_theta_a (this plan's geometry, any a, any theta: every knob except "a" fixed); block_a_ge (this plan with a' >= a,
    on this theta and its supersets); block_all_theta (this plan on ANY theta, theta-independent failure; the knobs not named in `free` fixed).
    free (v2): the knobs the failure does NOT depend on.  The constraint blocks the plan's values of the OTHER knobs for every value of the free knobs
    (one clause over the remaining knobs).  block_all_theta_a and block_a_ge always treat "a" as free (so free=() is the v1 behaviour)."""
    kind: str
    plan: dict
    theta: Optional[Theta]
    reason: str
    free: tuple = ()


@dataclass
class OracleResult:
    ok: bool
    rows: list = field(default_factory=list)       # [dict(name, ok, margin, guarantee, why)]
    learned: list = field(default_factory=list)    # [Learned]; when ok is False: at least one ("block_exact" at minimum)
    wall_s: float = 0.0
    cached: bool = False


Oracle = Callable[[dict, Theta], OracleResult]
