"""checks/row.py -- one result shape for every check.  margin > 0 = inside (safe / admitted), margin < 0 = outside; unit says what the margin is in.

guarantee (AGENTS.md rule 7, plus two non-admission kinds):
  proof | sampled | refuted   admission types; combine() takes the weakest (refuted < missing < sampled < proof)
  missing                     the check is not built (coverage shows it); never admits anything by itself
  runtime                     a monitor row evaluated on a state, not an admission; excluded from the verdict's guarantee
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from world_model.v2.stack.types import Bound, Verdict

ORDER = ("refuted", "missing", "sampled", "proof")      # weakest first
KINDS = ORDER + ("runtime",)


@dataclass
class Row:
    name: str
    ok: bool
    margin: float | None = None          # > 0 inside; None when not evaluated (missing)
    unit: str = ""
    guarantee: str = "runtime"
    worst: Any = None                    # the binding case / per-term margins
    n: int | None = None                 # samples behind a sampled row
    eps_upper: float | None = None       # one-sided upper bound on the failure rate (sampled rows)
    assumptions: dict = field(default_factory=dict)
    why: str = ""

    def __post_init__(self):
        if self.guarantee not in KINDS:
            raise ValueError(f"unknown guarantee {self.guarantee!r}; have {KINDS}")


def bound_margin(value: float, b: Bound) -> float:
    """distance of `value` inside the interval [b.lo, b.hi] (negative = outside); +-inf ends do not bind."""
    m = math.inf
    if math.isfinite(b.lo):
        m = min(m, value - b.lo)
    if math.isfinite(b.hi):
        m = min(m, b.hi - value)
    return m


def inside(margin: float, b: Bound) -> bool:
    """is a value with this margin inside `b`?  A strict Bound (tasks.Lim) excludes its own boundary (the old code's comparator)."""
    return margin > 0.0 if getattr(b, "strict", False) else margin >= 0.0


def bound_dict(b: Bound) -> dict:
    return dict(lo=b.lo, hi=b.hi, tag=b.tag, src=b.src, unit=b.unit, strict=bool(getattr(b, "strict", False)))


def missing(name: str, why: str) -> Row:
    return Row(name, ok=False, margin=None, guarantee="missing", why=why)


def combine(rows, checker: str = "check") -> Verdict:
    """rows -> Verdict.  admit = every non-missing, non-runtime row ok.  guarantee = weakest over the admission rows (a missing row downgrades
    it to "missing"); runtime rows are listed under assumptions["runtime_rows"] and do not enter the guarantee."""
    rows = list(rows)
    adm = [r for r in rows if r.guarantee not in ("runtime", "missing")]
    miss = [r for r in rows if r.guarantee == "missing"]
    run = [r for r in rows if r.guarantee == "runtime"]
    ok = bool(adm) and all(r.ok for r in adm)
    kinds = [r.guarantee for r in adm + miss]
    g = min(kinds, key=ORDER.index) if kinds else "missing"
    if any(not r.ok for r in adm):          # a failed row is a counterexample (exact corner or sampled world)
        g = "refuted"
    ass: dict = {}
    for r in adm + miss:
        ass[r.name] = r.assumptions
    ass["missing_rows"] = [r.name for r in miss]
    ass["runtime_rows"] = [r.name for r in run]
    why = "; ".join(f"{r.name}: {r.why}" for r in rows if (not r.ok) and r.why)
    return Verdict(checker=checker, admit=ok, guarantee=g, margins={r.name: r.margin for r in rows if r.margin is not None},
                   assumptions=ass, why=why)
