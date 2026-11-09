"""checks/monitors.py -- the unsafe set U and the task-ready set R as tables of (quantity, Bound) from TaskDef, evaluated on a state.  SIMULATION ONLY.
U is read on state.worst (window worst), R on state.now.  Each quantity's margin = distance inside its Bound in the quantity's unit; inside is decided by
the Bound's own comparator (strict for the old code's `>=` / `<=` violations, see tasks.Lim).
A quantity the state lacks is NEVER passed: the row is a `missing` row (ok=False, guarantee "missing", why names the quantity), so an outcome that rests on
it is unknown (success None), not True.  Same for a task whose U or R table is empty.  If another quantity is violated the row is a plain failed row."""
from __future__ import annotations

from world_model.v2.stack.checks.row import Row, bound_dict, bound_margin, inside
from world_model.v2.stack.state import StepState, has, read
from world_model.v2.stack.tasks import TaskDef


def _eval(table, values):
    """-> (margins {q: m}, violated [q] in table order, absent [q])."""
    margins, bad, absent = {}, [], []
    for q, b in table:
        if has(values, q):
            margins[q] = bound_margin(read(values, q), b)
            if not inside(margins[q], b):
                bad.append(q)
        else:
            absent.append(q)
    return margins, bad, absent


def first_violation(state: StepState, t: TaskDef):
    """-> (name | None, margins).  The first TABLE entry (monitor order) outside its Bound.  This is NOT the first in time: u_row reports the backend's
    time-ordered state.first_viol first and uses this only as a fallback."""
    margins, bad, _ = _eval(t.U, state.worst)
    return (bad[0] if bad else None), margins


def _missing_row(name, why, margins, absent, table):
    return Row(name, False, margin=min(margins.values()) if margins else None, unit="mixed (see worst)", guarantee="missing", worst=margins,
               assumptions=dict(not_evaluated=absent, **{q: bound_dict(b) for q, b in table}), why=why)


def u_row(state: StepState, t: TaskDef) -> Row:
    if not t.U:
        return _missing_row("U", f"task {t.name} has no U table (unsafe set not defined)", {}, [], ())
    margins, bad, absent = _eval(t.U, state.worst)
    bf = state.first_viol
    first_in_time = (bf or {}).get("kind")
    first = first_in_time or (bad[0] if bad else None)
    if bad or bf is not None:
        ok, guar, why = False, "runtime", f"U entered: {first}"
    elif absent:
        ok, guar, why = False, "missing", "U quantity not measured: " + ", ".join(absent)
    else:
        ok, guar, why = True, "runtime", ""
    return Row("U", ok, margin=min(margins.values()) if margins else None, unit="mixed (see worst)", guarantee=guar,
               worst=dict(margins, _backend_first_viol=bf, _table_first=bad[0] if bad else None),
               assumptions=dict(not_evaluated=absent, **{q: bound_dict(b) for q, b in t.U}), why=why)


def unsafe(state: StepState, t: TaskDef):
    """True | False | None (None = a U quantity was not measured and nothing else shows a violation: unknown)."""
    r = u_row(state, t)
    return None if r.guarantee == "missing" else (not r.ok)


def recoverable(state: StepState, t: TaskDef, include_U: bool | None = None, version: str = "v1") -> Row:
    """R: the task-ready set on state.now.  include_U adds 'U:<kind>' when the episode entered U (as carry.in_R does, carry.py:639); None = the task's
    own rule (TaskDef.r_includes_U: carry True, pan False as segments.Episode.in_R).  version selects the pan R table (v1 | v2)."""
    table = t.R_table(version)
    if not table:
        return _missing_row("R", f"task {t.name} has no R table (task-ready set not defined)", {}, [], ())
    margins, bad, absent = _eval(table, state.now)
    why = list(bad)
    if t.r_includes_U if include_U is None else include_U:
        ur = u_row(state, t)
        if not ur.ok and ur.guarantee != "missing":
            why.append(ur.why)
    if why:
        ok, guar = False, "runtime"
    elif absent:
        ok, guar, why = False, "missing", ["R quantity not measured: " + ", ".join(absent)]
    else:
        ok, guar = True, "runtime"
    return Row("R", ok, margin=min(margins.values()) if margins else None, unit="mixed (see worst)", guarantee=guar, worst=margins,
               assumptions=dict(not_evaluated=absent, version=version, **{q: bound_dict(b) for q, b in table}), why="; ".join(why))
