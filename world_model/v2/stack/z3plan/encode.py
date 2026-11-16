"""Z3 encoding: one Int index per knob into Domain.values, range constraints, Learned -> constraint, lexicographic cost.  Pure."""
from __future__ import annotations

import z3

from .types import Domain, Learned

A_KNOB = "a"


class Encoding:
    def __init__(self, domains, a_name: str = A_KNOB):
        self.domains = {d.name: d for d in domains}
        if len(self.domains) != len(domains):
            raise ValueError("duplicate knob names")
        for d in domains:
            if not d.values:
                raise ValueError(f"empty domain {d.name}")
        self.a_name = a_name
        self.vars = {d.name: z3.Int(f"i_{d.name}") for d in domains}

    # ---- constraints
    def base(self) -> list:
        return [z3.And(v >= 0, v < len(self.domains[n].values)) for n, v in self.vars.items()]

    def _idx(self, knob: str, value):
        vals = self.domains[knob].values
        try:
            return vals.index(value)
        except ValueError:
            return None

    def _match(self, plan: dict, skip=()):
        """conjunction knob == plan value over the knobs not in skip; None if a plan value is off-grid (nothing to exclude)."""
        terms = []
        for n in self.domains:
            if n in skip:
                continue
            i = self._idx(n, plan[n])
            if i is None:
                return None
            terms.append(self.vars[n] == i)
        return z3.And(*terms) if terms else z3.BoolVal(True)

    def forbid(self, l: Learned):
        """constraint over the indices.  Theta applicability is decided by the caller (loop.LearnedStore).
        l.free = knobs the failure does not depend on (not constrained); block_all_theta_a / block_a_ge always free "a"."""
        free = set(getattr(l, "free", ()) or ())
        bad = free - set(self.domains)
        if bad:
            raise ValueError(f"Learned.free names unknown knobs {sorted(bad)}")
        if l.kind in ("block_exact", "block_theta_superset", "block_all_theta"):
            m = self._match(l.plan, skip=free)
        elif l.kind == "block_all_theta_a":
            m = self._match(l.plan, skip=free | {self.a_name})
        elif l.kind == "block_a_ge":
            m = self._match(l.plan, skip=free | {self.a_name})
            if m is not None:
                a0 = l.plan[self.a_name]
                ge = [i for i, v in enumerate(self.domains[self.a_name].values) if v >= a0]
                m = z3.And(m, z3.Or(*[self.vars[self.a_name] == i for i in ge])) if ge else None
        else:
            raise ValueError(f"unknown Learned kind {l.kind}")
        return z3.BoolVal(True) if m is None else z3.Not(m)

    def forbid_projection(self, plan: dict, knobs):
        """block every plan that agrees with `plan` on `knobs` (a prefilter rejection that reads only those knobs)."""
        m = self._match(plan, skip=set(self.domains) - set(knobs))
        return z3.BoolVal(True) if m is None else z3.Not(m)

    # ---- model <-> plan
    def decode(self, model) -> dict:
        return {n: self.domains[n].values[model.eval(v, model_completion=True).as_long()] for n, v in self.vars.items()}

    def value_expr(self, knob: str):
        """numeric Z3 expression for the knob's value (If-chain over the index)."""
        vals = self.domains[knob].values
        if not all(isinstance(v, (int, float)) for v in vals):
            raise ValueError(f"knob {knob} is not numeric; cannot be an objective")
        e = z3.RealVal(float(vals[-1]))
        for i in range(len(vals) - 2, -1, -1):
            e = z3.If(self.vars[knob] == i, z3.RealVal(float(vals[i])), e)
        return e

    def objective(self, opt, cost):
        """cost = [(knob, "max"|"min"), ...] lexicographic, first = most important.  opt must be a z3.Optimize."""
        opt.set(priority="lex")
        for knob, sense in cost:
            e = self.value_expr(knob)
            if sense == "max":
                opt.maximize(e)
            elif sense == "min":
                opt.minimize(e)
            else:
                raise ValueError(sense)

    def new_solver(self, cost=None):
        s = z3.Optimize() if cost else z3.Solver()
        s.add(*self.base())
        if cost:
            self.objective(s, cost)
        return s
