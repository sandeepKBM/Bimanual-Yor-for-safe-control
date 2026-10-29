"""SATPlan-style bounded planning in Z3 + the constraint API the VLM's program is written against.

Encoding (horizon H): fluent Bools F[f,t] (t = 0..H), action Bools A[a,t] (t = 0..H-1); init fixed at t=0; goal at t=H;
at most one action per step (sequential; empty steps allowed, so plans of length <= H); action -> preconditions at t and
effects at t+1; explanatory frame axioms (a fluent changes only if an action at t adds/deletes it).  Extra clauses come from
the VLM program (ConstraintAPI) and from temporary state-action blocks (refinement failures).
"""
from __future__ import annotations

import z3


class Encoding:
    def __init__(self, task, H):
        self.task, self.H = task, H
        self.F = {(f, t): z3.Bool(f"F|{' '.join(f)}|{t}") for f in task.fluents for t in range(H + 1)}
        self.A = {(i, t): z3.Bool(f"A|{ga}|{t}") for i, ga in enumerate(task.actions) for t in range(H)}
        s = self.s = z3.Solver()
        for f in task.fluents:
            s.add(self.F[f, 0] if f in task.init else z3.Not(self.F[f, 0]))
        for g in sorted(task.goal):
            s.add(self.F[g, H])
        for t in range(H):
            acts = [self.A[i, t] for i in range(len(task.actions))]
            s.add(z3.AtMost(*acts, 1))
            for i, ga in enumerate(task.actions):
                a = self.A[i, t]
                s.add(z3.Implies(a, z3.And([self.F[p, t] for p in sorted(ga.pre_pos)] + [z3.Not(self.F[p, t]) for p in sorted(ga.pre_neg)] or [z3.BoolVal(True)])))
                s.add(z3.Implies(a, z3.And([self.F[p, t + 1] for p in sorted(ga.add)] + [z3.Not(self.F[p, t + 1]) for p in sorted(ga.dele)] or [z3.BoolVal(True)])))
            for f in task.fluents:
                adders = [self.A[i, t] for i, ga in enumerate(task.actions) if f in ga.add]
                dels = [self.A[i, t] for i, ga in enumerate(task.actions) if f in ga.dele]
                s.add(z3.Implies(z3.And(z3.Not(self.F[f, t]), self.F[f, t + 1]), z3.Or(adders) if adders else z3.BoolVal(False)))
                s.add(z3.Implies(z3.And(self.F[f, t], z3.Not(self.F[f, t + 1])), z3.Or(dels) if dels else z3.BoolVal(False)))

    def state_eq(self, state, t):
        return z3.And([self.F[f, t] if f in state else z3.Not(self.F[f, t]) for f in self.task.fluents])

    def solve(self, extra=()):
        self.s.push()
        for c in extra:
            self.s.add(c)
        r = self.s.check()
        plan = None
        if r == z3.sat:
            mdl = self.s.model()
            plan = []
            for t in range(self.H):
                for i, ga in enumerate(self.task.actions):
                    if z3.is_true(mdl.eval(self.A[i, t], model_completion=True)):
                        plan.append(ga)
        self.s.pop()
        return plan


class ConstraintAPI:
    """What the VLM program sees (documented in the VRM prompt).  Everything returns Z3 expressions; api.add() asserts them."""

    def __init__(self, enc):
        self.enc, self.clauses = enc, []
        self.H = enc.H

    def steps(self):
        """the action time steps 0..H-1"""
        return range(self.enc.H)

    def action(self, name, *args, t):
        """Bool: ground action (name args...) is taken at step t (False if no such ground action)."""
        for i, ga in enumerate(self.enc.task.actions):
            if ga.name == name and ga.args == tuple(a.lower() for a in args):
                return self.enc.A[i, t]
        return z3.BoolVal(False)

    def actions(self, name=None, t=None):
        """list of (name, args, t, Bool) for every ground action matching name (all if None) at step t (all if None).
        args are spelled as in the PDDL file (e.g. 'L', 'handle'); action()/holds() accept any case."""
        ts = range(self.enc.H) if t is None else [t]
        nm = self.enc.task.names
        return [(ga.name, tuple(nm.get(x, x) for x in ga.args), tt, self.enc.A[i, tt]) for i, ga in enumerate(self.enc.task.actions)
                if name in (None, ga.name) for tt in ts]

    def holds(self, pred, *args, t):
        """Bool: fact (pred args...) is true at time t (state BEFORE the action of step t).  Static facts are constants."""
        a = (pred.lower(),) + tuple(x.lower() for x in args)
        if a[0] in self.enc.task.static_preds:
            return z3.BoolVal(a in self.enc.task.statics)
        return self.enc.F.get((a, t), z3.BoolVal(False))

    def disallow(self, name, *args, when=None):
        """block the action at every step (optionally only when when(t) holds)."""
        for t in self.steps():
            a = self.action(name, *args, t=t)
            self.add(z3.Not(z3.And(a, when(t))) if when else z3.Not(a))

    def add(self, *exprs):
        self.clauses += list(exprs)


SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ("range", "len", "any", "all", "list", "tuple", "dict", "set", "enumerate", "zip", "min", "max", "abs",
                           "str", "int", "float", "bool", "isinstance", "sorted", "print", "Exception", "ValueError")}
Z3_NAMES = dict(And=z3.And, Or=z3.Or, Not=z3.Not, Implies=z3.Implies, If=z3.If, BoolVal=z3.BoolVal, AtMost=z3.AtMost,
                AtLeast=z3.AtLeast, PbEq=z3.PbEq, Sum=z3.Sum)


def apply_program(code, enc):
    """exec the VLM program (restricted builtins, no imports) and return its clauses.  Raises on error (fed back to the VLM).
    NOTE: this is exec of model-written code; restricted namespace only, not a security sandbox."""
    g = dict(__builtins__=SAFE_BUILTINS, **Z3_NAMES)
    exec(compile(code, "<vlm_constraints>", "exec"), g)
    if "constraints" not in g:
        raise ValueError("the program must define constraints(api)")
    api = ConstraintAPI(enc)
    g["constraints"](api)
    return api.clauses


def plan(task, code=None, blocks=(), h_max=8):
    """incremental horizon 1..h_max; returns (plan or None, horizon, n_vlm_clauses).  blocks: list of (state, ground action)."""
    for H in range(1, h_max + 1):
        enc = Encoding(task, H)
        extra = apply_program(code, enc) if code else []
        n_vlm = len(extra)
        for state, ga in blocks:
            i = task.actions.index(ga)
            extra += [z3.Not(z3.And(enc.A[i, t], enc.state_eq(state, t))) for t in range(H)]
        p = enc.solve(extra)
        if p is not None:
            return p, H, n_vlm
    return None, h_max, None


def states_along(task, plan_):
    s = set(task.init)
    out = []
    for ga in plan_:
        out.append(frozenset(s))
        s = (s - ga.dele) | ga.add
    return out
