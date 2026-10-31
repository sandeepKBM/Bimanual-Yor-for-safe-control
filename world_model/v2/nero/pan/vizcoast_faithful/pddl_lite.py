"""Minimal PDDL reader + grounder: typed STRIPS with negative preconditions and (not (= ?x ?y)).  Enough for pan.pddl."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field


def sexpr(text):
    text = re.sub(r";[^\n]*", "", text).lower()
    toks = re.findall(r"\(|\)|[^\s()]+", text)
    stack, cur = [], []
    for t in toks:
        if t == "(":
            stack.append(cur)
            cur = []
        elif t == ")":
            done, cur = cur, stack.pop()
            cur.append(done)
        else:
            cur.append(t)
    return cur


def _typed(lst):
    out, pend = [], []
    i = 0
    while i < len(lst):
        if lst[i] == "-":
            out += [(n, lst[i + 1]) for n in pend]
            pend, i = [], i + 2
        else:
            pend.append(lst[i])
            i += 1
    return out + [(n, "object") for n in pend]


def _lits(e):
    """-> list of (positive, atom-tuple) ; atom ('=', a, b) for equality."""
    if not e:
        return []
    if e[0] == "and":
        return [x for s in e[1:] for x in _lits(s)]
    if e[0] == "not":
        return [(not p, a) for p, a in _lits(e[1])]
    return [(True, tuple(e))]


@dataclass(frozen=True)
class GroundAction:
    name: str
    args: tuple
    pre_pos: frozenset
    pre_neg: frozenset
    add: frozenset
    dele: frozenset

    def __str__(self):
        return f"({self.name} {' '.join(self.args)})" if self.args else f"({self.name})"


@dataclass
class Task:
    actions: list
    fluents: list
    init: frozenset
    goal: frozenset
    statics: frozenset
    static_preds: frozenset
    names: dict = field(default_factory=dict)       # lower-case object/constant -> spelling in the PDDL file (what the VLM saw)


def load(path):
    raw = open(path).read()
    blocks = sexpr(raw)
    dom = next(b for b in blocks if b[0] == "define" and b[1][0] == "domain")
    prob = next(b for b in blocks if b[0] == "define" and b[1][0] == "problem")
    objs, acts, preds = {}, [], []
    for sec in dom[2:]:
        if sec[0] == ":constants":
            objs.update(dict(_typed(sec[1:])))
        elif sec[0] == ":predicates":
            preds = [p[0] for p in sec[1:]]
        elif sec[0] == ":action":
            d = dict(zip(sec[2::2], sec[3::2]))
            acts.append((sec[1], _typed(d.get(":parameters", [])), _lits(d.get(":precondition", [])), _lits(d.get(":effect", []))))
    init, goal = set(), set()
    for sec in prob[2:]:
        if sec[0] == ":objects":
            objs.update(dict(_typed(sec[1:])))
        elif sec[0] == ":init":
            init = {tuple(x) for x in sec[1:]}
        elif sec[0] == ":goal":
            goal = {a for p, a in _lits(sec[1]) if p}
    changed = {a[0] for _, _, _, eff in acts for _, a in eff}
    static_preds = frozenset(p for p in preds if p not in changed)
    statics = frozenset(a for a in init if a[0] in static_preds)
    ground = []
    for name, params, pre, eff in acts:
        doms = [[o for o, t in objs.items() if t == ty] for _, ty in params]
        for combo in itertools.product(*doms):
            sub = {p[0]: v for p, v in zip(params, combo)}
            g = lambda a: tuple(sub.get(x, x) for x in a)
            ok, pp, pn = True, set(), set()
            for pos, a in pre:
                a = g(a)
                if a[0] == "=":
                    ok &= (a[1] == a[2]) == pos
                elif a[0] in static_preds:
                    ok &= (a in statics) == pos
                else:
                    (pp if pos else pn).add(a)
            if not ok:
                continue
            add = frozenset(g(a) for pos, a in eff if pos)
            dele = frozenset(g(a) for pos, a in eff if not pos)
            ground.append(GroundAction(name, tuple(combo), frozenset(pp), frozenset(pn), add, dele))
    fl = sorted({a for ga in ground for a in ga.pre_pos | ga.pre_neg | ga.add | ga.dele} | {a for a in init if a[0] not in static_preds} | goal)
    names = {t.lower(): t for t in re.findall(r"[^\s()]+", re.sub(r";[^\n]*", "", raw)) if t.lower() in objs}
    return Task(ground, fl, frozenset(a for a in init if a[0] not in static_preds), frozenset(goal), statics, static_preds, names)
