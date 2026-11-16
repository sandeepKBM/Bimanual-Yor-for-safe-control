"""propose (Z3) -> oracle -> learn -> repeat.  Pure.

solve(domains, oracle, theta, learned_store, cost=None, max_iters=500, prefilter=None) -> (plan | None, log)
  status = log[-1]["status"] in {"found", "unsat", "max_iters"} (unsat = Z3 proved no plan is left; max_iters = the cap, NOT "no plan exists").
  prefilter(vals) -> None | (rejected: bool, reason: str[, reads: tuple of knob names]): a physics-free rule.  Before the loop every plan of the (finite) domain product
  is tried; each rejected plan is blocked in Z3 (projected onto `reads` when given: all plans agreeing on those knobs).  log[0] is then a "prefilter" entry.

LearnedStore reuse rule (the only cross-theta reuse):
  block_theta_superset / block_a_ge found on T  applies to every query Q  with Q ⊇ T
  block_all_theta_a                              applies to every query Q
  block_exact                                    applies only to Q == T
Pass cache: plan passed on T  =>  passes on every Q ⊆ T (exact: admission is "for all theta in the set"), unless the oracle sets
  pass_monotone = False (then only Q == T: its checks depend on theta beyond "for all theta", e.g. a path model built at a_admit(theta)).
  Used (a) as an oracle shortcut for any proposed plan, (b) when cost is None, as an immediate answer before calling Z3
  (with a cost it is not used as the answer, since a cached passer need not be cost-optimal).
"""
from __future__ import annotations

import itertools
import time

from .encode import Encoding
from .types import Domain, Learned, OracleResult, Theta, plan_key


def _applies(l: Learned, q: Theta) -> bool:
    if l.kind in ("block_all_theta_a", "block_all_theta"):
        return True
    if l.theta is None:
        return False
    if l.kind == "block_exact":
        return l.theta == q
    return q.contains(l.theta)          # superset kinds


class LearnedStore:
    def __init__(self):
        self.learned: list = []
        self._seen: set = set()
        self.passes: list = []           # [(plan, Theta)]

    def add(self, l: Learned) -> bool:
        k = (l.kind, plan_key(l.plan), (l.theta.lo, l.theta.hi) if l.theta else None, tuple(sorted(getattr(l, "free", ()) or ())))
        if k in self._seen:
            return False
        self._seen.add(k)
        self.learned.append(l)
        return True

    def applicable(self, q: Theta) -> list:
        return [l for l in self.learned if _applies(l, q)]

    def add_pass(self, plan: dict, t: Theta):
        self.passes.append((dict(plan), t))

    def passed_superset(self, plan: dict, q: Theta, monotone: bool = True):
        k = plan_key(plan)
        return any(plan_key(p) == k and _covers(t, q, monotone) for p, t in self.passes)

    def any_pass(self, q: Theta, monotone: bool = True):
        for p, t in self.passes:
            if _covers(t, q, monotone):
                return p
        return None


def _covers(t: Theta, q: Theta, monotone: bool) -> bool:
    """pass on t carries to q: q subset of t when the oracle is pass-monotone, else only q == t (review 11-16: the contents oracle's Rule 1
    stage builds its path model at a_admit(theta), which grows as theta shrinks, so a pass on t does not imply a pass on a subset)."""
    return t.contains(q) and (monotone or q.contains(t))


MAX_PREFILTER_PLANS = 200_000


def _apply_prefilter(enc, s, domains, prefilter) -> dict:
    """enumerate the domain product, add one blocking clause per distinct rejected projection.  -> stats dict (n_plans, n_rejected, n_clauses, by_rule, wall_s)."""
    n = 1
    for d in domains:
        n *= len(d.values)
    if n > MAX_PREFILTER_PLANS:
        raise ValueError(f"prefilter: the domain product has {n} plans (> {MAX_PREFILTER_PLANS}); enumeration refused")
    t0 = time.perf_counter()
    names = [d.name for d in domains]
    seen, by_rule, n_rej = set(), {}, 0
    for combo in itertools.product(*[d.values for d in domains]):
        vals = dict(zip(names, combo))
        r = prefilter(vals)
        if not r or not r[0]:
            continue
        n_rej += 1
        reads = tuple(r[2]) if len(r) > 2 and r[2] else tuple(names)
        rule = str(r[1]).split(":")[0][:80]
        by_rule[rule] = by_rule.get(rule, 0) + 1
        k = (reads, tuple(vals[x] for x in reads))
        if k in seen:
            continue
        seen.add(k)
        s.add(enc.forbid_projection(vals, reads))
    return dict(n_plans=n, n_rejected=n_rej, n_clauses=len(seen), by_rule=by_rule, wall_s=time.perf_counter() - t0)


def solve(domains, oracle, theta: Theta, learned_store: LearnedStore, cost=None, max_iters: int = 500, prefilter=None):
    enc = Encoding(domains)
    log: list = []
    mono = bool(getattr(oracle, "pass_monotone", True))      # an oracle sets pass_monotone = False when "passes on T => passes on Q subset T" is not true for it
    if cost is None:
        hit = learned_store.any_pass(theta, mono)
        if hit is not None:
            log.append(dict(it=0, plan=dict(hit), ok=True, fail_rows=[], learned_kinds=[], z3_s=0.0, oracle_s=0.0,
                            source="pass_cache", status="found"))
            return dict(hit), log
    s = enc.new_solver(cost)
    n_applied = 0
    if prefilter is not None:
        pf = _apply_prefilter(enc, s, domains, prefilter)
        log.append(dict(it=0, plan=None, ok=False, fail_rows=[], learned_kinds=[], z3_s=0.0, oracle_s=pf.pop("wall_s"), source="prefilter", **pf))
    for l in learned_store.applicable(theta):
        s.add(enc.forbid(l))
        n_applied += 1
    for it in range(1, max_iters + 1):
        t0 = time.perf_counter()
        r = s.check()
        z3_s = time.perf_counter() - t0
        if str(r) != "sat":
            log.append(dict(it=it, plan=None, ok=False, fail_rows=[], learned_kinds=[], z3_s=z3_s, oracle_s=0.0,
                            source=("unsat" if str(r) == "unsat" else "z3_" + str(r)), status=("unsat" if str(r) == "unsat" else "z3_" + str(r)),
                            n_prior_learned=n_applied))
            return None, log
        plan = enc.decode(s.model())
        if learned_store.passed_superset(plan, theta, mono):
            res, src, o_s = OracleResult(ok=True, cached=True), "pass_cache", 0.0
        else:
            t1 = time.perf_counter()
            res, src = oracle(plan, theta), "oracle"
            o_s = time.perf_counter() - t1
        fails = [x.get("name") for x in res.rows if not x.get("ok", True)]
        if res.ok:
            learned_store.add_pass(plan, theta)
            log.append(dict(it=it, plan=plan, ok=True, fail_rows=[], learned_kinds=[], z3_s=z3_s, oracle_s=o_s,
                            source=src, cached=res.cached, status="found"))
            return plan, log
        ls = list(res.learned)
        if not any(l.kind == "block_exact" for l in ls) and not any(l.kind != "block_exact" for l in ls):
            ls.append(Learned("block_exact", dict(plan), theta, "fallback: oracle returned no Learned"))
        for l in ls:
            if l.theta is None and l.kind not in ("block_all_theta_a", "block_all_theta"):
                l = Learned(l.kind, l.plan, theta, l.reason, getattr(l, "free", ()))       # fixed at the queried theta (sound: the failure was observed there)
            learned_store.add(l)
            if _applies(l, theta):
                s.add(enc.forbid(l))
        # the proposed plan itself must be excluded even if the oracle returned only non-excluding Learned
        s.add(enc.forbid(Learned("block_exact", plan, theta, "proposed plan failed")))
        log.append(dict(it=it, plan=plan, ok=False, fail_rows=fails, learned_kinds=[l.kind for l in ls], z3_s=z3_s,
                        oracle_s=o_s, source=src, cached=res.cached))
    log.append(dict(it=max_iters + 1, plan=None, ok=False, fail_rows=[], learned_kinds=[], z3_s=0.0, oracle_s=0.0,
                    source="max_iters", status="max_iters"))
    return None, log
