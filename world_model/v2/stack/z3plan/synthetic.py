"""Toy oracle for smoke/tests.  Pure.

SyntheticOracle(geom, phys, a_monotone=True): 
  geom(plan) -> (bool, why)   depends on the plan only (failure => block_all_theta_a: any a, any theta)
  phys(plan, theta) -> (bool, why)  MUST be monotone in theta (fails on T => fails on every superset) and, when a_monotone,
      monotone in a (fails at a => fails at all a' >= a).  The oracle does not check this; brute_force + the caller's predicate do.
Failures return block_theta_superset (+ block_a_ge when a_monotone) and block_all_theta_a for geometry failures.
"""
from __future__ import annotations

import itertools
import time

from .types import Learned, OracleResult, Theta


class SyntheticOracle:
    def __init__(self, geom, phys, a_monotone: bool = True):
        self.geom, self.phys, self.a_monotone = geom, phys, a_monotone
        self.calls = 0

    def __call__(self, plan: dict, theta: Theta) -> OracleResult:
        t0 = time.perf_counter()
        self.calls += 1
        g_ok, g_why = self.geom(plan)
        if not g_ok:
            return OracleResult(False, [dict(name="geom", ok=False, margin=None, guarantee="proof", why=g_why)],
                                [Learned("block_all_theta_a", dict(plan), None, "geometry row: independent of a and theta")],
                                time.perf_counter() - t0)
        p_ok, p_why = self.phys(plan, theta)
        rows = [dict(name="geom", ok=True, margin=None, guarantee="proof", why=""),
                dict(name="phys", ok=p_ok, margin=None, guarantee="proof", why=p_why)]
        if p_ok:
            return OracleResult(True, rows, [], time.perf_counter() - t0)
        ls = [Learned("block_theta_superset", dict(plan), theta, "phys fails on theta => fails on supersets")]
        if self.a_monotone:
            ls.append(Learned("block_a_ge", dict(plan), theta, "phys monotone in a: fails at a => fails at a' >= a"))
        return OracleResult(False, rows, ls, time.perf_counter() - t0)


def brute_force(domains, oracle, theta: Theta) -> list:
    """every plan passing the oracle on theta (calls the oracle for each grid point; use a fresh oracle)."""
    names = [d.name for d in domains]
    out = []
    for combo in itertools.product(*[d.values for d in domains]):
        plan = dict(zip(names, combo))
        if oracle(plan, theta).ok:
            out.append(plan)
    return out
