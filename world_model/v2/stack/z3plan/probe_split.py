"""Z3 chooses how to split a probe reading: exists edges, forall bins, exists plan (CEGIS over the edges).  Pure.

Reading model: y = offset + c + n, |n| <= noise, c = contents.  Subtracting the offset, the reading axis is r = y - offset = c + n.
Bin j is the reading interval r in [e_j, e_{j+1}] (first bin open below, last open above; edges e_1..e_{k-1} are the cut points).
Worlds in bin j: c in theta0 with exists n, |n|<=noise, c + n in [e_j, e_{j+1}]   <=>   c in [e_j - noise, e_{j+1} + noise]
  theta_j = [max(theta0.lo, e_j - noise), min(theta0.hi, e_{j+1} + noise)]       (one noise width per side)
NOTE: the task text wrote 2*noise; with |n| <= noise and the reading interval defined on r = y - offset, the exact formula is 1*noise.
(2*noise would be the interval-arithmetic result if the edges were the readings of two separate measurements.)  The offset cancels.

Edges: Z3 Ints over a grid of the contents axis (step GRID_STEP_KG over theta0, ASSUMED), strictly increasing.
CEGIS: k = 1 (no probe), 2, ... max_bins.  For each candidate edge set every bin is solved (solve() with the shared LearnedStore);
an unsolvable bin theta T excludes every future bin whose theta contains T (a plan failing for T fails for its supersets), as integer
bounds on that bin position's lower/upper edge index.  The unsolvable list persists across k; solved bins are cached by the pass cache.
"""
from __future__ import annotations

import math

import z3

from .loop import LearnedStore, solve
from .types import ReadingModel, Theta

GRID_STEP_KG = 0.05      # ASSUMED edge grid on the contents axis


def bin_thetas(theta0: Theta, noise: float, edges) -> list:
    """edges = increasing cut points on the contents axis (len k-1) -> [Theta_j or None if empty] for the k bins."""
    lo_e = [-math.inf] + list(edges)
    hi_e = list(edges) + [math.inf]
    out = []
    for a, b in zip(lo_e, hi_e):
        lo, hi = max(theta0.lo, a - noise), min(theta0.hi, b + noise)
        out.append(Theta(lo, hi) if lo <= hi else None)
    return out


def split(domains, oracle, theta0: Theta, reading: ReadingModel, max_bins: int = 4, cost=None, store: LearnedStore = None,
          grid_step: float = GRID_STEP_KG, max_rounds: int = 2000, max_iters: int = 500, prefilter=None):
    store = store if store is not None else LearnedStore()
    noise = reading.noise_kg
    n_g = int(math.floor((theta0.hi - theta0.lo) / grid_step + 1e-9))
    grid = [theta0.lo + i * grid_step for i in range(n_g + 1)]
    log: list = []
    unsolvable: list = []          # [Theta]
    solved: dict = {}              # (lo, hi) -> plan
    tol = 1e-9

    for k in range(1, max_bins + 1):
        s = z3.Solver()
        E = [z3.Int(f"e{j}") for j in range(1, k)]        # grid indices of the k-1 cut points
        for j, e in enumerate(E):
            s.add(e >= 0, e <= n_g)
            if j:
                s.add(e > E[j - 1])

        def add_excl(T: Theta):
            # bin position p has lower edge L (-1 = -inf) and upper edge U (n_g+1 = +inf); theta_p contains T iff
            # L-edge - noise <= T.lo and U-edge + noise >= T.hi.   a = largest grid idx with g - noise <= T.lo; b = smallest with g + noise >= T.hi
            a_idx = max([i for i, g in enumerate(grid) if g - noise <= T.lo + tol], default=-1)
            b_idx = min([i for i, g in enumerate(grid) if g + noise >= T.hi - tol], default=n_g + 1)
            for p in range(k):
                L = E[p - 1] if p > 0 else None
                U = E[p] if p < k - 1 else None
                cl = z3.BoolVal(True) if L is None else (L <= a_idx)
                cu = z3.BoolVal(True) if U is None else (U >= b_idx)
                s.add(z3.Not(z3.And(cl, cu)))

        for T in unsolvable:
            add_excl(T)
        for rnd in range(max_rounds):
            if str(s.check()) != "sat":
                log.append(dict(k=k, round=rnd, event="unsat_edges"))
                break
            m = s.model()
            edges = [grid[m.eval(e, model_completion=True).as_long()] for e in E]
            thetas = bin_thetas(theta0, noise, edges)
            bins, bad = [], []
            for j, T in enumerate(thetas):
                if T is None:
                    continue
                key = (T.lo, T.hi)
                if key in solved:
                    plan = solved[key]
                else:
                    plan, sl = solve(domains, oracle, T, store, cost, max_iters=max_iters, prefilter=prefilter)
                    log.append(dict(k=k, round=rnd, bin=j, theta=key, solved=plan is not None, iters=len(sl),
                                    oracle_calls=sum(1 for x in sl if x.get("source") == "oracle"),
                                    z3_s=sum(x["z3_s"] for x in sl), oracle_s=sum(x["oracle_s"] for x in sl), solve_log=sl))
                    if plan is not None:
                        solved[key] = plan
                    elif sl[-1].get("status") != "unsat":       # max_iters or z3_unknown (review 11-16): not a proof of "unsolvable"
                        # the cap is NOT "unsolvable": calling it so would make the split edges unsound.  Stop with no verdict.
                        log.append(dict(k=k, round=rnd, event="capped", theta=key, max_iters=max_iters, status=sl[-1].get("status")))
                        return None, log
                if plan is None:
                    bad.append(T)
                else:
                    bins.append((T, plan))
            if not bad:
                log.append(dict(k=k, round=rnd, event="found", edges=edges))
                return bins, log
            for T in bad:
                if not any(u.contains(T) and T.contains(u) for u in unsolvable):
                    unsolvable.append(T)
                add_excl(T)
            log.append(dict(k=k, round=rnd, event="cegis_cut", edges=edges, bad=[(t.lo, t.hi) for t in bad]))
        else:                                              # max_rounds used up without an unsat over the edges: a cap, not "no split with k bins"
            log.append(dict(k=k, event="capped", max_rounds=max_rounds))
            return None, log
    log.append(dict(event="no_split_up_to", max_bins=max_bins))
    return None, log
