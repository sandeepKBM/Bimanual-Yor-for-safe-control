"""checkers/probe4.py -- CHECKER 4: probe decision value over SET-VALUED beliefs.  SIMULATION ONLY (all probe models are ASSUMPTIONS, named below).

Belief      b = {m in [m_lo, m_hi] kg} x {mu in [mu_lo, mu_hi]} (mu optional).  A set, never a point: a VLM can only WIDEN it, evidence NARROWS it.
Admissible  A(b) = candidate plans p with check(p, m, mu) true for EVERY member of b.  For checkers monotone in m (heavier is harder) and mu (slipperier is harder)
            this is the 4 corners of the box; `n_interior` adds interior samples for a non-monotone checker.  b subset b'  =>  A(b') subset A(b).
Checker     check(cand, m, mu) -> bool | (bool, why).  Default `assign_checker` = pick/assign.py (single: width <= jaw & m <= SINGLE_MAX_KG; bimanual: a hand axis in
            [MIN_HAND_SEP, MAX_SPAN] & m <= 2 SINGLE_MAX_KG; mu ignored).  Checker 3 (checkers/grasp3.py, load/slip) plugs in by passing any callable with this
            signature (e.g. `lambda c, m, mu: grasp3.check_grasp(...)`) -- nothing else here changes.
Probe       an experiment with a noise model.  True state (m*, mu*) in b  ->  posterior b' = b  intersect  [m* -+ K_SIGMA sigma_m] x [mu* -+ K_SIGMA sigma_mu]
            (K_SIGMA = 2).  Outcomes are the true states, sampled on a uniform grid (N_M x N_MU) over b = "uniform over the belief".
Value       frac(probe)  = fraction of outcomes whose posterior has A(b') non-empty ("progress" probability);   worst(probe) = 1 iff ALL outcomes do.
            net(probe)   = frac - cost_s / T_BUDGET_S.        Choose argmax net over SAFE, AVAILABLE probes with frac > 0 (criterion="worst": lexicographic (worst, net)).
decide(b, cands, probes) -> ("execute", cand)            if A(b) != {}  (first in candidate order = preference)
                         -> ("probe", probe, value_dict)  if some safe probe can make a plan admissible
                         -> ("infeasible", reason)        otherwise (no plan admissible even at the best case m_lo/mu best, or no safe probe narrows enough)
Safe probe  a lift probe is run at REDUCED acceleration (quasi-static, A_PROBE m/s^2); it is allowed only if its own load limit holds for the WORST case m_hi:
            m_hi <= n_arms * ARM_PAYLOAD_RATED_KG * PROBE_LOAD_FRACTION  and the grasp geometry exists (check(cand, ~0 kg) true for the cand's mode).  ASK is always safe.
Soundness   `execute` is returned only for a plan admissible on the whole belief (pure A(b) test); probes/ask never relax a limit.  Probe noise models are assumptions
            -- the posterior is claimed only under them (Gaussian-ish +-2 sigma; sigma values are NOT measured on the YOR model).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from world_model.v2.house.pick import assign as A

K_SIGMA = 2.0               # ASSUMPTION: posterior half-width = K_SIGMA sigma
N_M, N_MU = 41, 5           # outcome grid (true masses x true frictions)
T_BUDGET_S = 120.0          # ASSUMPTION: time budget that converts probe seconds into a unit-free cost
A_PROBE = 0.2               # m/s^2, ASSUMPTION: reduced lift acceleration of a probe
PROBE_LOAD_FRACTION = 0.9   # ASSUMPTION: quasi-static fraction of the RATED payload usable at A_PROBE (the 0.5 derate is for dynamic lifts, not this)
G = 9.81
M_TINY = 1e-3


@dataclass(frozen=True)
class Belief:
    m_lo: float
    m_hi: float
    mu_lo: Optional[float] = None
    mu_hi: Optional[float] = None

    def __post_init__(self):
        if not (self.m_lo <= self.m_hi):
            raise ValueError(f"empty mass interval [{self.m_lo}, {self.m_hi}]")
        if (self.mu_lo is None) != (self.mu_hi is None) or (self.mu_lo is not None and self.mu_lo > self.mu_hi):
            raise ValueError("bad mu interval")

    @property
    def has_mu(self):
        return self.mu_lo is not None

    def widen(self, m_lo=None, m_hi=None):
        """widen-only update (the VLM channel): the union hull with the new range."""
        return Belief(min(self.m_lo, self.m_lo if m_lo is None else m_lo), max(self.m_hi, self.m_hi if m_hi is None else m_hi), self.mu_lo, self.mu_hi)

    def corners(self):
        ms = {self.m_lo, self.m_hi}
        mus = {self.mu_lo, self.mu_hi} if self.has_mu else {None}
        return [(m, mu) for m in sorted(ms) for mu in sorted(mus, key=lambda v: -1 if v is None else v)]

    def to_json(self):
        return asdict(self)


@dataclass
class Candidate:
    """a plan option the checker can judge.  mode 'single'|'bimanual'; width = grasp width [m] (jaw width / hand-to-hand span); axis = bimanual squeeze yaw."""
    mode: str
    arm: str
    width: float
    axis: float = 0.0
    why: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def name(self):
        return f"{self.mode}:{self.arm}"

    def to_json(self):
        return asdict(self)


def assign_checker(cand: Candidate, m: float, mu: Optional[float] = None):
    """default Checker (stand-in for Checker 3): pick/assign.py feasibility.  mu is ignored (assign.py has no friction model)."""
    if cand.mode == "single":
        return A.single_feasible(cand.width, m)
    if cand.mode == "bimanual":
        ok = A.MIN_HAND_SEP_M <= cand.width <= A.MAX_SPAN_M and m <= 2 * A.SINGLE_MAX_KG
        why = []
        if not (A.MIN_HAND_SEP_M <= cand.width <= A.MAX_SPAN_M):
            why.append(f"hand span {cand.width*100:.0f} cm outside [{A.MIN_HAND_SEP_M*100:.0f}, {A.MAX_SPAN_M*100:.0f}] cm")
        if m > 2 * A.SINGLE_MAX_KG:
            why.append(f"mass {m:.2f} kg > two-arm limit {2*A.SINGLE_MAX_KG:.2f} kg")
        return ok, "; ".join(why)
    return False, f"unknown mode {cand.mode}"


def _ok(res):
    return bool(res[0] if isinstance(res, (tuple, list)) else res)


def admissible(cand, belief: Belief, checker=assign_checker, n_interior=0):
    """cand is admissible iff the checker accepts EVERY member of the belief (corners [+ interior samples])."""
    pts = list(belief.corners())
    for i in range(1, n_interior + 1):
        f = i / (n_interior + 1)
        m = belief.m_lo + f * (belief.m_hi - belief.m_lo)
        pts.append((m, None if not belief.has_mu else belief.mu_lo + f * (belief.mu_hi - belief.mu_lo)))
    return all(_ok(checker(cand, m, mu)) for m, mu in pts)


def admissible_set(belief, cands, checker=assign_checker, n_interior=0):
    return [c for c in cands if admissible(c, belief, checker, n_interior)]


# ---------------------------------------------------------------------------------------------------------- probes (ASSUMPTION models)
@dataclass(frozen=True)
class Probe:
    name: str
    kind: str                       # 'lift' | 'ask' | 'push'
    arms: int                       # arms used by the probe (0 for ask / push-with-no-load)
    sigma_m: Optional[float]        # [kg] std of the mass read-out (None: probe does not inform mass)
    sigma_mu: Optional[float]       # std of the friction read-out (None: no information on mu)
    cost_s: float
    needs: Optional[str] = None     # candidate mode whose grasp geometry the probe needs: 'single' | 'bimanual' | None
    note: str = ""

    def to_json(self):
        return asdict(self)


# every number here is an ASSUMPTION (not measured on the YOR model; the real Piper joint-torque read-out is coarse, 8-bit MIT-mode)
LIFT_TEST_1 = Probe("LIFT_TEST_1", "lift", 1, 0.20, None, 6.0, "single", "lift a few mm with ONE arm at A_PROBE; load from joint-torque (gravity) estimate")
LIFT_TEST_2 = Probe("LIFT_TEST_2", "lift", 2, 0.15, None, 8.0, "bimanual", "lift a few mm with TWO arms at A_PROBE; summed joint-torque load")
WEIGH = Probe("WEIGH", "lift", 2, 0.05, None, 15.0, "bimanual", "full bimanual lift-and-hold (quasi-static), averaged over the hold -> tighter load")
ASK = Probe("ASK", "ask", 0, 0.10, 0.10, 30.0, None, "human states the mass / slipperiness (assumed +-0.1 kg, +-0.1 mu); t_ask")
PUSH_TEST = Probe("PUSH_TEST", "push", 0, None, 0.05, 5.0, None, "slide the object with a finger; friction from push force (optional, mu only)")
DEFAULT_PROBES = (LIFT_TEST_1, LIFT_TEST_2, WEIGH, ASK, PUSH_TEST)


def probe_load_limit(probe: Probe):
    return probe.arms * A.ARM_PAYLOAD_RATED_KG * PROBE_LOAD_FRACTION


def probe_available(probe: Probe, belief: Belief, cands, checker=assign_checker):
    """(safe_and_available, why).  Safe = worst-case load within the quasi-static limit AND the grasp geometry exists."""
    if probe.sigma_mu is not None and probe.sigma_m is None and not belief.has_mu:
        return False, "no friction belief to narrow"
    if probe.kind == "lift":
        if belief.m_hi > probe_load_limit(probe):
            return False, f"unsafe: worst-case mass {belief.m_hi:.2f} kg > {probe.arms}-arm probe limit {probe_load_limit(probe):.2f} kg at a={A_PROBE} m/s^2"
        if not any(c.mode == probe.needs and _ok(checker(c, M_TINY, belief.mu_hi)) for c in cands):
            return False, f"no {probe.needs} grasp geometry to lift with"
    return True, ""


def posterior(belief: Belief, probe: Probe, m_true: float, mu_true: Optional[float] = None) -> Belief:
    """belief after the probe if the truth is (m_true, mu_true): intersect with the +-K_SIGMA sigma window."""
    lo, hi = belief.m_lo, belief.m_hi
    if probe.sigma_m is not None:
        lo, hi = max(lo, m_true - K_SIGMA * probe.sigma_m), min(hi, m_true + K_SIGMA * probe.sigma_m)
    mlo, mhi = belief.mu_lo, belief.mu_hi
    if belief.has_mu and probe.sigma_mu is not None and mu_true is not None:
        mlo, mhi = max(mlo, mu_true - K_SIGMA * probe.sigma_mu), min(mhi, mu_true + K_SIGMA * probe.sigma_mu)
    return Belief(lo, hi, mlo, mhi)


def _grid(lo, hi, n):
    return [lo] if hi <= lo or n <= 1 else [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def evaluate_probe(belief, cands, probe, checker=assign_checker, t_budget_s=T_BUDGET_S, n_interior=0):
    ok, why = probe_available(probe, belief, cands, checker)
    v = dict(probe=probe.name, available=ok, why_not=why, frac=0.0, worst=0.0, cost_s=probe.cost_s, net=-probe.cost_s / t_budget_s, n_outcomes=0,
             post_m_width=None, best_plan_counts={})
    if not ok:
        return v
    mus = _grid(belief.mu_lo, belief.mu_hi, N_MU) if belief.has_mu else [None]
    n_ok = n = 0
    counts = {}
    for m in _grid(belief.m_lo, belief.m_hi, N_M):
        for mu in mus:
            n += 1
            ad = admissible_set(posterior(belief, probe, m, mu), cands, checker, n_interior)
            if ad:
                n_ok += 1
                counts[ad[0].name] = counts.get(ad[0].name, 0) + 1
    v.update(frac=n_ok / n, worst=float(n_ok == n), net=n_ok / n - probe.cost_s / t_budget_s, n_outcomes=n, best_plan_counts=counts,
             post_m_width=None if probe.sigma_m is None else min(belief.m_hi - belief.m_lo, 2 * K_SIGMA * probe.sigma_m))
    return v


def decide(belief: Belief, cands, probes=DEFAULT_PROBES, checker=assign_checker, criterion="expected", t_budget_s=T_BUDGET_S, n_interior=0):
    """-> ("execute", cand) | ("probe", Probe, value_dict) | ("infeasible", reason_str).  criterion: 'expected' (max net) | 'worst' (lexicographic worst-case then net)."""
    assert criterion in ("expected", "worst"), criterion
    cands = list(cands)
    ad = admissible_set(belief, cands, checker, n_interior)
    if ad:
        return ("execute", ad[0])
    best_case = Belief(belief.m_lo, belief.m_lo, belief.mu_hi, belief.mu_hi) if belief.has_mu else Belief(belief.m_lo, belief.m_lo)
    if not admissible_set(best_case, cands, checker):
        return ("infeasible", f"no plan admissible even at the best case m={belief.m_lo:.2f} kg: " +
                "; ".join(f"{c.name}: {_why(checker, c, belief.m_lo, best_case.mu_lo)}" for c in cands))
    vals = [(p, evaluate_probe(belief, cands, p, checker, t_budget_s, n_interior)) for p in probes]
    useful = [(p, v) for p, v in vals if v["available"] and v["frac"] > 0]
    if not useful:
        return ("infeasible", f"no safe probe can make a plan admissible on [{belief.m_lo:.2f}, {belief.m_hi:.2f}] kg: " +
                "; ".join(f"{p.name}: " + (v["why_not"] or "posterior still inadmissible for every outcome") for p, v in vals))
    key = (lambda pv: (pv[1]["worst"], pv[1]["net"])) if criterion == "worst" else (lambda pv: pv[1]["net"])
    p, v = max(useful, key=key)
    v = dict(v, criterion=criterion, considered={q.name: dict(avail=w["available"], frac=round(w["frac"], 3), net=round(w["net"], 3), why_not=w["why_not"]) for q, w in vals})
    return ("probe", p, v)


def _why(checker, c, m, mu):
    r = checker(c, m, mu)
    return r[1] if isinstance(r, (tuple, list)) and len(r) > 1 and r[1] else "inadmissible"


def apply_probe(belief: Belief, probe: Probe, m_meas: float, mu_meas: Optional[float] = None) -> Belief:
    """closed loop: the belief after a probe returned the reading (m_meas, mu_meas): posterior around the READING (== `posterior` with the truth at the reading)."""
    m_meas = min(max(m_meas, belief.m_lo), belief.m_hi)       # a reading outside the belief is clamped (the belief is the hard envelope)
    if mu_meas is not None and belief.has_mu:
        mu_meas = min(max(mu_meas, belief.mu_lo), belief.mu_hi)
    return posterior(belief, probe, m_meas, mu_meas)
