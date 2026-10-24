"""Checker 4 (probe decision value): admissibility over set beliefs, probe selection, infeasible, monotonicity / soundness.  Pure python, no sim."""
import itertools

from world_model.v2.house.checkers import probe4 as PB
from world_model.v2.house.pick import assign as A

SINGLE = PB.Candidate("single", "left", 0.05)
BIM = PB.Candidate("bimanual", "both", 0.34, 0.0)
L1, L2 = A.SINGLE_MAX_KG, 2 * A.SINGLE_MAX_KG


def test_admissible_needs_all_members():
    assert PB.admissible(SINGLE, PB.Belief(0.1, L1))
    assert not PB.admissible(SINGLE, PB.Belief(0.1, L1 + 0.01))
    assert PB.admissible(BIM, PB.Belief(0.1, L2)) and not PB.admissible(BIM, PB.Belief(0.1, L2 + 0.01))


def test_execute_prefers_candidate_order():
    d = PB.decide(PB.Belief(0.1, 0.5), [SINGLE, BIM])
    assert d[0] == "execute" and d[1].name == "single:left"


def test_basket_belief_probes_with_a_safe_probe():
    b = PB.Belief(1.0, 2.5)                       # widened mass hi 2.5 > two-arm limit 1.5
    d = PB.decide(b, [BIM])
    assert d[0] == "probe" and d[1].name in ("WEIGH", "LIFT_TEST_2", "ASK")
    assert 0 < d[2]["frac"] <= 1 and d[2]["cost_s"] > 0
    # the chosen lift probe is safe at the worst case
    if d[1].kind == "lift":
        assert b.m_hi <= PB.probe_load_limit(d[1])
    assert d[2]["considered"]["LIFT_TEST_1"]["avail"] is False                # no single geometry (hand span only)


def test_unsafe_probe_excluded_and_ask_is_fallback():
    b = PB.Belief(1.0, 3.5)                       # 3.5 kg > 2-arm quasi-static limit (2.7): no lift probe is safe
    d = PB.decide(b, [BIM])
    assert d[0] == "probe" and d[1].name == "ASK"


def test_no_probe_can_help_is_infeasible():
    # best case already above the two-arm limit
    d = PB.decide(PB.Belief(2.0, 3.0), [SINGLE, BIM])
    assert d[0] == "infeasible" and "best case" in d[1]
    # geometry impossible (hand span too small, too wide for a jaw)
    d = PB.decide(PB.Belief(0.1, 0.9), [PB.Candidate("single", "left", 0.10), PB.Candidate("bimanual", "both", 0.10)])
    assert d[0] == "infeasible"
    # the whole belief is the single point at the limit but the probe window cannot get hi below it
    d = PB.decide(PB.Belief(L2, L2 + 1.0), [BIM], probes=(PB.WEIGH, PB.LIFT_TEST_2))
    assert d[0] == "infeasible" and "no safe probe" in d[1]


def test_criterion_worst_prefers_probe_that_resolves_everything():
    b = PB.Belief(1.3, 1.7)                       # narrow: WEIGH (+-0.1) cannot always bring hi under 1.5, ASK (+-0.2) neither; worst case = all outcomes
    d = PB.decide(b, [BIM], criterion="worst")
    assert d[0] == "probe"
    assert d[2]["worst"] in (0.0, 1.0)


def test_closed_loop_probe_then_execute():
    b = PB.Belief(1.0, 2.5)
    d = PB.decide(b, [BIM])
    assert d[0] == "probe"
    b2 = PB.apply_probe(b, d[1], m_meas=1.1)      # the reading says light: posterior hi = 1.1 + 2 sigma <= 1.5
    assert PB.decide(b2, [BIM])[0] == "execute"
    b3 = PB.apply_probe(b, d[1], m_meas=2.2)      # the reading says heavy: stays inadmissible, best case above the limit -> infeasible
    assert PB.decide(b3, [BIM])[0] == "infeasible"


def test_friction_probe_with_custom_checker():
    # a Checker-3-like plug-in: grasp slips if mu < 0.4 (monotone in mu).  Only PUSH_TEST / ASK inform mu.
    def chk(c, m, mu):
        return (m <= L1) and (mu is None or mu >= 0.4)
    b = PB.Belief(0.2, 0.5, 0.2, 0.8)
    d = PB.decide(b, [SINGLE], checker=chk)
    assert d[0] == "probe" and d[1].name in ("PUSH_TEST", "ASK")
    assert PB.decide(PB.Belief(0.2, 0.5, 0.5, 0.8), [SINGLE], checker=chk)[0] == "execute"


def test_monotone_set_inclusion_and_soundness():
    """b subset b'  =>  A(b') subset A(b);  an executed plan is admissible on every member; widening can only move execute -> probe/infeasible."""
    order = {"execute": 0, "probe": 1, "infeasible": 2}
    cands = [SINGLE, BIM]
    his = [0.3, 0.7, 0.75, 1.0, 1.5, 1.6, 2.5]
    for lo, hi in itertools.product([0.1, 0.5, 1.0], his):
        if lo > hi:
            continue
        for dlo, dhi in itertools.product([0.0, 0.3], [0.0, 0.2, 1.0]):
            b, bw = PB.Belief(lo, hi), PB.Belief(max(0.0, lo - dlo), hi + dhi)
            assert {c.name for c in PB.admissible_set(bw, cands)} <= {c.name for c in PB.admissible_set(b, cands)}
            dn, dw = PB.decide(b, cands), PB.decide(bw, cands)
            if dw[0] == "execute":
                assert PB.admissible(dw[1], b)               # inadmissible-on-narrow can never be executed on wide
                for m in (bw.m_lo, bw.m_hi, 0.5 * (bw.m_lo + bw.m_hi)):
                    assert PB.assign_checker(dw[1], m)[0]            # sound: feasible for every member
            if dn[0] != "execute":
                assert dw[0] != "execute"                              # widening never creates an execute
            if dhi >= 0 and dlo == 0 and dn[0] == "infeasible":        # raising only the upper end never rescues an infeasible belief
                assert dw[0] == "infeasible"
            assert order[dw[0]] >= order[dn[0]] or dlo > 0             # (lowering lo may turn infeasible into probe: that belief contains lighter, hence resolvable, masses)
