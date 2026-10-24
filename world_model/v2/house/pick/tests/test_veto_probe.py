"""planner veto_mode: default 'rule' unchanged; 'probe' closes the veto gap (VLM may only widen; vetoed + rule inadmissible -> probe / infeasible, not the rule's plan)."""
import json
import math
import time

import numpy as np

from world_model.v2.house.pick import assign as A, planner as PL

EST_BASKET = dict(extent=(0.34, 0.22, 0.18), centroid=np.array([0.55, -0.21, 0.8]), yaw=math.pi / 2)        # prior 1.35 kg, bimanual
EST_LIGHT = dict(extent=(0.34, 0.22, 0.06), centroid=np.array([0.55, -0.21, 0.8]), yaw=math.pi / 2)         # prior 0.45 kg, bimanual by width only
EST_BIG = dict(extent=(0.40, 0.40, 0.40), centroid=np.array([0.55, -0.21, 0.8]), yaw=0.0)                    # prior 6.4 kg: nothing can lift it


def _planner(mode, answer, veto_mode=None):
    from world_model.v2.house.lang import backend as BE
    gp = PL.GraspPlanner(planner=mode, backend="mock", veto_mode=veto_mode)
    gp._be = BE.MockVLM(rules=[(r"Boxed object", lambda *a: json.dumps(answer)), (r"valid JSON", lambda *a: json.dumps(answer))], use_default=False)
    return gp


def _decide(gp, est):
    k = gp.request(1, "basket", est)
    for _ in range(100):
        if gp.status(k) == "done":
            break
        time.sleep(0.02)
    return gp.decide(1, est)


NANO = dict(arms=2, arm="both", grasp="squeeze", mass_kg=[1.5, 2.5], why="large basket")


def test_default_is_rule_and_unchanged():
    import inspect
    assert inspect.signature(PL.GraspPlanner.__init__).parameters["veto_mode"].default is None
    gp = _planner("vlm+veto", NANO)
    assert gp.veto_mode == "rule"
    d = _decide(gp, EST_BASKET)
    assert d["vetoed"] and d["plan"].mode == "bimanual" and d["plan"] is d["rule"]       # the gap: the rule's plan is returned
    assert "probe" not in d and "veto_mode" not in d


def test_probe_mode_basket_returns_probe():
    gp = _planner("vlm+veto", NANO, "probe")
    d = _decide(gp, EST_BASKET)
    assert d["vetoed"] and d["plan"].mode == "probe" and d["plan"].probe in ("WEIGH", "LIFT_TEST_2", "ASK")
    assert d["source"] == "vlm+veto+probe" and d["veto_mode"] == "probe"
    assert d["probe"]["kind"] == "probe" and d["probe"]["belief"]["m_hi"] == 2.5 and abs(d["probe"]["belief"]["m_lo"] - 1.3464) < 1e-3
    assert d["rule"].mode == "bimanual" and d["vlm"]["arms"] == 2                  # records stay consistent
    assert d["plan"].to_json()["probe"] == d["plan"].probe
    assert any("PROBE" in r for r in d["reasons"])


def test_probe_mode_widening_still_admits_returns_rule_plan():
    # VLM says one arm for a 34 cm basket (geometry veto) with a mass range that stays under the two-arm limit -> the veto is geometric, the rule plan is admissible
    ans = dict(arms=1, arm="left", grasp="pinch", mass_kg=[0.3, 0.6], why="small")
    d = _decide(_planner("vlm+veto", ans, "probe"), EST_LIGHT)
    assert d["vetoed"] and d["plan"].mode == "bimanual" and d["plan"] is d["rule"] and d["probe"] is None


def test_probe_mode_no_probe_can_fix_is_infeasible():
    ans = dict(arms=2, arm="both", grasp="squeeze", mass_kg=[5.0, 9.0], why="heavy")
    d = _decide(_planner("vlm+veto", ans, "probe"), EST_BASKET)
    assert d["plan"].mode == "infeasible" and d["probe"]["kind"] == "infeasible"
    d2 = _decide(_planner("vlm+veto", NANO, "probe"), EST_BIG)
    assert d2["plan"].mode == "infeasible"


def test_probe_mode_never_changes_unvetoed_or_rule_planner():
    ok = dict(arms=1, arm="left", grasp="pinch", mass_kg=0.2, why="small")
    est = dict(extent=(0.056, 0.056, 0.22), centroid=np.array([0.6, 0.08, 0.83]), yaw=0.0)
    for vm in ("rule", "probe"):
        d = _decide(_planner("vlm+veto", ok, vm), est)
        assert not d["vetoed"] and d["plan"].mode == "single" and d["source"] == "vlm+veto"
    d = PL.GraspPlanner(planner="rule", backend="mock", veto_mode="probe").decide(1, est)
    assert d["source"] == "rule" and d["plan"].mode == "single"


def test_widening_never_creates_an_inadmissible_execute():
    """soundness through the planner: for any VLM mass hi, probe-mode's final mode is 'single'/'bimanual' only if admissible on the widened belief."""
    for hi in (0.3, 0.7, 1.0, 1.4, 1.6, 2.5, 5.0):
        ans = dict(arms=2, arm="both", grasp="squeeze", mass_kg=[0.2, hi], why="x")
        d = _decide(_planner("vlm+veto", ans, "probe"), EST_BASKET)
        p = d["plan"]
        if p.mode == "bimanual":
            assert max(A.est_mass(EST_BASKET["extent"]), hi) <= 2 * A.SINGLE_MAX_KG
        if hi > 2 * A.SINGLE_MAX_KG:
            assert p.mode != "bimanual" and p.mode != "single"
