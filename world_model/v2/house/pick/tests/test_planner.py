"""planner: VLM parse, hard veto, caching, fallbacks -- MOCK backend only (no daemon)."""
import json
import math
import time

import numpy as np

from world_model.v2.house.pick import assign as A, planner as PL

EST_BOTTLE = dict(extent=(0.056, 0.056, 0.22), centroid=np.array([0.6, 0.08, 0.83]), yaw=0.0)
EST_CRATE = dict(extent=(0.30, 0.20, 0.14), centroid=np.array([0.5, -0.21, 0.79]), yaw=math.pi / 2)


def _wait(gp, k):
    for _ in range(100):
        if gp.status(k) == "done":
            return
        time.sleep(0.02)
    raise AssertionError("VLM request did not finish")


def _planner(mode, rules):
    from world_model.v2.house.lang import backend as BE
    gp = PL.GraspPlanner(planner=mode, backend="mock")
    gp._be = BE.MockVLM(rules=rules, use_default=False)
    return gp


def test_parse_defensive():
    assert PL.parse_vlm_plan('blah {"arms": 1, "arm": "left", "grasp": "pinch", "why": "x"} tail')["arm"] == "left"
    assert PL.parse_vlm_plan("{'arms': 2, 'arm': 'both',}")["arms"] == 2
    assert PL.parse_vlm_plan('{"arms": 1, "arm": "middle"}') is None
    assert PL.parse_vlm_plan("no json") is None
    assert PL.parse_vlm_plan("") is None


def test_rule_only_never_calls_vlm():
    gp = PL.GraspPlanner(planner="rule", backend="mock")
    gp.request(1, "bottle", EST_BOTTLE)
    d = gp.decide(1, EST_BOTTLE)
    assert d["source"] == "rule" and gp.records == [] and gp._be is None


def test_veto_overrides_infeasible_vlm_plan():
    bad = json.dumps(dict(arms=1, arm="left", grasp="pinch", mass_kg=0.5, why="looks small"))        # one arm for the 30 cm crate -> width > jaw opening
    gp = _planner("vlm+veto", [(r"Boxed object", lambda *a: bad), (r"valid JSON", lambda *a: bad)])
    k = gp.request(7, "crate", EST_CRATE)
    _wait(gp, k)
    d = gp.decide(7, EST_CRATE)
    assert d["vetoed"] and d["plan"].mode == "bimanual"                       # rule's plan
    assert any("jaw opening" in r for r in d["reasons"])
    assert d["vlm"]["arms"] == 1 and d["rule"].mode == "bimanual"
    assert gp.agree(d) is False
    assert gp.records[0]["raw"][0] == bad and gp.records[0]["latency_s"] >= 0


def test_vlm_choice_kept_when_feasible_and_unvetoed():
    ok = json.dumps(dict(arms=1, arm="right", grasp="pinch", mass_kg=0.2, why="either arm"))           # feasible, but a different arm than the rule's (left)
    gp = _planner("vlm+veto", [(r"Boxed object", lambda *a: ok)])
    k = gp.request(2, "bottle", EST_BOTTLE)
    _wait(gp, k)
    d = gp.decide(2, EST_BOTTLE)
    assert not d["vetoed"] and d["plan"].arm == "right" and d["rule"].arm == "left" and gp.agree(d) is False


def test_veto_on_unreachable_arm():
    ok = json.dumps(dict(arms=1, arm="left", grasp="pinch", mass_kg=0.2, why="x"))
    gp = _planner("vlm+veto", [(r"Boxed object", lambda *a: ok)])
    k = gp.request(3, "bottle", EST_BOTTLE)
    _wait(gp, k)
    d = gp.decide(3, EST_BOTTLE, reach=dict(left=False, right=True))
    assert d["vetoed"] and "cannot reach" in d["reasons"][0]


def test_veto_mass_guess_cannot_relax_limit():
    heavy = json.dumps(dict(arms=1, arm="left", grasp="pinch", mass_kg=3.0, why="solid"))
    gp = _planner("vlm+veto", [(r"Boxed object", lambda *a: heavy)])
    k = gp.request(4, "bottle", EST_BOTTLE)
    _wait(gp, k)
    d = gp.decide(4, EST_BOTTLE)
    assert d["vetoed"] and any("payload" in r for r in d["reasons"])


def test_fallbacks_down_unparseable_pending_and_cache():
    def boom(*a):
        raise RuntimeError("daemon down")
    gp = _planner("vlm+veto", [(r"Boxed object", boom)])
    k = gp.request(5, "bottle", EST_BOTTLE)
    _wait(gp, k)
    d = gp.decide(5, EST_BOTTLE)
    assert d["source"] == "rule" and "daemon down" in d["fallback"] and d["vlm"] is None
    gp2 = _planner("vlm+veto", [(r"Boxed object", lambda *a: "I think left."), (r"valid JSON", lambda *a: "still no")])
    k = gp2.request(6, "bottle", EST_BOTTLE)
    _wait(gp2, k)
    d = gp2.decide(6, EST_BOTTLE)
    assert "unparseable" in d["fallback"] and gp2.records[0]["repaired"] and len(gp2.records[0]["raw"]) == 2
    n = len(gp2.backend().calls)
    gp2.request(6, "bottle", EST_BOTTLE)                                       # cached: no new call
    assert len(gp2.backend().calls) == n
    d0 = gp2.decide(99, EST_BOTTLE)                                            # never requested
    assert d0["status"] == "pending" and d0["plan"].mode == "single"


def test_default_mock_prior_runs_and_is_json():
    gp = PL.GraspPlanner(planner="vlm+veto", backend="mock")
    k = gp.request(8, "crate", EST_CRATE)
    _wait(gp, k)
    d = gp.decide(8, EST_CRATE)
    assert d["vlm"]["arms"] == 2 and not d["vetoed"] and d["plan"].mode == "bimanual"
    json.dumps(gp.records)
