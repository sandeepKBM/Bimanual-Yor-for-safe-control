"""Unit tests for the pan constraint layer (mock proposer, recorded evidence, fake Stage B).  SIMULATION ONLY, no sim runs."""
from world_model.v2.nero.pan.vizcoast import certify as CE
from world_model.v2.nero.pan.vizcoast import domain as D
from world_model.v2.nero.pan.vizcoast import evidence as E
from world_model.v2.nero.pan.vizcoast import facts as F
from world_model.v2.nero.pan.vizcoast import planner as P


def _plan(*acts, belief=D.PREREG):
    p = D.Plan(acts=[], beliefs=[], final=belief, state=D.INIT)
    for a in acts:
        s, b = D.apply(p.state, p.final, a)
        p = D.Plan(acts=p.acts + [a], beliefs=p.beliefs + [p.final], final=b, state=s)
    return p


UR = lambda frac=0.7, push=0.07: [D.Act("pinch", "L", "handle", frac=frac), D.Act("push", push=push), D.Act("clamp_rim", "R", "rim"),
                                  D.Act("lift")]


def test_parser_last_json_grammar_and_widen_only():
    txt = 'thinking {"a": 1} ... ' + F.WRONG
    cons, b, log = F.parse(txt, D.PREREG)
    assert cons == [] and b.mass == (0.0, 0.4) and b.dxy == D.PREREG.dxy
    assert any("narrowing" in x for x in log) and any("not trusted" in x for x in log)
    cons, b, _ = F.parse('{"facts": {"contents_mass_kg": [0, 0.9]}, "constraints": [{"forbid": {"skill": "fly"}}, '
                         '{"require": {"skill": "lift", "atom": "lid_secured"}}]}', D.PREREG)
    assert b.mass == (0.0, 0.9) and cons == [("require", "lift", "lid_secured")]
    assert F.parse(F.GARBAGE, D.PREREG)[0] == []


def test_lid_and_one_arm_rejected_with_counterexamples():
    db = E.load()
    v = CE.check(_plan(D.Act("pinch", "L", "lid"), D.Act("lift")), db)
    assert not v.ok and v.rule == "R1_lid" and v.counterexample is not None
    v = CE.check(_plan(D.Act("pinch", "L", "handle", frac=0.7), D.Act("lift")), db)
    assert not v.ok and v.rule == "R2_one_arm" and v.counterexample["hidden_mass"] == 0.0   # even the empty pan is too heavy


def test_recorded_failures_and_pregrasp_generalisation():
    db = E.load()
    v = CE.check(_plan(*UR(0.7, 0.07)), db)
    assert not v.ok and v.rule == "R3_recorded" and v.support["total"] == 60
    v = CE.check(_plan(*UR(0.6, 0.08)), db)                        # never run with push 0.08, but 0.6 pre-grasp failures transfer
    assert not v.ok and v.counterexample["cat"] == "no_clean_pregrasp"
    small = _plan(D.Act("observe_pose"), *UR(0.7, 0.07))           # +-5 mm / 2 deg still holds a recorded failure (A25:
    v = CE.check(small, db)                                         # mu 1.11, 0.38 kg off-centre, slip/tilt)
    assert not v.ok and v.counterexample["src"].endswith("robust_L70/A25.json")
    probed = _plan(D.Act("observe_pose"), D.Act("pinch", "L", "handle", frac=0.7), D.Act("probe_mass"), D.Act("push", push=0.07),
                   D.Act("clamp_rim", "R", "rim"), D.Act("lift"))
    v = CE.check(probed, db)                                        # payload known: no recorded run -> admitted, Stage B decides
    assert v.ok and sum("ASSUMED" in a for a in v.assumptions) == 2


def test_good_proposer_prunes_search_but_same_admission():
    db = E.load()
    pr = F.produce(F.mock_backend("good"), D.PREREG)
    rg = P.solve(pr["belief"], pr["constraints"], list(db))
    rn = P.solve(D.PREREG, [], list(db))
    assert rg["plan"] == rn["plan"] and "clamp_rim" in rg["plan"]
    assert rg["stats"]["certificate_calls"] < rn["stats"]["certificate_calls"]


def test_overprune_reports_no_plan_and_cegis_replans_on_fake_stageB():
    db = E.load()
    pr = F.produce(F.mock_backend("overprune"), D.PREREG)
    assert P.solve(pr["belief"], pr["constraints"], list(db))["plan"] is None
    calls = []

    def fake_sim(plan):                    # every admitted plan without observe_pose fails once at the dy=+dxy vertex
        calls.append(str(plan))
        b = plan.aim_belief()
        held = b.dxy <= D.OBS_DXY
        th = dict(dx=0.0, dy=b.dxy, yaw_deg=b.yaw, mu_scale=1.0, hidden_mass=0.0, placement="none")
        return [dict(sig=plan.signature(), theta=th, held=held, cat="held" if held else "right_claw_lost_or_missed", src="fake")]

    r = P.solve(D.PREREG, [], list(db), sim=fake_sim, max_rounds=10)
    assert r["stage"] == "B" and r["plan"].startswith("observe_pose") and len(calls) >= 2
