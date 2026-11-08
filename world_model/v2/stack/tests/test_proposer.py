"""Proposer tests: schema validator, prompt hygiene, propose / reask with the MOCK backend (no network, no model), compile -> closed-chain plan.
SIMULATION ONLY, CPU.   PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_proposer.py
Compiling with build=True plans the geometry (~35 s per task the first time, then the on-disk plan cache of closed_chain)."""
import copy
import json
import os

import numpy as np
import pytest

from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack.planners import closed_chain as CCP
from world_model.v2.stack.proposer import compile as PC
from world_model.v2.stack.proposer import mock as MK
from world_model.v2.stack.proposer import propose as PP
from world_model.v2.stack.proposer import prompt as PR
from world_model.v2.stack.proposer import schema as S

TASKS = ("box", "tray6", "pot")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")


def sup(task):
    return tuple(PR.task_spec(task)["supports"])


# ------------------------------------------------------------------------------------------------------------------ validator
@pytest.mark.parametrize("task", TASKS)
def test_fixture_reply_validates(task):
    tp = S.validate(MK.fixture_plan(task), task, sup(task))
    assert [s.call.skill for s in tp.steps] == ["lift", "move", "place", "release"]
    assert all(s.recoveries for s in tp.steps[:3]) and tp.steps[3].recoveries == []
    assert set(tp.belief) == set(S.params(task))


def bad(task, mut):
    d = copy.deepcopy(MK.fixture_plan(task))
    mut(d)
    with pytest.raises(S.PlanRejected) as e:
        S.validate(d, task, sup(task))
    return e.value


def test_validator_rejects_every_class_of_bad_reply():
    e = bad("box", lambda d: d["steps"].__setitem__(1, dict(skill="teleport", args={}, why="x")))
    assert e.stage == "schema" and any("unknown skill 'teleport'" in r for r in e.reasons)
    assert any("extra" in r or "unknown keys" in r for r in bad("box", lambda d: d.update(notes="hi")).reasons)
    assert any("missing keys" in r for r in bad("box", lambda d: d.pop("fallback")).reasons)
    assert any("outside [0.02, 0.2]" in r for r in bad("box", lambda d: d["steps"][0]["args"].update(dz=0.5)).reasons)          # no clipping
    assert any("finite number" in r for r in bad("box", lambda d: d["steps"][0]["args"].update(dz="0.1")).reasons)
    assert any("must be release()" in r for r in bad("box", lambda d: d["steps"].pop()).reasons)
    assert any("not a support" in r for r in bad("tray6", lambda d: d["steps"][2]["args"].update(support="shelf")).reasons)
    assert any("needs at least one recovery" in r for r in bad("box", lambda d: d["recoveries"].pop("1")).reasons)
    assert any("unknown recovery skill" in r for r in bad("box", lambda d: d["recoveries"]["0"].__setitem__(0, dict(skill="dance", args={}))).reasons)
    assert any("no arguments" in r for r in bad("box", lambda d: d["recoveries"]["0"].__setitem__(0, dict(skill="hold", args=dict(t=1)))).reasons)
    assert any("must be the index of a motion step" in r for r in bad("box", lambda d: d["recoveries"].update({"3": [dict(skill="hold", args={})]})).reasons)
    assert any("sanity range" in r for r in bad("box", lambda d: d["belief"].update(contents_kg=[0, 99])).reasons)
    assert any("lo 5" in r for r in bad("box", lambda d: d["belief"].update(contents_kg=[5, 1])).reasons)
    assert any("missing keys" in r for r in bad("box", lambda d: d["belief"].pop("contents_x")).reasons)
    assert any("not a (valid) belief parameter" in r for r in bad("box", lambda d: d["recoveries"]["0"][0].update(covers=dict(mass=[1, 2]))).reasons)
    assert any("outside the sanity range" in r for r in bad("box", lambda d: d["recoveries"]["0"][0].update(covers=dict(contents_kg=[0, 25]))).reasons)
    assert any("1..3 recovery" in r for r in bad("box", lambda d: d.update(fallback=[])).reasons)


def test_parse_takes_last_json_and_never_repairs():
    d = MK.fixture_plan("box")
    assert S.parse_reply("thinking {\"a\": 1} then final:\n```json\n" + json.dumps(d) + "\n```") == d
    for t in ("", "   ", "no json here"):
        with pytest.raises(S.PlanRejected) as e:
            S.parse_reply(t)
        assert e.value.stage == "parse"
    with pytest.raises(S.PlanRejected):                                       # truncated reply: only an inner fragment decodes; it cannot pass the schema
        S.validate_reply("{\"belief\": {\"contents_kg\": [0, 5]}, \"steps\": [ lift", "box")


# ------------------------------------------------------------------------------------------------------------------ prompt
def test_prompt_has_no_hidden_truth():
    w = SC.BoxWorld(contents_kg=3.1234, contents_xy=(0.0123, -0.0456))
    txt = PR.SYSTEM + PR.user_prompt("box", world=w)
    for needle in ("3.1234", "0.0123", "-0.0456", "contents_mu"):
        assert needle not in txt
    tw = SC.TrayWorld(n_cups=6, fills_kg=[0.0777] * 6)
    assert "0.0777" not in PR.user_prompt("tray6", world=tw)
    assert "(0.5, 5.0)" not in txt and "0.5, 5.0" not in txt                      # the prior box is not shown
    f = PR.scene_facts("box")
    assert f["task"]["goal_xy"] == [0.18, -0.22] and f["task"]["goal_support"] == "shelf"
    assert f["robot"]["payload_kg_per_arm_datasheet"] == 3.0
    assert "teleport" not in txt and set(f["vocabulary"]["recoveries"]) == set(S.RECOVERY_SKILLS)
    assert PR.user_prompt("pot") == PR.user_prompt("pot")                          # deterministic (cache replay)


# ------------------------------------------------------------------------------------------------------------------ propose / reask (mock backend)
def test_propose_writes_cache_then_replays_offline(tmp_path):
    cache = tmp_path / "c"
    plan, rec = PP.propose("box", backend_spec="mock:fixture", cache_dir=cache)
    assert plan["status"] == "ok" and rec["status"] == "ok" and rec["cached"] is False
    assert rec["model"] == "scripted" and rec["spec"] == "mock:fixture" and rec["request_hash"] and os.path.exists(rec["raw_reply_path"])
    assert rec["caveats"] and rec["reasoning"] is not None and rec["plan_id"] == plan["plan_id"]
    saved = json.load(open(rec["raw_reply_path"]))
    assert "TASK_ID: box" in saved["request"]["text"] and saved["request"]["system"]
    plan2, rec2 = PP.propose("box", backend_spec="mock:fixture", cache_dir=cache, offline=True)       # replay
    assert rec2["cached"] is True and plan2 == plan and rec2["request_hash"] == rec["request_hash"]
    with pytest.raises(PP.ProposerCacheMiss):                                                         # a miss offline fails loudly
        PP.propose("pot", backend_spec="mock:fixture", cache_dir=cache, offline=True)


def test_offline_real_spec_never_builds_a_network_backend(tmp_path, monkeypatch):
    import world_model.v2.nero.pan.llm as LLM
    monkeypatch.setattr(LLM, "make", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network backend built")))
    with pytest.raises(PP.ProposerCacheMiss):
        PP.propose("box", backend_spec="codex:gpt-5.6-sol:high", cache_dir=tmp_path / "c", offline=True)
    with pytest.raises(PP.ProposerCacheMiss):                                                         # env BIMANUAL_NO_NETWORK=1 (autouse) does the same without offline=
        PP.propose("box", backend_spec="codex:gpt-5.6-sol:high", cache_dir=tmp_path / "c")
    # a recorded codex reply replays: write it through a mock-made CachedVLM entry keyed like the real one
    from world_model.v2.house.lang.backend import CachedVLM
    be = PP.make_backend("codex:gpt-5.6-sol:high", tmp_path / "c2", offline=True)
    assert isinstance(be, CachedVLM) and be.mode == "ro" and be.inner.model == "gpt-5.6-sol"


def test_bad_replies_are_plan_rejected_not_repaired(tmp_path):
    p, rec = PP.propose("box", backend_spec="mock:malformed", cache_dir=tmp_path / "m")
    assert p["status"] == "plan_rejected" and p["stage"] in ("parse", "schema") and rec["status"] == "plan_rejected" and rec["reply_head"]
    p, rec = PP.propose("box", backend_spec="mock:unknown_skill", cache_dir=tmp_path / "u")
    assert p["status"] == "plan_rejected" and p["stage"] == "schema" and any("teleport" in r for r in p["reasons"])
    assert rec["reasons"] == p["reasons"]


def test_reask_names_the_uncovered_part(tmp_path):
    plan, _ = PP.propose("box", backend_spec="mock:narrow", cache_dir=tmp_path / "c")
    b = plan["belief"]["contents_kg"]
    probe = dict(contents_kg=[b[1] + 1.0, b[1] + 2.0])
    new, rec = PP.reask("box", plan, probe, backend_spec="mock:narrow", cache_dir=tmp_path / "c")
    assert new["status"] == "ok" and rec["kind"] == "reask" and rec["probe_set"] == probe
    assert rec["uncovered"] == {"contents_kg": [[probe["contents_kg"][0], probe["contents_kg"][1]]]} and rec["previous_plan_id"] == plan["plan_id"]
    txt = json.load(open(rec["raw_reply_path"]))["request"]["text"]
    assert "NEW MEASUREMENT" in txt and str(probe["contents_kg"][0]) in txt
    with pytest.raises(ValueError):
        PP.reask("box", dict(status="plan_rejected"), probe, backend_spec="mock:narrow", cache_dir=tmp_path / "c")


def test_probe_in_belief():
    r = PC.probe_in_belief("box", dict(contents_kg=[1.0, 3.0], contents_x=[-0.05, 0.05], contents_y=[-0.08, 0.08]), 1.2, 2.9)
    assert r["inside"] and r["param"] == "contents_kg" and set(r["not_probed"]) == {"contents_x", "contents_y"}
    assert not PC.probe_in_belief("box", dict(contents_kg=[1.0, 3.0], contents_x=[0, 0], contents_y=[0, 0]), 0.5, 2.9)["inside"]
    t = PC.probe_in_belief("tray6", dict(fill_total_kg=[0.0, 0.5]), 6 * 0.30 + 0.1, 6 * 0.30 + 0.4)             # tray: mugs' own mass subtracted
    assert t["inside"] and abs(t["probe"][0] - 0.1) < 1e-9


def test_probe_in_belief_uses_b_within_prior():
    """11-07 (clarity report (f)1): B is intersected with the prior box; a B covering the prior is flagged (Rule 2 cannot trigger)."""
    xy = dict(contents_x=[0, 0], contents_y=[0, 0])
    wide = PC.probe_in_belief("box", dict(contents_kg=[0.0, 20.0], **xy), 1.2, 2.9)              # the sanity range the 11-05 replies used
    assert wide["inside"] and wide["covers_prior"] and wide["prior"] == [0.5, 5.0] and wide["B_eff"] == [0.5, 5.0]
    part = PC.probe_in_belief("box", dict(contents_kg=[0.0, 3.0], **xy), 0.6, 2.9)                # clipped below: B_eff = [0.5, 3]
    assert part["inside"] and not part["covers_prior"] and part["B_eff"] == [0.5, 3.0] and part["B"] == [0.0, 3.0]
    assert not PC.probe_in_belief("box", dict(contents_kg=[0.0, 3.0], **xy), 0.6, 3.5)["inside"]
    far = PC.probe_in_belief("box", dict(contents_kg=[6.0, 8.0], **xy), 1.2, 2.9)                 # B disjoint from the prior
    assert far["B_eff"] is None and not far["inside"] and not far["covers_prior"]


# ------------------------------------------------------------------------------------------------------------------ compile -> closed-chain plan
@pytest.fixture(scope="module", params=TASKS)
def compiled(request):
    task = request.param
    plan = dict(MK.fixture_plan(task), status="ok", task=task)
    return task, PC.compile_plan(plan, task)


def test_compile_fixture_matches_default_path_and_reaches_goal(compiled):
    task, cp = compiled
    assert cp.names == ["s0_lift", "s1_move", "s2_place"] and cp.wp.backups == ("set_down",) * 3 and cp.fallback == ("let_go",)
    assert cp.goal["miss_m"] < 1e-9 and cp.goal["end_support"] == cp.goal["goal_support"]
    g = CCP.geometry_for(CCP._nominal_inputs(task, cp.wp))
    g0 = CCP.geometry_for(CCP._nominal_inputs(task))                                      # the default plan
    assert list(g.bounds) == cp.names and np.abs(g.nodes - g0.nodes).max() < 1e-9 and abs(g.info["dy"] - g0.info["dy"]) < 1e-12
    for s in CCP.SIDES:
        assert g.fk_pos[s].max() < 1e-6 and g.margin[s].min() > 0.0
    w = CCP._nominal_inputs(task).world
    assert np.allclose(w.obj_xy[0] + g.offs[-1][0], cp.goal["goal_xy"][0]) and np.allclose(w.obj_xy[1] + g.offs[-1][1], cp.goal["goal_xy"][1])
    plan = CCP.plan_for(CCP._nominal_inputs(task, cp.wp), 0.5)                              # timing built for every segment, named after the steps
    assert list(plan.pieces) == cp.names and all(plan.pieces[n].T > 0 for n in cp.names)
    j = CCP.ClosedChainPlanner().plan(None, CCP.SkillCall("carry_task", dict(task=task, a_plan=0.5)), None, inputs=CCP._nominal_inputs(task, cp.wp))
    assert [s.mode for s in j.segments] == cp.names + ["release"]
    assert j.segments[2].meta["backup_phases"][1] == ("retrace", ("s2_place", "s1_move", "s0_lift")) and j.segments[3].meta["backup_phases"] is None
    j0 = CCP.ClosedChainPlanner().plan(None, CCP.SkillCall("carry_task", dict(task=task, a_plan=0.5)), None, inputs=CCP._nominal_inputs(task))
    assert [s.mode for s in j0.segments] == list(CCP.SEGS) and j0.segments[2].meta["backup_phases"][1] == ("retrace", ("place", "carry", "lift"))   # default unchanged


def test_compiled_plan_roundtrips_json(compiled):
    task, cp = compiled
    cp2 = PC.CompiledPlan.from_json(json.loads(json.dumps(cp.to_json())))
    assert cp2.wp == cp.wp and cp2.belief == cp.belief and cp2.fallback == cp.fallback and hash(cp2.wp) == hash(cp.wp)


def test_compile_rejections_name_the_reason():
    def comp(task, mut):
        d = dict(MK.fixture_plan(task), status="ok", task=task)
        mut(d)
        with pytest.raises(S.PlanRejected) as e:
            PC.compile_plan(d, task, build=False)
        assert e.value.stage == "compile"
        return e.value.reasons

    assert any("tolerance" in r for r in comp("tray6", lambda d: d["steps"][1]["args"].update(dx=-0.25)))                  # ends 15 cm past the goal

    def drag(d):                                                                                                           # slides along the counter without lifting
        d["steps"] = [dict(skill="move", args=dict(dx=-0.1, dy=0.0, dz=0.0), why="slide"), d["steps"][2], d["steps"][3]]
        d["recoveries"] = {"0": [dict(skill="hold", args={})], "1": [dict(skill="hold", args={})]}
    assert any("no dragging" in r for r in comp("tray6", drag))

    def below(d):
        d["steps"][1]["args"].update(dx=0.0, dz=-0.03)                                                                      # lift 5 cm, then down 3 cm, then place: fine ...
        d["steps"][0]["args"].update(dz=0.02)                                                                               # ... but 2 cm lift then -3 cm goes below the start support
    assert any("below it" in r for r in comp("tray6", below))
    assert any("not over the support" in r for r in comp("tray6", lambda d: d["steps"][1]["args"].update(dy=0.25, dx=-0.1)))   # centre off the counter
    assert any("task goal is on 'shelf'" in r for r in comp("box", lambda d: d["steps"][2]["args"].update(support="counter")))


def test_planner_refusal_is_plan_rejected_with_reason():
    d = dict(MK.fixture_plan("box"), status="ok", task="box")
    d["steps"][0]["args"]["dz"] = 0.17                                                              # underside 2 cm above the shelf top: less than the planner's 3 cm clearance
    with pytest.raises(S.PlanRejected) as e:
        PC.compile_plan(d, "box")
    assert e.value.stage == "planner" and any("collides" in r or "unreachable" in r or "reason=" in r for r in e.value.reasons)
