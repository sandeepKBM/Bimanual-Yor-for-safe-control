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
def _no_network(monkeypatch, request):
    """every test is offline.  Tests whose name contains "_v2" run the schema-v2 format (the default); all others run the 11-05..11-09 format (BIMANUAL_LEGACY_PROPOSER_V1=1),
    which stays replayable for the old response caches."""
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")
    if "_v2" in request.node.name:
        monkeypatch.delenv("BIMANUAL_LEGACY_PROPOSER_V1", raising=False)
    else:
        monkeypatch.setenv("BIMANUAL_LEGACY_PROPOSER_V1", "1")


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


# ------------------------------------------------------------------------------------------------------------------ schema v2 (11-09)
BEHIND = ("tray6_behind", "box_behind")
ALL_V2 = TASKS + BEHIND


def sup2(task):
    return tuple(PR.task_spec(task)["supports"])


def bad2(task, mut, plan=None):
    d = copy.deepcopy(plan if plan is not None else MK.fixture_plan(task))
    mut(d)
    with pytest.raises(S.PlanRejected) as e:
        S.validate(d, task, sup2(task))
    assert e.value.stage == "schema"
    return e.value.reasons


@pytest.mark.parametrize("task", ALL_V2)
def test_v2_fixture_validates(task):
    assert not S.legacy_v1()
    d = MK.fixture_plan(task)
    assert set(d) == set(S.V2_KEYS) and "belief" not in d
    tp = S.validate(d, task, sup2(task))
    assert tp.schema == "v2" and tp.belief == {} and tp.probes == ["weigh"]
    assert [p["id"] for p in tp.phenomena if "measure" in p["handle"]] == ["PH1"]
    td = tp.to_dict()
    assert set(td) == {"status", "task", "phenomena", "probes", "steps", "recoveries", "fallback"}
    assert S.reply_of(dict(td, plan_id="x")) == dict({k: td[k] for k in S.V2_KEYS}, probes=[{"skill": q} for q in td["probes"]])      # reply format: probes are objects (round trip fixed 11-09)
    skills = [s.call.skill for s in tp.steps]
    assert skills == (["lift", "base_move", "place", "release"] if task in BEHIND else ["lift", "move", "place", "release"])


def test_v2_every_bad_reply_class_is_rejected_with_its_reason():
    t = "tray6"
    assert any("belief: not part of this format" in r for r in bad2(t, lambda d: d.update(belief=dict(fill_total_kg=[0, 1]))))
    assert any("covers: not part of this format" in r for r in bad2(t, lambda d: d["recoveries"]["0"][0].update(covers=dict(x=[1, 2]))))
    assert any("unknown phenomenon 'PH9'" in r for r in bad2(t, lambda d: d["phenomena"][0].update(id="PH9")))
    assert any("duplicate phenomenon 'PH1'" in r for r in bad2(t, lambda d: d["phenomena"].append(copy.deepcopy(d["phenomena"][0]))))
    assert any("does not measure 'PH2'" in r for r in bad2(t, lambda d: d["phenomena"][1].update(handle=dict(measure="weigh"))))
    assert any("unknown skill 'tap'" in r for r in bad2(t, lambda d: d["phenomena"][0].update(handle=dict(measure="tap"))))
    assert any("not built" in r for r in bad2(t, lambda d: d["phenomena"][1].update(handle=dict(measure="hold_moment"))))
    assert any("no phenomenon names it" in r for r in bad2(t, lambda d: d["phenomena"][0].update(handle=dict(unmeasurable="skip"))))            # probe without a handle
    assert any("is not in probes" in r for r in bad2(t, lambda d: d.update(probes=[])))                                                       # handle without a probe
    assert any("listed twice" in r for r in bad2(t, lambda d: d["probes"].append(dict(skill="weigh"))))
    assert any("not a built measuring skill" in r for r in bad2(t, lambda d: d["probes"].append(dict(skill="shake_listen"))))
    assert any("exactly one of" in r for r in bad2(t, lambda d: d["phenomena"][0].update(handle=dict(measure="weigh", unmeasurable="x"))))
    assert any("missing keys" in r for r in bad2(t, lambda d: d.pop("probes")))
    assert any("unknown keys" in r for r in bad2(t, lambda d: d["phenomena"][0].update(mass_kg=3)))
    assert any("a number (3.2)" in r for r in bad2(t, lambda d: d["phenomena"][0].update(handle=dict(unmeasurable=3.2))))
    assert any("numbers are allowed only in step args" in r for r in bad2(t, lambda d: d["fallback"][0].update(args=dict(t=1))))
    assert any("missing keys ['name']" in r for r in bad2(t, lambda d: d["phenomena"].append(dict(id="NEW", why="w", handle=dict(unmeasurable="u")))))
    assert any("name" in r and "unknown keys" in r for r in bad2(t, lambda d: d["phenomena"][0].update(name="x")))                          # name only on NEW
    assert any("must be a non-empty string" in r for r in bad2(t, lambda d: d["phenomena"][0].update(why="x" * 401)))
    assert any("unknown skill 'teleport'" in r for r in bad2(t, lambda d: d["steps"].__setitem__(1, dict(skill="teleport", args={}, why="x"))))     # step rules unchanged
    assert any("needs at least one recovery" in r for r in bad2(t, lambda d: d["recoveries"].pop("1")))


def test_v2_new_phenomenon_is_accepted_and_recorded():
    d = MK.fixture_plan_v2("box", "new_phenomenon")
    tp = S.validate(d, "box", sup2("box"))
    new = [p for p in tp.phenomena if p["id"] == "NEW"]
    assert len(new) == 1 and new[0]["name"] == "contents shift" and "unmeasurable" in new[0]["handle"]
    # omitted PH1 and no probe is a valid reply (PH1 then sits on its fallback)
    tp = S.validate(MK.fixture_plan_v2("box", "omit_ph1"), "box", sup2("box"))
    assert tp.probes == [] and all(p["id"] != "PH1" for p in tp.phenomena)
    # 'unmeasurable' for a phenomenon that has a built skill is accepted and recorded
    d = MK.fixture_plan("box")
    d["phenomena"][0]["handle"] = dict(unmeasurable="I do not trust it")
    d["probes"] = []
    assert S.validate(d, "box", sup2("box")).phenomena[0]["handle"] == dict(unmeasurable="I do not trust it")


@pytest.mark.parametrize("kind,frag", [("wrong_skill", "does not measure 'PH2'"), ("numeric", "belief: not part of this format"), ("narrow", "belief: not part of this format")])
def test_v2_mock_variants_through_propose(tmp_path, kind, frag):
    p, rec = PP.propose("box", backend_spec=f"mock:{kind}", cache_dir=tmp_path / kind)
    assert p["status"] == "plan_rejected" and p["stage"] == "schema" and any(frag in r for r in p["reasons"]) and rec["status"] == "plan_rejected"


def test_v2_mock_variants_that_validate(tmp_path):
    for kind in ("fixture", "omit_ph1", "new_phenomenon"):
        p, rec = PP.propose("pot", backend_spec=f"mock:{kind}", cache_dir=tmp_path / kind)
        assert p["status"] == "ok" and "phenomena" in p and "belief" not in p and p["probes"] == ([] if kind == "omit_ph1" else ["weigh"])
    p, _ = PP.propose("pot", backend_spec="mock:malformed", cache_dir=tmp_path / "m")
    assert p["status"] == "plan_rejected" and p["stage"] in ("parse", "schema")


def test_v2_prompt_has_no_contents_prior_no_sanity_range_no_truth():
    from world_model.v2.stack import tasks as TK
    for task in ALL_V2:
        txt = PR.system() + PR.user_prompt(task)
        assert PR.SYSTEM == PR.system() and "belief" not in txt.lower() and "sanity" not in txt.lower() and "hidden_parameters" not in txt and "prior" not in txt.lower()
        assert "covers" in txt and "Recoveries take no" in txt                                  # the rule that forbids covers is the only mention
        f = PR.scene_facts(task)
        assert "hidden_parameters" not in f and f["phenomena"] and [c["id"] for c in f["measuring_skills"]] == ["weigh"]
        lo, hi = TK.prior_box(task)                                                             # the task's contents prior box: its numbers appear in no field of the facts
        assert f"[{lo}, {hi}]" not in txt and f"({lo}, {hi})" not in txt
        assert PR.user_prompt(task) == PR.user_prompt(task)                                     # deterministic
    w = SC.BoxWorld(contents_kg=3.1234, contents_xy=(0.0123, -0.0456))
    t = PR.SYSTEM + PR.user_prompt("box", world=w)
    for needle in ("3.1234", "0.0123", "-0.0456", "contents_mu"):
        assert needle not in t
    assert "0.5, 5.0" not in t and "(0.0, 20.0)" not in t and "20.0" not in PR.user_prompt("box")


def test_v2_behind_prompt_keeps_base_move_and_scene_facts():
    for task in BEHIND:
        txt = PR.user_prompt(task)
        f = PR.scene_facts(task)
        assert "base_move" in json.dumps(f["vocabulary"]) and "Known-good route R1" in txt and "Turning in place at the pick pose hits counter 1" in txt
        assert "R1" in json.dumps(f["task"]["base"]["routes_known_good"]) and f["phenomena"]
    assert "base_move" not in json.dumps(PR.scene_facts("tray6")["vocabulary"])


def test_v2_reask_prompt_is_built_and_rejection_prompt_uses_v2_keys(tmp_path):
    plan, _ = PP.propose("box", backend_spec="mock:fixture", cache_dir=tmp_path / "c")
    assert PR.REASK_MARK in PR.reask_prompt("box", plan, dict(trigger="T2", readings=[]))          # build step 6 (11-09); full tests in test_coverage.py
    txt = PR.rejection_prompt("box", [dict(plan=plan, reply="r", stage="planner", reasons=["x"])])
    assert '"phenomena"' in txt and '"belief"' not in txt and "REJECTION 1: stage=planner" in txt


def test_v2_compile_accepts_a_v2_plan_dict():
    plan = dict(MK.fixture_plan("tray6"), status="ok", task="tray6")
    cp = PC.compile_plan(plan, "tray6", build=False)
    assert cp.names == ["s0_lift", "s1_move", "s2_place"] and cp.belief == {}


def test_v1_replay_switch_still_gives_the_old_format():                       # runs under BIMANUAL_LEGACY_PROPOSER_V1=1 (no "_v2" in the name)
    assert S.legacy_v1() and set(MK.fixture_plan("box")) == set(S.V1_KEYS)
    assert '"belief"' in PR.SYSTEM and "sanity_range" in json.dumps(PR.scene_facts("box")) and "hidden_parameters" in PR.scene_facts("box")
    assert "belief" not in json.dumps(PR.scene_facts("box")["vocabulary"])
