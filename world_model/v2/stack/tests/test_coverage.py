"""checks/coverage.py (probe-selection build step 5) + the v2 Rule 2 prompt / mock re-ask (step 6, stack part).  SIMULATION ONLY, CPU, no network.
PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_coverage.py"""
import itertools
import json
import types

import pytest

from world_model.v2.stack import probing as PB
from world_model.v2.stack.checks import admit as AD
from world_model.v2.stack.checks import coverage as COV
from world_model.v2.stack.proposer import mock as MK
from world_model.v2.stack.proposer import propose as PP
from world_model.v2.stack.proposer import prompt as PR
from world_model.v2.stack.proposer import schema as S
from world_model.v2.stack.skills import catalog as CAT

T0 = {"contents_mass_kg": (0.5, 5.0)}


class Fake:
    id = "weigh"

    def __init__(self, reading):
        self._r = reading

    def limit_rows(self, th):
        return []

    def run(self, th):
        return self._r


def run_weigh(lo=1.0, hi=2.0):
    return PB.run_probes(["weigh"], {"weigh": Fake(PB.Reading("weigh", {"contents_mass_kg": (lo, hi)}))}, T0)


@pytest.fixture(autouse=True)
def _v2(monkeypatch):
    monkeypatch.delenv("BIMANUAL_LEGACY_PROPOSER_V1", raising=False)
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")


def test_measured_fallback_and_out_of_scope_note():
    cov = CAT.coverage("box", run_weigh())
    st = {r.phi: (r.status, r.by) for r in cov.rows}
    assert st["PH1"] == ("measured", "weigh") and st["PH2"][0] == "fallback" and st["PH3"][0] == "fallback" and cov.covered and not cov.refuses
    assert cov.theta_t["contents_mass_kg"] == (1.0, 2.0) and COV.contents_mass(cov) == (1.0, 2.0)
    assert "PH4" in cov.out_of_scope and "PH5" in cov.out_of_scope and "PH6" in cov.out_of_scope
    assert set(cov.to_dict()) >= {"rows", "covered", "uncovered", "theta_t", "out_of_scope", "named_by_llm"}
    assert not (set(CAT.OUT_OF_SCOPE) & set(CAT.phi("box")) | set(CAT.OUT_OF_SCOPE) & set(CAT.phi("tray6_behind")))      # out of scope = out of PHI
    skipped = CAT.coverage("box", PB.run_probes(["weigh"], {"weigh": Fake(None)}, T0, static_same=lambda th: True))
    assert {r.phi: r.status for r in skipped.rows}["PH1"] == "fallback" and skipped.covered and skipped.theta_t["contents_mass_kg"] == (0.5, 5.0)


def test_uncovered_phi_refuses_in_enforce_mode_only():
    phi = ("PH1", "PH2", "PH4")                                          # PH4: out of scope = status out_of_scope = no approved fallback
    cov = CAT.coverage("tray6", None, theta0=T0, phi_override=phi)
    assert cov.uncovered == ["PH4"] and cov.refuses and not cov.covered
    assert [r.status for r in cov.rows] == ["fallback", "fallback", "uncovered"]
    assert not CAT.coverage("tray6", None, theta0=T0, phi_override=phi, mode="report").refuses
    r = COV.row(cov)
    assert not r.ok and r.guarantee == "missing" and "PH4" in r.why
    assert COV.row(CAT.coverage("box", run_weigh())).ok
    with pytest.raises(ValueError):
        CAT.coverage("box", None, mode="nope")
    # a phenomenon with no entry at all is uncovered as well; an explicit approved fallback covers it
    cov = CAT.coverage("box", None, theta0=T0, phi_override=("PH9",), fallbacks={})
    assert cov.uncovered == ["PH9"]
    cov = CAT.coverage("box", None, theta0=T0, phi_override=("PH9",), fallbacks={"PH9": CAT.Fallback("interior", "x", "MODEL", "t", "approved")})
    assert cov.covered


def _verdict(cov):
    d = cov.to_dict()
    d.pop("named_by_llm")
    d.pop("new_names")
    return json.dumps(d, sort_keys=True), cov.refuses


@pytest.mark.parametrize("task", ["box", "tray6", "pot", "box_behind", "tray6_behind"])
def test_property_llm_phenomena_list_never_changes_the_verdict(task):
    """drop or shuffle the LLM's phenomena, same readings -> same coverage verdict (the LLM's list only changes which probes run)."""
    ids = [p for p in CAT.PHENOMENA if CAT.kind_of(task) in CAT.PHENOMENA[p].objects] + ["NEW"]
    ref = _verdict(CAT.coverage(task, run_weigh(), named=ids))
    for k in range(len(ids) + 1):
        for sub in itertools.islice(itertools.permutations(ids, k), 40):
            assert _verdict(CAT.coverage(task, run_weigh(), named=list(sub), new_names=["x"] * k)) == ref
    for refr in (ref, _verdict(CAT.coverage(task, None, theta0=T0, named=[]))):   # a different reading set may differ, only through theta / measured
        assert isinstance(refr[1], bool)
    # the same holds for the admission: admit() with the coverage refuses or not purely on the coverage
    spec = types.SimpleNamespace(contents=types.SimpleNamespace(m_lo=0.5, m_hi=5.0), open_limits=[])
    cov_bad = CAT.coverage(task, None, theta0=T0, phi_override=("PH1", "PH4"))
    for named in ([], ["PH1"], ["PH4", "PH1"]):
        cv = CAT.coverage(task, None, theta0=T0, phi_override=("PH1", "PH4"), named=named)
        r = AD.admit(types.SimpleNamespace(rows=()), spec, None, None, None, coverage=cv)      # refuses before it touches path / backend
        assert r.action == "refuse (uncovered)" and not r.verdict.admit and r.verdict.guarantee == "missing" and r.planner == "coverage"
    assert cov_bad.refuses


def test_admit_without_coverage_is_unchanged_signature():
    import inspect
    assert inspect.signature(AD.admit).parameters["coverage"].default is None


# ------------------------------------------------------------------------------------------------------------------ Rule 2 prompt (schema v2)
def test_reask_prompt_v2_has_evidence_and_no_truth():
    plan, _ = PP.propose("box", backend_spec="mock:fixture")
    ev = dict(trigger="T2", readings=[dict(skill="weigh", output="lower_bound", fields={"contents_mass_kg": [1.6, None]}, why="rung_not_admissible")],
              probes_skipped=[], failing_rows=[dict(row="torque_payload[s0_lift]", u_max=1.07, a=0.5, worst=dict(m=5.0, c=[0.02, 0.0, 0.1]))],
              worst=dict(m=5.0, c=[0.02, 0.0, 0.1], wp="p3"), action="refuse")
    txt = PR.reask_prompt("box", plan, ev)
    assert PR.REASK_MARK in txt and "(T2)" in txt and "torque_payload[s0_lift]" in txt and "1.6" in txt and "WORST VERTEX" in txt
    assert "YOUR PREVIOUS PLAN" in txt and "phenomena" in txt and "belief" not in txt.split("YOUR PREVIOUS PLAN")[1].split("RULE 2")[0]
    for w in ("seed", "theta_world", "M_true", "truth"):
        assert w not in txt
    t1 = PR.reask_prompt("box", plan, dict(trigger="T1", readings=[], phenomena_shown=["PH2"]))
    assert "(T1)" in t1 and "PH2" in t1
    assert PR.reask_prompt("box", plan, ev) == txt                                       # deterministic (cache key)


def test_reask_v2_with_mock_and_rejection_loop(tmp_path):
    plan, rec = PP.propose("box", backend_spec="mock:t2_reask", cache_dir=tmp_path / "c")
    assert plan["status"] == "ok" and plan["probes"] == [] and all(p["id"] != "PH1" for p in plan["phenomena"])      # first reply omits PH1: no probe
    ev = dict(trigger="T2", readings=[], failing_rows=[dict(row="torque_payload[s0_lift]", u_max=1.2)])
    new, rq = PP.reask("box", plan, ev, backend_spec="mock:t2_reask", cache_dir=tmp_path / "c")
    assert new["status"] == "ok" and new["probes"] == ["weigh"] and rq["kind"] == "reask" and rq["trigger"] == "T2" and rq["previous_plan_id"] == plan["plan_id"]
    assert "RULE 2 RE-ASK" in json.load(open(rq["raw_reply_path"]))["request"]["text"]
    calls = []

    def check(pl):
        calls.append(pl["plan_id"])
        if len(calls) == 1:
            raise S.PlanRejected("planner", ["no lift offset reachable"])
        return ("ok", pl["plan_id"])
    n2, acc, rec2, info = PP.reask_rule2_until_accepted("box", plan, ev, check, backend_spec="mock:t2_reask", cache_dir=tmp_path / "c2")
    assert acc[0] == "ok" and info["n_asks"] == 2 and len(info["attempts"]) == 1 and info["attempts"][0]["stage"] == "planner" and rec2["kind"] == "reask_rejected"
    txt = json.load(open(rec2["raw_reply_path"]))["request"]["text"]
    assert "RULE 2 RE-ASK" in txt and "REJECTED PLAN 1" in txt                          # the evidence stays in the request after a rejection
    with pytest.raises(PP.ProposerCacheMiss):
        PP.reask_rule2_until_accepted("box", plan, ev, check, backend_spec="codex:gpt-5.6-sol:high", cache_dir=tmp_path / "nocache", offline=True)


def test_probe_in_belief_retired_from_the_v2_path():
    from world_model.v2.stack.proposer import compile as PC
    with pytest.raises(RuntimeError, match="retired"):
        PC.probe_in_belief("box", {}, 1.0, 2.0)


def test_mock_t2_reask_kind_registered():
    assert "t2_reask" in MK.KINDS and MK.MockProposer("t2_reask").kind == "t2_reask"


@pytest.mark.parametrize("task", ["box", "tray6", "pot", "box_behind"])
def test_proposed_v2_plan_round_trips_into_compile(task, tmp_path):
    """regression (11-09): a plan dict from propose() lists probes as ids; compile_plan re-validates it as a reply, so reply_of must restore [{"skill": id}]."""
    from world_model.v2.stack.proposer import compile as PC
    plan, _ = PP.propose(task, backend_spec="mock:fixture", cache_dir=tmp_path / "c")
    assert plan["status"] == "ok" and plan["probes"] == ["weigh"] and S.reply_of(plan)["probes"] == [{"skill": "weigh"}]
    PC.compile_plan(plan, task, build=False, level="L1" if task.endswith("_behind") else None)


def test_identical_rejected_reply_loop_stops(monkeypatch):
    from world_model.v2.house.lang.backend import Backend
    class Stuck(Backend):
        name, model = "stuck", "stuck-model"
        def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
            return dict(texts=[json.dumps(MK.fixture_plan("box"))] * n, seconds=0.0, wall_s=0.0, cached=False, backend=self.name)
    monkeypatch.setattr(PP, "make_backend", lambda spec, cache_dir, offline=False: Stuck())
    def check(pl):
        raise S.PlanRejected("planner", ["never"])
    plan, acc, rec, info = PP.propose_until_accepted("box", check, backend_spec="x", cache_dir=None)
    assert acc is None and plan["stage"] == "proposer" and "identical" in info["proposer_error"] and info["n_reasks"] == PP.MAX_IDENTICAL_REJECTED
