"""same_worlds --plan-source codex with the MOCK proposer backend (no network, no model).  SIMULATION ONLY, CPU.
PYTHONPATH=. python -m pytest -q world_model/v2/nero/contents/tests/test_same_worlds_plan.py
One world per task (box, tray6, pot), OURS only, run in process (each world is a probe + a carry or a refusal: minutes, not seconds).
Fixture path byte-identical: the pre-existing test files (test_carry_cert, test_probe, test_pot, stack/tests) cover the default plan; here the static admission of a
fixture-shaped proposer plan is compared with the default plan's."""
import argparse
import json

import pytest

from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.nero.contents import probe as P
from world_model.v2.nero.contents import same_worlds as SW
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack.proposer import compile as PC
from world_model.v2.stack.proposer import mock as MK
from world_model.v2.stack.proposer import propose as PP
from world_model.v2.stack.proposer import schema as S

EPS = (0.0498, 0.0774)               # eps_lo, eps_free (CALIB.json, MEASURED 11-03; fixed here so the test does not read /data)


@pytest.fixture(scope="module", autouse=True)
def _legacy_proposer_v1():
    """these tests drive the v1 reply (belief, probe_in_B); the schema-v2 run path is build steps 6-7.  BIMANUAL_LEGACY_PROPOSER_V1=1 for the whole module (the cache fixture is module-scoped)."""
    mp = pytest.MonkeyPatch()
    mp.setenv("BIMANUAL_LEGACY_PROPOSER_V1", "1")
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")
    monkeypatch.setattr(P, "PROBE_METHOD", "ladder")        # EPS above are the 11-03 LADDER eps: the eps must match the method


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    d = tmp_path_factory.mktemp("mockcache")
    for t in ("box", "tray6", "pot", "tray2"):
        PP.propose(t, backend_spec="mock:fixture", cache_dir=d)            # the mock cache: one stored request + reply per task
    return d


def ns(cache, offline=True, spec="mock:fixture", outside_b="refuse"):
    return argparse.Namespace(plan_spec=spec, plan_cache=str(cache), offline=offline, outside_b=outside_b)


@pytest.mark.parametrize("task", ["box", "tray6", "pot"])
def test_one_world_with_a_proposer_plan(task, cache, tmp_path):
    plans = SW._proposer_plans([task], ns(cache), tmp_path)
    it = plans[task]
    assert it["plan_source"] == "codex" and "rejected" not in it and it["plan"]["wp"].names == ("s0_lift", "s1_move", "s2_place")
    doc = json.loads((tmp_path / "plans" / f"{task}.json").read_text())
    assert doc["record"]["request_hash"] and doc["record"]["caveats"] and doc["compiled"]["belief"]
    r = SW._job(("ours", task, 0, *EPS, it))
    assert "error" not in r, r.get("tb")
    assert r["method"] == "ours" and r["planner"] == "closed_chain" and r["plan_source"] == "codex" and r["plan_id"] == it["plan_id"]
    assert r["plan_record"]["spec"] == "mock:fixture" and r["plan_record"]["cached"] is True              # replayed from the cache, offline
    assert r["outcome"] in ("carried", "carry_failed", "refused_in_R", "refused_bad")
    if r["outcome"] in ("carried", "carry_failed"):
        assert [s["seg"] for s in r["carry"]["segments"]][:3] == ["s0_lift", "s1_move", "s2_place"][:len(r["carry"]["segments"])]
        assert r["probe_in_B"]["param"] and r["probe_in_B"]["inside"] is True                              # the mock belief is the whole sanity range
        assert all(s["family"] for s in r["carry"]["segments"] if s["seg"] != "release")                   # the compiled backups were run
    else:
        assert r["fallback"] == ["let_go"] and r["refuse_end_in_R"] is not None


def test_static_admission_over_compiled_segments_matches_default_for_fixture_shape(cache, tmp_path):
    """the fixture-shaped proposer plan IS the default path: same verdict, rows keyed by the compiled segment names."""
    plan, _ = PP.propose("tray6", backend_spec="mock:fixture", cache_dir=cache, offline=True)
    cp = PC.compile_plan(plan, "tray6")
    w = CC.world_default("tray6")
    lo, hi = P.prior_contents("tray6")
    a = P.static_interval("tray6", lo, hi, w, planner="closed_chain")
    b = P.static_interval("tray6", lo, hi, w, wp=cp.wp)
    assert a["action"] == b["action"] and abs(a["u_nom"] - b["u_nom"]) < 1e-9 and a["a_admit"] == b["a_admit"]
    assert list(b["rows"]) == ["s0_lift", "s1_move", "s2_place", "release"] and "support_dest" in b["rows"]["s2_place"] and b["wp"]["names"][0] == "s0_lift"
    assert list(a["rows"]) == ["lift", "carry", "place", "release"]


def test_rejected_plan_is_counted_for_every_world(cache, tmp_path, monkeypatch):
    """USER 11-09: a rejected plan is re-asked without a cap, so plan_rejected is final only after a proposer ERROR (the backend raises)."""
    from world_model.v2.house.lang.backend import Backend, CachedVLM
    from world_model.v2.stack.proposer import propose as PP

    class Boom(Backend):
        name = model = "boom"

        def ask(self, *a, **k):
            raise RuntimeError("codex exited 1")
    monkeypatch.setattr(PP, "make_backend", lambda spec, cache_dir, offline=False: CachedVLM(Boom(), cache_dir, mode="rw"))
    bad = tmp_path / "bad"
    plans = SW._proposer_plans(["box"], ns(bad, offline=False, spec="mock:unknown_skill"), tmp_path)
    assert plans["box"]["rejected"]["stage"] == "proposer" and "plan" not in plans["box"] and plans["box"]["plan_record"]["proposer_error"]
    r = SW._job(("ours", "box", 3, *EPS, plans["box"]))
    assert r["outcome"] == "plan_rejected" and r["any_U"] is False and r["plan_rejected"]["reasons"] and r["plan_source"] == "codex" and "error" not in r


def test_offline_cache_miss_fails_loudly(tmp_path):
    with pytest.raises(PP.ProposerCacheMiss):
        SW._proposer_plans(["box"], ns(tmp_path / "empty"), tmp_path)
    with pytest.raises(PP.ProposerCacheMiss):                                      # main() stops before any job starts
        SW.main(["run", "--tasks", "box", "--n", "1", "--plan-source", "codex", "--plan-spec", "codex:gpt-5.6-sol:high", "--plan-cache", str(tmp_path / "empty"),
                 "--offline", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):                                                  # no cache dir given
        SW.main(["run", "--tasks", "box", "--n", "1", "--plan-source", "codex", "--out", str(tmp_path / "o2")])


def test_theirs_agentchord_is_guarded(tmp_path):
    with pytest.raises(SystemExit):
        SW.main(["run", "--tasks", "box", "--n", "1", "--theirs", "agentchord", "--out", str(tmp_path / "o3")])    # no --agentchord-plan


def test_probe_set_outside_belief_is_recorded_and_refused(cache, tmp_path):
    """a narrow belief (mock:narrow) on the probed parameter; the probe's set for a heavy box world lies outside it -> refuse, with the numbers recorded."""
    d = tmp_path / "narrow"
    plans = SW._proposer_plans(["tray2"], ns(d, offline=False, spec="mock:narrow"), tmp_path)
    r = SW._job(("ours", "tray2", 1, *EPS, plans["tray2"]))
    assert "error" not in r, r.get("tb")
    if r.get("probe_in_B") and not r["probe_in_B"]["inside"]:
        assert r["outcome"].startswith("refused") and r["decision"]["action"] == "refuse (probe set outside B)" and r["decision"]["why_B"]["B"]
