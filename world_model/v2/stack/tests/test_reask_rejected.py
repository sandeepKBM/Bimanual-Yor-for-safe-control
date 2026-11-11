"""Re-ask on a REJECTED plan (USER 11-09, no cap): fake proposer, no network, no model.   PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_reask_rejected.py"""
import json

import pytest

from world_model.v2.house.lang.backend import Backend, CachedVLM
from world_model.v2.stack.proposer import mock as MK
from world_model.v2.stack.proposer import propose as PP
from world_model.v2.stack.proposer import prompt as PR
from world_model.v2.stack.proposer import schema as S

TRUTH_WORDS = ("seed", "theta", "truth", "true_", "ground")


class Fake(Backend):
    """reply 0: unparsable; reply 1: a valid plan the checker refuses (planner); reply 2: a valid plan the checker accepts (a different lift => different plan id)."""
    name, model = "fake", "fake-model"

    def __init__(self):
        self.prompts = []

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        self.prompts.append(text)
        k = len(self.prompts) - 1
        if k == 0:
            reply = "here is a plan: {\"belief\": [lift"
        else:
            d = MK.fixture_plan("box")
            if k >= 2:
                d["steps"][0]["args"]["dz"] = 0.17                    # a different step => a different plan id (schema v2 has no belief)
            reply = json.dumps(d)
        return dict(texts=[reply] * n, seconds=0.0, wall_s=0.0, cached=False, backend=self.name)


def _check(pl):
    if len(CHECKED) == 0:
        CHECKED.append(pl["plan_id"])
        raise S.PlanRejected("planner", ["no lateral lift offset on the grid has an in-limit configuration", "reason=unreachable", "arm=right"])
    return ("compiled", pl["plan_id"])


CHECKED = []


@pytest.fixture
def be(monkeypatch, tmp_path):
    CHECKED.clear()
    fake = Fake()
    monkeypatch.setattr(PP, "make_backend", lambda spec, cache_dir, offline=False: CachedVLM(fake, cache_dir, mode="ro" if offline else "rw"))
    return fake


def test_fails_twice_then_passes(be, tmp_path):
    plan, acc, rec, info = PP.propose_until_accepted("box", _check, backend_spec="fake:x", cache_dir=str(tmp_path / "c"))
    assert info["final"] == "accepted" and info["n_reasks"] == 2 and acc == ("compiled", plan["plan_id"])
    assert [a["attempt"] for a in info["attempts"]] == [1, 2] and info["attempts"][0]["stage"] in ("parse", "schema")
    assert info["attempts"][1]["stage"] == "planner" and info["attempts"][1]["reasons"][1] == "reason=unreachable"
    assert info["accepted_plan_id"] == plan["plan_id"] and info["proposer_wall_s"] >= 0 and info["proposer_error"] is None
    assert len(be.prompts) == 3 and rec["kind"] == "reask_rejected" and rec["n_prior_rejections"] == 2


def test_feedback_has_reasons_and_no_truth(be, tmp_path):
    PP.propose_until_accepted("box", _check, backend_spec="fake:x", cache_dir=str(tmp_path / "c"))
    p1, p2 = be.prompts[1], be.prompts[2]
    assert "REJECTION 1: stage=" in p1 and "here is a plan" in p1                      # the raw unparsable reply is shown back
    assert "REJECTION 2: stage=planner" in p2 and "arm=right" in p2 and "REJECTED PLAN 2" in p2 and "REJECTED PLAN 1" in p2
    base = PR.user_prompt("box").rsplit("\n\nReturn the JSON object now.", 1)[0]
    assert p1.startswith(base) and "Return a COMPLETE" not in p1
    low = (p1 + p2)[len(base) * 2:].lower()
    assert not any(w in low for w in TRUTH_WORDS)
    assert PR.SYSTEM not in p1                                                         # system prompt is sent separately, unchanged


def test_offline_replay_is_deterministic(be, tmp_path):
    c = str(tmp_path / "c")
    r1 = PP.propose_until_accepted("box", _check, backend_spec="fake:x", cache_dir=c)
    n = len(be.prompts)
    CHECKED.clear()
    r2 = PP.propose_until_accepted("box", _check, backend_spec="fake:x", cache_dir=c, offline=True)
    assert len(be.prompts) == n                                                        # nothing asked again: every request incl. the feedback ones came from the cache
    assert r2[0]["plan_id"] == r1[0]["plan_id"] and r2[3]["n_reasks"] == 2
    CHECKED.clear(); CHECKED.append("x")                                               # a different rejection history => different request => offline miss, loudly
    with pytest.raises(PP.ProposerCacheMiss):
        PP.propose_until_accepted("box", lambda pl: (_ for _ in ()).throw(S.PlanRejected("planner", ["other"])), backend_spec="fake:x", cache_dir=c, offline=True)


def test_proposer_error_ends_loop(monkeypatch, tmp_path):
    class Boom(Backend):
        name, model = "boom", "boom"

        def ask(self, *a, **k):
            raise RuntimeError("codex exited 1")
    monkeypatch.setattr(PP, "make_backend", lambda spec, cache_dir, offline=False: CachedVLM(Boom(), cache_dir, mode="rw"))
    plan, acc, rec, info = PP.propose_until_accepted("box", _check, backend_spec="fake:x", cache_dir=str(tmp_path / "c"))
    assert acc is None and plan["stage"] == "proposer" and info["final"] == "proposer_error" and "codex exited 1" in info["proposer_error"]
