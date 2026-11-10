"""regressions from the 11-09 code review."""
import json

import pytest

from world_model.v2.house.lang.backend import Backend, CachedVLM
from world_model.v2.stack import tasks as TK


class _B(Backend):
    name = "fake"

    def __init__(self, reasoning):
        self.model, self.reasoning = "m", reasoning

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        return dict(texts=[f"reply-{self.reasoning}"])


def test_cache_does_not_replay_another_reasoning_level(tmp_path):
    hi = CachedVLM(_B("high"), tmp_path)
    assert hi.ask("q")["texts"] == ["reply-high"]
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["reasoning"] == "high"
    lo = CachedVLM(_B("low"), tmp_path, mode="ro")
    with pytest.raises(KeyError, match="reasoning"):
        lo.ask("q")
    assert CachedVLM(_B("high"), tmp_path, mode="ro").ask("q")["cached"]
    f = next(tmp_path.glob("*.json")); d = json.loads(f.read_text()); d.pop("reasoning"); f.write_text(json.dumps(d))
    r = CachedVLM(_B("low"), tmp_path, mode="ro")
    assert r.ask("q")["cached"] and r.trace[-1]["reasoning_checked"] is False       # legacy record: replayed, marked unchecked


def test_behind_tasks_do_not_reach_fixed_base_goal():
    from world_model.v2.stack.planners import closed_chain as CC
    from world_model.v2.stack.proposer import prompt as PR
    for n in ("box_behind", "tray6_behind"):
        with pytest.raises(NotImplementedError):
            CC.kind_of(n)
        with pytest.raises(NotImplementedError):
            PR.task_spec(n)
    assert CC.kind_of("box") == "box" and CC.kind_of("tray6") == "tray"
