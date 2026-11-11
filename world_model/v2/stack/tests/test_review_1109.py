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
    """11-09 D1: the NotImplementedError guards are replaced by real handling: a *_behind task keeps its own kind, its goal is TaskDef.goal_dxy on counter2 (never the fixed-base GOAL_DXY),
    and its plan is a base WaypointPlan (lift -> base_move R1 -> place)."""
    from world_model.v2.stack import tasks as TK
    from world_model.v2.stack.planners import closed_chain as CC
    from world_model.v2.stack.proposer import prompt as PR
    assert CC.kind_of("box_behind") == "box" and CC.kind_of("tray6_behind") == "tray"
    for n in ("box_behind", "tray6_behind"):
        sp = PR.task_spec(n)
        assert sp["goal_support"] == "counter2" and TK.task(n).goal_dxy == TK._BEHIND_DXY
        assert abs(sp["goal_xy"][0] - (sp["start_xy"][0] + TK._BEHIND_DXY[0])) < 1e-3          # not start + GOAL_DXY[kind] = (-0.10, 0) / (-0.20, 0)
    assert CC.kind_of("box") == "box" and CC.kind_of("tray6") == "tray"
