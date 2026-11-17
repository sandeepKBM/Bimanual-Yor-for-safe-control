"""arm-reach line in planner refusals (USER 11-16 option b: "go with b").  Owner-written gate.  SIMULATION ONLY, CPU, offline (no proposer call).

Checked:
  1. reach_summary on box_bookcase at z = 0.02 m equals the owner's independent scan (11-16, same SRS solver, 1 cm steps): forward 0.06, back 0.25, left 0.05, right 0.03 m;
  2. a real refused Codex reply (run1 reply 3: lift 2 cm, push 15 cm) is still refused at the planner, with the SAME first reason as before, plus one "arm reach" line;
  3. BIMANUAL_LEGACY_NO_REACH_TEXT=1 gives the old reasons exactly (no reach line);
  4. a plan the planner admits (the mock default) is unchanged (no reach work on success).
"""
import json
import pathlib

import pytest

from world_model.v2.stack.planners import closed_chain as CCP
from world_model.v2.stack.proposer import compile as PC, schema as S

TASK = "box_bookcase"
REPLY3 = json.loads((pathlib.Path(__file__).parent / "data" / "codex_bookcase_reply3.json").read_text())["reply_text"]
OLD_FIRST = "left: no SRS solution inside the joint limits at path sample 11/25 (object offset [0.0469, 0.0344, 0.02])"


def _plan(text):
    tp = S.validate_reply(text, TASK, ("counter", "bookcase_l1"))
    d = tp.to_dict()
    d.update(task=TASK, plan_id=S.plan_id(d))
    return d


def _reasons(monkeypatch, legacy):
    monkeypatch.setenv("BIMANUAL_PLANNER", "closed_chain")
    if legacy:
        monkeypatch.setenv("BIMANUAL_LEGACY_NO_REACH_TEXT", "1")
    else:
        monkeypatch.delenv("BIMANUAL_LEGACY_NO_REACH_TEXT", raising=False)
    with pytest.raises(S.PlanRejected) as ei:
        PC.compile_plan(_plan(REPLY3), TASK, level="L1")
    assert ei.value.stage == "planner"
    return list(ei.value.reasons)


def test_reach_summary_matches_owner_scan():
    r = CCP.reach_summary(CCP._nominal_inputs(TASK), 0.02)
    assert r == dict(forward=0.06, back=0.25, left=0.05, right=0.03), r


def test_refusal_gets_one_reach_line(monkeypatch):
    rs = _reasons(monkeypatch, legacy=False)
    assert rs[0].startswith(OLD_FIRST), rs[0]
    reach = [x for x in rs if x.startswith("arm reach")]
    assert len(reach) == 1 and "forward 0.06 m" in reach[0] and "robot's right 0.03 m" in reach[0], reach


def test_legacy_flag_restores_old_reasons(monkeypatch):
    rs = _reasons(monkeypatch, legacy=True)
    assert rs[0].startswith(OLD_FIRST) and not any(x.startswith("arm reach") for x in rs) and len(rs) == 3, rs


def test_admitted_plan_unchanged(monkeypatch):
    from world_model.v2.stack.proposer import mock as MK
    monkeypatch.setenv("BIMANUAL_PLANNER", "closed_chain")
    cp = PC.compile_plan(_plan(json.dumps(MK.fixture_plan_v2(TASK))), TASK, level="L1")
    assert cp.planner and cp.planner["margin_min"]
