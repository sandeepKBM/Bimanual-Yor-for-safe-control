"""Offline tests for agentchord_probe.nero: canned plans only (no LLM, no network beyond the loopback gateway).  Run from the worktree root with
/data/local/ss5772/agentchord/venv_nero/bin/python -m pytest world_model/v2/agentchord_probe/nero/tests -x -q   (MUJOCO_GL=osmesa, CPU)."""
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("mujoco")

from world_model.v2.agentchord_probe.nero import canned, plan as P, tasks as T      # noqa: E402

RGB = np.full((480, 640, 3), 90, np.uint8)       # stand-in image (the real render is exercised in the smoke run)


def _plan(tmp_path, kind, task="tray4", **kw):
    cf = tmp_path / f"{kind}.json"
    cf.write_text(json.dumps(canned.replies(kind, task, **kw)))
    return P.make_plan(task, tmp_path / f"plan_{kind}", backend="canned", canned=str(cf), rgb=RGB)


def _run(meta, task, i=0, cell_idx=0, **kw):
    from world_model.v2.agentchord_probe.nero.execute import run_world
    cell = T.cell_list(task)[cell_idx]
    return run_world(dict(plan=str(pathlib.Path(meta["graph"]).parent / "plan.json"), task=task, cell=list(cell), i=i, **kw))


def test_facts_text_reaches_the_planner(tmp_path):
    meta = _plan(tmp_path, "two_arm")
    from world_model.v2.vlm import contents_eval as CE
    facts = CE.facts_and_truth("tray4", T.rep_world("tray4"))[0]
    assert facts in meta["task_prompt"] and meta["facts_text"] == facts
    rows = [json.loads(l) for l in open(tmp_path / "plan_two_arm" / "llm_log.jsonl")]
    assert len(rows) == 2                                              # task agent + recovery agent (no canonicalize call)
    first = json.dumps(rows[0]["messages"])
    assert facts.split(". ")[0] in first and "image_url" in first       # the request the gateway got: facts text + an image part
    assert "Task goal" in first and "Available atomic actions" in first    # their template, unchanged
    assert "hidden" not in meta["task_prompt"].lower() or "hidden items" in facts   # no truth/mass hint beyond the facts text
    assert meta["models"] and meta["setup_deviations"] and meta["backend"] == "canned"


def test_box_prompt_has_shelf_and_facts(tmp_path):
    text, facts = T.task_prompt("box")
    assert "shelf" in text and facts in text and "0.97" in text


def test_two_arm_plan_executes(tmp_path):
    meta = _plan(tmp_path, "two_arm", "tray2")
    r = _run(meta, "tray2")
    assert r["outcome"] == "completed", (r.get("reason"), r.get("tb"))
    assert r["n_edges_executed"] == 4 and r["n_rejected"] == 0 and not r["monitors_fired"]
    st = [e["status"] for e in r["edges"]]
    assert st == ["executed"] * 4
    assert r["edges"][0]["info"]["note"].startswith("no-op")           # grasp under the scripted weld
    assert r["goal"]["xy_err"] < 0.05 and "tau_ratio" in r["track"]


def test_one_arm_edge_rejected(tmp_path):
    meta = _plan(tmp_path, "one_arm", "tray2")
    r = _run(meta, "tray2")
    assert r["outcome"] == "plan_rejected" and r["reason"] == "rejected_one_arm"
    assert r["n_edges_executed"] == 2 and r["n_rejected"] == 1         # grasp + lift ran, the carry edge did not, nothing after it
    assert len(r["edges"]) == 3 and not r["ac_success"]


def test_rotate_eef_rejected(tmp_path):
    meta = _plan(tmp_path, "rotate", "tray2")
    r = _run(meta, "tray2")
    assert r["outcome"] == "plan_rejected" and r["reason"] == "unmappable_atom"
    assert r["rejections"][0]["detail"]["fn"] == "rotate_eef"


def test_monitor_fires_and_recovery_runs(tmp_path, monkeypatch):
    from world_model.v2.agentchord_probe.nero import atoms
    monkeypatch.setattr(atoms, "MAX_EDGES", 5)
    meta = _plan(tmp_path, "loop", "tray2", thr=0.0005)
    r = _run(meta, "tray2")
    assert r["monitors_fired"] and r["n_recoveries_run"] >= 1
    assert r["outcome"] in ("budget", "completed", "graph_error")
    assert any(e["status"] == "aborted_by_monitor" for e in r["edges"])


GPT_BOX_PLAN = pathlib.Path("/common/users/ss5772/bimanual_data/agentchord_node/plan_box/plan.json")   # gpt-6.1-sol box plan (11-03)


@pytest.mark.skipif(not GPT_BOX_PLAN.exists(), reason="gpt-6.1-sol box plan not staged")
def test_arms_move_independently_after_release():
    """11-03 fix: after open_gripper the two arms may move by different deltas (the box plan's withdraw dy +0.1 / -0.1)."""
    from world_model.v2.agentchord_probe.nero.execute import run_world
    cell = T.cell_list("box")[0]
    r = run_world(dict(plan=str(GPT_BOX_PLAN), task="box", cell=list(cell), i=0, seeds="baseline"))
    st = {e["edge_id"]: e["status"] for e in r["edges"]}
    assert st.get("e06_withdraw_grippers") == "executed", (r.get("reason"), r.get("rejections"))
    assert r["n_rejected"] == 0
    e6 = next(e for e in r["edges"] if e["edge_id"] == "e06_withdraw_grippers")
    assert e6["info"].get("released") is True
