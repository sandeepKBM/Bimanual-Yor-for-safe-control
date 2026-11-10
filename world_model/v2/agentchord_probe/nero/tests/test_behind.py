"""Offline tests for agentchord_probe.nero.behind (table-behind tasks): canned plans only (no model call, no network beyond the loopback gateway).  Run from the worktree root:
OMP_NUM_THREADS=1 PYTHONPATH=. MUJOCO_GL=osmesa /data/local/ss5772/agentchord/venv_nero/bin/python -m pytest world_model/v2/agentchord_probe/nero/tests/test_behind.py -x -q"""
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("mujoco")

from world_model.v2.agentchord_probe.nero import atoms, behind as BH, canned as CN, plan as P     # noqa: E402
from world_model.v2.stack import tasks as TK                                                       # noqa: E402

RGB = np.full((480, 640, 3), 90, np.uint8)
L, R = CN.L, CN.R
R1 = [list(s) for s in TK.task("tray6_behind").routes[0][1]]


def _edges(task, route_r=None, lift=0.05):
    td = TK.task(task)
    w = BH.rep_world(task)
    gx, gy = float(w.obj_xy[0] + td.goal_dxy[0]), float(w.obj_xy[1] + td.goal_dxy[1])
    obj = td.kind
    both = lambda fn, **kw: (CN.act(fn, L, **kw), CN.act(fn, R, **kw))
    base = (CN.act("move_base", L, route=R1), CN.act("move_base", R, route=route_r or R1))
    return [("grasp", *both("grasp", obj_name=obj, pre_grasp_dis=0.1)),
            ("lift", *both("move_by_relative_offset", dx=0.0, dy=0.0, dz=lift, mode="extrinsic")),
            ("base", *base),
            ("place", *both("place_on_table", obj_name=obj, x=gx, y=gy, pre_place_dis=0.05))]


def _plan(tmp_path, task, name, recovery=None, **kw):
    g = CN.graph(f"carry_{TK.task(task).kind}", _edges(task, **kw))
    rec = recovery(g) if recovery else CN.empty_recovery(g)
    cf = tmp_path / f"{name}.json"
    cf.write_text(json.dumps([json.dumps(g), json.dumps(rec)]))
    return BH.make_plan(task, tmp_path / f"plan_{name}", backend="canned", canned=str(cf), rgb=RGB)


def _run(meta, task, i=0, level="L2", **kw):
    return BH.run_world(dict(plan=str(pathlib.Path(meta["graph"]).parent / "plan.json"), task=task, i=i, level=level, **kw))


def test_prompt_and_deviations(tmp_path):
    BH.install()
    for row in BH.SETUP_DEVIATIONS_BEHIND:
        assert row in P.SETUP_DEVIATIONS
    text = BH.atom_actions_text()
    theirs = (pathlib.Path(__import__("world_model.v2.agentchord_probe.nero.shim", fromlist=["x"]).THEIR_SRC) / "embodichain/agents/prompts/atom_actions.txt").read_text().strip()
    assert text.startswith(theirs) and '"move_base":' in text and "def move_base(robot_name: str, route: list" in text
    meta = _plan(tmp_path, "tray6_behind", "tray")
    assert (tmp_path / "plan_tray" / "atom_actions.txt").read_text() == text
    tp = meta["task_prompt"]
    assert BH.TURN_FACT in tp and json.dumps(R1) in tp and "counter2" in tp and "Known facts about the load:" in tp
    assert meta["behind"]["monitor_frame"] == "base" and any("move_base" in s for s in meta["setup_deviations"])
    # the model-facing request carries the new atom + the facts (what the gateway got)
    rows = [json.loads(l) for l in open(tmp_path / "plan_tray" / "llm_log.jsonl")]
    assert "move_base" in json.dumps(rows[0]["messages"]) and BH.TURN_FACT in json.dumps(rows[0]["messages"])


@pytest.mark.parametrize("task", ["tray6_behind", "box_behind"])
def test_canned_move_base_plan_runs_light_world(tmp_path, task):
    meta = _plan(tmp_path, task, "ok", lift=(0.05 if task.startswith("tray") else 0.03))
    r = _run(meta, task, level="L2")
    assert r["outcome"] == "completed", (r.get("reason"), r.get("tb"))
    assert [e["status"] for e in r["edges"]] == ["executed"] * 4 and r["n_rejected"] == 0
    b = r["base"]
    assert b["n_base_edges"] == 1 and b["n_aborted"] == 0 and set(b["U_flags"]) == set(BH.BASE_FLAGS)
    assert abs(b["base_end_pose"][0] - (r["base_log"][0]["nominal_end"][0])) < 0.05           # drove R1 (end near the nominal end)
    assert "success" in r["result"] and "xy_err" in r["measured"]
    assert r["edges"][3]["info"]["support"] == "counter2"


def test_arm_mismatch_rejected(tmp_path):
    meta = _plan(tmp_path, "tray6_behind", "mm", route_r=[["line", -0.4], ["turn", 180.0], ["line", 0.4]])
    r = _run(meta, "tray6_behind")
    assert r["outcome"] == "plan_rejected" and r["reason"] == "rejected_arm_mismatch"
    assert r["n_edges_executed"] == 2 and r["base"]["n_base_edges"] == 0            # grasp + lift ran, the base edge did not


def test_monitor_abort_brakes_then_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(BH, "MONITOR_FRAME", "world")           # world-frame object_moved fires on the base motion itself (L2: 0.05 m per 0.1 s sample > 0.02 m default)
    monkeypatch.setattr(atoms, "MAX_EDGES", 5)

    def rec(g):
        return dict(task=g["task"], recovery_bindings=[dict(edge_id=g["edges"][2]["id"], failure_name="moved", monitors=[
            dict(type="object_moved", objects=["tray"], threshold=0.02)], recovery=[dict(type="retry_failed_edge")], merge="target", repeat_until_success=False)])
    meta = _plan(tmp_path, "tray6_behind", "mon", recovery=rec)
    r = _run(meta, "tray6_behind", level="L2")
    fired = [e for e in r["edges"] if e.get("monitor")]
    assert fired and fired[0]["edge_id"].endswith("base") and fired[0]["status"] == "aborted_by_monitor"
    b0 = r["base_log"][0]
    assert b0["aborted"] and b0["abort_t_in_edge_s"] > 0 and max(abs(v) for v in b0["v_at_abort"]) > 0.05       # it was moving when the monitor hit
    assert b0["stop_dist_m"] > 0.0                                                                             # and braked over a distance (not an instant stop)
    assert r["n_recoveries_run"] >= 1 and r["outcome"] in ("budget", "completed", "graph_error")
