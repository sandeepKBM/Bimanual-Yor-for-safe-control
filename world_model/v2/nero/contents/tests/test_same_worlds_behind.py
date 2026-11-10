"""same_worlds on the table-behind tasks: one world list for both methods, the base-level pick, the counter2 support row, the plan mapping.  SIMULATION ONLY, CPU, no model call.
The cross-method world test needs the AgentChord venv (run with /data/local/ss5772/agentchord/venv_nero/bin/python); it is skipped without it."""
import pytest

from world_model.v2.adapters import base_envelope as BE
from world_model.v2.nero.base import smoke as SM
from world_model.v2.nero.contents import carry_cert as CC, probe as P, same_worlds as SW
from world_model.v2.stack import tasks as TK

BEHIND = ("tray6_behind", "box_behind")


@pytest.mark.parametrize("task", BEHIND)
def test_one_world_list_for_both_methods(task):
    pytest.importorskip("langchain_core")
    from world_model.v2.agentchord_probe.nero import behind as BH
    for i in range(3):
        s1, w1 = SM.behind_world(task, i)
        s2, w2 = BH._world(task, i)                                     # AgentChord's runner builds its world through the same function
        assert s1 == s2 == P._seed("e2e", task[:-len("_behind")], i) and repr(w1) == repr(w2)
    assert repr(SM.behind_world(task, 0)[1]) != repr(SM.behind_world(task, 1)[1])
    if task == "box_behind":
        assert SM.behind_world(task, 0)[1].shelf is False


@pytest.mark.parametrize("task,bt", zip(BEHIND, ("tray6", "box")))
def test_pick_level_rows_and_monotone_in_mass(task, bt):
    lo, hi = P.prior_contents(bt)
    lv_lo, rows_lo = BE.pick_level(task, P.interval_spec(bt, lo, lo))
    lv_hi, rows_hi = BE.pick_level(task, P.interval_spec(bt, hi, hi))
    for rows in (rows_lo, rows_hi):
        assert set(rows) == {"L1", "L2", "L3"}
        names = {"base_tip", "base_torque"} | ({"base_open_content"} if task.startswith("tray") else set())
        assert all(set(r) == names for r in rows.values())
        assert all(r.guarantee == "sampled" for rr in rows.values() for k, r in rr.items() if k in ("base_tip", "base_torque"))     # measured card = sampled
    order = {None: 0, "L1": 1, "L2": 2, "L3": 3}
    assert order[lv_hi] <= order[lv_lo]                                  # a heavier contents set never gets a higher level
    best = lv_lo
    assert best is None or all(r.ok for r in rows_lo[best].values())


def test_support_row_is_on_counter2_for_behind_tasks():
    w = SM.task_world("tray6")
    spec = P.interval_spec("tray6", *P.prior_contents("tray6"))
    row = CC.support_row("tray6_behind", spec, spec.contents, w, None)
    from world_model.v2.nero.base import scene as BS
    c2 = BS.support_tops_behind(w)["counter2"]
    assert row["support"]["x"] == c2["x"] and row["support"]["y"] == c2["y"]       # counter 2, not counter 1
    assert row["ok"] in (True, False) and "margin_m" in row
    # box_behind: no shelf, no KeyError
    wb = SM.task_world("box")
    sb = P.interval_spec("box", *P.prior_contents("box"))
    assert "margin_m" in CC.support_row("box_behind", sb, sb.contents, wb, None)


def test_plan_map_and_summary_columns(tmp_path):
    assert SW._plan_map("a", ["x", "y"]) == {"x": "a", "y": "a"}
    assert SW._plan_map("x=d1,y=d2", ["y", "x"]) == {"y": "d2", "x": "d1"}
    with pytest.raises(SystemExit):
        SW._plan_map("x=d1", ["x", "y"])
    import json
    rec = lambda m, t, lv, ok: dict(task=t, method=m, i=0, any_U=False, outcome="carried", base_level=lv, theta_sound=True,
                                    result=dict(success=ok, unsafe=False, in_R=True, safe_success=ok, refused=False))
    (tmp_path / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in [rec("ours", "tray6_behind", "L2", True), rec("theirs", "tray6_behind", "L3", False),
                                                                          rec("ours", "tray6_behind", None, False)]) + "\n")
    SW.summary(tmp_path)
    txt = (tmp_path / "SUMMARY.md").read_text()
    assert "base level (count)" in txt and "{'L2': 1, 'none': 1}" in txt and "{'L3': 1}" in txt
    assert TK.task("tray6_behind").base
