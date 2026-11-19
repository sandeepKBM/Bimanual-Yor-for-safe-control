"""Owner gate (11-17): Rule 1 of the base_move step through adapters/rule1_contents.run_cell, on the Codex run-2 bookcase plan (reply 3), cell (1.0, 1.5) kg.
USER 11-16 "go with a and build the base stop check"; USER 11-17 rulings: resume = finish the planned route (base_route.remaining_route), safe = on a support,
the measured extra overshoot is always added.  SIMULATION ONLY, offline.  Slow (minutes); the result records go to BIMANUAL_GATE_OUT if set.

What is pinned here is the STRUCTURE.  The verdicts themselves are printed and written out (they depend on the MEASURED stop card)."""
import json
import os
import pathlib

import pytest

from world_model.v2.adapters import rule1_contents as RC
from world_model.v2.stack.planners import closed_chain as CCP


PLAN = pathlib.Path(__file__).resolve().parents[2] / "nero" / "results" / "bookcase_codex_2026-11-16" / "run2" / "plans" / "box_bookcase.json"
CELL = (1.0, 1.5)
EXPECT_ROWS = {"base_stop.recovery", "base_stop.in_route", "base_stop.at_rest", "base_stop.tip", "base_stop.torque", "base_stop.landing_hold",
               "base_stop.continuation", "base_stop.goal_support", "base_stop.next_step"}


def _wp():
    return CCP.WaypointPlan.from_data(json.loads(PLAN.read_text())["compiled"]["wp"])


def _run(tmp_path, env=None):
    old = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        return RC.run_cell("box_bookcase", CELL, tmp_path, n_sampled=0, planner="closed_chain", fallback=False, wp=_wp())
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _dump(name, res):
    d = os.environ.get("BIMANUAL_GATE_OUT")
    if d:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(d) / name).write_text(json.dumps(res, indent=1, default=str))


def test_base_move_record_has_every_row(tmp_path):
    res = _run(tmp_path)
    _dump("rule1_box_bookcase_default.json", res)
    segs = [r["seg"] for r in res]
    assert "base_move" in segs, segs
    b = res[segs.index("base_move")]
    names = {x["name"] for x in b["all_rows"]}
    assert names == EXPECT_ROWS, names ^ EXPECT_ROWS
    assert b["programs"] == ["base_stop_finish"] and b["landings"] == {"base_stop_finish": "set_down:goal"}
    nxt = next(x for x in b["all_rows"] if x["name"] == "base_stop.next_step")
    assert nxt["guarantee"] != "missing", "link_base_next must fill the next-step row from the later steps"
    assert b["card"] is not None, "the MEASURED stop card must be read (CARD.json)"
    # a verdict is either admitted with every row ok, or refused naming only failing rows
    if b.get("admitted"):
        assert all(x["ok"] for x in b["all_rows"]) and b["program"] == "base_stop_finish" and b["guarantee"] in ("sampled", "proof")
    else:
        assert b["rows"] and not any(x["ok"] for x in b["rows"]) and b["refusal"].startswith("base_stop recovery:")
    for r in res:                                   # no hold landing counts as safe any more (USER 11-17)
        for lf in (r.get("rows") or []):
            if lf["name"].startswith("landing[hold:"):
                assert not lf["ok"], lf
    print({r["seg"]: (r.get("admitted"), r.get("program"), r.get("guarantee"), [x["name"] for x in (r.get("rows") or []) if not x.get("ok")]) for r in res})
