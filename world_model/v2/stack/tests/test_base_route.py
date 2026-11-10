"""D1 (table-behind, 11-09): base_move skill (schema / compile / prompt / mock) and stack/planners/base_route.py.  SIMULATION ONLY.
Pure tests first; the MuJoCo ones (adapters/base_sweep) build the two-table swerve scene (a few seconds each)."""
import copy
import math

import numpy as np
import pytest

from world_model.v2.stack.planners import base_route as BR
from world_model.v2.stack.planners import closed_chain as CCP
from world_model.v2.stack.proposer import compile as CP
from world_model.v2.stack.proposer import mock as MK
from world_model.v2.stack.proposer import prompt as PR
from world_model.v2.stack.proposer import schema as S

R1 = (("line", -0.5), ("turn", 180.0), ("line", 0.5))


# ---- pure ---------------------------------------------------------------------------------------------------------------------------
def test_route_pure():
    assert BR.normalize_route([{"line": -0.5}, ["turn", 180], ("line", 0.5)]) == R1
    with pytest.raises(ValueError):
        BR.normalize_route([{"line": 1, "turn": 2}])
    with pytest.raises(ValueError):
        BR.normalize_route([{"spin": 1}])
    x, y, yaw = BR.route_end((0.0, -0.2, 0.0), R1)
    assert abs(x + 1.0) < 1e-12 and abs(y + 0.2) < 1e-12 and abs(yaw - math.pi) < 1e-12           # R1: net 1.0 m backwards, facing -x
    ps = BR.route_poses((0.0, -0.2, 0.0), R1)
    assert ps[0] == (0, 0.0, -0.2, 0.0) and len(ps) == 1 + 50 + 90 + 50
    assert np.allclose(ps[-1][1:], BR.route_end((0.0, -0.2, 0.0), R1))
    assert [p[0] for p in ps] == sorted(p[0] for p in ps)
    assert BR.check_ranges(R1) == []
    assert len(BR.check_ranges((("line", 1.5),))) == 1 and len(BR.check_ranges((("turn", 181.0),))) == 1
    assert len(BR.check_ranges((("line", 0.1),) * 5)) == 1 and len(BR.check_ranges(())) == 1


def test_trap_duration():
    assert abs(BR.trap_T(1.0, 0.25, 0.25) - (0.25 / 0.25 + 1.0 / 0.25)) < 1e-12                       # trapezoid
    assert abs(BR.trap_T(0.1, 0.25, 0.25) - 2 * math.sqrt(0.1 / 0.25)) < 1e-12                        # triangle
    lv = dict(L1=(0.25, 0.25, 0.5, 0.5))
    T = BR.route_duration(R1, "L1", lv)
    assert abs(T - (BR.trap_T(0.5, .25, .25) * 2 + BR.trap_T(math.pi, .5, .5) + 3 * BR.GAP_S)) < 1e-9


# ---- schema / prompt / mock ----------------------------------------------------------------------------------------------------------------
def _valid(task="tray6_behind"):
    d = MK.fixture_plan(task)
    return d, S.validate(d, task, ("counter", "counter2"))


def test_schema_accepts_fixture_and_rejects_bad_routes():
    d, tp = _valid()
    assert [s.call.skill for s in tp.steps] == ["lift", "base_move", "place", "release"]
    assert tp.steps[1].call.args == {"route": [{"line": -0.5}, {"turn": 180.0}, {"line": 0.5}]}
    for bad, frag in (([{"line": 1.5}], "outside"), ([{"turn": 200}], "outside"), ([{"line": 0.1}] * 5, "segments"), ([], "non-empty"),
                      ([{"line": 1, "turn": 2}], "exactly one key"), ([{"line": "x"}], "finite"), ("R1", "non-empty")):
        d2 = copy.deepcopy(d)
        d2["steps"][1]["args"]["route"] = bad
        with pytest.raises(S.PlanRejected) as e:
            S.validate(d2, "tray6_behind", ("counter", "counter2"))
        assert frag in " ".join(e.value.reasons), (bad, e.value.reasons)
    d2 = copy.deepcopy(d)                                              # a second base_move
    d2["steps"].insert(2, copy.deepcopy(d2["steps"][1]))
    d2["recoveries"] = {str(i): [dict(skill="hold", args={})] for i in range(4)}
    with pytest.raises(S.PlanRejected) as e:
        S.validate(d2, "tray6_behind", ("counter", "counter2"))
    assert "at most one base_move" in " ".join(e.value.reasons)
    d2 = copy.deepcopy(d)                                              # let_go is accepted by the schema (the checker refuses it)
    d2["recoveries"]["1"] = [dict(skill="let_go", args={})]
    S.validate(d2, "tray6_behind", ("counter", "counter2"))
    d2["recoveries"].pop("1")                                          # base_move is a motion step: it needs a recovery
    with pytest.raises(S.PlanRejected) as e:
        S.validate(d2, "tray6_behind", ("counter", "counter2"))
    assert "needs at least one recovery" in " ".join(e.value.reasons)


def test_fixed_base_schema_unchanged():
    assert "base_move" not in S.vocabulary()["steps"] and "base_move" not in S.vocabulary(("counter", "shelf"), base=False)["steps"]
    assert S.vocabulary(("counter", "counter2"), base=True)["steps"]["base_move"]["route"]["segments"] == "1..4"
    for t in ("tray6", "box", "pot"):
        d = MK.fixture_plan(t)
        S.validate(d, t, ("counter", "shelf"))
        d2 = copy.deepcopy(d)
        d2["steps"].insert(1, dict(skill="base_move", args=dict(route=[dict(line=-0.5)]), why="x"))
        with pytest.raises(S.PlanRejected) as e:
            S.validate(d2, t, ("counter", "shelf"))
        assert "unknown skill 'base_move'" in " ".join(e.value.reasons)
    assert S.step_skills("tray6") == S.STEP_SKILLS and "base_move" in S.step_skills("box_behind")


def test_prompt_behind():
    sp = PR.task_spec("tray6_behind")
    assert set(sp["supports"]) == {"counter", "counter2"} and sp["goal_support"] == "counter2"
    assert abs(sp["goal_xy"][0] + 1.38) < 1e-3 and abs(sp["goal_xy"][1] + 0.1991) < 1e-3
    t = sp["instruction"]
    for frag in ("counter 2", "base_move", "+x axis turns with the base", "Known-good route R1", "Turning in place at the pick pose hits counter 1", "recovery"):
        assert frag in t, frag
    f = PR.scene_facts("tray6_behind")
    assert "base_move" in f["vocabulary"]["steps"] and "hidden_parameters" not in f and f["phenomena"] and f["task"]["base"]["routes_known_good"]["R1"]     # schema v2 (11-09)
    assert "prior" not in str(f).lower().replace("prior box", "") and "sanity" not in str(f).lower()
    f6 = PR.scene_facts("tray6")                                       # fixed task: no base block, no base_move, no ranges either
    assert "base" not in f6["task"] and "hidden_parameters" not in f6 and "base_move" not in f6["vocabulary"]["steps"]
    import os
    os.environ["BIMANUAL_LEGACY_PROPOSER_V1"] = "1"                    # the 11-05..11-09 facts stay replayable
    try:
        f1, f61 = PR.scene_facts("tray6_behind"), PR.scene_facts("tray6")
    finally:
        del os.environ["BIMANUAL_LEGACY_PROPOSER_V1"]
    assert "sanity_range" not in str(f1["hidden_parameters"]) and "sanity_range" in f61["hidden_parameters"]["fill_total_kg"]
    assert PR.task_spec("box_behind")["object"]["name"] == "closed crate"


# ---- compile (no geometry build) --------------------------------------------------------------------------------------------------------
def _compile(d, task="tray6_behind", level="L1", build=False):
    tp = S.validate(d, task, ("counter", "counter2"))
    return CP.compile_plan(tp, task, build=build, level=level)


def test_compile_behind_ok_and_rules():
    d = MK.fixture_plan("tray6_behind")
    cp = _compile(d)
    assert cp.wp.names == ("s0_lift", "s2_place") and cp.wp.base == ((1, R1, "L1"),)
    assert cp.wp.deltas[1] == (0.0, 0.0, -0.05)                        # base-frame offsets: the object keeps its base-frame position across the base move
    assert cp.goal["miss_m"] < 1e-6 and cp.goal["end_support"] == "counter2" and abs(cp.goal["base_end_pose"][2] - math.pi) < 1e-6
    assert CP.CompiledPlan.from_json(cp.to_json()).wp == cp.wp         # level survives the JSON round trip
    with pytest.raises(S.PlanRejected) as e:                           # level None -> no level passes the tip row
        _compile(d, level=None)
    assert e.value.stage == "planner" and "no base level passes the tip row" in " ".join(e.value.reasons)
    d2 = copy.deepcopy(d)                                              # route that ends elsewhere: goal miss
    d2["steps"][1]["args"]["route"] = [dict(line=-0.5), dict(turn=180.0), dict(line=0.3)]
    with pytest.raises(S.PlanRejected) as e:
        _compile(d2)
    assert "from the goal" in " ".join(e.value.reasons) or "not over the support" in " ".join(e.value.reasons)
    d2 = copy.deepcopy(d)                                              # object not lifted at base_move start
    d2["steps"][0]["args"]["dz"] = 0.02
    d2["steps"][0]["args"]["dz"] = 0.02
    cp2 = _compile(d2)                                                 # 2 cm = MIN_CLEAR_M is allowed by compile (the planner's CLEAR_M check is stricter)
    assert cp2.wp.deltas[0] == (0.0, 0.0, 0.02)
    d2 = copy.deepcopy(d)                                              # no base_move: the goal is behind the robot
    d2["steps"].pop(1)
    d2["recoveries"] = {"0": [dict(skill="set_down", args={})], "1": [dict(skill="hold", args={})]}
    with pytest.raises(S.PlanRejected) as e:
        _compile(d2)
    assert "no base_move" in " ".join(e.value.reasons)
    d2 = copy.deepcopy(d)                                              # place on counter 1
    d2["steps"][2]["args"]["support"] = "counter"
    with pytest.raises(S.PlanRejected):
        _compile(d2)


def test_fixed_base_compile_unchanged():
    d = MK.fixture_plan("tray6")
    tp = S.validate(d, "tray6", ("counter", "shelf"))
    cp = CP.compile_plan(tp, "tray6", build=False)
    assert cp.wp.base == () and cp.wp.names == ("s0_lift", "s1_move", "s2_place")
    assert "base" not in cp.wp.key_data() and "base" not in cp.to_json()["wp"]


# ---- MuJoCo: sweep + planner ----------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tray_inp():
    return CCP._nominal_inputs("tray6_behind")


def test_default_plan_is_r1_and_sweeps_clear(tray_inp):
    wp = tray_inp.wp
    assert wp.names == ("s0_lift", "s2_place") and wp.base[0][1] == R1
    g = CCP.geometry_for(tray_inp)
    b = g.info["base"]
    assert b["sweep_min_dist_m"] >= b["clearance_min_m"] and b["n_poses"] == 191
    assert b["z_under_counter2_m"] >= CCP.CLEAR_M["tray"] - 1e-9 and b["table2_margin_min"] > 0
    assert min(b["sweep_by_segment"].values()) == b["sweep_min_dist_m"]
    pl = CCP.ClosedChainPlanner().plan(None, CCP.SkillCall("plan", {"task": "tray6_behind", "a_plan": 0.5}), inputs=tray_inp)
    modes = [s.mode for s in pl.segments]
    assert modes == ["s0_lift", "carry", "s2_place", "release"]
    sb = pl.segments[1]
    assert sb.meta["phases"] == [("base_move", sb.meta["base_ref"])] and sb.meta["base_ref"]["level"] == "L1" and sb.meta["backup_phases"] is None
    assert sb.meta["accel"] == dict(source="base_card", level="L1") and sb.t1 - sb.t0 > 5.0
    assert all(pl.segments[i].t1 <= pl.segments[i + 1].t0 + 1e-9 for i in range(3))
    xo = pl.segments[2].x_obj(0.0)                                     # after the base move: the object sits over the counter-2 goal in the WORLD frame
    assert abs(xo[0] + 1.38) < 5e-3 and abs(xo[1] + 0.1991) < 5e-3 and abs(xo[2] - 0.05) < 1e-6


def test_turn_in_place_at_pick_pose_refused(tray_inp):
    w = tray_inp.wp
    wp = CCP.WaypointPlan(w.names, w.deltas, w.backups, True, ((1, (("turn", 180.0),), "L1"),))
    inp = CCP.PlanInputs(tray_inp.m, tray_inp.world, tray_inp.q0, "tray6_behind", wp=wp)
    with pytest.raises(CCP.PlanRefused) as e:
        CCP.geometry_for(inp)
    assert e.value.info["reason"] == "base_sweep" and e.value.info["segment"] == 0 and e.value.info["dist_m"] < BR.default_backend().CLEARANCE_MIN
    assert "base_move segment 0 (turn 180 deg) collides" in str(e.value) and len(e.value.info["pair"]) == 2


def test_height_over_counter2_refused(tray_inp):
    w = tray_inp.wp
    wp = CCP.WaypointPlan(w.names, ((0.0, 0.0, 0.03), (0.0, 0.0, -0.03)), w.backups, True, w.base)
    inp = CCP.PlanInputs(tray_inp.m, tray_inp.world, tray_inp.q0, "tray6_behind", wp=wp)
    with pytest.raises(CCP.PlanRefused) as e:
        CCP.geometry_for(inp)
    assert e.value.info["reason"] == "base_height"
