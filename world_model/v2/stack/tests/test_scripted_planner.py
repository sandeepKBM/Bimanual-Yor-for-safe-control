"""ScriptedPlanner: plan structure only (no MuJoCo)."""
from world_model.v2.nero.contents import carry as CR
from world_model.v2.stack.planners.scripted import ScriptedPlanner
from world_model.v2.stack.types import Planner, SkillCall


def test_tray2_four_segments_legacy():
    p = ScriptedPlanner()
    assert isinstance(p, Planner)
    plan = p.plan(None, SkillCall("carry_task", dict(task="tray", a_plan=0.5)), None)
    assert [s.mode for s in plan.segments] == ["lift", "carry", "place", "release"]
    assert plan.planner == "scripted" and plan.provenance["legacy"] is True
    for s in plan.segments:
        assert s.meta["legacy"] and s.meta["phases"]
    assert plan.segments[0].meta["phases"][0][:2] == ("move", (0.0, 0.0, 0.05))
    assert plan.segments[3].meta["phases"][1] == ("weld", False)
    assert plan.segments[3].meta["backup_phases"] is None and plan.segments[3].meta["accel"] is None
    assert plan.segments[2].meta["backup_phases"][0] == ("stop", 0.3)
    assert plan.segments[0].t0 == 0.0 and plan.segments[1].t0 == plan.segments[0].t1


def test_matches_carry_helpers():
    plan = ScriptedPlanner().plan(None, SkillCall("carry_task", dict(task="box", a_plan=0.4)), None)
    for s in plan.segments:
        if s.mode != "release":
            assert s.meta["accel"]["a_plan"] == 0.4
    assert set(CR.PLANS) == {"tray", "box"}
