"""box_bookcase gate (owner-written, 11-16).  SIMULATION ONLY, CPU.  USER 11-16: "take the box and turn around, move and place it inside a shelf a little
steps away"; 2-level bookcase, ~2.5 m, a real checked task; sizes "as long as a solution is possible"; then "Wide + plinth" (the 0.60 m / full-floor / 0.10-0.20 m
insert sizes are infeasible with the end-tab grasp: arm links at |dy| 0.44-0.52 m and 6-7 cm below the support top; insert reach ~0.05 m).

Checked here:
  1. the TaskDef goal equals the scene (place_pose_xy - obj_xy), the goal support is the plinth top at counter height, the route matches the scene constants;
  2. the old table-behind scenes are untouched (no bookcase geom, counter2 present) and the bookcase scene has no counter2;
  3. the default plan builds; collision rows pass; every arm-world distance to a bookcase geom over insert + place is >= 20 mm (ASSUMED margin of this gate);
  4. Rule 1 admits lift, insert and place; base_move is refused as MISSING (no stored base recovery; known gap).
"""
import numpy as np
import pytest

from world_model.v2.adapters import contents_specs as CS
from world_model.v2.nero.base import scene as BS
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack import tasks as TK

TASK = "box_bookcase"
BOOKCASE_GEOMS = ("bookcase_l1_base", "bookcase_plinth", "bookcase_l1_floor", "bookcase_side_left", "bookcase_side_right", "bookcase_mid", "bookcase_back", "bookcase_top")
ARM_BOOKCASE_MIN_M = 0.020   # ASSUMED: this gate's margin (the agent measured 36.9 mm; collision_rows d_min is the 5 mm D9 bound)


def _geom_names(task):
    import mujoco
    w = CS.build_world(task)
    spec = BS.build_spec(w)
    m = spec.compile()
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)}, w


def test_goal_and_route_match_the_scene():
    t = TK.task(TASK)
    w = CS.build_world(TASK)
    assert w.bookcase and not w.shelf
    assert t.base == "swerve" and t.goal_support == "bookcase_l1" and t.moves == ()
    assert np.allclose(t.goal_dxy, BS.place_pose_xy(w) - np.asarray(w.obj_xy), atol=1e-4)
    assert 2.4 <= float(np.hypot(*t.goal_dxy)) <= 2.6                                   # "about 2.5 m"
    sup = BS.support_tops_behind(w)
    assert sup["bookcase_l1"]["top"] == pytest.approx(float(w.counter_top))              # plinth top = counter top (same arm pose as at pick)
    route = dict(t.routes)[t.routes[0][0]]
    lines = [v for k, v in route if k == "line"]
    assert ("turn", 180.0) in route and lines[0] == pytest.approx(-BS.BC_RETREAT_M) and sum(lines[1:]) == pytest.approx(BS.BC_ADVANCE_M)


def test_old_scenes_untouched_and_bookcase_scene_has_its_geoms():
    for n in ("box_behind", "tray6_behind"):
        names, _ = _geom_names(n)
        assert "counter2" in names and not any(x and x.startswith("bookcase") for x in names), n
    names, _ = _geom_names(TASK)
    assert set(BOOKCASE_GEOMS) <= names and "counter2" not in names


@pytest.fixture(scope="module")
def plan():
    from world_model.v2.stack.planners import closed_chain as CCP
    return CCP.plan_for(CCP._nominal_inputs(TASK), 0.5)


def test_plan_builds_and_clears_the_bookcase(plan):
    from world_model.v2.adapters import collision_rows as CR_
    rows = CR_.collision_rows(plan)
    rows = rows if isinstance(rows, (list, tuple)) else [rows]
    assert all(r.ok for r in rows), [(r.name, r.why) for r in rows if not r.ok]
    sc = CR_._Scene(plan)
    m = sc.m
    allp = list(sc.pairs["arm_world"])
    worst = {}
    for nm in BOOKCASE_GEOMS:
        sc.pairs["arm_world"] = [(g, wg) for g, wg in allp if wg >= 0 and m.geom(wg).name == nm]
        if not sc.pairs["arm_world"]:
            continue
        for sg in ("s1_insert", "s2_place"):
            _, res = CR_.piece_distances(plan, sg, sc)
            worst[nm] = min(worst.get(nm, 9.0), float(np.min(res["arm_world"]["d"])))
    sc.pairs["arm_world"] = allp
    assert worst and min(worst.values()) >= ARM_BOOKCASE_MIN_M, worst


@pytest.mark.slow
def test_rule1_admits_the_arm_steps_and_flags_the_base_step(tmp_path):
    from world_model.v2.adapters import rule1_contents as RC
    recs = {r["seg"]: r for r in RC.run_cell(TASK, (1.0, 1.5), tmp_path, n_sampled=0, planner="closed_chain", fallback=False)}
    for seg in ("s0_lift", "s1_insert", "s2_place"):
        assert recs[seg]["admitted"], (seg, recs[seg].get("refusal"))
    base = [r for s, r in recs.items() if "base" in s]
    assert base and not base[0]["admitted"] and "recovery" in (base[0].get("refusal") or "")
