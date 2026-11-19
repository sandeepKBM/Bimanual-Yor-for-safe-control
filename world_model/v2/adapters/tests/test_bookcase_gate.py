"""box_bookcase gate (owner-written, 11-16).  SIMULATION ONLY, CPU.  USER 11-16: "take the box and turn around, move and place it inside a shelf a little
steps away"; 2-level bookcase, ~2.5 m, a real checked task; sizes "as long as a solution is possible"; then "Wide + plinth" (the 0.60 m / full-floor / 0.10-0.20 m
insert sizes are infeasible with the end-tab grasp: arm links at |dy| 0.44-0.52 m and 6-7 cm below the support top; insert reach ~0.05 m).

Checked here:
  1. the TaskDef goal equals the scene (place_pose_xy - obj_xy), the goal support is the plinth top at counter height, the route matches the scene constants;
  2. the old table-behind scenes are untouched (no bookcase geom, counter2 present) and the bookcase scene has no counter2;
  3. the default plan builds; collision rows pass; every arm-world distance to a bookcase geom over insert + place is >= 20 mm (ASSUMED margin of this gate);
  3b. 11-17 tilt-aware row (USER): the load tilt is MEASURED for the default plan (tilt_measure.measure_plan, n = 20 ASSUMED, cell 1.0-1.5, a = 0.5) and the
     collision row and Rule 1 use it (sampled).  Owner check 11-18: collision ok (sampled); Rule 1 admits all four steps.
  4. Rule 1 admits lift, insert and place; base_move is checked by the base-stop rows (11-17: program base_stop_finish, MEASURED per-level stop card):
     admitted, or refused by named base_stop.* rows -- never the old "no recovery" MISSING.
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
def wp():
    from world_model.v2.stack.planners import closed_chain as CCP
    return CCP.default_waypoint(TASK, CS.build_world(TASK))


@pytest.fixture(scope="module")
def plan(wp):
    from world_model.v2.stack.planners import closed_chain as CCP
    return CCP.plan_for(CCP._nominal_inputs(TASK, wp), 0.5)


@pytest.fixture(scope="module")
def tilt(wp, tmp_path_factory):
    """the per-plan load tilt, MEASURED as probe_select measures it (n = 20 ASSUMED, sim only)."""
    from world_model.v2.adapters import collision_rows as RCOL
    from world_model.v2.nero.contents import tilt_measure as TM
    tab, info = TM.measure_plan(TASK, wp, (1.0, 1.5), 0.5, n=20, procs=8, out_dir=tmp_path_factory.mktemp("tilt"), plan_id="bookcase_gate")
    return RCOL.tilt_from_table(tab, TM.cell_key(TASK, (1.0, 1.5)), 0.5, source=info["table_path"])


def test_plan_builds_and_clears_the_bookcase(plan, tilt):
    from world_model.v2.adapters import collision_rows as CR_
    rows = CR_.collision_rows(plan, tilt=tilt)
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
def test_rule1_admits_the_arm_steps_and_flags_the_base_step(tmp_path, wp, tilt):
    from world_model.v2.adapters import rule1_contents as RC
    recs = {r["seg"]: r for r in RC.run_cell(TASK, (1.0, 1.5), tmp_path, n_sampled=0, planner="closed_chain", fallback=False, wp=wp, a=0.5, tilt=tilt)}
    for seg in ("s0_lift", "s1_insert", "s2_place"):
        assert recs[seg]["admitted"], (seg, recs[seg].get("refusal"))
    base = [r for s, r in recs.items() if "base" in s]
    assert base, sorted(recs)
    b0 = base[0]
    names = [r.get("name") or r.get("row") for r in b0.get("all_rows", [])]
    assert names and all(str(n).startswith("base_stop.") for n in names), names         # the base step is checked by the base-stop rows
    assert "base_stop.next_step" in names                                                 # run_cell links the steps after the base move
    if not b0["admitted"]:
        bad = [(r.get("name") or r.get("row"), r.get("why")) for r in b0["all_rows"] if not r.get("ok")]
        assert bad and all(str(n).startswith("base_stop.") for n, _ in bad), bad
        assert "no stored" not in (b0.get("refusal") or "") and "not built" not in (b0.get("refusal") or ""), b0.get("refusal")
    else:
        assert b0.get("program") == "base_stop_finish" and b0.get("guarantee") in ("sampled", "proof"), (b0.get("program"), b0.get("guarantee"))
