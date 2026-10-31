"""scenarios tests (CPU; builds a few TeleopSim rigs, ~15 s each).  python -m pytest -q world_model/v2/house/scenarios/tests"""
import json
import math

import numpy as np
import pytest

from world_model.v2.house.scenarios import (ScenSim, ScenarioError, corridor as C, occlusion as O, patched, dist_to_route, run_scen as RS)
from world_model.v2.house.teleop import run as RUN, scene as SC
from world_model.v2.house.teleop.sim import Cmd, TeleopSim

ROUTE = SC.get_scene("h5").out_route


# ------------------------------------------------------------------------------------------------ construction (no sim)
def test_corridor_scene_deterministic_and_single_person():
    a, b = C.build_scene(C.CorridorSpec(0.8)), C.build_scene(C.CorridorSpec(0.8))
    assert [p.name for p in a.people] == [C.NAME] and len(b.people) == 1
    assert np.array_equal(a.people[0].pts, b.people[0].pts)
    assert not np.array_equal(a.people[0].pts, C.build_scene(C.CorridorSpec(1.0)).people[0].pts)
    assert a.out_route == ROUTE                                    # the route itself is untouched


@pytest.mark.parametrize("d", [0.4, 0.6, 0.9, 1.4])
@pytest.mark.parametrize("side", ["right", "left"])
def test_corridor_offset_matches_request(d, side):
    spec = C.CorridorSpec(d=d, arc_s=3.6, side=side)
    try:
        sc = C.build_scene(spec)
    except ScenarioError:                                          # left side (east) is wall-limited at 1.4: must be refused, not silently moved
        assert side == "left" and d > C.max_valid_d(spec)
        return
    xy = sc.people[0].pts[0, 1:3]
    assert abs(dist_to_route(xy, ROUTE) - d) < 0.02
    assert np.allclose(sc.people[0].pts[:, 1:3], xy)               # constant track (zero velocity forever)


def test_corridor_invalid_placement_raises():
    with pytest.raises(ScenarioError):
        C.build_scene(C.CorridorSpec(d=1.4, arc_s=2.75, side="right"))      # inside the west wall of the passage


def test_occlusion_deterministic_seeded_jitter_and_speed_cap():
    a = O.track_points(O.OcclusionSpec(seed=3))
    assert np.array_equal(a, O.track_points(O.OcclusionSpec(seed=3)))
    ts = {round(O.effective_t_appear(O.OcclusionSpec(seed=s)), 6) for s in range(6)}
    assert len(ts) > 1 and all(abs(t - 4.5) <= 0.5 + 1e-9 for t in ts)
    assert np.all(np.diff(a[:, 0]) > 0)
    seg_v = np.hypot(*np.diff(a[:-1, 1:3], axis=0).T) / np.diff(a[:-1, 0])
    assert seg_v.max() <= 1.2 + 1e-9
    with pytest.raises(ScenarioError):
        O.track_points(O.OcclusionSpec(speed=1.5))
    z = O.track_points(O.OcclusionSpec(t_appear=0.2, jitter_s=0.0))                # starts walking at t=0 (start point moved along the path)
    assert z[0, 0] == 0.0 and np.all(np.diff(z[:, 0]) > 0)


def test_patched_restores_and_registry():
    old = RUN.TeleopSim
    key = C.install(C.CorridorSpec(0.7))
    assert key in SC.SCENES
    with patched():
        assert RUN.TeleopSim is ScenSim
    assert RUN.TeleopSim is old


def test_run_scen_params_and_summary():
    assert RS.parse_params("0.4:1.4:0.5") == [0.4, 0.9, 1.4] and RS.parse_params("0.6,1.0") == [0.6, 1.0]
    assert RS.parse_seeds("0-2") == [0, 1, 2]
    rows = [dict(scenario="corridor", param=0.6, config="x", seed=s, error=None, task_done=bool(s), task_time_s=9.0 * s, stall_s=1.0,
                 min_person_distance_m=0.3 + s, min_wall_clearance_m=0.2, person_contact_ticks=0, wall_collision_ticks=0, physical_house_contacts=0,
                 spilled=False, final_dist_to_goal_m=1.0, gk=dict(n_commit=2, step_ms_max=5.0)) for s in (0, 1)]
    s = RS.summarize(rows)["corridor|0.6|x"]
    assert s["n"] == 2 and s["done_rate"] == 0.5 and s["mean_time_done_s"] == 9.0 and s["min_person_m"] == 0.3 and s["gk"]["n_commit"] == 4


# ------------------------------------------------------------------------------------------------ with a sim
@pytest.fixture(scope="module")
def corridor_sim():
    key = C.install(C.CorridorSpec(d=0.8))
    return ScenSim(key, seed=0)


def test_standing_person_at_offset_and_does_not_move(corridor_sim):
    sim = corridor_sim
    assert sim.people_clock == "time" and len(sim.people_state) == 1
    xy0 = np.array(sim.people_state[0][0])
    assert abs(dist_to_route(xy0, sim.scene.out_route) - 0.8) < 0.02
    assert np.allclose(sim.d.mocap_pos[sim.rig.person_mocap[0]][:2], xy0)
    for _ in range(60):
        sim.tick(Cmd())
    xy, v = sim.people_state[0]
    assert np.allclose(xy, xy0, atol=1e-12) and np.allclose(v, 0.0)
    assert np.allclose(sim.d.mocap_pos[sim.rig.person_mocap[0]][:2], xy0)


@pytest.fixture(scope="module")
def occ_run():
    """stationary robot at the spawn pose (head camera looks SW over the north approach); the person walks out at t_appear = 2.5 s (no jitter)."""
    spec = O.OcclusionSpec(t_appear=2.5, jitter_s=0.0, speed=1.2)
    sim = ScenSim(O.install(spec), seed=0)
    vs, ps = [], []
    first = sim.vis_log                                            # list is appended in place
    f0 = sim._visibility().of(sim, np.array(sim.people_state[0][0]))
    for _ in range(int(7.0 / 0.02)):
        sim.tick(Cmd())
        vs.append(float(np.hypot(*sim.people_state[0][1]))); ps.append(sim.people_state[0][0].copy())
    return sim, f0, np.array(vs), np.array(ps)


def test_occlusion_person_initially_hidden_then_visible(occ_run):
    sim, f0, vs, ps = occ_run
    assert not f0[1]                                                # t=0, robot at the spawn: NOT visible (the frustum may contain the point, the wall blocks the line of sight)
    log = sim.vis_log
    assert log and not log[0][2]                                    # first logged sample: not visible
    seen = [r[0] for r in log if r[2]]
    assert seen, "person never became visible"
    assert 2.0 <= seen[0] <= 5.0                                    # becomes visible as it leaves cover (t_appear = 2.5 s + a fraction of the crossing)
    assert sim.extras()["first_visible_t"] == seen[0]


def test_occlusion_hide_point_hidden_from_every_approach_pose(occ_run):
    import mujoco
    from world_model.v2.house import build as B
    sim = occ_run[0]
    vis = sim._visibility()
    hide = np.array(O.OcclusionSpec().hide)
    for x, y, yaw in [(3.1, 6.09, -2.33), (2.26, 5.4, -2.02), (2.07, 4.08, -1.46), (2.13, 3.5, -1.57), (2.13, 3.1, -1.57), (2.14, 2.46, -1.5)]:
        sim.base.x, sim.base.y, sim.base.yaw = x, y, yaw
        sim.d.mocap_pos[0] = (x, y, 0.0)
        sim.d.mocap_quat[0] = B.yaw_quat(yaw)
        mujoco.mj_forward(sim.m, sim.d)
        assert not vis.of(sim, hide)[1], (x, y, yaw)


def test_occlusion_person_walks_across_route_at_speed_cap(occ_run):
    sim, f0, vs, ps = occ_run
    assert vs.max() <= 1.2 + 1e-6 and vs.max() > 1.0
    assert ps[0][0] < 1.0 and abs(ps[-1][0] - 3.0) < 1e-6           # hidden side -> far side of the route
    assert np.any((ps[:, 0] > 2.0) & (ps[:, 0] < 2.27))             # crosses the route centreline x = 2.134


def test_run_scen_one_short_episode_per_scenario():
    for scen, param, kw in (("corridor", 0.8, {}), ("occlusion", 1.0, {"jitter_s": 0.0})):
        r = RS.run_job((scen, param, "shield_brake", 0, 2.0, kw, RS.CONFIGS))
        assert r["error"] is None, r["error"]
        assert r["scenario"] == scen and r["config"] == "shield_brake" and r["mode"] == "shield_brake"
        for k in ("task_done", "min_person_distance_m", "person_contact_ticks", "physical_house_contacts", "spilled", "stall_s", "sim_s"):
            assert k in r
        assert 1.9 <= r["sim_s"] <= 2.1
        json.dumps(r, default=str)
    assert "first_visible_t" in r
