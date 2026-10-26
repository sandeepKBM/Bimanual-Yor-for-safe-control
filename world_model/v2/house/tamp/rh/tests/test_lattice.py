"""LatticeProposer + the shared receding-horizon loop helpers (tamp/rh).  SIMULATION ONLY, no rendering, no network.
    python -m pytest -q world_model/v2/house/tamp/rh/tests/test_lattice.py            (the short-episode test needs RH_SLOW=1: it builds two sims and ticks physics)"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest

from world_model.v2.house.gatekeeper.checks import check1
from world_model.v2.house.tamp import world as W
from world_model.v2.house.tamp.rh import api
from world_model.v2.house.tamp.rh import loop as LP
from world_model.v2.house.tamp.rh.lattice import LatticeProposer, cost_field
from world_model.v2.house.teleop import geom as GM

POSE0 = (2.14, 3.56, -1.578)                                    # sweep_1022c2 ours_r_s1 stand-off: robot at rest, remaining route = down the corridor
WALKER = (np.array([2.6, 3.8]), np.zeros(2))
P_STAND = (np.array([2.13449 - 1.32, 3.5]), np.zeros(2))


@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


@pytest.fixture(scope="module")
def goal(sim):
    return np.array(sim.scene.waypoints["cafe_counter_pick"][:2], float)


@pytest.fixture(scope="module")
def prop():
    return LatticeProposer()


def _state(pose=POSE0, v=(0.0, 0.0, 0.0), people=(WALKER, P_STAND)):
    return api.State(tuple(pose), tuple(v), 0.0, list(people))


def _cup_hw_ok(tr, world, pose):
    gk = dataclasses.replace(world.gk, runtime_rows="cup")
    bel = world.belief_at(pose, True, "tray")
    return check1(tr, bel, gk)


# ------------------------------------------------------------------------------------------------ cost-to-go field
def test_cost_field_exact_at_goal_and_follows_the_corridor(world, goal):
    f = cost_field(world.bel0.grid, goal)
    assert abs(f.v(*goal)) < 1e-9
    assert abs(f.v(2.134, 3.5) - 1.5) < 0.05                                    # straight corridor: distance = remaining y
    assert abs(f.v(3.1, 6.09) - 4.49) < 0.15                                    # route length 4.45 m
    d = f.guide(2.14, 3.56)
    assert d[1] < -0.95 and abs(d[0]) < 0.3                                     # down the corridor
    assert f.v(2.134, 3.0) < f.v(2.134, 3.5) < f.v(2.134, 4.0)                  # monotone along the route


# ------------------------------------------------------------------------------------------------ construction invariants
@pytest.mark.parametrize("v0", [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, -0.15, 0.2)])
def test_every_candidate_ends_at_rest_and_passes_cup_and_hw_rows(world, goal, prop, v0):
    cfg = api.RHConfig()
    st = _state(v=v0)
    cands = prop.propose(st, world, goal, None, cfg)
    assert 2 <= len(cands) <= prop.n_out + 1
    for c in cands:
        tr = c.traj
        assert tr.i_rest is not None and np.all(tr.V[-1] == 0.0), c.source              # ends at rest
        assert np.allclose(tr.V[0], v0), c.source                                        # starts at the state's twist
        assert tr.t_sw <= cfg.horizon_s + 1e-9 and abs(tr.t0 - st.t) < 1e-12
        nom = tr.V[: int(round(tr.t_sw / tr.h)) + 1]
        assert np.hypot(nom[:, 0], nom[:, 1]).max() <= cfg.v_max + 1e-6, c.source        # never above v_max
        ok, m, why, mg = _cup_hw_ok(tr, world, st.pose)
        assert ok, (c.source, m, why, mg)
    assert cands[-1].source == "lattice:stop" and cands[-1].traj.t_sw <= 0.0 and cands[-1].meta["depth"] == 0   # "stop now" LAST
    assert all(c.traj.t_sw > 0 for c in cands[:-1])


def test_stop_now_is_ranked_first_only_when_it_ends_at_the_goal(world, goal, prop):
    cands = prop.propose(_state((goal[0] + 0.01, goal[1] - 0.01, -1.57), people=()), world, goal, None, api.RHConfig())
    assert cands[0].source == "lattice:stop" and cands[0].traj.t_sw <= 0.0
    assert sum(c.source == "lattice:stop" for c in cands) == 1
    far = prop.propose(_state(people=()), world, goal, None, api.RHConfig())
    assert far[-1].source == "lattice:stop" and far[0].source != "lattice:stop"


def test_horizon_bounds_the_depth_and_v_max_is_respected(world, goal):
    p = LatticeProposer()
    for hz in (0.5, 1.0):
        cands = p.propose(_state(), world, goal, None, api.RHConfig(horizon_s=hz))
        assert max(c.traj.t_sw for c in cands) <= hz + 1e-9
    cands = p.propose(_state(), world, goal, None, api.RHConfig(v_max=0.15))
    for c in cands:
        nom = c.traj.V[: int(round(c.traj.t_sw / c.traj.h)) + 1]
        assert np.hypot(nom[:, 0], nom[:, 1]).max() <= 0.15 + 1e-6


def test_candidates_are_ranked_by_cost_at_the_end_of_the_nominal(world, goal, prop):
    cands = prop.propose(_state(), world, goal, None, api.RHConfig())
    keys = [c.meta["rank_cost"] for c in cands[:-1]]
    assert keys == sorted(keys)
    assert cands[0].progress > 0.3 and cands[0].progress >= max(c.progress for c in cands[:-1]) - 0.1
    assert abs(cands[0].progress - (cost_field(world.bel0.grid, goal).v(*POSE0[:2]) - cands[0].meta["cost_end"])) < 1e-9


def test_proposed_footprints_do_not_cross_a_wall(world, goal, prop):
    grid = world.bel0.grid
    pose = (2.14, 4.9, -1.578)                                                  # in the doorway region: walls close on both sides
    cands = prop.propose(_state(pose, people=()), world, goal, None, api.RHConfig())
    pts = GM.boundary_pts(step=0.05)
    for c in cands:
        P = c.traj.poses()
        c_, s_ = math.cos(pose[2]), math.sin(pose[2])
        for p in P[:: 2]:
            x = pose[0] + c_ * p[0] - s_ * p[1]
            y = pose[1] + s_ * p[0] + c_ * p[1]
            yaw = pose[2] + p[2]
            w = pts @ GM.rot(yaw).T + np.array([x, y])
            assert grid.lookup(w[:, 0], w[:, 1]).min() > 0.0, c.source


# ------------------------------------------------------------------------------------------------ admission (the same api.admit for every planner)
def test_standoff_snapshot_admits_a_candidate_that_moves_away_along_the_route(world, goal, prop):
    cfg = api.RHConfig()
    st = _state()
    cands = prop.propose(st, world, goal, None, cfg)
    adm, log = api.admit(cands, st, world, world.gk, cfg)
    assert adm is not None, log[:5]
    assert adm.source != "lattice:stop" and adm.progress > 0.0
    P = adm.traj.poses()
    n = int(round(adm.traj.t_sw / adm.traj.h))
    c, s = math.cos(POSE0[2]), math.sin(POSE0[2])
    end = (POSE0[0] + c * P[n, 0] - s * P[n, 1], POSE0[1] + s * P[n, 0] + c * P[n, 1])
    assert POSE0[1] - end[1] > 0.1 and abs(end[0] - POSE0[0]) < 0.3             # moved down the corridor (away from the walker), not sideways
    assert [r[0] for r in log].index(adm.source) == len(log) - 1                # admit returns the FIRST that certifies


def test_candidate_driving_into_a_person_in_its_path_is_rejected(world, goal, prop):
    cfg = api.RHConfig()
    blocker = (np.array([2.14, 2.75]), np.zeros(2))                             # 0.2 m beyond the footprint front (y = 2.98), centred on the corridor
    st = _state(people=(blocker,))
    cands = prop.propose(st, world, goal, None, cfg)
    fwd = cands[0]                                                              # the best-ranked candidate drives down the corridor, straight at the person
    assert fwd.progress > 0.3 and fwd.meta["depth"] == 3
    adm, log = api.admit([fwd], st, world, world.gk, cfg)
    assert adm is None and not log[0][1] and log[0][2] < 0.0
    adm, log = api.admit(cands, st, world, world.gk, cfg)                        # whatever is admitted never gets closer to the person (the robot starts 0.23 m from its centre,
    d0 = GM.rect_dist_point(blocker[0], POSE0)[0]                               # inside the 0.3 m keep-out: only the separating-motion exit rule lets it move at all)
    if adm is not None:
        P = adm.traj.poses()
        c, s = math.cos(POSE0[2]), math.sin(POSE0[2])
        for p in P:
            pose = (POSE0[0] + c * p[0] - s * p[1], POSE0[1] + s * p[0] + c * p[1], POSE0[2] + p[2])
            assert GM.rect_dist_point(blocker[0], pose)[0] >= d0 - 1e-6


def test_admit_at_zero_delay_is_api_admit_and_a_delay_only_tightens(world, goal, prop):
    cfg = api.RHConfig()
    st = _state()
    cands = prop.propose(st, world, goal, None, cfg)
    a0, l0 = api.admit(cands, st, world, world.gk, cfg)
    a1, l1 = LP.admit_at(cands, st, world, world.gk, cfg, True, 0.0)
    assert a0.source == a1.source and l0 == l1
    a2, l2 = LP.admit_at(cands, st, world, world.gk, cfg, True, 0.3)
    m0 = {r[0]: r[2] for r in l0}
    for r in l2:
        if r[0] in m0:
            assert r[2] <= m0[r[0]] + 1e-9                                        # the observation is older: the person disc is bigger -> margins can only shrink


def test_planner_time_is_reported(world, goal):
    p = LatticeProposer()
    p.propose(_state(), world, goal, None, api.RHConfig())
    assert p.last_stats["n_out"] >= 2 and p.last_stats["total_ms"] > 0.0


# ------------------------------------------------------------------------------------------------ episode loop (slow: two sims + physics)
@pytest.mark.skipif(os.environ.get("RH_SLOW") != "1", reason="set RH_SLOW=1 (builds two TeleopSims and steps the physics)")
def test_short_episode_runs_end_to_end(tmp_path):
    out = LP.rh_episode(LatticeProposer(), 0, 1.61, dict(delay_s=0.01, speed=0.571, x0=4.34), api.RHConfig(), max_s=4.0, out=tmp_path, verbose=False)
    assert out["task"] == "base_drive_to_counter" and out["planner"] == "lattice"
    assert out["n_replans"] >= 1 and out["n_falsified"] == 0 and not out["cup_lost"] and out["person_contact_ticks"] == 0
    assert (tmp_path / "lattice_s0.json").exists()
