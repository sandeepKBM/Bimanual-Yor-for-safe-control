"""tests for tamp/rh/mppi_prop.py.  SIMULATION ONLY, no physics stepping.

CPU part (always): GoalField, convert_twists (cup/hw rows by construction, ends at rest), candidates_from_twists (ranking, stop last, progress).
GPU part: only if CUDA is available AND MPPI_GPU_TESTS=1 (so a stray pytest never touches a GPU that is not ours; on westeros use CUDA_VISIBLE_DEVICES=4 + the
gpu4 lock, see AGENTS.md):
    python -m pytest -q world_model/v2/house/tamp/rh/tests/test_mppi_prop.py                       # CPU only
    MPPI_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=4 python -m pytest -q ...                                 # + GPU
"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest
import torch

from world_model.v2.house.gatekeeper import check1, check2
from world_model.v2.house.tamp import plan_cert as PC
from world_model.v2.house.tamp import world as W
from world_model.v2.house.tamp.rh import mppi_prop as MP
from world_model.v2.house.tamp.rh.api import RHConfig, State, admit, plan_gk

GPU = torch.cuda.is_available() and os.environ.get("MPPI_GPU_TESTS") == "1"
POSE_SO = (2.14, 3.56, -1.578)
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
    return np.array(sim.scene.waypoints["cafe_counter_pick"], float)


@pytest.fixture(scope="module")
def gfield(world, goal):
    return MP.GoalField(world.bel0.grid, goal[:2])


def _bel(world, pose=POSE_SO):
    return PC.plan_belief(world, pose, True, "tray", [], 0.0)


def _runtime_cup_ok(tr, world, bel):
    g = dataclasses.replace(world.gk, runtime_rows="cup")
    o1, m1, r1, g1 = check1(tr, bel, g)
    o2, m2, r2, g2 = check2(tr, bel, g)
    return o1 and o2, (m1, r1, g1), (m2, r2, g2)


# ------------------------------------------------------------------------------------------------ GoalField
def test_goal_field_zero_at_goal_and_monotone_along_the_scene_route(world, gfield, goal):
    assert gfield.v(goal[:2]) < 0.25                                   # the dock may sit inside the inflated band: the goal snaps to the nearest free cell
    route = np.array(world.routes[0][1], float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])
    pts = np.array([np.interp(np.linspace(0, s[-1], 60), s, route[:, k]) for k in (0, 1)]).T
    v = gfield.v(pts)
    assert np.all(np.isfinite(v)) and np.all(np.diff(v) < 0.08)        # decreases toward the goal (octile slack on the corners)
    assert v[0] > 4.0 and v[0] < 1.2 * s[-1]                            # a bit longer or equal to the polyline, not wildly different
    assert gfield.reachable(route[0])


def test_goal_field_gradient_points_down_the_corridor(world, gfield):
    g = gfield.grad(np.array([2.14, 3.56]))
    assert g[1] > 0.5 and abs(g[0]) < 0.6 * abs(g[1])                   # -grad = toward -y (the cafe counter)


def test_goal_field_inside_walls_pushes_back_into_free_space(world, gfield):
    # a point deep inside a wall: V is larger than at the nearest free cell (out_slope)
    grid = world.bel0.grid
    x = np.array([1.0 - 0.02, 3.5])                                    # corridor wall at x ~ 1.0 (EDT ~ 0 there)
    assert grid.lookup(x[0], x[1]) < 0.05
    assert gfield.v(x) > gfield.v(np.array([1.9, 3.5])) - 0.2


# ------------------------------------------------------------------------------------------------ conversion
SEQS = {
    "straight": lambda H: np.tile([0.3, 0.0, 0.0], (H, 1)),
    "reverse": lambda H: np.tile([-0.3, 0.0, 0.0], (H, 1)),
    "lateral": lambda H: np.tile([0.0, 0.3, 0.0], (H, 1)),
    "arc": lambda H: np.tile([0.3, 0.0, 0.5], (H, 1)),
    "diag_spin": lambda H: np.tile([0.3, 0.3, -0.5], (H, 1)),
    "zigzag": lambda H: np.array([[0.3, 0.3 * (-1) ** (i // 3), 0.0] for i in range(H)]),
}


@pytest.mark.parametrize("name", list(SEQS))
@pytest.mark.parametrize("v0", [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.3)])
def test_converted_traj_is_cup_hw_feasible_ends_at_rest_and_is_cut_at_the_horizon(world, name, v0):
    bel, gk = _bel(world), world.gk
    tr = MP.convert_twists(v0, SEQS[name](15), 0.1, 30, bel, gk, t0=1.25, v_trans_max=0.3)
    assert tr.i_rest is not None and np.abs(tr.V[-1]).max() == 0.0     # ends at rest
    assert abs(tr.t_sw - 30 * gk.h) < 1e-12 and tr.t0 == 1.25 and abs(tr.T_S - tr.t_sw) < 1e-12
    np.testing.assert_allclose(tr.V[0], v0)
    assert np.hypot(tr.V[:, 0], tr.V[:, 1]).max() <= 0.3 + 1e-9        # speed cap (norm) of the sampled twists
    ok, c1, c2 = _runtime_cup_ok(tr, world, bel)
    assert ok, (c1, c2)                                                # runtime_rows="cup": hw + cup rows of check1, no_rest / rest_speed of check2
    assert c1[2]["cup"] >= 0.0 and c1[2]["hw_slew"] >= -1e-9
    # the full plan-time certificate (walls + cup + hw) on a person-free corridor pose also holds for the straight run
    if name == "straight" and v0 == (0.0, 0.0, 0.0):
        ok2, m, why, mg = PC.certify_plan(tr, world, plan_gk(gk, RHConfig()), people=[], t0=0.0, pose0=(2.14, 6.0, -math.pi / 2))
        assert ok2, (m, why, mg)


def test_pure_backup_is_the_stop_candidate_and_is_rest_safe(world):
    bel, gk = _bel(world), world.gk
    st = State(pose=POSE_SO, v=(0.2, 0.0, 0.0), t=0.0, people=[])
    gf = MP.GoalField(world.bel0.grid, np.array([2.135, 2.0]))
    cands = MP.candidates_from_twists([SEQS["straight"](15)], st, bel, gk, gf, 0.1, 1.5, v_trans_max=0.3)
    stop = cands[-1]
    assert stop.source == "mppi:stop" and stop.traj.t_sw <= 0 and stop.progress == 0.0
    assert stop.traj.i_rest is not None and np.abs(stop.traj.V[-1]).max() == 0.0
    assert _runtime_cup_ok(stop.traj, world, bel)[0]


def test_candidates_are_ranked_by_end_distance_progress_matches_and_stop_is_last(world, gfield, goal):
    bel, gk = _bel(world), world.gk
    st = State(pose=POSE_SO, v=(0.0, 0.0, 0.0), t=3.0, people=[])
    Us = [SEQS["lateral"](15), SEQS["straight"](15), SEQS["reverse"](15), np.zeros((15, 3)), SEQS["arc"](15)]
    cands = MP.candidates_from_twists(Us, st, bel, gk, gfield, 0.1, 1.5)
    assert len(cands) == 5 and cands[-1].source == "mppi:stop"          # the all-zero sequence is dropped (== stop now)
    v_end = [c.meta["v_end"] for c in cands[:-1]]
    assert v_end == sorted(v_end)
    v0 = float(gfield.v(np.array(POSE_SO[:2])))
    for c in cands:
        assert abs(c.progress - (v0 - c.meta["v_end"])) < 1e-9 and c.traj.t0 == 3.0
    best = cands[0]
    assert best.progress > 0.2 and best.source in ("mppi:s1", "mppi:s4")   # straight (down the corridor) beats lateral and reverse
    assert cands[-2].progress < 0                                       # reverse loses ground


def test_diverse_picks_distinct_sequences():
    base = np.zeros((6, 15, 3)); base[1] += 0.001; base[2] += 0.2; base[3, :, 0] = 0.3; base[4, :, 1] = -0.3; base[5] += 0.2005
    keep = MP.diverse(base, 4, 0.08, [0.3, 0.3, 0.5])
    assert keep[0] == 0 and 1 not in keep and 5 not in keep and len(keep) == 4


def test_admit_certifies_a_converted_candidate_in_open_corridor(world, goal):
    st = State(pose=(2.14, 6.0, -math.pi / 2), v=(0.0, 0.0, 0.0), t=0.0, people=[])
    gf = MP.GoalField(world.bel0.grid, goal[:2])
    bel = _bel(world, st.pose)
    cands = MP.candidates_from_twists([SEQS["straight"](15)], st, bel, world.gk, gf, 0.1, 1.5)
    ch, log = admit(cands, st, world, world.gk, RHConfig())
    assert ch is not None and ch.source == "mppi:mean" and log[0][1] is True


# ------------------------------------------------------------------------------------------------ GPU (opt in)
@pytest.mark.skipif(not GPU, reason="needs CUDA and MPPI_GPU_TESTS=1 (westeros: CUDA_VISIBLE_DEVICES=4 under the gpu4 lock)")
def test_gpu_proposer_standoff_and_warm_start(world, goal):
    cfg = RHConfig()
    prop = MP.MPPIProposer(world, cfg, K=1024, goal_xy=goal[:2])
    st = State(pose=POSE_SO, v=(0.0, 0.0, 0.0), t=0.0, people=[WALKER, P_STAND])
    c1 = prop.propose(st, world, goal, None, cfg)
    assert c1[-1].source == "mppi:stop" and 2 <= len(c1) <= 12
    bel = _bel(world)
    for c in c1:
        assert c.traj.t0 == 0.0 and np.abs(c.traj.V[-1]).max() == 0.0 and c.traj.t_sw <= cfg.horizon_s + 1e-9
        assert _runtime_cup_ok(c.traj, world, bel)[0]
    v_end = [c.meta["v_end"] for c in c1[:-1]]
    assert v_end == sorted(v_end)
    s1 = dict(prop.last_stats)
    assert s1["plan_ms"] > 0 and s1["convert_ms"] > 0
    st2 = State(pose=POSE_SO, v=(0.0, 0.0, 0.0), t=0.5, people=[WALKER, P_STAND])
    c2 = prop.propose(st2, world, goal, None, cfg)                       # warm start: shifted mean
    assert prop.last_stats["elapsed_shift"] == pytest.approx(0.5)
    assert c2[-1].source == "mppi:stop"
    st3 = State(pose=POSE_SO, v=(0.0, 0.0, 0.0), t=0.1, people=[WALKER, P_STAND])      # time went backwards: reset, no shift
    prop.propose(st3, world, goal, None, cfg)
    assert prop.last_stats["elapsed_shift"] == 0.0
