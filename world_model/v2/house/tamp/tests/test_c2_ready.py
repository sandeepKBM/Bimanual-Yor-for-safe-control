"""opt-in C2-ready ranking + lateral body-offset candidates in segment mode (tamp/segments.py, gatekeeper/ready.py).  SIMULATION ONLY: no physics stepping,
no rendering, no network.  Placements below were found by probe scans on 10-22 (state = segment k=2 of the no-people route at 0.3 m/s)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from world_model.v2.house.tamp import segments as SEG
from world_model.v2.house.tamp import streams as ST
from world_model.v2.house.tamp import world as W


@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


@pytest.fixture(scope="module")
def route(world):
    pts = tuple(map(tuple, world.routes[0][1]))
    path = SEG.Path(pts)
    s_stop = path.total - ST.Streams(world, cert_mode="segments").end_skip
    segs, why = SEG.plan_route_segments(path, False, 0.3, world.gk, lambda pose, s, t: W.runtime_belief_at(world, pose, False, None, people=False), s_stop)
    assert segs is not None, why
    return path, s_stop, segs


def _with_person(world, xy, v=(0.0, 0.0)):
    b = world.bel0.with_(people=(np.asarray(xy, float),), people_vel=(np.asarray(v, float),), people_e_p=(), people_e_v=(), people_stale=())
    return dataclasses.replace(world, bel_people=b)


def _state(route, world, k, a, d):
    """segment k's start state and a person a m ahead of / d m left of the route point there."""
    path, s_stop, segs = route
    sg = segs[k]
    p0, h = path.point(sg.s_start), path.heading(sg.s_start)
    xy = p0 + a * np.array([math.cos(h), math.sin(h)]) + d * np.array([-math.sin(h), math.cos(h)])
    w = _with_person(world, xy)
    bel = W.runtime_belief_at(w, sg.pose_start, False, None, people=True)
    return sg, bel


def _ncs(route, sg, bel, gk, **kw):
    path, s_stop, _ = route
    return SEG.next_certified_segment(path, False, 0.3, sg.pose_start, sg.V[0], sg.s_start, bel, gk, s_stop, **kw)


def test_defaults_are_off_and_bit_identical_to_explicit_defaults(world, route):
    gk = world.gk
    assert gk.c2_ready == "off" and gk.seg_offsets == ()
    gk2 = dataclasses.replace(gk, c2_ready="off", seg_offsets=())
    for k, a, d in ((2, 0.5, 0.6), (2, 1.3, 0.2), (4, 2.5, 1.5), (7, 0.9, 0.5)):
        sg, bel = _state(route, world, k, a, d)
        A, B = _ncs(route, sg, bel, gk), _ncs(route, sg, bel, gk2)
        assert (A[0] is None) == (B[0] is None) and A[5] == B[5]
        if A[0] is not None:
            np.testing.assert_array_equal(A[1]["V"], B[1]["V"])
            assert A[1]["ready"] is None and A[1]["offset"] == 0.0 and A[4] == B[4]
    # plan_route_segments: offsets stay 0, ready stays -1
    assert all(sg.offset == 0.0 and sg.ready == -1 for sg in route[2])


def test_no_people_rank_on_equals_rank_off_and_all_ready(world, route):
    path, s_stop, segs = route
    gk = dataclasses.replace(world.gk, c2_ready="rank")
    on, why = SEG.plan_route_segments(path, False, 0.3, gk, lambda pose, s, t: W.runtime_belief_at(world, pose, False, None, people=False), s_stop)
    assert on is not None, why
    assert len(on) == len(segs)
    for a, b in zip(on, segs):
        np.testing.assert_array_equal(a.V, b.V)
        assert a.T_S == b.T_S and a.s_end == b.s_end
    assert all(sg.ready == 2 for sg in on)                     # every candidate fully ready (final segment: by definition)


def test_rank_prefers_a_ready_shorter_commitment_near_a_person(world, route):
    """person 1.3 m ahead, 0.2 m left of the route at segment 2: rank off commits T_S=0.3 whose rest pose has NO certified next move; rank on takes T_S=0.1
    (ready under both hypotheses).  Ranking never admits what check1+check2 reject (T_S=0.1 is itself admitted without ranking)."""
    sg, bel = _state(route, world, 2, 1.3, 0.2)
    gk_on = dataclasses.replace(world.gk, c2_ready="rank")
    off = _ncs(route, sg, bel, world.gk)
    on = _ncs(route, sg, bel, gk_on)
    assert off[0] is not None and on[0] is not None
    assert off[4] == 0.3 and on[4] == 0.1 and on[1]["ready"] == on[1]["ready_n"] == 2
    path, s_stop, _ = route
    from world_model.v2.house.gatekeeper.ready import ready_score
    sc_off = ready_score(off[0], bel, gk_on, SEG._ready_moves(path, False, off[1]["s_end"], s_stop, gk_on))[0]
    assert sc_off < on[1]["ready"]
    # the ranked candidate is one the plain loop also admits
    assert SEG.check_segment(on[0], bel, world.gk)[0]


@pytest.mark.parametrize("k,a,d,sign", [(2, 0.9, 0.5, -1), (2, 0.5, -1.1, +1), (4, 1.0, -0.8, +1), (7, 1.0, 0.4, -1)])
def test_offsets_pick_the_side_away_from_the_person(world, route, k, a, d, sign):
    """offset 0 fails for every T_S; with seg_offsets the segment shifts AWAY from the person (left normal +: person on the left -> negative offset)."""
    sg, bel = _state(route, world, k, a, d)
    assert _ncs(route, sg, bel, world.gk)[0] is None
    gk = dataclasses.replace(world.gk, seg_offsets=(0.15, -0.15, 0.3, -0.3))
    tr, res, m, row, T, why = _ncs(route, sg, bel, gk)
    assert tr is not None, why
    assert np.sign(res["offset"]) == sign and SEG.check_segment(tr, bel, gk)[0]


def test_failure_reason_format_unchanged_without_offsets(world, route):
    sg, bel = _state(route, world, 2, 0.5, 0.0)
    r = _ncs(route, sg, bel, world.gk)
    assert r[0] is None and r[5].startswith("T_S=1.0: ") and "off=" not in r[5]
    r2 = _ncs(route, sg, bel, dataclasses.replace(world.gk, seg_offsets=(0.15, -0.15)))
    if r2[0] is None:
        assert r2[5].startswith("off=0: T_S=1.0: ") and "off=0.15: T_S=" in r2[5] and "off=-0.15: T_S=" in r2[5]


def test_constant_offset_is_tracked_and_tapers_to_the_route_end(world, route):
    path, s_stop, _ = route
    gk, off = world.gk, 0.2
    bel = W.runtime_belief_at(world, SEG.start_pose(path, False), False, None, people=False)
    pose, v, s = SEG.start_pose(path, False), np.zeros(3), 0.0
    lat = []
    for _ in range(400):
        res = SEG.make_segment(path, False, 0.3, pose, v, s, bel, gk, 1.0, s_stop, offset=off)
        assert res is not None and res["offset"] == off
        # walk the nominal part to log the lateral error vs the route
        P = Traj_poses(res, pose, gk)
        for q in P:
            sq = path.project(q[:2], s)
            h = path.heading(sq)
            lat.append((sq, (q[0] - path.point(sq)[0]) * -math.sin(h) + (q[1] - path.point(sq)[1]) * math.cos(h)))
        pose, v, s = res["end_pose"], res["end_v"], res["s_end"]
        if res["final"]:
            break
    else:
        pytest.fail("did not finish")
    lat = np.array(lat)
    mid = lat[(lat[:, 0] > 1.2) & (lat[:, 0] < s_stop - gk.seg_offset_taper - 0.3)]
    assert len(mid) > 20
    straight = mid[:, 1]
    assert abs(float(np.median(straight)) - off) < 0.05, np.median(straight)
    h_end = path.heading(s)
    e_end = (pose[0] - path.point(s)[0]) * -math.sin(h_end) + (pose[1] - path.point(s)[1]) * math.cos(h_end)
    assert abs(e_end) < 0.03 and abs(s - s_stop) < 0.05


def Traj_poses(res, pose0, gk):
    from world_model.v2.house.gatekeeper.traj import Traj
    n = int(round(res["t_sw"] / gk.h))
    P = Traj(res["V"][:n + 1], gk.h).poses()
    c, s = math.cos(pose0[2]), math.sin(pose0[2])
    return np.stack([pose0[0] + c * P[:, 0] - s * P[:, 1], pose0[1] + s * P[:, 0] + c * P[:, 1]], 1)


def test_ready_moves_is_depth_one(world, route, monkeypatch):
    path, s_stop, segs = route
    gk = dataclasses.replace(world.gk, c2_ready="rank", seg_offsets=(0.15, -0.15))
    calls = []
    real = SEG.next_certified_segment

    def spy(*a, **kw):
        calls.append((kw.get("ready"), tuple(a[7].seg_offsets), kw.get("offset", 0.0)))
        return real(*a, **kw)
    monkeypatch.setattr(SEG, "next_certified_segment", spy)
    sg = segs[2]
    bel = W.runtime_belief_at(world, sg.pose_start, False, None, people=False)
    mv = SEG._ready_moves(path, False, sg.s_end, s_stop, gk, offsets=(0.15, 0.0))
    out = mv(bel)
    assert out is not None and out.startswith("route v=") and "off=0.15" in out
    assert calls and all(r is False and so == () for r, so, _ in calls)
    assert {o for _, _, o in calls} <= {0.15, 0.0}
    # at the goal: no inner call
    calls.clear()
    assert SEG._ready_moves(path, False, s_stop, s_stop, gk)(bel.with_(pose_wall=np.r_[path.point(s_stop), path.heading(s_stop)])) == "at_goal" and not calls
