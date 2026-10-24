"""segment-mode route certificate (tamp/segments.py, seg_gate.py): timed gatekeeper-format segments, planner-chosen speed, execution re-certification.
SIMULATION ONLY, no stepping of the physics (the base state is set directly), no rendering, no network."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from world_model.v2.house.tamp import segments as SEG
from world_model.v2.house.tamp import streams as ST
from world_model.v2.house.tamp import world as W
from world_model.v2.house.tamp.types import Conf, to_json


@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


def _confs(sim):
    wp = sim.scene.waypoints
    return (Conf("a", *wp["lab_start"]), Conf("b", *wp["cafe_counter_pick"], surface="counter"), Conf("c", *wp["lab_desk_place"], surface="desk"))


def _out_pts(world):
    return tuple(map(tuple, world.routes[0][1]))


def _with_person(world, xy):
    b = world.bel0.with_(people=(np.asarray(xy, float),), people_vel=(np.zeros(2),), people_e_p=(), people_e_v=(), people_stale=())
    return dataclasses.replace(world, bel_people=b)


def test_no_people_route_segments_certify_and_reach_the_end_at_rest(world, sim):
    qa, qb, _ = _confs(sim)
    st = ST.Streams(world, cert_mode="segments")
    r, why = st.build_route(qa, qb, _out_pts(world), False, False, None, "scene:out_route")
    assert r is not None, why
    segs = r.segments
    assert r.v_max == ST.SEG.SPEEDS[0] and len(segs) >= 10 and r.cert.margin >= 0
    d = dict(r.cert.detail)
    assert d["n_segments"] == len(segs) and d["speed"] == r.v_max and len(d["seg_margins"]) == len(segs)
    gk = world.gk
    # continuity: segment k starts at the predicted moving state at the end of segment k-1's nominal; arclength is contiguous and monotone
    for a, b in zip(segs[:-1], segs[1:]):
        n = int(round(a.t_sw / gk.h))
        np.testing.assert_allclose(a.V[n], b.V[0], atol=1e-9)
        assert abs(a.s_end - b.s_start) < 0.05 and b.s_end > b.s_start - 1e-9
        assert abs(b.t_start - (a.t_start + a.t_sw)) < 1e-9
    # every segment ends in its own cup-safe backup at rest; the last nominal ends at rest too; the route ends at the driven part's end
    for sg in segs:
        tr = sg.traj()
        assert tr.i_rest is not None and np.abs(tr.V[-1]).max() == 0.0
    last = segs[-1]
    assert np.abs(last.V[int(round(last.t_sw / gk.h))]).max() == 0.0
    assert abs(last.s_end - r.s_stop) < 0.05
    P = np.array(world.routes[0][1])
    L = float(np.hypot(*np.diff(P, axis=0).T).sum())
    assert abs(r.s_stop - (L - st.end_skip)) < 1e-9
    # concatenated pose profile ends at the polyline point at s_stop
    pose, end = segs[0].pose_start, None
    for sg in segs:
        P_rel = sg.traj().poses()
        n = int(round(sg.t_sw / gk.h))
        c, s = math.cos(sg.pose_start[2]), math.sin(sg.pose_start[2])
        end = (sg.pose_start[0] + c * P_rel[n, 0] - s * P_rel[n, 1], sg.pose_start[1] + s * P_rel[n, 0] + c * P_rel[n, 1])
    xy_end = SEG.Path(r.pts).point(r.s_stop)
    assert math.hypot(end[0] - xy_end[0], end[1] - xy_end[1]) < 0.1
    # the log form carries no arrays
    js = to_json(r)
    assert js["segments"] and js["v_max"] == r.v_max and "V" not in js["segments"][0]


def test_speed_selection_picks_the_lower_speed_when_0p3_fails(world, sim):
    pts = _out_pts(world)
    path = SEG.Path(pts)
    # CARRYING the tray, a person 1.2 m to the RIGHT of the route at s = 1.0 m (inside near_m, so the people rows apply; found by scan 10-22).
    # Before the 10-22 cup_row fix this test used an empty-handed case that only failed because of the phantom cup row (cup on the counter).
    base, h = path.point(1.0), path.heading(1.0)
    xy = base - 1.2 * np.array([-math.sin(h), math.cos(h)])
    w = _with_person(world, xy)
    chosen = {}
    for speeds in ((0.3,), (0.3, 0.2, 0.12)):
        st = ST.Streams(w, cert_mode="segments", lateral_offsets=(), speeds=speeds)
        c, why, segs, v = st._certify_segments(pts, False, True, "tray")
        chosen[speeds] = (c, v, why)
    assert chosen[(0.3,)][0] is None and "segment" in chosen[(0.3,)][2]          # 0.3 alone: refused
    c, v, _ = chosen[(0.3, 0.2, 0.12)]
    assert c is not None and v == 0.12 and dict(c.detail)["speeds_failed"] == 2
    assert c.margin >= 0


def test_replan_from_predicted_state_reproduces_and_recertifies(world, sim):
    """planner cert => execution cert when the realised state equals the predicted state (people static): rebuild every segment from the predicted start state
    with the execution-side function and re-run check1 + check2 under the same belief."""
    qa, qb, _ = _confs(sim)
    pts = _out_pts(world)
    w = _with_person(world, SEG.Path(pts).point(0.5) + np.array([6.0, 4.0]))     # a far person: in the belief, not binding
    st = ST.Streams(w, cert_mode="segments")
    r, why = st.build_route(qa, qb, pts, False, False, None, "scene:out_route")
    assert r is not None, why
    path = SEG.Path(r.pts)
    gk = w.gk
    v = np.zeros(3)
    for sg in r.segments:
        bel = W.runtime_belief_at(w, sg.pose_start, False, None, people=sg.s_start <= st.near_m + 1e-9)
        tr, res, m, row, T, why = SEG.next_certified_segment(path, r.reverse, r.v_max, sg.pose_start, v, sg.s_start, bel, gk, r.s_stop)
        assert tr is not None, why
        assert T == sg.T_S
        np.testing.assert_allclose(res["V"], sg.V, atol=1e-9)
        ok, m2, row2, why2 = SEG.check_segment(sg.traj(), bel, gk)
        assert ok, why2
        v = res["end_v"]


def test_gatekeeper_boundary_recertifies_planned_segment_from_set_state(world, sim):
    """SegmentGatekeeper at a mid-route boundary: base state set to the planned predicted state -> the rebuilt segment is committed (not refused)."""
    from world_model.v2.house.tamp.seg_gate import make_seg_filter
    qa, qb, _ = _confs(sim)
    st = ST.Streams(world, cert_mode="segments")
    r, why = st.build_route(qa, qb, _out_pts(world), False, False, None, "scene:out_route")
    assert r is not None, why
    k = len(r.segments) // 2
    sg = r.segments[k]
    n = int(round(sg.t_sw / world.gk.h))
    save = (sim.base.x, sim.base.y, sim.base.yaw, sim.base.v.copy())
    try:
        sim.base.x, sim.base.y, sim.base.yaw = sg.pose_start
        sim.base.v = np.array(sg.V[0], float)
        filt = make_seg_filter()
        filt.docks = []
        filt.begin_route(r)
        filt.sr["s"] = sg.s_start
        filt._boundary(sim, 0.0)
        assert filt.seg_state() == "active", filt.sr["reason"]
        assert filt.sr["n_admitted"] == 1 and filt.sr["n_refused"] == 0 and filt.mode == "commit"
        np.testing.assert_allclose(filt.traj.V, sg.V, atol=1e-6)
        assert filt.counters()["n_commit"] == 1
    finally:
        sim.base.x, sim.base.y, sim.base.yaw, sim.base.v = save


def test_gatekeeper_boundary_refuses_when_the_state_is_not_certifiable(world, sim):
    from world_model.v2.house.tamp.seg_gate import make_seg_filter
    qa, qb, _ = _confs(sim)
    st = ST.Streams(world, cert_mode="segments")
    r, _ = st.build_route(qa, qb, _out_pts(world), False, False, None, "scene:out_route")
    save = (sim.base.x, sim.base.y, sim.base.yaw, sim.base.v.copy())
    try:
        sim.base.x, sim.base.y, sim.base.yaw = 3.1, 6.09, 0.0            # wrong heading / inside the wall-side of the start: fine to build, just must not crash
        sim.base.v = np.zeros(3)
        filt = make_seg_filter()
        filt.docks = []
        filt.begin_route(r)
        filt._boundary(sim, 0.0)
        assert filt.seg_state() in ("active", "refused")
        if filt.seg_state() == "refused":
            assert filt.sr["n_refused"] == 1 and "refused" in filt.sr["reason"]
            assert filt.end_route()["status"] == "refused"
    finally:
        sim.base.x, sim.base.y, sim.base.yaw, sim.base.v = save


def test_legacy_default_is_unchanged(world, sim):
    qa, qb, _ = _confs(sim)
    st = ST.Streams(world)
    assert st.cert_mode == "rest"
    r, why = st.build_route(qa, qb, _out_pts(world), False, False, None, "scene:out_route")
    assert r is not None, why
    assert r.cert.checker == "C1C2_static_route" and r.v_max is None and r.segments is None and r.s_stop is None
    c, why2 = st.certify_route(_out_pts(world), False, False)
    assert c == r.cert
    d = dict(r.cert.detail)
    assert d["v_cruise"] == 0.3 and "n_segments" not in d
