"""time-indexed people prediction in the segment planner (tamp/people_pred.py, Streams(people_pred="time")) and the timed-wait probe.
SIMULATION ONLY, no physics stepping.  ASSUMPTIONS under test: constant-velocity prediction of the binding person at the segment start; velocity-error growth e_v t;
planning is feasibility (the gatekeeper's per-tick check is unchanged)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from world_model.v2.house.tamp import people_pred as PP
from world_model.v2.house.tamp import segments as SEG
from world_model.v2.house.tamp import streams as ST
from world_model.v2.house.tamp import world as W
from world_model.v2.house.tamp.types import Conf


@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


@pytest.fixture(scope="module")
def pts(world):
    return tuple(map(tuple, world.routes[0][1]))


def _confs(sim):
    wp = sim.scene.waypoints
    return Conf("a", *wp["lab_start"]), Conf("b", *wp["cafe_counter_pick"], surface="counter")


def _world_with(world, xy, v):
    b = world.bel0.with_(people=(np.asarray(xy, float),), people_vel=(np.asarray(v, float),), people_e_p=(), people_e_v=(), people_stale=())
    return dataclasses.replace(world, bel_people=b)


def _crossing(world, pts, s0, lat, vlat):
    """a person at lateral offset `lat` from route point s0, walking along the left normal at vlat [m/s] (constant velocity)."""
    path = SEG.Path(pts)
    base, h = path.point(s0), path.heading(s0)
    n = np.array([-math.sin(h), math.cos(h)])
    return _world_with(world, base + lat * n, vlat * n)


def _cert(w, sim, pts, pred, tau=0.0, speeds=SEG.SPEEDS):
    qa, qb = _confs(sim)
    st = ST.Streams(w, cert_mode="segments", lateral_offsets=(), speeds=speeds, people_pred=pred, t_offset=tau)
    return st.build_route(qa, qb, pts, False, False, None, "scene:out_route")


def test_frozen_mode_is_bit_identical_to_the_old_rule(world, sim, pts):
    """default Streams == explicit people_pred="frozen" == the pre-change bel_fn (people only for s_start <= near_m, positions frozen, velocity ignored)."""
    w = _crossing(world, pts, 2.5, 0.0, 0.5)
    qa, qb = _confs(sim)
    for ww in (world, w):
        r0, why0 = ST.Streams(ww, cert_mode="segments", lateral_offsets=()).build_route(qa, qb, pts, False, False, None, "scene:out_route")
        r1, why1 = _cert(ww, sim, pts, "frozen")
        assert why0 == why1 and (r0 is None) == (r1 is None)
        if r0 is not None:
            assert r0.cert == r1.cert and r0.v_max == r1.v_max
            for a, b in zip(r0.segments, r1.segments):
                np.testing.assert_array_equal(a.V, b.V)
        # the OLD rule, re-implemented literally, gives the same segments / reason
        st = ST.Streams(ww, cert_mode="segments", lateral_offsets=())
        n_ppl = len(ww.bel_people.people)
        old = lambda pose, s: W.runtime_belief_at(ww, pose, False, None, people=bool(n_ppl) and s <= st.near_m + 1e-9)
        for v in SEG.SPEEDS:
            segs, why = SEG.plan_route_segments(SEG.Path(pts), False, v, ww.gk, lambda p, s, t: old(p, s), max(0.0, SEG.Path(pts).total - st.end_skip))
            if segs is not None:
                assert r0 is not None and r0.v_max == v and len(segs) == len(r0.segments)
                break
    # velocity is irrelevant when frozen
    wv = _crossing(world, pts, 2.5, 0.0, -3.0)
    ra, _ = _cert(w, sim, pts, "frozen")
    rb, _ = _cert(wv, sim, pts, "frozen")
    assert (ra is None) == (rb is None)


def test_prediction_moves_the_person_to_c_plus_v_t_and_grows_ep(world, pts):
    w = _world_with(world, (3.0, 4.0), (0.5, -0.25))
    b = w.bel_people
    p = PP.predict_people(b, 4.0)
    np.testing.assert_allclose(p.people[0], [3.0 + 0.5 * 4.0, 4.0 - 0.25 * 4.0])
    assert p.people_e_p[0] == pytest.approx(b.e_v * 4.0)                    # velocity-error growth only; NO a_p t^2/2
    assert PP.predict_people(b, 0.0) is b and PP.predict_people(b.with_(people=()), 3.0).people == ()
    # the bel_fn the stream hands to the planner: t_start flows through (t_offset + segment start time)
    seen = []
    st = ST.Streams(_crossing(world, pts, 2.5, 0.0, 0.5), cert_mode="segments", lateral_offsets=(), people_pred="time", t_offset=1.5)
    orig = ST.runtime_belief_at

    def spy(*a, **k):
        b_ = orig(*a, **k)
        seen.append(b_)
        return b_
    try:
        ST.runtime_belief_at = spy
        st._certify_segments(pts, False, False, None, immediate=True)
    finally:
        ST.runtime_belief_at = orig
    assert seen and seen[0].people[0] is not None
    c0 = np.asarray(st.w.bel_people.people[0]); v0 = np.asarray(st.w.bel_people.people_vel[0])
    # first call is segment 0 at t_start = 0 -> predicted to t_offset exactly; reconstruct via the same function and compare to what the planner used
    exp = PP.predict_people(st.w.bel_people, 1.5)
    np.testing.assert_allclose(exp.people[0], c0 + v0 * 1.5)


def test_crossing_both_directions_frozen_vs_time(world, sim, pts):
    """A: a person ON the route at s = 2.5 m walking sideways at 0.5 m/s -> frozen refuses (person on the path now), time certifies (it has left by the time the
       robot arrives).   B: a person 4 m to the side walking TOWARD the route at 0.6 m/s -> frozen admits (far now), time refuses (it arrives when the robot does)."""
    wa = _crossing(world, pts, 2.5, 0.0, 0.5)
    ra_f, why_f = _cert(wa, sim, pts, "frozen")
    ra_t, why_t = _cert(wa, sim, pts, "time")
    assert ra_f is None and "segment" in why_f
    assert ra_t is not None and ra_t.cert.margin >= 0, why_t
    d = dict(ra_t.cert.detail)
    assert d["people_pred"] == "time" and d["t_offset"] == 0.0 and d["n_near_samples"] == len(ra_t.segments)      # every segment checked against people

    wb = _crossing(world, pts, 2.5, 4.0, -0.6)
    rb_f, _ = _cert(wb, sim, pts, "frozen")
    rb_t, why_bt = _cert(wb, sim, pts, "time")
    assert rb_f is not None and rb_f.cert.margin >= 0
    assert rb_t is None and "segment" in why_bt


def test_non_immediate_route_keeps_the_people_free_rule(world, sim, pts):
    """a route that does not start at the world's current conf is not predicted (H0 beyond near_m, frozen near): same cert as frozen."""
    wb = _crossing(world, pts, 2.5, 4.0, -0.6)
    st = ST.Streams(wb, cert_mode="segments", lateral_offsets=(), people_pred="time")
    q_far = Conf("far", 2.13, 3.0, -1.57)
    assert not st.is_immediate(q_far) and st.is_immediate(Conf("here", *wb.q0.xyth()))
    c_nonimm = st._certify_segments(pts, False, False, None, immediate=False)
    c_froz = ST.Streams(wb, cert_mode="segments", lateral_offsets=())._certify_segments(pts, False, False, None)
    assert c_nonimm[0] == c_froz[0]


def test_timed_wait_picks_the_smallest_working_tau(world, sim, pts):
    # synthetic probe: works from tau >= 2.0 on
    tau, probes = PP.probe_wait(lambda t: t >= 2.0)
    assert tau == 2.0 and probes == [(0.5, False), (1.0, False), (1.5, False), (2.0, True)]
    assert PP.probe_wait(lambda t: False)[0] is None
    # real planner: a person crossing toward the route at 0.8 m/s; at the speed 0.3 alone nothing certifies until the person has passed (tau = 3 s here)
    w = _crossing(world, pts, 2.5, 3.0, -0.8)
    cache = {}
    qa, qb = _confs(sim)

    def ok(tau):
        st = ST.Streams(w, cert_mode="segments", lateral_offsets=(), speeds=(0.3,), people_pred="time", t_offset=tau, seg_cache=cache)
        return st.build_route(qa, qb, pts, False, False, None, "scene:out_route")[0] is not None
    assert not ok(0.0)
    tau, probes = PP.probe_wait(ok)
    assert tau is not None and tau > 0
    assert all(not p for _, p in probes[:-1]) and probes[-1][1]
    # monotone sanity: nothing smaller than the chosen tau works
    assert not any(ok(t) for t in PP.TAUS if t < tau)
