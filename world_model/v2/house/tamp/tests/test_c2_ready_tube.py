"""opt-in C2-ready TRACKED-TUBE ranking term + scope switch (tamp/segments.py next_certified_segment site/tube, GKConfig.c2_ready_scope/tube_h/tube_clear, seg_gate boundaries).
SIMULATION ONLY: no rendering, no network.  Placements were found by probe scans on 10-22 (route 0 of the h5 scene runs along x ~ 2.13 from y = 5 down to y = 2.5)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from world_model.v2.house.gatekeeper.belief import GKConfig
from world_model.v2.house.tamp import hop as HOP
from world_model.v2.house.tamp import plan_monitor as PM
from world_model.v2.house.tamp import seg_gate as SG
from world_model.v2.house.tamp import segments as SEG
from world_model.v2.house.tamp import streams as ST
from world_model.v2.house.tamp import world as W
from world_model.v2.house.tamp.types import Conf
from world_model.v2.house.teleop import geom as GM
from world_model.v2.house.teleop.sim import Cmd, TeleopSim


@pytest.fixture(scope="module")
def sim():
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


@pytest.fixture(scope="module")
def route(world):
    path = SEG.Path(tuple(map(tuple, world.routes[0][1])))
    s_stop = path.total - ST.Streams(world, cert_mode="segments").end_skip
    segs, why = SEG.plan_route_segments(path, False, 0.3, world.gk, lambda pose, s, t: W.runtime_belief_at(world, pose, False, None, people=False), s_stop)
    assert segs is not None, why
    return path, s_stop, segs


def _state(route, world, k, a, d, vx=0.0, vy=0.0):
    """segment k's start state; a person a m ahead of / d m left of the route point there, moving (vx along, vy left) m/s."""
    path, s_stop, segs = route
    sg = segs[k]
    p0, h = path.point(sg.s_start), path.heading(sg.s_start)
    e, n = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
    b = world.bel0.with_(people=(p0 + a * e + d * n,), people_vel=(vx * e + vy * n,), people_e_p=(), people_e_v=(), people_stale=())
    return sg, W.runtime_belief_at(dataclasses.replace(world, bel_people=b), sg.pose_start, False, None, people=True)


def _ncs(route, sg, bel, gk, **kw):
    path, s_stop, _ = route
    return SEG.next_certified_segment(path, False, 0.3, sg.pose_start, sg.V[0], sg.s_start, bel, gk, s_stop, **kw)


def _same(A, B):
    assert (A[0] is None) == (B[0] is None) and A[5] == B[5]
    if A[0] is not None:
        np.testing.assert_array_equal(A[1]["V"], B[1]["V"])
        assert A[4] == B[4] and A[1]["offset"] == B[1]["offset"] and A[1]["ready"] == B[1]["ready"]


CROSS = dict(k=2, a=2.0, d=1.0, vx=0.0, vy=-0.4)        # walker 2 m ahead, 1 m left of the route, walking across it


def _brute_force_clear(tr, bel, gk):
    """independent re-implementation of the tube test: min over t in [t_rest, t_rest + h] of dist(footprint, tube disc) - d_person - clear >= 0 for every person."""
    from world_model.v2.house.gatekeeper.checks import _ev, _pe
    from world_model.v2.house.gatekeeper.ready import rest_poses
    _pw, pp, t_rest = rest_poses(tr, bel)
    vel = PM.people_vels(bel)
    t = t_rest
    while t <= t_rest + gk.c2_ready_tube_h + 1e-9:
        for i, c in enumerate(bel.people):
            ctr = np.asarray(c, float)[:2] + vel[i] * t
            if GM.rect_dist_point(ctr, pp, bel.fp)[0] - (GM.PERSON_CLEAR_R + _pe(bel, i) + _ev(bel, i) * t) <= bel.d_person + gk.c2_ready_tube_clear:
                return False
        t += 0.25
    return True


# ------------------------------------------------------------------------------------------------------------- defaults
def test_new_fields_default_off():
    g = GKConfig()
    assert g.c2_ready_scope == "both" and g.c2_ready_tube_h == 0.0 and g.c2_ready_tube_clear == 0.0
    assert W.RUNTIME_GK.get("c2_ready_tube_h", 0.0) == 0.0 and W.RUNTIME_GK.get("c2_ready_scope", "both") == "both"


@pytest.mark.parametrize("rank", ["off", "rank"])
def test_defaults_bit_identical_whatever_the_site(world, route, rank):
    gk = dataclasses.replace(world.gk, c2_ready=rank)
    for st in (dict(k=2, a=0.5, d=0.6), dict(k=2, a=1.3, d=0.2), dict(k=4, a=2.5, d=1.5), dict(k=7, a=0.9, d=0.5), CROSS):
        sg, bel = _state(route, world, **st)
        base = _ncs(route, sg, bel, gk)
        _same(base, _ncs(route, sg, bel, gk, site="gate"))
        _same(base, _ncs(route, sg, bel, gk, site="plan"))
        assert "tube_off" not in (base[1] or {})


def test_hop_out_of_tubes_unchanged_by_the_shared_helper(world):
    """the pre-refactor inline implementation of hop.out_of_tubes (copied) versus the delegating one."""
    from world_model.v2.house.gatekeeper.checks import _ev, _pe

    def old(bel, pose, t_now=0.0, horizon=4.0, dt=0.25):
        n = len(bel.people)
        if n == 0:
            return True
        S = [(t_now + k * dt, tuple(float(x) for x in pose), np.zeros(3), "rest") for k in range(int(round(horizon / dt)) + 1)]
        return PM.conflict_time(S, bel.people, HOP._vels(bel), [GM.PERSON_CLEAR_R + _pe(bel, i) for i in range(n)], [_ev(bel, i) for i in range(n)], t_now, bel.d_person, bel.fp) == math.inf
    rng = np.random.default_rng(0)
    for _ in range(40):
        c = rng.uniform(-3, 3, 2)
        v = rng.uniform(-0.6, 0.6, 2)
        b = world.bel0.with_(people=(c,), people_vel=(v,), people_e_p=(), people_e_v=(), people_stale=())
        pose = (*rng.uniform(-3, 3, 2), rng.uniform(-3, 3))
        t0 = float(rng.uniform(0, 3))
        assert HOP.out_of_tubes(b, pose, t0) == old(b, pose, t0)
        assert HOP.out_of_tubes(b, pose, t0, 2.0, 0.5) == old(b, pose, t0, 2.0, 0.5)


# ------------------------------------------------------------------------------------------------------------- scope switch
def test_scope_gate_ranks_only_at_the_gatekeeper_site(world, route):
    sg, bel = _state(route, world, 2, 1.3, 0.2)                 # the test_c2_ready placement: rank off T_S=0.3, rank on T_S=0.1
    both = dataclasses.replace(world.gk, c2_ready="rank")
    gate = dataclasses.replace(world.gk, c2_ready="rank", c2_ready_scope="gate")
    off = _ncs(route, sg, bel, world.gk)
    assert off[4] == 0.3
    assert _ncs(route, sg, bel, both)[4] == 0.1 and _ncs(route, sg, bel, both, site="plan")[4] == 0.1       # "both" = today everywhere
    _same(_ncs(route, sg, bel, gate, site="plan"), off)                                                       # planner: cheap first-passing rule
    _same(_ncs(route, sg, bel, gate), off)
    g = _ncs(route, sg, bel, gate, site="gate")
    _same(g, _ncs(route, sg, bel, both, site="gate"))
    assert g[4] == 0.1 and g[1]["ready"] == g[1]["ready_n"] == 2


def test_planner_unchanged_under_scope_gate_even_with_the_tube_term(world, route):
    path, s_stop, segs = route
    gk = dataclasses.replace(world.gk, c2_ready="rank", c2_ready_scope="gate", c2_ready_tube_h=4.0, c2_ready_tube_clear=0.5)
    plan, why = SEG.plan_route_segments(path, False, 0.3, gk, lambda pose, s, t: W.runtime_belief_at(world, pose, False, None, people=False), s_stop)
    assert plan is not None, why
    assert len(plan) == len(segs)
    for a, b in zip(plan, segs):
        np.testing.assert_array_equal(a.V, b.V)
        assert a.T_S == b.T_S and a.ready == -1


# ------------------------------------------------------------------------------------------------------------- the tube term
def test_walker_crossing_ahead_prefers_the_off_tube_shorter_commitment(world, route):
    sg, bel = _state(route, world, **CROSS)
    gk0 = world.gk
    gk1 = dataclasses.replace(gk0, c2_ready_tube_h=4.0)
    today = _ncs(route, sg, bel, gk0)
    on = _ncs(route, sg, bel, gk1, site="gate")
    assert today[0] is not None and on[0] is not None
    assert today[4] == 0.6 and on[4] == 0.3                                           # longer T_S parks in the tube; the shorter one rests before it
    assert not SEG.tube_off(today[0], bel, gk1) and not _brute_force_clear(today[0], bel, gk1)
    assert on[1]["tube_off"] and on[1]["tube_changed"] and SEG.tube_off(on[0], bel, gk1) and _brute_force_clear(on[0], bel, gk1)
    # rest pose is BEFORE the crossing along the route (smaller arclength), and the candidate is one the plain C1+C2 loop admits (safety neutrality)
    assert on[1]["s_end"] < today[1]["s_end"]
    assert SEG.check_segment(on[0], bel, gk0)[0]
    # the term is a boundary-site term: planner site and default site are untouched
    _same(_ncs(route, sg, bel, gk1, site="plan"), today)
    _same(_ncs(route, sg, bel, gk1), today)


def test_tube_term_agrees_with_an_independent_brute_force_over_many_states(world, route):
    gk0 = world.gk
    gk1 = dataclasses.replace(gk0, c2_ready_tube_h=4.0, c2_ready_tube_clear=0.2)
    n_changed = 0
    for k in (2, 4):
        for a in (1.0, 2.0, 3.0):
            for d in (-1.5, 1.0):
                for vy in (0.0, 0.4 * -np.sign(d)):
                    sg, bel = _state(route, world, k, a, d, 0.0, vy)
                    today, on = _ncs(route, sg, bel, gk0), _ncs(route, sg, bel, gk1, site="gate")
                    assert (today[0] is None) == (on[0] is None)               # never admits / rejects
                    if today[0] is None:
                        continue
                    assert on[1]["tube_off"] == _brute_force_clear(on[0], bel, gk1) == SEG.tube_off(on[0], bel, gk1)
                    assert SEG.check_segment(on[0], bel, gk0)[0]
                    if on[1]["tube_off"]:
                        n_changed += on[4] != today[4]
                    else:                                                      # nothing off-tube among the passing candidates -> today's choice
                        np.testing.assert_array_equal(on[1]["V"], today[1]["V"])
    assert n_changed >= 1


def test_no_people_same_choice_as_today(world, route):
    path, s_stop, segs = route
    bel_fn = lambda pose, s, t: W.runtime_belief_at(world, pose, False, None, people=False)
    for rank in ("off", "rank"):
        gk = dataclasses.replace(world.gk, c2_ready=rank)
        gkt = dataclasses.replace(gk, c2_ready_tube_h=4.0, c2_ready_tube_clear=0.5)
        for sg in segs[:6]:
            bel = bel_fn(sg.pose_start, sg.s_start, 0.0)
            A, B = _ncs(route, sg, bel, gk, site="gate"), _ncs(route, sg, bel, gkt, site="gate")
            np.testing.assert_array_equal(A[1]["V"], B[1]["V"])
            assert A[4] == B[4] and A[1]["ready"] == B[1]["ready"] and B[1]["tube_off"] and not B[1]["tube_changed"]


@pytest.mark.parametrize("rank", ["off", "rank"])
def test_no_off_tube_candidate_falls_back_to_todays_key(world, route, rank):
    """a huge required clearance puts every passing candidate on the tube -> exactly today's choice (first passing / today's ranking)."""
    gk = dataclasses.replace(world.gk, c2_ready=rank)
    gkt = dataclasses.replace(gk, c2_ready_tube_h=4.0, c2_ready_tube_clear=50.0)
    hits = 0
    for st in (dict(k=2, a=1.3, d=0.2), dict(k=2, a=2.0, d=1.0, vy=-0.4), dict(k=4, a=2.0, d=-1.5, vy=0.4)):
        sg, bel = _state(route, world, **st)
        A, B = _ncs(route, sg, bel, gk), _ncs(route, sg, bel, gkt, site="gate")
        _same(A, B)
        if B[0] is not None:
            hits += 1
            assert B[1]["tube_off"] is False and B[1]["tube_changed"] is False
    assert hits >= 2


def test_rank_and_tube_together_choose_among_off_tube_candidates_by_readiness(world, route):
    sg, bel = _state(route, world, **CROSS)
    gk = dataclasses.replace(world.gk, c2_ready="rank", c2_ready_tube_h=4.0)
    r = _ncs(route, sg, bel, gk, site="gate")
    assert r[0] is not None and r[1]["tube_off"] and r[1]["ready"] is not None
    assert SEG.check_segment(r[0], bel, world.gk)[0]


# ------------------------------------------------------------------------------------------------------------- gatekeeper level
class _Walker(SG.SegmentGatekeeper):
    """belief people scripted: a walker at (x0, y0) moving with (vx, vy), stopping dead at sim time t_stop (None = never).  Records, at every commit, whether the committed
    trajectory's REST pose is off the tube of the belief it was committed under (tube test with gk_tube)."""
    X0, Y0, V, T_STOP = 6.5, 3.5, (-0.3, 0.0), None
    gk_tube = None

    def _bel_now(self, sim, t):
        b = super()._bel_now(sim, t)
        tt = sim.t if self.T_STOP is None else min(sim.t, self.T_STOP)
        moving = self.T_STOP is None or sim.t < self.T_STOP
        p = np.array([self.X0 + self.V[0] * tt, self.Y0 + self.V[1] * tt])
        b = b.with_(people=(p,), people_vel=(np.array(self.V if moving else (0.0, 0.0)),), people_e_p=(), people_e_v=(), people_stale=())
        self.last_bel = b
        return b

    def _commit(self, tr, T, a, bel, t):
        if self.gk_tube is not None and self.sr is not None:
            self.rec.append((float(t), float(T), SEG.tube_off(tr, bel, self.gk_tube), bel))
        return super()._commit(tr, T, a, bel, t)


def _run_walker(tube_h, t_stop=None, plan_monitor="off", max_ticks=900):
    s2 = TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)
    w2 = W.world_from_sim(s2)
    extra = dict(c2_ready_scope="gate", c2_ready_tube_h=tube_h, plan_monitor=plan_monitor, plan_r_rel=50.0, plan_decision="brake")
    f = _Walker(gk=dataclasses.replace(GKConfig(), **{**W.RUNTIME_GK, **extra}))
    f.docks = []
    f.rec = []
    f.gk_tube = dataclasses.replace(f.gk, c2_ready_tube_h=4.0)
    _Walker.T_STOP = t_stop
    wp = s2.scene.waypoints
    r, why = ST.Streams(w2, cert_mode="segments").build_route(Conf("a", *wp["lab_start"]), Conf("b", *wp["cafe_counter_pick"], surface="counter"),
                                                              tuple(map(tuple, w2.routes[0][1])), False, False, None, "scene:out_route")
    assert r is not None, why
    f.begin_route(r)
    if plan_monitor == "on":
        f.set_plan_prediction([(np.array([_Walker.X0, _Walker.Y0]), np.array(_Walker.V))], s2.t)      # the plan: the walker keeps walking at v
    for _ in range(max_ticks):
        safe, _ = f.filter(s2, Cmd())
        s2.tick(safe)
        if f.seg_state() in ("done", "refused", "plan_falsified"):
            break
    return s2, f


def test_executor_commits_off_tube_backups_while_the_walker_crosses_ahead():
    s2, f = _run_walker(4.0)
    assert f.seg_state() == "done" and f.tube_log
    f.end_route()
    assert all(r["tube_off"] == rec[2] for r, rec in zip(f.tube_log, f.rec))          # the logged flag is the independent recomputation on the committed trajectory
    assert any(r["changed"] for r in f.tube_log) and all(r["tube_off"] for r in f.tube_log if r["changed"])
    assert f.seg_log[-1]["tube_hist"]["changed"] == sum(r["changed"] for r in f.tube_log)
    assert f.counters()["tube_log"]
    _, f0 = _run_walker(0.0)                                                          # term off: no log, no extra keys
    f0.end_route()
    assert not f0.tube_log and "tube_hist" not in f0.seg_log[-1] and "tube_log" not in f0.counters()


def test_committed_backup_is_off_the_tube_before_the_plan_monitor_flag_fires():
    """the walker (plan: keeps walking at v) stops dead at t = 5.5 s -> the monitor flags (decision brake: follow the committed trajectory).  The trajectory it follows was committed at the
    last boundary BEFORE the flag: its rest pose is off the walker's tracked tube (belief of that boundary).  Without the term the same boundary's commit is on the tube."""
    outs = {}
    for th in (4.0, 0.0):
        s2, f = _run_walker(th, t_stop=5.5, plan_monitor="on")
        assert f.plan_flag_log, f"no flag (th={th}, state {f.seg_state()})"
        t_flag = f.plan_flag_log[0]["gk_t"]
        pre = [r for r in f.rec if r[0] <= t_flag]
        assert pre
        outs[th] = (t_flag, pre[-1][2])
    assert outs[4.0][1] is True, outs
    assert outs[0.0][1] is False, outs
