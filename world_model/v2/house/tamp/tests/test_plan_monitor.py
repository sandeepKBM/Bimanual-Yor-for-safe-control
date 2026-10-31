"""opt-in plan-prediction monitor (tamp/plan_monitor.py, SegmentGatekeeper.coast / predict_state, segments.plan_route_segments(start=...)).  SIMULATION ONLY.
Fast: the monitor logic is tested standalone; the gatekeeper-level tests use one short TeleopSim (no people, injected belief people), no full episodes."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from world_model.v2.house.gatekeeper.belief import GKConfig
from world_model.v2.house.gatekeeper.traj import Traj
from world_model.v2.house.tamp import plan_monitor as PM
from world_model.v2.house.tamp import seg_gate as SG
from world_model.v2.house.tamp import segments as SEG
from world_model.v2.house.tamp import streams as ST
from world_model.v2.house.tamp import world as W
from world_model.v2.house.tamp.types import Conf
from world_model.v2.house.teleop import base as BS
from world_model.v2.house.teleop.sim import Cmd, TICK

STEP = 0.1                                     # gatekeeper step [s]
PATH = SEG.Path(((0.0, 0.0), (10.0, 0.0)))


def _run(mon, obs_fn, t0, t1, s_cur=0.0):
    """feed the monitor every STEP from t0 to t1; obs_fn(t) -> (people, vel).  -> (flag or None, number of checks run)."""
    n = 0
    t = t0
    while t <= t1 + 1e-9:
        p, v = obs_fn(t)
        n += 1
        fl = mon.check(p, v, t, PATH, s_cur, 10.0)
        if fl is not None:
            return fl, n
        t = round(t + STEP, 6)
    return None, n


# ---------------------------------------------------------------------------------------------------- 1. no flag when people follow the prediction
def test_no_flag_when_prediction_holds():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2)), (np.array([6.0, -1.0]), np.array([0.4, 0.0]))], t_obs=2.0)
    fl, n = _run(mon, lambda t: ([np.array([3.0, 1.0]), np.array([6.0 + 0.4 * (t - 2.0), -1.0])], [np.zeros(2), np.array([0.4, 0.0])]), 2.0, 12.0)
    assert fl is None and n > 90 and mon.active and mon.n_viol == 0


def test_small_noise_inside_eps_does_not_flag():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2))], t_obs=0.0)
    fl, _ = _run(mon, lambda t: ([np.array([3.0 + 0.25, 1.0])], [np.array([0.15, 0.0])]), 0.0, 5.0)
    assert fl is None


# ---------------------------------------------------------------------------------------------------- 2. flag when a standing person starts walking / drifts
def test_standing_person_starts_walking_flags_vel_within_confirm_checks():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2))], t_obs=0.0)

    def obs(t):
        w = max(0.0, t - 1.0)
        return [np.array([3.0 + 0.5 * w, 1.0])], [np.array([0.5 if t >= 1.0 else 0.0, 0.0])]
    fl, n = _run(mon, obs, 0.0, 5.0)
    assert fl is not None and fl["kind"] == "vel" and fl["person"] == 0
    assert abs(fl["t"] - 1.1) < 1e-6                   # first violating check t = 1.0, second (confirming) t = 1.1  (n_confirm = 2)
    assert fl["vel_err"] == pytest.approx(0.5)
    assert not mon.active                              # off after the flag until a new prediction


def test_slow_drift_flags_pos():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2))], t_obs=0.0)
    fl, _ = _run(mon, lambda t: ([np.array([3.0 + 0.05 * t, 1.0])], [np.array([0.05, 0.0])]), 0.0, 20.0)      # 5 cm/s: vel inside eps_v, position exceeds 0.30 m at t = 6 s
    assert fl is not None and fl["kind"] == "pos" and fl["t"] > 6.0 and fl["pos_err"] > 0.30


def test_rearm_after_flag():
    mon = PM.PlanMonitor(n_confirm=1)
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2))], t_obs=0.0)
    assert mon.check([np.array([4.0, 1.0])], [np.zeros(2)], 0.1, PATH, 0.0, 10.0) is not None
    assert mon.check([np.array([4.0, 1.0])], [np.zeros(2)], 0.2, PATH, 0.0, 10.0) is None          # one flag per prediction
    mon.set_prediction([(np.array([4.0, 1.0]), np.zeros(2))], t_obs=0.2)
    assert mon.active and mon.check([np.array([4.0, 1.0])], [np.zeros(2)], 0.3, PATH, 0.0, 10.0) is None


# ---------------------------------------------------------------------------------------------------- 3. debounce
def test_single_check_glitch_does_not_flag():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2))], t_obs=0.0)
    glitch = {0.5}
    fl, _ = _run(mon, lambda t: ([np.array([3.0 + (1.0 if round(t, 6) in glitch else 0.0), 1.0])], [np.zeros(2)]), 0.0, 3.0)
    assert fl is None and mon.n_viol == 0
    glitch.update({1.0, 1.1})                              # two consecutive -> flag
    fl, _ = _run(mon, lambda t: ([np.array([3.0 + (1.0 if round(t, 6) in glitch else 0.0), 1.0])], [np.zeros(2)]), 0.0, 3.0)
    assert fl is not None and fl["kind"] == "pos"


def test_confirm_parameter():
    mon = PM.PlanMonitor(n_confirm=4)
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2))], t_obs=0.0)
    fl, n = _run(mon, lambda t: ([np.array([4.0, 1.0])], [np.zeros(2)]), 0.0, 3.0)
    assert fl is not None and n == 4


# ---------------------------------------------------------------------------------------------------- 4. relevance / new person
def test_motion_change_far_from_remaining_route_is_ignored():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([3.0, 8.0]), np.zeros(2))], t_obs=0.0)                       # 8 m from the route
    fl, _ = _run(mon, lambda t: ([np.array([3.0 + 0.5 * t, 8.0])], [np.array([0.5, 0.0])]), 0.0, 5.0)
    assert fl is None
    # same person near the route: flags
    mon.set_prediction([(np.array([3.0, 1.5]), np.zeros(2))], t_obs=0.0)
    fl, _ = _run(mon, lambda t: ([np.array([3.0 + 0.5 * t, 1.5])], [np.array([0.5, 0.0])]), 0.0, 5.0)
    assert fl is not None


def test_relevance_uses_remaining_route_only():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([1.0, 1.0]), np.zeros(2))], t_obs=0.0)                       # 1 m off the route, but the robot is already past it
    fl, _ = _run(mon, lambda t: ([np.array([1.0 + 0.5 * t, 1.0])], [np.array([0.5, 0.0])]), 0.0, 1.0, s_cur=8.0)
    assert fl is None                                                                           # remaining route [8, 10]: 7+ m away
    fl, _ = _run(mon, lambda t: ([np.array([1.0 + 0.5 * t, 1.0])], [np.array([0.5, 0.0])]), 0.0, 1.0, s_cur=0.0)
    assert fl is not None


def test_new_person_flags_only_when_relevant():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2))], t_obs=0.0)
    far = lambda t: ([np.array([3.0, 1.0]), np.array([5.0, 9.0])], [np.zeros(2), np.zeros(2)])
    assert _run(mon, far, 0.0, 2.0)[0] is None
    near = lambda t: ([np.array([3.0, 1.0]), np.array([5.0, 1.2])], [np.zeros(2), np.zeros(2)])
    fl, _ = _run(mon, near, 0.0, 2.0)
    assert fl is not None and fl["kind"] == "new" and fl["person"] == 1


def test_nearest_neighbour_association_when_counts_differ():
    mon = PM.PlanMonitor()
    mon.set_prediction([(np.array([3.0, 1.0]), np.zeros(2)), (np.array([7.0, 1.0]), np.zeros(2))], t_obs=0.0)
    # only the second person is observed (index 0 in the belief): it must match the SECOND prediction, not the first
    fl, _ = _run(mon, lambda t: ([np.array([7.0, 1.0])], [np.zeros(2)]), 0.0, 2.0)
    assert fl is None


def test_dist_to_polyline():
    p = SEG.Path(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)))
    assert PM.dist_to_polyline((2.0, 3.0), p, 0.0, 8.0) == pytest.approx(2.0)
    assert PM.dist_to_polyline((2.0, 3.0), p, 4.0, 8.0) == pytest.approx(2.0)     # remaining part is the leg x = 4
    assert PM.dist_to_polyline((0.0, 0.0), p, 4.0, 8.0) == pytest.approx(4.0)


# ---------------------------------------------------------------------------------------------------- 6. predicted start state at t_f + L (stub sim, hand-made trajectory)
class _StubBase:
    def __init__(self, v):
        self.v = np.asarray(v, float)


class _StubSim:
    def __init__(self, pose, v):
        self._pose, self.base, self.a_hw = pose, _StubBase(v), BS.A_HW.copy()

    def pose(self):
        return np.asarray(self._pose, float)


def _committed():
    """0.3 m/s forward for 1.0 s (nominal), then a linear brake to rest over 1.0 s (decel 0.3 m/s^2), grid h = 0.05."""
    h = 0.05
    V = [[0.3, 0.0, 0.0]] * 21 + [[0.3 * (1 - k / 20.0), 0.0, 0.0] for k in range(1, 21)]
    return Traj(np.array(V), h, 0.0, 1.0, 1.0)


@pytest.fixture(scope="module")
def stub_filter():
    gk = dataclasses.replace(GKConfig(), plan_monitor="on", **{k: v for k, v in W.RUNTIME_GK.items() if k in GKConfig.__dataclass_fields__ and not k.startswith("plan_")})
    f = SG.SegmentGatekeeper(gk=gk)
    f.traj = _committed()
    f.n_calls = 0
    return f


def _ref(tr, dt):
    k = int(round(dt / tr.h))
    P = tr.poses()
    return P[k], tr.V[k]


@pytest.mark.parametrize("dt,kind", [(0.0, "nominal"), (0.5, "nominal"), (1.0, "nominal"), (1.5, "brake"), (1.9, "brake"), (2.0, "rest"), (3.0, "rest")])
def test_predicted_state_equals_committed_trajectory_sample(stub_filter, dt, kind):
    sim = _StubSim((0.0, 0.0, 0.0), (0.3, 0.0, 0.0))
    st = stub_filter.predict_state(sim, dt)
    pose_ref, v_ref = _ref(stub_filter.traj, min(dt, 2.0))
    assert st["kind"] == kind
    assert np.allclose(st["pose"], pose_ref, atol=1e-6)
    assert np.allclose(st["v"], v_ref, atol=1e-6)
    if kind == "rest":
        assert np.all(st["v"] == 0.0)


def test_predicted_state_from_a_rotated_start_pose(stub_filter):
    sim = _StubSim((1.0, 2.0, math.pi / 2), (0.3, 0.0, 0.0))
    st = stub_filter.predict_state(sim, 3.0)
    assert st["kind"] == "rest"
    assert st["pose"][0] == pytest.approx(1.0, abs=1e-9) and st["pose"][1] == pytest.approx(2.0 + 0.45, abs=1e-6)       # 0.45 m driven along the heading (+y)


# ---------------------------------------------------------------------------------------------------- 7. planner side: optional start state
@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


def _route_setup(world):
    pts = tuple(map(tuple, world.routes[0][1]))
    path = SEG.Path(pts)
    s_stop = path.total - ST.Streams(world, cert_mode="segments").end_skip
    bel_fn = lambda pose, s, t: W.runtime_belief_at(world, pose, False, None, people=False)
    return path, s_stop, bel_fn


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x.V), np.asarray(y.V)) and x.t_sw == y.t_sw and x.T_S == y.T_S and x.s_end == y.s_end and tuple(x.pose_start) == tuple(y.pose_start)


def test_plan_route_segments_start_none_is_the_original(world):
    path, s_stop, bel_fn = _route_setup(world)
    a, _ = SEG.plan_route_segments(path, False, 0.3, world.gk, bel_fn, s_stop)
    b, _ = SEG.plan_route_segments(path, False, 0.3, world.gk, bel_fn, s_stop, start=None)
    c, _ = SEG.plan_route_segments(path, False, 0.3, world.gk, bel_fn, s_stop, start=(SEG.start_pose(path, False), np.zeros(3), 0.0, 0.0))
    assert a is not None
    _same(a, b)
    _same(a, c)                                                                    # an explicit rest start at the route start is the same state


def test_plan_route_segments_moving_start_uses_given_twist(world):
    path, s_stop, bel_fn = _route_setup(world)
    v0 = np.array([0.2, 0.0, 0.0])
    p0 = SEG.start_pose(path, False)
    segs, why = SEG.plan_route_segments(path, False, 0.3, world.gk, bel_fn, s_stop, start=(p0, v0, 0.0, 0.0))
    assert segs is not None, why
    assert np.array_equal(np.asarray(segs[0].V[0]), v0)
    assert tuple(segs[0].pose_start) == tuple(p0) and segs[0].s_start == 0.0


def test_streams_start_v_default_unchanged_and_applied_to_immediate_route(world):
    pts = tuple(map(tuple, world.routes[0][1]))
    q0 = world.q0
    qa = Conf("qa", *SEG.start_pose(SEG.Path(pts), False))
    w2 = dataclasses.replace(world, q0=qa)
    st0 = ST.Streams(w2, cert_mode="segments", people_pred="time")
    r0, why0 = st0.build_route(qa, Conf("qb", pts[-1][0], pts[-1][1], 0.0), pts, False, False, None, "scene:out_route")
    st1 = ST.Streams(w2, cert_mode="segments", people_pred="time", start_v=(0.2, 0.0, 0.0))
    r1, why1 = st1.build_route(qa, Conf("qb", pts[-1][0], pts[-1][1], 0.0), pts, False, False, None, "scene:out_route")
    assert r0 is not None and r1 is not None, (why0, why1)
    assert np.allclose(r0.segments[0].V[0], 0.0)
    assert np.allclose(r1.segments[0].V[0], [0.2, 0.0, 0.0])


def test_world_from_sim_robot_pose_override(sim):
    w = W.world_from_sim(sim, robot_pose=(1.0, 2.0, 0.5))
    assert (w.q0.x, w.q0.y, w.q0.yaw) == (1.0, 2.0, 0.5)
    w0 = W.world_from_sim(sim)
    assert tuple(np.round([w0.q0.x, w0.q0.y, w0.q0.yaw], 9)) == tuple(np.round(sim.pose(), 9))


# ---------------------------------------------------------------------------------------------------- 8. defaults off
def test_defaults_are_off():
    g = GKConfig()
    assert g.plan_monitor == "off" and (g.plan_eps_p, g.plan_eps_v, g.plan_r_rel, g.plan_confirm) == (0.30, 0.20, 2.0, 2)
    f = SG.make_seg_filter()
    assert f.pm is None and not f._coast
    f.set_plan_prediction([(np.zeros(2), np.zeros(2))], 0.0)                       # no-op
    assert f.pm is None and "plan_flags" not in f.counters()


# ---------------------------------------------------------------------------------------------------- 5. gatekeeper level: no new commits after a flag
class _InjectedFilter(SG.SegmentGatekeeper):
    """SegmentGatekeeper whose belief carries a person we script (the sim has none): standing at xy until `walk_t`, then walking +x at 0.5 m/s."""
    person0 = np.array([4.0, 6.0])
    walk_t = math.inf

    def _bel_now(self, sim, t):
        b = super()._bel_now(sim, t)
        w = max(0.0, sim.t - self.walk_t)
        moving = sim.t >= self.walk_t
        b = b.with_(people=(self.person0 + np.array([0.5 * w, 0.0]),), people_vel=(np.array([0.5 if moving else 0.0, 0.0]),), people_e_p=(), people_e_v=(), people_stale=())
        self.last_bel = b
        return b


def _driver(sim, world, walk_t, decision="brake"):
    from world_model.v2.house.teleop.sim import TeleopSim  # noqa: F401
    gk = dataclasses.replace(GKConfig(), **{**W.RUNTIME_GK, "plan_monitor": "on", "plan_r_rel": 50.0, "plan_decision": decision})
    f = _InjectedFilter(gk=gk)
    f.walk_t = walk_t
    f.docks = []
    wp = sim.scene.waypoints
    pts = tuple(map(tuple, world.routes[0][1]))
    qa, qb = Conf("a", *wp["lab_start"]), Conf("b", *wp["cafe_counter_pick"], surface="counter")
    st = ST.Streams(world, cert_mode="segments")
    r, why = st.build_route(qa, qb, pts, False, False, None, "scene:out_route")
    assert r is not None, why
    return f, r


def test_no_new_segments_after_flag_and_coast_never_commits(sim, world, monkeypatch):
    from world_model.v2.house.teleop.sim import TeleopSim
    s2 = TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)
    w2 = W.world_from_sim(s2)
    f, route = _driver(s2, w2, walk_t=2.0)
    calls = {"n": 0}
    orig = SEG.next_certified_segment

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(SEG, "next_certified_segment", spy)
    f.begin_route(route)
    f.set_plan_prediction([(f.person0.copy(), np.zeros(2))], s2.t)                 # the plan: that person stands there for ever
    flagged_at = None
    for k in range(400):
        safe, _ = f.filter(s2, Cmd())
        s2.tick(safe)
        if f.seg_state() == "plan_falsified":
            flagged_at = k
            break
    assert flagged_at is not None, "monitor never flagged"
    assert f.pm.flag is not None and f.pm.flag["kind"] == "vel" and f._coast
    assert 2.0 <= s2.t <= 2.5                                                      # flag within ~n_confirm checks of the walk start (0.2 s + tick alignment)
    n_calls_flag, n_commit_flag, traj_flag = calls["n"], f.n_commit, f.traj
    assert traj_flag is not None and f.mode == "keep"
    # the executor would now end_route; coast for 4 s: only the committed trajectory is followed
    f.end_route("plan_falsified")
    assert f.sr is None
    modes = set()
    for k in range(int(4.0 / TICK)):
        safe, info = f.filter(s2, Cmd())
        s2.tick(safe)
        modes.add(f.mode)
    assert calls["n"] == n_calls_flag, "a segment was built after the flag"
    assert f.n_commit == n_commit_flag, "something was committed in coast mode"
    assert modes <= {"keep", "regime2"}
    assert float(np.abs(s2.base.v).max()) <= SG.REST_V                             # the committed brake ran to rest
    # leaving coast mode restores the plain gatekeeper (next filter call may commit again)
    f.coast(False)
    for _ in range(10):
        safe, _ = f.filter(s2, Cmd())
        s2.tick(safe)
    assert f.n_commit > n_commit_flag


def test_monitor_off_never_flags(sim, world):
    from world_model.v2.house.teleop.sim import TeleopSim
    s2 = TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)
    w2 = W.world_from_sim(s2)
    gk = dataclasses.replace(GKConfig(), **W.RUNTIME_GK)
    f = _InjectedFilter(gk=gk)
    f.walk_t = 0.5
    f.docks = []
    wp = s2.scene.waypoints
    pts = tuple(map(tuple, w2.routes[0][1]))
    r, why = ST.Streams(w2, cert_mode="segments").build_route(Conf("a", *wp["lab_start"]), Conf("b", *wp["cafe_counter_pick"], surface="counter"), pts, False, False, None, "scene:out_route")
    f.begin_route(r)
    f.set_plan_prediction([(f.person0.copy(), np.zeros(2))], s2.t)
    for _ in range(150):
        safe, _ = f.filter(s2, Cmd())
        s2.tick(safe)
    assert f.seg_state() in ("active", "refused", "done") and f.pm is None


# ---------------------------------------------------------------------------------------------------- amendment: timing test (continue / brake / switch)
def test_conflict_time_analytic():
    # robot drives +x at 0.3 m/s from the origin; a person 2 m ahead, 1.5 m to the side, walking toward the line at 0.5 m/s
    S = [(0.05 * k, (0.3 * 0.05 * k, 0.0, 0.0), np.array([0.3, 0.0, 0.0]), "nominal") for k in range(0, 201)]
    far = PM.conflict_time(S, [np.array([20.0, 9.0])], [np.zeros(2)], [0.3], [0.2], 0.0, 0.1)
    assert far == math.inf
    t = PM.conflict_time(S, [np.array([2.0, 1.5])], [np.array([0.0, -0.5])], [0.3], [0.0], 0.0, 0.1)
    assert 0.0 < t < 10.0
    # a stationary person right next to the start conflicts at t_f itself
    assert PM.conflict_time(S, [np.array([0.3, 0.2])], [np.zeros(2)], [0.3], [0.0], 0.0, 0.1) == 0.0
    # larger velocity error can only bring the conflict earlier
    t2 = PM.conflict_time(S, [np.array([2.0, 1.5])], [np.array([0.0, -0.5])], [0.3], [0.2], 0.0, 0.1)
    assert t2 <= t


class _ScriptedFilter(SG.SegmentGatekeeper):
    """belief people scripted by the test: P = [(xy, v), ...] (static between assignments)."""
    P = []

    def _bel_now(self, sim, t):
        b = super()._bel_now(sim, t)
        b = b.with_(people=tuple(np.array(p, float) for p, _ in self.P), people_vel=tuple(np.array(v, float) for _, v in self.P), people_e_p=(), people_e_v=(), people_stale=())
        self.last_bel = b
        return b


def _scripted(people, decision="timing", L=2.0, margin=0.5):
    from world_model.v2.house.teleop.sim import TeleopSim
    s2 = TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)
    w2 = W.world_from_sim(s2)
    gk = dataclasses.replace(GKConfig(), **{**W.RUNTIME_GK, "plan_monitor": "on", "plan_r_rel": 50.0, "plan_decision": decision, "plan_conf_margin": margin})
    f = _ScriptedFilter(gk=gk)
    f.docks = []
    f.set_plan_latency(L)
    wp = s2.scene.waypoints
    pts = tuple(map(tuple, w2.routes[0][1]))
    r, why = ST.Streams(w2, cert_mode="segments").build_route(Conf("a", *wp["lab_start"]), Conf("b", *wp["cafe_counter_pick"], surface="counter"), pts, False, False, None, "scene:out_route")
    assert r is not None, why
    _ScriptedFilter.P = [(p, v) for p, v in people]
    f.begin_route(r)
    f.set_plan_prediction([(np.array(p, float), np.zeros(2)) for p, _ in people], s2.t)         # the plan: everybody stands still
    return s2, f


def _step(s2, f, n):
    for _ in range(n):
        safe, _ = f.filter(s2, Cmd())
        s2.tick(safe)


def _until_flag(s2, f, walker_v):
    """walk person 0 (starting to move at once) until the monitor flags."""
    for _ in range(400):
        _ScriptedFilter.P = [(_ScriptedFilter.P[0][0], walker_v)] + _ScriptedFilter.P[1:]
        _step(s2, f, 1)
        if f.plan_flag_log:
            return True
    return False


def test_far_or_no_conflict_continues_and_keeps_committing_old_route_segments():
    # person changes motion (walks east, away) but never comes near the route -> t_conf = inf -> continue
    s2, f = _scripted([((6.0, 8.0), (0.0, 0.0))])
    assert _until_flag(s2, f, (0.5, 0.0))
    fl = f.plan_flag_log[-1]
    assert fl["decision"] == "continue" and fl["t_conf"] == math.inf and f.plan_continue and not f._coast
    assert f.seg_state() == "plan_falsified" and f.sr["status"] == "active"
    n_adm, n_commit = f.sr["n_admitted"], f.n_commit
    _step(s2, f, int(1.5 / TICK))
    assert f.sr["n_admitted"] > n_adm and f.n_commit > n_commit, "continue must keep committing old-route segments (normal certification)"
    assert f.plan_flag_log[-1]["decision"] == "continue"
    rec = f.end_route("plan_falsified_handover")                                   # the ladder's hand-over at t_ready
    assert rec["status"] == "plan_falsified_handover"


def test_far_finite_conflict_continues_until_t_ready_then_handover():
    s2, f = _scripted([((5.5, 4.5), (0.0, 0.0))])
    assert _until_flag(s2, f, (-0.5, 0.0))                                         # walks toward the route line; conflict is a few seconds away
    fl = f.plan_flag_log[-1]
    assert fl["decision"] == "continue", fl
    assert math.isfinite(fl["t_conf"]) and fl["t_conf"] > fl["t_ready"] + fl["t_stop"] + fl["margin"]
    n0 = f.n_commit
    while f.n_calls * TICK < fl["t_ready"] - 1e-9:
        _ScriptedFilter.P = [(_ScriptedFilter.P[0][0], (-0.5, 0.0))]
        _step(s2, f, 1)
    assert f.n_commit > n0 and f.plan_flag_log[-1]["decision"] == "continue"


def test_near_conflict_brakes_and_commits_no_new_old_route_segment(monkeypatch):
    s2, f = _scripted([((3.6, 4.5), (0.0, 0.0))])
    assert _until_flag(s2, f, (-0.5, 0.0))
    fl = f.plan_flag_log[-1]
    assert fl["decision"] == "brake" and fl["t_conf"] < fl["t_ready"] + fl["t_stop"] + fl["margin"]
    assert f._coast and f.sr["status"] == "plan_falsified"
    n_commit = f.n_commit
    _step(s2, f, int(3.0 / TICK))
    assert f.n_commit == n_commit


def test_mid_continue_person_change_switches_to_brake_at_next_boundary():
    s2, f = _scripted([((5.5, 4.5), (0.0, 0.0))])
    assert _until_flag(s2, f, (-0.5, 0.0))
    assert f.plan_flag_log[-1]["decision"] == "continue"
    n_commit = f.n_commit
    _ScriptedFilter.P = [((3.3, 4.5), (-1.0, 0.0))]                                # now much closer and faster: t_conf moves earlier
    for _ in range(int(3.0 / TICK)):
        _step(s2, f, 1)
        if f._coast:
            break
    fl = f.plan_flag_log[-1]
    assert fl["decision"] == "switched_to_brake" and fl["decision_initial"] == "continue" and f._coast
    assert fl["t_conf_switch"] < fl["t_ready"] + fl["t_stop_switch"] + fl["margin"]
    n_after = f.n_commit
    _step(s2, f, int(2.0 / TICK))
    assert f.n_commit == n_after, "a segment was committed after the switch to brake"
    assert f.n_commit >= n_commit


def test_decision_brake_forces_brake_even_without_conflict():
    s2, f = _scripted([((6.0, 8.0), (0.0, 0.0))], decision="brake")
    assert _until_flag(s2, f, (0.5, 0.0))
    assert f.plan_flag_log[-1]["decision"] == "brake" and f._coast
