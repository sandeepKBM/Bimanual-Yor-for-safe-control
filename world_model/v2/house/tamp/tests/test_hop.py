"""opt-in HOP primitive (tamp/hop.py, SegmentGatekeeper.hop_plan / hop_commit, run_tamp --hop, tamp/hop_e3.py).  SIMULATION ONLY.
Fast: checker calls on injected beliefs, one short TeleopSim for the execution tests, a stubbed filter/planner for the ladder-order tests (no planner search, no full episodes)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from world_model.v2.house.gatekeeper.belief import GKConfig
from world_model.v2.house.tamp import hop as HOP
from world_model.v2.house.tamp import hop_e3 as E3
from world_model.v2.house.tamp import run_tamp as RT
from world_model.v2.house.tamp import seg_gate as SG
from world_model.v2.house.tamp import segments as SEG
from world_model.v2.house.tamp import world as W
from world_model.v2.house.teleop.sim import Cmd, TICK, TeleopSim

H = 0.05


@pytest.fixture(scope="module")
def sim():
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


@pytest.fixture(scope="module")
def gk(world):
    return world.gk


def _bel(world, people, vels=None, pose=(0.0, 0.0, 0.0), holding=False):
    """walls removed, truth people (e_p = 0) at `people`, velocity estimates `vels`."""
    n = len(people)
    b = world.bel0.with_(people=tuple(np.array(p, float) for p in people), people_vel=tuple(np.array(v, float) for v in (vels or [(0.0, 0.0)] * n)),
                         people_e_p=(), people_e_v=(), people_stale=(), grid=None)
    w2 = dataclasses.replace(world, bel_people=b)
    return W.runtime_belief_at(w2, pose, holding, "tray" if holding else None, people=True).with_(grid=None)


# ---------------------------------------------------------------------------------------------------- 1. the trajectory
@pytest.mark.parametrize("t_h", [0.1, 0.15, 0.2, 0.35, 0.4, 0.6])
@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.5])
def test_hop_traj_rest_to_rest_pure_translation(t_h, yaw):
    a_r = 1.2
    d = (math.cos(0.7), math.sin(0.7))                                      # world direction
    tr = HOP.hop_traj(d, yaw, t_h, a_r, H)
    n = int(round(t_h / H))
    assert tr.N == n and np.all(tr.V[0] == 0.0) and np.all(tr.V[-1] == 0.0)
    assert tr.i_rest == n and tr.t_rest == pytest.approx(t_h)                  # ends EXACTLY at rest, not before
    assert np.all(tr.V[:, 2] == 0.0)                                           # wz = 0
    P = tr.poses()
    assert np.allclose(P[:, 2], 0.0)                                           # yaw constant
    wd = np.array([math.cos(yaw) * P[-1, 0] - math.sin(yaw) * P[-1, 1], math.sin(yaw) * P[-1, 0] + math.cos(yaw) * P[-1, 1]])
    assert np.allclose(wd / np.hypot(*wd), d, atol=1e-9)                       # moves along the WORLD direction
    disp = HOP.displacement(tr)
    assert abs(disp - a_r * (n * H) ** 2 / 4) <= a_r * H * H / 4 + 1e-9        # a_r t^2/4 within one grid step (exact for even n)
    if n % 2 == 0:
        assert disp == pytest.approx(a_r * t_h ** 2 / 4)
    A = np.linalg.norm(np.diff(tr.V[:, :2], axis=0) / H, axis=1)
    assert A.max() <= a_r + 1e-9
    assert tr.t_sw == pytest.approx((n // 2) * H)


def test_hop_duration_rounds_down_to_the_grid():
    assert HOP.hop_traj((1, 0), 0.0, 0.37, 1.0, H).N == 7
    assert HOP.durations_grid([0.6, 0.35, 0.34, 0.1, 0.03], H) == [12, 7, 6, 2]


def test_hop_a_r_sources(world, gk):
    # empty hands, cup row attached (RUNTIME_GK): only the hardware slew box / speed cap bind (a_hw = 4 along body x)
    be = _bel(world, [], holding=False)
    assert HOP.hop_a_r(be, gk, np.array([1.0, 0.0]), 0.3) == pytest.approx(min(4.0, 2 * 0.9 / 0.3))
    assert HOP.hop_a_r(be, gk, np.array([0.0, 1.0]), 0.3) == pytest.approx(3.0)               # vy cap 0.45: 2 * 0.45 / 0.3
    assert HOP.hop_a_r(be, gk, np.array([0.0, 1.0]), 0.1) == pytest.approx(4.0)
    assert HOP.hop_a_r(be, gk, np.array([1.0, 0.0]), 0.6) == pytest.approx(3.0)               # speed cap: 2 * 0.9 / 0.6
    # tray held: the gatekeeper's cup bound plan_frac * a_cup_max
    bh = _bel(world, [], holding=True)
    assert HOP.hop_a_r(bh, gk, np.array([0.6, 0.8]), 0.3) == pytest.approx(gk.plan_frac * bh.a_cup_max)
    assert gk.plan_frac * bh.a_cup_max == pytest.approx(0.97 * 0.9 * 0.144 * 9.81)


def test_hop_checks_use_the_unchanged_checker(world, gk):
    """the hop is certified by gatekeeper.checks.check1/check2 as they are: the rows' margins reproduce by calling them directly."""
    from world_model.v2.house.gatekeeper.checks import check1, check2
    bel = _bel(world, [(2.0, 0.0)], holding=True)
    r = HOP.longest_admissible(bel, gk, (1.0, 0.0), HOP.HOP_DURATIONS, HOP.HOP_MIN_M)
    assert r is not None
    o1, _, _, g1 = check1(r["traj"], bel, gk)
    o2, _, _, g2 = check2(r["traj"], bel, gk)
    assert o1 and o2 and g1 == r["info"]["c1"] and g2 == r["info"]["c2"]
    assert g1["cup"] >= 0 and g1["hw_slew"] >= -1e-6                                         # cup row and hardware rows present and met


# ---------------------------------------------------------------------------------------------------- 2. monotone in distance
def test_admissible_duration_is_monotone_in_person_distance(world, gk):
    ds = [0.45, 0.5, 0.55, 0.65, 0.8, 1.0, 1.2, 1.5, 2.5]
    for holding in (False, True):
        th = []
        for d in ds:
            r = HOP.longest_admissible(E3.belief_with_person(world, d, holding), gk, (1.0, 0.0), [H * n for n in range(40, 1, -1)], 0.0)
            th.append(0.0 if r is None else r["t_h"])
        assert all(b >= a - 1e-12 for a, b in zip(th, th[1:])), th
        assert th[-1] > th[0]


# ---------------------------------------------------------------------------------------------------- 3. frozen radius
def test_frozen_radius_and_far_hop(world, gk):
    near = E3.belief_with_person(world, 0.36)
    for k in range(8):                                                                       # no hop in ANY direction inside the frozen radius
        d = (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4))
        assert HOP.longest_admissible(near, gk, d, HOP.HOP_DURATIONS, 0.0) is None
    fr = E3.frozen_radius(gk, near)
    assert min(fr.values()) == pytest.approx(0.4075)
    assert HOP.longest_admissible(E3.belief_with_person(world, 0.8), gk, (1.0, 0.0), HOP.HOP_DURATIONS, HOP.HOP_MIN_M) is not None
    assert HOP.longest_admissible(E3.belief_with_person(world, 0.45), gk, (1.0, 0.0), [H * n for n in range(12, 1, -1)], 0.0) is not None   # just outside the frozen radius (0.1 s / 1 cm)


def test_min_displacement_rejects_tiny_hops(world, gk):
    b = E3.belief_with_person(world, 0.45)                                                     # only a 0.1 s / 1 cm hop fits here
    assert HOP.longest_admissible(b, gk, (1.0, 0.0), HOP.HOP_DURATIONS, 0.0) is not None
    assert HOP.longest_admissible(b, gk, (1.0, 0.0), HOP.HOP_DURATIONS, HOP.HOP_MIN_M) is None


# ---------------------------------------------------------------------------------------------------- 4. direction choice
def test_prog_wins_when_admissible(world, gk):
    path = SEG.Path(((-2.0, 0.0), (6.0, 0.0)))
    bel = _bel(world, [(4.0, 3.0)])
    c = HOP.choose_hop(bel, gk, HOP.HOP_DIRS, HOP.HOP_DURATIONS, HOP.HOP_MIN_M, route_path=path, back_target=(-1.0, 0.0, 0.0))
    assert c is not None and c["kind"] == "prog" and np.allclose(c["d"], (1.0, 0.0))
    assert c["tried"] == [("prog", "ok")] and c["disp"] >= HOP.HOP_MIN_M


@pytest.mark.parametrize("x, want", [(1.2, -1.0), (-1.2, 1.0)])
def test_clear_picks_the_side_away_from_a_crossing_walkers_path(world, gk, x, want):
    # walker at (x, -1.2) walking +y at 0.5 m/s: its tube crosses the robot's x-axis; the perpendicular of v pointing away from the tube is -sign(x)
    bel = _bel(world, [(x, -1.2)], [(0.0, 0.5)])
    d = HOP.dir_clear(bel, bel.pose_wall)
    assert d is not None and d[0] * want > 0.95
    c = HOP.choose_hop(bel, gk, ("clear",), HOP.HOP_DURATIONS, HOP.HOP_MIN_M)
    assert c is not None and c["kind"] == "clear" and c["d"][0] * want > 0.95


def test_clear_standing_person_goes_straight_away(world, gk):
    bel = _bel(world, [(0.9, 0.6)], pose=(0.0, 0.0, 0.0))
    d = HOP.dir_clear(bel, bel.pose_wall)
    c = np.array([0.9, 0.6])
    assert np.allclose(d, -c / np.hypot(*c))


def test_back_is_the_fallback(world, gk):
    bel = _bel(world, [])                                                                       # nobody: no "clear"; no route: no "prog"
    c = HOP.choose_hop(bel, gk, HOP.HOP_DIRS, HOP.HOP_DURATIONS, HOP.HOP_MIN_M, route_path=None, back_target=(-1.0, -1.0, 0.0))
    assert c is not None and c["kind"] == "back" and np.allclose(c["d"], (-1 / math.sqrt(2), -1 / math.sqrt(2)))
    # prog too short to be a hop (person 0.5 m ahead of the front edge, route straight at it: <= 2 cm), moving away is not -> order respected, back wins
    path = SEG.Path(((-2.0, 0.0), (6.0, 0.0)))
    bel2 = _bel(world, [(0.577 + 0.5, 0.0)])
    c2 = HOP.choose_hop(bel2, gk, ("prog", "back"), HOP.HOP_DURATIONS, 0.05, route_path=path, back_target=(-1.0, 0.0, 0.0))
    assert c2 is not None and c2["kind"] == "back" and c2["tried"][0] == ("prog", "none")


def test_compass_is_last_resort_and_off_by_default(world, gk):
    bel = _bel(world, [])
    assert HOP.choose_hop(bel, gk, HOP.HOP_DIRS) is None                                         # no route, no person, no rest pose: nothing without compass
    c = HOP.choose_hop(bel, gk, HOP.HOP_DIRS, compass=True)
    assert c is not None and c["kind"].startswith("compass")


def test_out_of_tubes(world, gk):
    assert HOP.out_of_tubes(_bel(world, []), (0.0, 0.0, 0.0))
    assert HOP.out_of_tubes(_bel(world, [(6.0, 6.0)], [(0.0, 0.0)]), (0.0, 0.0, 0.0))
    assert not HOP.out_of_tubes(_bel(world, [(1.0, 0.0)], [(0.0, 0.0)]), (0.0, 0.0, 0.0))
    assert not HOP.out_of_tubes(_bel(world, [(3.0, -1.0)], [(-0.5, 0.5)]), (0.0, 0.0, 0.0))     # walks into the robot within 4 s


# ---------------------------------------------------------------------------------------------------- 5. execution through the SegmentGatekeeper
class _Filter(SG.SegmentGatekeeper):
    """belief people scripted (the sim has none)."""
    P = ()

    def _bel_now(self, sim, t):
        b = super()._bel_now(sim, t)
        b = b.with_(people=tuple(np.array(p, float) for p, _ in self.P), people_vel=tuple(np.array(v, float) for _, v in self.P), people_e_p=(), people_e_v=(), people_stale=())
        self.last_bel = b
        return b


def _fresh():
    s2 = TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)
    f = _Filter(gk=dataclasses.replace(GKConfig(), **W.RUNTIME_GK))
    f.docks = []
    _Filter.P = ()
    return s2, f


def _settle(s2, f, n=12):
    for _ in range(n):
        safe, _ = f.filter(s2, Cmd())
        s2.tick(safe)


def test_commit_follows_the_hop_in_coast_mode_and_ends_at_rest():
    s2, f = _fresh()
    _settle(s2, f)
    p0 = np.array(s2.pose())
    ch = f.hop_plan(s2, kinds=(), compass=True)
    assert ch is not None
    n_commit0 = f.n_commit
    rec = f.hop_commit(s2, ch["d"], ch["t_h"], ch["kind"])
    assert rec["ok"] and f._coast and f.n_commit == n_commit0 + 1 and f.traj is not None and f.traj.i_rest == int(round(ch["t_h"] / H))
    assert rec["margin"] >= -1e-9 and rec["c1"] is not None and rec["c2"] is not None
    vmax = 0.0
    for k in range(int(round(ch["t_h"] / TICK)) + 40):
        safe, info = f.filter(s2, Cmd())
        s2.tick(safe)
        assert f.n_commit == n_commit0 + 1, "something was committed while coasting"
        vmax = max(vmax, float(np.abs(s2.base.v).max()))
    p1 = np.array(s2.pose())
    assert vmax > 0.01 and float(np.abs(s2.base.v).max()) <= SG.REST_V and f.mode in ("keep", "commit")
    disp = float(np.hypot(*(p1[:2] - p0[:2])))
    assert disp == pytest.approx(rec["disp_planned"], abs=0.01)
    assert abs(p1[2] - p0[2]) < 1e-3                                                               # no rotation
    mv = (p1[:2] - p0[:2]) / disp
    assert float(mv @ ch["d"]) > 0.98                                                              # along the chosen WORLD direction
    f.coast(False)
    assert not f._coast and f.counters()["hops"][-1]["ok"]


def test_recheck_from_the_actual_state_refuses_and_does_not_move():
    s2, f = _fresh()
    _settle(s2, f)
    p0 = np.array(s2.pose())
    ch = f.hop_plan(s2, kinds=(), compass=True)
    assert ch is not None
    # the world changed between choosing and executing: a person is now 0.3 m in front of the hop direction -> the fresh belief must refuse it
    c = p0[:2] + 0.7 * np.asarray(ch["d"])
    _Filter.P = [(tuple(c), (0.0, 0.0))]
    n_commit0, traj0 = f.n_commit, f.traj
    rec = f.hop_commit(s2, ch["d"], ch["t_h"], ch["kind"])
    assert not rec["ok"] and rec["why"].startswith("c1 person")
    assert not f._coast and f.n_commit == n_commit0 and f.traj is traj0
    _settle(s2, f, 30)
    assert float(np.hypot(*(np.array(s2.pose())[:2] - p0[:2]))) < 1e-3                             # the base did not move


def test_hop_refused_when_not_at_rest_or_route_active():
    s2, f = _fresh()
    _settle(s2, f)
    s2.base.v = np.array([0.2, 0.0, 0.0])
    assert f.hop_plan(s2, kinds=(), compass=True) is None
    rec = f.hop_commit(s2, (1.0, 0.0), 0.2)
    assert not rec["ok"] and rec["why"] == "not_at_rest"
    s2.base.v = np.zeros(3)


def test_default_filter_has_no_hop_state():
    f = SG.make_seg_filter()
    assert f.hop_log == [] and "hops" not in f.counters()


# ---------------------------------------------------------------------------------------------------- 6. ladder
class _Bel:
    pass


class _StubFilter:
    """enough of the filter for the ladder's unsolved branch: records hop calls; hops 'succeed' without moving."""

    def __init__(self, solve_after=None):
        self.docks = []
        self.plan_calls = 0
        self.commits = 0
        self.fp_now = None
        self.solve_after = solve_after
        self.fn = []

    def filter(self, sim, cmd):
        return Cmd(), {}

    def coast(self, on):
        pass

    def hop_plan(self, sim, kinds, durations, min_m, route_path=None, back_target=None, compass=False):
        self.plan_calls += 1
        return dict(kind="prog", d=np.array([1.0, 0.0]), t_h=0.1, a_r=1.0, bel=_Bel(), tried=[("prog", "ok")])

    def hop_commit(self, sim, d, t_h, kind="hop", min_m=0.02):
        self.commits += 1
        return dict(ok=True, kind=kind, t_h=t_h, a_r=1.0, margin=0.1, pose_before=[float(x) for x in sim.pose()], why="")


class _StubEx:
    def _log(self, *a, **k):
        pass

    def episode_metrics(self):
        return {}


def _ladder(sim, monkeypatch, hop, solve_at=None, hop_max=3, max_waits=1, max_recoveries=6, span_s=12.0):
    """run RT._ladder with a planner that never solves (or solves on its solve_at-th call) and a stub filter."""
    s2, max_s = sim, sim.t + span_s
    calls = {"n": 0}

    def fake_solve(sim, goal, held, plan_time, **kw):
        calls["n"] += 1
        ok = solve_at is not None and calls["n"] >= solve_at
        return dict(solved=ok, plan=[] if ok else None, time_s=0.0, cost=0.0, stream_log=[], checker_stats={})
    monkeypatch.setattr(RT, "_solve", fake_solve)
    monkeypatch.setattr(RT.GM, "rect_dist_point", lambda *a, **k: (1.0,))
    f = _StubFilter()
    summ = RT._ladder(s2, f, _StubEx(), "gatekeeper", False, 0, "(On tray desk)", max_s, 10.0, 1.0, False, None, 0.5, max_waits, max_recoveries, 0.0, cert="segments",
                      people_pred="time", wait_mode="fixed", hop=hop, hop_cfg=dict(max_hops=hop_max))
    return summ, f, calls


def _kinds(summ):
    return [e["kind"] for e in summ["recoveries"]]


def test_ladder_hop_off_makes_no_hop_events(sim, monkeypatch):
    summ, f, _ = _ladder(sim, monkeypatch, "off")
    assert f.plan_calls == 0 and f.commits == 0 and "hop" not in _kinds(summ) and "hop_chains" not in summ and "n_hops" not in summ


def test_ladder_before_wait_runs_the_chain_before_the_first_wait(sim, monkeypatch):
    summ, f, _ = _ladder(sim, monkeypatch, "before_wait", hop_max=3)
    k = _kinds(summ)
    assert k[0] == "hop" and k[1] == "wait"
    ev = summ["recoveries"][0]
    assert ev["n_hops"] == 3 and f.commits == sum(e.get("n_hops", 0) for e in summ["recoveries"]) and ev["stop"] == "max_hops" and not ev["solved"]            # hop-max respected; the chain is ONE event
    assert ev["directions"] == ["prog"] * 3 and len(ev["displacements"]) == 3
    assert k.count("hop") >= 1 and all(e["kind"] != "hop" or e["n_hops"] <= 3 for e in summ["recoveries"])


def test_ladder_after_wait_runs_the_chain_after_the_waits_and_before_the_retreat(sim, monkeypatch):
    summ, f, _ = _ladder(sim, monkeypatch, "after_wait", hop_max=2, max_waits=2)
    k = _kinds(summ)
    assert k[:3] == ["wait", "wait", "hop"]
    assert summ["recoveries"][2]["n_hops"] == 2
    assert k.index("hop") < k.index("retreat_uncertifiable")


def test_ladder_chain_stops_and_hands_over_the_plan_when_it_solves(sim, monkeypatch):
    summ, f, calls = _ladder(sim, monkeypatch, "before_wait", solve_at=3, hop_max=10)         # call 1 = initial plan; the chain's 2nd replan solves
    ev = summ["recoveries"][0]
    assert ev["kind"] == "hop" and ev["n_hops"] == 2 and ev["solved"] and ev["stop"] == "plan solved" and summ["completed"]
    assert _kinds(summ) == ["hop"]
    assert summ["recoveries"][0]["replan_solved"] is True


def test_ladder_hop_chain_counts_as_one_recovery_for_max_recoveries(sim, monkeypatch):
    summ, f, _ = _ladder(sim, monkeypatch, "before_wait", hop_max=5, max_recoveries=1)
    assert _kinds(summ) == ["hop", "wait"] and summ["aborted"] == "max recoveries reached" and summ["recoveries"][0]["n_hops"] == 5


# ---------------------------------------------------------------------------------------------------- 7. E3: checker vs the T4(c) lemma
@pytest.mark.parametrize("holding", [False, True])
def test_e3_checker_is_consistent_with_the_lemma(world, gk, holding):
    rows = E3.e3_rows(world, gk, holding=holding)
    for r in rows:
        assert r["t_h"] <= r["t_bar"] + H + 1e-9, r               # the checker never admits a hop longer than the lemma allows (one grid step: it checks the last MOVING sample)
        if r["t_bar"] > 0:
            assert r["t_h"] >= r["t_bar"] - H - 1e-9, r           # ... and does not refuse hops the lemma admits (by more than a grid step)
        else:
            assert r["t_h"] == 0.0
        assert r["t_bar"] <= r["t_bar_raw"] + 1e-9                # the draft's formula as written (no d_person, no sweep) is optimistic
    assert [r["t_h"] for r in rows if r["d"] <= 0.40] == [0.0] * len([r for r in rows if r["d"] <= 0.40])
    assert all(r["t_h"] > 0 for r in rows if r["d"] >= 0.45)
