"""plan-time certifier (tamp/plan_cert.py), Streams(plan_cert="pred"), runtime_rows="cup" in the segment executor, CLI flags.  SIMULATION ONLY, no physics stepping,
no rendering, no network.  python -m pytest -q world_model/v2/house/tamp/tests/test_plan_cert.py"""
from __future__ import annotations

import dataclasses
import json
import math
import sys

import numpy as np
import pytest

from world_model.v2.house.gatekeeper import GKConfig, check1, check2, plan_candidate
from world_model.v2.house.tamp import plan_cert as PC
from world_model.v2.house.tamp import run_tamp as RT
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


def _confs(sim):
    wp = sim.scene.waypoints
    return Conf("a", *wp["lab_start"]), Conf("b", *wp["cafe_counter_pick"], surface="counter")


def _pts(world):
    return tuple(map(tuple, world.routes[0][1]))


def _with_person(world, xy, v=(0.0, 0.0)):
    b = world.bel0.with_(people=(np.asarray(xy, float),), people_vel=(np.asarray(v, float),), people_e_p=(), people_e_v=(), people_stale=())
    return dataclasses.replace(world, bel_people=b)


# ------------------------------------------------------------------------------------------------ defaults
def test_defaults_are_unchanged(world, sim):
    qa, qb = _confs(sim)
    st = ST.Streams(world, cert_mode="segments")
    assert st.plan_cert == "off" and st._gk is world.gk
    assert world.gk.runtime_rows == "all" and world.gk.plan_cert == "off"
    r0, w0 = st.build_route(qa, qb, _pts(world), False, False, None, "scene:out_route")
    r1, w1 = ST.Streams(world, cert_mode="segments", plan_cert="off").build_route(qa, qb, _pts(world), False, False, None, "scene:out_route")
    assert r0 is not None and r1 is not None and r0.cert == r1.cert
    for a, b in zip(r0.segments, r1.segments):
        np.testing.assert_array_equal(a.V, b.V)
    with pytest.raises(AssertionError):
        ST.Streams(world, plan_cert="pred")                                 # needs cert_mode="segments"


def test_planner_always_checks_walls_and_people_even_when_the_runtime_gk_is_cup(world, sim):
    qa, qb = _confs(sim)
    wc = dataclasses.replace(world, gk=dataclasses.replace(world.gk, runtime_rows="cup"))
    st = ST.Streams(wc, cert_mode="segments")
    assert st._gk.runtime_rows == "all"
    r0, _ = ST.Streams(world, cert_mode="segments").build_route(qa, qb, _pts(world), False, False, None, "scene:out_route")
    r1, _ = st.build_route(qa, qb, _pts(world), False, False, None, "scene:out_route")
    assert r0.cert == r1.cert and len(r0.segments) == len(r1.segments)
    for a, b in zip(r0.segments, r1.segments):
        np.testing.assert_array_equal(a.V, b.V)


# ------------------------------------------------------------------------------------------------ seed-1 doorway standoff smoke
POSE0 = (2.14, 3.56, -math.pi / 2)                                  # sweep_1022c2 ours_r_s1: robot at the standoff, remaining route = down the corridor to the cafe counter
WALKER = (np.array([2.6, 3.8]), np.zeros(2))                        # scripted walker frozen ~0.145 m from the footprint
P_STAND = (np.array([2.134 - 1.32, 3.5]), np.zeros(2))


def _route_traj(world):
    bel = PC.plan_belief(world, POSE0, True, "tray", [WALKER, P_STAND], 0.0)
    return plan_candidate(np.zeros(3), np.array([0.3, 0.0, 0.0]), 5.0, bel, world.gk), bel      # 1.5 m (route: 1.56 m to (2.135, 2.0)) straight at 0.3 m/s, cup-safe ramp + backup


def test_seed1_standoff_remaining_route_passes_pred_and_fails_the_speed_model(world, capsys):
    tr, _ = _route_traj(world)
    P = tr.poses()
    assert abs(P[-1, 0] - 1.5) < 0.05 and abs(P[-1, 1]) < 1e-9           # straight along the body x axis (= down the corridor for yaw -pi/2)
    # prediction-error growth e_v t only.  NOT the defaults: at e_v = 0.2 the margin is -0.078 m, at the default a_p = 2 the disc r0 + e_v t + a_p t^2/2 swallows the corridor
    # within ~1.5 s (margin -22 m); measured in the report.  e_v <= 0.15, a_p = 0 passes.
    gk = dataclasses.replace(world.gk, e_v=0.1, a_p=0.0)
    ok, m, why, mg = PC.certify_plan(tr, world, gk, [WALKER, P_STAND], 0.0, POSE0)
    # the same trajectory under the default speed model (reachable disc at 1.5 m/s), same belief otherwise
    bs = PC.plan_belief(world, POSE0, True, "tray", [WALKER, P_STAND], 0.0, gk).with_(person_model="speed", people_vel=(), people_stale=())
    o1, m1, r1, g1 = check1(tr, bs, world.gk)
    o2, m2, r2, g2 = check2(tr, bs, world.gk)
    with capsys.disabled():
        print(f"\n[seed-1 smoke] pred(e_v=0.1,a_p=0): ok={ok} min={m:.4f} ({why}) {({k: round(v, 4) for k, v in mg.items()})}")
        print(f"[seed-1 smoke] speed(v_p=1.5):      ok={o1 and o2} c1={m1:.4f} ({r1}) c2={m2:.4f} ({r2})")
    assert ok, (m, why, mg)
    assert mg["person_exit"] >= 0.0 and mg["wall"] > 0.0 and mg["cup"] >= 0.0 and mg["rest_person"] >= 0.0
    assert not (o1 and o2) and m1 < 0.0 and r1 == "person"


def test_seed1_standoff_moving_toward_the_walker_fails_pred(world):
    bel = PC.plan_belief(world, POSE0, True, "tray", [WALKER, P_STAND], 0.0)
    gk = dataclasses.replace(world.gk, e_v=0.2, a_p=0.0)
    tr = plan_candidate(np.zeros(3), np.array([-0.3, 0.0, 0.0]), 1.0, bel, gk)      # backs up UP the corridor, towards the walker at (2.6, 3.8)
    ok, m, why, mg = PC.certify_plan(tr, world, gk, [WALKER, P_STAND], 0.0, POSE0)
    assert not ok and mg["person_exit"] < 0.0


# ------------------------------------------------------------------------------------------------ certify_plan / certify_segments
def test_certify_plan_no_people_equals_check1_check2_rows(world):
    pose = (2.14, 6.0, -math.pi / 2)
    bel = PC.plan_belief(world, pose, True, "tray", [], 0.0)
    tr = plan_candidate(np.zeros(3), np.array([0.3, 0, 0]), 1.0, bel, world.gk)
    ok, m, why, mg = PC.certify_plan(tr, world, world.gk, [], 0.0, pose)
    c1, c2 = check1(tr, bel, dataclasses.replace(world.gk, runtime_rows="all")), check2(tr, bel, dataclasses.replace(world.gk, runtime_rows="all"))
    assert ok == (c1[0] and c2[0]) and set(mg) == set(c1[3]) | set(c2[3]) and "person" not in mg
    assert ok and why not in PC.TIGHT_BY_CONSTRUCTION


def test_certify_plan_ignores_the_runtime_rows_setting(world):
    pose = (2.14, 6.0, -math.pi / 2)
    bel = PC.plan_belief(world, pose, True, "tray", [(np.array([2.14 + 0.6, 6.0 - 1.0]), np.zeros(2))], 0.0)
    tr = plan_candidate(np.zeros(3), np.array([0.3, 0, 0]), 1.0, bel, world.gk)
    a = PC.certify_plan(tr, world, world.gk, [(np.array([2.14 + 0.6, 6.0 - 1.0]), np.zeros(2))], 0.0, pose)
    b = PC.certify_plan(tr, world, dataclasses.replace(world.gk, runtime_rows="cup"), [(np.array([2.14 + 0.6, 6.0 - 1.0]), np.zeros(2))], 0.0, pose)
    assert a[0] == b[0] and a[3] == b[3] and "wall" in a[3] and "person" in a[3]


def test_delay_t0_grows_the_error_and_advances_the_person(world):
    pose = (2.14, 6.0, -math.pi / 2)
    tr = plan_candidate(np.zeros(3), np.array([0.3, 0, 0]), 1.0, PC.plan_belief(world, pose, True, "tray", [], 0.0), world.gk)
    gk = dataclasses.replace(world.gk, e_v=0.1, a_p=0.1)
    still = [(np.array([3.8, 5.5]), np.zeros(2))]                          # 1.5 m beside the footprint, stationary
    walker = [(np.array([3.8, 5.5]), np.array([-0.4, 0.0]))]               # same place, walking toward the route at 0.4 m/s
    m0 = PC.certify_plan(tr, world, gk, still, 0.0, pose)[3]["person"]
    m1 = PC.certify_plan(tr, world, gk, still, 1.0, pose)[3]["person"]
    w0, w1 = (PC.certify_plan(tr, world, gk, walker, t, pose)[3]["person"] for t in (0.0, 1.0))
    assert m0 > m1 > 0.0                                                   # a trajectory that starts 1 s after the observation faces a wider disc
    assert w0 - w1 > m0 - m1 + 0.1                                         # ... and, for a walker, one that has advanced 0.4 m toward the route


def test_certify_segments_chain_matches_streams_segments_and_flags_a_person_on_the_route(world, sim):
    qa, qb = _confs(sim)
    gk0 = dataclasses.replace(world.gk, e_v=0.0, a_p=0.0)
    w0 = dataclasses.replace(world, gk=gk0)
    r, why = ST.Streams(w0, cert_mode="segments", plan_cert="pred").build_route(qa, qb, _pts(w0), False, True, "tray", "scene:out_route")
    assert r is not None, why
    trajs = [sg.traj() for sg in r.segments]
    ok, m, reason, mg = PC.certify_segments(trajs, w0, gk0, [], 0.0, SEG.start_pose(SEG.Path(_pts(w0)), False), holding=True, item="tray")
    assert ok and m >= 0.0
    path = SEG.Path(_pts(w0))
    xy = path.point(6.0) + 0.05 * np.array([-math.sin(path.heading(6.0)), math.cos(path.heading(6.0))])
    ok2, m2, reason2, _ = PC.certify_segments(trajs, w0, gk0, [(xy, np.zeros(2))], 0.0, SEG.start_pose(path, False), holding=True, item="tray")
    assert not ok2 and m2 < 0 and reason2.startswith("seg ") and reason2.endswith("person")


def test_streams_pred_checks_the_whole_route_no_near_m_cutoff(world, sim):
    qa, qb = _confs(sim)
    gk0 = dataclasses.replace(world.gk, e_v=0.0, a_p=0.0)
    path = SEG.Path(_pts(world))
    far = path.point(6.0)                                                    # 6 m along the route: beyond near_m = 2 m, so the legacy rule never looks at this person
    w1 = dataclasses.replace(_with_person(world, far), gk=gk0)
    legacy, _ = ST.Streams(w1, cert_mode="segments", lateral_offsets=()).build_route(qa, qb, _pts(w1), False, False, None, "scene:out_route")
    assert legacy is not None
    pred, why = ST.Streams(w1, cert_mode="segments", lateral_offsets=(), plan_cert="pred").build_route(qa, qb, _pts(w1), False, False, None, "scene:out_route")
    assert pred is None and "person" in why
    # no people: pred == off (same segments)
    r_off, _ = ST.Streams(dataclasses.replace(world, gk=gk0), cert_mode="segments").build_route(qa, qb, _pts(world), False, False, None, "scene:out_route")
    r_pred, _ = ST.Streams(dataclasses.replace(world, gk=gk0), cert_mode="segments", plan_cert="pred").build_route(qa, qb, _pts(world), False, False, None, "scene:out_route")
    assert len(r_off.segments) == len(r_pred.segments)
    for a, b in zip(r_off.segments, r_pred.segments):
        np.testing.assert_array_equal(a.V, b.V)
    assert dict(r_pred.cert.detail)["plan_cert"] == "pred"


# ------------------------------------------------------------------------------------------------ executor: cup mode refuses the route, no Shield
def test_segment_gatekeeper_cup_mode_pose_falsification_refuses_route_and_never_enters_the_shield(world, sim, monkeypatch):
    from world_model.v2.house.teleop import shield as SH
    from world_model.v2.house.tamp.seg_gate import SegmentGatekeeper
    qa, qb = _confs(sim)
    r, why = ST.Streams(world, cert_mode="segments").build_route(qa, qb, _pts(world), False, False, None, "scene:out_route")
    assert r is not None, why
    sg = r.segments[len(r.segments) // 2]
    calls = []
    monkeypatch.setattr(SH.Shield, "filter_base", lambda self, s, u: calls.append(1) or (_ for _ in ()).throw(AssertionError("Shield.filter_base entered")))
    save = (sim.base.x, sim.base.y, sim.base.yaw, sim.base.v.copy())
    try:
        sim.base.x, sim.base.y, sim.base.yaw = sg.pose_start
        sim.base.v = np.array(sg.V[0], float)
        filt = SegmentGatekeeper(gk=dataclasses.replace(GKConfig(), **dict(W.RUNTIME_GK, runtime_rows="cup")))
        filt.docks = []
        filt.begin_route(r)
        filt.sr["s"] = sg.s_start
        filt._boundary(sim, 0.0)
        assert filt.seg_state() == "active" and filt.mode == "commit"
        filt.gk_step(sim, np.zeros(3), 0.0)                                  # on the plan (the sim is not stepped, so t = commit time): the T1 KEEP rule keeps the segment
        assert filt.seg_state() == "active" and filt.mode == "keep"
        sim.base.x += 0.3                                                    # pose falsified
        filt.gk_step(sim, np.zeros(3), 0.0)
        assert filt.seg_state() == "refused" and filt.sr["reason"].startswith("falsified: pose_")
        assert filt.mode == "cup_brake" and filt.traj.i_rest is not None and filt.traj.t_sw <= 0.0 and filt.n_falsified == 1
        for k in range(10):
            u, lab = filt.filter_base(sim, np.zeros(3))
            filt.n_calls += 1
        assert "regime2" not in lab and filt.n_regime2_ticks == 0 and not calls          # the base gatekeeper then holds zero (backup to rest), never the Shield
        assert filt.end_route()["status"] == "refused"                      # the executor turns this into "segment_refused" -> the run_tamp ladder replans
    finally:
        sim.base.x, sim.base.y, sim.base.yaw, sim.base.v = save


# ------------------------------------------------------------------------------------------------ CLI
def test_cli_flags_are_parsed_defaults_recorded_and_runtime_gk_untouched_by_default(tmp_path, monkeypatch):
    seen = {}

    def stub(*a, **k):
        seen["gk"] = dict(W.RUNTIME_GK)
        return dict(completed=True, n_replans=0, aborted="", sim_s=0.0, wall_s=0.0, episode={}, segments=None, n_recoveries=0, recoveries_by_kind={}, final_route_sources=[])
    save = dict(W.RUNTIME_GK)
    monkeypatch.setattr(RT, "run_episode", stub)
    try:
        monkeypatch.setattr(sys, "argv", ["run_tamp", "--out", str(tmp_path), "--name", "d"])
        RT.main()
        d = json.loads((tmp_path / "d.json").read_text())
        assert (d["runtime_rows"], d["plan_cert"], d["pred_ev"], d["pred_ap"]) == ("all", "off", GKConfig().e_v, GKConfig().a_p)
        assert not ({"runtime_rows", "plan_cert", "e_v", "a_p"} & set(seen["gk"]))            # default run: RUNTIME_GK exactly as before
        W.RUNTIME_GK.clear(); W.RUNTIME_GK.update(save)
        monkeypatch.setattr(sys, "argv", ["run_tamp", "--out", str(tmp_path), "--name", "p", "--filter", "gatekeeper", "--cert", "segments", "--runtime-rows", "cup",
                                          "--plan-cert", "pred", "--pred-ev", "0.1", "--pred-ap", "0.4"])
        RT.main()
        d = json.loads((tmp_path / "p.json").read_text())
        assert (d["runtime_rows"], d["plan_cert"], d["pred_ev"], d["pred_ap"]) == ("cup", "pred", 0.1, 0.4)
        assert (seen["gk"]["runtime_rows"], seen["gk"]["plan_cert"], seen["gk"]["e_v"], seen["gk"]["a_p"]) == ("cup", "pred", 0.1, 0.4)
    finally:
        W.RUNTIME_GK.clear(); W.RUNTIME_GK.update(save)
