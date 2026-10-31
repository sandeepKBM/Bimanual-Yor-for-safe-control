"""Tests for the opt-in T1 fixes: keep_rule="monitor", belief_filter="intersect", cup_chord.  CPU only.  python -m pytest -q world_model/v2/house/gatekeeper/tests/test_monitor.py"""
import dataclasses
import math

import numpy as np
import pytest

from world_model.v2.house.teleop import base as BS, shield as SH
from world_model.v2.house.gatekeeper import GKConfig, Belief, Gatekeeper, plan_candidate, check1
from world_model.v2.house.gatekeeper.monitor import disc_enclose, PersonSetFilter
from world_model.v2.house.gatekeeper.traj import acup, acup_chord, acup_series
from world_model.v2.house.gatekeeper.tests.test_gatekeeper import FakeSim, bel, R_CUP, GK


def _run(g, sim, u, n):
    labels = []
    for _ in range(n):
        v, lab = g.filter_base(sim, u)
        g.n_calls += 1
        sim.step(v)
        labels.append(lab)
    return labels


def test_defaults_are_the_old_behaviour():
    c = GKConfig()
    assert (c.keep_rule, c.belief_filter, c.cup_chord) == ("recertify", "none", False)
    g = Gatekeeper()
    assert g._pf is None and "falsify_reasons" not in g.counters() and "n_obs_inconsistent" not in g.counters()


# ------------------------------------------------------------------------------------------------ disc intersection
def test_disc_enclose_cases():
    c, r = disc_enclose([0, 0], 2.0, [0.5, 0], 0.5)                      # containment -> the smaller disc
    assert r == 0.5 and np.allclose(c, [0.5, 0])
    c, r = disc_enclose([0, 0], 0.7, [0, 0], 1.0)                        # concentric
    assert r == 0.7
    assert disc_enclose([0, 0], 1.0, [3, 0], 1.0) is None               # disjoint
    c, r = disc_enclose([0, 0], 1.0, [2, 0], 1.0)                        # tangent -> the point
    assert r == pytest.approx(0.0, abs=1e-6) and np.allclose(c, [1, 0])
    c, r = disc_enclose([0, 0], 1.0, [1, 0], 1.0)                        # symmetric lens: chord is the diameter, centre (0.5, 0), r = sqrt(3)/2
    assert np.allclose(c, [0.5, 0]) and r == pytest.approx(math.sqrt(3) / 2)
    c, r = disc_enclose([0, 0], 1.0, [1.2, 0], 0.5)                      # lens, smaller than both
    assert r < 0.5
    c, r = disc_enclose([0, 0], 1.0, [1.0, 0], 3.0)                      # big disc 2 covers c1: a<=0 branch -> disc 1 contains the lens (d + r1 <= r2 -> containment actually)
    assert r == 1.0


def test_disc_enclose_randomised_soundness():
    rng = np.random.default_rng(0)
    n_lens = 0
    for _ in range(3000):
        c1, c2 = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
        r1, r2 = rng.uniform(0, 2.5, 2)
        out = disc_enclose(c1, r1, c2, r2)
        P = c1 + rng.uniform(-1, 1, (400, 2)) * r1
        P = P[np.hypot(*(P - c1).T) <= r1]
        P = P[np.hypot(*(P - c2).T) <= r2]                               # true points in BOTH discs
        if out is None:
            assert len(P) == 0 and np.hypot(*(c2 - c1)) > r1 + r2 - 1e-9
            continue
        c, r = out
        assert r <= min(r1, r2) + 1e-9
        if len(P):
            n_lens += 1
            assert (np.hypot(*(P - c).T) <= r + 1e-9).all()
    assert n_lens > 500


def test_set_filter_truth_is_noop_and_counts_inconsistency():
    gk = GKConfig()
    f = PersonSetFilter(gk)
    for k in range(5):
        c, r = f.update(["a"], [np.array([1.0 + 0.1 * k, 2.0])], [0.0], [None], [0.2], 0.1 * k)
        assert np.allclose(c[0], [1.0 + 0.1 * k, 2.0]) and r[0] == 0.0
    assert f.n_inconsistent == 0
    c, r = f.update(["a"], [np.array([9.0, 2.0])], [0.0], [None], [0.2], 0.5)       # teleport beyond v_p_max*dt -> falsified, reset to the observation
    assert f.n_inconsistent == 1 and np.allclose(c[0], [9.0, 2.0])
    f = PersonSetFilter(GKConfig())                                                   # tracker-like radii: the lens is smaller than the raw disc
    f.update(["a"], [np.array([0.0, 0.0])], [0.3], [None], [0.2], 0.0)
    c, r = f.update(["a"], [np.array([0.3, 0.0])], [0.3], [None], [0.2], 0.05)        # predicted disc radius 0.375 around 0; obs 0.3 around 0.3 -> proper lens
    assert r[0] < 0.3 and f.n_inconsistent == 0


def test_filter_random_true_point_stays_inside():
    rng = np.random.default_rng(1)
    for accel in (False, True):
        gk = GKConfig(person_model="accel" if accel else "speed")
        f = PersonSetFilter(gk)
        x = rng.uniform(-1, 1, 2); v = rng.uniform(-1, 1, 2)
        for k in range(60):
            t = 0.1 * k
            a = rng.uniform(-1, 1, 2) * gk.a_p / math.sqrt(2)
            v = v + a * 0.1
            sp = np.hypot(*v)
            v = v if sp <= gk.v_p_max else v / sp * gk.v_p_max
            x = x + v * 0.1
            e = 0.25
            obs = x + rng.uniform(-1, 1, 2) * e / math.sqrt(2)                     # |obs - x| <= e
            vh = v + rng.uniform(-1, 1, 2) * 0.1
            c, r = f.update(["p"], [obs], [e], [vh], [0.2], t)
            assert np.hypot(*(x - c[0])) <= r[0] + 1e-9, (accel, k)
        assert f.n_inconsistent == 0


def test_truth_source_intersect_is_noop_in_the_gatekeeper():
    out = []
    for bf in ("none", "intersect"):
        g = Gatekeeper(gk=GKConfig(person_model="accel", belief_filter=bf))
        sim = FakeSim((1.5, 5.0, 0.0), people=[(3.8, 7.0)])
        out.append((_run(g, sim, np.array([0.9, 0, 0]), 150), sim.pose().copy(), g.n_commit, g.n_falsified))
    assert out[0][0] == out[1][0] and np.array_equal(out[0][1], out[1][1]) and out[0][2:] == out[1][2:]
    assert g._pf.n_inconsistent == 0 and g._pf.stats()["mean_filtered_radius"] == 0.0 and "n_obs_inconsistent" in g.counters()


# ------------------------------------------------------------------------------------------------ monitor
def _setup(keep_rule, person=(4.1, 5.9), n=1):
    g = Gatekeeper(cfg=SH.ShieldCfg(use_ir_veto=False), gk=GKConfig(keep_rule=keep_rule))
    sim = FakeSim((1.5, 5.0, 0.0), people=[person])
    g.gk_step(sim, np.array([0.9, 0, 0]), 0.0)
    assert g.mode == "commit" and g.traj is not None and g.traj.i_rest not in (0, None)
    return g, sim


def test_monitor_keeps_a_still_safe_plan_where_recertify_fails():
    """the observation age grows (latency 0.4 s: a wider re-anchored belief) after the commit, so the committed plan no longer passes Checker 1 under the CURRENT belief
    (every hold fails too), but the observation is consistent with the commit-time prediction -> the monitor keeps it, recertify falls into regime 2."""
    res = {}
    for kr in ("recertify", "monitor"):
        g, sim = _setup(kr, person=(3.4, 5.6))
        for _ in range(5):                                                   # follow the commit for a few ticks
            u = g.traj.u_at(0.1); sim.step(u)
        g.gk.latency_s = 0.4
        g.gk_step(sim, np.array([0.9, 0, 0]), 0.1)
        res[kr] = (g.mode, g.n_falsified, g.n_keep_backup, dict(g.falsify_reasons))
    assert res["recertify"][0] == "regime2" and res["recertify"][1] == 1
    assert res["monitor"][0] == "keep" and res["monitor"][1:3] == (0, 1)


def test_monitor_flags_observation_outside_predicted_set():
    g, sim = _setup("monitor", person=(6.0, 9.0))
    sim.people_state = [(np.array([sim.base.x + 1.2, sim.base.y]), np.zeros(2))]      # teleport in front of the moving robot
    g.gk_step(sim, np.array([0.9, 0, 0]), 0.1)
    assert g.mode == "regime2" and g.n_falsified == 1 and g.falsify_reasons == {"person_unmatched_near": 1}


def test_monitor_ignores_unmatched_person_that_cannot_reach_the_motion():
    from world_model.v2.house.gatekeeper.monitor import falsification_monitor
    g, sim = _setup("monitor", person=(6.0, 9.0))
    for _ in range(3):
        sim.step(g.traj.u_at(0.1))
    rem = g.traj.shifted(0.1)
    far = ([np.array([2.0, 9.5])], [0.0])                                              # jumped out of its predicted set, but 4.5 m from the path
    near = ([np.array([sim.base.x + 1.2, sim.base.y])], [0.0])
    assert falsification_monitor(g._cb[0], g._cb[1], g._cb[2], rem, g.last_bel, far, sim.pose(), 0.1, g.gk)[0]
    assert falsification_monitor(g._cb[0], g._cb[1], g._cb[2], rem, g.last_bel, near, sim.pose(), 0.1, g.gk) == (False, "person_unmatched_near")


def test_monitor_pose_falsification():
    g, sim = _setup("monitor", person=(6.0, 9.0))
    for _ in range(5):
        sim.step(g.traj.u_at(0.1))
    g.gk.latency_s = 0.0
    # off-plan pose: shift by 0.3 m
    sim.base.x += 0.3
    from world_model.v2.house.gatekeeper.monitor import falsification_monitor
    ok, why = falsification_monitor(g._cb[0], g._cb[1], g._cb[2], g.traj.shifted(0.1), g.last_bel, ([], []), sim.pose(), 0.1, g.gk)
    assert not ok and why == "pose_xy"


def test_monitor_end_to_end_no_false_alarm_when_assumptions_hold():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        g = Gatekeeper(gk=GKConfig(keep_rule="monitor", person_model="accel", cup_chord=True))
        sim = FakeSim((1.5, 5.0, 0.0), people=[(3.8, 7.0)])
        u_op = np.zeros(3)
        for k in range(500):
            if k % 25 == 0:
                u_op = rng.uniform(-1, 1, 3) * BS.V_STICK * rng.choice([0.0, 1.0])
            v, _ = g.filter_base(sim, u_op); g.n_calls += 1; sim.step(v)
        assert g.n_falsified == 0 and g.n_commit > 0, (seed, g.falsify_reasons)
        assert g.counters()["falsify_reasons"] == {}


# ------------------------------------------------------------------------------------------------ cup chord
def test_cup_chord_reduces_margin_by_the_bound_and_is_sound():
    b = bel()
    V = np.array([[0.0, 0.0, 0.0], [0.07, 0.03, 0.10]])                      # one segment, big alpha and a_b
    from world_model.v2.house.gatekeeper.traj import Traj
    tr = Traj(V, 0.05)
    off = check1(tr, b, GK)[3]["cup"]
    on = check1(tr, b, dataclasses.replace(GK, cup_chord=True))[3]["cup"]
    bound = float(acup_chord(V, 0.05, R_CUP)[0])
    A = (V[1] - V[0]) / 0.05
    assert bound == pytest.approx(0.05 ** 2 / 8 * (2 * abs(A[2]) * math.hypot(A[0], A[1]) + 2 * A[2] ** 2 * math.hypot(*R_CUP))) and bound > 0
    assert off - on == pytest.approx(bound, rel=1e-9)
    # soundness of the bound: max over t in the segment <= max(end values) + bound, random segments
    rng = np.random.default_rng(3)
    for _ in range(300):
        V0 = rng.uniform(-1, 1, 3) * BS.V_STICK; V1 = V0 + rng.uniform(-1, 1, 3) * BS.A_HW * 0.05
        Vs = np.array([V0, V1]); h = 0.05; a = (V1 - V0) / h
        ts = np.linspace(0, h, 101)
        mx = max(math.hypot(*acup(V0 + a * t, a, R_CUP)) for t in ts)
        s = acup_series(Vs, h, R_CUP)
        assert mx <= max(s[0], s[1]) + acup_chord(Vs, h, R_CUP)[0] + 1e-9
    # default is untouched
    assert check1(tr, b, GK)[3]["cup"] == off
