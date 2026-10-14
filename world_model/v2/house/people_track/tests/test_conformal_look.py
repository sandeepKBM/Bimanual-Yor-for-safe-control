"""people_track: conformal margin, look-before-move glance, people-source switch (unit level, no episodes)."""
import math

import numpy as np
import pytest

from world_model.v2.house.people_track.conformal import ConformalMargin, cq, bin_of


def test_cq_is_split_conformal_quantile():
    s = np.arange(1, 100)                                  # n = 99
    assert cq(s, 0.1) == 90                               # ceil(100 * 0.9) = 90th smallest
    assert cq(s[:5], 0.05) == float("inf")                # too few samples for 95%


def test_fit_bins_fallback_and_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    def draw(n):                                           # score grows when the track is coasting
        su = rng.choice([0.0, 0.3, 1.5], n)
        ag = rng.uniform(0, 4, n)
        e = np.abs(rng.normal(0, 0.05 + 0.1 * (su > 0.1) + 0.2 * (su > 1.0), n))
        return np.stack([su, ag, e], 1)
    m = ConformalMargin.fit(draw(6000), alpha=0.1, cap=5.0)
    cov = m.coverage(draw(6000))
    assert abs(cov["overall"] - 0.9) < 0.03
    assert m.radius(1.5, 3.0) > m.radius(0.0, 3.0)         # coasting track gets the wider margin
    # empty cell -> falls back to the visibility marginal, then global
    m2 = ConformalMargin.fit(np.array([[0.0, 3.0, 0.1]] * 100), alpha=0.1)
    assert m2.radius(1.0, 0.1) == pytest.approx(m2.glob[1])
    # json round trip
    m3 = ConformalMargin.from_json(m.to_json())
    assert m3.radius(0.3, 1.0) == m.radius(0.3, 1.0)
    assert bin_of(0.0, 0.0) == (0, 0) and bin_of(1.0, 5.0) == (2, 2)


def test_cap_applies():
    m = ConformalMargin.fit(np.array([[0.0, 3.0, 2.0]] * 100), alpha=0.1, cap=0.6)
    assert m.radius(0.0, 3.0) == 0.6


@pytest.fixture(scope="module")
def simenv():
    from world_model.v2.house.people_track.sim_fix import TeleopSimWalking
    from world_model.v2.house.people_track import perception as PC
    sim = TeleopSimWalking("h5", grasped=True, seed=0)
    perc = PC.PerceptionStack(sim, PC.PerceptionCfg(detector="param"))
    return sim, perc


def test_conformal_inflation_replaces_ksigma_in_shield_rows(simenv):
    from world_model.v2.house.people_track.perception import SimProxy
    from world_model.v2.house.people_track.shield_pt import PerceivedShield, UncertaintyCfg
    from world_model.v2.house.people_track.tests.test_people_track import _FakeTrack
    sim, perc = simenv
    pose = sim.pose()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    xy = tuple(pose[:2] + np.array([c, s]) * 1.2)
    cm = ConformalMargin.fit(np.array([[0.0, 3.0, 0.12]] * 100 + [[2.0, 3.0, 0.40]] * 100), alpha=0.1, cap=0.6)
    sh = PerceivedShield(perc, None, occ=sim.occ, a_hw=sim.a_hw, ucfg=UncertaintyCfg(unseen_cap=False, conformal=cm))
    P = np.diag([4.0, 4.0, 1.0, 1.0])                      # huge covariance: k-sigma would saturate at 0.5; conformal ignores it
    t = _FakeTrack(0, xy, (0, 0), P)
    t.last_upd, t.born = sim.k, sim.k - 200                # just updated, old track
    sh._keep_rows(SimProxy(sim, [t]), sim.base.v)
    assert sh.last_infl == pytest.approx(0.12)
    t.last_upd = sim.k - 100                               # coasting 2 s
    sh._keep_rows(SimProxy(sim, [t]), sim.base.v)
    assert sh.last_infl == pytest.approx(0.40)


def test_looker_glance_logic(simenv):
    from world_model.v2.house.people_track import perception as PC
    from world_model.v2.house.people_track.look import Looker, LookCfg, wrap
    from world_model.v2.house.teleop.sim import Cmd
    sim, _ = simenv
    perc = PC.PerceptionStack(sim, PC.PerceptionCfg(detector="param"))
    L = Looker(perc, LookCfg(lost_track=False))
    o, yaw = L._cam()
    # a track just seen at a bearing 70 deg to the left of the camera axis, then lost: trigger (b) yields a left glance
    L2 = Looker(perc, LookCfg(corridor=False))
    class T:                                               # a coasting track at 70 deg left, 2 m away
        pass
    t = T(); t.last_upd = sim.k - 50
    ang = yaw + math.radians(70)
    t.x = np.array([o[0] + 2.0 * math.cos(ang), o[1] + 2.0 * math.sin(ang), 0.0, 0.0])
    perc.tracks = [t]
    cmd = Cmd(twist=np.array([0.4, 0.0, 0.0]))
    out = L2.apply(sim, cmd)
    assert L2.active and out.twist[2] > 0 and out.twist[0] == 0.0 and L2.n_trig_lost == 1
    # a track BEHIND the robot (beyond max_glance_deg): no glance (speed cap only)
    L3 = Looker(perc, LookCfg(corridor=False))
    ang = yaw + math.radians(170)
    t.x = np.array([o[0] + 2.0 * math.cos(ang), o[1] + 2.0 * math.sin(ang), 0.0, 0.0])
    assert L3.apply(sim, cmd) is cmd and not L3.active
    # a track already inside the FOV: no glance
    L4 = Looker(perc, LookCfg(corridor=False))
    t.x = np.array([o[0] + 2.0 * math.cos(yaw), o[1] + 2.0 * math.sin(yaw), 0.0, 0.0])
    assert L4.apply(sim, cmd) is cmd and not L4.active
    perc.tracks = []


def test_people_source_default_is_truth_shield(simenv):
    from world_model.v2.house.teleop import shield as SH
    sim, _ = simenv
    assert sim.people_source == "truth"
    sh = sim.make_shield()
    assert type(sh) is SH.Shield
    from world_model.v2.house.people_track.shield_pt import PerceivedShield
    sim.people_source = "tracker"
    try:
        sh2 = sim.make_shield(perception="param:0.1:3:0.0", look=True)
        assert isinstance(sh2, PerceivedShield) and sh2.look is not None
    finally:
        sim.people_source = "truth"


def test_soft_floor_never_creates_an_instantaneous_violation(simenv):
    from world_model.v2.house.people_track.perception import SimProxy
    from world_model.v2.house.people_track.shield_pt import PerceivedShield, UncertaintyCfg
    from world_model.v2.house.people_track.tests.test_people_track import _FakeTrack
    from world_model.v2.house.teleop import geom as GM
    sim, perc = simenv
    pose = sim.pose()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    xy = tuple(pose[:2] + np.array([c, s]) * 1.0)
    dist, _, _ = GM.rect_dist_point(np.array(xy), pose)
    P = np.diag([4.0, 4.0, 1.0, 1.0])
    for floor in (None, 0.05):
        sh = PerceivedShield(perc, None, occ=sim.occ, a_hw=sim.a_hw, ucfg=UncertaintyCfg(unseen_cap=False, soft_floor=floor))
        sh._keep_rows(SimProxy(sim, [_FakeTrack(0, xy, (0, 0), P)]), sim.base.v)
        if floor is None:
            assert sh.last_infl == pytest.approx(0.5) and dist - (GM.PERSON_CLEAR_R + sh.last_infl) < 0      # k-sigma at its cap violates the margin
        else:
            assert sh.last_infl == pytest.approx(max(0.0, dist - GM.PERSON_CLEAR_R - floor)) and dist - (GM.PERSON_CLEAR_R + sh.last_infl) >= floor - 1e-9
