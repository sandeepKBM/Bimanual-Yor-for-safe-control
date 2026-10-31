"""tracker people source of the gatekeeper: belief mapping from tracks, per-person monotonicity, assumption audit, sweep configs.  SIMULATION ONLY."""
import math
import types

import numpy as np
import pytest

from world_model.v2.house.teleop import base as BS, geom as GM
from world_model.v2.house.gatekeeper import GKConfig, Belief, Gatekeeper, plan_candidate, check1, check2, build_belief
from world_model.v2.house.gatekeeper.tests.test_gatekeeper import FakeSim, GRID, R_CUP, bel, admit, GK
from world_model.v2.house.people_track.conformal import ConformalMargin


def _ti(xy, v=(0.0, 0.0), sp=0.05, sv=0.2, since=0.0, age=5.0, rng=2.0, name="trk0"):
    P = np.diag([sp ** 2, sp ** 2, sv ** 2, sv ** 2]).astype(float)
    return dict(name=name, xy=np.array(xy, float), v=np.array(v, float), P=P, since_upd_s=since, age_s=age, range_m=rng)


class FakePerc:
    def __init__(self, tis):
        self.tis = tis
        self.n_steps = 0

    def step(self, sim):
        self.n_steps += 1

    def proxy(self, sim):
        return types.SimpleNamespace(track_info=list(self.tis))


def make_gk(tis, truth=(), **kw):
    sim = FakeSim((1.5, 5.0, 0.0), people=truth)
    g = Gatekeeper(gk=GKConfig(people_source="tracker", **kw), perc=FakePerc(tis))
    return g, sim


def test_defaults_are_truth_and_unchanged():
    gk = GKConfig()
    assert gk.people_source == "truth" and gk.conformal is None and gk.k_sigma == 2.0
    g = Gatekeeper()
    assert not g.tracker and "n_belief_miss" not in g.counters()
    b = Belief()
    assert b.people_e_p == () and b.people_e_v == () and b.people_stale == ()
    from world_model.v2.house.gatekeeper import sweep
    assert sweep.DEFAULT_CONFIGS == ("shield_brake", "shield_escape", "gk_speed_hold", "gk_accel_hold")
    assert sweep.CONFIGS["gk_speed_hold"] == dict(person_model="speed", candidates="hold") and sweep.CONFIGS["gk_accel_hold"] == dict(person_model="accel", candidates="hold")
    assert sweep.CONFIGS["shield_brake"] is None and sweep.CONFIGS["shield_escape"] is None


def test_truth_belief_margins_identical_with_and_without_empty_per_person_fields():
    rng = np.random.default_rng(1)
    for model in ("speed", "accel"):
        for _ in range(40):
            pose = np.array([rng.uniform(1.0, 4.4), rng.uniform(1.0, 9.0), rng.uniform(-3, 3)])
            ppl = tuple(pose[:2] + rng.uniform(-3, 3, 2) for _ in range(2))
            kw = dict(pose_wall=pose, pose_people=pose, people=ppl, e_p=0.03, latency_s=0.05, person_model=model, people_vel=(np.array([.3, 0]), np.array([0, -.4])))
            b0 = bel(**kw)
            b1 = bel(people_e_p=(0.0, 0.0), people_e_v=(0.0, 0.0), people_stale=(0.0, 0.0), **kw)       # explicit zeros == no per-person fields (numerically)
            tr = plan_candidate(rng.uniform(-1, 1, 3) * BS.V_STICK, rng.uniform(-1, 1, 3) * BS.V_STICK, 0.6, b0, GK)
            assert check1(tr, b0, GK)[1] == check1(tr, b1, GK)[1] and check2(tr, b0, GK)[1] == check2(tr, b1, GK)[1]


def test_belief_mapping_from_tracks_ksigma():
    ti = _ti((3.0, 5.0), v=(-0.4, 0.1), sp=0.05, sv=0.2, since=0.12)
    g, sim = make_gk([ti], truth=[(3.0, 5.0)], person_model="accel", k_sigma=2.0)
    g.perceive(sim)
    trk = g._track_list()
    assert trk[0]["e_p"] == pytest.approx(2.0 * 0.05) and trk[0]["e_v"] == pytest.approx(2.0 * 0.2) and trk[0]["stale"] == pytest.approx(0.12)
    b = build_belief(sim, g.cfg, g.gk, fp=g._shield(sim).footprint_of(sim), tracks=trk)
    assert len(b.people) == 1 and np.allclose(b.people[0], [3.0, 5.0]) and np.allclose(b.people_vel[0], [-0.4, 0.1])
    assert b.people_e_p == (pytest.approx(0.1),) and b.people_e_v == (pytest.approx(0.4),) and b.people_stale == (pytest.approx(0.12),)
    assert b.e_p == 0.0 and b.e_v == 0.0 and b.latency_s == 0.0 and b.person_model == "accel"
    # anisotropic P: lambda_max is used
    P = np.diag([0.01 ** 2, 0.09 ** 2, 0.1 ** 2, 0.3 ** 2])
    ti2 = dict(ti, P=P)
    g2, _ = make_gk([ti2])
    g2.perceive(sim)
    t2 = g2._track_list()[0]
    assert t2["e_p"] == pytest.approx(2 * 0.09) and t2["e_v"] == pytest.approx(2 * 0.3)


def test_belief_mapping_from_tracks_conformal_radius():
    cm = ConformalMargin(alpha=0.05, n_min=10, cap=0.6, glob=(100, 0.31))        # every bin falls back to the global quantile
    ti = _ti((3.0, 5.0), since=0.3, age=1.0, rng=1.2)
    g, sim = make_gk([ti], conformal=cm)
    g.perceive(sim)
    t = g._track_list()[0]
    assert t["e_p"] == pytest.approx(0.31) and t["e_v"] == pytest.approx(2 * 0.2) and t["stale"] == pytest.approx(0.3)     # e_v is k-sigma even with conformal e_p


def _rand_track_case(rng, model):
    pose = np.array([rng.uniform(1.0, 4.4), rng.uniform(1.0, 9.0), rng.uniform(-math.pi, math.pi)])
    n = int(rng.integers(1, 3))
    ppl = tuple(pose[:2] + rng.uniform(-3, 3, 2) for _ in range(n))
    b = bel(pose_wall=pose, pose_people=pose, people=ppl, person_model=model, people_vel=tuple(rng.uniform(-1, 1, 2) for _ in range(n)), e_v=0.0,
            people_e_p=tuple(rng.uniform(0, 0.15, n)), people_e_v=tuple(rng.uniform(0, 0.4, n)), people_stale=tuple(rng.uniform(0, 0.3, n)),
            v_p_max=rng.uniform(0.3, 1.2), mu_lo=rng.uniform(0.2, 0.35))
    tr = plan_candidate(rng.uniform(-1, 1, 3) * BS.V_STICK, rng.uniform(-1, 1, 3) * BS.V_STICK, rng.choice([1.0, 0.6, 0.3, 0.1]), b, GK)
    return b, tr


@pytest.mark.parametrize("model", ["speed", "accel"])
@pytest.mark.parametrize("field", ["people_e_p", "people_e_v", "people_stale"])
def test_monotonicity_per_track_fields_never_admit(model, field):
    rng = np.random.default_rng(11)
    n_adm = 0
    for _ in range(150):
        b, tr = _rand_track_case(rng, model)
        i = int(rng.integers(0, len(b.people)))
        cur = list(getattr(b, field))
        cur[i] += {"people_e_p": 0.06, "people_e_v": 0.3, "people_stale": 0.2}[field]
        b2 = b.with_(**{field: tuple(cur)})
        a1, a2 = admit(tr, b), admit(tr, b2)
        n_adm += a1
        assert not (a2 and not a1), f"{model}/{field}: enlarging the track belief turned a reject into an admit"
    assert n_adm > 5


def test_stale_track_grows_only_its_own_disc():
    b = bel(people=(np.array([3.4, 5.0]), np.array([1.5, 8.5])), people_e_p=(0.05, 0.05), people_e_v=(0.2, 0.2), people_stale=(0.0, 0.0), person_model="accel", people_vel=(np.zeros(2),) * 2, e_v=0.0)
    tr = plan_candidate(np.array([0.3, 0, 0]), np.array([0.5, 0, 0]), 0.6, b, GK)
    m0 = check1(tr, b, GK)[3]["person"]
    m_far = check1(tr, b.with_(people_stale=(0.0, 1.0)), GK)[3]["person"]
    m_near = check1(tr, b.with_(people_stale=(1.0, 0.0)), GK)[3]["person"]
    assert m_near <= m0 and m_far <= m0
    assert m_near < m0 - 1e-6                                          # the nearer track binds -> its staleness shrinks the margin
    assert m_far == pytest.approx(m0)


def test_belief_ep_ev_positive_and_decisions_use_tracks_not_truth():
    """the truth person is far away, the TRACK is right in front of the robot: the gatekeeper must treat it as the hazard (no truth in decisions)."""
    ti = _ti((2.6, 5.0), sp=0.06, sv=0.2)
    g, sim = make_gk([ti], truth=[(1.5, 9.0)], person_model="accel")
    g.perceive(sim)
    for k in range(200):
        g.perceive(sim)
        u, lab = g.filter_base(sim, np.array([0.9, 0, 0])); g.n_calls += 1; sim.step(u)
    b = g.last_bel
    assert b.people_e_p[0] > 0 and b.people_e_v[0] > 0
    assert sim.base.x < 1.5 + 0.2                                      # never advanced toward the (phantom) track at 2.6 m
    # same run with the track far away: robot moves freely
    g2, sim2 = make_gk([_ti((3.0, 9.5))], truth=[(2.6, 5.0)], person_model="accel")
    for k in range(200):
        g2.perceive(sim2)
        u, lab = g2.filter_base(sim2, np.array([0.9, 0, 0])); g2.n_calls += 1; sim2.step(u)
    assert sim2.base.x > 1.5 + 0.8                                     # the TRUE person at 2.6 m is invisible to the belief (open assumption) ...
    assert g2.n_untracked_near > 0                                     # ... and the audit counts it


def test_assumption_audit_counters():
    ti = _ti((3.0, 5.0), sp=0.05)                                      # e_p = 0.1
    # (a) true person inside e_p of the track -> no miss
    g, sim = make_gk([ti], truth=[(3.05, 5.05)])
    g.perceive(sim); g.gk_step(sim, np.zeros(3), 0.0)
    assert (g.n_audit_person_steps, g.n_audit_matched, g.n_belief_miss, g.n_untracked_near) == (1, 1, 0, 0)
    # (b) 0.35 m off (> e_p = 0.1, < match radius 1 m) -> belief miss, excess 0.25
    g, sim = make_gk([ti], truth=[(3.35, 5.0)])
    g.perceive(sim); g.gk_step(sim, np.zeros(3), 0.0)
    assert (g.n_belief_miss, g.n_untracked_near) == (1, 0) and g.max_miss_excess_m == pytest.approx(0.25)
    # (c) no track near a true person within 3 m of the robot -> untracked_near; far away (> 3 m from the footprint) -> not counted
    g, sim = make_gk([ti], truth=[(1.5, 7.0), (1.5, 9.8)])
    g.perceive(sim); g.gk_step(sim, np.zeros(3), 0.0)
    assert (g.n_audit_person_steps, g.n_belief_miss, g.n_untracked_near) == (2, 0, 1)
    # (d) no tracks at all
    g, sim = make_gk([], truth=[(2.5, 5.0)])
    g.perceive(sim); g.gk_step(sim, np.zeros(3), 0.0)
    assert g.n_untracked_near == 1 and len(g.last_bel.people) == 0
    c = g.counters()
    assert c["n_untracked_near"] == 1 and c["n_belief_miss"] == 0 and c["n_audit_steps"] == 1
    # truth people never read when the belief is built (the belief people are the tracks)
    g, sim = make_gk([ti], truth=[(1.5, 9.0)])
    g.perceive(sim); g.gk_step(sim, np.zeros(3), 0.0)
    assert len(g.last_bel.people) == 1 and np.allclose(g.last_bel.people[0], [3.0, 5.0])


def test_cpu_perception_conditions_only():
    sim = FakeSim((1.5, 5.0, 0.0))
    g = Gatekeeper(gk=GKConfig(people_source="tracker", perception="neural"))
    with pytest.raises(AssertionError):
        g._ensure_perc(sim)


# ------------------------------------------------------------------------------------------ new sweep configs: one short real episode each (CPU)
@pytest.mark.parametrize("name", ["shield_brake_trk", "shield_escape_trk", "gk_accel_hold_trk"])
def test_sweep_trk_configs_run_one_short_episode(name):
    from world_model.v2.house.gatekeeper import sweep
    r = sweep._job((name, 0, 2.0, "carry", "h5"))
    assert r["error"] is None, r
    assert r["config"] == name and r.get("min_person_distance_m") is not None
    if name.startswith("gk"):
        g = r["gk"]
        assert g["n_audit_steps"] > 0 and g["n_belief_miss"] >= 0 and g["n_untracked_near"] >= 0 and g["n_gk_steps"] > 0
    S = sweep.summarize([r])
    assert name in S
    if name.startswith("gk"):
        assert "n_belief_miss" in S[name] and "n_untracked_near" in S[name]
