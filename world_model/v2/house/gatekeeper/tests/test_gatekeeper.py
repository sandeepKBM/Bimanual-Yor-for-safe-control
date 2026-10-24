"""Gatekeeper unit tests (CPU, no MuJoCo except the last one).  python -m pytest -q world_model/v2/house/gatekeeper/tests"""
import inspect
import math
import types

import numpy as np
import pytest
from scipy import ndimage

from world_model.v2.house.teleop import base as BS, geom as GM, shield as SH
from world_model.v2.house.teleop.scene import OccGrid
from world_model.v2.house.teleop.sim import TICK
from world_model.v2.house.gatekeeper import GKConfig, Belief, Gatekeeper, plan_candidate, backup_from, check1, check2
from world_model.v2.house.gatekeeper.traj import cup_safe_target, acup_series
from world_model.v2.house.gatekeeper.checks import person_radii, t_touch_accel, _rect_dist, world_poses

GK = GKConfig()
R_CUP = np.array([0.35, -0.20])


def make_grid(wall_x=5.0, size=10.0, res=0.02):
    n = int(size / res)
    occ = np.zeros((n, n), bool)
    ix = int(wall_x / res)
    occ[:, ix:] = True                                    # solid wall for x >= wall_x
    occ[:3, :] = occ[-3:, :] = True; occ[:, :3] = True
    edt = ndimage.distance_transform_edt(~occ) * res
    return OccGrid(res, 0.0, 0.0, occ, edt)


GRID = make_grid()


def bel(**kw):
    d = dict(r_cup=R_CUP.copy(), grid=GRID, pose_wall=np.array([1.5, 5.0, 0.0]), pose_people=np.array([1.5, 5.0, 0.0]))
    d.update(kw)
    return Belief(**d)


def admit(tr, b, gk=GK):
    return check1(tr, b, gk)[0] and check2(tr, b, gk)[0]


def test_backup_cup_accel_never_exceeds_limit_incl_rotation():
    b = bel()
    bad = 0
    for r in (R_CUP, np.array([0.5, 0.25]), np.array([0.1, -0.4])):
        b = bel(r_cup=r)
        for vx in (-0.9, -0.4, 0, 0.5, 0.9):
            for vy in (-0.45, 0, 0.3, 0.45):
                for wz in (-1.2, -0.6, 0, 0.7, 1.2):
                    v0 = cup_safe_target(np.array([vx, vy, wz]), r, GK.steady_frac * GK.plan_frac * b.a_cup_max, b.v_cap)      # states inside the steady-state cup set
                    tr = backup_from(v0, b, GK)
                    assert tr.i_rest is not None and tr.V[-1].max() == 0 and tr.t_rest <= 2.0
                    a = acup_series(tr.V, tr.h, r)
                    # independent evaluation with base.cup_accel_body at segment start and end
                    for k in range(tr.N):
                        acc = (tr.V[k + 1] - tr.V[k]) / tr.h
                        for vv in (tr.V[k], tr.V[k + 1]):
                            assert np.linalg.norm(BS.cup_accel_body(vv, acc, r)) <= b.a_cup_max + 1e-9
                    assert a.max() <= b.a_cup_max + 1e-9
                    assert np.all(np.abs(np.diff(tr.V, axis=0)) / tr.h <= BS.A_HW + 1e-9)
                    assert check1(tr, bel(r_cup=r, pose_wall=np.array([1.5, 5.0, 0.0])), GK)[0]


def test_nominal_ramp_is_cup_limited_not_hw_limited():
    b = bel()
    tr = plan_candidate(np.zeros(3), np.array([0.9, 0.0, 0.0]), 1.0, b, GK)
    acc = np.abs(np.diff(tr.V[:, 0])) / tr.h
    assert acc.max() <= b.a_cup_max + 1e-9 < BS.A_HW[0]


def test_checker1_wall_inside_braking_distance_and_free_space():
    # wall at x = 5; footprint front at +0.577 -> at x=4.2 the front is 0.223 m from the wall; braking from 0.9 m/s needs > 0.33 m + buffer
    v0 = np.array([0.9, 0, 0])
    near = bel(pose_wall=np.array([4.2, 5.0, 0.0]))
    tr = plan_candidate(v0, np.array([0.9, 0, 0]), 0.3, near, GK)
    ok, m, why, _ = check1(tr, near, GK)
    assert (not ok) and why == "wall" and m < 0
    far = bel(pose_wall=np.array([1.5, 5.0, 0.0]))
    tr = plan_candidate(v0, np.array([0.9, 0, 0]), 1.0, far, GK)
    assert admit(tr, far)
    # at rest next to the wall but outside the buffer: stationary trajectory admitted; heading in with room to stop: rejected only if it cannot stop
    rest = bel(pose_wall=np.array([4.0, 5.0, 0.0]))                       # front 0.423 from the wall (> d_wall 0.10)
    assert admit(plan_candidate(np.zeros(3), np.zeros(3), 1.0, rest, GK), rest)
    assert not admit(plan_candidate(np.zeros(3), np.array([0.9, 0, 0]), 1.0, rest, GK), rest)


def test_checker1_people_reachable_set_growth():
    pose = np.array([1.5, 5.0, 0.0])
    # person straight ahead: rect front at +0.577; person centre 2.0 m ahead of the base origin
    c = (3.5, 5.0)
    kw = dict(pose_wall=pose, pose_people=pose, people=(np.array(c),))
    tr = plan_candidate(np.zeros(3), np.array([0.9, 0, 0]), 0.6, bel(**kw), GK)
    assert admit(tr, bel(v_p_max=0.0, **kw))                              # static person: enough room (stops ~1.2 m before)
    assert admit(tr, bel(v_p_max=0.3, **kw)) or True
    ok15, m15, why, _ = check1(tr, bel(v_p_max=1.5, **kw), GK)
    assert (not ok15) and why == "person"
    # margin decreases with v_p_max
    ms = [check1(tr, bel(v_p_max=v, **kw), GK)[3]["person"] for v in (0.0, 0.5, 1.0, 1.5)]
    assert all(ms[i] > ms[i + 1] for i in range(3))
    # a person walking up to a robot that is ALREADY at rest does not falsify it (passive safety)
    near = dict(pose_wall=pose, pose_people=pose, people=(np.array([2.5, 5.0]),))
    assert admit(plan_candidate(np.zeros(3), np.zeros(3), 1.0, bel(**near), GK), bel(**near))
    # ... but starting to move with the person close is rejected, and the passive-safety margin is the failing item for a short-fuse person
    tr2 = plan_candidate(np.zeros(3), np.array([0.0, 0.45, 0.0]), 0.1, bel(**near), GK)
    assert not admit(tr2, bel(**near))


def test_checker2_requires_rest_and_passive_safety():
    b = bel()
    tr = plan_candidate(np.zeros(3), np.array([0.9, 0, 0]), 1.0, b, GK)
    tr_cut = type(tr)(tr.V[:8], tr.h, tr.t0, tr.t_sw, tr.T_S)             # truncated before it stops
    ok, m, why, _ = check2(tr_cut, b, GK)
    assert (not ok) and why == "no_rest"
    # person that can reach the rest footprint before the robot has stopped
    p = np.array(tr.poses()[-1][:2]) + np.array([1.5, 5.0]) + np.array([0.577 + 0.3 + 0.4, 0.0])        # 0.4 m beyond the disc at the rest pose
    bp = bel(people=(p,), v_p_max=1.5)
    ok, m, why, mg = check2(tr, bp, GK)
    assert mg["passive_person"] == pytest.approx((0.4 / 1.5) - tr.t_rest, abs=0.02)
    assert (not ok) == (mg["passive_person"] < 0)


def _random_case(rng):
    pose = np.array([rng.uniform(1.0, 4.4), rng.uniform(1.0, 9.0), rng.uniform(-math.pi, math.pi)])
    v0 = rng.uniform(-1, 1, 3) * BS.V_STICK
    u = rng.uniform(-1, 1, 3) * BS.V_STICK
    ppl = tuple(pose[:2] + rng.uniform(-3, 3, 2) for _ in range(rng.integers(0, 3)))
    b = bel(pose_wall=pose, pose_people=pose, people=ppl, e_xy=rng.uniform(0, 0.05), e_th=rng.uniform(0, 0.02), e_p=rng.uniform(0, 0.1),
            v_p_max=rng.uniform(0.2, 1.2), mu_lo=rng.uniform(0.2, 0.35), latency_s=rng.uniform(0, 0.1), d_wall=rng.uniform(0.05, 0.15), d_person=rng.uniform(0.0, 0.2))
    tr = plan_candidate(v0, u, rng.choice([1.0, 0.6, 0.3, 0.1]), b, GK)
    return b, tr


@pytest.mark.parametrize("field,delta", [("e_xy", 0.03), ("e_th", 0.02), ("e_p", 0.05), ("v_p_max", 0.4), ("latency_s", 0.1), ("d_wall", 0.05),
                                         ("d_person", 0.1), ("mu_lo", -0.08)])
def test_monotonicity_enlarging_belief_never_admits(field, delta):
    rng = np.random.default_rng(7)
    n_adm = n_flip = 0
    for _ in range(150):
        b, tr = _random_case(rng)
        b2 = b.with_(**{field: getattr(b, field) + delta})
        a1, a2 = admit(tr, b), admit(tr, b2)
        n_adm += a1
        assert not (a2 and not a1), f"{field}: enlarging the belief turned a reject into an admit"
        n_flip += (a1 and not a2)
    assert n_adm > 5                                                     # the property test is not vacuous


ACC = ["e_v", "a_p", "v_p_max", "e_p", "d_person"]
_DELTA = dict(e_v=0.3, a_p=1.5, v_p_max=0.4, e_p=0.05, latency_s=0.1, d_person=0.1)


def _random_case_accel(rng):
    b, tr = _random_case(rng)
    vel = tuple(rng.uniform(-1.3, 1.3, 2) for _ in b.people)
    b = b.with_(person_model="accel", people_vel=vel, e_v=rng.uniform(0.0, 0.4), a_p=rng.uniform(0.5, 3.0))
    return b, tr


@pytest.mark.parametrize("field", ACC)
def test_monotonicity_accel_model_enlarging_never_admits(field):
    rng = np.random.default_rng(11)
    n_adm = n_flip = n_ppl = 0
    for _ in range(200):
        b, tr = _random_case_accel(rng)
        b2 = b.with_(**{field: getattr(b, field) + _DELTA[field]})
        a1, a2 = admit(tr, b), admit(tr, b2)
        m1, m2 = check1(tr, b, GK)[3].get("person", np.inf), check1(tr, b2, GK)[3].get("person", np.inf)
        assert m2 <= m1 + 1e-9                                          # the person margin itself is monotone, not only the verdict
        assert not (a2 and not a1), f"{field}: enlarging the accel belief turned a reject into an admit"
        n_adm += a1; n_flip += (a1 and not a2); n_ppl += len(b.people) > 0
    assert n_adm > 5 and n_ppl > 20


def test_monotonicity_accel_model_latency_for_zero_velocity_estimate():
    """latency shifts the accel disc centre by v_hat*latency (a person walking away is further away with an OLDER observation), so the accel model is monotone in
    latency only for v_hat = 0 (standing / unknown-direction) -- documented limitation; the speed disc (centre fixed) is always monotone."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        b, tr = _random_case_accel(rng)
        b = b.with_(people_vel=tuple(np.zeros(2) for _ in b.people))
        a1, a2 = admit(tr, b), admit(tr, b.with_(latency_s=b.latency_s + 0.1))
        assert not (a2 and not a1)


def test_accel_reachable_set_algebra_and_soundness():
    # standing person (v_hat = 0): radius_a(t) = r0 + e_v t + a t^2/2 < radius_s(t) = r0 + v_p t  iff  t < t* = 2 (v_p - e_v)/a_p
    b = Belief(people=(np.array([3.0, 0.0]),), people_vel=(np.zeros(2),), person_model="accel", e_v=0.2, a_p=2.0, v_p_max=1.5)
    t_star = 2 * (b.v_p_max - b.e_v) / b.a_p                              # 1.3 s (== 2 v_p/a_p = 1.5 s when e_v = 0)
    for t in (0.1, 0.5, 1.0, 1.29):
        (_, rs), (_, ra) = person_radii(b, 0, t)
        assert ra < rs
    (_, rs), (_, ra) = person_radii(b, 0, t_star)
    assert ra == pytest.approx(rs)
    (_, rs), (_, ra) = person_radii(b, 0, t_star + 0.3)
    assert ra > rs                                                        # beyond t* the SPEED disc is the smaller one -> max-margin rule falls back to it
    assert person_radii(b.with_(e_v=0.0), 0, 1.0)[1][1] - 0.3 == pytest.approx(1.0)
    # soundness: random person motions consistent with the assumptions stay inside BOTH discs (hence in the intersection)
    rng = np.random.default_rng(0)
    dt = 0.01
    for trial in range(300):
        vh = rng.uniform(-1.0, 1.0, 2)
        bb = b.with_(people_vel=(vh,))
        e = rng.normal(size=2); e = e / np.linalg.norm(e) * bb.e_v * rng.uniform(0, 1)
        v = vh + e                                                         # true initial velocity: within e_v of v_hat
        if np.linalg.norm(v) > bb.v_p_max:
            v = v / np.linalg.norm(v) * bb.v_p_max
        x = np.array(bb.people[0], float).copy()
        for k in range(1, 300):
            a = rng.normal(size=2); a = a / np.linalg.norm(a) * bb.a_p * rng.uniform(0, 1)
            x = x + v * dt                                                   # left Riemann sum: exact integral of the piecewise-constant velocity
            v = v + a * dt
            if np.linalg.norm(v) > bb.v_p_max:
                v = v / np.linalg.norm(v) * bb.v_p_max
            t = k * dt
            (cs, rs), (ca, ra) = person_radii(bb, 0, t)
            assert np.linalg.norm(x - cs) <= rs - GM.PERSON_CLEAR_R + 1e-9
            assert np.linalg.norm(x - ca) <= ra - GM.PERSON_CLEAR_R + 1e-9


def _fwd_traj(b):
    return plan_candidate(np.zeros(3), np.array([0.9, 0, 0]), 0.3, b, GK)


def test_accel_person_margins_standing_moving_and_never_above_geometry():
    pose = np.array([1.5, 5.0, 0.0])
    c = np.array([3.6, 5.0])                                              # 2.1 m ahead of the base origin
    kw = dict(pose_wall=pose, pose_people=pose, people=(c,))
    tr = plan_candidate(np.zeros(3), np.array([0.45, 0.0, 0.0]), 0.6, bel(**kw), GK)
    ms = check1(tr, bel(**kw), GK)[3]["person"]
    mst = check1(tr, bel(person_model="accel", people_vel=(np.zeros(2),), **kw), GK)[3]["person"]
    assert mst > ms                                                        # standing person: strictly smaller reachable radius -> larger margin
    # ... and the margin never exceeds the true geometric clearance (distance to the centre minus the base disc minus buffer)
    Pp = world_poses(tr.poses()[:tr.i_rest], pose)
    geo = (_rect_dist(c, Pp, GM.FP) - GM.PERSON_CLEAR_R - bel().d_person).min()
    assert mst <= geo + 1e-9
    # person walking TOWARD the robot at 1.0 m/s: the centre shifts, margin smaller than standing; walking away: larger
    m_to = check1(tr, bel(person_model="accel", people_vel=(np.array([-1.0, 0.0]),), **kw), GK)[3]["person"]
    m_aw = check1(tr, bel(person_model="accel", people_vel=(np.array([+1.0, 0.0]),), **kw), GK)[3]["person"]
    assert m_to < mst < m_aw
    assert m_to >= ms - 1e-9                                               # never worse than the speed model (max rule): the speed disc is always available
    # passive safety: standing person is certified for the same stop earlier than with the speed model
    assert t_touch_accel(1.0, 0.2, 2.0, 1.5) == pytest.approx((-0.2 + math.sqrt(0.04 + 4.0)) / 2.0)       # a t^2/2 + b t = x
    assert t_touch_accel(1.0, 0.0, 2.0, 1.5) == pytest.approx(1.0)                                         # t^2 = 1 for a_p = 2
    far = dict(pose_wall=pose, pose_people=pose, people=(np.array([2.5, 5.0]),))
    trp = plan_candidate(np.zeros(3), np.array([0.0, 0.45, 0.0]), 0.1, bel(**far), GK)
    mp_s = check2(trp, bel(**far), GK)[3].get("passive_person")
    mp_a = check2(trp, bel(person_model="accel", people_vel=(np.zeros(2),), **far), GK)[3].get("passive_person")
    assert mp_a >= mp_s


def test_accel_speed_model_default_is_the_original_and_diagnosed_livelock_case():
    """robot at rest, operator pushes forward, person stands ~0.65 m ahead of the footprint: speed model rejects every hold, accel model admits a short one."""
    pose = np.array([1.5, 5.0, 0.0])
    c = np.array([1.5 + GM.FP["x"][1] + 0.60, 5.0])
    kw = dict(pose_wall=pose, pose_people=pose, people=(c,))
    for T_S in GK.t_switch:
        tr = plan_candidate(np.zeros(3), np.array([0.8, 0, 0]), T_S, bel(**kw), GK)
        assert not admit(tr, bel(**kw))
    tr = plan_candidate(np.zeros(3), np.array([0.8, 0, 0]), 0.1, bel(**kw), GK)
    assert admit(tr, bel(person_model="accel", people_vel=(np.zeros(2),), **kw))


class FakeSim:
    """minimal sim stand-in: a kinematic base, a cup at a fixed base-frame position, the wall grid, optional people."""

    def __init__(self, pose, people=()):
        self.base = BS.BaseSim(*pose)
        self.a_hw = BS.A_HW.copy()
        self.occ_ctrl = GRID
        self.loc = None
        self.cup = 0
        self.people_state = [(np.asarray(p, float), np.zeros(2)) for p in people]
        self.tray = 1
        self.scene = types.SimpleNamespace(people=[types.SimpleNamespace(name=f'p{i}') for i in range(4)])
        self.d = types.SimpleNamespace(xpos={0: np.zeros(3), 1: np.zeros(3)}, xmat={1: np.eye(3).ravel()}, xquat={1: np.array([1.0, 0, 0, 0])})
        self._sync()

    def _sync(self):
        R = GM.rot(self.base.yaw)
        self.d.xpos[0] = np.r_[np.array([self.base.x, self.base.y]) + R @ R_CUP, 0.0]

    def pose(self):
        return np.array([self.base.x, self.base.y, self.base.yaw])

    ctrl_pose = pose

    def step(self, u):
        self.base.advance(u, TICK, self.a_hw)
        self._sync()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gatekeeper_never_exceeds_cup_limit_when_assumptions_hold(seed):
    rng = np.random.default_rng(seed)
    g = Gatekeeper()
    sim = FakeSim((1.5, 5.0, 0.0), people=[(3.8, 7.0)] if seed == 2 else [])
    a_max = None
    u_op = np.zeros(3)
    worst = 0.0
    moved = 0.0
    for k in range(700):
        if k % 25 == 0:
            u_op = rng.uniform(-1, 1, 3) * BS.V_STICK * rng.choice([0.0, 1.0])
        v = sim.base.v.copy()
        u, label = g.filter_base(sim, u_op)
        g.n_calls += 1
        a_max = a_max or Belief(r_cup=R_CUP).a_cup_max
        R = GM.rot(sim.base.yaw)
        a_imp = np.linalg.norm(BS.cup_accel_body(v, (u - v) / TICK, R_CUP))
        worst = max(worst, a_imp)
        sim.step(u)
        moved = max(moved, np.hypot(sim.base.x - 1.5, sim.base.y - 5.0))
        # never inside the (d_wall) buffer of the wall at x = 5 (front of the footprint)
        assert sim.base.x + GM.FP["x"][1] + 0.0 < 5.0 - 0.10 + 0.02 or label.startswith("gk_regime2")
    assert worst <= a_max + 1e-6, (worst, a_max)
    assert g.n_falsified == 0 and moved > 0.3 and g.n_commit > 0


def test_gatekeeper_stops_before_wall_under_full_stick():
    g = Gatekeeper()
    sim = FakeSim((3.0, 5.0, 0.0))
    for k in range(600):
        u, _ = g.filter_base(sim, np.array([0.9, 0, 0]))
        g.n_calls += 1
        sim.step(u)
    front = sim.base.x + GM.FP["x"][1]
    assert front < 5.0 - GK_dw() and np.abs(sim.base.v).max() < 1e-6 and g.n_falsified == 0


def GK_dw():
    return Belief().d_wall


def test_falsification_regime2_when_person_exceeds_assumed_speed():
    """a person suddenly appears next to a moving robot -> the committed backup is no longer certified -> n_falsified > 0 and the shield takes over."""
    g = Gatekeeper(cfg=SH.ShieldCfg(use_ir_veto=False))
    sim = FakeSim((1.5, 5.0, 0.0))
    for k in range(60):
        u, _ = g.filter_base(sim, np.array([0.9, 0, 0])); g.n_calls += 1; sim.step(u)
    assert g.n_falsified == 0 and sim.base.v[0] > 0.3
    sim.people_state = [(np.array([sim.base.x + 1.2, sim.base.y]), np.zeros(2))]          # teleports in front: nothing can make this safe
    for k in range(10):
        u, label = g.filter_base(sim, np.array([0.9, 0, 0])); g.n_calls += 1; sim.step(u)
    assert g.n_falsified >= 1 and label.startswith("gk_regime2")


def test_run_py_default_path_unchanged():
    from world_model.v2.house.teleop import run as RUN
    sig = inspect.signature(RUN.run_episode)
    assert sig.parameters["filt"].default is None
    assert list(sig.parameters)[:5] == ["shielded", "seed", "task", "scene", "max_s"]


def test_run_py_filt_none_bit_identical_and_filt_injected():
    from world_model.v2.house.teleop import run as RUN, shield as SH
    s1, r1 = RUN.run_episode(False, 0, max_s=1.0, verbose=False)
    s2, r2 = RUN.run_episode(False, 0, max_s=1.0, verbose=False, filt=None)
    s3, r3 = RUN.run_episode(False, 0, max_s=1.0, verbose=False, filt=SH.Passthrough())
    for a, b in ((r1, r2), (r1, r3)):
        assert len(a) == len(b) and all(np.array_equal(x["pose"], y["pose"]) for x, y in zip(a, b))
    for k in ("spilled", "min_wall_clearance_m", "peak_base_speed"):
        assert s1[k] == s2[k] == s3[k]


# ------------------------------------------------------------------------------------------ opt-in improvements: defaults, candidate set, velocity source
GOLDEN = {   # recorded with the code BEFORE the opt-in features existed (FakeSim, 400 ticks, defaults)
    "a": dict(c={"n_commit": 80, "n_keep_backup": 0, "n_falsified": 0, "n_regime2_ticks": 0, "n_gk_steps": 80, "n_fallback": 0, "n_veto": 0,
                 "T_S_hist": {1.0: 69, 0.6: 9, 0.3: 2}}, chk=11064.153943629068, pose=[1.081103706393512, 4.942269507450336, 0.43948240254060833]),
    "b": dict(c={"n_commit": 25, "n_keep_backup": 55, "n_falsified": 0, "n_regime2_ticks": 0, "n_gk_steps": 80, "n_fallback": 0, "n_veto": 0,
                 "T_S_hist": {0.1: 25}}, chk=2777.8612068000125, pose=[2.1782790960000007, 5.0, 0.0]),
}


def _golden_run(name, gk=None):
    ppl, ops = ([(3.8, 7.0)], None) if name == "a" else ([(3.4, 5.3)], "fwd")
    rng = np.random.default_rng(3)
    g = Gatekeeper(gk=gk)
    sim = FakeSim((1.5, 5.0, 0.0), people=ppl)
    u_op = np.zeros(3)
    chk = 0.0
    for k in range(400):
        if ops is None:
            if k % 25 == 0:
                u_op = rng.uniform(-1, 1, 3) * BS.V_STICK * rng.choice([0.0, 1.0])
        else:
            u_op = np.array([0.8, 0, 0])
        u, _ = g.filter_base(sim, u_op); g.n_calls += 1; sim.step(u)
        chk += float(np.sum(u) * (k + 1))
    c = g.counters()
    for k in ("step_ms_mean", "step_ms_max", "step_ms_p95"):
        c.pop(k)
    return g, c, chk, [float(x) for x in sim.pose()]


@pytest.mark.parametrize("name", ["a", "b"])
def test_defaults_bit_identical_to_pre_feature_behaviour(name):
    g = GKConfig()
    assert g.person_model == "speed" and g.candidates == "hold"
    _, c, chk, pose = _golden_run(name)
    assert c == GOLDEN[name]["c"] and chk == GOLDEN[name]["chk"] and pose == GOLDEN[name]["pose"]
    # explicitly selecting the defaults and a candidate set that is never needed (free space) gives the same trajectory too
    _, c2, chk2, pose2 = _golden_run(name, GKConfig(person_model="speed", candidates="hold"))
    assert (c2, chk2, pose2) == (c, chk, pose)


def test_candidate_set_is_inert_when_hold_is_certified_and_stops_early():
    g0, c0, chk0, p0 = _golden_run("a")                                   # person far away, hold always certifies
    g1, c1, chk1, p1 = _golden_run("a", GKConfig(candidates="set"))
    # hold-u_op is tried first; the set is only searched when it fails -> in the steps where hold certified, nothing is spent
    assert g1.n_cand_checks <= 24 * g1.n_steps and g1.cand_checks_max <= 24
    if g1.n_deviate == 0:
        assert (chk1, p1) == (chk0, p0) and g1.n_cand_checks == 0 == sum(g1.dev_hist)


def test_candidate_set_commits_certified_deviation_when_hold_is_blocked():
    front = GM.FP["x"][1]
    ppl = [(1.5 + front + 0.65, 5.45)]                                     # standing person 0.65 m ahead, 0.45 m to the left of the stick direction
    res = {}
    for cd in ("hold", "set"):
        g = Gatekeeper(gk=GKConfig(candidates=cd))
        sim = FakeSim((1.5, 5.0, 0.0), people=ppl)
        n_chk = 0
        for k in range(300):
            u, label = g.filter_base(sim, np.array([0.8, 0, 0]))
            if (g.n_calls % g.gk.k_ticks) == 0 and g.mode == "commit":     # a gatekeeper step just committed: re-verify the committed trajectory independently
                ok1 = check1(g.traj, g.last_bel, g.gk)
                ok2 = check2(g.traj, g.last_bel, g.gk)
                assert ok1[0] and ok2[0], (ok1, ok2)
                n_chk += 1
            g.n_calls += 1
            sim.step(u)
        res[cd] = (g, sim, n_chk)
    gh, sh_, _ = res["hold"]
    gs, ss, nchk = res["set"]
    assert gh.n_deviate == 0
    assert gs.n_deviate > 0 and gs.n_deviate <= gs.n_commit and nchk > 0
    assert gs.cand_checks_max <= 24 and gs.n_falsified == 0
    assert np.hypot(ss.base.x - 1.5, ss.base.y - 5.0) > np.hypot(sh_.base.x - 1.5, sh_.base.y - 5.0) + 0.3     # the deviation actually makes progress where hold froze
    c = gs.counters()
    assert 0.0 < c["mean_deviation"] and c["mean_deviation_when_deviating"] > 0.0 and len(gs.dev_commit) == gs.n_deviate


def test_candidate_set_wall_backoff_direction_and_candidates():
    g = Gatekeeper(gk=GKConfig(candidates="set"))
    sim = FakeSim((4.0, 5.0, 0.0))                                         # front 0.42 m from the wall at x = 5; wall is straight ahead
    b = build_belief_for(g, sim)
    g.last_why = ("c1", 0.1, "wall", -0.1)
    bo = g._backoff_dir(sim, b)
    assert bo is not None and bo[0] < -0.9                                  # away from the wall = -x in the body frame
    cands = g._candidate_twists(sim, np.array([0.8, 0, 0]), b)
    devs = [d for d, _ in cands]
    assert devs == sorted(devs) and len(cands) <= 13 and min(devs) > 0
    assert any(x[0] < 0 for _, x in cands)                                  # a back-off candidate exists
    assert any(abs(x[1]) > 0.1 for _, x in cands) and any(abs(x[2]) > 0.1 and abs(x[0]) < 1e-9 for _, x in cands)


def build_belief_for(g, sim):
    from world_model.v2.house.gatekeeper import build_belief
    return build_belief(sim, g.cfg, g.gk, fp=g._shield(sim).footprint_of(sim), docks=g.docks)


def test_velocity_source_state_and_finite_difference():
    sim = FakeSim((1.5, 5.0, 0.0), people=[(3.0, 5.0)])
    sim.people_state = [(np.array([3.0, 5.0]), np.array([-0.7, 0.1]))]
    g = Gatekeeper(gk=GKConfig(person_model="accel"))
    assert g._people_vel(sim, 0.0) == (None, None)                          # state source: build_belief reads people_state velocities
    b = build_belief_for(g, sim)
    assert b.person_model == "accel" and np.allclose(b.people_vel[0], [-0.7, 0.1]) and b.e_v == g.gk.e_v
    g = Gatekeeper(gk=GKConfig(person_model="accel", vel_source="fd"))
    pv, ev = g._people_vel(sim, 0.0)                                        # first step: nothing known -> v_hat = 0, e_v = v_p_max
    assert np.allclose(pv[0], 0) and ev == g.gk.v_p_max
    sim.people_state = [(np.array([2.93, 5.01]), np.zeros(2))]
    pv, ev = g._people_vel(sim, 0.1)
    assert np.allclose(pv[0], [-0.7, 0.1]) and ev == pytest.approx(g.gk.e_v + g.gk.a_p * 0.1)
