"""Unit tests for people_track (no GPU, no display, no robot).  python -m pytest -q world_model/v2/house/people_track/tests"""
import math

import numpy as np
import pytest

from world_model.v2.house.people_track.tracker import (MultiTracker, TrackerCfg, Det, predict_xP, kf_update, TICK, Track)


def det(x, y, s=0.05):
    return Det(np.array([x, y], float), np.eye(2) * s ** 2)


# --------------------------------------------------------------------------------------------- KF
def test_kf_predict_is_constant_velocity_and_grows_covariance():
    x = np.array([1.0, 2.0, 0.5, -0.25]); P = np.diag([0.01, 0.01, 0.04, 0.04])
    x1, P1 = predict_xP(x, P, 0.5, 0.03)
    assert np.allclose(x1, [1.25, 1.875, 0.5, -0.25])
    assert P1[0, 0] > P[0, 0] + 0.04 * 0.25 - 1e-12 and np.allclose(P1, P1.T) and np.linalg.eigvalsh(P1).min() > 0


def test_kf_update_shrinks_and_converges_to_constant_velocity_target():
    rng = np.random.default_rng(0)
    x = np.zeros(4); P = np.diag([1, 1, 1, 1.0])
    truth_v = np.array([0.3, -0.2])
    for k in range(1, 400):
        x, P = predict_xP(x, P, TICK, 0.03)
        if k % 2 == 0:
            z = truth_v * k * TICK + rng.normal(0, 0.05, 2)
            x, P = kf_update(x, P, z, np.eye(2) * 0.05 ** 2)
    assert np.allclose(x[2:], truth_v, atol=0.08)
    assert np.sqrt(P[0, 0]) < 0.06 and np.allclose(P, P.T) and np.linalg.eigvalsh(P).min() > 0


# --------------------------------------------------------------------------------------------- tracker
def run_ticks(trk, k0, k1):
    for k in range(k0, k1):
        trk.predict_to(k)


def test_association_hungarian_keeps_identities_when_targets_cross_close():
    trk = MultiTracker(TrackerCfg(confirm_hits=1))
    trk.predict_to(0)
    trk.process_frame([det(0.0, 0.0), det(1.0, 0.0)], 0)
    ids = {t.id: t.x[:2].copy() for t in trk.tracks}
    assert len(ids) == 2
    for k in range(1, 40):
        trk.predict_to(k)
        if k % 2 == 0:
            a = np.array([0.0088 * k, 0.0]); b = np.array([1.0 - 0.0088 * k, 0.0])
            trk.process_frame([det(*b), det(*a)], k)                  # detections delivered in the opposite order
    t0 = [t for t in trk.tracks if t.id == 0][0]; t1 = [t for t in trk.tracks if t.id == 1][0]
    assert t0.x[0] < t1.x[0] and abs(t0.x[0] - 0.33) < 0.1 and abs(t1.x[0] - 0.67) < 0.1 and len(trk.tracks) == 2


def test_gate_rejects_far_detection_and_births_new_track():
    trk = MultiTracker(TrackerCfg())
    trk.predict_to(0)
    trk.process_frame([det(0, 0)], 0)
    for k in range(1, 5):
        trk.predict_to(k); trk.process_frame([det(0.0, 0.0)], k)
    n0 = len(trk.tracks)
    trk.predict_to(5); trk.process_frame([det(0, 0), det(3.0, 3.0)], 5)
    assert len(trk.tracks) == n0 + 1 and np.hypot(*trk.tracks[0].x[:2]) < 0.1


def test_confirmation_and_tentative_deletion():
    trk = MultiTracker(TrackerCfg(confirm_hits=2, tent_max_miss=2))
    trk.predict_to(0); trk.process_frame([det(1, 1)], 0)
    assert trk.output() == [] and len(trk.tracks) == 1                    # tentative: not published
    trk.predict_to(2); trk.process_frame([det(1, 1)], 2)
    assert len(trk.output()) == 1                                         # 2nd hit confirms
    trk2 = MultiTracker(TrackerCfg())
    trk2.predict_to(0); trk2.process_frame([det(1, 1)], 0)
    for k in (2, 4):
        trk2.predict_to(k); trk2.process_frame([], k)
    assert len(trk2.tracks) == 0                                          # two missed frames kill a tentative track


def test_coasting_through_occlusion_grows_covariance_then_dies():
    cfg = TrackerCfg(confirm_hits=2, coast_s=2.0)
    trk = MultiTracker(cfg)
    for k in range(0, 40, 2):
        trk.predict_to(k)
        trk.process_frame([det(0.01 * k, 0.0)], k)
    t = trk.output()[0]
    s0 = t.sigma_pos(); x0 = t.x.copy()
    for k in range(40, 40 + 60):
        trk.predict_to(k); trk.predict_frame_tick(k)                       # occluded: no frames
    t = trk.output()[0]
    assert t.sigma_pos() > 3 * s0 and t.x[0] > x0[0] + 0.3                  # predicts forward, uncertainty grows
    for k in range(100, 100 + 80):
        trk.predict_to(k); trk.predict_frame_tick(k)
    assert trk.output() == []                                              # coast_s exceeded -> deleted


def test_latency_retrodiction_matches_zero_latency():
    """a frame captured at tick 20 and delivered at tick 23 gives the same state at tick 23 as processing it at 20 and predicting."""
    def build():
        t = MultiTracker(TrackerCfg(confirm_hits=1))
        for k in range(0, 20, 2):
            t.predict_to(k); t.process_frame([det(0.02 * k, 0.0)], k)
        return t
    A = build()
    A.predict_to(20); A.process_frame([det(0.40, 0.0)], 20); A.predict_to(23)
    B = build()
    B.predict_to(23); B.process_frame([det(0.40, 0.0)], 20)
    ta, tb = A.tracks[0], B.tracks[0]
    assert np.allclose(ta.x, tb.x, atol=1e-9) and np.allclose(ta.P, tb.P, atol=1e-9)
    C = build(); C.predict_to(23)                                            # without the late frame the posterior is worse
    assert B.tracks[0].P[0, 0] < C.tracks[0].P[0, 0]


# --------------------------------------------------------------------------------------------- sim-dependent
@pytest.fixture(scope="module")
def simenv():
    from world_model.v2.house.people_track.sim_fix import TeleopSimWalking
    from world_model.v2.house.people_track import perception as PC
    sim = TeleopSimWalking("h5", grasped=True, seed=0)
    perc = PC.PerceptionStack(sim, PC.PerceptionCfg(detector="param"))
    return sim, perc


def _place(sim, x, y, yaw):
    import mujoco
    from world_model.v2.house import build as B
    sim.d.mocap_pos[0] = (x, y, 0.0)
    sim.d.mocap_quat[0] = B.yaw_quat(yaw)
    mujoco.mj_forward(sim.m, sim.d)


def test_fov_and_occlusion_gating(simenv):
    sim, perc = simenv
    _place(sim, 2.1, 4.0, -math.pi / 2)                                   # lab corridor, facing -y (south)
    fwd = np.array([0.0, -1.0])
    ahead = np.array([2.1, 4.0]) + 2.5 * fwd
    behind = np.array([2.1, 4.0]) - 2.0 * fwd
    side = np.array([2.1, 4.0]) + np.array([2.0, 0.2])
    fov, vis, rng = perc.vis.of(sim, ahead)
    assert fov and vis and abs(rng - 2.5) < 0.1
    assert not perc.vis.of(sim, behind)[0]                                # behind the robot: outside the frustum
    assert not perc.vis.of(sim, side)[0]                                  # beside the robot: outside the frustum
    _place(sim, 1.0, 6.0, math.pi)                                        # facing -x toward the wall at x~0 (spans y 3.76..7.76)
    f2, v2, _ = perc.vis.of(sim, np.array([-1.2, 6.0]))
    assert f2 and not v2                                                  # inside the frustum but behind the wall: occluded
    _place(sim, 3.1, 6.09, -2.33121118)


def test_parametric_detector_respects_gating_dropout_noise(simenv):
    from world_model.v2.house.people_track.detectors import ParametricDetector
    sim, perc = simenv
    truth = [(np.array([1.0, 1.0]), np.zeros(2)), (np.array([2.0, 2.0]), np.zeros(2))]
    vis = [(True, True, 2.0), (False, False, 3.0)]
    d = ParametricDetector(sigma=0.1, p_drop=0.0, seed=1)
    outs = [d.detect(None, perc.cam, sim, vis=vis, truth=truth) for _ in range(400)]
    assert all(len(o) == 1 and o[0].truth == 0 for o in outs)             # the non-visible person is never reported
    err = np.array([o[0].xy - truth[0][0] for o in outs])
    assert abs(err.std() - 0.1) < 0.015 and abs(err.mean()) < 0.02
    d2 = ParametricDetector(sigma=0.1, p_drop=0.5, seed=1)
    n = sum(len(d2.detect(None, perc.cam, sim, vis=vis, truth=truth)) for _ in range(600))
    assert 240 < n < 360


class _FakeTrack:
    def __init__(self, i, xy, v, P):
        self.id, self.x, self.P, self.status = i, np.array([*xy, *v], float), P, "conf"
        self.name = f"trk{i}"


def _shield(sim, perc, **u):
    from world_model.v2.house.people_track.shield_pt import PerceivedShield, UncertaintyCfg
    return PerceivedShield(perc, None, occ=sim.occ, a_hw=sim.a_hw, ucfg=UncertaintyCfg(unseen_cap=False, **u))


def test_proxy_pairs_tracks_and_delegates(simenv):
    from world_model.v2.house.people_track.perception import SimProxy
    sim, perc = simenv
    P = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
    tr = [_FakeTrack(7, (2.0, 3.0), (0.1, 0.0), P), _FakeTrack(3, (5.0, 1.0), (0.0, -0.2), P)]
    px = SimProxy(sim, tr)
    assert [p.name for p in px.scene.people] == ["trk7", "trk3"] and len(px.people_state) == 2
    assert np.allclose(px.people_state[0][0], [2.0, 3.0]) and np.allclose(px.people_state[1][1], [0.0, -0.2])
    assert px.d is sim.d and px.base is sim.base and px.occ is sim.occ and px.scene.name == sim.scene.name and px.scene.waypoints is sim.scene.waypoints
    assert px.without_people().people_state == [] and px.without_people().scene.people == []


def test_tags_match_tracks_and_equal_original_rows_when_no_inflation(simenv):
    from world_model.v2.house.teleop import shield as SH
    from world_model.v2.house.people_track.perception import SimProxy
    sim, perc = simenv
    pose = sim.pose()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    near = pose[:2] + np.array([c, s]) * 0.9 + np.array([-s, c]) * -0.3
    far = pose[:2] + np.array([c, s]) * 1.6
    P = np.zeros((4, 4))
    tr = [_FakeTrack(1, tuple(near), (0.2, -0.1), P), _FakeTrack(2, tuple(far), (-0.1, 0.3), P)]
    px = SimProxy(sim, tr)
    sh = _shield(sim, perc, k_sigma=0.0, d_buffer=0.10)
    A, b, tags = sh._keep_rows(px, sim.base.v)
    plain = SH.Shield(None, occ=sim.occ, a_hw=sim.a_hw)
    sim_state = sim.people_state
    try:
        sim.people_state = [(t.x[:2], t.x[2:]) for t in tr]
        sim_scene_people = sim.scene.people
        sim.scene.people = [type("N", (), {"name": t.name})() for t in tr]
        A0, b0, tags0 = plain._keep_rows(sim, sim.base.v)
    finally:
        sim.people_state = sim_state
        sim.scene.people = sim_scene_people
    assert tags == tags0 and np.allclose(A, A0) and np.allclose(b, b0)      # k_sigma = 0: identical to the original rows
    assert any(t == "person:trk1" for t in tags) and any(t == "person:trk2" for t in tags)


def test_inflation_scales_with_track_covariance_and_is_capped(simenv):
    from world_model.v2.house.people_track.perception import SimProxy
    sim, perc = simenv
    pose = sim.pose()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    xy = tuple(pose[:2] + np.array([c, s]) * 1.2)
    res = {}
    for name, sig in (("tight", 0.02), ("loose", 0.15), ("huge", 2.0)):
        P = np.diag([sig ** 2] * 2 + [1e-6] * 2)
        sh = _shield(sim, perc, k_sigma=2.0, t_horizon=0.5, max_infl=0.5)
        A, b, tags = sh._keep_rows(SimProxy(sim, [_FakeTrack(0, xy, (0, 0), P)]), sim.base.v)
        res[name] = (sh.last_infl, b[-1])
    assert res["tight"][0] < res["loose"][0] < res["huge"][0] + 1e-12
    assert abs(res["loose"][0] - 0.30) < 0.05                               # 2 * 0.15
    assert abs(res["huge"][0] - 0.5) < 1e-9                                 # capped
    assert res["tight"][1] > res["loose"][1] >= res["huge"][1]              # larger uncertainty -> tighter row


def test_stack_end_to_end_tracks_a_visible_static_person(simenv):
    from world_model.v2.house.people_track import perception as PC
    from world_model.v2.house.people_track.tracker import TrackerCfg
    sim, _ = simenv
    _place(sim, 2.1, 4.0, -math.pi / 2)
    sim.people_state = [(np.array([2.0, 1.8]), np.zeros(2)), (np.array([2.0, 7.0]), np.zeros(2))]      # one ahead, one behind
    perc = PC.PerceptionStack(sim, PC.PerceptionCfg(detector="param", sigma=0.05, latency_ticks=3))
    for k in range(60):
        sim.k = k
        perc.step(sim)
    assert len(perc.tracks) == 1 and np.hypot(*(perc.tracks[0].x[:2] - [2.0, 1.8])) < 0.1
    _place(sim, 3.1, 6.09, -2.33121118)


def test_circle_fit_beats_centroid_on_a_partial_arc():
    from world_model.v2.house.people_track import detectors as D
    rng = np.random.default_rng(0)
    c = np.array([3.0, 1.0]); cam = np.array([0.0, 0.0, 1.15])
    th = np.linspace(math.pi - 1.2, math.pi - 0.2, 40)
    pts = np.stack([c[0] + 0.22 * np.cos(th), c[1] + 0.22 * np.sin(th)], 1) + rng.normal(0, 0.01, (40, 2))
    c0 = pts.mean(0); u = c0 - cam[:2]; guess = c0 + D.AXIS_OFFSET * u / np.linalg.norm(u)
    fit = D._circle_fit(pts, guess)
    assert np.linalg.norm(fit - c) < 0.06 < np.linalg.norm(guess - c)
