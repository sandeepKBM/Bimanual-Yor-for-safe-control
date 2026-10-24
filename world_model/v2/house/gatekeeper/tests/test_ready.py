"""C2-ready ranking helpers (gatekeeper/ready.py) + the new opt-in GKConfig fields.  SIMULATION ONLY, no sim: Belief and Traj built directly."""
import numpy as np
import pytest

from world_model.v2.house.gatekeeper import GKConfig
from world_model.v2.house.gatekeeper import ready as RD
from world_model.v2.house.gatekeeper.belief import Belief
from world_model.v2.house.gatekeeper.traj import Traj

H = 0.05


def _traj(rest=True):
    V = [[0.3, 0, 0]] * 3 + [[0.2, 0, 0], [0.1, 0, 0]] + ([[0, 0, 0]] * 3 if rest else [[0.05, 0, 0]])
    return Traj(np.array(V, float), H, 0.0, 3 * H, 3 * H)


def _bel(**kw):
    d = dict(pose_wall=np.array([1.0, 2.0, np.pi / 2]), pose_people=np.array([1.0, 2.0, np.pi / 2]), people=(np.array([5.0, 5.0]), np.array([0.0, 0.0])),
             people_vel=(np.array([-1.0, 0.5]), np.array([0.2, 0.0])), people_stale=(0.3, 0.4), people_e_p=(0.1, 0.1), people_e_v=(0.05, 0.05), latency_s=0.2,
             e_xy=0.07, e_p=0.03)
    d.update(kw)
    return Belief(**d)


def test_defaults_off():
    g = GKConfig()
    assert g.c2_ready == "off" and g.seg_offsets == () and g.c2_ready_dmin == 0.05 and g.c2_ready_speeds == (0.3, 0.2, 0.12)
    assert (g.seg_k_lat, g.seg_v_lat_max, g.seg_offset_taper) == (1.5, 0.15, 0.6)


def test_rest_poses_none_without_rest_and_anchor_otherwise():
    b = _bel()
    assert RD.rest_poses(_traj(False), b) is None
    tr = _traj()
    pw, pp, t = RD.rest_poses(tr, b)
    assert t == pytest.approx(tr.i_rest * H) and tr.i_rest == 5
    # heading pi/2: travel along +x of the start frame maps to +y of the world
    P = tr.poses()[5]
    assert pw[0] == pytest.approx(1.0 - 0.0 * P[0] - P[1]) and pw[1] == pytest.approx(2.0 + P[0]) and P[0] > 0.04
    np.testing.assert_allclose(pw, pp)


def test_belief_at_rest_moves_people_and_sets_hypotheses():
    b = _bel()
    pw, pp, t = np.array([1.0, 2.4, 1.5]), np.array([1.0, 2.4, 1.5]), 0.25
    for hyp in RD.HYPS:
        r = RD.belief_at_rest(b, pw, pp, t, hyp)
        np.testing.assert_allclose(r.people[0], [5.0 - 0.25, 5.0 + 0.125])
        np.testing.assert_allclose(r.people[1], [0.05, 0.0])
        assert r.people_stale == (0.0, 0.0)
        np.testing.assert_allclose(r.pose_wall, pw)
        np.testing.assert_allclose(r.pose_people, pp)
        assert r.e_xy == b.e_xy and r.e_p == b.e_p and r.latency_s == b.latency_s and r.people_e_p == b.people_e_p and r.people_e_v == b.people_e_v
    np.testing.assert_allclose(RD.belief_at_rest(b, pw, pp, t, "walk").people_vel[0], [-1.0, 0.5])
    np.testing.assert_allclose(RD.belief_at_rest(b, pw, pp, t, "stop").people_vel[0], [0.0, 0.0])
    np.testing.assert_allclose(b.people[0], [5.0, 5.0])          # the input belief is not mutated
    assert b.people_stale == (0.3, 0.4)


def test_no_velocity_estimate_both_hypotheses_identical():
    b = _bel(people_vel=(), people_stale=())
    pw = np.array([1.0, 2.4, 0.0])
    w, s = (RD.belief_at_rest(b, pw, pw, 0.25, h) for h in RD.HYPS)
    assert w.people_vel == () and s.people_vel == () and w.people_stale == ()
    for a, c in zip(w.people, s.people):
        np.testing.assert_allclose(a, c)
    np.testing.assert_allclose(w.people[0], [5.0, 5.0])


def test_ready_score_counts_hypotheses_and_no_rest():
    b, tr = _bel(), _traj()
    calls = []

    def mv(br):
        calls.append(br)
        return None if np.allclose(br.people_vel[0], 0) else "route v=0.3"
    score, n, detail = RD.ready_score(tr, b, GKConfig(), mv)
    assert (score, n) == (1, 2) and len(calls) == 2 and detail == ["walk:route v=0.3", "stop:none"]
    calls.clear()
    assert RD.ready_score(_traj(False), b, GKConfig(), mv) == (0, 2, ["no_rest"]) and not calls
    assert RD.ready_score(tr, b, GKConfig(), lambda br: "x")[0] == 2
