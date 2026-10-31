"""Stage-0 shadow mode (tamp/rh/shadow.py): metric helpers on synthetic data + observe="truth" leaves the episode unchanged.  SIMULATION ONLY.
    python -m pytest -q world_model/v2/house/tamp/rh/tests/test_shadow.py        (the episode test needs RH_SLOW=1: ~1.5 min, builds two sims)"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from world_model.v2.house.tamp.rh import shadow as SH


def test_se2_roundtrip_and_place():
    p = (1.0, 2.0, 0.7)
    q = SH.se2_mul(p, SH.se2_inv(p))
    assert np.allclose(q, (0, 0, 0), atol=1e-12)
    # estimate == truth -> identical prediction
    pred = (1.3, 2.1, 0.9)
    assert np.allclose(SH.place_est_state(pred, p, p), pred)
    # estimate shifted by (0.1, 0) and yawed by 0.2: the predicted state keeps its RELATIVE body motion
    est = (1.1, 2.0, 0.9)
    pe = SH.place_est_state(pred, p, est)
    assert np.allclose(SH.se2_mul(SH.se2_inv(est), pe), SH.se2_mul(SH.se2_inv(p), pred))
    assert SH.pose_error((0, 0, 3.1), (0, 0, -3.1))[1] == pytest.approx(2 * math.pi - 6.2)


def test_match_people_hungarian_and_gate():
    T = [[0.0, 0.0], [2.0, 0.0], [5.0, 5.0]]
    K = [[2.1, 0.1], [0.05, 0.0], [9.0, 9.0]]
    pairs, missed, false_t = SH.match_people(T, K, 0.5)
    assert {(i, j) for i, j, _ in pairs} == {(0, 1), (1, 0)}
    assert missed == [2] and false_t == [2]
    assert dict(((i, j), d) for i, j, d in pairs)[(0, 1)] == pytest.approx(0.05)
    # global assignment, not greedy: a nearer track must go to the person it is closer to overall
    pairs, _, _ = SH.match_people([[0, 0], [0.4, 0]], [[0.3, 0]], 0.5)
    assert len(pairs) == 1 and pairs[0][0] == 1
    # gate: 0.6 m apart is not a match
    pairs, missed, false_t = SH.match_people([[0, 0]], [[0.6, 0]], 0.5)
    assert pairs == [] and missed == [0] and false_t == [0]
    assert SH.match_people([], [[1, 1]]) == ([], [], [0])
    assert SH.match_people([[1, 1]], []) == ([], [0], [])


def test_classify_visibility():
    # camera at origin looking along +x (camera -z = world +x): R columns (right, up, back) = (-y, z, -x)
    R = np.stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], 1).astype(float)
    K = dict(f=200.0, cx=320.0, cy=180.0)
    pts = [[2, 0, 0], [2, 5, 0], [-2, 0, 0], [10, 0, 0], [3, 0.2, 0]]
    blocked = lambda o, P: abs(P[0] - 3) < 1e-9
    c = SH.classify_visibility(pts, [0, 0, 0], R, K, 640, 360, blocked)
    assert c == ["visible", "out_of_view", "out_of_view", "out_of_view", "occluded"]


def _row(fs=False, fu=False, ea="a", ta="a", **pe):
    p = dict(n_truth=2, n_match=1, miss_visible=1, miss_occluded=0, miss_out_of_view=0, miss_no_frame=0, n_false_tracks=1, n_match_visible=1, n_match_occluded=0,
             pos_err=[0.1], vel_err=[0.2])
    p.update(pe)
    return dict(false_safe=fs, false_unsafe=fu, est_adm=ea, truth_adm=ta, pose_err_xy=0.02, pose_err_yaw=0.01, people=p)


def test_summarize_counts_and_percentiles():
    rows = [_row(fs=True), _row(fu=True, ea=None), _row()]
    s = SH.summarize(rows, [10.0, 20.0, 30.0], [100.0, 200.0])
    assert (s["n_decisions"], s["n_false_safe"], s["n_false_unsafe"], s["n_est_admit"], s["n_truth_admit"], s["n_same_candidate"]) == (3, 1, 1, 2, 3, 2)
    P = s["people"]
    assert P["counts"]["miss_visible"] == 3 and P["n_visible_person_decisions"] == 6 and P["miss_rate_visible"] == pytest.approx(0.5)
    assert P["miss_rate_occluded"] is None
    assert P["pos_err_p50_m"] == pytest.approx(0.1) and P["vel_err_p95_ms"] == pytest.approx(0.2)
    assert s["stereo_ms_per_tick"]["p50"] == pytest.approx(20.0) and s["stereo_ms_per_frame"]["n_frames"] == 2
    assert SH.pct([], 50) is None


GOLD = Path(__file__).parent / "golden_truth_s0_4s.json"


@pytest.mark.skipif(not os.environ.get("RH_SLOW"), reason="short episode (~1.5 min); set RH_SLOW=1")
def test_observe_truth_unchanged():
    """observe='truth' (the default) = the pre-change loop: first admit_log rows + counters of a 4 s lattice seed-0 run match a golden recorded from the unmodified loop.py."""
    from world_model.v2.house.tamp.rh import loop as LP
    from world_model.v2.house.tamp.rh.api import RHConfig
    from world_model.v2.house.tamp.rh.lattice import LatticeProposer
    g = json.loads(GOLD.read_text())
    sc = LP.load_scenario(0)
    walker = dict(delay_s=sc["walker_delay"], speed=sc["walker_speed"], x0=sc["walker_x0"])
    r = LP.rh_episode(LatticeProposer(), 0, sc["stand_offset"], walker, RHConfig(), max_s=4.0, verbose=False)
    assert "shadow" not in r and "observe" not in r
    r = json.loads(json.dumps({**{k: r[k] for k in g if k != "admit_log"}, "admit_log": r["admit_log"][:6]}, default=str))
    assert r == g
