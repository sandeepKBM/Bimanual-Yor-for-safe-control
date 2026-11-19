"""Owner gate (11-17) for stack/checks/base_stop.py and base_route.remaining_route / route_start: PURE, no physics.
USER 11-16 "go with a and build the base stop check"; USER 11-17: resume = finish the planned route, safe = on a support, in_route always adds the measured extra."""
import math

import pytest

from world_model.v2.stack.checks import base_stop as BS
from world_model.v2.stack.checks import fixability as FX
from world_model.v2.stack.planners import base_route as BR

L2 = (0.5, 0.5, 1.0, 1.0)                       # (v, a, w, alpha) of level L2 (base_sweep.levels)
ROUTE = (("line", -0.5), ("turn", 180.0), ("line", 1.0), ("line", 0.3))      # Codex run-2 reply 3
GAP = 0.5
P0 = (0.0, -0.2095, 0.0)


def _segs(route=ROUTE, caps=L2):
    return BS.segment_table(route, caps, GAP)


def _times(segs, dt=0.25):
    T = segs[-1]["t1"]
    return [round(k * dt, 6) for k in range(int(T / dt) + 2)]


def test_trapezoid_ends_and_triangle():
    for D, vm, am in ((1.0, 0.5, 0.5), (0.1, 0.5, 0.5), (math.pi, 1.0, 1.0)):
        T = BS._trap_T(D, vm, am)
        s0, v0 = BS.trap_state(0.0, D, vm, am)
        s1, v1 = BS.trap_state(T, D, vm, am)
        assert s0 == 0.0 and v0 == 0.0 and abs(s1 - D) < 1e-9 and abs(v1) < 1e-9
        vp = max(BS.trap_state(T * k / 200, D, vm, am)[1] for k in range(201))
        assert vp <= vm + 1e-9
    s, v = BS.trap_state(BS._trap_T(0.1, 0.5, 0.5) / 2, 0.1, 0.5, 0.5)          # triangle: peak at T/2 = sqrt(D a)
    assert abs(v - math.sqrt(0.1 * 0.5)) < 1e-9 and abs(s - 0.05) < 1e-9


def test_in_route_zero_extra_stays_inside_every_segment():
    segs = _segs()
    r = BS.in_route_row(segs, _times(segs), dict(line=0.0, turn=0.0), {})
    assert r.ok and r.guarantee == "sampled" and r.margin >= -BS.TOL


def test_in_route_extra_always_added_needs_sweeps():
    segs = _segs()
    ts = _times(segs)
    ex = dict(line=0.0063, turn=0.009)
    cases = BS.overshoot_cases(segs, ts, ex)
    assert cases, "the MEASURED extra must make the decel-phase stops run past the segment end (USER 11-17: no shortcut)"
    assert all(c["excess"] > 0 for c in cases)
    assert BS.in_route_row(segs, ts, ex, {}).guarantee == "missing"                      # no sweep -> MISSING, never a pass
    clear = {BS.case_key(c): dict(ok=True, min_dist_m=0.05, clearance_min_m=0.005, worst=None) for c in cases}
    r = BS.in_route_row(segs, ts, ex, clear)
    assert r.ok and r.guarantee == "sampled"
    bad = dict(clear)
    k = BS.case_key(cases[0])
    bad[k] = dict(ok=False, min_dist_m=-0.01, clearance_min_m=0.005, worst="right_arm_link2_col vs counter")
    r = BS.in_route_row(segs, ts, ex, bad)
    assert not r.ok and r.guarantee == "refuted" and abs(r.margin - (-0.015)) < 1e-9 and "past the segment end" in r.why


def test_overshoot_pose_pushes_past_the_segment_end():
    segs = _segs()
    c = dict(segment=0, kind="line", excess=0.01, t=0.0)
    x, y, yaw = BS.overshoot_pose(P0, ROUTE, c)
    assert abs(x - (-0.51)) < 1e-9 and abs(y - P0[1]) < 1e-9 and yaw == 0.0               # line -0.5 driven backwards: 1 cm further back
    c = dict(segment=1, kind="turn", excess=0.02, t=0.0)
    assert abs(BS.overshoot_pose(P0, ROUTE, c)[2] - (math.pi + 0.02)) < 1e-9


def test_missing_card_is_missing():
    segs = _segs()
    ts = _times(segs)
    assert BS.in_route_row(segs, ts, None).guarantee == "missing"
    assert BS.at_rest_row(segs, ts, None).guarantee == "missing"


def test_at_rest_row():
    segs = _segs()
    ts = _times(segs)
    ok = BS.at_rest_row(segs, ts, 0.31)
    assert ok.ok and ok.margin > 0
    bad = BS.at_rest_row(segs, ts, 3.997)                                                  # the pre-fix card's settle (end-of-route timeouts)
    assert not bad.ok and bad.guarantee == "refuted"


@pytest.mark.parametrize("skills,ok,g", [(None, True, "proof"), (["hold", "retrace"], True, "proof"), (["set_down"], False, "refuted"),
                                         (["let_go"], False, "refuted"), (["retrace"], False, "missing")])
def test_recovery_row(skills, ok, g):
    r = BS.recovery_row(skills)
    assert r.ok is ok and r.guarantee == g


def test_route_start_inverts_route_end():
    p_end = BR.route_end(P0, ROUTE)
    s = BR.route_start(p_end, BR.normalize_route(ROUTE))
    assert all(abs(a - b) < 1e-9 for a, b in zip(s, P0))


def test_remaining_route_finishes_the_planned_route():
    R = BR.normalize_route(ROUTE)
    # landed 0.2 m into segment 0 (line -0.5: driving backwards): 0.3 m of it left, then the rest unchanged
    rem = BR.remaining_route(P0, R, 0, (-0.2, P0[1], 0.0))
    assert rem[0][0] == "line" and abs(rem[0][1] - (-0.3)) < 1e-9 and rem[1:] == R[1:]
    # overshot the end of segment 0 by 1 cm -> drive 1 cm forward, then the rest
    rem = BR.remaining_route(P0, R, 0, (-0.51, P0[1], 0.0))
    assert abs(rem[0][1] - 0.01) < 1e-9 and rem[1:] == R[1:]
    # landed 90 deg into the 180 deg turn -> the other 90 deg, then the lines
    rem = BR.remaining_route(P0, R, 1, (-0.5, P0[1], math.pi / 2))
    assert rem[0][0] == "turn" and abs(rem[0][1] - 90.0) < 1e-6 and rem[1:] == R[2:]
    # at the segment end (below MIN_LINE_M) -> that piece is dropped
    assert BR.remaining_route(P0, R, 0, (-0.5, P0[1], 0.0)) == R[1:]
    assert BR.remaining_route(P0, R, len(R), (0, 0, 0)) == ()


def test_interrupted_index_rest_gap_is_next_segment():
    segs = _segs()
    assert BS.interrupted_index(segs, 0.0) == 0
    assert BS.interrupted_index(segs, segs[0]["t1"] + GAP / 2) == 1
    assert BS.interrupted_index(segs, segs[-1]["t1"] + 1.0) == len(segs)


def test_next_step_row():
    assert BS.next_step_row(None).guarantee == "missing"
    assert BS.next_step_row([("s2_place", True, "")]).ok
    r = BS.next_step_row([("s2_place", False, "landing.not_supported")])
    assert not r.ok and r.guarantee == "refuted" and "s2_place" in r.why


def test_every_new_row_name_is_tagged():
    names = ["base_stop.recovery", "base_stop.in_route", "base_stop.at_rest", "base_stop.tip", "base_stop.torque", "base_stop.landing_hold",
             "base_stop.continuation", "base_stop.goal_support", "base_stop.next_step", "landing.not_supported"]
    tags = {n: FX.tag_leaf(n, "refuted", "").kind for n in names}
    assert FX.UNTAGGED not in tags.values(), tags
    assert tags["base_stop.in_route"] == FX.NOT_FIXABLE and tags["base_stop.continuation"] == FX.FIXABLE and tags["landing.not_supported"] == FX.FIXABLE
