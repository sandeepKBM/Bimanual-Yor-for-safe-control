"""Owner gate (11-17): resume after a base stop = FINISH THE PLANNED ROUTE (USER 11-17).  PURE, no physics.
PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_remaining_route_gate.py

Checks stack/planners/base_route.remaining_route:
  - the remaining turn is the SHORTEST rotation; an exact +-180 takes the planned turn's sign;
  - a landing within MIN_LINE_M / MIN_TURN_DEG of the segment end (before or past it) counts as arrived;
  - the returned route ends at the planned end pose route_end(p0, route) (final correction), also with sideways / yaw drift."""
import math

import numpy as np
import pytest

from world_model.v2.stack.planners import base_route as BR

ROUTE = (("line", -0.5), ("turn", 180.0), ("line", 1.0), ("line", 0.3))
P0 = (0.0, 0.0, 0.0)


def _end_err(p_land, rem, p_end):
    e = BR.route_end(p_land, rem)
    dxy = math.hypot(e[0] - p_end[0], e[1] - p_end[1])
    dyaw = abs((e[2] - p_end[2] + math.pi) % (2 * math.pi) - math.pi)
    return dxy, dyaw


def test_turn_180_from_rest_takes_the_planned_sign():
    p_land = BR.route_end(P0, ROUTE[:1])                       # stopped exactly at the end of the first line, before the 180 turn
    rem = BR.remaining_route(P0, ROUTE, 1, p_land)
    assert rem[0][0] == "turn" and rem[0][1] == pytest.approx(180.0)
    assert _end_err(p_land, rem, BR.route_end(P0, ROUTE)) == pytest.approx((0.0, 0.0), abs=1e-9)
    neg = (("line", -0.5), ("turn", -180.0), ("line", 1.0))
    assert BR.remaining_route(P0, neg, 1, p_land)[0][1] == pytest.approx(-180.0)


def test_shortest_rotation_after_a_turn_overshoot():
    x, y, _ = BR.route_end(P0, ROUTE[:1])
    p_land = (x, y, math.radians(270.0))                        # 90 deg past the planned 180
    rem = BR.remaining_route(P0, ROUTE, 1, p_land)
    assert rem[0] == ("turn", pytest.approx(-90.0))


@pytest.mark.parametrize("past_m, expect_piece", [(0.004, False), (-0.004, False), (0.007, True), (-0.007, True)])
def test_line_end_tolerance(past_m, expect_piece):
    route = (("line", 1.0), ("line", 0.3))
    p_land = (1.0 + past_m, 0.0, 0.0)                           # +: past the end of segment 0, -: short of it
    rem = BR.remaining_route(P0, route, 0, p_land)
    first = rem[0]
    if expect_piece:
        assert first[0] == "line" and first[1] == pytest.approx(-past_m)
    else:
        assert first == ("line", 0.3)


def test_random_landings_end_at_the_planned_pose():
    rng = np.random.default_rng(1117)
    p_end = BR.route_end(P0, ROUTE)
    for _ in range(400):
        i = int(rng.integers(0, len(ROUTE)))
        k, v = ROUTE[i]
        frac = float(rng.uniform(0.0, 1.0))
        part = ROUTE[:i] + (((k, v * frac),) if frac > 0 else ())
        x, y, yaw = BR.route_end(P0, part)
        p_land = (x + rng.uniform(-0.03, 0.03), y + rng.uniform(-0.03, 0.03), yaw + rng.uniform(-0.05, 0.05))   # stop overshoot + sideways / yaw drift
        rem = BR.remaining_route(P0, ROUTE, i, p_land)
        dxy, dyaw = _end_err(p_land, rem, p_end)
        assert dxy <= BR.MIN_LINE_M + 1e-9 and math.degrees(dyaw) <= BR.MIN_TURN_DEG + 1e-9, (i, frac, p_land, rem)
        assert all(abs(v_) <= 180.0 + 1e-9 for k_, v_ in rem if k_ == "turn")


def test_after_last_segment_returns_empty():
    assert BR.remaining_route(P0, ROUTE, len(ROUTE), BR.route_end(P0, ROUTE)) == ()
