"""stack/checks/base_tip: analytic cases (no sim)."""
import numpy as np
import pytest

from world_model.v2.nero.contents.objects import Contents
from world_model.v2.stack.checks import base_tip as T

POLY = T.Polygon(-0.1, 0.1, -0.15, 0.15)


def _model(x=0.0, z=0.5, m=10.0):
    return T.MassModel(np.array([m]), np.array([[x, 0.0, z]]), np.eye(3), np.zeros(3))


def test_line_closed_form_matches_a_tip():
    mm = _model(x=0.05, z=0.5)
    none = Contents(0.0, 0.0, np.zeros((1, 3)))
    a_tip = T.a_tip(mm, 0.0, [0, 0, 0], POLY)
    assert a_tip == pytest.approx((0.1 - 0.05) * T.G / 0.5)
    for a, ok in ((0.99 * a_tip, True), (1.01 * a_tip, False)):
        r = T.tip_row(mm, none, {"line": POLY}, {"line": T.Accel(a_lin=a)})
        assert r.ok is ok and r.guarantee == "proof"                      # command source: proof of the model only
    rm = T.tip_row(mm, none, {"line": POLY}, {"line": T.Accel(a_lin=0.5, source="measured", n=4)})
    assert rm.guarantee == "sampled" and rm.n == 4


def test_turn_centripetal_pushes_outward():
    mm = _model(x=0.05, z=0.5)
    none = Contents(0.0, 0.0, np.zeros((1, 3)))
    m0, _ = T.zmp_margin(mm.m, mm.p, POLY, "turn", T.Accel(alpha=0.0, w=0.0))
    m1, w1 = T.zmp_margin(mm.m, mm.p, POLY, "turn", T.Accel(alpha=0.0, w=3.0))
    # ZMP shift = + w^2 z r / g outward: 9 * 0.5 * 0.05 / 9.81
    assert m0 - m1 == pytest.approx(9 * 0.5 * 0.05 / T.G) and w1["edge"] == [1, 0]
    assert T.tip_row(mm, none, {"turn": POLY}, {"turn": T.Accel(w=3.0)}).ok


def test_contents_vertex_and_level_choice():
    mm = _model(x=0.0, z=0.5, m=10.0)
    c = Contents(0.0, 10.0, np.array([[0.09, 0.0, 0.5], [-0.09, 0.0, 0.5]]))     # 10 kg at the front edge region is the worst vertex
    best, rows = T.pick_level(mm, c, {"line": POLY}, {"L1": {"line": T.Accel(a_lin=0.1)}, "L2": {"line": T.Accel(a_lin=1.0)}})
    # worst: m=10 at x=0.09: c_x 0.045, S_z/M = 0.5 -> a_tip = 0.055 g / 0.5 = 1.079
    assert best == "L2" and rows["L2"].ok
    best, rows = T.pick_level(mm, c, {"line": POLY}, {"L1": {"line": T.Accel(a_lin=1.2)}})
    assert best is None and not rows["L1"].ok and rows["L1"].worst["contents_m"] == 10.0
