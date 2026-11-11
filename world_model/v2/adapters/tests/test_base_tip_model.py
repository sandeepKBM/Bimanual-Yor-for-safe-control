"""adapters/base_tip_model vs the compiled swerve scene (SIMULATION ONLY, CPU)."""
import numpy as np
import pytest

from world_model.v2.adapters import base_tip_model as BT
from world_model.v2.adapters import contents_specs as CS
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks import base_tip as T


def test_polygon_from_pivots_and_offsets():
    P = BT.polygons()
    assert P["line"].x_hi == pytest.approx(0.105, abs=2e-3) and P["line"].y_hi == pytest.approx(0.156 - 0.021, abs=2e-3)
    assert P["turn"].x_hi < P["line"].x_hi and P["turn"].y_hi > P["line"].y_hi


def test_mass_model_matches_sim_truth():
    from world_model.v2.nero.contents import scenes as SC
    for task in ("tray6_behind", "box_behind"):
        mm = BT.mass_model(task)
        w = CS.build_world(task)
        ep = BT._episode(w)
        t = SC.object_truth(ep.m, ep.d)
        spec = CS.spec_from_cell(task, *TK.prior_box(task))
        assert mm.m[-1] == pytest.approx(t["empty_mass"], rel=0.05)                        # spec empty mass = the sim shell
        assert np.allclose(mm.obj_R.T @ (mm.p[-1] - mm.obj_p), spec.empty_com) and mm.labels["robot_mass"] > 30


def test_measured_row_refuses_heavy_tray_at_L2_not_L1():
    """G0b 11-09: heaviest tray6 world lifted the back axle at L2 (and tipped at L3), level at L1."""
    from world_model.v2.nero.base import smoke as SM
    w = SM.task_world("tray6", SM.heaviest_seed(task="tray6")[1])
    best, rows = BT.pick_level("tray6_behind", type("S", (), {"contents": BT.true_contents("tray6_behind", w)})(), world=w)
    assert best == "L1" and not rows["L2"].ok and not rows["L3"].ok
    assert rows["L1"].assumptions["accel"]["line"]["source"] == "measured"
    assert rows["L1"].guarantee == "sampled" and rows["L1"].n == BT.card()["levels"]["L1"]["n"]        # max over n runs, not a bound (rules 6-7)
    assert "base_tip" in TK.task("box_behind").rows and "base_tip" not in TK.task("box").rows
