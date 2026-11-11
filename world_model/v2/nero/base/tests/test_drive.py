"""base/drive.py + monitors.py: route parsing / nominal end (pure), monitors importable from smoke."""
import numpy as np
import pytest

from world_model.v2.nero.base import drive as DR, monitors as MON, scene as BS, smoke as SM


def test_route_segments_deg_to_rad_and_validation():
    assert DR.route_to_segments([["line", -0.5], ["turn", 180.0]]) == [("line", -0.5), ("turn", pytest.approx(np.pi))]
    with pytest.raises(ValueError):
        DR.route_to_segments([["hop", 1.0]])


def test_nominal_end_of_R1_from_any_heading():
    R1 = [["line", -0.5], ["turn", 180.0], ["line", 0.5]]
    e = DR.nominal_end((0.0, -0.21, 0.0), R1)
    assert np.allclose(e[:2], [-1.0, -0.21]) and abs(e[2] - np.pi) < 1e-12
    e2 = DR.nominal_end((1.0, 2.0, np.pi / 2), R1)                 # line along the CURRENT +x
    assert np.allclose(e2[:2], [1.0, 2.0 - 1.0], atol=1e-12)


def test_smoke_imports_the_moved_monitors():
    assert SM.EDGES is MON.EDGES and SM.LIFTOFF_STEPS == MON.LIFTOFF_STEPS and SM.ROUTE_TOL_M == MON.ROUTE_TOL_M == 0.02 and SM.ROUTE_TOL_DEG == MON.ROUTE_TOL_DEG == 2.0


def test_driver_dt_is_the_carry_dt():
    from world_model.v2.nero.contents import carry as CA
    assert DR.DT == CA.DT


def test_support_tops_behind_mirrors_the_counter():
    from world_model.v2.nero.contents import scenes as SC
    w = SC.TrayWorld(n_cups=6)
    t = BS.support_tops_behind(w)
    assert t["counter"]["top"] == t["counter2"]["top"] == w.counter_top
    cx = 0.5 * sum(t["counter2"]["x"])
    assert abs(cx - BS.mirror(w.counter_centre[:2], BS.mirror_centre())[0]) < 1e-12
    assert np.isclose(t["counter2"]["x"][1] - t["counter2"]["x"][0], 2 * w.counter_half[0])
