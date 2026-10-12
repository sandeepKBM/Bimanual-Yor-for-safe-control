"""Synthetic-grid tests for nav.py (no house, no MuJoCo model): run with
   PYTHONPATH=<repo> python -m pytest world_model/v2/house/tests/test_nav_phase2.py   or   python <this file>"""
import math
import numpy as np
from world_model.v2.house import nav as N


def _grid(gap=0.884, res=0.01):
    """10 x 6 m room split by a wall at x=5 with a door gap centred at y=3."""
    ny, nx = int(6 / res), int(10 / res)
    occ = np.zeros((ny, nx), bool)
    occ[0:3, :] = occ[-3:, :] = occ[:, 0:3] = occ[:, -3:] = True
    occ[:, 495:505] = True
    lo, hi = int((3 - gap / 2) / res), int((3 + gap / 2) / res)
    occ[lo:hi, 495:505] = False
    return N.Grid(res, 0.0, 0.0, occ).with_edt()


def test_door_passage_fit_and_no_fit():
    door = dict(centre=np.array([5.0, 3.0]), wall_dir=np.array([0.0, 1.0]), normal=np.array([1.0, 0.0]))
    ok = N.door_passage(_grid(0.884), N.DEFAULT_FP, door, +1)
    assert ok["ok"] and 0.04 < ok["clearance"] < 0.07          # (0.884 - 0.774) / 2 = 5.5 cm
    assert abs(ok["u"] - 0.257) < 0.02                         # off-centre footprint: origin offset ~ +0.257 (heading +x)
    no = N.door_passage(_grid(0.75), N.DEFAULT_FP, door, +1)
    assert not no["ok"]


def test_plan_route_and_limits():
    F = _grid(1.6)
    C = N.coarsen(F)
    door = dict(id="d", centre=np.array([5.0, 3.0]), wall_dir=np.array([0.0, 1.0]), normal=np.array([1.0, 0.0]),
                clear_w=1.6, design_w=1.6)
    out = N.plan_route(0, (1.8, 1.4), (8.2, 4.4), door, F=F, C=C, verbose=False, amax=0.25, vmax=0.3)
    assert out["route"] is not None, out["reason"]
    r = out["route"]
    assert r["peak_speed"] <= 0.31 and r["peak_acc"] <= 0.26 and r["end_err_m"] < 1e-3
    assert r["min_clearance_door_zone_m"] >= 0.05


if __name__ == "__main__":
    test_door_passage_fit_and_no_fit(); test_plan_route_and_limits(); print("ok")
