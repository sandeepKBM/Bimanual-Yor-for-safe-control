"""adapters/base_sweep.py (table-behind D1, 11-09): the kinematic sweep of a base route.  SIMULATION ONLY (CPU, ~10 s)."""
import numpy as np
import pytest

from world_model.v2.adapters import base_sweep as BSW
from world_model.v2.stack.planners import base_route as BR

R1 = (("line", -0.5), ("turn", 180.0), ("line", 0.5))


@pytest.fixture(scope="module")
def world():
    return BSW.world_default("tray6_behind")


def test_backend_facts(world):
    sup = BSW.supports(world)
    assert sup["counter"]["top"] == sup["counter2"]["top"]
    assert np.allclose(sup["counter2"]["x"], (-1.53, -1.13), atol=1e-3) and np.allclose(sup["counter2"]["y"], (-0.3491, -0.0491), atol=1e-3)   # MODEL: mirror of counter 1
    x, y, yaw = BSW.start_pose(world)
    assert abs(x) < 1e-6 and abs(y + 0.20955) < 1e-4 and abs(yaw) < 1e-6
    assert BSW.levels()["L1"] == (0.25, 0.25, 0.5, 0.5)
    assert set(BSW.hold_q(world)) == {"left", "right"}
    gx, gy = BSW.place_pose_xy(world)
    assert abs(gx + 1.38) < 1e-3 and abs(gy + 0.1991) < 1e-3


def test_sweep_start_pose_matches_static_distances(world):
    p0 = BSW.start_pose(world)
    r = BSW.sweep(world, [(0,) + tuple(p0)], obj_off=(0, 0, 0.05))
    assert r["n_poses"] == 1 and r["ok"] and 0.005 <= r["min_dist"] < 0.1 and r["worst"]["group"] in ("robot", "load")


def test_sweep_r1_clear_and_turn_at_pick_hits(world):
    p0 = BSW.start_pose(world)
    r1 = BSW.sweep(world, BR.route_poses(p0, R1), obj_off=(0, 0, 0.05))
    assert r1["ok"] and r1["min_dist"] >= BSW.CLEARANCE_MIN and set(r1["by_segment"]) == {0, 1, 2}
    r2 = BSW.sweep(world, BR.route_poses(p0, (("turn", 180.0),)), obj_off=(0, 0, 0.05))
    assert not r2["ok"] and r2["min_dist"] < BSW.CLEARANCE_MIN and r2["worst"]["seg"] == 0
    assert "counter" in " ".join(r2["worst"]["pair"]) or "table" in " ".join(r2["worst"]["pair"])
