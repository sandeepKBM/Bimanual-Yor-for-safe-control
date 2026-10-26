"""zonotope obstacles (tamp/spec_zono.py): cluster / split / fit on synthetic walls, clearance vs a synthetic occupancy grid.  SIMULATION ONLY, no sensing.
    python -m pytest -q world_model/v2/house/tamp/tests/test_spec_zono.py"""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.house.tamp import spec_zono as SZ
from world_model.v2.house.teleop.scene import OccGrid


def wall_points(p0, p1, z=(0.15, 1.15), vox=0.02, thick=0.02, rng=None):
    L = float(np.hypot(*(np.asarray(p1) - np.asarray(p0))))
    n = int(L / vox) + 1
    t = np.linspace(0, 1, n)
    xy = np.asarray(p0)[None] + t[:, None] * (np.asarray(p1) - np.asarray(p0))[None]
    zs = np.arange(z[0], z[1], vox)
    P = np.array([[x, y, zz] for x, y in xy for zz in zs])
    if rng is not None:
        P[:, :2] += rng.normal(0, thick / 2, (len(P), 2))
    return P


def test_split_pieces_are_short_straight_and_cover_everything():
    rng = np.random.default_rng(0)
    P = np.concatenate([wall_points((0, 0), (4, 0), rng=rng), wall_points((4, 0), (4, 3), rng=rng)])         # an L: 4 m + 3 m
    cl = SZ.clusters_xy(P)
    assert len(cl) == 1
    pieces = SZ.split_pieces(P[:, :2], cl[0])
    assert sorted(np.concatenate(pieces).tolist()) == list(range(len(P)))                                    # a partition
    for p in pieces:
        X = P[p, :2]
        if len(X) > 1:
            u = np.linalg.svd(X - X.mean(0))[2][0]
            assert np.ptp((X - X.mean(0)) @ u) <= SZ.MAX_LEN + 1e-9
    assert len(pieces) >= 6


def test_split_handles_ties_and_sparse_spread_out_sets():
    col = np.tile([[1.33, 6.39]], (9, 1))                          # a vertical pole: identical xy, different z
    far = np.array([[2.81, 4.25]])
    X = np.concatenate([col, far])
    pieces = SZ.split_pieces(X, np.arange(len(X)))
    assert sorted(np.concatenate(pieces).tolist()) == list(range(10))
    for p in pieces:
        assert np.ptp(X[p, 0]) < 1.0 and np.ptp(X[p, 1]) < 1.0      # the far point is not boxed together with the pole
    two = np.array([[1.03, 3.43], [2.75, 3.41]])                   # two points 1.7 m apart, almost the same y: still two pieces
    assert len(SZ.split_pieces(two, np.arange(2))) == 2


def test_fit_obstacles_contain_the_points_and_stay_tight():
    rng = np.random.default_rng(1)
    P = np.concatenate([wall_points((2.7, 2.0), (2.7, 5.0), rng=rng), wall_points((1.0, 3.4), (2.0, 3.4), rng=rng)])
    route = np.array([[2.1, 5.0], [2.1, 2.0]])
    zs, info, Pb, pieces = SZ.fit_obstacles(P, route, verbose=False)
    assert SZ.containment(zs, Pb, pieces) == 1.0
    # tight: the wall at x = 2.7 gives a clearance of 0.5 m to a footprint edge at x = 2.2 within a few cm
    pose = np.array([[2.2 - 0.5770901, 2.5, 0.0]])                                  # front edge at x = 2.2
    cz = SZ.zono_clearance(zs, pose, far=5.0)
    assert abs(cz[0] - 0.5) < 0.05


def test_compare_with_grid_counts_false_safe_and_false_unsafe():
    # synthetic grid: occupied band x in [3.0, 3.1], y in [0, 10] ; 1 cm cells
    res, x0, y0 = 0.01, -0.3, -0.3
    nx, ny = 500, 1100
    occ = np.zeros((ny, nx), bool)
    occ[:, int((3.0 - x0) / res):int((3.1 - x0) / res)] = True
    from scipy import ndimage
    grid = OccGrid(res, x0, y0, occ, ndimage.distance_transform_edt(~occ) * res)
    from world_model.v2.tray_rep.sets.zono import Zonotope
    zon_ok = [Zonotope.from_box([3.0, 0.0, 0.15], [3.1, 10.0, 1.15])]            # perceives the wall
    zon_missing = [Zonotope.from_box([3.0, 8.0, 0.15], [3.1, 10.0, 1.15])]       # only the far part: the wall near y = 5 is NOT perceived
    poses = np.array([[3.0 - 0.5770901 - 0.30, 5.0, 0.0], [3.0 - 0.5770901 - 0.05, 5.0, 0.0], [3.0 - 0.5770901 + 0.05, 5.0, 0.0]])    # 0.30 m, 0.05 m, -0.05 m (colliding)
    r, cg, cz = SZ.compare_with_grid(zon_ok, grid, poses, d_wall=0.10)
    assert r["false_safe"] == 0 and r["false_unsafe"] == 0 and r["both_safe"] == 1 and r["both_unsafe"] == 2 and r["colliding_true"] == 1
    r2, _, _ = SZ.compare_with_grid(zon_missing, grid, poses, d_wall=0.10)
    assert r2["false_safe"] == 3 - 1 and r2["worst_false_safe_m"] < 0 and r2["false_safe_idx"] == [1, 2]
    zon_big = [Zonotope.from_box([2.0, 0.0, 0.15], [3.1, 10.0, 1.15])]            # a phantom slab in front of the wall
    r3, _, _ = SZ.compare_with_grid(zon_big, grid, poses[:1], d_wall=0.10)
    assert r3["false_unsafe"] == 1


def test_route_poses_follow_the_polyline():
    P = SZ.route_poses([[0, 0], [1, 0], [1, 1]], 0.25)
    assert len(P) == 9 and abs(P[0, 2]) < 1e-12 and abs(P[5, 2] - math.pi / 2) < 1e-12 and np.allclose(P[-1, :2], [1, 1])
