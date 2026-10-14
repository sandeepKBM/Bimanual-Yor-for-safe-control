"""house/loc unit tests (CPU, no GL, no network).  pytest world_model/v2/house/loc/tests"""
import math

import numpy as np

from world_model.v2.house.loc import estimator as ES, mapper as MP


def _drive(est, T=20.0, v=0.5, dt=0.02):
    p = np.zeros(3)
    t = 0.0
    for k in range(int(T / dt)):
        w = 0.3 if 5 < t < 9 else 0.0
        p = ES.compose(p, [v * dt, 0, w * dt]); t += dt
        est.update(t, p, dt)
    return p


def test_perfect_sensors_track_truth():
    cfg = ES.scaled(ES.LocCfg(), 0.0)
    e = ES.PoseEstimator(cfg, seed=0)
    _drive(e)
    r = e.errors()
    assert r["ate_max_m"] < 5e-3 and r["n_slip"] == 0 and r["n_dropout"] == 0


def test_deterministic_and_seeded():
    a, b, c = ES.PoseEstimator(seed=3), ES.PoseEstimator(seed=3), ES.PoseEstimator(seed=4)
    _drive(a); _drive(b); _drive(c)
    assert np.array_equal(a.pose, b.pose) and not np.array_equal(a.pose, c.pose)


def test_error_bounded_vs_odometry_only():
    # the VIO correction must beat pure odometry dead-reckoning on a long drive
    cfg = ES.LocCfg()
    fused = ES.PoseEstimator(cfg, seed=1)
    _drive(fused, T=40.0)
    odo_only = ES.PoseEstimator(ES.LocCfg(gain_xy=0.0, gain_yaw=0.0), seed=1)
    _drive(odo_only, T=40.0)
    assert fused.errors()["ate_rmse_m"] < odo_only.errors()["ate_rmse_m"]


def test_compose_inverse():
    p0, p1 = np.array([1.0, 2.0, 0.3]), np.array([1.4, 2.2, 0.9])
    inc = ES.body_increment(p0, p1)
    assert np.allclose(ES.compose(p0, inc), p1)


def test_integrate_wall():
    # a ray pair hitting a wall at x=1.0 between z in band marks the wall cell occupied and clears the cells in front
    res, x0, y0 = 0.05, 0.0, 0.0
    L = np.zeros((20, 40, 40))
    ex, ey, ez = np.array([1.0]), np.array([1.0]), np.array([0.5])
    MP._integrate(L, res, x0, y0, 0.2, 1.0, 0.5, ex, ey, ez, np.array([True]), 0.15, 1.15, 0.85, -0.4, 4.0, None, None, None, 0.05)
    assert L[:, 20, 20].max() > 0 and L[:, 20, 10].min() < 0 and L[:, 20, 5].min() < 0


def test_integrate_floor_hit_not_occupied():
    L = np.zeros((20, 40, 40))
    MP._integrate(L, 0.05, 0.0, 0.0, 0.2, 1.0, 1.1, np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([True]), 0.15, 1.15, 0.85, -0.4, 4.0, None, None, None, 0.05)
    assert (L > 0).sum() == 0


def test_shield_default_is_truth_pose():
    from world_model.v2.house.teleop import shield as SH

    class S:
        def pose(self):
            return np.array([1.0, 2.0, 3.0])
    assert np.array_equal(SH.Shield._wall_pose(S()), S().pose())
