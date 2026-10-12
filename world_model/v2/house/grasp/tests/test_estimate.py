"""Pure-numpy tests of the tray-pose estimator (no MuJoCo model).  python -m pytest -q world_model/v2/house/grasp/tests"""
import math

import numpy as np

from world_model.v2.house.grasp import estimate as E
from world_model.v2.house.grasp.wrist import WristFrame
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def synth_frame(p_true, R_true, rng, view="near_edge_plus_x", sigma=0.002, n=1500, with_shelf=True, with_cup=True):
    """noisy plate points (top surface), shelf points 6 mm lower around it, a cup blob; all returned in WORLD frame, geom 1 = tray."""
    pts, gid = [], []
    # plate: visible part depends on the view (a strip next to the +x edge, as the left wrist camera sees it)
    x = rng.uniform(-HX, HX, n)
    y = rng.uniform(-HY, HY, n)
    if view == "near_edge_plus_x":
        keep = x > 0.06
    elif view == "far":
        keep = np.ones(n, bool)
    else:
        keep = x < -0.03
    x, y = x[keep], y[keep]
    pl = np.c_[x, y, rng.normal(0, sigma, len(x))]
    pts.append(pl)
    gid.append(np.ones(len(pl), int))
    if with_shelf:
        m = 900
        xs, ys = rng.uniform(-0.26, 0.26, m), rng.uniform(-0.22, 0.22, m)
        out = (np.abs(xs) > HX + 0.002) | (np.abs(ys) > HY + 0.002)
        sh = np.c_[xs[out], ys[out], -0.006 + rng.normal(0, sigma, out.sum())]
        if view == "near_edge_plus_x":
            sh = sh[sh[:, 0] > 0.04]
        pts.append(sh)
        gid.append(np.zeros(len(sh), int))
    if with_cup:
        th = rng.uniform(0, 2 * math.pi, 150)
        zc = rng.uniform(0.01, 0.1, 150)
        cu = np.c_[-0.045 + 0.04 * np.cos(th), 0.04 * np.sin(th), zc]
        if view != "near_edge_plus_x":
            pts.append(cu)
            gid.append(np.zeros(len(cu), int) + 2)
    P = np.vstack(pts)
    g = np.concatenate(gid)
    w = p_true + P @ R_true.T
    return WristFrame("left", 0, 0.0, np.zeros(3), np.eye(3), np.zeros(3), np.eye(3), w, g.astype(np.int32), np.ones(len(w)))


def _run(mode, view, seed):
    rng = np.random.default_rng(seed)
    prior_p = np.array([1.0, 2.0, 0.85])
    prior_R = _rz(0.4)
    d_true = np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-0.004, 0.004)])
    dyaw = math.radians(rng.uniform(-4, 4))
    p_true = prior_p + prior_R @ d_true
    R_true = prior_R @ _rz(dyaw)
    fr = synth_frame(p_true, R_true, rng, view=view)
    est = E.estimate_tray(fr, prior_p, prior_R, mode=mode, tray_geoms=np.array([1]), seed=seed)
    e, en, ang = E.pose_error(est.p, est.R, p_true, R_true)
    return est, e, ang


def test_edge_distance_and_height_recovered_from_a_strip():
    # a strip next to the +x edge: the edge distance (tray x) and plate height (tray z) are observable to a few mm, yaw to ~2 deg
    for mode in ("priv", "depth"):
        for seed in range(5):
            est, e, ang = _run(mode, "near_edge_plus_x", seed)
            assert est.ok and est.conf > 0.4, (mode, seed, est.conf)
            assert abs(e[0]) < 7.0 and abs(e[2]) < 3.0, (mode, seed, e)
            assert ang < 5.5, (mode, seed, ang)


def test_full_view_recovers_full_pose():
    for mode in ("priv", "depth"):
        for seed in range(4):
            est, e, ang = _run(mode, "far", seed)
            assert est.ok and est.conf > 0.3
            assert np.abs(e).max() < 6.0 and ang < 4.0, (mode, seed, e, ang)


def test_empty_and_sparse_frames_report_low_confidence():
    fr = WristFrame("left", 0, 0.0, np.zeros(3), np.eye(3), np.zeros(3), np.eye(3), np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0))
    est = E.estimate_tray(fr, np.zeros(3), np.eye(3), mode="depth")
    assert not est.ok and est.conf == 0.0
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 0.05, (200, 3)) + np.array([3.0, 3.0, 3.0])             # nothing near the prior volume
    fr = WristFrame("left", 0, 0.0, np.zeros(3), np.eye(3), np.zeros(3), np.eye(3), pts, np.zeros(200, np.int32), np.ones(200))
    est = E.estimate_tray(fr, np.zeros(3), np.eye(3), mode="depth")
    assert (not est.ok) or est.conf < 0.3


def test_ransac_plane_prefers_prior_height_over_shelf():
    rng = np.random.default_rng(1)
    top = np.c_[rng.uniform(0.05, 0.15, 400), rng.uniform(-0.1, 0.1, 400), rng.normal(0, 0.0015, 400)]
    shelf = np.c_[rng.uniform(0.15, 0.25, 500), rng.uniform(-0.1, 0.1, 500), -0.006 + rng.normal(0, 0.0015, 500)]
    abc = E.ransac_plane(np.vstack([top, shelf]), rng, z0=0.0)
    assert abc is not None and abs(abc[2]) < 0.0035
