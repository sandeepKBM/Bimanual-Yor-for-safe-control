"""Tests for house/perception_bridge.py.  SIMULATION ONLY, CPU.  Run:
    PYTHONPATH=. python -m pytest -q world_model/v2/house/tests/test_perception_bridge.py        (about 40 s: composes house 17 once)
"""
from dataclasses import replace

import numpy as np
import pytest
import mujoco

from world_model.v2.house import perception_bridge as PB
from world_model.v2.house import phase1 as P1
from world_model.v2.tray_rep.perception import ball_estimator as BE

IDEAL = dict(sigma_floor=0.0, sigma_disp_px=0.0, p_drop=0.0, p_out=0.0, quant_m=0.0, cos_min=0.0)


@pytest.fixture(scope="module")
def sc():
    C = P1.new_hold(17, (3.55, 10.05), 0.3, 0.005)
    s = PB.HouseScene(C)
    s.place_base((3.55, 10.05), 0.3)
    return s


def test_groups_reassigned(sc):
    m = sc.m
    assert all(m.geom_group[g] == PB.G_TRAY for g in (*sc.gid_tray, sc.gid_ball))
    assert (m.geom_group[sc.wall_geoms] == PB.G_TRAY).all() and len(sc.wall_geoms) > 0
    robot = np.isin(sc.cls, list("LRB")) & (sc.group0 == 3)
    assert robot.sum() > 20 and not (sc.is_house & (sc.group0 == 3)).any()          # g3 = robot collision only


def test_hide_restores_groups(sc):
    g0 = sc.m.geom_group.copy()
    with sc.hide("house"):
        assert (sc.m.geom_group[sc.is_house] == PB.G_HIDE).all()
    assert (sc.m.geom_group == g0).all()


def test_head_camera_follows_base_and_ball_distance_exact(sc):
    cams, _ = PB.make_rig(sc, "head_aim", **IDEAL)
    cam = cams[0]
    for xy, yaw in (((3.55, 10.05), 0.3), ((3.0, 9.5), -1.0)):
        sc.place_base(xy, yaw)
        o, R = cam.pose()
        Rb = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        assert np.allclose(o, np.array([*xy, 0.0]) + Rb @ np.array([1.1, -0.21, 1.05]), atol=1e-9)
        tp, tR = sc.tray_pose()
        f = cam.capture(sc, tp, tR, 0, 0, True, self_filter=False)
        bw = sc.d.xpos[sc.body_ball]
        hit = f.geom == sc.gid_ball
        if hit.sum() == 0:
            continue
        # ideal sensor: every ball return lies on the sphere
        r = np.linalg.norm(f.pts_w[hit] - bw, axis=1)
        assert np.abs(r - PB.BALL_R).max() < 1e-6


def test_self_filter_removes_robot_points_and_tracker_is_accurate(sc):
    sc.place_base((3.55, 10.05), 0.3)
    cams, specs = PB.make_rig(sc, "head_aim+wrist", **IDEAL)
    tp, tR = sc.tray_pose()
    raw = [c.capture(sc, tp, tR, 0, 0, True, self_filter=False) for c in cams]
    fil = [c.capture(sc, tp, tR, 0, 0, True) for c in cams]
    is_robot = lambda f: np.isin(sc.geom_cat[f.geom], ["arm_L", "arm_R", "robot_base"]).sum()
    assert sum(is_robot(f) for f in raw) > 100
    assert sum(is_robot(f) for f in fil) < 0.02 * sum(is_robot(f) for f in raw)
    pay = BE.measure(fil, BE.EstCfg(), specs)
    truth = sc.ball_in_tray()
    assert pay[BE.I_VALID] > 0.5 and np.linalg.norm(pay[BE.I_X:BE.I_X + 2] - truth[:2]) < 2e-3


def test_place_base_keeps_hold_relative_pose(sc):
    rel = lambda: sc.tray_pose()[1].T @ (sc.d.xpos[sc.m.body("left_arm_gripper_base").id] - sc.tray_pose()[0])
    sc.place_base((3.55, 10.05), 0.3)
    a = rel()
    sc.place_base((5.0, 8.0), 1.2, (0.05, 0.02))
    assert np.allclose(rel(), a, atol=1e-9)
    assert np.allclose(sc.ball_in_tray(), [0.05, 0.02, PB.BALL_R + 0.0005], atol=1e-9)
    sc.place_base((3.55, 10.05), 0.3)


def test_roi_cost_smoke_and_ray_budget(sc):
    import time
    sc.place_base((3.55, 10.05), 0.3)
    cams, _ = PB.make_rig(sc, "head_aim")
    tp, tR = sc.tray_pose()
    f0 = cams[0].capture(sc, tp, tR, 0, 0, True)
    f1 = cams[0].capture(sc, tp, tR, 0, 0, True, max_rays=300)
    assert f1.n_rays <= 330 < f0.n_rays
    t = time.perf_counter()
    cams[0].capture(sc, tp, tR, 1, 0, True, max_rays=600)
    assert (time.perf_counter() - t) < 0.2           # loose: machine is shared; the study reports the real number


def test_meshobj_truth_matches_ray_hits(sc):
    key = next(k for k, v in sc.md.items() if v["category"] == "Bowl")
    o = PB.MeshObj(sc, key)
    lo, hi = o.aabb()
    ctr = (lo + hi) / 2
    spec = replace(PB.D.CAMERAS["head_aim"], mount="world", pos=tuple(ctr + [0.0, -0.5, 0.4]), R=tuple(PB._lookat_R(ctr + [0, -0.5, 0.4], ctr).ravel()), **IDEAL)
    labels = {g: 1 for g in PB.object_geoms(sc, key)}
    cl = PB.capture_cloud(sc, [spec], ctr, float(np.linalg.norm(hi - lo)) / 2 + 0.05, labels)
    P = cl.pts[cl.label == 1]
    assert len(P) > 50 and np.abs(o.sdf(P)).max() < 4e-3
