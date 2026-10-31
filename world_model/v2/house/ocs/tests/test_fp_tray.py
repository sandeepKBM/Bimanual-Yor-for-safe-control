"""ocs.fp_tray.fit_tray on SYNTHETIC depth point clouds (no GL, no GPU): a flat 30x20 cm tray at a known offset / yaw / tilt from the 'FK' tray frame, a cup + a
gripper-sized occluder, a far floor, depth noise, dropout.  The fit sees only the point cloud + the camera origin."""
import math

import numpy as np
import pytest

from world_model.v2.house.ocs import fp_tray as FT
from world_model.v2.house.ocs import util as U


def _camera_rays(W=320, H=180, fovy=70.0):
    f = (H / 2) / math.tan(math.radians(fovy) / 2)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - (W - 1) / 2) / f, (v - (H - 1) / 2) / f, np.ones_like(u, float)], -1).reshape(-1, 3)
    return d


def _scene_cloud(dx, dy, yaw_deg, tilt_deg=0.0, occlude=True, seed=0, sigma=0.004, drop=0.02, with_cup=True):
    """camera above/behind the tray looking down at ~36 deg (FK tray frame: x along the long axis, z up); the TRUE tray = FK frame shifted (dx, dy), rotated yaw, tilted.
    Returns (P (N,3) valid points in the FK frame, cam_o, T_true (4,4) true tray pose in the FK frame)."""
    rng = np.random.default_rng(seed)
    pitch = math.radians(50.0)                                       # ray direction: forward (+y_fk is 'away' here) and down
    fwd = np.array([0.0, math.cos(pitch), -math.sin(pitch)])
    up = np.array([0.0, math.sin(pitch), math.cos(pitch)])
    right = np.cross(fwd, up)
    cam_o = np.array([0.0, -0.30, 0.45])
    d = _camera_rays()
    dirs = d[:, 0:1] * right + (-d[:, 1:2]) * up + d[:, 2:3] * fwd
    ty, yw = math.radians(tilt_deg), math.radians(yaw_deg)
    Rz = np.array([[math.cos(yw), -math.sin(yw), 0], [math.sin(yw), math.cos(yw), 0], [0, 0, 1.0]])
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    Rt = Rz @ Ry
    pt = np.array([dx, dy, 0.0])
    n = Rt[:, 2]
    s_t = (n @ pt - n @ cam_o) / (dirs @ n)                          # ray / tray-plane intersection
    X = cam_o + s_t[:, None] * dirs
    loc = (X - pt) @ Rt
    on_tray = (np.abs(loc[:, 0]) <= U.HX) & (np.abs(loc[:, 1]) <= U.HY) & (s_t > 0)
    s_floor = (-0.7 - cam_o[2]) / dirs[:, 2]
    s = np.where(on_tray, s_t, s_floor)                              # else the floor 70 cm below
    if with_cup:                                                      # a cup (r 3.5 cm, h 10 cm) standing on the tray near its centre: nearer surface
        cc = pt + Rt @ np.array([0.03, -0.02, 0.0])
        a = dirs[:, :2]
        o = cam_o[:2] - cc[:2]
        A = (a ** 2).sum(1)
        B = 2 * (a * o).sum(1)
        C = (o ** 2).sum() - 0.035 ** 2
        disc = B * B - 4 * A * C
        sc = (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A)
        zc = cam_o[2] + sc * dirs[:, 2]
        hit = (disc > 0) & (sc > 0) & (zc >= 0.0) & (zc <= 0.10) & (sc < s)
        s = np.where(hit, sc, s)
    valid = rng.random(len(s)) > drop
    if occlude:                                                       # 'gripper': a 6 cm box over one short end of the tray (self-filtered pixels -> no depth)
        Xs = cam_o + s[:, None] * dirs
        loc2 = (Xs - pt) @ Rt
        valid &= ~((loc2[:, 0] > U.HX - 0.06) & (np.abs(loc2[:, 1]) < 0.05))
    z = s * np.maximum(d[:, 2], 1e-9)                                 # noise along the ray (depth-like)
    s_n = s + sigma * rng.normal(size=len(s))
    P = cam_o + s_n[:, None] * dirs
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rt, pt
    return P[valid], cam_o, T


def _err(T_fit, T_true):
    D = np.linalg.inv(T_true) @ T_fit
    yaw = abs(math.degrees(math.atan2(D[1, 0], D[0, 0])))
    tilt = math.degrees(math.acos(float(np.clip(D[2, 2], -1, 1))))
    return np.linalg.norm(D[:2, 3]), abs(D[2, 3]), yaw, tilt


@pytest.mark.parametrize("dx,dy,yaw", [(0.0, 0.0, 0.0), (0.04, -0.03, 4.0), (-0.045, 0.02, -5.0), (0.03, 0.045, 3.0)])
def test_offset_recovered(dx, dy, yaw):
    P, co, T = _scene_cloud(dx, dy, yaw, seed=1)
    r = FT.fit_tray(P, co, rng=np.random.default_rng(0))
    assert r["ok"], r["why"]
    e_xy, e_z, e_yaw, e_tilt = _err(r["T"], T)
    assert e_xy < 0.006 and e_z < 0.003 and e_yaw < 2.0 and e_tilt < 1.0, (e_xy, e_z, e_yaw, e_tilt)


def test_tilted_tray_and_height():
    P, co, T = _scene_cloud(0.02, 0.01, 2.0, tilt_deg=4.0, seed=2)
    r = FT.fit_tray(P, co, rng=np.random.default_rng(0))
    assert r["ok"], r["why"]
    e_xy, e_z, e_yaw, e_tilt = _err(r["T"], T)
    assert e_xy < 0.008 and e_z < 0.004 and e_tilt < 1.5, (e_xy, e_z, e_yaw, e_tilt)


def test_better_than_fk_and_robust_to_occlusion():
    """the FK identity is 4-5 cm off; the fit must beat it by a wide margin with the gripper-end occlusion + cup in view."""
    P, co, T = _scene_cloud(0.045, -0.04, 5.0, occlude=True, seed=3)
    r = FT.fit_tray(P, co, rng=np.random.default_rng(0))
    assert r["ok"]
    e_xy, *_ = _err(r["T"], T)
    assert e_xy < 0.25 * np.linalg.norm(T[:2, 3])


def test_no_tray_fails_gracefully():
    P = np.random.default_rng(0).uniform([-0.4, -0.4, -0.8], [0.4, 0.4, -0.6], (2000, 3))      # only far floor: no plane in the gate
    r = FT.fit_tray(P, np.array([0.0, -0.3, 0.45]))
    assert not r["ok"] and np.allclose(r["T"], np.eye(4))


def test_re_express_round_trip():
    T = np.eye(4)
    T[:3, 3] = [0.03, -0.02, 0.001]
    yw = math.radians(4.0)
    T[:2, :2] = [[math.cos(yw), -math.sin(yw)], [math.sin(yw), math.cos(yw)]]
    p_fk = np.array([0.05, 0.01, 0.0])
    pv, Rv = FT.re_express(T, p_fk, np.eye(3))
    assert np.allclose(T[:3, :3] @ pv + T[:3, 3], p_fk, atol=1e-9)
    assert np.allclose(Rv[:, 2], T[:3, :3].T @ np.array([0.0, 0.0, 1.0]), atol=1e-9)
