"""tests for the ZED stereo demo (SIMULATION only; no hardware).

    cd /common/users/ss5772/bimanual
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<n != 0> PYTHONPATH=. python -m pytest -q world_model/v2/house/zed_demo/tests
(the MuJoCo-render test is skipped when no GL context can be created; the numpy tests always run.)
"""
import os

import numpy as np
import pytest

from world_model.v2.house.zed_demo import stereo as ST
from world_model.v2.house.zed_demo import rig as Z


def _speckle(h, w, seed=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.normal(size=(h, w)), 1.5) + 0.5 * gaussian_filter(rng.normal(size=(h, w)), 4.0)
    n = (n - n.min()) / (n.max() - n.min())
    return (40 + 200 * n).astype(np.uint8)


def test_sgbm_recovers_depth_of_fronto_parallel_textured_plane_numpy():
    """analytic pair: a fronto-parallel plane at Z shifts every pixel by d = f B / Z (right image = left shifted)."""
    K = Z.intrinsics(640, 360)
    Zt = 1.0
    d_true = K["f"] * K["B"] / Zt
    base = _speckle(360, 640 + 400)
    shift = int(round(d_true))
    left = base[:, 200:200 + 640]
    right = base[:, 200 + shift:200 + shift + 640]            # right(x) = left(x + d): a left feature at x appears at x - d in the right image
    L = np.repeat(left[:, :, None], 3, 2)
    Rr = np.repeat(right[:, :, None], 3, 2)
    disp, _ = ST.compute_disparity(L, Rr, ST.make_sgbm(numDisparities=128))
    depth = ST.depth_from_disparity(disp, K)
    core = depth[60:300, 200:600]
    assert np.isfinite(core).mean() > 0.85
    Zhat = np.nanmedian(core)
    Z_expected = K["f"] * K["B"] / shift                       # integer-disparity ground truth
    assert abs(Zhat - Z_expected) < 0.01, (Zhat, Z_expected)
    assert abs(Zhat - Zt) < 0.03


def test_cloud_projection_roundtrip():
    K = Z.intrinsics(640, 360)
    depth = np.full((360, 640), 1.3)
    origin = np.array([0.5, -0.2, 1.1])
    th = np.deg2rad(30.0)
    fwd = np.array([np.cos(th), 0, -np.sin(th)])
    up = np.array([np.sin(th), 0, np.cos(th)])
    zc = -fwd
    xc = np.cross(up, zc)
    R = np.stack([xc, up, zc], 1)
    P, idx = ST.depth_to_cloud(depth, K, origin, R)
    uv, zz = ST.project(P, K, origin, R)
    u0 = idx % 640
    v0 = idx // 640
    assert np.abs(uv[:, 0] - u0).max() < 1e-3 and np.abs(uv[:, 1] - v0).max() < 1e-3
    assert np.abs(zz - 1.3).max() < 1e-6


def test_zonotope_contains_its_segment_points_and_inflation_grows_it():
    from world_model.v2.perception_zono.fit import fit_hull_zono
    from world_model.v2.perception_zono.zonoutil import ZUnion
    from world_model.v2.house.zed_demo import perceive as PC
    rng = np.random.default_rng(1)
    # a tilted, noisy box surface patch (two visible faces) like a stereo segment
    n = 4000
    top = np.column_stack([rng.uniform(-0.1, 0.1, n), rng.uniform(-0.05, 0.05, n), np.full(n, 0.08)])
    front = np.column_stack([rng.uniform(-0.1, 0.1, n), np.full(n, -0.05), rng.uniform(0.0, 0.08, n)])
    P = np.concatenate([top, front]) + rng.normal(0, 0.002, (2 * n, 3))
    P = P @ np.array([[np.cos(0.4), -np.sin(0.4), 0], [np.sin(0.4), np.cos(0.4), 0], [0, 0, 1.0]]).T + [4.0, 5.0, 0.7]
    for order in (2, 3, 4):
        Zo = fit_hull_zono(P, order=order)
        assert ZUnion([Zo]).contains(P).mean() == 1.0, order
    Zi = PC.inflate_world(Zo, 0.02)
    assert ZUnion([Zi]).contains(P).mean() == 1.0
    v0 = ZUnion([Zo]).volume_sum()
    assert ZUnion([Zi]).volume_sum() > v0
    Zx = PC.extrude_unseen(Zo, (0.0, 1.0))
    assert ZUnion([Zx]).contains(P).mean() == 1.0


def _gl_ok():
    try:
        import mujoco
        sp = mujoco.MjSpec()
        sp.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1, 1, 1])
        m = sp.compile()
        r = mujoco.Renderer(m, 32, 32)
        r.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gl_ok(), reason="no MuJoCo GL context (set MUJOCO_GL=egl and a free CUDA_VISIBLE_DEVICES)")
def test_mujoco_rendered_textured_plane_stereo_depth(tmp_path):
    """two rendered parallel cameras looking at a speckle-textured wall at Z = 1.2 m: SGBM depth within 1.5 cm (median) of the true z."""
    import mujoco
    import cv2
    sp = mujoco.MjSpec()
    sp.visual.global_.offwidth = 1280
    sp.visual.global_.offheight = 720
    sp.visual.headlight.ambient = [0.6, 0.6, 0.6]
    png = Z.make_speckle_png(tmp_path / "s.png")
    sp.add_texture(name="t", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
    mat = sp.add_material(name="mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "t"
    mat.texuniform = True
    mat.texrepeat = [2.5, 2.5]
    Zwall = 1.2
    sp.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[2.0, 0.02, 1.5], pos=[0, Zwall, 0], material="mat")
    q = np.zeros(4)
    # MuJoCo camera looks along its -z; rotate so -z -> world +y (x right = world +x, y up = world +z)
    mujoco.mju_mat2Quat(q, np.stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]], 1).ravel())
    B = Z.ZED_BASELINE
    sp.worldbody.add_camera(name="L", pos=[-B / 2, 0, 0], quat=q.tolist(), fovy=Z.ZED_FOVY)
    sp.worldbody.add_camera(name="R", pos=[+B / 2, 0, 0], quat=q.tolist(), fovy=Z.ZED_FOVY)
    m = sp.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 720, 1280)
    r.update_scene(d, camera="L")
    L = r.render().copy()
    r.update_scene(d, camera="R")
    Rr = r.render().copy()
    r.close()
    K = Z.intrinsics(1280, 720)
    disp, _ = ST.compute_disparity(L, Rr, ST.make_sgbm())
    depth = ST.depth_from_disparity(disp, K)
    # the wall's front face is at y = Zwall - 0.02
    z_true = Zwall - 0.02
    core = depth[150:570, 300:1100]
    assert np.isfinite(core).mean() > 0.85
    assert abs(np.nanmedian(core) - z_true) < 0.015, (np.nanmedian(core), z_true)
