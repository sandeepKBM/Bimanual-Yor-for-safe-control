"""stereo tests: pure numpy / cv2 (no GL, no sim).  Run: pytest -q world_model/v2/house/stereo/tests"""
import math

import numpy as np

from world_model.v2.house.stereo import cfg as C, payload as PL, adapters as AD, backends as BK


def test_cfg_geometry():
    c = C.StereoCfg()
    K = c.K()
    assert abs(K["f"] - (180.0 / math.tan(math.radians(35)))) < 1e-6
    assert c.n_disp() % 16 == 0 and c.n_disp() >= c.f * c.baseline / c.z_min
    R = C.head_axes(c)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
    fwd = -R[:, 2]
    assert abs(math.degrees(math.asin(-fwd[2])) - c.pitch_deg) < 1e-6


def test_backend_interface_and_pad():
    c = C.StereoCfg(W=160, H=90, num_disp=32, pad_left=True)
    b = BK.make_backend(c)
    b.warmup(n=1)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (90, 200, 3), dtype=np.uint8)
    L, R = base[:, 20:180], base[:, 24:184]           # a point at left column x sits at right column x - 4: disparity +4 px
    z, ok, ms = b.depth(L, R, c.K(), c.baseline)
    assert z.shape == (90, 160) and ok.dtype == bool and ms > 0
    assert np.isnan(z[~ok]).all()
    d = c.f * c.baseline / np.nanmedian(z)
    assert abs(d - 4.0) < 0.6
    assert ok[:, :32].mean() > 0.2                    # the padded strip is no longer dead


def test_head_frame_selfmask_and_zs():
    c = C.StereoCfg(W=64, H=36)
    depth = np.full((36, 64), 2.0, np.float32)
    mask = np.zeros((36, 64), bool)
    mask[:, :32] = True
    fr = dict(tick=3, depth=depth, origin=np.zeros(3), R=np.eye(3), self_mask=mask)
    rows, cols, x, y = AD.grid_rays(c, 4)
    hf = AD.frame_to_head_frame(fr, c, stride=4, zs_grid=np.full(x.shape, 2.5), z_lim=(-10, 10))
    assert hf["pix"][:, 1].min() >= 8                  # masked half dropped (grid col index >= 8 for stride 4)
    assert np.allclose(hf["z"], 2.0) and np.allclose(hf["zs"], 2.5)
    assert np.allclose(hf["pts"][:, 2], -2.0)           # camera looks along -z with identity R


def test_slide_window():
    a = np.concatenate([np.random.default_rng(0).uniform(-0.15, 0.15, 2000), [0.20, 0.21, -0.19]])   # plate + 3 stray points
    c, w = PL._slide(a, 0.15, 0.002, 0.97)
    assert abs(c) < 0.01 and w < 0.02


def _cyl_depth(o, d, base, r, h):
    """ray (o + t d) vs a vertical cylinder (axis through base, +z, radius r, height h): first hit t or inf (side + top rim ignored for the inside)."""
    ox, oy = o[0] - base[0], o[1] - base[1]
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    cc = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * cc
    t = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a + 1e-12), np.inf)
    z = o[2] + t * d[:, 2]
    return np.where((t > 0) & (z >= base[2]) & (z <= base[2] + h), t, np.inf)


def test_payload_recovers_synthetic_tray_and_cup():
    c = C.StereoCfg(W=320, H=180)
    K = c.K()
    Rc = C.head_axes(c)
    lift = 0.0
    oL = np.asarray(c.mount) + np.array([0, 0, lift]) - 0.5 * c.baseline * Rc[:, 0]
    pt, Rt = np.array([0.45, -0.21, 0.85]), np.eye(3)
    v, u = np.mgrid[0:c.H, 0:c.W]
    dirs_cam = np.stack([(u - K["cx"]) / K["f"], -(v - K["cy"]) / K["f"], -np.ones_like(u, float)], -1).reshape(-1, 3)
    dirs = dirs_cam @ Rc.T
    t_plane = (pt[2] - oL[2]) / dirs[:, 2]
    hit = oL + t_plane[:, None] * dirs
    on_plate = (np.abs(hit[:, 0] - pt[0]) < 0.15) & (np.abs(hit[:, 1] - pt[1]) < 0.10) & (t_plane > 0)
    cup_xy = np.array([pt[0] - 0.045, pt[1]])
    base = np.array([cup_xy[0], cup_xy[1], pt[2]])
    # include the far inner wall by testing both roots: keep the nearest only (outer wall) -- sufficient for the fit
    t_cup = _cyl_depth(oL, dirs, base, 0.032, 0.08)
    t = np.where(on_plate, t_plane, np.inf)
    t = np.minimum(t, t_cup)
    zdep = np.where(np.isfinite(t), t * (-dirs_cam[:, 2] * 0 + 1) * 1.0, np.nan)
    # t is a ray length along unit-ish dirs (dirs built with z=-1 -> t = z-depth)
    depth = zdep.reshape(c.H, c.W).astype(np.float32)
    rgb = np.full((c.H, c.W, 3), 120, np.uint8)
    rgb[(t_cup < t_plane).reshape(c.H, c.W) & np.isfinite(t_cup).reshape(c.H, c.W)] = (200, 30, 30)
    fr = dict(depth=depth, left=rgb)
    e = PL.estimate(fr, c, (pt + np.array([0.0, 0.015, 0.0]), Rt), lift)         # FK prior off by 15 mm in y
    assert e["ok"] and e["tray"] is not None
    assert abs(math.degrees(e["tray"]["tilt"])) < 0.5
    assert abs(e["tray"]["p"][2] - pt[2]) < 0.003
    assert e["cup"]["ok"]
    rel = Rt.T @ (np.array([cup_xy[0], cup_xy[1], pt[2]]) - e["tray"]["p"])
    assert np.linalg.norm((e["cup"]["p"][:2] - rel[:2])) < 0.006, (e["cup"]["p"], rel)
    assert e["cup"]["tilt"] < math.radians(6)
