"""tests for the pure-geometry parts of ocs/fp_render.py (no GL, no GPU, no recordings).  SIMULATION ONLY."""
import numpy as np

from world_model.v2.house.ocs import fp_render as FR, util as U


def _rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_K_principal_point_and_focal():
    K = FR.K_from_fovy(53.1, 320, 180)
    assert np.isclose(K[0, 2], 159.5) and np.isclose(K[1, 2], 89.5) and np.isclose(K[0, 0], K[1, 1])
    assert np.isclose(2 * np.degrees(np.arctan(90.0 / K[1, 1])), 53.1)


def test_mujoco_to_opencv_camera_convention():
    # MuJoCo camera: x right, y up, looks along -z.  OpenCV: x right, y down, looks +z.  A point straight ahead (-z_mj) has z_cv > 0; a point above (+y_mj) has y_cv < 0.
    R_cv = np.eye(3) @ FR.FLIP
    ahead_mj, up_mj = np.array([0, 0, -1.0]), np.array([0, 1.0, 0])
    assert (R_cv.T @ ahead_mj)[2] > 0 and (R_cv.T @ up_mj)[1] < 0


def test_cv_cam_in_tray_composes_with_world_poses():
    rng = np.random.default_rng(0)
    for _ in range(10):
        Rt, pt = _rot(rng), rng.normal(size=3)
        Rc, oc = _rot(rng), rng.normal(size=3)
        T = FR.cv_cam_in_tray(Rt, pt, oc, Rc)
        x_cam = rng.normal(size=3) + np.array([0, 0, 2.0])                    # OpenCV camera-frame point
        x_world = oc + (Rc @ FR.FLIP) @ x_cam
        x_tray = Rt.T @ (x_world - pt)
        assert np.allclose(T[:3, :3] @ x_cam + T[:3, 3], x_tray)


def _topdown_scene(W=160, H=90, fovy=53.1, h_cam=0.30, cup_r=0.0325, cup_h=0.08, cup_xy=(0.02, -0.01), block=None):
    """analytic z-depth of a camera straight above a tray (top face z=0): flat plane + a cylinder (the cup) [+ a block outside the tray rectangle]."""
    K = FR.K_from_fovy(fovy, W, H)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_dir, y_dir = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    depth = np.full((H, W), h_cam)                                              # plane z = 0 seen from above: z-depth = h_cam
    X, Y = x_dir * (h_cam - cup_h), y_dir * (h_cam - cup_h)                      # where the ray crosses the cup top plane
    # camera frame: tray x = cam x, tray y = -cam y (looking down with OpenCV axes flipped about x): handled by the pose below
    Xt, Yt = X, -Y
    inside = np.hypot(Xt - cup_xy[0], Yt - cup_xy[1]) < cup_r
    depth[inside] = h_cam - cup_h
    truth = inside.copy()
    if block is not None:                                                        # a 5 cm tall block centred at tray (0.22, 0): outside the 0.15 half-length
        hb = 0.05
        Xb, Yb = x_dir * (h_cam - hb), -y_dir * (h_cam - hb)
        ib = (np.abs(Xb - block[0]) < 0.02) & (np.abs(Yb - block[1]) < 0.02)
        depth[ib] = h_cam - hb
    return depth, K, truth


def test_tray_gate_mask_selects_cup_not_plane_or_outside_block():
    h_cam = 0.30
    # camera above the tray centre looking straight down: MuJoCo camera R = diag(1,1,1) (looks along -z world = down)
    T = FR.cv_cam_in_tray(np.eye(3), np.zeros(3), np.array([0.0, 0.0, h_cam]), np.eye(3))
    depth, K, truth = _topdown_scene(h_cam=h_cam, block=(0.22, 0.0))
    mask, pts = FR.tray_gate_mask(depth, K, T, U.HX, U.HY, shrink=0.02, h_lo=0.005, h_hi=0.11, min_px=20)
    iou = (mask & truth).sum() / (mask | truth).sum()
    assert iou > 0.95, iou
    assert np.allclose(pts[:, 2], 0.08, atol=1e-6)                               # points are the cup top, 8 cm above the tray top
    assert np.abs(pts[:, 0].mean() - 0.02) < 2e-3 and np.abs(pts[:, 1].mean() + 0.01) < 2e-3   # centroid in the TRAY frame = the cup xy


def test_tray_gate_mask_empty_when_nothing_above_plane():
    T = FR.cv_cam_in_tray(np.eye(3), np.zeros(3), np.array([0.0, 0.0, 0.3]), np.eye(3))
    depth = np.full((90, 160), 0.3)
    mask, pts = FR.tray_gate_mask(depth, FR.K_from_fovy(53.1, 160, 90), T, U.HX, U.HY, min_px=20)
    assert mask.sum() == 0 and len(pts) == 0


def test_canonical_yaw_keeps_axis_and_is_a_rotation():
    rng = np.random.default_rng(1)
    for _ in range(20):
        R = _rot(rng)
        Rc = FR.canonical_yaw(R)
        assert np.allclose(Rc[:, 2], R[:, 2]) and np.allclose(Rc.T @ Rc, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(Rc), 1.0)
        # x-axis lies in the plane spanned by the tray x and the cup axis
        assert abs(np.linalg.det(np.stack([Rc[:, 0], np.array([1.0, 0, 0]), Rc[:, 2]], 1))) < 1e-9


def test_wrist_depth_noise_is_seeded_and_respects_valid_range():
    from world_model.v2.tray_rep.perception import depth_sim as D
    spec = D.CAMERAS["wrist_r"]
    K = FR.K_from_fovy(spec.fovy, 160, 90)
    z = np.full((90, 160), 0.4)
    z[:, :20] = 0.05                                                              # nearer than z_min = 0.10 m -> invalid
    a = FR.wrist_depth_noise(z, K, spec, np.random.default_rng(3))
    b = FR.wrist_depth_noise(z, K, spec, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert (a[:, :20] > 0).mean() < 0.01                                         # only the rare gross outliers (0.2 %) can land back in range
    ok = a[:, 40:120] > 0
    assert 0.9 < ok.mean() < 1.0                                                   # dropout ~2 % + grazing
    assert abs(np.median(a[:, 40:120][ok]) - 0.4) < 2e-3
    robot = np.zeros_like(z, bool)
    robot[:, 100:] = True
    c = FR.wrist_depth_noise(z, K, spec, np.random.default_rng(3), robot)
    assert (c[:, 100:] == 0).all()
