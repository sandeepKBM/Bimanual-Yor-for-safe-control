"""depth-side geometry of the proposals (pure numpy; no GL, no GPU)."""
import math

import numpy as np

from world_model.v2.house.pick import propose as PR


def _box_cloud(c, size, yaw=0.0, n=4000, seed=0, front_only=True):
    """points on the TOP face and the camera-facing side faces of a box (camera at -x looking +x, above): what a head camera sees."""
    rng = np.random.default_rng(seed)
    sx, sy, sz = size
    top = np.stack([rng.uniform(-sx / 2, sx / 2, n), rng.uniform(-sy / 2, sy / 2, n), np.full(n, sz / 2)], 1)
    front = np.stack([np.full(n // 2, -sx / 2), rng.uniform(-sy / 2, sy / 2, n // 2), rng.uniform(-sz / 2, sz / 2, n // 2)], 1)
    P = np.concatenate([top, front])
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return P @ R.T + np.asarray(c)


def test_top_face_footprint_is_unbiased_by_visible_side_faces():
    c = np.array([0.55, -0.21, 0.78])
    o = PR.estimate_obb(_box_cloud(c, (0.20, 0.30, 0.20), yaw=0.0))
    assert np.allclose(o["centroid"][:2], c[:2], atol=0.004)                 # a whole-cloud mean would be ~2.5 cm toward the camera
    a, b, h = o["extent"]
    assert abs(a - 0.30) < 0.01 and abs(b - 0.20) < 0.01 and abs(h - 0.20) < 0.01
    assert abs(abs(math.sin(o["yaw"])) - 1.0) < 0.05                          # long axis along robot y


def test_isotropic_footprint_snaps_to_robot_axes():
    o = PR.estimate_obb(_box_cloud([0.5, 0.2, 0.7], (0.055, 0.055, 0.1), yaw=0.3, n=3000))
    assert o["yaw"] in (0.0, ) or abs(abs(o["yaw"]) - math.pi / 2) < 1e-9 or abs(o["yaw"]) < 1e-9
    assert abs(o["extent"][0] - 0.055) < 0.015


def test_outliers_are_rejected():
    P = _box_cloud([0.5, 0.0, 0.75], (0.05, 0.05, 0.06), n=1500)
    P = np.concatenate([P, np.array([[0.9, 0.3, 1.2], [0.1, -0.4, 0.5]])])     # gross depth outliers (0.2 % of pixels, 5-20 cm)
    o = PR.estimate_obb(P)
    assert np.allclose(o["centroid"][:2], [0.5, 0.0], atol=0.006)


def test_fuse_estimates_median():
    mk = lambda cx: dict(_gt="a", centroid=np.array([cx, 0.0, 0.7]), extent=(0.05, 0.05, 0.1), top_z=0.75, base_z=0.65, yaw=0.0, box2d=[0, 0, 5, 5])
    f = PR.fuse_estimates([[mk(0.50)], [mk(0.52)], [mk(0.90)]])               # one wild frame
    assert len(f) == 1 and abs(f[0]["centroid"][0] - 0.52) < 1e-9 and f[0]["n_frames"] == 3


def test_robot_frame_roundtrip():
    base = np.array([3.1, 6.09, -2.33])
    p_robot = np.array([[0.5, -0.2, 0.8]])
    c, s = math.cos(base[2]), math.sin(base[2])
    p_world = np.array([[base[0] + c * 0.5 - s * -0.2, base[1] + s * 0.5 + c * -0.2, 0.8]])
    assert np.allclose(PR.to_robot_frame(p_world, base), p_robot)


def test_draw_boxes_marks_selected_and_modes():
    im = np.zeros((360, 640, 3), np.uint8)
    props = [dict(id=1, box2d=[10, 10, 60, 90], label="bottle", mode="single", arm="left"), dict(id=2, box2d=[200, 100, 400, 250], label="crate", mode="bimanual", arm="both")]
    out = PR.draw_boxes(im, props, selected=2)
    assert out.shape == im.shape and out.sum() > 0 and im.sum() == 0              # drew on a copy
