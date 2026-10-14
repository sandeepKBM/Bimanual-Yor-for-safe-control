"""offline unit tests (numpy only): floor-gate mask recovers a box on the floor; IPC layout helpers agree between client and server."""
import numpy as np

from world_model.v2.house.foundation import masks as M
from world_model.v2.house.foundation import stereo_fs as SF


def test_floor_gate_mask_box():
    H, W = 120, 160
    K = np.array([[100.0, 0, 79.5], [0, 100.0, 59.5], [0, 0, 1]])
    # camera 1 m above the floor looking straight down (OpenCV: z forward = world -z)
    Tcw = np.eye(4)
    Tcw[:3, :3] = np.diag([1.0, -1.0, -1.0])
    Tcw[:3, 3] = [0, 0, 1.0]
    depth = np.full((H, W), 1.0, np.float32)
    depth[40:80, 60:100] = 0.9            # 10 cm tall box
    m = M.floor_gate_mask(depth, K, Tcw, seed_xy=[0, 0], floor_z=0.0, h_min=0.03, radius=0.5)
    assert m[60, 80] and not m[5, 5]
    assert M.iou(m, np.pad(np.ones((40, 40), bool), ((40, 40), (60, 60)))) > 0.9


def test_layout_total():
    lay = SF._layout(64, 48)
    assert lay["total"] == 64 * 48 * (3 + 3 + 4 + 1)
