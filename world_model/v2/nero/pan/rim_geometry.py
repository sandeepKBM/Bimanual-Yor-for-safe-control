"""pan/rim_geometry.py -- constants and helpers that live code still needs from the archived friction-pinch pipeline.
Moved VERBATIM (values and code unchanged, 2026-11-03) from pan/under_rim.py (RIGHT_APPROACH, RIGHT_JAW, APPROACH_IN, EDGE_CLEAR,
_perturb_pose) and pan/skills/grasp.py (_rot).  SIMULATION ONLY."""
from __future__ import annotations

import mujoco
import numpy as np

RIGHT_APPROACH, RIGHT_JAW = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])   # from the far side towards the pan, fingers above/below
APPROACH_IN = 0.04            # ASSUMED (11-04, was design 0.08): stand-off distance the right claw travels in [m].  With the NYU gripper the 8 cm stand-off pose has 0 IK solutions (tilt 0-30 deg, checked 11-04); 4 cm is the largest tried that has solutions
EDGE_CLEAR = 0.006            # design: right fingertips stop this far short of the stove edge [m]


def _perturb_pose(m, d, centre_xy, dx, dy, yaw_deg):
    """Move pan + lid + hidden puck rigidly: rotate yaw about the pan's nominal centre (z axis), then shift (dx, dy)."""
    c, s_ = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    qz = np.array([np.cos(np.radians(yaw_deg) / 2), 0, 0, np.sin(np.radians(yaw_deg) / 2)])
    for name in ("pan", "lid", "hidden"):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            continue
        a = m.jnt_qposadr[m.body(name).jntadr[0]]
        pos = d.qpos[a:a + 3].copy()
        rel = pos - np.array([centre_xy[0], centre_xy[1], 0.0])
        d.qpos[a:a + 3] = Rz @ rel + np.array([centre_xy[0] + dx, centre_xy[1] + dy, 0.0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qz, d.qpos[a + 3:a + 7].copy())
        d.qpos[a + 3:a + 7] = q
    mujoco.mj_forward(m, d)


def _rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
