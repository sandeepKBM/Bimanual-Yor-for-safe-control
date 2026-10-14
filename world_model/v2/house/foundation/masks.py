"""masks.py -- NON-privileged object masks for onboarding / first-frame prompts (numpy + cv2 only).

floor_gate_mask: the object rests alone on a KNOWN plane (floor z = floor_z, known from odometry / kinematics).  Mask = pixels whose 3-D point lies
  h_min < height-above-floor < h_max  AND  within `radius` (horizontal) of a seed point (where the robot placed the object, ~3 cm prior),
  largest connected component, small closing + hole fill.  Uses depth + camera pose only: no rendered segmentation.
iou: evaluation helper (the privileged rendered segmentation is used ONLY to score a mask, never to make one).
"""
from __future__ import annotations

import cv2
import numpy as np


def backproject(depth, K):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], -1)


def floor_gate_mask(depth, K, cam_in_world, seed_xy, floor_z=0.0, h_min=0.003, h_max=0.40, radius=0.25, min_px=30):
    P = backproject(depth, K)
    Pw = P @ cam_in_world[:3, :3].T + cam_in_world[:3, 3]
    h = Pw[..., 2] - floor_z
    d_xy = np.linalg.norm(Pw[..., :2] - np.asarray(seed_xy)[None, None], axis=-1)
    m = (depth > 0.05) & (h > h_min) & (h < h_max) & (d_xy < radius)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(depth.shape, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_px:
        return np.zeros(depth.shape, bool)
    out = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return filled.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0
