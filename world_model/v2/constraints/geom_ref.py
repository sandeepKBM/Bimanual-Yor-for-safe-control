"""geom_ref.py -- hand-coded reference geometry (independent of zono.py / ir.py): exact signed distance between two oriented boxes.
separated -> min over vertex-to-box and edge-to-edge distances (exact for convex polyhedra); overlapping -> minus the minimum overlap over the
15 SAT axes (= the minimal translation that separates them). Also point-to-box. Pure numpy."""
from __future__ import annotations

import itertools

import numpy as np

_SIGNS = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.abs(_SIGNS[i] - _SIGNS[j]).sum() == 2.0]


def verts(R, p, h):
    return p + (_SIGNS * np.asarray(h)) @ np.asarray(R).T


def point_box_dist(x, R, p, h):
    """Euclidean distance from x to the solid box (0 inside)."""
    q = np.asarray(R).T @ (x - p)
    return float(np.linalg.norm(np.maximum(np.abs(q) - h, 0.0)))


def seg_seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-14 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def box_box_sd(RA, pA, hA, RB, pB, hB):
    RA, RB = np.asarray(RA), np.asarray(RB)
    t = pB - pA
    axes = [RA[:, i] for i in range(3)] + [RB[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(RA[:, i], RB[:, j])
            n = np.linalg.norm(c)
            if n > 1e-9:
                axes.append(c / n)
    ov = []
    for ax in axes:
        rA = np.abs(RA.T @ ax) @ hA
        rB = np.abs(RB.T @ ax) @ hB
        ov.append(rA + rB - abs(t @ ax))
    if min(ov) > 0:                                            # no separating axis: overlapping
        return -float(min(ov))
    VA, VB = verts(RA, pA, hA), verts(RB, pB, hB)
    d = min([point_box_dist(v, RB, pB, hB) for v in VA] + [point_box_dist(v, RA, pA, hA) for v in VB])
    for (i, j) in _EDGES:
        for (k, l) in _EDGES:
            d = min(d, seg_seg_dist(VA[i], VA[j], VB[k], VB[l]))
    return float(d)
