"""Run: PYTHONPATH=. MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=4 python -m pytest -q world_model/v2/vlm/grounding/tests   (SIMULATION; CPU + EGL)"""
import os

import numpy as np
import pytest

from world_model.v2.vlm.grounding import common as C
from world_model.v2.vlm.grounding import downstream as D


def test_parsers_native_formats():
    assert C.parse_points("qwen3-vl-8b", '```json\n[{"point_2d": [512, 300], "label": "x"}]\n```') == [(512.0, 300.0)]
    assert C.parse_boxes("qwen3.5-9b", '[{"bbox_2d": [1, 2, 30, 40], "label": "x"}]') == [(1.0, 2.0, 30.0, 40.0)]
    assert C.parse_points("robobrain2.5-4b", "[(512, 300), (1, 2)]")[0] == (512.0, 300.0)
    assert C.parse_boxes("robobrain2.5-4b", "[10, 20, 30, 40]") == [(10.0, 20.0, 30.0, 40.0)]
    assert C.parse_points("molmo2-er", '<points coords="1 1 523 412">mug</points>') == [(523.0, 412.0)]
    assert C.parse_points("molmo2-er", '<point x="61.5" y="40.1" alt="a">a</point>') == [(61.5, 40.1)]
    assert C.parse_points("qwen3-vl-8b", "I cannot see it") == []
    assert C.parse_boxes("molmo2-er", '<boxes coords="1 1 470 580 490 600 2 480 550 500 570">x</boxes>') == [(470.0, 580.0, 490.0, 600.0)]
    assert C.parse_boxes("molmo2-er", "x1 535 506 x2 601 600") == [(535.0, 506.0, 601.0, 600.0)]
    assert C.parse_boxes("molmo2-er", "[(0.262, 0.262), (0.309, 0.294)]") == [(262.0, 262.0, 309.0, 294.0)]


def test_to_px_relative_1000():
    p = C.to_px("qwen3-vl-8b", (500, 500), 1280, 720)
    assert np.allclose(p, (640, 360))
    assert np.allclose(C.to_px("qwen3-vl-8b", (10, 20, 30, 40), 1280, 720, "abs"), (10, 20, 30, 40))


def test_metrics():
    m = np.zeros((10, 10), bool)
    m[2:5, 3:7] = True
    assert C.mask_bbox(m) == (3.0, 2.0, 7.0, 5.0)
    assert C.point_in_mask(m, (3.5, 2.5)) and not C.point_in_mask(m, (0.5, 0.5)) and not C.point_in_mask(m, (-1, 3))
    assert C.box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert abs(C.box_iou((0, 0, 2, 2), (1, 0, 3, 2)) - 1 / 3) < 1e-9


def test_downstream_helpers():
    assert D.iou_sets([1, 2, 3], [2, 3, 4]) == 0.5
    z = np.array([1.0, 1.01, 1.02] * 10 + [2.0] * 40)       # near group (30 pts) is kept over the larger far group
    g = D.cleanup_house(z)
    assert len(g) == 30 and z[g].max() < 1.1
    P = np.concatenate([np.random.default_rng(0).normal(0, 0.003, (200, 3)), np.random.default_rng(1).normal(0, 0.003, (50, 3)) + 0.1])
    assert len(D.cleanup_tab(P)) > 150


def test_tabletop_render_aligned_with_cloud():
    pytest.importorskip("mujoco")
    from world_model.v2.perception_zono import geometry as G, sensing as S
    from world_model.v2.vlm.grounding import make_tabletop as MT
    try:
        objs, colors, _ = MT.build_scene("mix4", 0)
        sc = G.Scene(objs)
        rm, gl = MT.render_xml(objs, colors)
        import mujoco
        d = mujoco.MjData(rm)
        mujoco.mj_forward(rm, d)
        cl = S.capture(sc, "head_hi", "n3mm", 0)
        v = cl.views[0]
        rgb, seg = MT.render_rgb_seg(rm, d, MT.free_cam(v.origin, v.R), gl)
    except Exception as e:                              # no EGL/GPU
        pytest.skip(f"render unavailable: {e}")
    uv, z = MT.project(cl.pts, v.origin, v.R, MT.F_PX)
    u, w = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < MT.W) & (w >= 0) & (w < MT.H)
    assert ok.mean() > 0.99
    assert (seg[w[ok], u[ok]] == cl.label[ok]).mean() > 0.98        # measured 0.994-0.997
