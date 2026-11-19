"""Owner gate (11-19): the four lab cameras of the room (`room4` mode of nero/contents/plan_images.py; USER 11-19: lab-fixed sensors allowed,
Codex gets their RGB photos, no scene numbers; design note docs/status/geometry_probe_design_2026-11-19.md §3).  Task = box_bookcase (USER 11-19:
the fixed-base box scene is dropped).  SIMULATION ONLY, no network.
PYTHONPATH=. MUJOCO_GL=osmesa python -m pytest -q world_model/v2/nero/contents/tests/test_room4_gate.py   (or MUJOCO_GL=egl on a free GPU)

Checks, built independently of the module's renderer:
  1. work area = xy box of the static scene (world-body geoms, no planes), computed here; room = that box + ROOM_MARGIN_M on every side;
     four labels in order, each camera at the middle of its own wall at ROOM_CAM_H_M, all aimed at the work-area centre at counter height;
  2. render_views(box_bookcase, 2, room4): four distinct images of ROOM_SIZE, deterministic, the room4 note, ASSUMED tags, no ZED mount claimed;
  3. what each camera sees (segmentation pass, own camera construction from pos / look_at): the crate is visible in every photo, the bookcase in
     every photo; the hidden contents in none (no hidden truth in the images);
  4. room4 refuses a fixed-base task.
Owner check 11-19 (not a test: renderer-dependent hashes): the zed / zed+look / zed+wrist images are byte-identical to the pre-edit module (CPU osmesa)."""
import hashlib
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as iio
import mujoco

from world_model.v2.nero.base import scene as BS
from world_model.v2.nero.base import smoke as SM
from world_model.v2.nero.contents import plan_images as PI
from world_model.v2.nero.contents import scenes as SC

TASK, I = "box_bookcase", 2
LABELS = ["room_front", "room_left", "room_back", "room_right"]
MIN_PX = 100                # ASSUMED (this gate): "visible" = at least 100 pixels of the body at ROOM_SIZE


@pytest.fixture(scope="module")
def scene():
    _, w = SM.behind_world(TASK, I, nominal=True)          # the room4 layout numbers are the nominal scene (varied geometry: test_geom_probe_gate / geom_noise study)
    m = BS.build_spec(w).compile()
    d = mujoco.MjData(m)
    SC.settle(m, d, PI.SETTLE_S)
    mujoco.mj_forward(m, d)
    return w, m, d


def _work_area(m, d):
    """own computation: xy box of every world-body geom except planes (AABB of the rotated box/mesh by its xy half-extent bound)."""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != 0 or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c = d.geom_xpos[g][:2]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3)
            e = np.abs(R[:2, :]) @ m.geom_size[g]
        else:
            e = np.full(2, m.geom_rbound[g])
        lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
    return lo, hi


def test_layout(scene):
    w, m, d = scene
    lo, hi = _work_area(m, d)
    assert lo == pytest.approx([-2.45, -0.77], abs=0.02) and hi == pytest.approx([0.80, 0.37], abs=0.02), (lo, hi)   # the bookcase scene, world 2
    cams = PI.room4_cameras(m, d, w)
    assert [c["label"] for c in cams] == LABELS
    cx, cy = 0.5 * (lo + hi)
    cz = float(SC.support_tops(w)["counter"]["top"])
    g = PI.ROOM_MARGIN_M
    want = {"room_front": (hi[0] + g, cy), "room_left": (cx, hi[1] + g), "room_back": (lo[0] - g, cy), "room_right": (cx, lo[1] - g)}
    for c in cams:
        p, a = np.asarray(c["pos"], float), np.asarray(c["look_at"], float)
        assert p[:2] == pytest.approx(want[c["label"]], abs=1e-6) and p[2] == pytest.approx(PI.ROOM_CAM_H_M), c
        assert a == pytest.approx([cx, cy, cz], abs=1e-6), c


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    d = tmp_path_factory.mktemp("room4")
    return PI.render_views(TASK, I, "room4", d / "a"), PI.render_views(TASK, I, "room4", d / "b")


def test_render_views_record(views):
    a, b = views
    assert a.cameras == LABELS and len(a.paths) == 4
    ha = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in a.paths]
    hb = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in b.paths]
    assert ha == hb, "not deterministic"
    assert len(set(ha)) == 4, "two cameras gave the same image"
    w, h = PI.ROOM_SIZE
    for p in a.paths:
        assert iio.imread(p).shape[:2] == (h, w), p
    assert a.note == PI.image_note("room4") and a.note.startswith("IMAGES: images 1-4 are fixed lab cameras"), a.note
    ass = a.assumptions
    for k in ("room_margin_m", "cam_height_m", "fovy_deg"):
        assert ass[k]["tag"].startswith("ASSUMED"), (k, ass[k])
    assert "work_area_xy" in ass
    assert not any("zed" in str(k).lower() for k in ass), sorted(ass)          # room4 must not claim a ZED mount
    assert "depth" in str(ass.get("noise", "")).lower()


def test_seen_and_hidden(scene):
    w, m, d = scene
    W, H = PI.ROOM_SIZE
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    m.vis.global_.fovy = PI.ROOM_FOVY_DEG
    body = lambda name: {g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name == name}
    crate, hidden = body("obj"), body("contents")
    shelf = {g for g in range(m.ngeom) if m.geom(g).name.startswith("bookcase")}
    assert crate and hidden and shelf
    m.vis.quality.offsamples = 0     # segmentation ids must not be blended: with the default 4x MSAA, EGL averages id colours at edges (11-19: 1 'contents' px on room_back; 81/81 sub-pixel ray casts hit tab_left)
    r = mujoco.Renderer(m, H, W)
    counts = {}
    try:
        r.enable_segmentation_rendering()
        for c in PI.room4_cameras(m, d, w):
            p, a = np.asarray(c["pos"], float), np.asarray(c["look_at"], float)
            f = (a - p) / np.linalg.norm(a - p)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = float(np.linalg.norm(a - p))
            cam.lookat[:] = a
            cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
            cam.elevation = float(np.degrees(np.arcsin(f[2])))
            r.update_scene(d, camera=cam)
            seg = r.render()
            ids, isgeom = seg[..., 0], seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            n = lambda S: int((np.isin(ids, list(S)) & isgeom).sum())
            counts[c["label"]] = dict(crate=n(crate), bookcase=n(shelf), contents=n(hidden))
    finally:
        r.close()
    for lab in LABELS:
        c = counts[lab]
        assert c["contents"] == 0, (lab, counts)
        assert c["crate"] >= MIN_PX and c["bookcase"] >= MIN_PX, (lab, counts)


def test_room4_refuses_fixed_task(tmp_path):
    with pytest.raises(ValueError):
        PI.render_views("box", 0, "room4", tmp_path)
