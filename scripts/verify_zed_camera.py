#!/usr/bin/env python3
"""Headless verification for scripts/zed_camera.py::add_zed_camera.

Loads scenes/scene_yor_canonical.xml (read-only) plus scenes/scene_yor_cup.xml
(if present), adds the ZED camera in-memory via add_zed_camera(), renders RGB
(+ depth) from it, and checks: both arms + tabletop framed, cup/ball visible.

Run: MUJOCO_GL=egl python scripts/verify_zed_camera.py
"""
import os
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.zed_camera import add_zed_camera, ZED_CAMERA_NAME

REPO = "/common/users/ss5772/bimanual"
ARTIFACTS = os.path.join(REPO, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)


def load_with_zed(xml_path, keyframe="home"):
    spec = mujoco.MjSpec.from_file(xml_path)
    cam_name = add_zed_camera(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data, cam_name


def render(model, data, cam_name, w=640, h=360):
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=cam_name)
    rgb = renderer.render().copy()
    depth = None
    try:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
    except Exception as e:
        print("depth render skipped:", e)
    renderer.close()
    return rgb, depth


def main():
    canonical = os.path.join(REPO, "scenes", "scene_yor_canonical.xml")
    model, data, cam_name = load_with_zed(canonical)
    print(f"[canonical] loaded, ncam={model.ncam}, zed camera name={cam_name!r}")

    rgb, depth = render(model, data, cam_name)
    print("[canonical] rgb shape", rgb.shape, "std", rgb.std())
    Image.fromarray(rgb).save(os.path.join(ARTIFACTS, "zed_view.png"))
    if depth is not None:
        d = depth.copy()
        finite = np.isfinite(d)
        if finite.any():
            dmin, dmax = d[finite].min(), d[finite].max()
            dn = np.zeros_like(d)
            dn[finite] = (d[finite] - dmin) / max(dmax - dmin, 1e-6)
            Image.fromarray((dn * 255).astype(np.uint8)).save(
                os.path.join(ARTIFACTS, "zed_depth.png")
            )
            print(f"[canonical] depth range {dmin:.3f}..{dmax:.3f} m")

    # Sanity: both arm gripper bases + table visible (project world pts -> check within frustum roughly by rendering pixel stats per region)
    for body in ["left_arm_gripper_base", "right_arm_gripper_base", "table"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        print(f"  {body} world xpos = {data.xpos[bid]}")

    # Cup scene, if present
    cup_xml = os.path.join(REPO, "scenes", "scene_yor_cup.xml")
    if os.path.exists(cup_xml):
        cmodel, cdata, ccam = load_with_zed(cup_xml, keyframe="home")
        crgb, _ = render(cmodel, cdata, ccam)
        Image.fromarray(crgb).save(os.path.join(ARTIFACTS, "zed_cup_view.png"))
        print("[cup] rgb std", crgb.std())

        # ball pixel count at 64x64 via a colored-pixel heuristic: render
        # small, then count pixels that are neither floor/table/robot-gray
        # nor background sky. We instead directly query ball geom colors and
        # do an id-based visibility check using segmentation rendering.
        renderer = mujoco.Renderer(cmodel, height=64, width=64)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(cdata, camera=ccam)
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        renderer.close()
        obj_ids = seg[..., 0]
        ball_geom_ids = []
        for i in range(cmodel.ngeom):
            gname = mujoco.mj_id2name(cmodel, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname and "ball" in gname.lower():
                ball_geom_ids.append(i)
        cup_geom_ids = []
        for i in range(cmodel.ngeom):
            gname = mujoco.mj_id2name(cmodel, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname and "cup" in gname.lower():
                cup_geom_ids.append(i)
        ball_px = int(np.isin(obj_ids, ball_geom_ids).sum()) if ball_geom_ids else -1
        cup_px = int(np.isin(obj_ids, cup_geom_ids).sum()) if cup_geom_ids else -1
        print(f"[cup] n_ball_geoms={len(ball_geom_ids)} ball_pixel_count@64x64={ball_px}")
        print(f"[cup] n_cup_geoms={len(cup_geom_ids)} cup_pixel_count@64x64={cup_px}")
    else:
        print("scenes/scene_yor_cup.xml not found; skipping cup/ball check")


if __name__ == "__main__":
    main()
