"""
Smoke test: load the YOR bimanual (two AgileX Piper arms + table/floor) MuJoCo scene,
print model introspection info, run mj_forward, and render an offscreen PNG.

Usage:
    MUJOCO_GL=egl conda run -n safe_env python scripts/smoke_load_scene.py

This script is READ-ONLY with respect to the YOR repo -- it only loads the XML.
Outputs (if any) are written under artifacts/.
"""
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

REPO_ROOT = "/common/users/ss5772/bimanual"
SCENE_DIR = os.path.join(REPO_ROOT, "YOR", "robot", "arm", "mujoco")
SCENE_XML = os.path.join(SCENE_DIR, "scene_bimanual_table.xml")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
OUT_PNG = os.path.join(ARTIFACTS_DIR, "scene_bimanual_table_initial.png")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def main():
    print(f"[info] MUJOCO_GL = {os.environ.get('MUJOCO_GL')}")
    print(f"[info] mujoco version = {mujoco.__version__}")
    print(f"[info] loading scene: {SCENE_XML}")

    # cd into the mujoco dir so relative include/meshdir paths resolve regardless of
    # how MuJoCo's compiler treats relative vs absolute xml paths.
    prev_cwd = os.getcwd()
    os.chdir(SCENE_DIR)
    try:
        model = mujoco.MjModel.from_xml_path("scene_bimanual_table.xml")
    finally:
        os.chdir(prev_cwd)

    data = mujoco.MjData(model)

    print("\n=== MODEL SIZES ===")
    print(f"nbody  = {model.nbody}")
    print(f"njnt   = {model.njnt}")
    print(f"nq     = {model.nq}")
    print(f"nv     = {model.nv}")
    print(f"nu     = {model.nu}")
    print(f"ngeom  = {model.ngeom}")
    print(f"nsite  = {model.nsite}")
    print(f"nmesh  = {model.nmesh}")

    print("\n=== JOINT NAMES ===")
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    for i, name in enumerate(joint_names):
        jtype = model.jnt_type[i]
        jrange = model.jnt_range[i]
        print(f"  [{i}] {name!r}  type={jtype}  range={jrange}")

    print("\n=== ACTUATOR NAMES ===")
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    for i, name in enumerate(actuator_names):
        print(f"  [{i}] {name!r}")

    print("\n=== BODY NAMES (all) ===")
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    for i, name in enumerate(body_names):
        print(f"  [{i}] {name!r}")

    print("\n=== GEOM NAMES (all) ===")
    geom_names = []
    for i in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        geom_names.append(gname)
        gtype = model.geom_type[i]
        gpos = model.geom_pos[i]
        gsize = model.geom_size[i]
        bodyid = model.geom_bodyid[i]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodyid)
        print(f"  [{i}] name={gname!r} body={bname!r} type={gtype} pos={gpos} size={gsize}")

    print("\n=== GEOMS matching table/floor/box/object (case-insensitive) ===")
    pattern = re.compile(r"table|floor|box|object", re.IGNORECASE)
    found_any = False
    for i in range(model.ngeom):
        gname = geom_names[i] or ""
        bodyid = model.geom_bodyid[i]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodyid) or ""
        if pattern.search(gname) or pattern.search(bname):
            found_any = True
            gtype = model.geom_type[i]
            gpos_local = model.geom_pos[i]
            gsize = model.geom_size[i]
            type_names = {
                mujoco.mjtGeom.mjGEOM_PLANE: "plane",
                mujoco.mjtGeom.mjGEOM_BOX: "box",
                mujoco.mjtGeom.mjGEOM_MESH: "mesh",
                mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
                mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
                mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
            }
            tname = type_names.get(gtype, str(gtype))
            print(f"  geom[{i}] name={gname!r} body={bname!r} type={tname} local_pos={gpos_local} size={gsize}")
    if not found_any:
        print("  (none found -- this scene may not contain an actual table/box mesh, "
              "only a ground-plane geom named 'floor' and mocap-target boxes for pinch sites)")

    print("\n=== mj_forward ===")
    mujoco.mj_forward(model, data)
    print("mj_forward OK")

    print("\n=== ARM BASE WORLD POSITIONS (post mj_forward, via data.xpos) ===")
    for bname in ("left/base_link", "right/base_link"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid == -1:
            print(f"  {bname!r}: NOT FOUND")
        else:
            xpos = data.xpos[bid]
            print(f"  {bname!r}: world pos = {xpos}")

    print("\n=== FLOOR GEOM WORLD Z (top surface, since it's the only ground geom) ===")
    for i in range(model.ngeom):
        gname = geom_names[i] or ""
        if "floor" in gname.lower():
            gtype = model.geom_type[i]
            bodyid = model.geom_bodyid[i]
            # world position of this geom (post-forward)
            gxpos = data.geom_xpos[i]
            gsize = model.geom_size[i]
            print(f"  geom[{i}] name={gname!r} type={gtype} world_pos={gxpos} size={gsize}")
            if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                print(f"    -> plane top surface z = world_pos[2] = {gxpos[2]}")

    print("\n=== SITE WORLD POSITIONS: left/pinch_site, right/pinch_site, left/ee, right/ee ===")
    for site_name in ("left/pinch_site", "right/pinch_site", "left/ee", "right/ee"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid == -1:
            print(f"  site {site_name!r}: NOT FOUND")
        else:
            spos = data.site_xpos[sid]
            print(f"  site {site_name!r}: world pos = {spos}")

    print("\n=== KEYFRAMES ===")
    for k in range(model.nkey):
        kname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k)
        print(f"  key[{k}] name={kname!r} qpos={model.key_qpos[k]}")

    # ------------------------------------------------------------------
    # Offscreen render
    # ------------------------------------------------------------------
    print("\n=== OFFSCREEN RENDER ===")
    width, height = 800, 600
    # The XML doesn't set <visual><global offwidth=".." offheight=".."/>, so the
    # compiled default offscreen framebuffer is only 640x480. Bump it in the
    # already-loaded model object (does not touch the YOR XML files on disk).
    if width > model.vis.global_.offwidth or height > model.vis.global_.offheight:
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)

    # Use the 'home' keyframe if present so the arms are in a non-singular pose,
    # else just use qpos0 (already set by mj_forward above at qpos0/zero).
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        print(f"  using keyframe 'home' (id={key_id}) for render pose")
    else:
        print("  no 'home' keyframe found; using default qpos0 pose")

    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    # No named camera is defined in the XML (checked: no <camera> elements), so
    # configure a free camera to frame both arms + floor.
    cam.lookat = np.array([0.0, 0.0, 0.15])
    cam.distance = 1.6
    cam.azimuth = 130
    cam.elevation = -25

    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()

    from PIL import Image
    img = Image.fromarray(pixels)
    img.save(OUT_PNG)
    print(f"  saved render to: {OUT_PNG}")
    print(f"  file size: {os.path.getsize(OUT_PNG)} bytes, image size: {img.size}")

    print("\n[SUCCESS] smoke test complete.")


if __name__ == "__main__":
    main()
