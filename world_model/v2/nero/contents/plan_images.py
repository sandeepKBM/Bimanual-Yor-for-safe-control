"""nero/contents/plan_images.py -- the images the proposer model may be sent with its plan request (opt-in, default OFF).  SIMULATION ONLY.   (11-12)

    render_views(task, i, mode, out_dir)  -> PlanImages(paths, note, cameras, ...)    mode "none" | "zed" | "zed+wrist"
    PlanImages.proposer_cfg()             -> dict(images=[paths], image_note=str)      merged into plan["proposer"] (Rule 2 / rejection re-asks attach the SAME initial images)

Rendered from the world's INITIAL state (arms at their reset pose, base at rest, objects settled 0.5 s), BEFORE any probe: no probe reading, no hidden truth is in the picture
beyond what a camera sees (the box lid is closed; the mugs' fill is not visible).  Only cameras the real robot has:
  zed    the torso ZED (left imager).  Mount = scripts/zed_camera.py: Lift_Top, position PROPOSED (not hardware-documented; confirm on hardware), fovy 70 (ZED nominal vertical FOV,
         datasheet default).  The PITCH is ASSUMED: ZED_PITCH_DEG (the doc'd 44 deg frames the old canonical table; with the table-behind scene's table top at 0.8 m it cuts the
         object off).  Intrinsics, depth and the right imager are NOT modelled (left RGB only; no noise, no lens model).
  wrist  left_arm_wrist_rgb: the RealSense D435i RGB of the left NYU gripper (nero/nyu_gripper.d435i_body_xml; mount/tilt ASSUMED, see nyu_gripper CAM_*).
Image: 640x360 PNG (ZED HD720 is 1280x720; this is a 2x downscale in size, not a re-render of the real image pipeline).  Files: <out_dir>/<sha256[:16] of the PNG bytes>.png, so a
changed render is a new file and the CachedVLM key (which hashes the file bytes) changes with it.  Rendering is CPU (MUJOCO_GL=osmesa) or EGL; it is deterministic for a world.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODES = ("none", "zed", "zed+wrist", "zed+look")
LOOK_BACK_M = 0.0                      # ASSUMED (zed+look, USER 11-17): the look-turn stand-in turns about the base rotation centre with NO retreat.  BIMANUAL_LEGACY_LOOK_BACK=1: 0.50 m (copied from route RB's first segment) and the old caption
LOOK_BACK_LEGACY_M = 0.50
SIZE = (640, 360)                      # (width, height) ASSUMED: modest, keeps the request small
ZED_FOVY_DEG = 70.0                    # scripts/zed_camera.ZED_FOVY_DEG (ZED nominal vertical FOV; datasheet default, not read from the SDK)
ZED_LOCAL_POS = (0.049604, 0.000450, 0.385865)   # scripts/zed_camera._ZED_POS_LOCAL, in Lift_Top's frame: PROPOSED torso mount, confirm-on-hardware
ZED_PITCH_DEG = 28.0                   # ASSUMED pitch down (scripts/zed_camera uses 44 for the old canonical table); framed by eye on box_behind / tray6_behind
WRIST_CAMERA = "left_arm_wrist_rgb"    # RealSense D435i RGB on the left NYU gripper (mount ASSUMED)
SETTLE_S = 0.5                         # as CarryEpisode: objects at rest

_NOTES = {
    "zed": "IMAGES: image 1 is the torso stereo camera (ZED, left imager) at the start of the task, before any probe.",
    "zed+look": "IMAGES: image 1 is the torso stereo camera (ZED, left imager) at the start of the task, before any probe; image 2 is the same camera after the base turns in place to look behind the robot.",
    "zed+look_legacy": "IMAGES: image 1 is the torso stereo camera (ZED, left imager) at the start of the task, before any probe; image 2 is the same camera after the base backs up 0.5 m and turns 180 degrees in place (a look-around), facing what is behind the robot.",
    "zed+wrist": "IMAGES: image 1 is the torso stereo camera (ZED, left imager) at the start of the task, before any probe; image 2 is the left wrist camera (RealSense RGB, facing the gripper fingers) at the same moment.",
}


@dataclass
class PlanImages:
    mode: str
    paths: list = field(default_factory=list)
    note: str = ""
    cameras: list = field(default_factory=list)
    size: tuple = SIZE
    task: str = ""
    i: int = 0
    assumptions: dict = field(default_factory=dict)

    def proposer_cfg(self):
        return dict(images=list(self.paths), image_note=self.note) if self.paths else {}

    def record(self):
        return dict(mode=self.mode, paths=list(self.paths), cameras=list(self.cameras), size=list(self.size), world=dict(task=self.task, i=self.i), assumptions=self.assumptions)


def _legacy_look_back():
    return os.environ.get("BIMANUAL_LEGACY_LOOK_BACK") == "1"


def look_back_m():
    """how far the look-turn stand-in backs up before turning: 0.0 (USER 11-17), 0.5 with BIMANUAL_LEGACY_LOOK_BACK=1."""
    return LOOK_BACK_LEGACY_M if _legacy_look_back() else LOOK_BACK_M


def image_note(mode):
    """the one prompt line for a mode ('' for none)."""
    if mode not in MODES:
        raise ValueError(f"plan images mode {mode!r} not in {MODES}")
    if mode == "zed+look" and _legacy_look_back():
        return _NOTES["zed+look_legacy"]
    return _NOTES.get(mode, "")


def _zed_free_camera(m, d):
    """MjvCamera equal to the ZED pose: position LIFT_TOP-local ZED_LOCAL_POS, looking along the Lift_Top frame's (-cos p, 0, -sin p) (scripts/zed_camera: the Lift_Top frame is
    yawed 180 deg to the world in these scenes, so the camera looks toward the table), pitched down ZED_PITCH_DEG."""
    import mujoco
    bid = m.body("Lift_Top").id
    R = d.xmat[bid].reshape(3, 3)
    pos = d.xpos[bid] + R @ np.array(ZED_LOCAL_POS)
    p = np.deg2rad(ZED_PITCH_DEG)
    fwd = R @ np.array([-np.cos(p), 0.0, -np.sin(p)])
    fwd /= np.linalg.norm(fwd)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.0
    cam.lookat[:] = pos + cam.distance * fwd           # MuJoCo free camera: position = lookat - distance * forward
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    return cam, pos, fwd


def _png_bytes(img):
    import io
    import imageio.v2 as iio
    b = io.BytesIO()
    iio.imwrite(b, img, format="png")
    return b.getvalue()


def render_world(world, mode, size=SIZE):
    """-> [(camera label, HxWx3 uint8)] for a world object (a task world of nero/contents/scenes), from its initial settled state.  CPU or EGL per MUJOCO_GL."""
    import mujoco
    from world_model.v2.nero.base import scene as BS
    from world_model.v2.nero.contents import scenes as SC
    if mode == "none":
        return []
    spec = BS.build_spec(world)
    m = spec.compile()
    d = mujoco.MjData(m)
    SC.settle(m, d, SETTLE_S)
    mujoco.mj_forward(m, d)
    w, h = size
    m.vis.global_.offwidth, m.vis.global_.offheight = max(w, m.vis.global_.offwidth), max(h, m.vis.global_.offheight)
    out = []
    if mode == "zed+look":                                     # 11-16: brighter headlight for this mode only (the older modes' images, hence their cache keys, are unchanged)
        m.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
        m.vis.headlight.diffuse[:] = (0.75, 0.75, 0.75)
    r = mujoco.Renderer(m, h, w)
    try:
        m.vis.global_.fovy = ZED_FOVY_DEG                      # free camera uses the global fovy; the wrist camera has its own (fixed camera, own fovy)
        cam, _, _ = _zed_free_camera(m, d)
        r.update_scene(d, camera=cam)
        out.append(("zed_left", r.render().copy()))
        if mode == "zed+look":                                 # 11-16 stand-in for an UNBUILT look-turn skill: the ZED pose moved as the base would move (back look_back_m() = 0 by default, turn 180
            from world_model.v2.adapters import base_sweep as BSW  # about the rotation centre); the robot's own arms are NOT re-posed in this view (they are behind / beside the camera).
            c0x, c0y, yaw0 = BSW.start_pose(world)
            _, pos, fwd = _zed_free_camera(m, d)
            lb = look_back_m()
            c1 = np.array([c0x, c0y]) + lb * -np.array([np.cos(yaw0), np.sin(yaw0)])
            rel = pos[:2] - np.array([c0x, c0y]) - lb * -np.array([np.cos(yaw0), np.sin(yaw0)])
            p2 = np.array([*(c1 - rel), pos[2]])                 # 180 deg about c1: rel -> -rel
            f2 = np.array([-fwd[0], -fwd[1], fwd[2]])
            cam2 = mujoco.MjvCamera()
            cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam2.distance = 1.0
            cam2.lookat[:] = p2 + cam2.distance * f2
            cam2.azimuth = float(np.degrees(np.arctan2(f2[1], f2[0])))
            cam2.elevation = float(np.degrees(np.arcsin(f2[2])))
            r.update_scene(d, camera=cam2)
            out.append(("zed_left_after_look_turn", r.render().copy()))
        if mode == "zed+wrist":
            r.update_scene(d, camera=WRIST_CAMERA)
            out.append((WRIST_CAMERA, r.render().copy()))
    finally:
        r.close()
    return out


def render_views(task, i, mode, out_dir, size=SIZE):
    """the proposer's views of world i of a table-behind `task` (nero/base/smoke.behind_world: the same world list as the runs).  -> PlanImages.  mode 'none' -> no files."""
    if mode not in MODES:
        raise ValueError(f"plan images mode {mode!r} not in {MODES}")
    if mode == "none":
        return PlanImages(mode="none", task=task, i=i)
    from world_model.v2.nero.base import smoke as SM
    _, world = SM.behind_world(task, i)
    views = render_world(world, mode, size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, cams = [], []
    for label, img in views:
        b = _png_bytes(img)
        p = out_dir / f"{hashlib.sha256(b).hexdigest()[:16]}.png"
        if not p.exists():
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(b)
            os.replace(tmp, p)
        paths.append(str(p))
        cams.append(label)
    ass = dict(zed_mount="PROPOSED torso mount (Lift_Top, scripts/zed_camera.py), confirm on hardware", zed_pitch_deg=dict(value=ZED_PITCH_DEG, tag="ASSUMED"),
               zed_fovy_deg=dict(value=ZED_FOVY_DEG, tag="ASSUMED (ZED nominal vertical FOV)"), left_image_only=True, wrist_mount="ASSUMED (nyu_gripper CAM_*)",
               noise="none: clean sim render, no lens model, no depth")
    return PlanImages(mode=mode, paths=paths, note=image_note(mode), cameras=cams, size=tuple(size), task=task, i=i, assumptions=ass)
