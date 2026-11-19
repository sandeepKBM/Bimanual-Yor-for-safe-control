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

MODES = ("none", "zed", "zed+wrist", "zed+look", "room4")
LOOK_BACK_M = 0.0                      # ASSUMED (zed+look, USER 11-17): the look-turn stand-in turns about the base rotation centre with NO retreat.
SIZE = (640, 360)                      # (width, height) ASSUMED: modest, keeps the request small
ZED_FOVY_DEG = 70.0                    # scripts/zed_camera.ZED_FOVY_DEG (ZED nominal vertical FOV; datasheet default, not read from the SDK)
ZED_LOCAL_POS = (0.049604, 0.000450, 0.385865)   # scripts/zed_camera._ZED_POS_LOCAL, in Lift_Top's frame: PROPOSED torso mount, confirm-on-hardware
ZED_PITCH_DEG = 28.0                   # ASSUMED pitch down (scripts/zed_camera uses 44 for the old canonical table); framed by eye on box_behind / tray6_behind
WRIST_CAMERA = "left_arm_wrist_rgb"    # RealSense D435i RGB on the left NYU gripper (mount ASSUMED)
SETTLE_S = 0.5                         # as CarryEpisode: objects at rest
ROOM_MARGIN_M = 1.0                    # ASSUMED (11-19, USER approved framing): the room = the work-area xy box grown by this on every side
ROOM_SIZE = (2208, 1242)               # ASSUMED (11-19, user: 2K): room4 photo size (width, height) = the default of room_size()
ROOM_CAM_H_M = 2.2                     # ASSUMED (11-19): lab camera height above the floor
ROOM_FOVY_DEG = 70.0                   # ASSUMED (11-19): ZED-2i-like vertical field of view, all four cameras

_NOTES = {
    "zed": "IMAGES: image 1 is the torso stereo camera (ZED, left imager) at the start of the task, before any probe.",
    "zed+look": "IMAGES: image 1 is the torso stereo camera (ZED, left imager) at the start of the task, before any probe; image 2 is the same camera after the base turns in place to look behind the robot.",
    "room4": "IMAGES: images 1-4 are fixed lab cameras, one on each wall of a rectangular room (front, left, back, right as seen from the robot), at the start of the task, before any probe.",
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


def room_size():
    """room4 photo size (width, height): ROOM_SIZE (2K)."""
    return ROOM_SIZE


def look_back_m():
    """how far the look-turn stand-in backs up before turning: 0.0 (USER 11-17)."""
    return LOOK_BACK_M


def image_note(mode):
    """the one prompt line for a mode ('' for none)."""
    if mode not in MODES:
        raise ValueError(f"plan images mode {mode!r} not in {MODES}")
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


def work_area_xy(m, d):
    """xy bounding box [[xlo, ylo], [xhi, yhi]] of the static scene geoms: every geom on the world body (geom_bodyid == 0) except planes (the floor), in the settled compiled model.
    Per-geom world extent = |R| @ half-size (exact for boxes; a bounding box for the other types, with half-size from geom_size: sphere r, capsule/cylinder r and half-length)."""
    import mujoco
    T = mujoco.mjtGeom
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != 0 or m.geom_type[g] == T.mjGEOM_PLANE:
            continue
        t, sz = m.geom_type[g], m.geom_size[g]
        if t == T.mjGEOM_SPHERE:
            hs = np.full(3, sz[0])
        elif t in (T.mjGEOM_CAPSULE, T.mjGEOM_CYLINDER):
            hs = np.array([sz[0], sz[0], sz[1] + (sz[0] if t == T.mjGEOM_CAPSULE else 0.0)])
        elif t == T.mjGEOM_BOX:
            hs = np.array(sz[:3])
        else:
            hs = np.full(3, m.geom_rbound[g])
        R = d.geom_xmat[g].reshape(3, 3)
        he = np.abs(R) @ hs
        c = d.geom_xpos[g]
        lo, hi = np.minimum(lo, c[:2] - he[:2]), np.maximum(hi, c[:2] + he[:2])
    return [[float(lo[0]), float(lo[1])], [float(hi[0]), float(hi[1])]]


def room4_cameras(m, d, world):
    """the four lab-fixed room cameras (USER 11-19) -> [dict(label, pos (3,), look_at (3,))], order front (+x wall), left (+y), back (-x, behind the robot), right (-y).
    Room = the work-area box (work_area_xy(m)) grown by ROOM_MARGIN_M; one camera at the middle of each wall at ROOM_CAM_H_M, all aimed at (work-area centre x, y, counter top z).
    Pure apart from reading m / d (d must be forwarded)."""
    from world_model.v2.nero.contents import scenes as SC
    (x0, y0), (x1, y1) = work_area_xy(m, d)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    look = np.array([cx, cy, float(SC.support_tops(world)["counter"]["top"])])
    z, g = ROOM_CAM_H_M, ROOM_MARGIN_M
    spots = (("room_front", (x1 + g, cy)), ("room_left", (cx, y1 + g)), ("room_back", (x0 - g, cy)), ("room_right", (cx, y0 - g)))
    return [dict(label=lb, pos=np.array([x, y, z]), look_at=look.copy()) for lb, (x, y) in spots]


def _look_free_camera(pos, look_at):
    """MjvCamera FREE equal to a camera at `pos` looking at `look_at` (same construction as _zed_free_camera)."""
    import mujoco
    fwd = np.asarray(look_at, float) - np.asarray(pos, float)
    dist = float(np.linalg.norm(fwd))
    fwd = fwd / dist
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = dist
    cam.lookat[:] = np.asarray(look_at, float)
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    return cam


def _png_bytes(img):
    import io
    import imageio.v2 as iio
    b = io.BytesIO()
    iio.imwrite(b, img, format="png")
    return b.getvalue()


def render_world(world, mode, size=SIZE, info=None):
    """-> [(camera label, HxWx3 uint8)] for a world object (a task world of nero/contents/scenes), from its initial settled state.  CPU or EGL per MUJOCO_GL.
    info (room4): a dict that receives cameras and work_area_xy."""
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
        if mode == "room4":
            m.vis.global_.fovy = ROOM_FOVY_DEG
            m.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)    # ASSUMED (11-19): same values as zed+look
            m.vis.headlight.diffuse[:] = (0.75, 0.75, 0.75)
            cams = room4_cameras(m, d, world)
            if info is not None:
                info.update(cameras=cams, work_area_xy=work_area_xy(m, d))
            for c in cams:
                r.update_scene(d, camera=_look_free_camera(c["pos"], c["look_at"]))
                out.append((c["label"], r.render().copy()))
            return out
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


def render_views(task, i, mode, out_dir, size=None):
    """the proposer's views of world i of a table-behind `task` (nero/base/smoke.behind_world: the same world list as the runs).  -> PlanImages.  mode 'none' -> no files."""
    if mode not in MODES:
        raise ValueError(f"plan images mode {mode!r} not in {MODES}")
    if mode == "none":
        return PlanImages(mode="none", task=task, i=i)
    from world_model.v2.stack import tasks as TK
    if mode == "room4" and str(task) not in TK.FIXED_OF:
        raise ValueError("room4 is for base tasks (11-19)")
    size = (room_size() if mode == "room4" else SIZE) if size is None else size
    from world_model.v2.nero.base import smoke as SM
    _, world = SM.behind_world(task, i)
    info = {}
    views = render_world(world, mode, size, info=info)
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
    if mode == "room4":
        ass = dict(room_margin_m=dict(value=ROOM_MARGIN_M, tag="ASSUMED (11-19, USER approved framing)"), work_area_xy=info["work_area_xy"],
                   cam_height_m=dict(value=ROOM_CAM_H_M, tag="ASSUMED (11-19)"), fovy_deg=dict(value=ROOM_FOVY_DEG, tag="ASSUMED (11-19)"),
                   size=dict(value=list(size), tag="ASSUMED (11-19)"), lighting="render only: brighter headlight",
                   cameras={c["label"]: dict(pos=[float(x) for x in c["pos"]], look_at=[float(x) for x in c["look_at"]]) for c in info["cameras"]},
                   noise="none: clean sim render, RGB only, no depth")
        return PlanImages(mode=mode, paths=paths, note=image_note(mode), cameras=cams, size=tuple(size), task=task, i=i, assumptions=ass)
    ass = dict(zed_mount="PROPOSED torso mount (Lift_Top, scripts/zed_camera.py), confirm on hardware", zed_pitch_deg=dict(value=ZED_PITCH_DEG, tag="ASSUMED"),
               zed_fovy_deg=dict(value=ZED_FOVY_DEG, tag="ASSUMED (ZED nominal vertical FOV)"), left_image_only=True, wrist_mount="ASSUMED (nyu_gripper CAM_*)",
               noise="none: clean sim render, no lens model, no depth")
    return PlanImages(mode=mode, paths=paths, note=image_note(mode), cameras=cams, size=tuple(size), task=task, i=i, assumptions=ass)
