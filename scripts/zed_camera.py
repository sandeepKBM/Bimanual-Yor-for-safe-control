#!/usr/bin/env python3
"""Add the real YOR ZED stereo camera to a MuJoCo spec/scene as a workspace view.

The ZED is a REAL sensor on YOR (see `YOR/robot/zed_pub_node.py`, which
publishes zed/image, zed/depth, zed/pcd, zed/pose from a ZED stereo camera
via the ZED SDK). This module does NOT invent a new camera -- it represents
that real sensor in sim, attached to the robot body that is closest to its
plausible physical mount (see MOUNT NOTE below), with real-ish ZED intrinsics
(fovy) so it is usable as a workspace-facing world-model observation.

=== MOUNT NOTE (read before trusting the numbers) ===

`zed_pub_node.py` DOES hard-code a real rigid extrinsic transform (used to
convert the ZED's own SLAM pose into a robot "base" pose for localization):

    angle = deg2rad(23.8)          # tilt about local x
    Tcb   = rot_x(angle)
    Tcb_t = translate(0.0603, 0.0, 0.2143)
    Tcb_2 = [[0,-1,0],[0,0,1],[0,0,0],[-1,0,0,0]]-ish axis remap
    Twb = Twc @ Tcb @ Tcb_t @ Tcb_2   # base pose in world = camera pose in world @ (base expressed in camera frame)

Inverting this (see scripts/zed_camera.py git history / derivation below) to
get the CAMERA's pose in the robot's "base" frame gives translation
`(x=+0.2143, y=+0.0603, z=0.0)` m (forward, slightly left, SAME HEIGHT as
base origin, i.e. ~0.10-0.30 m off the floor) with a 23.8 deg tilt.

That is a **low, forward-facing, floor-level mount** -- consistent with a
navigation/floor-plane/SLAM camera on the mobile base, NOT a camera aimed at
the tabletop workspace (table top z=0.55, arms mounted at z~0.67). Reusing
those exact numbers for a "workspace ZED" would misrepresent real calibration
data as something it isn't.

=> CONCLUSION: the *existence* of a real ZED and its *intrinsics query
   pattern* (live fx/fy/cx/cy from the ZED SDK, HD720/VGA resolution, NEURAL
   depth) are code-documented. The *workspace-facing mount pose* used below
   is **PROPOSED, NOT hardware-documented** -- confirm against the physical
   robot (is there a second ZED / iPhone-style mast mount, or is the one
   documented ZED repositioned for manipulation experiments?). Flagged here
   and in scenes/ZED_CAMERA.md.

The proposed mount: on `Lift_Top` (the torso/lift body both Piper arm bases
attach to, world z~0.61-0.67 in scene_yor_canonical.xml's home keyframe),
offset up and back from the arms and pitched down ~44 deg to frame the
tabletop workspace + both arms -- the same body class of mount as a
mast-mounted stereo camera on a mobile manipulator.

Usage:
    import mujoco
    from scripts.zed_camera import add_zed_camera

    spec = mujoco.MjSpec.from_file("scenes/scene_yor_canonical.xml")
    cam_name = add_zed_camera(spec)          # default: mounts on Lift_Top
    model = spec.compile()
"""

from __future__ import annotations

import numpy as np

ZED_CAMERA_NAME = "zed_cam"
ZED_MOUNT_BODY = "Lift_Top"

# ZED 2 / ZED mini nominal stereo FOV: ~110 deg H x ~70 deg V (native, no
# crop). We use vertical FOV directly since MuJoCo <camera fovy=.../> is the
# *vertical* field of view. This value is a datasheet default, NOT read from
# `zed_pub_node.py` (which queries fx/fy/cx/cy live from the SDK at runtime
# and does not hardcode FOV in degrees).
ZED_FOVY_DEG = 70.0

# Native ZED resolution options actually used in zed_pub_node.py: HD720
# (1280x720, class default) and VGA (672x376, used in main()). We default the
# sim render size to a 16:9 crop of that for a WM-friendly image while
# keeping the real aspect ratio.
ZED_RESOLUTION = (1280, 720)

# --- Proposed mount pose, expressed in Lift_Top's LOCAL body frame ---------
# Derived by: (1) picking a world-frame camera position/target that frames
# the shared workspace (table center ~(0.42,-0.21,0.55), both arm mounts at
# world z~0.67, y=-0.117/-0.302), then (2) transforming into Lift_Top's local
# frame using Lift_Top's world pose at the `home` keyframe of
# scenes/scene_yor_canonical.xml (xpos=(-0.0001,-0.2096,0.6142), a ~180 deg
# yaw about world z relative to world axes -- base/lift are welded/rigid in
# this scene, so this local pose is constant regardless of when it's added).
# World-frame design point: cam at (-0.05, -0.21, 1.00), looking at
# (0.42, -0.21, 0.55) -> ~44 deg pitch-down, camera y-axis (right) along
# world -y, i.e. camera faces +x (forward) and down.
_ZED_POS_LOCAL = (0.049604, 0.000450, 0.385865)
_ZED_XYAXES_LOCAL = (0.0, 1.0, 0.0, -0.692147, 0.0, 0.721757)


def add_zed_camera(
    spec,
    body_name: str = ZED_MOUNT_BODY,
    name: str = ZED_CAMERA_NAME,
    fovy: float = ZED_FOVY_DEG,
    pos=_ZED_POS_LOCAL,
    xyaxes=_ZED_XYAXES_LOCAL,
):
    """Attach the real ZED camera to an in-memory MjSpec, at load time.

    Args:
      spec: a `mujoco.MjSpec` (e.g. `mujoco.MjSpec.from_file(...)`), NOT yet
        compiled. Callers own the spec (this scene file itself is never
        edited on disk).
      body_name: robot body to attach the camera to. Default `"Lift_Top"`
        (the torso/lift body the arm bases mount on) -- see MOUNT NOTE above
        re: real-vs-proposed.
      name: MuJoCo camera name. Default `"zed_cam"`.
      fovy: vertical FOV in degrees (MuJoCo convention). Default 70 (ZED2/
        mini nominal vertical FOV).
      pos, xyaxes: camera pose in `body_name`'s local frame. Defaults frame
        the tabletop workspace + both arms (see derivation above); override
        if attaching to a different body or scene.

    Returns:
      The camera name (str) that was added -- pass this to
      `mj_name2id(model, mjtObj.mjOBJ_CAMERA, name)` / `Renderer(camera=name)`.
    """
    body = spec.body(body_name)
    if body is None:
        raise ValueError(
            f"add_zed_camera: body '{body_name}' not found in spec; "
            "pass the correct body_name for this scene's robot mjcf."
        )
    body.add_camera(
        name=name,
        pos=list(pos),
        xyaxes=list(xyaxes),
        fovy=float(fovy),
        mode=mujoco_mode_fixed(),
    )
    return name


def mujoco_mode_fixed():
    import mujoco

    return mujoco.mjtCamLight.mjCAMLIGHT_FIXED
