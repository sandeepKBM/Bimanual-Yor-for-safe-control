# ZED camera -- workspace-facing sim representation

The ZED is a **real sensor** on YOR (`YOR/robot/zed_pub_node.py`: publishes
`zed/image`, `zed/depth`, `zed/pcd`, `zed/pose` via the ZED SDK, stereo RGB-D
+ visual-inertial pose). This adds it to the sim as an in-memory MjSpec
camera -- **no scene files were edited**; consumers call the helper at load
time.

## Documented vs. proposed (read this before trusting the pose)

**Documented in code (`zed_pub_node.py`):**
- It's a real ZED (SDK `pyzed.sl`), resolution `VGA` (672x376, `main()`) or
  `HD720` (1280x720, class default), `depth_mode=NEURAL`, positional
  tracking with `set_floor_as_origin=True`.
- Intrinsics (`fx, fy, cx, cy`) are **queried live from the SDK calibration**
  at `open_and_start()` -- not hardcoded, i.e. genuinely real hardware
  numbers on the physical unit, not something we can replicate exactly in
  sim without the calibration file.
- A hardcoded rigid extrinsic *does* exist: `Twb = Twc @ Tcb @ Tcb_t @ Tcb_2`
  with `Tcb` = 23.8 deg rotation about local x, `Tcb_t` = translate
  `(0.0603, 0, 0.2143)` m, `Tcb_2` an axis remap. This converts the ZED's own
  SLAM pose into a "base" pose for localization.
- **Inverting that transform** (camera pose in the "base" frame) gives
  translation `(x=+0.2143, y=+0.0603, z=0)` m and the same 23.8 deg tilt --
  i.e. a **low, forward, floor-level mount** (~0.10-0.30 m off the ground
  once composed with the `base` body's own height). This reads as a
  **navigation/floor-plane SLAM mount**, not a workspace-facing one: it
  cannot see a table top at z=0.55 m or arms mounted at z~0.67 m.

**Conclusion:** the ZED's *existence* and *intrinsics style* are real and
documented; its **workspace-viewing mount pose is NOT documented anywhere**
in this repo. Reusing the nav-mount numbers for a "look at the table" camera
would misrepresent real calibration data.

**PROPOSED, CONFIRM AGAINST HARDWARE:** we mount the sim ZED on `Lift_Top`
(the torso/lift body both Piper arm bases attach to), offset up/back from
the arms and pitched down ~44 deg, as a plausible mast/torso placement for a
workspace-overview stereo camera on this robot. **Flag for hardware
confirmation:** is there a second ZED, or is the documented one repositioned,
for manipulation work? If the real unit truly only exists at the low nav
mount, it cannot serve as the WM's tabletop-workspace camera and the WM
should fall back to the wrist egocentric cams (per `docs/design/state_spaces.md`,
already the primary real WM image source).

## Spec

- Helper: `scripts/zed_camera.py::add_zed_camera(spec, body_name="Lift_Top",
  name="zed_cam", fovy=70.0, pos=..., xyaxes=...)` -- takes an uncompiled
  `mujoco.MjSpec`, adds a `<camera>` as a child of `Lift_Top`, returns the
  camera name.
- Call: `cam_name = add_zed_camera(mujoco.MjSpec.from_file("scenes/scene_yor_canonical.xml"))`
  then `model = spec.compile()`.
- Camera name: `zed_cam`.
- Mount body: `Lift_Top` (torso/lift, world xpos ~(0, -0.21, 0.61) at the
  `home` keyframe; base + lift are welded/rigid in this scene so this local
  pose is constant).
- Local pose (in `Lift_Top` frame): `pos=(0.0496, 0.0004, 0.3859)`,
  `xyaxes=(0, 1, 0, -0.6921, 0, 0.7218)` -- equivalent to a world-frame
  camera at `(-0.05, -0.21, 1.00)` looking at `(0.42, -0.21, 0.55)` (table
  center), ~44 deg pitch down.
- `fovy=70` deg (ZED2/mini nominal vertical FOV; horizontal ~110 deg deg at
  16:9). Datasheet default -- not read from code (code queries fx/fy live
  from the SDK, doesn't hardcode FOV in degrees).
- Resolution: sim renders rendered at 640x360 (16:9, matches ZED aspect) in
  verification; `ZED_RESOLUTION=(1280,720)` constant documents the real
  HD720 mode.

## Verification (`scripts/verify_zed_camera.py`, `MUJOCO_GL=egl`)

- Loaded `scenes/scene_yor_canonical.xml` (read-only) + `add_zed_camera()`:
  `ncam=6` (5 upstream/canonical + `zed_cam`), rgb std=65.9 (real content,
  not blank), depth range 0.11-5.31 m.
- **Framing (canonical, `artifacts/zed_view.png`):** both arm grippers +
  full tabletop clearly in frame, workspace centered, robot base visible in
  foreground (realistic torso-down view). Good WM main-observation
  candidate.
- **Cup scene (`scenes/scene_yor_cup.xml`, `artifacts/zed_cup_view.png`):**
  cup clearly visible and centered on table (30 px @ 64x64 segmentation).
  Balls at rest are **inside** the opaque cup -> 0 px @64x64 at the `home`
  keyframe, which is physically correct (a closed cup occludes its
  contents; that's not a framing failure).
- **Spill visibility check (`artifacts/zed_spill_view.png`):** manually
  displaced 4 balls onto the table around the cup (simulating a spill) ->
  **3 px @ 64x64**, **102 px @ 640x360** segmentation count, clearly visible
  in the RGB render. Confirms the ZED view **can** see a spill once balls
  leave the cup, satisfying the WM's spill-detection requirement.

## Files touched (all new; nothing under `YOR/` or shared scenes edited)

- `scripts/zed_camera.py` (new) -- `add_zed_camera()` helper.
- `scripts/verify_zed_camera.py` (new) -- verification/render script.
- `scenes/ZED_CAMERA.md` (this file, new).
- `artifacts/zed_view.png`, `artifacts/zed_depth.png`,
  `artifacts/zed_cup_view.png`, `artifacts/zed_spill_view.png` (new).

## Rollback

Delete `scripts/zed_camera.py`, `scripts/verify_zed_camera.py`,
`scenes/ZED_CAMERA.md`, and the `artifacts/zed_*.png` files listed above.
Nothing under `YOR/`, `scenes/scene_yor_canonical.xml`, or
`scenes/yor_robot_derived.mjcf` was touched.
