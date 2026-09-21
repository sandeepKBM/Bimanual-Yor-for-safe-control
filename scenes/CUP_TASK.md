# Don't-spill-the-cup scene

Files: `scenes/scene_bimanual_cup.xml` (new), `scripts/cup_task_utils.py` (new),
`scripts/test_cup_scene.py` (new). Derived from `scenes/scene_bimanual_pickup.xml`
/ `scenes/bimanual_piper_pickup.xml` (both untouched, read-only for this task).

## Cup

Body `cup` (freejoint `cup_free`), pos `0.32 0.13 0.20` (on the table top,
z=0.20, near the LEFT arm's reach envelope -- same region as M0's `box_left`).
Built from 13 geoms, all in local frame relative to the body origin (bottom
center):
- `cup_bottom`: cylinder, radius 0.030 m, half-thickness 0.0015 m, at local z=0.0015.
- `cup_wall_0..11`: 12 boxes forming a dodecagon shell (approximating a hollow
  cylinder). Centerline radius R=0.030 m, wall thickness t=0.003 m, height
  H=0.08 m, tangential half-width 0.00785 m (== half the per-segment arc
  length, so adjacent segments meet edge-to-edge with ~0 gap -- no leak paths).
  Positions/quats generated analytically (12-fold rotation about local Z).
- Rim (world) height at rest = 0.20 + 0.08 = 0.28 m. Usable interior radius
  ~0.0285 m (R minus half wall thickness).
- Material `cup_mat` rgba `0.85 0.15 0.15 0.95`. Friction `0.6 0.01 0.0001`,
  condim=4 on all cup geoms. Explicit `<inertial mass="0.03" .../>` (light,
  ~30 g, easily graspable/liftable).

## Balls

N=8 free bodies `ball_0`..`ball_7`, sphere radius 0.008 m (slightly under the
spec's ~1 cm to fit comfortably inside a 6 cm cup with room to move -- noted
deviation). `density="900"` (~2.4 g/ball, "light"). Friction `0.5 0.005
0.0003` (sliding, torsional, rolling) + `condim="6"` so rolling resistance
lets them settle to rest instead of rolling indefinitely. Initial pose: two
4-ball layers stacked inside the cup (z ~0.2085 and ~0.226), all within the
usable interior radius.

## Cameras

- `head_cam`: world-fixed, declared directly in the XML (`worldbody`), pos
  `0.95 0.05 0.60`, `mode="targetbody" target="cup"` (keeps the cup framed
  even while tilted), fovy=45.
- `right/wrist_cam`: **not** in the XML -- `bimanual_piper_pickup.xml` is an
  existing file, so per the "new filenames only" constraint the camera is
  attached at load time via `mujoco.MjSpec` in
  `scripts/cup_task_utils.py:build_cup_spec()/load_cup_model()`, as a child of
  body `right/gripper_base`. Mount: local pos `0 0 0.12` (near the fingertip
  level -- fingers attach to this body at local z=0.1358), euler
  `0.4 3.14159 0` (180 deg about Y so the camera's view axis aligns with the
  body's local +Z, i.e. "forward" past the fingers, plus a ~23 deg downward
  tilt), fovy=70. An empirical 12-candidate orientation sweep
  (scratchpad `probe_wrist_cam.py`) confirmed this convention: reusing the
  `right/ee` site's own euler (`0 -1.5708 0`) instead makes the camera stare
  back into the gripper's own mesh (std~29, all gripper-body pixels); the
  chosen mount gives a clear outward/downward view (checker floor/wall
  visible, no self-occlusion).
- Any script loading this scene for cameras/observations must go through
  `cup_task_utils.load_cup_model()` (or `build_cup_spec()` +
  `spec.compile()`), not `mujoco.MjModel.from_xml_path()` directly, or the
  wrist camera will be missing.

## Spill detector (`cup_task_utils.count_spilled`)

For each `ball_*` body, transform its world position into the `cup` body's
own local frame using the cup's *current* world position + orientation
(`data.xpos`/`data.xmat`), so this stays correct even while the cup is
tilted:
```
local = cup_rot.T @ (ball_world_pos - cup_world_pos)
radial = hypot(local.x, local.y)
height = local.z
spilled if radial > CUP_RADIUS(0.030) or height < -0.002 or height > CUP_HEIGHT(0.080)
```
i.e. spilled == outside the upright cylinder (radius, 0..rim-height) that
approximates the cup's interior in its own frame. Resting balls have
radial<0.03 and height in ~[0.005, 0.03], well inside the cylinder, so they
are never falsely flagged.

## Verification results (`scripts/test_cup_scene.py`)

| Check | Result | Evidence |
|---|---|---|
| scene loads | PASS | nq=79, nu=16, nbody=31 |
| cameras present | PASS | `['head_cam', 'right/wrist_cam']` |
| cameras render RGB | PASS | head_cam 640x480x3 std=43.96; wrist_cam 640x480x3 std=29.16 |
| balls rest before shake | PASS | `count_spilled`=0 after a 1.0 s settle |
| tilt spills balls | PASS | `count_spilled` 0 -> 8 after scripted tilt |

Scripted tilt (open-loop, cup is a free body -- no gripper grasp or IK exists
in this repo, so the cup's freejoint qpos is directly overwritten each step,
same "teleport" technique `scripts/test_pickup_scene.py` uses for the box):
lift 0.10 m clear of the table (0.3 s), rotate 100 deg about the world X axis
(1.0 s ramp), hold (0.4 s), settle (0.6 s). All 8 balls end up spilled onto
the table (see `artifacts/cup_spill_montage.png`: before | mid-tilt | after).

## Accel-induced spill while the cup stays upright

**Achieved and tuned.** With the cup's orientation kinematically forced to
stay upright (identity quat, `run_upright_shake_test` in
`test_cup_scene.py`) and only its lateral (x) position driven by a fast
sinusoid, sweeping amplitude/frequency gives a graded (not all-or-nothing)
threshold -- evidence it is a real friction/inertia effect, not a solver
artifact:

| amplitude | freq | balls spilled |
|---|---|---|
| 0.01 m | 4 Hz | 0 |
| 0.02 m | 5 Hz | 0 |
| 0.03 m | 6 Hz | 5 / 8 |
| 0.04 m | 6 Hz | 8 / 8 |
| 0.06 m | 8 Hz | 8 / 8 |

Peak lateral acceleration at the partial-spill point (0.03 m, 6 Hz) is
`A*(2*pi*f)^2 ~= 42 m/s^2 ~= 4.3 g`. The default test script uses
amp=0.06 m / 8 Hz (full spill, 8/8) for a clear PASS signal.

## Blockers / notes

- None blocking. Minor: at the arms' "home" keyframe the right arm's default
  resting pose does not happen to point `right/wrist_cam` at the table/cup
  (right arm points away from the workspace at rest) -- the render check only
  requires non-degenerate RGB (satisfied), not that the cup be in frame;
  framing the cup would need the arm scripted into a reach pose, out of scope
  here.
- Ball radius used (0.008 m) is slightly smaller than the spec's ~1 cm
  suggestion, to fit 8 balls comfortably inside a 6 cm-diameter cup.

## Rollback

Delete the 3 new files (`scenes/scene_bimanual_cup.xml`,
`scripts/cup_task_utils.py`, `scripts/test_cup_scene.py`) and the 3 new
artifacts (`artifacts/cup_headcam.png`, `artifacts/cup_wristcam.png`,
`artifacts/cup_spill_montage.png`). Nothing else in the repo was modified.
