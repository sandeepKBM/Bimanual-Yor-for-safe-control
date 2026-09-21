# YOR_CANONICAL interface lock

Canonical bimanual scene going forward. Supersedes the old simplified rig
(`scenes/bimanual_piper_pickup.xml` / `scene_bimanual_pickup.xml`, arms fixed
0.5m apart on the floor, no lift, no base). Re-pointing `experiments/` and
`collection/` at this scene is a **later** step, not done here.

## Files

- `YOR/robot/yor-description/robot-welded-base-and-lift.mjcf` -- original,
  **read-only, unmodified**. Swerve base (welded/fixed, no freejoint) + 2-DOF
  vertical lift (joints commented out -- rigid) + two Piper arms.
- `scenes/yor_robot_derived.mjcf` -- **derived copy**. Only change vs.
  upstream: adds ACTUATED parallel-jaw gripper fingers (upstream's gripper is
  a single static mesh, `gripper_body.stl`, fingers baked in, non-actuable).
  See the file's header comment for the full diff. Everything else
  (base/lift/wheels, contact excludes, sensors, existing cameras) is
  byte-for-byte identical to upstream.
- `scenes/scene_yor_canonical.xml` -- top-level scene: includes
  `yor_robot_derived.mjcf` + floor + table + one graspable test box +
  `head_cam` + a full-scene `home` keyframe. **Load this file.**
- `scripts/verify_yor_canonical.py` -- headless load/grasp/camera check
  (`MUJOCO_GL=egl python scripts/verify_yor_canonical.py`).

## Old -> new name mapping

The old rig (`scenes/bimanual_piper_pickup.xml`) used bare `left/`, `right/`
prefixes on a floor-mounted pair of arms with no base/lift. Map old -> new:

| Old (bimanual_piper_pickup.xml) | New (yor_robot_derived.mjcf) |
|---|---|
| `left/base_link` | `left_arm_link0` |
| `left/joint1` .. `left/joint6` | `left_arm_joint1` .. `left_arm_joint6` |
| `left/link1` .. `left/link6` | `left_arm_link1` .. `left_arm_link6` |
| `left/gripper_base` | `left_arm_gripper_base` |
| `left/joint7`, `left/joint8` (fingers) | `left_arm_joint7`, `left_arm_joint8` (**re-added**, same offsets/ranges) |
| `left/link7`, `left/link8` | `left_arm_link7`, `left_arm_link8` |
| `left/ee` (site) | `left_arm_ee` (existing, at link6 origin) **or** `left_arm_ee_grasp` (**new**, fingertip depth, recommended for grasp IK) |
| `right/*` | `right_arm_*` (all analogous) |
| *(none -- old rig had no wrist cams)* | `left_arm_egocentric`, `right_arm_egocentric` (real `<camera>`, already existed upstream) |
| *(none)* | `head_cam` (**new**, fixed world-observation camera) |
| Arms 0.5m apart, floor-mounted, no base | Arms 0.185m apart, lift-mounted at z~0.67m, welded swerve base |

## Joint names + ranges (verified via `mj_name2id`/`jnt_range`)

| Joint | Type | Range | Status |
|---|---|---|---|
| `left_arm_joint1` / `right_arm_joint1` | hinge | [-2.61, 2.61] rad | upstream |
| `left_arm_joint2` / `right_arm_joint2` | hinge | [0, 3.13] rad | upstream |
| `left_arm_joint3` / `right_arm_joint3` | hinge | [-2.965, 0] rad | upstream |
| `left_arm_joint4` / `right_arm_joint4` | hinge | [-1.74, 1.74] rad | upstream |
| `left_arm_joint5` / `right_arm_joint5` | hinge | [-1.2, 1.2] rad | upstream |
| `left_arm_joint6` / `right_arm_joint6` | hinge | [-3, 3] rad | upstream |
| `left_arm_joint7` / `right_arm_joint7` | slide | [0, 0.035] m | **added** (finger) |
| `left_arm_joint8` / `right_arm_joint8` | slide | [-0.035, 0] m | **added** (finger) |

Total `nq`=16 for the robot alone (12 arm + 4 finger); `nq`=23 in the full
scene (+7 for the test box's freejoint).

## Actuator names (all `<position>`)

`left_arm_joint{1..6}_pos`, `right_arm_joint{1..6}_pos` (upstream, kp 5-10) and
`left_arm_joint{7,8}_pos`, `right_arm_joint{7,8}_pos` (**added**, class
`finger`, kp=200, forcerange -20..20). Fingers: **(0, 0) = closed**,
(0.035, -0.035) = fully open; joint7/joint8 have independent actuators (no
tendon coupling) -- command both together, e.g. `ctrl[joint7]=0.03,
ctrl[joint8]=-0.03` to open by 3cm.

## Collision-proxy bodies (3 per arm, for the arm-arm collision sphere set)

Old equivalent (`link3`/`link5`/`gripper_base`) -> new:

- `left_arm_link3` (upper-arm/elbow segment), `left_arm_link5` (forearm/wrist
  segment), `left_arm_gripper_base` (end-effector) -- mirrored for `right_arm_*`.

## EE / grasp sites

- `left_arm_ee` / `right_arm_ee` -- upstream, at link6 origin.
- `left_arm_ee_grasp` / `right_arm_ee_grasp` -- **added**, at the fingertip
  midpoint (`pos 0 0 0.1` in `*_gripper_base` frame), matching the old rig's
  `left/ee` convention. Use this one for grasp-point IK targeting.

## Cameras

| Name | Type | Notes |
|---|---|---|
| `left_arm_egocentric` | real `<camera>`, `mode="fixed"`, fovy 53.1 | wrist cam, mounted in `left_arm_camera_body` (was already a real camera upstream, not just a body -- no change needed) |
| `right_arm_egocentric` | same | right wrist cam |
| `head_cam` | **added**, `mode="fixed"`, `pos="1.1 -0.21 1.05"`, `xyaxes="0 1 0 -0.5924 0 0.8057"`, fovy 55 | fixed world-model camera framing the table + both arms; static pose, not targetbody-tracking |
| `front_camera`, `side_camera` | upstream, `mode="track"` attached to the `base` body | base is welded/fixed so these are effectively static too; kept as-is (secondary, not primary observation cams) |

## Table + box (shared workspace)

- Table: body `table` at `pos="0.42 -0.21 0"`; top surface **z = 0.55 m**
  (`table_top` geom center z=0.535, half-thickness 0.015). Footprint 0.60m x
  0.60m, legs down to the floor. Position chosen to be forward of the base/
  wheel footprint (verified: zero contacts between table and base geoms) and
  centered on the arms' mount-point midline.
- Box: body `test_box`, freejoint `test_box_free`, `pos="0.42 -0.21 0.57"`
  (resting on table top), 4cm cube (`size="0.02 0.02 0.02"`), mass 0.05 kg
  (50 g), friction `1.0 0.01 0.0001`, condim 4.

## Arm mount geometry + reachability (world frame, base welded/fixed)

- `left_arm_link0` world pos (at home keyframe) = `(0.005, -0.117, 0.672)`;
  `right_arm_link0` = `(0.005, -0.302, 0.672)`. **Mount separation: 0.185 m**
  in y (matches the decision-context figure), both at the same height, well
  above the floor (lift-mounted).
- Reachable EE envelope (random FK sweep, 20-80k samples over full joint
  ranges, `*_arm_ee` site, world frame):
  - left: x in [-0.60, 0.63], y in [-0.58, 0.60], z in [0.20, 1.38]
  - right: x in [-0.59, 0.63], y in [-1.01, 0.17], z in [0.21, 1.38]
  - naive AABB overlap: x in [-0.59, 0.63], y in [-0.58, 0.17], z in [0.21, 1.38]
- Both arms' EE sites can reach a common point at `(0.42, -0.21, z)` (the
  mount-point y-midline, 0.42m forward) to within ~1-2cm for z in
  approximately [0.45, 0.70] -- this is the **shared/overlap region** where
  arm-arm collisions occur; the table + box were placed at its center. A
  bimanual IK test that sends both grasp sites to the same point produces
  visibly crossed/tangled arms (see `artifacts/view_reach` exploration),
  confirming genuine overlap, not just a bounding-box coincidence.

## Gripper actuation -- what was added

Upstream `left_arm_gripper_base` / `right_arm_gripper_base` used a single
static mesh (`gripper_body.stl`, scaled 0.001, fingers baked in) with no
joint. In `yor_robot_derived.mjcf`:
1. Swapped that mesh for `gripper_base.stl` (fingers-free; this exact swap is
   pre-staged, commented out, in `YOR/robot/arm/mujoco/piper_left.xml` at the
   identical body pose -- high-confidence match, not a guess).
2. Added `left_arm_link7`/`left_arm_link8` (and `right_arm_*` mirrors) as
   child bodies of `*_gripper_base`, at `pos="0 0 0.1358"` with the Piper
   URDF's documented joint7/joint8 offsets (verified against
   `YOR/piperlib/urdf/piper_cone-e_left.urdf`), using finger meshes ported
   (absolute path, not copied) from `YOR/robot/arm/mujoco/assets/{link7,link8}.stl`.
3. Added `left_arm_joint7`/`left_arm_joint8` slide joints (+ `right_arm_*`)
   and 4 position actuators, same convention as `scenes/bimanual_piper_pickup.xml`.

## Verification (headless, `scripts/verify_yor_canonical.py`)

- **Load**: `nq=23, nv=22, nu=16, ncam=5`. All 16 actuators and 5 cameras
  enumerate correctly.
- **Grasp + lift (PASS)**: IK (damped-least-squares, seeded by an
  orientation-aware coarse random search so the solution reaches down with
  the finger open-axis horizontal) drives the left arm to the box, closes
  `left_arm_joint7/8`, then IK-lifts 10cm. Result: 4-6 finger-pad/box contacts,
  max normal force ~0.9-1.1 N, **box lifted 9.7 cm**.
- **Cameras (PASS)**: `left_arm_egocentric` std=53.5, `right_arm_egocentric`
  std=38.1, `head_cam` std=45.9 (all >>1, i.e. real rendered content, not
  blank).
- Table/base collision check: 0 contacts between table geoms and base/wheel
  geoms at rest.

## Figures

- `artifacts/yor_canonical_initial.png` -- 2-angle montage, home pose, box on table.
- `artifacts/yor_canonical_grasp.png` -- left gripper holding the lifted box.
- `artifacts/yor_canonical_wristcam.png` -- left+right wrist-cam montage during the grasp.
- `artifacts/yor_canonical_headcam.png` -- fixed head_cam view of the table/workspace.

## Rollback

Delete `scenes/yor_robot_derived.mjcf`, `scenes/scene_yor_canonical.xml`,
`scenes/YOR_CANONICAL.md`, `scripts/verify_yor_canonical.py`, and the
`artifacts/yor_canonical_*.png` files. Nothing under `YOR/` was touched.
