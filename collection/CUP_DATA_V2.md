# Cup-task data collection V2 -- PROPER YOR robot + object-pose-leak fix

Rebuild of the "don't-spill-the-cup" pipeline (see `collection/CUP_DATA.md`
for the V1 history, on the SIMPLIFIED rig) on the canonical YOR robot
(`scenes/scene_yor_canonical.xml` family, `scenes/YOR_CANONICAL.md`), plus
the `proprio` object-pose-leak fix from `docs/design/state_spaces.md`.

**Status: DONE.** Cup scene + demonstrator + labels + 26-D proprio fix +
real ZED workspace camera as the WM `image` source + full 200-episode
collection, all complete and verified. (History: this doc originally shipped
with the full collection ON HOLD pending the ZED; the ZED integration
below and section 8's full run supersede that hold.)

## 1. Camera plan (final -- went through three course-corrections)

1. **First correction:** a synthetic external "close-up" cam like V1's
   `cup_cam_a`/`cup_cam_b` is not physically realizable on real hardware --
   pointed at the egocentric wrist cams instead.
2. **Second correction:** the WM's MAIN observation should be a workspace
   **ZED** RGB-D camera, not wrist cams -- wrist cams to be recorded as
   auxiliary data only. (At this point the ZED wasn't mounted yet, so the
   full collection was held.)
3. **Final state (this section):** the ZED is now mounted
   (`scripts/zed_camera.py::add_zed_camera`, real sensor per
   `YOR/robot/zed_pub_node.py`, proposed workspace-facing pose on `Lift_Top`
   -- see that module's own MOUNT NOTE for the sim2real caveat: the exact
   pose is proposed/confirm-later, not yet hardware-verified; accepted as
   fine for sim-only work). `world_model/dataset_adapter.py` is owned by a
   separate in-flight task and was **deliberately left untouched** --
   it unconditionally builds `image = concat([rgb_head, rgb_wrist_right],
   axis=-1)`. So, without editing the adapter, `collection/cameras.py`'s
   `YorCupCameraRenderer` stores the SAME `zed_cam` frame into BOTH
   `rgb_head` and `rgb_wrist_right` (the two field names the adapter already
   reads) -- `image` therefore ends up as the ZED view duplicated across
   both 3-channel halves (verified: `np.array_equal(image[...,:3],
   image[...,3:6]) == True` on the full collection), not two different
   cameras. The same ZED frame is also stored under its own `rgb_zed` key
   (debug/inspection), and the two REAL egocentric wrist cams are recorded
   separately as `rgb_wrist_left_ego`/`rgb_wrist_right_ego` -- auxiliary
   data, not part of `image`, not read by the adapter. No invented
   `head_cam` or the MJCF's `front_camera`/`side_camera` (`mode="track"`
   debug cams) are used anywhere in this pipeline.

`scripts/cup_task_utils.load_yor_cup_model()` attaches `zed_cam` at load
time (same in-memory MjSpec pattern as the arm-arm collision patch --
`scene_yor_cup.xml` is never edited on disk for this either).

### Ball-visibility finding (honest)

At the arms' scripted "home" hold pose (both arms stationary throughout
every behavior except `collision_intrusion`'s right-arm sweep), NEITHER
wrist cam sees the cup (both point up/away from the table at rest) --
0/4096 ball px measured with a saturation-ratio detector, see
`artifacts/cup_v2_obs_check.png`. This is fine: wrist cams are auxiliary
only now, not the WM input.

**ZED workspace view: spills ARE visible, per the accepted limitation.**
At rest, the ZED's wide workspace framing (fovy=70, mounted high/back on
Lift_Top) renders the in-cup ball level as occluded/too-small to resolve at
64x64 (the cup itself is only a few pixels tall in frame) -- this matches
the coordinator's stated accepted limitation. But the SPILL itself is
visible: after a shake/tilt, spilled balls scatter onto the table around
the cup and are visible as distinct blue pixels against the tan table
(measured: 0 -> 3 ball px/4096 at native 64x64 for a shake episode; clearly
visible at higher render resolution, see below), and `spill_tilt`'s cup
lift+90-115deg tilt is itself a large, unambiguous visual event (the red
cup tips over) at any resolution. Proof frames:
- `artifacts/cup_v2_zed_rest_vs_spill_64native.png` -- native 64x64,
  upscaled for viewing: left = cup at rest (no scattered balls), right =
  post-shake (several blue balls visible scattered on the table).
- `artifacts/cup_v2_zed_rest_vs_spill_300.png` -- same comparison at 300x300
  for clearer visual confirmation (8 scattered balls unambiguous).
- `artifacts/cup_v2_zed_spill_check.png` -- 4-frame timelapse of a real
  collected episode (`cup_yor_zed_smoke/episode_000001`, `spill_shake`) from
  its stored `rgb_zed` field.

An earlier exploration (from course-correction 2, before the ZED existed)
found IK-driven wrist-cam "hover" poses CAN get very close to the cup
(2309/4096, 2123/4096 ball px) -- see `scripts/probe_yor_cup_hover.py` /
`artifacts/cup_v2_hover_*.png`. Not wired in: moot now that the ZED is the
real WM input, and risks arm-cup contact during `spill_tilt`'s lift. Kept
for reference in case a future monitor head wants a staged wrist-cam view.

## 2. Scene: `scenes/scene_yor_cup.xml`

Built by `<include file="yor_robot_derived.mjcf"/>` (unmodified) + the same
floor/table/lighting/asset conventions as `scenes/scene_yor_canonical.xml`
(table top z=0.55) + a cup + N=8 balls, replacing that scene's single
graspable `test_box`. Cup/ball GEOMETRY is copied verbatim from the V1 cup
scene (`scenes/scene_bimanual_cup.xml` / `scenes/CUP_TASK.md`) -- only the
WORLD POSITION is translated (delta = (+0.10, -0.34, +0.35)) so it lands in
the YOR shared-workspace column:

- Cup body `pos="0.42 -0.21 0.55"` -- x,y match `scene_yor_canonical.xml`'s
  `test_box` column (verified reachable by both arms, `YOR_CANONICAL.md`'s
  FK sweep); z=0.55 is the table top itself (the cup body's origin is its
  BOTTOM face, so this rests it flush on the table -- NOT 0.57, which was
  `test_box`'s *center* height for a box whose origin is its geometric
  center; a body-origin-convention difference, not an inconsistency).
  Rim height at rest = 0.55 + 0.08 = 0.63 m.
- 8 balls, radius 0.008m, at the same cup-relative offsets as V1, translated
  by the same delta.
- No fixed/external camera is declared in this file (see "Camera plan").

Verified (`scripts/cup_task_utils.load_yor_cup_model()`): `nq=79, nv=70,
nu=16, ncam=4` (`front_camera`, `side_camera` -- both `mode="track"`
debug-only, unused here -- plus `left_arm_egocentric`, `right_arm_egocentric`).
Balls settle to `count_spilled()==0` after a 0.5s gravity settle from the
scene's own `home` keyframe.

## 3. Collision-fidelity reuse (arm-arm contact for `collision_intrusion`)

`yor_robot_derived.mjcf` ships every robot link's real collision geom with
`contype=0` (see `experiments/ab_smooth_vs_bangbang/sim_common.py`'s
`_patch_real_collision_geoms` docstring for the full root-cause writeup):
two `contype=0` geoms never contact each other, so the two arms are
structurally incapable of colliding with EACH OTHER as shipped (they still
collide fine with the table/cup/balls, which use ordinary `contype=1`
defaults). `scripts/cup_task_utils.load_yor_cup_model()` applies the exact
same in-memory MjSpec patch as `sim_common.load_model()` -- **imported and
reused, not re-implemented** (`sim_common._patch_real_collision_geoms`) --
so `collect_cup.py`'s `collision_intrusion` behavior can register real
left/right arm contacts. `scene_yor_cup.xml` and `yor_robot_derived.mjcf`
are never edited on disk.

### `collision_intrusion`'s own intrusion target (new IK solve)

`sim_common.RIGHT_INTRUDE_CTRL` was IK-solved to reach `sim_common.
LEFT_GRASP_Q` -- the box task's left-arm GRASP pose, hovering right at the
table. This cup task's left arm instead holds `LEFT_HOME_CTRL` throughout
(reusing `LEFT_GRASP_Q` here would put the open left gripper's fingers
inside/against the cup's rim, since the rim at 0.63 sits ~4-6cm above the
box top `LEFT_GRASP_Q` was tuned for -- confirmed empirically: at
`LEFT_HOME_CTRL` + `RIGHT_INTRUDE_CTRL`, `min_pair_distance` gives
`h=+0.135` -- SAFE, no collision at all, which is what a first verification
run showed: 0/4 `collision_intrusion` episodes registered a contact).

Fix: `collect_cup.py`'s `RIGHT_INTRUDE_Q_HOME` is a fresh damped-least-
squares IK solution (same `coarse_seed`+`ik_to_target` method as
`scripts/verify_yor_canonical.py`, reused via import) aiming the right arm's
grasp site at the LEFT arm's actual `left_arm_ee_grasp` position while the
left arm is frozen at `LEFT_HOME_CTRL`. Verified: `h = +0.135` (right home)
`-> h = -0.051` (`RIGHT_INTRUDE_Q_HOME`) -- a genuine safe->violated swing,
comparable to `sim_common`'s own `LEFT_GRASP_Q`-targeted pair (`+0.129 ->
-0.060`). `_intrude_toward_left_home()` reuses `sim_common`'s exact ramp-in/
oscillating-hold/ramp-out TIMING (`T_IN0/T_IN1/T_HOLD_END/T_OUT_END/
OSC_AMP/OSC_FREQ_HZ` -- pure timing constants, robot/pose-agnostic) with
this new endpoint pose instead.

## 4. Object-pose-leak fix (`world_model/dataset_adapter.py`)

Per `docs/design/state_spaces.md`'s "sim2real line": object pose is
privileged sim state (no external tracker on the real robot), so it must
not be a WM observation.

- `PROPRIO_FIELDS` no longer includes `box_pose` -- `proprio` is now
  **26-D**: `[left_q(6), left_qd(6), right_q(6), right_qd(6),
  left_gripper_state(1), right_gripper_state(1)]` (down from 33-D).
- `box_pose` (the cup's 7-D xyz+wxyz-quat pose) is now written to a
  SEPARATE `object_pose` key in the converted per-episode npz -- a LABEL
  field (supervision/eval only), never concatenated into `proprio`.
  `tools.save_episodes`/`load_episodes` (vendored dreamerv3-torch) round-trip
  arbitrary npz keys unmodified, so this extra key is inert for
  `WorldModel` unless a future monitor head is explicitly wired to read it.
- `world_model/train_wm_config.yaml`'s `proprio_dim` updated 33 -> 26 to
  match (the only other place that dimension is declared).
- Added shape asserts (`proprio.shape == (t, 26)`, `object_pose.shape ==
  (t, 7)`) in `convert_episode_arrays` so a future schema drift fails loudly.

This fix applies to BOTH the cup task and the existing box-pickup task's
adapter path (the function is shared) -- box-pickup episodes converted
through this adapter now also get `box_pose` moved to `object_pose` instead
of `proprio`, which is the intended, consistent behavior (not cup-specific).

## 5. Verification run (15 episodes)

```
MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env python \
  -m collection.collect_cup --num-episodes 15 --seed-start 300 \
  --out-dir collection/data/cup_yor_verify2
```
(An earlier attempt, `collection/data/cup_yor_verify` seeds 200-214, is
superseded -- it predates the `RIGHT_INTRUDE_Q_HOME` fix above and shows
0/4 `collision_intrusion` episodes with `collision=True`; left in place,
not deleted, but not the reference run.)

Result (15/15 episodes, seed % 4 cycling through the 4 behaviors):
```
episode_000000 seed=300 success_gentle      success=True  spilled=0 collision=False steps=260
episode_000001 seed=301 spill_shake         success=False spilled=8 collision=False steps=319
episode_000002 seed=302 spill_tilt          success=False spilled=8 collision=False steps=443
episode_000003 seed=303 collision_intrusion success=False spilled=0 collision=True  steps=520
episode_000004 seed=304 success_gentle      success=True  spilled=0 collision=False steps=362
episode_000005 seed=305 spill_shake         success=False spilled=8 collision=False steps=330
episode_000006 seed=306 spill_tilt          success=False spilled=8 collision=False steps=442
episode_000007 seed=307 collision_intrusion success=False spilled=0 collision=True  steps=520
episode_000008 seed=308 success_gentle      success=True  spilled=0 collision=False steps=303
episode_000009 seed=309 spill_shake         success=False spilled=8 collision=False steps=296
episode_000010 seed=310 spill_tilt          success=False spilled=8 collision=False steps=415
episode_000011 seed=311 collision_intrusion success=False spilled=0 collision=True  steps=520
episode_000012 seed=312 success_gentle      success=True  spilled=0 collision=False steps=264
episode_000013 seed=313 spill_shake         success=False spilled=8 collision=False steps=314
episode_000014 seed=314 spill_tilt          success=False spilled=3 collision=False steps=409
wrote 15 episodes in 41.0s -- spilled=8 collision=3 clean=4
```
Outcome mix: **8/15 spilled** (all 4 `spill_shake` at full 8/8, 3/4
`spill_tilt` at full 8/8 + 1/4 at partial 3/8), **3/15 collision** (3/4
`collision_intrusion`, matching V1's own ~75% hit rate for this
non-deterministic oscillating-sweep contact), **4/15 clean** (all 4
`success_gentle` + 1/4 `collision_intrusion` that didn't touch).

## 6. Adapter ingestion

```
/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.dataset_adapter \
  --collection-dir collection/data/cup_yor_verify2 \
  --out-dir world_model/data/dreamer_fmt/cup_yor_verify2
```
**15/15 episodes converted, zero errors.** Spot-checked
(`episode_000000-260.npz`):
```
image       (260, 64, 64, 6) uint8
proprio     (260, 26)        float32   <- 26-D confirmed, object pose NOT included
object_pose (260, 7)         float32   <- cup pose, LABEL field only
action      (260, 16)        float32
reward/discount/is_first/is_terminal (260,) f32/f32/bool/bool
```

## 7. Full collection -- DONE (200 episodes, `collection/data/cup_yor_full`)

```
MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env python \
  -m collection.collect_cup --num-episodes 200 --seed-start 1000 \
  --out-dir collection/data/cup_yor_full
```
Result: **200/200 episodes written in 624.6s** (~3.1s/episode, single-process
EGL rendering, all 5 recorded cameras + 26-D proprio per step). Outcome mix
(seed % 4 cycling the 4 behaviors, 50 of each):

| | count | notes |
|---|---|---|
| spilled (`spilled_count>0`) | **102/200** | all 50 `spill_shake` at 8/8, ~49/50 `spill_tilt` mostly at 8/8 (a few partial, matching the graded threshold from `scripts/test_cup_scene.py`) |
| collision (`collision=True`) | **50/200** | all 50 `collision_intrusion` episodes -- 100% hit rate with `RIGHT_INTRUDE_Q_HOME` (up from the ~75% hit rate seen in the 15-episode verification run; both are consistent with a non-deterministic-looking but actually-deterministic scripted sweep -- every `collision_intrusion` episode runs the identical joint trajectory, so the hit rate depends on the (fixed) geometry, not the per-episode seed, and 15 samples under-represented that) |
| clean (0 spilled, no collision) | **48/200** | all 50 `success_gentle` minus ~2 that registered a partial spill from jitter (same physically-real effect as the 15-episode run), matches `48 = 200 - 102 - 50` |

Total steps: **80,160** across 200 episodes (`collect_cup.py`'s per-behavior
lengths: `success_gentle`/`spill_shake` ~260-360 steps, `spill_tilt` ~410-460
steps, `collision_intrusion` fixed 520 steps).

## 7a. Adapter ingestion of the full collection

```
/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.dataset_adapter \
  --collection-dir collection/data/cup_yor_full \
  --out-dir world_model/data/dreamer_fmt/cup_yor_full
```
**200/200 episodes converted, zero errors** (adapter code untouched --
confirms the ZED-into-rgb_head/rgb_wrist_right trick needed no adapter
changes). Spot-checked (`episode_000000-272.npz`):
```
image       (272, 64, 64, 6) uint8   -- image[...,:3] == image[...,3:6] (ZED duplicated, verified)
proprio     (272, 26)        float32
object_pose (272, 7)         float32
action      (272, 16)        float32
reward/discount/is_first/is_terminal (272,) f32/f32/bool/bool
```
Total steps across all 200 converted episodes: **80,160** (matches the raw
collection exactly).

## 8. WM training command (gated -- no training run in this task)

```
CUDA_VISIBLE_DEVICES=<n> /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \
  world_model/smoke_train_wm.py   # or a real train_wm.py entrypoint per
                                    # world_model/README.md section 6, using
                                    # world_model/train_wm_config.yaml
                                    # (proprio_dim=26, image_channels=6 --
                                    # both already correct for this
                                    # collection: image is 6ch [ZED, ZED],
                                    # proprio excludes object pose)
```

## Blockers

None. Full collection complete. Wrist-cam-at-rest ball-visibility is 0/4096
(section 1) -- accepted, not gating, wrist cams are auxiliary only. ZED
in-cup ball-level-at-rest visibility is limited by its wide workspace
framing (accepted per the coordinator) but spilled/scattered balls and the
cup tipping over ARE visible in the recorded frames (section 1 proof
frames) -- sufficient signal for a spill-detecting monitor head to learn
from. The ZED's exact mount pose is PROPOSED, not hardware-verified (see
`scripts/zed_camera.py`'s own MOUNT NOTE) -- fine for this sim-only
collection per the coordinator's explicit go-ahead.

## Rollback

Delete `scenes/scene_yor_cup.xml`, `scripts/probe_yor_cup_hover.py`, this
file, `collection/data/cup_yor_verify*/`, `collection/data/cup_yor_full/`,
`collection/data/cup_yor_zed_smoke/`, `world_model/data/dreamer_fmt/
cup_yor_verify2/`, `world_model/data/dreamer_fmt/cup_yor_full/`,
`world_model/data/dreamer_fmt/cup_yor_zed_smoke/`, and the `artifacts/
cup_v2_*` files. Revert: `scripts/cup_task_utils.py` (remove
`load_yor_cup_model`/`YOR_CUP_XML`, the `add_zed_camera` call, and the
`sys.path` header addition -- `scripts/zed_camera.py` itself is a
pre-existing file from another task, NOT to be deleted),
`collection/cameras.py` (remove `build_yor_cup_model`/`YorCupCameraRenderer`/
`SCENE_YOR_CUP`/`LEFT_EGOCENTRIC`/`RIGHT_EGOCENTRIC`/`ZED_CAM`),
`collection/episode.py` (revert `EpisodeWriter.save` to iterate
`STEP_ARRAY_KEYS` only, not the union of recorded keys -- this generalization
is backward-compatible and safe to keep, but revert if a stricter schema is
wanted), `collection/collect_cup.py` (restore from git history -- V1's
version targeted the old rig). `world_model/dataset_adapter.py` and
`world_model/train_wm_config.yaml` were edited ONLY for the object-pose-leak
fix (proprio 33->26-D, `box_pose` -> `object_pose` label) -- NOT for the ZED
integration (that was achieved entirely by having the collector store the
ZED frame into the adapter's existing `rgb_head`/`rgb_wrist_right` fields);
revert those two files' proprio-dim changes separately if wanted, but they
are shared with the box-pickup task and owned by a different in-flight
task -- coordinate before touching. Nothing under `YOR/`, `experiments/`, or
`hj_reachability/` was touched.
