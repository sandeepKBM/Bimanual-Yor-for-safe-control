# Cup-task data collection (world-model observability fix + spill labels)

Addresses the gating blocker for world-model training on the "don't-spill-
the-cup" task: **does the recorded RGB actually show the ball level?** Before
this change, no. `head_cam` (the only in-scene camera) sits ~0.7m from the
cup with `mode="targetbody"`; at a 64x64 render the cup is a small red blob
and the balls inside it are not visible at all (`right/wrist_cam`, injected
onto the right gripper, doesn't even frame the cup at the arm's "home" pose
-- see the two probe renders in the task history / `artifacts/cup_headcam.png`,
`artifacts/cup_wristcam.png`).

## 1. Camera fix

Added two new **fixed** (not `targetbody`-tracking) cameras directly to
`scenes/scene_bimanual_cup.xml`'s `<worldbody>`:

- `cup_cam_a`: pos `0.4138 0.13 0.3706`, `mode="fixed"`, close-in
  (~0.17m from the ball-pile centroid), angled partial-top-down.
- `cup_cam_b`: pos `0.40501 0.16964 0.3706`, `mode="fixed"`, same distance,
  25 deg azimuth offset from `cup_cam_a` around the cup's vertical axis (mild
  stereo-like baseline).

Both pose/orientation pairs were found by an empirical sweep over ~10
candidate (position, look-at-target, fovy) combinations (top-down, side,
oblique, several azimuths/distances), rendered at the ACTUAL 64x64 recording
resolution and scored by counting "ball-blue" pixels
(`b > r+15 and b > g+5`) -- not just eyeballed at high-res. Steep top-down
views under-perform (near rim occludes the pile from directly above); the
chosen oblique angle (~56 deg above horizontal) gave ~250-330 ball-colored
px / 4096 (out of a 64x64 frame) vs. 0-20 for worse candidates -- i.e. balls
render as multi-pixel resolvable blobs, not a 2px smear, and an emptied cup
after a spill is visibly distinguishable from a full one.

**Proof**: `artifacts/cup_obs_check.png` (rendered by
`scripts/render_cup_obs_check.py`) -- 2x2 montage, top row = `cup_cam_a` |
`cup_cam_b` at rest (blue balls visible inside the cup), bottom row = same
two cameras after an upright fast-shake spill (cup visibly empty). This is
the direct visual confirmation the gating question asked for.

The original `head_cam` (wide, `targetbody`-tracking) is left unchanged in
the XML for debug/visualization montages only -- it is NOT one of the two
cameras fed to the world model.

## 2. Camera reconciliation

`collection/cameras.py` previously only supported the pickup scene (MjSpec-
injected `head_cam` + `right/wrist_cam`, duplicating pose logic that lived
nowhere else). For the cup scene, there is now exactly ONE place cameras are
defined: `scenes/scene_bimanual_cup.xml` (`cup_cam_a`/`cup_cam_b`, declarative)
plus `scripts/cup_task_utils.py`'s existing MjSpec injection of
`right/wrist_cam` (kept for compatibility/future eye-in-hand use, but not
part of the WM feed). `collection/cameras.py` adds:

- `build_cup_model()` -- calls `scripts.cup_task_utils.load_cup_model()`
  (the single source of truth for the cup scene's cameras) and asserts
  `cup_cam_a`/`cup_cam_b` exist. No separate MjSpec camera injection for the
  cup scene lives in `cameras.py` -- nothing to drift out of sync.
- `CupDualCameraRenderer` -- renders `cup_cam_a` -> `rgb_head`,
  `cup_cam_b` -> `rgb_wrist_right` (episode.py's existing array key names,
  repurposed; see below). This is what feeds `world_model/dataset_adapter.py`'s
  `image = concat([rgb_head, rgb_wrist_right])` 2-cam -> 6-channel contract,
  unchanged.

**The two cameras feeding the world model for the cup task are `cup_cam_a`
and `cup_cam_b`** (both fixed, both close-in on the cup).

## 3. Spill labels wired into collection

New `collection/collect_cup.py` (parallel to `collect.py`, which stays
untouched and still serves the box-pickup task). Per-step,
`scripts.cup_task_utils.count_spilled(model, data)` is called every step;
`episode.py`'s existing schema (shared with the box task, not modified, so
`world_model/dataset_adapter.py` needs zero changes) is reused with two
fields **repurposed** and documented in `collect_cup.py`'s module docstring:

- `box_pose` -> the **cup** body's (xyz, wxyz-quat) pose each step (no "box"
  exists in this scene).
- `dropped` (bool) -> per-step **spill flag** (`count_spilled(t) > 0`)
  instead of "box fell". Combined with `collision`, `dataset_adapter.py`'s
  existing `is_terminal = collision | dropped` now means, with zero code
  changes, "collision OR spill this step" for cup episodes.
- `metadata["spilled_count"]` -- previously hardcoded `0` everywhere (the
  cup scene didn't exist when `episode.py`/`collect.py` were written) -- is
  now the REAL final-step `count_spilled(...)` value.
- `metadata["grasp_success"]` -> repurposed as "cup contents were not
  spilled" (nothing is grasped in this task).

New `perturbations.py` family `"cup_behavior"` (`CUP_BEHAVIOR_LEVELS`),
seeded via `Random(seed)` per episode, selects + parametrizes the
demonstrator motion (the cup has no grasp/IK in this repo, so it is moved by
directly overwriting its freejoint qpos each step, same "teleport" technique
already validated in `scripts/test_cup_scene.py`):

| level | motion | expected outcome |
|---|---|---|
| `success_gentle` | slow, small (0.5-1.2cm, 1-2.5Hz) upright lateral jitter | no spill |
| `spill_shake` | fast upright lateral shake (3.2-4.5cm @ 5.5-6.5Hz, tuned range from `scripts/test_cup_scene.py`'s `run_upright_shake_test`) | spills via acceleration alone, cup stays upright |
| `spill_tilt` | lift + tilt 90-115 deg (`run_tilt_test`-style) | spills via tipping |
| `collision_intrusion` | right arm runs `sim_common.intruder_ctrl` (the SAME sweep `collect.py`'s box task uses), cup left essentially undisturbed | exercises arm/arm collision; may incidentally spill but is not scripted to |

`collect_episode(..., level=None)` defaults to `CUP_BEHAVIOR_LEVELS[seed % 4]`
for an even mix; `--level <name>` forces one behavior for all episodes.

## 4. Small verification collection (16 episodes, seeds 100-115)

Command actually run:
```
MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env python \
  -m collection.collect_cup --num-episodes 16 --seed-start 100 \
  --out-dir collection/data/cup_verify
```
Result (16/16 episodes, seed % 4 cycling through the 4 behaviors, 4 of each):
```
episode_000000 seed=100 level=success_gentle      success=True  spilled_count=0 collision=False steps=881
episode_000001 seed=101 level=spill_shake         success=False spilled_count=8 collision=False steps=893
episode_000002 seed=102 level=spill_tilt          success=False spilled_count=1 collision=False steps=1123
episode_000003 seed=103 level=collision_intrusion success=True  spilled_count=0 collision=False steps=1300
episode_000004 seed=104 level=success_gentle      success=False spilled_count=1 collision=False steps=870
episode_000005 seed=105 level=spill_shake         success=False spilled_count=8 collision=False steps=858
episode_000006 seed=106 level=spill_tilt          success=False spilled_count=8 collision=False steps=1180
episode_000007 seed=107 level=collision_intrusion success=False spilled_count=0 collision=True  steps=1300
episode_000008 seed=108 level=success_gentle      success=True  spilled_count=0 collision=False steps=914
episode_000009 seed=109 level=spill_shake         success=False spilled_count=8 collision=False steps=809
episode_000010 seed=110 level=spill_tilt          success=False spilled_count=8 collision=False steps=1078
episode_000011 seed=111 level=collision_intrusion success=False spilled_count=0 collision=True  steps=1300
episode_000012 seed=112 level=success_gentle      success=True  spilled_count=0 collision=False steps=836
episode_000013 seed=113 level=spill_shake         success=False spilled_count=8 collision=False steps=844
episode_000014 seed=114 level=spill_tilt          success=False spilled_count=8 collision=False steps=1020
episode_000015 seed=115 level=collision_intrusion success=False spilled_count=0 collision=True  steps=1300
wrote 16 episodes in 145.7s -- spilled=9 collision=3 clean=4
```
Outcome mix (by ground truth, not scripted intent): **9/16 spilled** (all 4
`spill_shake`, 3/4 `spill_tilt` at full 8/8, 1/4 `spill_tilt` + 1/4
`success_gentle` at partial 1/8 -- physically real: a gentle jitter or an
under-rotated tilt can still nudge one ball over the rim), **3/16 collision**
(3 of 4 `collision_intrusion` episodes registered left/right arm contact
during the intrusion sweep's oscillating hold, matching `collect.py`'s own
box-task experience that this sweep produces contact only some of the time,
not deterministically), **4/16 clean** (0 spilled, 0 collision: 3
`success_gentle` + 1 `collision_intrusion` that didn't touch). No episode was
both spilled and collision in this batch.

Spot-checks (`scripts/inspect_cup_episode.py <episode.npz> <out.png>`,
montage = t=0 `cup_cam_a` | t=mid `cup_cam_a` | t=mid `cup_cam_b`):
- `artifacts/cup_spotcheck_spill_shake.png` (episode_000001): balls visible
  at t=0; cup visibly EMPTY at t=mid -- matches `metadata.spilled_count == 8`.
- `artifacts/cup_spotcheck_success_gentle.png` (episode_000000): balls
  visible in all 3 frames, `spilled_count == 0`, `collision == False`.
- `artifacts/cup_spotcheck_collision_intrusion.png` (episode_000003, probe
  run): balls visible throughout (no spill), `collision == True` for a
  window of steps during the intrusion sweep's hold phase, `spilled_count
  == 0`.
Labels match ground truth in every spot-checked episode.

## 5. Adapter ingestion

`world_model/dataset_adapter.py` was run UNMODIFIED against the 16 verify
episodes (using the `mujoco_ur5e` conda env, which has `torch`/`tensorboard`
-- `safe_env` does not and errors on `import tools`):
```
/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.dataset_adapter \
  --collection-dir collection/data/cup_verify \
  --out-dir world_model/data/dreamer_fmt/cup_verify
```
Result: **16/16 episodes converted, zero errors.** Spot-checked output
(`episode_000000-881.npz`): `image (881,64,64,6) uint8`, `proprio (881,33)
float32`, `action (881,16) float32`, `reward/discount/is_first/is_terminal
(881,) f32/f32/bool/bool` -- exactly the documented contract, no adapter
code changes needed.

## Full collection run (recommended)

```
MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env python \
  -m collection.collect_cup --num-episodes 200 --seed-start 0 \
  --out-dir collection/data/cup_full
```
200 episodes (seed % 4 cycling through all 4 behaviors -> ~50 each) at the
per-behavior episode lengths `collect_cup.py` derives automatically
(`success_gentle`/`spill_shake` ~0.7-1.3s = 350-850 steps,
`spill_tilt` ~1.2-2.3s = 600-1150 steps, `collision_intrusion` fixed 2.6s =
1300 steps) gives ~150k total steps, comparable in scale to `collect.py`'s
own recommended full run. Runtime: the 16-episode verification run above
took ~146s wall-clock (single process, EGL
rendering, 64x64x2 cams), so 200 episodes should take roughly 12-13x that
(~1x per behavior-length-weighted episode) -- parallelize across seed
ranges / submit via `collection/submit_collect_rlab.sh`-style sbatch
sharding if wall-clock matters (not wired up for `collect_cup.py` yet --
`collect.sbatch`/`submit_collect_rlab.sh` currently only invoke
`collection.collect`, not `collection.collect_cup`; a one-line module-name
change would point them at the cup task).

## Blockers

None blocking. Rendering requires `MUJOCO_GL=egl` (headless node, no
`DISPLAY`).

## Rollback

Delete `scenes/scene_bimanual_cup.xml`'s two new `<camera name="cup_cam_a"/
cup_cam_b">` lines (restores the pre-fix XML); delete
`collection/collect_cup.py`, `collection/CUP_DATA.md`,
`scripts/render_cup_obs_check.py`, `scripts/inspect_cup_episode.py`,
`artifacts/cup_obs_check.png`, and `collection/data/cup_verify/`; revert the
`cup_behavior` family addition in `collection/perturbations.py` and the
`build_cup_model`/`CupDualCameraRenderer` addition in `collection/cameras.py`.
Nothing in `collection/collect.py`, `collection/episode.py`'s schema, or
`world_model/dataset_adapter.py` was modified.
