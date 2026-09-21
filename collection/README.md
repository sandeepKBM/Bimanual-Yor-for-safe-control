# bimanual/collection

Minimal data-collection + training-support infra for a future Dreamer/RSSM
world-model, ported/adapted from `VLAism/ur5e_benchmark`. Scene:
`scenes/scene_bimanual_pickup.xml` (the cup scene, `scenes/scene_bimanual_cup.xml`,
does not exist yet -- see "Cup scene" below).

## World-model recon

No existing Dreamer/DreamerV3/RSSM/world-model trainer found on this machine
(`/common/users/ss5772`, the VLAism tree, no `*dreamer*`/`*world_model*`/
`*dino*wm*`/`*rssm*` dirs anywhere under `/common/users/ss5772` at depth <=4).
**None found -- a world-model trainer must be installed/written from scratch.**
Env `mujoco_ur5e` (`/common/users/ss5772/miniforge3/envs/mujoco_ur5e`) has
`mujoco==3.9.0` and `torch==2.12.1+cu130` -- usable as the training env once a
Dreamer/RSSM implementation is added; nothing was cloned or installed per
task instructions.

## Files

| File | Ported / Copied / Adapted |
|---|---|
| `episode.py` | Adapted from `ur5e_benchmark/benchmark/episode_schema.py` (dataclass + `build_episode_metadata` pattern) and `artifact_keys.py` (SSOT-tuple-of-required-keys idea). UR5 apple/box-placement fields dropped; bimanual step-array schema (images + both arms' q/qdot/action/gripper) is new. One `.npz` + one `.json` per episode (closer to `grasp_boundary/converter.py`'s per-episode-file idea than ur5e_benchmark's multi-file split). |
| `perturbations.py` | Ported the `PerturbationSpec` dataclass shape and `_sample_xy_offset` helper verbatim from `ur5e_benchmark/benchmark/perturbations.py` (both UR5-agnostic). Family set rebuilt for two arms + a box; `ball_config` is a **documented stub** (no cup/balls scene exists yet). |
| `cameras.py` | New. `scenes/scene_bimanual_pickup.xml` defines no cameras, and scene files were not touched (writes are collection/-only) -- cameras are injected at load time via MuJoCo's `MjSpec` API (`head_cam` on worldbody, `right/wrist_cam` on `right/gripper_base`), then compiled. |
| `collect.py` | New rollout loop. Reuses (imports, does not copy) `experiments/ab_smooth_vs_bangbang/sim_common.py` for model loading, joint bookkeeping, and the already-validated `intruder_ctrl` right-arm sweep (whose own docstring notes it can "shake the box loose" / cause arm-arm contact -- exactly the drop/collision signal we want). The grasp-init teleport technique is adapted from `sim_common.init_grasped_state`, parametrized by a per-episode grasp-strength scalar and an xy box-offset. |
| `collect.sbatch`, `submit_collect_rlab.sh` | Scaffold, modeled on `scripts/offload/slurm_box_episode_uploader.sbatch` / `submit_via_slurm_gateway.sh`. Gateway logic itself (`scripts/offload/slurm_gateway.sh`) reused unchanged. Not executed. |

## UR5-specific assumptions changed

- Single arm -> two arms (`left/*`, `right/*` joint/body namespacing already
  present in the scene).
- UR5's Cartesian-space `arm_start_shift` (uses a UR5 IK-adjacent pregrasp
  offset) is replaced here by a **joint-space** simplification: this minimal
  demonstrator has no IK, so `arm_start_shift` perturbs `q0_left[0:2]` /
  `q0_right[0:2]` directly, reusing the family's `*_dx_m/_dy_m` values as
  radian offsets via a fixed `3.0 rad/m` scale (`ARM_JOINTSPACE_SCALE_RAD_PER_M`
  in `collect.py`). Documented here rather than silently reinterpreted.
- UR5's apple/box "placement_success" + "object_shape" fields dropped (no
  analogous placement target in the pickup scene).
- No lift phase: the demonstrator only *holds* the post-grasp pose while the
  right arm runs the intrusion sweep. This was enough to produce success /
  dropped / collision episodes in testing (see Smoke result) without needing
  a real IK-based pick-and-lift controller, which was out of scope for this
  port.

## Gripper convention (the one, unambiguous choice)

```
gripper_state = qpos(<side>/joint7) / 0.035    # in [0, 1]
    0.0 == fully closed
    1.0 == fully open
```
`joint7`'s MJCF range is `[0, 0.035]` m (slide joint); `joint8` mirrors it
with the opposite sign and is not separately encoded (both fingers are
always commanded together). This one number is recorded per arm per step
(`left_gripper_state`, `right_gripper_state`).

## Episode format

`collection/data/<run>/episode_NNNNNN/`:
- `episode.npz` -- per-step arrays, all length `T = num_steps` (see
  `episode.STEP_ARRAY_KEYS` for the exact SSOT list):
  `rgb_head (T,H,W,3) u8`, `rgb_wrist_right (T,H,W,3) u8`,
  `left_q/left_qd/right_q/right_qd (T,6) f32`,
  `left_action/right_action (T,8) f32` (6 arm ctrl + 2 finger ctrl),
  `left_gripper_state/right_gripper_state (T,) f32`,
  `box_pose (T,7) f32` (xyz + wxyz quat),
  `collision (T,) bool`, `dropped (T,) bool`.
- `metadata.json` -- episode-level scalars: `episode_id`, `seed`,
  `perturbation_family/level/parameters`, `policy_name`, `num_steps`,
  `success`, `grasp_success`, `collision`, `spilled_count` (always `0` today
  -- see "Cup scene" below), `gripper_convention`, `schema_compat_version`.

This per-episode npz+json layout is what a Dreamer/RSSM dataloader wants
(one contiguous trajectory blob per episode, load-and-stack across episodes
for a replay buffer).

## Cup scene / `spilled_count`

`scenes/scene_bimanual_cup.xml` does not exist yet. `spilled_count` is kept
in the metadata schema now (so the writer/dataloader contract does not
change later) but is hardcoded to `0`; `perturbations.py`'s `ball_config`
family returns `{"applicable": false, ...}` rather than sampling anything.
**Smoke below ran against the pickup scene, per task fallback instructions.**
Once the cup scene lands: (1) wire `ball_config`'s `count`/`spread_radius_m`
sampling, (2) compute `spilled_count` per step as the number of `ball_*`
bodies whose xy leaves the cup rim radius, (3) add a `left/wrist_cam` /
`right/wrist_cam`-style camera on the cup-holding arm if not already covered
by `head_cam`.

## Collection command + smoke result

```
cd /common/users/ss5772/bimanual
MUJOCO_GL=egl /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \
  -m collection.collect --num-episodes 3 --num-steps 60 --render-size 64 \
  --out-dir collection/data/smoke
```
(`MUJOCO_GL=egl` is required -- this node is headless / no `DISPLAY`; GLX
fails, EGL via the NVIDIA driver works.)

Result (3 episodes, 60 steps = 0.12s each -- short on purpose per task
instructions, too short to reach the intrusion sweep's ramp-in at t=0.4s):
```
[collect] episode=episode_000000 seed=0 success=True grasp_success=True collision=False steps=60
[collect] episode=episode_000001 seed=1 success=True grasp_success=True collision=False steps=60
[collect] episode=episode_000002 seed=2 success=True grasp_success=True collision=False steps=60
[collect] wrote 3 episodes to collection/data/smoke in 2.1s (3/3 success)
```
`episode.npz` arrays verified: `rgb_head (60,64,64,3) uint8`,
`rgb_wrist_right (60,64,64,3) uint8`, `left_q/left_qd/right_q/right_qd
(60,6) float32`, `left_action/right_action (60,8) float32`,
`left_gripper_state/right_gripper_state (60,) float32`, `box_pose (60,7)
float32`, `collision`/`dropped (60,) bool`; `metadata.json` has all
`METADATA_REQUIRED_KEYS`.

An additional (not part of the required 2-3 episode smoke, kept only as
extra verification and left on disk under `collection/data/smoke_long/`) run
at full 3000-step / 6s length confirmed both failure modes actually fire:
```
[collect] episode=episode_000000 seed=10 success=False grasp_success=True  collision=True  steps=3000
[collect] episode=episode_000001 seed=11 success=False grasp_success=False collision=False steps=3000
[collect] episode=episode_000002 seed=12 success=False grasp_success=False collision=False steps=3000
[collect] episode=episode_000003 seed=13 success=False grasp_success=False collision=False steps=3000
```
i.e. with `--num-steps 3000` (full 6s, matching `sim_common`'s intrusion
timeline) the demonstrator does produce a mix of collision and drop
episodes, not just successes -- the 60-step smoke above is simply too short
for the (deterministic, time-gated) intrusion sweep to have engaged yet.

`bash -n collection/submit_collect_rlab.sh` and `bash -n
collection/collect.sbatch` both pass (syntax-only check; not executed).

## What's needed for a real (large) collection run

1. Use `--num-steps >= 3000` (the full 6s intrusion timeline) so the
   collision/drop branches are actually exercised; 60-step episodes are
   success-only.
2. Decide a real target episode count / seed range and submit via
   `collection/submit_collect_rlab.sh` (sets `NUM_EPISODES`, `OUT_DIR`, etc.,
   then calls the existing `scripts/offload/submit_via_slurm_gateway.sh`
   gateway) -- not run in this task.
3. Pick/port an actual Dreamer/RSSM trainer (none exists on this machine) and
   point its dataloader at `collection/data/<run>/episode_*/episode.npz`.
4. Once `scenes/scene_bimanual_cup.xml` exists, wire `ball_config` sampling
   and real `spilled_count` computation (see "Cup scene" above).
5. Consider a `left/wrist_cam` (symmetric to `right/wrist_cam`) if the
   world-model needs eye-in-hand views from both arms.

## Blockers

None blocking this port. Headless rendering requires `MUJOCO_GL=egl`
(no `DISPLAY` on this node) -- already set in `collect.sbatch`.

## Rollback

`rm -rf /common/users/ss5772/bimanual/collection/` -- no other files were
modified (scenes/, experiments/, scripts/offload/ were only read/imported,
never written).
