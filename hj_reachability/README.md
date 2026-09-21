# Bimanual HJ-Reachability Value Function (Piper-PAIR arm-arm avoidance)

SCAFFOLD/SMOKE ONLY. No training has run; no Slurm job has been submitted.
Goal: train a neural HJ value function to eventually replace the analytic
sphere-distance proxy in `experiments/ab_smooth_vs_bangbang/`
(`sim_common.py`'s `min_pair_distance`/`COMBINED_RADIUS` HOCBF-QP barrier).

## Status (this pass): REAL Piper geometry + a trainable DeepReach class

1. **Real FK.** `dynamics.py`'s placeholder serial-chain FK has been replaced
   by `mujoco_fk.py`, a MuJoCo-as-FK-oracle: for a sampled joint config it
   sets qpos on the SAME model as `experiments/ab_smooth_vs_bangbang/sim_common.py`
   (`scenes/scene_bimanual_pickup.xml`), runs `mj_kinematics` + `mj_comPos`,
   and reads the real link `xpos` for the same 3-sphere-per-arm proxy set
   (`left/link3,link5,gripper_base` / `right/...`). **Spot-check**: 6 random
   joint configs scored by both `mujoco_fk.MuJoCoFKOracle` and
   `sim_common.min_pair_distance` gave IDENTICAL distances (to float
   precision) in all 6 cases. `dynamics.py`'s `state_test_range()` now also
   uses the real per-joint Piper ranges (`mujoco_fk.JOINT_RANGES`, from
   `scenes/bimanual_piper_pickup.xml`'s `left/joint1..6` ranges -- confirmed
   the `right/joint1..6` ranges in that XML are identical) instead of a
   generic +/-170deg placeholder.

2. **DeepReach `Dynamics` port: `BimanualArmArmAvoidance`.**
   - **Fork branch**: `bimanual-armarm` in
     `/common/users/ss5772/DeepReach/deepreach` (currently checked out
     there). Changes are **working-tree only, uncommitted** (see "why not
     committed" below) and are **append/single-line-only**:
     - `dynamics/dynamics.py`: appended the new class (~150 lines) at
       end-of-file, after the last existing class. No existing class body
       was touched (verified: `UR5eGraspBoundary`, `Air3D`, `Dubins3D`,
       `MultiVehicleCollision`, `ParameterizedVertDrone2D` all still import
       fine; `run_experiment.py` still compiles).
     - `run_experiment.py`: one line changed --
       `TRAJECTORY_BACKED_DYNAMICS` tuple now includes
       `"BimanualArmArmAvoidance"` (required for `supervised_bootstrap`/
       trajectory-backed dataset construction, same as `UR5eGraspBoundary`).
     - Note: that repo already had substantial *pre-existing, unrelated*
       uncommitted local modifications to both files (someone else's
       in-progress work, `git diff --stat` showed 559/163 +/- lines before I
       touched anything). I did not stash, revert, or commit any of it --
       only added my own hunks on top, and left everything uncommitted so
       as not to fold unrelated work into a commit under this task's name.
   - **Our-repo copy**: `hj_reachability/deepreach_dynamics.py` (near-identical
     `BimanualArmArmAvoidance` + `_MuJoCoBoundaryFn`; importable standalone
     via `importlib`-file-path loading of both `mujoco_fk.py` and, if not
     already loaded, the fork's `dynamics/dynamics.py` -- see that file's
     docstring for a name-collision gotcha this works around: this repo's
     `dynamics.py` (plain module) and the fork's `dynamics/` (package) both
     want the top-level name `dynamics`, so a naive `sys.path.insert` +
     `import dynamics` from both directories at once silently poisons
     `sys.modules['dynamics']` with the wrong one).
   - **∇l (boundary gradient)**: NOT autograd-through-MuJoCo (impossible --
     MuJoCo's FK isn't a torch op). Implemented as the **analytic
     sphere-distance gradient through the geometric Jacobian**
     (`mujoco_fk.MuJoCoFKOracle.closest_pair_info`, using `mj_jac` at the
     current argmin sphere pair: `d(dist)/dq = n^T @ J_point(q)`, the same
     technique `sim_common.py`'s `hdot = n . (v_ego - v_int)` uses for
     velocities, specialized to positions), wired into `torch.autograd` via
     a custom `torch.autograd.Function` (`_MuJoCoBoundaryFn`): forward calls
     the oracle (value + gradient in one MuJoCo call), backward just scales
     the precomputed gradient by the upstream cotangent. **Verified against
     central finite differences**: max abs error ~1e-10 over 8 random
     configs (a finite-difference-only fallback was considered but not
     needed -- the analytic path is both exact and cheaper, ~1 MuJoCo call
     vs ~24 for a 12-dim central-difference gradient).
   - `dsdt`/`hamiltonian`/`optimal_control`/`optimal_disturbance` translate
     directly from `dynamics.py`'s numpy formulas (kinematic single-integrator
     game, state-independent dsdt -- the Hamiltonian itself needs only
     `dvds`, not `boundary_fn`'s gradient; ∇l only matters for a future
     `deepreach_model: diff/exact` residual formulation, which
     `train_config.yaml` does not currently use (`deepreach_model: vanilla`)
     -- implemented anyway per the task's ask, and it is exact, not a
     finite-difference stopgap).

3. **Eval interface: `value_runtime.py`.** Thin, read-only-study port of
   `ur5e_benchmark.hj_eval.hj_inference.TaskBHJInference` +
   `ur5e_benchmark.planning.hj_value_runtime.H10ValueRuntime` +
   `ur5e_benchmark.safety_models.interface.SafetyValueModel` (studied, not
   imported -- no new cross-repo dependency). Provides
   `BimanualHJValueRuntime` (checkpoint/config loading + batched
   `values(flat_states, tau_seconds)`), `BimanualHJInference`
   (`value_from_flat_state`), and `BimanualSafetyValueModel` (duck-types the
   `SafetyValueModel` protocol: `state_dim`, `safe_when_leq_zero`, `values`,
   `describe`). Simplified vs. the ur5e source: no
   variable-horizon/survival-risk-monitor multi-head support (our config is
   single-scalar `deepreach_model: vanilla`). **Sign-convention note**:
   `safe_when_leq_zero=True` here matches what a checkpoint trained against
   `export_dataset.py`'s CSV `target_value` would learn -- the OPPOSITE of
   `dynamics.boundary_fn`'s own raw `l(x)` sign (l>0 safe there). Import-only
   smoke-tested (no checkpoint exists yet to query end-to-end).

## Smoke results

- `python3 export_dataset.py --num-samples 200 --output-root <tmp>`: 0.79s,
  well-formed train/val/test CSVs + manifests, **both classes present**
  (187 safe / 13 unsafe out of 200).
- `python3 export_dataset.py --num-samples 2000 --output-root
  hj_reachability/datasets/bimanual_bootstrap_v0` (matches `train_config.yaml`'s
  `data_root`, so the class-instantiation test below is realistic): 1856
  safe / 144 unsafe.
- In `vlaism-deepreach` (after `pip install mujoco` there -- was missing,
  now `mujoco==3.3.7`, a pure smoke/import dependency, no CUDA/GPU
  involved): `BimanualArmArmAvoidance(data_root=...)` imports, instantiates,
  and a 5-sample batch runs `boundary_fn` forward **and backward**
  (`.sum().backward()` populates real per-row 12-dim gradients),
  `hamiltonian`, `optimal_control`, `optimal_disturbance`, `dsdt` -- all
  correct shapes. Existing dynamics classes (`UR5eGraspBoundary`, `Air3D`,
  `Dubins3D`, `MultiVehicleCollision`, `ParameterizedVertDrone2D`) still
  import; `run_experiment.py` still `py_compile`s. **No training was run,
  no sbatch was submitted.**

## Files here

- `mujoco_fk.py` -- **new**: the real-Piper-geometry FK/Jacobian oracle
  (`MuJoCoFKOracle`), standalone (does not import `sim_common.py` itself, to
  avoid that module's much larger `real_Cartpole/controller_core` QP-solver
  dependency chain for what is here just an FK read).
- `dynamics.py` -- numpy reference "system spec", now backed by
  `mujoco_fk.py` instead of a placeholder FK (see "Status" above).
- `deepreach_dynamics.py` -- **new**: our-repo copy of the torch
  `BimanualArmArmAvoidance` class (see "Status" above).
- `value_runtime.py` -- **new**: value-query runtime scaffold (see "Status").
- `export_dataset.py` -- unchanged logic; now scores REAL FK-based `l(x)`.
- `train_config.yaml` -- `dynamics_class: BimanualArmArmAvoidance` is now a
  real, registered class (fork branch `bimanual-armarm`); `data_root` points
  at the 2k-row smoke dataset generated above (NOT the real 200k one yet).
- `train.sbatch` -- Slurm job script (unchanged; still not run).
- `submit_rlab.sh` -- still hard-exits (`exit 1`) before submitting anything;
  message updated to reflect current status (class exists but only
  smoke-tested; dataset is smoke-scale; branch not merged) -- this task was
  scaffold/smoke only and explicitly must not sbatch.

## Update (2026-09-05): re-pointed to the canonical YOR scene, real 0.185m geometry

`mujoco_fk.py` now loads `scenes/scene_yor_canonical.xml` (the real YOR robot,
arms 0.185m apart, lift-mounted at z~0.672m -- see `scenes/YOR_CANONICAL.md`),
not the old simplified `scene_bimanual_pickup.xml` rig (arms fixed 0.5m apart
on the floor). Changes, all under `hj_reachability/`:

- `mujoco_fk.py`: `SCENE` -> `scenes/scene_yor_canonical.xml`;
  `LEFT_ARM_JOINTS`/`RIGHT_ARM_JOINTS` -> `left_arm_joint1..6`/
  `right_arm_joint1..6`; `EGO_BARRIER_BODIES`/`INT_BARRIER_BODIES` ->
  `left_arm_link3`/`left_arm_link5`/`left_arm_gripper_base` (+ `right_arm_*`).
  `JOINT_RANGES` unchanged numerically -- verified via `mj_name2id`/`jnt_range`
  on the new model that `left_arm_joint1..6` and `right_arm_joint1..6` have
  identical, symmetric ranges (matches `scenes/YOR_CANONICAL.md`'s table).
- `dynamics.py`, `export_dataset.py`, `deepreach_dynamics.py`: docstrings/
  comments updated to reference the new scene/body names; no logic changes
  (both modules already only touch geometry indirectly through `mujoco_fk.py`).
- `train_config.yaml`: `data_root` -> `datasets/bimanual_yor_v1`.

**Sanity check (unsafe fraction, real geometry).** 30,000 uniform random
joint-state samples over the (unchanged) joint ranges: **unsafe fraction
~4.8%** (`l(x) <= 0`), vs. the old rig's ~7%. This is the opposite direction
from the naive "closer mounts -> more overlap" expectation -- likely because
the old floor-mounted rig had both arms pointing toward each other for a
face-to-face pickup task, while the YOR rig's arms are lift-mounted side by
side (0.185m apart in y, same height, same forward-facing orientation), so
despite the smaller mount separation, a uniform sweep of the full 12-dim
joint-angle box produces proportionally fewer configurations where the
Cartesian keypoint clouds actually overlap. The failure region is real and
present (confirmed both classes appear at 200k scale), just not larger than
before -- reported as measured, not adjusted to match the expected direction.

**DeepReach fork implications: none required.** `deepreach_dynamics.py`'s
`BimanualArmArmAvoidance.boundary_fn` calls `mujoco_fk.get_oracle()` at
runtime (via `_MuJoCoBoundaryFn`, loaded by absolute file path from this
repo's `mujoco_fk.py`), not a precomputed/baked geometry -- so it
automatically tracks whatever `SCENE`/body names `mujoco_fk.py` points at.
The dataset CSVs/`dataset_description.json` are used only for `state_mean`/
`state_var`/`state_test_range` (bounds), which are unchanged (same joint
ranges). One stale docstring line (scene path in the class docstring) was
fixed; no behavioral fork edit needed. The actual DeepReach fork checkout at
`/common/users/ss5772/DeepReach/deepreach` (branch `bimanual-armarm`) was not
touched (out of scope: task says write only under `hj_reachability/`) --
its copy of `BimanualArmArmAvoidance` is byte-for-byte the same logic and
will pick up the new geometry the next time this repo's `mujoco_fk.py` is
loaded, no edit needed there either.

## What's missing before a REAL training run

1. **Full-scale dataset -- DONE.** Regenerated at `datasets/bimanual_yor_v1`
   with `--num-samples 200000` against the canonical YOR scene: 24.7s wall
   time, 200,000 rows (139,857 train / 30,063 val / 30,080 test), class
   counts 190,206 safe / 9,794 unsafe (4.90% unsafe, matching the sanity
   check above). The old `datasets/bimanual_bootstrap_v0` (2k rows, old
   scene) is left in place for reference/diffing, not deleted.
2. **Fork branch not merged / uncommitted.** `bimanual-armarm` exists with
   my two edits as *uncommitted working-tree changes* on top of that repo's
   own pre-existing uncommitted work (see "why not committed" above). A real
   training run needs that repo checked out on this branch (or the class
   ported to wherever `main` ends up) with those edits present.
3. **No training run yet, no checkpoint.** `value_runtime.py` is
   import-smoke-tested only -- there is nothing to load and query yet.
4. `submit_rlab.sh` intentionally still refuses to run (`exit 1`).

## Later launch command (once the above is resolved)

```bash
# 1) dataset already regenerated against the canonical YOR scene (see
#    "Update" section above); re-run only if joint ranges/geometry change again:
cd /common/users/ss5772/bimanual/hj_reachability
python3 export_dataset.py --num-samples 200000 \
  --output-root hj_reachability/datasets/bimanual_yor_v1

# 2) ensure /common/users/ss5772/DeepReach/deepreach is checked out on
#    branch bimanual-armarm (git checkout bimanual-armarm) with the
#    BimanualArmArmAvoidance edits present (currently true as working-tree
#    state; commit there first if you want it durable across `git checkout`).

# 3) remove the guard exit in submit_rlab.sh, then:
bash hj_reachability/submit_rlab.sh
# (equivalent to: bash scripts/offload/submit_via_slurm_gateway.sh hj_reachability/train.sbatch)
```

**GPU/time estimate** (unchanged from prior scaffold pass): `train_config.yaml`
mirrors `ur5e_grasp_boundary_bootstrap.yaml` (2-hidden-layer, 128-unit SIREN
MLP, 2000 epochs, batch 1024, single `a4000` GPU) -- budget **~1-3 GPU-hours**
(`train.sbatch` requests `--gres=gpu:a4000:1`, `--time=12:00:00` as a safety
ceiling).

## Blockers

- Dataset is smoke-scale only (2k rows); fork branch changes are uncommitted
  working-tree state on top of pre-existing unrelated dirty files in that
  repo; no checkpoint exists, so `value_runtime.py` is untested end-to-end.
  None of these block further scaffolding, only a real training run.

## Update (2026-09-06): FAITHFUL collision proxy + workspace-oversampled v2 dataset

`mujoco_fk.py` was a hand-copied, STALE duplicate of the OLD 3-point proxy
(`link3`/`link5`/`gripper_base`, `R_EGO=R_INT=0.05`/`D_MIN=0.08`) that
`experiments/ab_smooth_vs_bangbang/COLLISION_FIDELITY.md` found has a **19.8%
false-safe rate**. That doc's fix, implemented in `sim_common.py`, replaces
it with a FAITHFUL ~25-keypoint-per-arm proxy (0.00% false-safe): every major
link (`link0..6`, `gripper_base`, `link7`, `link8`, `iphone_body`) gets 2-3
spheres fit to its REAL fitted-capsule/box geometry, per-pair combined radius
`r_ego+r_int+D_MIN` (`D_MIN=0.03`) instead of one fixed scalar.

1. **`mujoco_fk.py` upgraded** to the SAME proxy (`LINK_NAMES`/
   `_body_keypoints`/`barrier_keypoints`/`D_MIN`, replicated verbatim from
   `sim_common.py`, plus its in-memory contype/conaffinity
   `_patch_real_collision_geoms` fix at `load_model()`). `closest_pair_info`'s
   analytic gradient now queries the Jacobian at the keypoint's WORLD OFFSET
   point (`_point_jac_at`), not the body origin (`d(dist)/dq = n^T @
   J_point(q, offset)`), since faithful-proxy keypoints are mostly not at
   body origins. **Spot-check**: 10 random joint configs, `mujoco_fk.
   MuJoCoFKOracle.boundary_fn(q_ego, q_int)` vs. `sim_common.min_pair_
   distance(model, data)`'s `h` on the SAME scene -- **0.00e+00 diff in all
   10** (exact float match, better than the ~1e-8 threshold checked).
   `dynamics.py`'s `boundary_fn`/`min_pair_distance` and
   `deepreach_dynamics.py`'s `_MuJoCoBoundaryFn` were updated to delegate to
   `oracle.boundary_fn`/`info.h` directly (the old
   `min_pair_distance(state) - combined_radius` form assumed one fixed
   scalar radius, which the faithful per-pair-radius proxy no longer has).

2. **Workspace oversampling.** Uniform sampling over the full 12-dim joint
   box gave only ~4.9% unsafe (see the 2026-09-05 update below) -- most
   configs are nowhere near the shared workspace. Added
   `BimanualPairAvoidance.sample_states_mixed` (`dynamics.py`): mixes
   `task_region_frac` (default **0.5**) fraction "task-region" samples --
   `q_ego = LEFT_GRASP_Q + N(0, 0.25rad)`, `q_int = interp(RIGHT_HOME_CTRL,
   RIGHT_INTRUDE_CTRL, Uniform(0,1)) + N(0, 0.25rad)`, i.e. both EEs near the
   shared workspace point (`(0.42,-0.21)`, ~0.55m height) -- with the
   remaining fraction pure-uniform. `LEFT_GRASP_Q`/`RIGHT_HOME_CTRL`/
   `RIGHT_INTRUDE_CTRL` are copied verbatim from `sim_common.py` (read-only
   import target per task scope, not edited). `export_dataset.py` gained
   `--task-region-frac` (default 0.5). **Result: 27,184/200,000 = 13.59%
   unsafe**, ~2.8x the old 4.90% uniform-only rate.

3. **Regenerated 200k -> `datasets/bimanual_yor_v2`.**
   `python3 export_dataset.py --num-samples 200000 --output-root
   datasets/bimanual_yor_v2 --task-region-frac 0.5`: **35m 19.7s wall-clock**
   (2116s CPU, single core) -- vs. v1's 24.7s, consistent with ~625 pairs/
   config (~70x the old 3-sphere proxy's 9 pairs) plus the doubled per-row
   oracle search (`min_pair_distance` + `boundary_fn` each re-run the O(625)
   closest-pair search; a follow-up could cache one combined call to roughly
   halve this). 200,000 rows (139,857 train / 30,063 val / 30,080 test --
   same split membership as v1, deterministic sha1-of-sample_id with the same
   seed=7), class counts **172,816 safe / 27,184 unsafe (13.59% unsafe)**.
   `train_config.yaml`'s `data_root` -> `datasets/bimanual_yor_v2`. The old
   `datasets/bimanual_yor_v1` (OLD proxy, uniform-only) is renamed
   `datasets/bimanual_yor_v1_STALE` (not deleted -- this sandbox blocks
   `rm -rf`) with a `STALE.md` marker; `datasets/bimanual_bootstrap_v0`
   (2k-row pre-YOR-scene smoke set) is untouched/irrelevant.

4. **DeepReach class re-confirmed against v2.** In conda env
   `vlaism-deepreach`: `BimanualArmArmAvoidance(data_root=".../
   datasets/bimanual_yor_v2")` imports and instantiates; a 5-sample batch
   runs `boundary_fn` forward (values match the faithful-proxy scale, e.g.
   `0.1347...` recurring, matching the mujoco_fk spot-check above) **and**
   backward (real per-row 12-dim gradients via the offset-aware Jacobian);
   `hamiltonian` runs. Confirms the class automatically inherited the
   faithful proxy purely by re-reading `mujoco_fk.py` at runtime, no
   DeepReach-repo-side edit needed for that part. **Caveat**: the actual
   external fork checkout (`/common/users/ss5772/DeepReach/deepreach`,
   branch `bimanual-armarm`, uncommitted working-tree copy of
   `BimanualArmArmAvoidance`) still has the OLD `info.dist -
   mujoco_fk.COMBINED_RADIUS` line in its own `dynamics/dynamics.py` copy --
   this task's write scope is `hj_reachability/` only, so that external copy
   was not touched. Before a real training run, hand-apply the same one-line
   fix there (`l_vals[i] = info.dist - mujoco_fk.COMBINED_RADIUS` ->
   `l_vals[i] = info.h`), mirroring `hj_reachability/deepreach_dynamics.py`'s
   canonical copy (kept in sync by hand per that file's own docstring).

5. **No training run, no sbatch.** `submit_rlab.sh` still `exit 1`s before
   submitting (message text updated to reflect the faithful-proxy v2
   dataset, guard behavior unchanged).

### Blockers / caveats (this pass)

- The external DeepReach fork's own `dynamics/dynamics.py` copy needs the
  same one-line `info.h` fix applied by hand (see item 4) before a real
  training run -- not done here (out of this task's write scope).
- Export time (35m for 200k) is dominated by the ~625-pair-per-config search
  run TWICE per row (`min_pair_distance` + `boundary_fn`); caching a single
  combined oracle call would roughly halve it if re-exporting at larger
  scale becomes a bottleneck.

### Rollback (this pass only)

```bash
cd /common/users/ss5772/bimanual/hj_reachability
mv datasets/bimanual_yor_v1_STALE datasets/bimanual_yor_v1   # undo the stale-rename
# then revert mujoco_fk.py / dynamics.py / export_dataset.py / deepreach_dynamics.py /
# train_config.yaml / submit_rlab.sh to their pre-2026-09-06 versions (git diff
# against this commit), and remove datasets/bimanual_yor_v2/.
```

## Rollback

```bash
# this repo
rm -rf /common/users/ss5772/bimanual/hj_reachability

# DeepReach fork: no commits were made on bimanual-armarm (it points at the
# same commit as main), so switching branches will NOT discard the edits --
# they're plain working-tree changes and follow you to whichever branch is
# checked out. To remove only this task's edits (leaving the repo's own
# pre-existing unrelated uncommitted work untouched):
#   - dynamics/dynamics.py: delete the appended block from the
#     "BimanualArmArmAvoidance -- added on branch `bimanual-armarm`" banner
#     comment to end-of-file.
#   - run_experiment.py: drop ", "BimanualArmArmAvoidance"" from the
#     TRAJECTORY_BACKED_DYNAMICS tuple.
cd /common/users/ss5772/DeepReach/deepreach
git branch -D bimanual-armarm   # just drops the branch pointer
```
