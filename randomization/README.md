# Domain randomization (WM/HJ generalization now, sim2real later)

SCAFFOLD + SMOKE. Module: `randomization/domain_randomization.py`.
Smoke test: `randomization/smoke_test.py` -> `artifacts/dr_samples.png`.

This module is **read-only** with respect to the rest of the repo: it never
edits scene XML files or `collection/`. It only mutates in-memory
`mujoco.MjSpec`/`mujoco.MjModel` objects handed to it by the caller.

## API

```python
from randomization.domain_randomization import RandomizationConfig, randomize, apply_sensor_noise
```

`randomize(spec_or_model, rng, config=None)` dispatches on type:
- `mujoco.MjSpec` -> **structural** edits only (currently: `scene`
  family's distractor objects -- must happen pre-compile). Caller must then
  call `spec.compile()`.
- `mujoco.MjModel` -> **numeric** in-place edits on the compiled model's
  flat arrays (visual, dynamics, and scene-object pose/count via
  `model.qpos0` + `model.body_pos`). Caller must reset+forward `MjData`
  afterwards (`qpos0` only takes effect on reset).

`rng` is a stdlib `random.Random`, seeded per-episode by the caller (e.g.
`random.Random(episode_seed)`), NOT numpy. Each family internally spawns an
independent child RNG from one draw of `rng`, so toggling one family's
`enabled` flag does not perturb another family's draw sequence (same
seeded-family idea as `collection/perturbations.py`, generalized).

`apply_sensor_noise(rgb, depth, rng, config=None)` is a **post-render**
hook (ZED depth/RGB noise stand-in) -- call it after rendering, not inside
`randomize()`, since no image exists at `randomize()`-call time.

## Families + ranges implemented

**visual** (`VisualConfig`):
- light pos jitter: +/-0.4 m per axis; intensity scale 0.5-1.6x; color
  jitter +/-0.15/channel.
- material color jitter +/-0.20/channel (covers table/cup/ball materials +
  bare-rgba geoms), reflectance resampled in [0, 0.4].
- camera pos jitter +/-0.01 m, angle jitter +/-2 deg (random axis), applied
  to all cameras in the compiled model (incl. `left/right_arm_egocentric`,
  the ZED-standin wrist cams).
- ZED depth/RGB noise hooks: `apply_sensor_noise()` -- gaussian RGB noise
  (std 6/255), gaussian depth noise (std 3mm) + 1% pixel dropout (missing
  return simulation).

**scene** (`SceneConfig`):
- cup xy radius 0.03 m + yaw +/-15 deg, within the shared reachable region
  documented in `scenes/YOR_CANONICAL.md`.
- balls: independent xy jitter +/-4mm each; "count" is randomized by
  parking a random subset (range 4-8 of the scene's fixed 8 balls) far
  below the floor (z=-1m and below, spread out) rather than deleting
  bodies -- deleting/adding ball bodies is a structural spec-level edit and
  the scene's 8 ball bodies are fixed in `scenes/scene_yor_cup.xml`; true
  count-varying would need spec-level geom deletion (documented, not
  implemented -- see Blockers).
- table pos jitter +/-0.015 m xy, applied to the table body AND added to
  the cup/ball offset so they move together and nothing falls off the
  table edge.
- distractor objects: 0-2 small random-colored box/sphere/cylinder bodies
  scattered on the table around the cup (structural, MjSpec-only).

**dynamics** (`DynamicsConfig`), all multiplicative:
- geom friction scale 0.7-1.3x (all geoms, all 3 friction cols).
- body mass scale 0.8-1.2x (skips massless/world bodies).
- dof damping scale 0.8-1.2x.
- actuator gain (kp) scale 0.85-1.15x, applied in lockstep to
  `gainprm[:,0]` and `biasprm[:,1]` to preserve MuJoCo's standard position-
  actuator identity `biasprm[1] == -gainprm[0]`.

## Smoke test results

Ran `randomization/smoke_test.py` against `scenes/scene_yor_cup.xml`
(loaded read-only) across seeds 0-5, rendered each via `front_camera`:

- **Montage rendered**: yes, `artifacts/dr_samples.png` (2x3 grid).
- **Variety confirmed**: yes -- visibly different background/skybox tint,
  table material color, and robot material tint per seed. Pairwise
  mean-abs-pixel-diff across the 6 renders: min=3.81, max=13.39, mean=7.28
  (all > the ~1.0 noise floor).
- **All seeds load valid**: yes, 6/6 PASS -- each randomized+compiled model
  has `left_arm_link0`, `right_arm_link0`, `table`, `cup` bodies present,
  `nu==16` actuators, `ncam>=4` cameras.

## Call-site snippet

### Collection loop (per episode)

```python
import random
import mujoco
from randomization.domain_randomization import RandomizationConfig, randomize

config = RandomizationConfig()  # or a task-tuned override

def build_episode_model(scene_path: str, episode_seed: int):
    rng = random.Random(episode_seed)
    spec = mujoco.MjSpec.from_file(scene_path)   # fresh spec per episode
    randomize(spec, rng, config)                 # structural: distractors
    model = spec.compile()
    randomize(model, rng, config)                # numeric: visual/dynamics/scene
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data
```

Compose with `collection/perturbations.py`'s existing
`build_perturbation_spec(family, level, seed)` calls -- that module seeds
demonstrator behavior / start-pose perturbation; this module seeds the
environment appearance/dynamics/layout. Use two independent seeds (or two
children of one episode seed) so toggling one doesn't reshuffle the other.

### Eval harness (per rollout, for WM/HJ generalization sweeps)

```python
for episode_idx in range(n_eval_episodes):
    model, data = build_episode_model(SCENE_PATH, episode_seed=eval_seed_base + episode_idx)
    # ... run policy/HJ-value rollout against (model, data) ...
    rgb = renderer.render()  # after mujoco.Renderer(model).update_scene(data, camera=...)
    rgb_noisy, _ = apply_sensor_noise(rgb, None, random.Random(eval_seed_base + episode_idx), config)
```

To hold one family fixed while sweeping another (e.g. sweep `dynamics`
only, freeze `visual`/`scene`), set `enabled=False` on the frozen
families' sub-configs -- the other families' RNG draws are unaffected.

## Blockers

- True ball-count variation (adding/removing ball bodies, not just
  parking) needs a structural MjSpec edit (delete/re-add ball geoms before
  compile) -- not implemented; current scaffold parks a random subset of
  the fixed 8 balls out of range instead, which is enough for occlusion/
  count-signal variety but not for changing sim cost (all 8 free joints
  still integrate).
- `scene` family's pose randomization (`_randomize_scene_poses`) is
  keyed to `scenes/scene_yor_cup.xml`'s specific body/joint names
  (`cup_free`, `ball_0_free`..`ball_7_free`, body `table`); it silently
  no-ops on other scenes (e.g. the older `scene_bimanual_cup.xml`) rather
  than erroring. `visual`/`dynamics` families are scene-agnostic (operate
  on whatever `nlight`/`nmat`/`ngeom`/`nbody`/`nu` the compiled model has).

## Rollback

Delete `randomization/` and `artifacts/dr_samples.png`. Nothing under
`scenes/`, `collection/`, or elsewhere was modified.
