# bimanual/world_model

DreamerV3 RSSM world model (encoder + latent transition + decoder), set up
for OFFLINE training on our `collection/` episodes. No online RL.
**INFRA + tiny smoke only** -- no full training, no sbatch was submitted.
Now includes: a real training driver (`train_wm.py`, checkpointing +
resume + tensorboard), a WM eval script (`eval_wm.py`, reconstruction +
open-loop imagined rollout + filmstrip), and the failure-monitor head
(`monitor_head.py`, latent-only MLP -> P(spill)/P(collision)). See section 7
below for all three; sections 1-6 are the original setup writeup.

## 1. Implementation chosen

- **Repo:** [`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch)
  (PyTorch port), cloned to `/common/users/ss5772/dreamerv3` (outside this
  repo, per task instructions).
- **Env:** reused `mujoco_ur5e` (`torch==2.12.1+cu130`, `cuda.is_available()==True`).
  **No new conda env was created.** `models.py`/`networks.py`/`tools.py` --
  the only three files we need for a world-model-only pipeline -- import
  nothing beyond `numpy`, `torch`, and `torch.utils.tensorboard` (already
  present in `mujoco_ur5e`); verified with a direct import. The repo's own
  `requirements.txt` (`torch==2.4.1`, `gym==0.22.0`, `dm_control`,
  `mujoco==2.3.5`, `einops`, `ruamel.yaml`, ...) is for its *env suites*
  (`dmc`/`atari`/`crafter`/...) via `dreamer.py`'s `make_env()`, which we
  never call -- see section 4. Nothing was installed into `mujoco_ur5e`;
  it is unmodified.
- **Why this port over danijar's JAX original:** torch-native, so it shares
  a runtime/CUDA stack with the rest of this repo (`mujoco_ur5e`) instead of
  needing a second JAX/TF install; the repo's README flags it as
  superseded by a faster successor (`r2dreamer`) but `dreamerv3-torch`'s
  `WorldModel`/`RSSM`/`MultiEncoder`/`MultiDecoder` code is unchanged
  DreamerV3 math and is what most forks/papers cite -- fine for our purpose
  (we only use the world model, not its actor/critic/env loop).

## 2. Verify-run result

Confirmed: `models.WorldModel._train(batch)` (full RSSM step -- encode,
`dynamics.observe`, KL loss, decoder/reward/cont heads, backward, optimizer
step) runs on this node's GPU (2080 Ti) with zero new dependencies. This was
verified directly via the smoke test in section 5 (real bimanual data, not
the repo's own toy config) -- see loss numbers there.

## 3. Data adapter

`world_model/dataset_adapter.py` converts `collection/data/<run>/episode_*/`
(schema: `collection/episode.py`) into dreamerv3-torch's own per-episode
flat-npz replay format (written via its vendored `tools.save_episodes`, so
it's byte-for-byte what `tools.load_episodes`/`sample_episodes` expect):

| dreamerv3-torch key | shape | from collection/ |
|---|---|---|
| `image` | `(T,64,64,6)` u8 | `concat([rgb_head, rgb_wrist_right], axis=-1)` |
| `proprio` | `(T,26)` f32 | `concat([left_q(6), left_qd(6), right_q(6), right_qd(6), left_gripper_state(1), right_gripper_state(1)])` |
| `object_pose` | `(T,7)` f32 | `box_pose` -- LABEL only, never fed to the WM (see below) |
| `action` | `(T,16)` f32 | `concat([left_action(8), right_action(8)])` |
| `reward` | `(T,)` f32 | zeros (no task reward defined yet -- see section 7's monitor head) |
| `discount` | `(T,)` f32 | ones |
| `is_first` | `(T,)` bool | `True` at t=0 only |
| `is_terminal` | `(T,)` bool | `collision[t] \| dropped[t]` (real per-step signal) |

**UPDATE (section 7's task):** `proprio` is now 26-D and `box_pose` (object
pose) has been moved out of it into a separate `object_pose` LABEL field
(never an encoder/decoder input) -- see `docs/design/state_spaces.md`'s
sim2real line and `dataset_adapter.py`'s own "OBJECT-POSE-LEAK FIX" comment.
`image` is still the two wrist/head cams concatenated as of this writing;
if it switches to ZED it stays 6-channel or changes shape entirely, and
**`train_wm.py`/`eval_wm.py`/`monitor_head.py` do not hardcode either
number** -- they read `image`/`proprio` shapes straight off the converted
data (`infer_obs_shapes()` in `train_wm.py`), so a future adapter change
needs no edits here.

**Why one 6-channel `image` key instead of two separate cam keys:**
dreamerv3-torch's `WorldModel.preprocess()` hardcodes `obs["image"] =
obs["image"] / 255.0` (a single literal key, not general multi-cam) even
though its `MultiEncoder`/`MultiDecoder` do support multiple regex-matched
`cnn_keys` (channel-concatenated internally). Channel-concatenating our two
cams into one `image` array upstream, in the adapter, lets us hit that
existing code path with **zero edits to the vendored repo**. A future
monitor head can split channels `[0:3]`/`[3:6]` back into head/wrist views.

Run it: `python -m world_model.dataset_adapter --collection-dir <dir> --out-dir <dir>`
(defaults point at `collection/data/smoke` -> `world_model/data/dreamer_fmt/smoke`).

`world_model/train_wm_config.yaml` holds the WM/RSSM hyperparameters
(dreamerv3-torch defaults, unchanged) plus our obs/action shapes and
offline data paths, for a real run; `smoke_train_wm.py` loads it and
overrides `batch_size`/`batch_length` down for the tiny smoke.

## 4. Offline-training support verdict

**Partial, with a documented workaround -- no repo edits needed for
world-model-only training.**

- `dreamer.py`'s `main()` *does* have first-class offline-replay support:
  `config.offline_traindir` -> `tools.load_episodes(directory)` skips the
  online `prefill`/`simulate` step entirely when set.
- **But** `main()` still unconditionally calls `make_env()` to build
  `train_envs[0]` purely to read `.observation_space`/`.action_space` for
  `Dreamer.__init__`, and it also runs online *evaluation* rollouts every
  `eval_every` steps (`tools.simulate(eval_policy, eval_envs, ...)`) --
  both require a real steppable env, which we don't have (and don't want:
  the task is offline WM-only, actor/critic and eval are out of scope).
- **Workaround (implemented, no repo surgery):** don't call `dreamer.py`'s
  `main()`/`Dreamer` agent at all. Construct `models.WorldModel` directly
  with a hand-built `gymnasium.spaces.Dict` obs_space / `Box` act_space
  (shapes known from the adapter -- section 3), and drive it with
  `tools.load_episodes` + `tools.sample_episodes` + `tools.from_generator`
  (also unchanged, vendored). This is exactly `smoke_train_wm.py`.
- **If a full `Dreamer` agent (with actor/critic, imagination rollouts, TB
  logging via `dreamer.py`) is wanted later:** that *would* need surgery --
  either (a) add a tiny dummy `gymnasium.Env` subclass exposing our fixed
  spaces and a new `elif suite == "bimanual":` branch in `make_env()`
  (~10 lines in the vendored repo), or (b) monkeypatch `make_env` from our
  own launcher script without touching the vendored file. Not needed for
  the world-model-only scope of this task.

## 5. Smoke result

```
CUDA_VISIBLE_DEVICES=3 <mujoco_ur5e>/bin/python world_model/smoke_train_wm.py
```
Converted the 3 real `collection/data/smoke/episode_00000{0,1,2}` (60 steps
each), built `WorldModel` (26.3M params) on `cuda:0`, ran 3 `_train()` steps
with `batch_size=2, batch_length=16` (seconds total, single GPU, no sbatch):

```
step 0: model_loss=4060.98 image_loss=4025.66 kl=7.16
step 1: model_loss=3801.94 image_loss=3780.02 kl=7.79
step 2: model_loss=3613.70 image_loss=3588.09 kl=9.45
```
Loss decreasing step-to-step on 3 real steps is expected (random init,
tiny batch) not evidence of convergence -- this is a "does it run and
compute a gradient" smoke, not a training result. Only warnings: cosmetic
`torch.cuda.amp.autocast`/`GradScaler` `FutureWarning`s (deprecated call
form in the vendored code, functional under torch 2.12; harmless).

## 6. Later real-training command + estimate

Superseded by section 7 -- `train_wm.py` now exists and was smoke-tested.

## 7. train_wm.py / eval_wm.py / monitor_head.py (this task)

### 7.1 train_wm.py -- real offline WM training driver

Same code path as `smoke_train_wm.py` (build `models.WorldModel` directly,
no `dreamer.py` main()/make_env()/Dreamer agent -- see section 4), but a
real loop over `config.train_steps` instead of 3 smoke steps, plus:

- **Shape inference, never hardcoded:** `infer_obs_shapes()` reads
  `image`/`proprio`/`action` shapes straight off one loaded episode's actual
  arrays (`episodes[k]["image"].shape[1:]` etc.), *not* off
  `train_wm_config.yaml`'s `image_channels`/`proprio_dim` fields (those are
  checked-not-trusted documentation/defaults -- a log line prints both and
  says data wins). Verified live: the smoke run below printed
  `image=(64, 64, 6) proprio_dim=26 num_actions=16` purely from the
  converted npz, matching but not sourced from the yaml.
- **Checkpointing:** `latest.pt` every `--ckpt-every` steps (resumed
  automatically on the next invocation unless `--no-resume`), `best.pt` on
  eval-loss improvement. Mirrors `dreamer.py`'s own
  `agent_state_dict`/`optims_state_dict` pattern (`tools.recursively_collect
  /load_optim_state_dict`) but for a bare `WorldModel`, not the full agent.
- **Tensorboard + jsonl logging:** via the vendored `tools.Logger` (the same
  class `dreamer.py` itself uses) -- `train/{image,proprio,reward,cont}_loss`,
  `train/kl`, `eval/*` (mirrored), `fps`.
- **Eval hook:** every `--eval-every` steps, one no-grad forward pass
  (`eval_forward()`) on a held-out episode split (`--eval-holdout N`, last N
  episodes by sorted id -- **NOT statistically rigorous with tiny N**, see
  caveats).

CLI: `python -m world_model.train_wm --config <yaml> --collection-dir <dir>
--dreamer-fmt-dir <dir> --logdir <dir> --train-steps N --batch-size B
--batch-length L --eval-every N --ckpt-every N --eval-holdout N [--no-resume]
[--skip-convert]`.

### 7.2 eval_wm.py -- WM eval on a held-out split

- (a) **Reconstruction error:** teacher-forced encode -> `dynamics.observe`
  -> decoder `.mode()`, plain pixel/proprio MSE (+ KL), no grad.
- (b) **Open-loop imagined rollout:** encode a `--prefix-len` prefix into
  posterior states, then `dynamics.imagine_with_action` for `--rollout-len`
  further steps using only the true action sequence (no more observations) --
  generalizes `models.WorldModel.video_pred`'s logic (same
  encode-prefix/imagine-suffix pattern) to also decode+compare `proprio`, not
  just `image`, and to arbitrary lengths.
- (c) **Filmstrip PNG** (`artifacts/wm_eval_<tag>.png`): rows = true /
  reconstructed-then-imagined / abs-error, columns = timesteps (labeled
  "obs" vs "imagined"). `_channels_to_panels()` renders any channel count
  generically (3ch shown directly, multiples of 3 tiled side-by-side as
  separate cams, anything else falls back to grayscale) -- **no hardcoded
  6-channel assumption**, so it keeps working if the image switches to a
  3-channel ZED view.

CLI: `python -m world_model.eval_wm --dreamer-fmt-dir <dir> --checkpoint
<path> --prefix-len N --rollout-len N --eval-holdout N --tag <name>`.

### 7.3 monitor_head.py -- the failure-monitor head

Small MLP (`latent_dim -> 64 -> 32 -> 2` logits) on the WM latent
`z_t = dynamics.get_feat(post)` from a **frozen** WorldModel checkpoint ->
`P(spill)`, `P(collision)`. **Input is the latent only** -- verified by
reading the code: `compute_latents_and_labels()` runs the WM encoder/RSSM
under `@torch.no_grad()` and the labels (`collision`, `spilled_count`) are
read straight from the **raw** `collection/<run>/episode_*/episode.npz` +
`metadata.json` (`_load_raw_labels()`), never routed through the WM's own
obs dict; `object_pose`/labels are structurally never in the tensor the
classifier's `forward()` sees.

**Label sourcing (current schema limitation, documented in the module
docstring):** `collision[t]` is a real per-step signal
(`collection/episode.py`'s `STEP_ARRAY_KEYS`). There is **no per-step spill
signal yet** -- only episode-level `spilled_count` in `metadata.json` (always
0 today per `episode.py`'s own docstring: no cup/balls scene wired up). The
monitor head broadcasts `spilled_count > 0` as a constant per-episode proxy
across all of that episode's timesteps; this is a placeholder for when a
real per-step spill flag lands.

**Metrics are computed with plain numpy** (`sklearn` is not installed in
`mujoco_ur5e`, and no new dependency was added): manual
`precision_recall()` (threshold 0.5), `roc_curve()` (threshold sweep +
trapezoidal AUC, returns `None`/"undefined" for a single-class eval split
rather than a misleading number), `calibration_curve()` (10-bin reliability
diagram). Saved to `artifacts/monitor_<tag>.png` (2x2: ROC + calibration for
each of spill/collision) and `artifacts/monitor_<tag>_metrics.json`.

CLI: `python -m world_model.monitor_head --dreamer-fmt-dir <dir>
--collection-dir <dir> --checkpoint <wm_checkpoint> --epochs N
--eval-holdout N --tag <name>`.

### 7.4 Smoke results (all on `collection/data/smoke`, GPU 3, seconds total)

```
CUDA_VISIBLE_DEVICES=3 python -m world_model.train_wm \
  --collection-dir collection/data/smoke --dreamer-fmt-dir world_model/data/dreamer_fmt/smoke \
  --logdir world_model/logdir/smoke_run --train-steps 3 --batch-size 2 --batch-length 16 \
  --eval-every 2 --ckpt-every 2 --eval-holdout 1
```
-> inferred `image=(64,64,6) proprio_dim=26 num_actions=16` from data (yaml
values matched but were not the source); 2 train / 1 eval episode split;
`step 1: train_model_loss=3753.06`, `step 2: train_model_loss=3504.88
eval_model_loss=3353.96` (new best -> `best.pt`), `step 3:
train_model_loss=3335.85`; `latest.pt`/`best.pt`/tensorboard events all
written. **Resume verified separately:** re-invoked with `--train-steps 5
--skip-convert` (no `--no-resume`) -> `resumed ... at step=3
best_eval_loss=3353.9607`, continued to step 5 with loss still decreasing
(`2966.58`). Both checkpoint dirs were deleted after verification (600MB+ of
disposable smoke checkpoints; not gitignored under `world_model/logdir/`,
unlike `world_model/data/` and `artifacts/`).

```
CUDA_VISIBLE_DEVICES=3 python -m world_model.eval_wm \
  --dreamer-fmt-dir world_model/data/dreamer_fmt/smoke --checkpoint world_model/logdir/smoke_run/latest.pt \
  --prefix-len 3 --rollout-len 5 --eval-holdout 1 --tag smoke --skip-convert
```
-> `(a) reconstruction: {'image_mse': 0.118, 'proprio_mse': 1.293, 'kl':
11.65}`; `(b) open-loop rollout: prefix=3 rollout=5 image_mse_prefix=0.122
image_mse_imagined=0.114 proprio_mse_imagined=1.337`; `(c) filmstrip saved
to artifacts/wm_eval_smoke.png` (visually confirmed: 8 columns, true row
shows the real scene, recon/imagined rows are still near-random noise after
only 5 train steps -- expected, not a bug: this is a "does it run and
produce a filmstrip" smoke, not a converged model).

```
CUDA_VISIBLE_DEVICES=3 python -m world_model.monitor_head \
  --dreamer-fmt-dir world_model/data/dreamer_fmt/smoke --collection-dir collection/data/smoke \
  --checkpoint world_model/logdir/smoke_run/latest.pt --epochs 20 --eval-holdout 1 --tag smoke --skip-convert
```
-> `latent_dim=1536` (`dyn_stoch*dyn_discrete + dyn_deter` = 32*32+512);
120 train / 60 eval timesteps; train BCE loss `0.719 -> 0.029` over 20
epochs (classifier fits fine); eval `precision=nan recall=nan
roc_auc=undefined (single class)` for **both** labels -- because all 3
smoke episodes have `collision=False, spilled_count=0` (verified directly
from their `metadata.json`; `collection/data/cup_verify`, collected
elsewhere, has real positive spill/collision episodes and is the right
dataset for a *meaningful* monitor-head run, but the task specified smoke on
`collection/data/smoke`, which is degenerate by construction). Plot
(`artifacts/monitor_smoke.png`) and metrics json
(`artifacts/monitor_smoke_metrics.json`) both saved and visually confirmed --
the plot honestly renders "AUC undefined" rather than a fake number.

**Confirmed: no hardcoded 6ch/33d/26d anywhere in `train_wm.py`/
`eval_wm.py`/`monitor_head.py`.** The only place fixed numbers appear is
`train_wm_config.yaml`'s `image_channels`/`proprio_dim` fields, which are
never read by the shape-determining code path (`infer_obs_shapes()` reads
the data; the yaml fields are print-compared, not used).

### 7.5 Real-run commands + rough estimate

```
python -m world_model.train_wm --collection-dir collection/data/run_001 \
  --dreamer-fmt-dir world_model/data/dreamer_fmt/train_001 \
  --logdir world_model/logdir/wm_run_001 --train-steps 2000 \
  --eval-every 50 --ckpt-every 50 --eval-holdout 5
# (also wired into world_model/wm_train.sbatch / submit_wm_rlab.sh -- NOT submitted)

python -m world_model.eval_wm --dreamer-fmt-dir world_model/data/dreamer_fmt/train_001 \
  --checkpoint world_model/logdir/wm_run_001/best.pt --prefix-len 5 --rollout-len 15 \
  --eval-holdout 5 --tag run_001

python -m world_model.monitor_head --dreamer-fmt-dir world_model/data/dreamer_fmt/train_001 \
  --collection-dir collection/data/run_001 \
  --checkpoint world_model/logdir/wm_run_001/best.pt --epochs 200 --eval-holdout 5 --tag run_001
```
Estimate unchanged from section 6's original analysis: ~2000 WM-only steps
at batch 16x64, 64x64x6 images, 26M-param model on a single 2080 Ti --
likely tens of minutes to ~1-2 hours (CPU-bound npz decompression vs GPU
compute dependent); a real "how many episodes / how long" number needs a run
with `collection/submit_collect_rlab.sh` (3000-step episodes so
collision/drop -- and eventually spill -- signals actually fire) first. The
monitor head itself is cheap (a 2-layer MLP on precomputed latents); its
cost is dominated by the one-time WM forward pass over all episodes to
extract latents, which is O(dataset size) at inference speed, not training
speed.

### 7.6 What's needed before a real (non-degenerate) monitor-head run

- **Real cup+ZED data with 26-D proprio:** `collection/data/smoke` has zero
  positive spill/collision labels by construction (a 60-step reach/grasp
  demo, no cup scene). A real run needs episodes long enough and varied
  enough (collisions induced, a cup scene with real spill physics) that both
  labels have positive examples in both train and eval splits --
  `collection/data/cup_verify` (spill/collision-positive, already collected
  elsewhere) or a fresh `collection/submit_collect_rlab.sh` run are
  candidates once the WM/adapter's ZED-image / 26-D-proprio switch (in
  progress elsewhere, see section 3's UPDATE note) is finalized and data
  collected against it.
- **A true per-step spill signal** in `collection/episode.py`'s schema (today
  only episode-level `spilled_count` exists) so `monitor_head.py`'s spill
  label stops being a constant-per-episode proxy.
- **A real held-out split:** `--eval-holdout` currently just takes the last N
  episodes by sorted id; a real run should use enough episodes that this is
  a statistically meaningful disjoint split, not 1-of-3.

## Blockers / caveats

- No hard dependency conflicts: WM-only code path needs nothing beyond what
  `mujoco_ur5e` already has.
- Offline surgery needed **only** if a full actor/critic `Dreamer` agent
  (not just the world model) is wanted later -- see section 4.
- `reward`/`discount` are placeholder zeros/ones (no task reward exists yet
  -- intentionally deferred to the future monitor head); `reward_head`/
  `cont_head` still train on this smoke data but on an uninformative
  target, which is fine for a WM-only setup/smoke.
- Resolution/obs-space: fixed at 64x64 (matches `collection/collect.py`'s
  `--render-size 64` used for the smoke episodes); a run collected at a
  different `--render-size` needs `size` in `train_wm_config.yaml` updated
  to match (`image_channels` is documentation only -- see section 3).
- `wm_train.sbatch`/`submit_wm_rlab.sh` (section 7.5) are scaffolds calling
  the now-real `train_wm.py` -- **neither was submitted/executed** by this
  task (no sbatch, per task instructions).
- Monitor-head smoke metrics are degenerate by construction (section 7.4) --
  not a bug, but not evidence the classifier works on real labels either.

## Rollback

- Delete the clone: `rm -rf /common/users/ss5772/dreamerv3`
- No new conda env was created, so there is nothing to `conda env remove`.
- No new pip/conda dependency was added anywhere in this task (monitor-head
  metrics are plain numpy, not sklearn -- see 7.3).
- Delete this directory's converted-data output (gitignored already via the
  repo's blanket `data/` rule): `rm -rf world_model/data/`
- Delete training logdirs/checkpoints (**not** gitignored -- can be large,
  e.g. a smoke run's `latest.pt`+`best.pt` was ~600MB for a 26M-param model):
  `rm -rf world_model/logdir/`
- Delete eval/monitor artifacts (gitignored via the repo's blanket
  `artifacts/` rule): `rm -f artifacts/wm_eval_*.png artifacts/monitor_*.png
  artifacts/monitor_*_metrics.json`
- To fully remove this task's new files: `rm -f world_model/{train_wm,eval_wm,monitor_head}.py
  world_model/wm_train.sbatch` and revert `world_model/submit_wm_rlab.sh` /
  `world_model/README.md` to their pre-task state (git has no prior commit
  for this repo yet, so there is no `git checkout` target -- see gitStatus).
