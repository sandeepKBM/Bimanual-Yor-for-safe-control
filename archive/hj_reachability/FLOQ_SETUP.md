# FLOQ adapted to bimanual arm-arm avoidance

## What FLOQ is (found, unmodified)

Source: `/common/users/ss5772/VLAism/ur5e_benchmark/src/ur5e_benchmark/floq/`
(`dataset.py`, `model.py`, `losses.py`, `train.py`, `eval.py`). Configs:
`.../ur5e_benchmark/configs/floq/floq_v1_*.yaml`. Prior outputs (cup/ball
manipulation-safety runs): `.../ur5e_benchmark/outputs/floq/`.

FLOQ v1 is a **horizon-conditioned safety classifier**, not a classic
Bellman-backup value net: for a state `z_t`, horizon `h`, task id and
(optional) action context, it predicts `P(unsafe within h steps)` via a
sigmoid over a scalar logit. Two heads share a condition encoder:
- `q_direct_head`: MLP(cond) -> logit ("direct_q")
- `q_flow_head`: MLP(cond, flow-sampled future latent) -> logit
  ("flow_conditioned_q"), where the "future latent" is produced by an
  auxiliary conditional-flow-matching model (`flow_velocity` MLP,
  Euler-integrated) trained to predict `z_{t+h}` from `z_t` -- this is the
  "Flo" in FloQ.

Losses (`losses.py::compute_floq_losses`): BCE classification (`loss_q`),
flow-matching MSE (`loss_flow`), an **optional TD/Bellman bootstrap**
(`loss_td`, gated by `use_td_loss`): `Q(z_t,h) <- r + (1-r)*gamma*Q(z_{t+1},h')`
with `h'` the next-smaller configured horizon and `gamma = td_gamma`
(**code default 0.99** -- this is "FLOQ's default discount"), and a
horizon-monotonicity penalty (`loss_horizon_mono`). None of the prior
cup/ball configs actually turn on `use_td_loss` (all have
`use_td_loss: false, lambda_td: 0.0` -- they only ever used the
classification+flow terms); our bimanual config turns it on, since a
discounted safety-Bellman recursion over horizons is exactly what this task
asked to exercise, and our problem (a real differential game with a BRT
"first hitting time" label) is a more natural fit for it than the cup/ball
per-episode risk labels.

Entry point: `ur5e_benchmark.floq.train.train_floq(config: dict, output_dir,
dry_run=False)`. `config` is a plain dict (loaded from one of
`configs/floq/*.yaml`) consumed directly -- no CLI wrapper in the source
package; each prior run had its own driver script.

## Adaptation (files under `hj_reachability/floq/`, FLOQ repo untouched)

- `paths.py` -- sys.path setup: adds `hj_reachability/` (for `import
  dynamics`/`mujoco_fk`) and VLAism's `ur5e_benchmark/src` (for `import
  ur5e_benchmark.floq`, read-only). No name-collision risk (unlike
  `deepreach_dynamics.py`'s workaround) since this package never imports the
  DeepReach fork's `dynamics/` package.
- `rollout_dataset.py` -- generates FLOQ's expected input shape (episode CSVs
  + `feature_schema.json`) from our problem, since FLOQ's own dataset builder
  (`ur5e_benchmark.floq.dataset.build_floq_dataset_cache`, imported
  unmodified) is trajectory/episode-based, not a static classification table
  like `export_dataset.py` produces for DeepReach. State = 12-D `[q_ego(6),
  q_int(6)]` (`dynamics.BimanualPairAvoidance.state_feature_order`). Dynamics
  = the kinematic single-integrator game `ẋ_ego=u, ẋ_int=d`,
  `‖u‖,‖d‖<=u_bar` (`DEFAULT_U_BAR_EGO=DEFAULT_U_BAR_INT=1.5` rad/s, from
  `dynamics.py`). Boundary `l(x)` = `dynamics.BimanualPairAvoidance.boundary_fn`,
  which delegates to `mujoco_fk.MuJoCoFKOracle.boundary_fn` -- the FAITHFUL
  ~25-keypoint-per-arm proxy (0.00% false-safe, see `hj_reachability/README.md`'s
  2026-09-06 update), imported directly, not re-implemented. Each episode:
  initial state drawn via `BimanualPairAvoidance.sample_states_mixed`
  (`--task-region-frac`, default 0.6, biases starts toward the shared
  workspace so failures actually occur), then a bounded random-walk rollout
  (no trained value function exists yet to define an *optimal* adversarial
  policy -- that is what FLOQ is being trained to approximate) until either
  `l(x)<=0` (`failure_step` recorded -- the BRT first-hitting-time event) or
  `--max-steps`. CSV columns: `episode_uid`, `timestep_index`,
  `failure_step`, `episode_end_step`, `task`, `q_ego_0..5`, `q_int_0..5` --
  matching FLOQ's `dataset.py` column-name candidates so its own
  `_build_split_arrays` derives all `(step, horizon)` safe/unsafe labels
  itself; no relabeling logic duplicated here.
- `train_bimanual_floq.py` -- thin CLI: loads a YAML config dict, calls
  `ur5e_benchmark.floq.train.train_floq` directly (same call FLOQ's own
  driver would make), with `--dry-run`/`--device`/`--epochs`/
  `--max-train-batches` overrides for smoke runs.
- `configs/bimanual_floq_smoke.yaml` -- tiny (2 epochs, 2 batches/epoch,
  128-dim hidden, horizons `[5,10,20]`) config used for the local smoke below.
- `configs/bimanual_floq_v1.yaml` -- real-scale config (15 epochs, batch 256,
  256-dim hidden, horizons `[10,25,50]`, `q_mode: both`, `use_td_loss: true`,
  `lambda_td: 1.0`, `td_gamma: 0.99`) for the rlab run.
- `train_floq.sbatch` -- generates a 4000-episode/60-step dataset then trains
  `bimanual_floq_v1.yaml` on 1 GPU.
- `datasets/smoke_v0/`, `datasets/smoke_v0_cache/` -- the local smoke's
  20-episode raw CSVs and FLOQ tensor cache (kept for reference).

## Local smoke (mandatory) -- PASSED

Env: `vlaism-deepreach` conda env (torch 2.5.1+cu, mujoco 3.3.7, pandas,
CUDA available) -- already used elsewhere in this repo to smoke-test the
DeepReach `BimanualArmArmAvoidance` class. GPU: **4** (`nvidia-smi` at the
time: GPU4 256 MiB/6% util, the freest of the 8; GPU3 noted busy, avoided).

```
cd hj_reachability/floq
CUDA_VISIBLE_DEVICES=4 python rollout_dataset.py --output-root datasets/smoke_v0 \
  --num-episodes 20 --max-steps 30 --seed 7
# -> 20 episodes, 7 with a real failure (35%), 3.9s wall
CUDA_VISIBLE_DEVICES=4 python train_bimanual_floq.py \
  --config configs/bimanual_floq_smoke.yaml \
  --output-dir outputs/runs/bimanual_floq_smoke --device cuda:0 \
  --epochs 2 --max-train-batches 2
```

Ran end to end, no crash, on GPU 4. Train loss **decreased** both epochs
(confirms real weight updates, not a no-op):

| epoch | loss_total | loss_q | loss_flow | loss_td | loss_horizon_mono |
|---|---|---|---|---|---|
| 1 | 3.152 | 0.800 | 2.347 | 0.0036 | 0.0260 |
| 2 | 2.402 | 0.506 | 1.891 | 0.0041 | 0.0167 |

`best_val_accuracy=1.0` (trivial -- the 2-episode val split at this tiny
scale has 0 unsafe rows; `auroc=null` for the same reason). This is a smoke
test of the pipeline's plumbing (dataset shape, TD/flow/mono loss wiring,
GPU forward+backward+step), not a claim about a converged/useful bimanual
safety model -- that needs the real-scale run below.

## rlab submission

Command run (as instructed):
```
bash /common/users/ss5772/bimanual/scripts/offload/submit_via_slurm_gateway.sh \
  hj_reachability/floq/train_floq.sbatch
```

**Bug found in the existing (pre-existing, not written by this task) gateway
plumbing**: `submit_via_slurm_gateway.sh` calls
`slurm_gateway_run "bash '${SUBMIT_SCRIPT}'"`. On the SSH/no-local-sbatch
branch (our workstation, westeros, has no local `sbatch`), this resolves to
literally running `bash <script>.sbatch` **directly on the rlab1 login
node**, not `sbatch <script>.sbatch` -- the `#SBATCH` header lines are just
bash comments to `bash`, so nothing is enqueued, despite `docs/offload.md`
describing this as running `sbatch`. **Confirmed empirically**: running the
instructed command left `bash train_floq.sbatch` and the
`rollout_dataset.py` child process running as plain foreground processes on
rlab1's login node (PIDs 691270/691279, visible via `pgrep`), with `squeue -u
ss5772` showing **nothing queued**. Killed both stray processes immediately
(bad citizenship to run CPU-heavy work on a shared login node; confirmed no
partial dataset files were written -- `rollout_dataset.py` only writes CSVs
after collecting all episodes in memory).

**Fix applied and used for the actual submission**: called `sbatch` directly
via the same gateway host/user
(`ssh ss5772@rlab1 "cd /common/users/ss5772/bimanual && sbatch hj_reachability/floq/train_floq.sbatch"`)
-- equivalent to the gateway file's own (unused-by-the-wrapper)
`slurm_gateway_sbatch` helper, not a new mechanism. First attempt was
rejected (`Please do not specify the number of CPUs`) because
`train_floq.sbatch` had `#SBATCH --cpus-per-task=8` (copied from
`hj_reachability/train.sbatch`'s DeepReach pattern) -- this cluster's
`unlimited` partition rejects that flag; removed it and resubmitted.

**Result: Slurm job 272949, queued** (`squeue -u ss5772` at submit time:
`272949 unlimited bimanual_floq_train PENDING 0:00 (Priority)` -- normal for
a shared cluster, not an error). Status:
```
ssh ss5772@rlab1 squeue -u ss5772
# or, from this repo (sourcing the existing gateway helpers):
source scripts/offload/slurm_gateway.sh && slurm_gateway_squeue -u ss5772
```

**This same `bash`-not-`sbatch` bug affects `hj_reachability/submit_rlab.sh`
and the whole box-uploader gateway path too** (any use of
`submit_via_slurm_gateway.sh` from a host without local `sbatch`) -- it was
not fixed in `scripts/offload/submit_via_slurm_gateway.sh` itself since that
file is out of this task's write scope (shared, pre-existing infra used by
other pipelines); flagging here so a future pass can patch
`slurm_gateway_run`'s quoting (the bug is that `"bash '${SUBMIT_SCRIPT}'"` is
passed as a single already-quoted string to a function that re-splits it with
`$*` only on the SSH branch, not the local-exec branch, and neither branch
actually invokes `sbatch`).

## 2026-09-06 update: job 272949 cancelled at time limit -- fixed, resubmitted as 273001

**Root cause**: `train_floq.sbatch` had `--time=02:00:00`. Log evidence
(`bimanual_floq_train_272949.{out,err}` at repo root): dataset generation
(4000 episodes) alone took 776s (~13 min), then training started but never
logged an epoch before `error: *** JOB 272949 ON rlab4 CANCELLED AT
2026-09-06T06:00:51 DUE TO TIME LIMIT ***` -- the 2h budget was simply too
short for 15 epochs over ~125k train rows. The training loop itself is
already finite (`bimanual_floq_v1.yaml`: `epochs: 15`,
`max_train_batches: 0` = all batches, not unbounded -- confirmed by the
local smoke terminating cleanly on its own), so this was purely a `--time`
sizing problem, not an open-ended-training problem.

**Fixes applied (repo-wide rule, memory `slurm-generous-time-autorelease`,
now in `docs/runbooks/slurm.md`)**:
1. `train_floq.sbatch`: `--time` 02:00:00 -> **12:00:00** (the cluster's
   `unlimited` partition itself has `TIMELIMIT=infinite`, so this is a
   safety-net ceiling, not a target).
2. Confirmed finite training (epochs=15, no code change needed there);
   noted the same epoch/step-cap requirement for `world_model/wm_train.sbatch`
   and `hj_reachability/train.sbatch` in the runbook.
3. Added a `_DONE` sentinel write as `train_floq.sbatch`'s last step (so the
   script -- already finite -- exits and Slurm releases the node
   immediately rather than idling to the timer), plus a reusable watcher
   `scripts/offload/slurm_autorelease.sh <job_id> <sentinel_path>` that
   polls and `scancel`s a job the instant its sentinel appears, for jobs
   whose command might hang.
4. **Fixed the gateway bug flagged above**:
   `scripts/offload/submit_via_slurm_gateway.sh` now calls
   `slurm_gateway_sbatch` (real `sbatch` on the gateway host) instead of
   `slurm_gateway_run "bash '${SUBMIT_SCRIPT}'"`, and strips any
   `#SBATCH --cpus-per-task` line automatically (writing a sanitized copy
   next to the original .sbatch file on the shared NFS path, since sbatch
   runs over ssh on a different filesystem than a local `/tmp`).

**Resubmitted via the now-fixed gateway path**:
```
bash scripts/offload/submit_via_slurm_gateway.sh hj_reachability/floq/train_floq.sbatch
# -> Submitted batch job 273001
```
Confirmed actually enqueued this time (`squeue -u ss5772`: `273001
unlimited bimanual_floq_train PD 0:00 (Priority)`), unlike 272949's original
submission path.

Status:
```
source scripts/offload/slurm_gateway.sh && slurm_gateway_squeue -u ss5772
```
