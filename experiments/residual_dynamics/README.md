# Residual dynamics for a feedback-linearized bimanual arm

Learns the gap between what a feedback-linearization (FL) controller
*commands* (an acceleration `a`, converted to torque via `tau = M_nom @ a +
c_nom`) and what a **mismatched real plant** actually *does* (its realized
`qacc`), so a learned residual term could later be added to the FL command
to compensate for unmodeled dynamics.

## Files

- `perturbed_plant.py` -- model loaders (`load_nominal()`,
  `make_perturbed_model()`), arm dof/qpos addressing (reuses
  `sim_common.joint_addrs`, not reinvented), the FL torque helper
  (`fl_torque`, matching `mpc_fl.py`'s exact convention), collision-free
  state sampling, and the nominal-plant sanity check.
- `collect_residual_data.py` -- CLI: generates the dataset.
- `models.py` -- `MLP` and `SIREN` torch models, `[q,qd,a](36) ->
  residual(12)`.
- `train_residual.py` -- CLI: trains either model, reports the decisive
  metrics, saves a checkpoint + `<out>.metrics.json`.

## The two models of the scene

Both are `scenes/scene_yor_canonical.xml` (via
`experiments/ab_smooth_vs_bangbang/sim_common.py`, unmodified on disk;
in-memory `MjSpec` patches only), with both arms' position servos disabled
(this experiment drives the arms purely through `qfrc_applied`, so a live
position servo fighting a manually-set `qpos` would otherwise inject a
large spurious torque -- this was in fact the first bug found while
building this experiment, see `perturbed_plant._disable_arm_position_servos`).

**Shared clean baseline.** The scene ships every arm joint with a
non-zero default `frictionloss=0.3` Nm. `frictionloss` is a *constraint*
force (`qfrc_constraint`), not part of `qfrc_bias`/`qfrc_passive`, so
`mpc_fl.py`'s FL law can never cancel it -- even on the model it was
derived from. Both `load_nominal()` and `make_perturbed_model()` therefore
start from the SAME baseline with arm `frictionloss` zeroed
(`_load_clean_baseline_spec`), so the nominal model really is one FL is
exact on, and the perturbed model's dynamics mismatch is *exactly* the
named `PERTURB_*` constants below -- nothing implicit.

**Perturbation** (`make_perturbed_model`, magnitudes as named constants at
the top of `perturbed_plant.py`), added on top of that shared baseline,
independently for every one of both arms' 6 joints / relevant links:

| perturbation | constant | value |
|---|---|---|
| joint frictionloss | `PERTURB_FRICTIONLOSS_ADD_NM` | 1.0 Nm |
| joint damping | `PERTURB_DAMPING_ADD_NM_S_PER_RAD` | 0.5 Nm·s/rad |
| per-link body_mass scale | `PERTURB_MASS_SCALE_LO/HI` | U[0.85, 1.15], drawn once per link, fixed seed `PERTURB_MASS_SEED=20260910` |

## Residual definition

Per sample: sample a collision-free `(q, qd)` (uniform in joint range /
`~N(0, 1.0^2)` clipped to `+-3.0` rad/s, rejecting any config with an
arm-involved contact -- see caveat below); sample `a ~ N(0, 3.0^2)`
rad/s^2 (12-D), rescaled per-arm if needed so `|M_nom@a + c_nom| <=
TAU_MAX`; `tau = M_nom @ a + c_nom` (nominal); apply `tau` to the
perturbed plant at the same state via `qfrc_applied` + `mj_forward`
(instantaneous, no integration error); `residual = qacc_realized - a`.

## Collision-free state filtering (caveat, found during smoke debugging)

Random full-joint-range configs *always* have >=1 MuJoCo contact (the
scene's `test_box` resting on the table), but that contact's Jacobian is
zero on the arm dofs and is harmless. An **arm-involved** contact (self-
collision, arm-vs-table, arm-vs-box) is a different story: its constraint
force is real, large, and has nothing to do with the FL/dynamics-model
mismatch this experiment targets. `sample_collision_free_state` rejects
and resamples until no contact involves a `left_arm_*`/`right_arm_*` body
(empirically ~4% of random full-range configs, so rejection sampling is
cheap; both the sanity check and `collect_residual_data.py` use it).

## Sanity check caveat: why the threshold isn't 1e-6

With the clean baseline above, FL *is* exact for the arm's own 6x6 mass-
matrix block -- but each arm also carries 2 finger dofs, and the scene has
an independent free-floating box, both coupled to the arm through the
*full* mass matrix's off-diagonal blocks. `mpc_fl.py`'s FL law (reused here
unchanged, per the task) only ever builds `M_e`/`c_e` from the arm's own
6x6 diagonal block -- it does not reflect finger/box inertia into the FL
command (same simplification the existing controller makes). That leftover
coupling alone produces a small nonzero nominal-plant residual: measured
over 30 random collision-free states, mean 0.015, max 0.035 rad/s^2.
`SANITY_NOMINAL_RESIDUAL_THRESHOLD = 0.1` sits comfortably above that
floor and >=1000x below the smallest observed *perturbed*-plant residual
(min 33 rad/s^2 over the same 30 states) -- it cleanly separates "FL
working as designed" from "dynamics mismatch present," which is the
property actually being checked.

## Running it

```bash
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python

# collect
$PY experiments/residual_dynamics/collect_residual_data.py \
    --num-samples 2000 --out experiments/residual_dynamics/data.npz --seed 0

# train (MLP)
$PY experiments/residual_dynamics/train_residual.py \
    --data experiments/residual_dynamics/data.npz --arch mlp --epochs 5 \
    --out experiments/residual_dynamics/ckpt_mlp.pt

# train (SIREN)
$PY experiments/residual_dynamics/train_residual.py \
    --data experiments/residual_dynamics/data.npz --arch siren --epochs 5 \
    --out experiments/residual_dynamics/ckpt_siren.pt
```

## SIREN caveat (found during the smoke)

`SIREN`'s paper-standard `w0_first=30.0` (the default in `models.py` and
`train_residual.py --siren-w0-first`) is tuned for coordinate-like inputs
normalized to `[-1,1]` (images/SDFs). Data collected here has a heavy-
tailed residual (wrist joints have small effective inertia, so the same
~1 Nm frictionloss mismatch produces a much larger `qacc` error there than
at the shoulder -- see per-dof std in the smoke run: joints 3/4/5 and
9/10/11, the wrist triples, have 3-10x the std of joints 0/1/2 and 6/7/8).
On that target, with only 5 epochs / 2000 samples, `w0_first=30` never
converges below the zero-prediction baseline (`fraction_removed` came out
**negative**, verified down to -0.40 across a learning-rate/batch-size/
hidden-width/loss-function sweep -- root cause confirmed by checking that
even closed-form linear regression on the same standardized data recovers
`fraction_removed=0.42`, i.e. the target is learnable, `w0=30` is just a
poor inductive bias for it at this sample/epoch budget). The mandatory
smoke below uses `--siren-w0-first 6.0`; the 30.0 default is left as the
documented "standard" choice for a longer real run, where more
epochs/data may make it converge fine.

## Smoke result (2000 samples, seed 0, 5 epochs each)

- Nominal-plant sanity: `max|residual| = 6.068e-03 rad/s^2` (< 0.1) -- **PASS**.
- Perturbed-plant residual is enormously larger at the same states/`a`:
  measured mean 87.8, max 2709 rad/s^2 (dataset-wide); a 30-state spot
  check gave perturbed-residual range [33, 159] vs nominal-residual range
  [0.003, 0.035] rad/s^2 -- 3-4 orders of magnitude separation, confirming
  the perturbation actually breaks FL cancellation as intended.

| arch | epochs | train loss (standardized MSE) 1->5 | baseline RMSE (val) | model val RMSE | fraction removed |
|---|---|---|---|---|---|
| MLP (256,256, relu) | 5 | 0.948 -> 0.424 | 188.64 rad/s^2 | 116.69 rad/s^2 | **0.381** |
| SIREN (256,256, w0_first=6.0) | 5 | 0.906 -> 0.396 | 188.64 rad/s^2 | 120.38 rad/s^2 | **0.362** |

Both nets: loss decreases monotonically every epoch, and both remove a
substantial (not marginal) fraction of the FL-vs-real-plant mismatch on
held-out data. Checkpoints + metrics: `ckpt_mlp_smoke.pt(.metrics.json)`,
`ckpt_siren_smoke.pt(.metrics.json)`; dataset: `smoke_data.npz`.

## Other caveats

- 2000 samples / 5 epochs is a smoke, not a trained model -- these numbers
  are expected to be weak (per the task) and are not meant to reflect a
  usable residual-compensation controller.
- The residual target is heavy-tailed (driven by wrist joints' small
  effective inertia -- see SIREN caveat above); a real training run should
  consider either much more data, per-joint standardization weighting, or
  a robust loss (Huber/L1) if wrist-joint accuracy matters most.
- `collect_residual_data.py` samples joint configs uniformly across the
  *entire* range (`model.jnt_range`), including near-singular wrist
  configurations that produce the heavy tail above -- this is a faithful
  "explore the whole state space" sampler, not a task-specific one; a
  demo-rollout-biased sampler (mentioned as a "bonus" in the task) was not
  implemented for this smoke.
