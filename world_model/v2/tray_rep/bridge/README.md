# bridge: keypoint-cost filter vs latent-HJ filter on the loaded-tray carry

Simulation and saved data only. CPU only. No GPU, no hardware, no network. No existing `tray_rep` file was edited.
Python: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`. Run from `/common/users/ss5772/bimanual`.

## The question

SafeBimanual (Deng et al., arXiv 2508.18268) steers a bimanual diffusion policy at test time with hand-written geometric
keypoint costs: object collision, behaviour alignment, gripper poking, tearing and collision. A language model picks which costs are
active in each task stage. One of their real tasks carries a loaded plate. Their costs read the current keypoints. They do not predict how the
payload will move.

Our filter uses a learned world model (WM) and a latent HJ value V(z): "can the ball still be saved?" It looks ahead.

Claim to test: a free ball on a rim-less tray can become unsavable several ticks before it reaches the edge. A myopic keypoint cost
cannot see that. The look-ahead value can.

Both filters act in the task space of the carried tray: `u = [tray accel (3), tray tilt rate (3)]`, at 50 Hz.

## What it runs

**Carries.** Full nominal carries from the TEST split (600 carries, about 190 ticks each). The stored snapshots start at tick 20, so each
carry is rebuilt from tick 0 from its stored seed, with the same code and the same random draws as `gen_nominal.run_episode`.
Check: with F0 the rebuilt carry drops the ball at exactly the stored tick, for all 20 smoke carries in both setups.

**Nominal command `u_nom`.** `closedloop` (default): the operator law `carry_u_clip`. It reads the true tray axis, so when a filter bends the tray the
operator reacts, as a person or a policy would. Its noise advances one step per tick whatever the filter does, so the carry stays time-aligned.
`--nominal replay`: the logged actions, open loop.

**Limits.** All filter outputs go through `project_box`: horizontal accel vector at most `5*a_scale` = 1.5 m/s^2, tilt rate at most `2*w_scale` = 1.0 rad/s per axis.
The env applies its own tilt-rate slew limit (alpha = 10 rad/s^2) to every command, filtered or not. The scaled 85-element fallback set sits inside the box.

**Setups.** `piper` = noise-free full obs (`data_piper_full`). `piper_trk` = full obs plus the realistic tracker (`data_piper_trk_full`: 3 mm noise, 60 ms latency,
30 Hz, 2% dropout). `piper_trk` is the headline. Models for F3 come from `runs/<setup>/<ARM>_full_b<budget>_s<seed>/` (`wm.pt`, `value/value.pt`).

## Filters

Each is a callable `f(obs_hist, act_hist, u_nom, t) -> u`. (`act_hist` is the realised-action history that the WM filter needs.)

| id | what it does |
|---|---|
| F0 | `u = u_nom`. |
| F1 | SafeBimanual-style keypoint cost, one-step guided selection. Details below. |
| F2 | Hand rule. Cap horizontal accel at `a_cap * a_max`. If the next-tick tilt would exceed `theta_max`, replace the tilt command by a levelling rate `3 * (c x z)`. No ball information. |
| F3 | Ours, least restrictive. Keep a running WM latent (`wm/policy.py` incremental filter). Apply `u_nom` while `V(f_WM(z, u_nom)) >= tau`. Otherwise apply `argmax_u V(f_WM(z, u))` over the scaled 85-set. `tau = inf` gives the always-fallback WM-greedy policy. |
| F4 | Optional oracle upper bound. Privileged MuJoCo look-ahead (see cost). |

### F1 in detail

Every tick the candidates are: `u_nom`, the scaled 85-set, and a local grid around `u_nom` (80 steps in a_x, a_y, w_x, w_y of 25% of the limit). F1 picks

    argmin  |u - u_nom|^2  +  l_align * C_align  +  l_prox * C_prox

- `|u - u_nom|^2`: accel / a_max and tilt rate / w_max, so the two parts weigh the same.
- `C_align = (tilt' / 0.2 rad)^2`. `tilt'` is the angle of the next-tick tray normal from vertical. One-step kinematic prediction of the tray only (same maths as the env: slew limit, 4 sub-steps, semi-implicit position).
- `C_prox = (relu(d0 - d') / d0)^2`, `d0 = 5 cm`. `d'` is the signed distance (inside is positive) from the perceived ball keypoint to the nearest tray-edge keypoint (50 keypoints, 2 cm apart, on the tray perimeter).
  The perceived ball keypoint is the ball position in `obs` (the tracker output in the trk setup).
  The keypoint is held fixed in the world while the tray moves one tick. This is a keypoint, not a ball model. There is no friction, no rolling, no gravity along a tilt.
  Without that world-fixed step, `C_prox` would not depend on `u` and could not change the choice.
- Ties keep `u_nom`. F1 never predicts the ball's dynamics. That is the defining property.
- Optional `vlook` (default 0) extrapolates the ball keypoint with its observed tray-frame velocity by `vlook` seconds. That is an ablation, not the faithful baseline. It is still not a payload model.

### Faithfulness note: how F1 differs from SafeBimanual

- They guide the diffusion denoising: cost gradients steer every denoising step of an action chunk. We have no diffusion policy here. We apply the same cost family as a one-step guided selection on a fixed nominal command.
- They use five costs and let GPT-4o switch them by task stage. We use two (alignment and ball-to-edge proximity). The stage switch is fixed: both always on. The weights `(l_align, l_prox)` are tuned on VAL instead.
- Their keypoints come from the gripper and object geometry in the scene. Ours are the tray plane, the tray edge and the perceived ball. Gripper poking, tearing and arm-collision costs have no meaning on this task-space testbed and are left out.
- The defining property is kept: the cost reads current keypoints, with no prediction of payload dynamics.
- Reading: if F1 fails here, it fails because it is myopic, not because it is badly tuned. It is tuned on VAL and compared at matched deviation.
- F2 is not in the paper. It is a plain "carry it level and gently" rule. In the smoke it was a strong baseline (see below), so expect it to be hard to beat on the drop rate. It pays for that with a large deviation and a high intervention rate.

## Metrics (per filter, setup, seed; over the TEST carries)

- ball-drop rate with a Wilson 95% CI (a drop = ball centre off the tray; a tunnel artifact counts as a drop and is counted separately);
- intervention rate = ticks with `u != u_nom` / ticks run;
- mean `|u - u_nom|`: raw (6-vector, mixed units) and normalised (accel / a_max, tilt rate / w_max). The Pareto plots use the normalised one;
- carry completion = fraction of the nominal path done before a drop or the end (time-indexed; it is time survived, not distance);
- **path progress** (added with the boed work) = arc length the tray origin actually travelled / nominal arc length (capped at 1), and the first tick within 2 cm of the end of the nominal path (`goal_tick`, -1 = never). New record fields: `path_len_nom`, `path_len_done`, `progress`, `goal_tick`, `goal_reached`. The report table has two new columns (`n/a` for result files written before the metric existed; those would have to be rerun). The stored nominal carries cover only about 22 to 38% of their path (the arm clips the plan's acceleration), so `goal reached` is 0 for F0 and nearly every filter; `progress` is the number to compare;
- drop rate on the carries the nominal law drops (does the filter SAVE them?) and on the carries it does not drop (does the filter CAUSE drops?), plus the counts saved / caused.
  "Nominal drops" come from the F0 run of the same setup. If F0 is missing the stored nominal onset is used. They agree.

## Run it (owner)

    cd /common/users/ss5772/bimanual ; PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
    L="$PY -m world_model.v2.tray_rep.bridge.launch_local"
    $L --stage tune --dry-run                  # list the 216 VAL jobs
    $L --stage tune   --workers 8 --jobs 1     # VAL sweeps: F1 grid, F2 grid, F3 tau x arm x seed
    $L --stage select                          # VAL -> <out>/main/chosen_params.json (+ notes: tau at grid edge, no F1 match)
    $L --stage test   --workers 8 --jobs 1     # TEST: F0, F1@N0, F1@Rc3k, F2, F3@N0, F3@Rc3k (3 seeds)
    $L --stage report                          # tables (md, json) and PNGs under <out>/main/
    $L --stage smoke  --workers 8              # the smoke (smoke1b models), writes <out>/smoke/
    $PY -m world_model.v2.tray_rep.bridge.smoke_checks --n 8 --workers 4   # determinism, resume, F0 = stored carry

One run by hand: `$PY -m world_model.v2.tray_rep.bridge.run --filter "F3:arm=Rc3k,tau=0.0" --setup piper_trk --seed 0 --split test --workers 8`.
Presets from VAL: `--filter @F1@Rc3k`, `@F1@N0`, `@F1@best`, `@F2`, `@F3@N0`, `@F3@Rc3k`. Flags: `--n-eps N` (first N carries), `--nominal replay`, `--runs-root`, `--model-setup`, `--tag`.
Output root: `/common/users/ss5772/bimanual_data/tray_rep/bridge/<tag>/<split>/<setup>/<filter>__<params>__s<seed>.{jsonl,json}`.
The jsonl has one line per carry, written as it finishes. Rerun the same command to resume. The json is the summary.
Slurm (CPU, 2 cores per task): `$L --stage tune --list-tasks`, then `sbatch --array=0-<N-1>%12 slurm/bridge_array.sbatch tune`. Run `--stage select`, then the same for `test`.
Run detached: `setsid nohup $L --stage test --workers 8 > $R/bridge/logs/test.log 2>&1 &`. Wait on a stored PID, not on `pgrep -f`.

Selection rules (VAL only, never TEST):
- F3 `tau`: per (setup, arm), seeds pooled. The `tau` with the lowest drop rate. Among taus within 1 percentage point of that, the lowest mean deviation.
- F2: same rule over its `(theta_max, a_cap)` grid.
- F1: of the grid points whose mean deviation is within 15% of F3's chosen point (per arm), the lowest drop rate. `F1@best` is the lowest drop rate on the whole grid.
  This is "operating point with mean deviation matched to F3's".
- Seeds. F3: model seed. Any filter: seed > 0 changes the tracker noise stream (seed 0 = the stored stream). F0 and F2 read no noisy input, and in the noise-free setup neither does F1. Those run seed 0 only.

Default grids: F1 `l_align` {0,1,4,16} x `l_prox` {0,10,40,160,640}; F2 `theta_max` {0.05,0.1,0.15,0.2,0.3} rad x `a_cap` {0.4,0.7,1}; F3 `tau` {-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4}, arms `N0` and `Rc3k`.
The `tau` grid is a guess. V lives in [-0.5, 1]. `report select` warns when the chosen `tau` sits at the edge of the grid. Then widen it with `--taus`.

## Smoke result (VERIFIED, 2026-10-05; smoke1b models are tiny and untrained, so the drop rates mean nothing)

Smoke outputs: `/common/users/ss5772/bimanual_data/tray_rep/bridge/smoke/` (1.4 MB: tables, PNGs, per-carry jsonl, `chosen_params.json`, `checks.json`).

- All filters (F0, F1, F2, F3 with N0 and Rc50, F4) ran end to end on 20 TEST carries in both setups, after a mini VAL sweep and `select` (10 carries). 0 failed jobs.
- `smoke_checks`: all 10 checks pass. F0 reproduces the stored nominal drop tick for every carry (20 of 20, both setups). The same run with 1 worker and with 4 workers gives identical records (F1 and F3, both setups). A resumed run equals a fresh run.
  Run seed 1 changes the tracker stream in `piper_trk` and changes nothing in `piper`. No hardware module was imported.
- Looks plausible, not evidence: on the 20 carries F0 drops 9; F2 (theta 0.1, a_cap 0.7) drops 4; the oracle F4 drops 1 (saves 8, causes 0, 9% of ticks changed).
  The hand rule saves a lot here, at a mean normalised deviation of 0.23 and 64% of ticks changed. Plan for F2 being a strong baseline.

Untested: F3 with trained values (only tiny untrained ones); the `Rc3k` arm name against real run directories (the smoke used `Rc50`; the loader maps `Rc3k` to `Rc_full_b3000_s*`);
the full 600-carry runs; the Slurm array files (`bridge_array.sbatch`, the task-file writer); `--nominal replay`; the quality of the default `tau` grid on a real value;
F4 at scale.

## Cost (single-core seconds per carry, measured on westeros at load average about 54, so pessimistic)

| filter | s per carry |
|---|---|
| F0 | 0.45 |
| F1 | 0.65 |
| F2 | 0.5 |
| F3 | 1.55 (torch on 1 CPU thread; the WM runs on CPU) |
| F4 (check every 5 ticks) | about 50 |

Full TEST run, 600 carries, both setups, 3 seeds (F3 6 runs per setup; F1 2 points, with 3 tracker seeds in trk; F0 and F2 one run):
about 116 CPU-min (noise-free) + 142 CPU-min (trk) = **about 4.3 CPU-h**, so about 33 min on 8 cores.
VAL tuning at 200 carries per run, default grids, 3 seeds: about 5.8 CPU-h (noise-free) + 7.2 CPU-h (trk) = **about 13 CPU-h**, about 1.7 h on 8 cores.
Cheaper tuning: `--seeds 0` cuts the F3 sweep by a third; `--val-n 100` halves everything (`--no-f1-seeds-trk` drops the extra F1 tracker seeds).
Total with defaults: **about 17 CPU-h**.

F4: 600 carries would cost about 8 CPU-h per setup, far over the 20 CPU-min limit, so the full TEST run skips it by default. It does not read the tracker, so one setup is enough.
For a subset: `--stage test --f4 --f4-n 100` (about 1.4 CPU-h per setup). A cheaper F4 (`check_every=10`) was not measured.

## Caveats

- F1 is a faithful family, not a copy. See the faithfulness note.
- The operator is a closed-loop law. If a filter tilts the tray, the operator reacts to the new tilt. F2's acceleration cap makes the operator's anti-roll tilt (planned for the full accel) a poor fit for the reduced accel. That is a property of F2, and the cap grid includes 1.0 (no cap) to separate the two effects.
- `mu_roll` (hidden) is read only by F4. No other filter sees it.
- The nominal drop rate is about 31% by design (`configs/piper.json`). Rates on the full TEST set will be near that for F0.
- The F3 `tau` choice and the F1 match both use VAL. TEST is touched once per filter.

## Files

| file | what |
|---|---|
| `filters.py` | F0-F4, spec strings, box projection, tray kinematics, arm loader |
| `episode.py` | rebuild a nominal carry from its seed and run it with a filter |
| `metrics.py` | per-carry records to summary (Wilson CI, saved / caused) |
| `run.py` | one filter x setup x seed, resumable, jsonl + json |
| `report.py` | `select` (VAL), `table`, `plots` (Pareto and tau-curve PNGs) |
| `launch_local.py` | local launcher (smoke, tune, select, test, report), Slurm task-file writer |
| `smoke_checks.py` | determinism, resume, F0 = stored carry, seed effect, no-hardware scan |
| `slurm/bridge_array.sbatch` | optional CPU array job (one line of the task file per task) |
