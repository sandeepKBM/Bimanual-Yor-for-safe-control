# robust: three tests of perception error on the latent-HJ filter (tray + ball)

Simulation and saved data only. CPU only. No hardware, no network, no GPU. No file outside `robust/` was edited.
Python: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`. Run from `/common/users/ss5772/bimanual`.
Threads are pinned by `tray_rep.common`. Output root: `/common/users/ss5772/bimanual_data/tray_rep/robust/`.

This is the harness. It was smoke-tested on a few states and carries. No full-scale job was run. The numbers in the smoke outputs
are plumbing, not evidence.

## The claim under test

Latent safety filters calibrate what the world model predicts and trust what the tracker reports. On a tray with a free ball, tracker
error may break the learned value near the failure boundary, worst during a track loss. A set built from the tracker, and grown with
the age of the last good sample, may repair it better than a set built from the world-model residual.
Background and closest work: `reports/Perception robust latent safety filters.md` (Cao & Bansal 2609.34300, UNISafe).
The draft pre-registration is `PREREG_ROBUST_DRAFT.md` in the data directory. The owner freezes it.

## How the corruption works

`corrupt.py`. The env runs with its own tracker OFF, so its obs ball is exact. The filter sees `obs_c`: the exact tray part plus a
corrupted ball part (x, y, vx, vy). The corruption is a function of the true ball stream, the condition and a seed. So all filters see
the same corrupted stream on the same carry.

- `Corruptor` has the same maths and the same random draws as `env.Tracker`. Check: `trk` with no burst equals `env.Tracker` bitwise.
  It adds dropout BURSTS: a window in which every due sample is lost and the last value is held. That is how the 2% random dropouts
  already behave in the training data. A condition with no corruption passes the exact ball through (equal to the stored obs).
- Conditions: `off`, `trk` (3 mm, 3 ticks, 30 Hz, 2% dropout), `trkn6`, `lat<k>`, `trklat<k>`, `trkB<L>s|f|r`, `custom:key=val,...`.
  `B10s` = a 10-tick burst that began 0-24 ticks before the decision tick (puts states in all three phases). `B10f` = a 10-tick burst
  that starts 0-25 ticks before the nominal failure. `r` = random time.
- Phases: steady; dropout (last good sample 5 or more ticks old); reacquisition (a dropout ended less than 10 ticks ago).
- `TrackerSet`: the tracker's own uncertainty. A constant-velocity Kalman covariance per axis, updated on every accepted sample and
  propagated by (latency + age) ticks, so it grows during a dropout. A Minkowski term `lam * v_max * dt * delay` covers the latency bias.
  `radius = scale * (sigma_p + extra)`. `scale` is the one calibrated number (`tune.calibrate_scale`: 90% coverage on VAL, over all ticks
  and over dropout ticks; smoke value 0.36-0.42 for `trk`/`trkB10f`). `support()` = the centre plus 8 boundary points, each with the
  conditional velocity shift.

## Test 1: false-safe map (`test1.py`)

For each certified-savable state (972 in TEST) and each hold-ok state (541): rebuild the corrupted obs stream of ticks 0..t from the TRUE
ball (`priv`), filter the WM latent, read V(z) and the tracker-set robust value `v_rob`, then restore MuJoCo and run the monitor's own
fallback (argmax over the 85-set of V(f(z,u))) for T = 50 ticks. Log distance to the edge `d0`, ball speed, phase, outcome.
False-safe = V >= tau and the fallback fails. Binned by distance: [0,2), [2,4), [4,6), [6,8), [8,11) cm. Inner = [0,2), outer = [6,11).
`report.py test1` writes the table by bin (n, rate, P(fail given safe), false-intervention rate on hold-ok), by phase, the R1 statistic
(inner minus outer, bootstrap CI over episodes) and the R1b statistic (the same, as an excess over the `off` condition on the same states).
Both setups, both arms, any seed. `off` is the privileged reference. Note: for `piper_trk` models `off` is an input they were not trained on.

## Test 2: five filters at matched conservatism (`rfilters.py`, `episode.py`, `run2.py`, `tune.py`)

Same learned value (arm, model seed). Each filter reduces a tick to one score s; it intervenes (fallback) when s is below thr.

| id | name | score s | fallback |
|---|---|---|---|
| plain | (i) F3 | V(f(z, u_nom)) | argmax V(f(z,u)) |
| conf | (ii) one conformal threshold | V(z) (state value); `test1.conformal_tau` gives tau with P(fail given V >= tau) <= eps | same |
| epi | (iii) epistemic | minus the std of V(f(z_j, u_nom)) over the 3 seed models; minus infinity if the ensemble mean is below 0 | argmax of the ensemble-mean Q |
| aci | (iv) Cao & Bansal style | min over 9 latents within radius r_t of V(f(z_i, u_nom)); r_t from the WM residual with ACI | argmax min_i V |
| rob | (v) ours: tracker-set margin | min over 9 ball observations from the tracker set of V(f(z_i, u_nom)) | argmax min_i V |

(iv) details: residual = norm of (decoded predicted obs minus normalised actual obs), all 13 numbers. ACI: alpha moves by gamma times
(target minus error indicator). r = the (1 - alpha) quantile of a VAL calibration pool plus recent residuals. The 9 latents are the latent
and +-r along each of the 4 ball columns. `stale=skip` (default, the generous reading) does not update on ticks without a fresh sample.
`stale=hold` updates every tick.

Matching. `tune.py tune`, one cell = (setup, arm, seed, condition), on `n_val` VAL carries: (1) rho* = the closed-loop intervention rate of
plain at the bridge's VAL-chosen tau; (2) each filter is run open loop to collect its scores; (3) threshold search by interpolation until the
closed-loop intervention rate is within 10% of rho* (at most 4 runs), keeping the closest. Everything lands in
`test2/chosen/<tag>__<cell>.json`, with the trace. `tune.py test` then runs the chosen specs (+ no filter) on TEST carries.
Results are split by phase: interventions per tick of each phase; drops by carry class (the worst phase in [onset-25, onset]) and by
the phase at the drop tick. Paired differences (by carry and seed) with bootstrap CIs are in `report.py test2`.

## Test 3: latency (`test3.py`, `budget.py`)

`test3 run`: injected latency {0,1,2,3,5} ticks, plain (bridge tau) and rob (matched thr), thresholds fixed. `piper` uses `lat<k>`,
`piper_trk` uses `trklat<k>` (the models were trained at 3).
`budget.py` derives the budget and measures it. Per savable state: lead = onset minus t (stored data); D = the largest even delay such
that the nominal continuation at t + D is still certified savable (oracle, `certify`); recovery = lead - D; turnaround = ticks the best
library sequence needs to reach its closest approach to the edge. From the bridge TEST jsonl: the median fallback ticks of F3 on the carries
it saved. Derived budget = 3.5 (the task's median lead, NOT re-derived) minus the median recovery, by both methods. It can come out negative.
`report.py test3` draws the drop rate against latency with the derived budget and the median D.

## Files

| file | what |
|---|---|
| `corrupt.py` | conditions, `Corruptor`, `TrackerSet`, phases |
| `lat.py` | running WM latent with residual, batched V and Q queries on obs variants |
| `test1.py` | Test 1 runner and analysis (false-safe, R1, R1b, conformal tau) |
| `rfilters.py` | the five filters |
| `episode.py` | carry runner with corrupted obs and per-phase bookkeeping |
| `run2.py` | one filter x setup x condition x seed, resumable |
| `tune.py` | calibrate the set scale, match intervention rates on VAL, run chosen specs on TEST |
| `test3.py`, `budget.py` | latency sweep; derived and measured budget |
| `oracle_val.py` | VAL oracle (labels written outside the dataset) |
| `report.py` | tables (md, json) and plots (png) |
| `launch_local.py`, `slurm/robust_array.sbatch` | local launcher; Slurm CPU array (pattern of `bridge_array2.sbatch`) |
| `smoke.py` | smoke tests |
| `util.py` | paths, helpers |

## Run it (owner)

```
cd /common/users/ss5772/bimanual ; PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
L="$PY -m world_model.v2.tray_rep.robust.launch_local"
$L --stage oracle_val --jobs 2 --workers 2       # then: $PY -m world_model.v2.tray_rep.robust.oracle_val --merge --n-parts 10
$L --stage test1  --jobs 2 --workers 2           # TEST states (7 conditions) + VAL states (3 conditions)
$L --stage budget --workers 2
$L --stage tune   --jobs 2 --workers 2           # needs test1 VAL for the conformal tau (optional)
$L --stage test2  --jobs 2 --workers 2
$L --stage test3  --jobs 2 --workers 2
$L --stage report
$L --stage <s> --dry-run                         # list the jobs
$L --stage <s> --list-tasks                      # write slurm/robust_tasks_<s>.txt; then sbatch --array=0-<N-1>%12 slurm/robust_array.sbatch <s>
```
Everything is resumable: rerun the same command. Use `--setups piper_trk --cond2 trk,trkB10f` for the headline cells only.
Smoke: `$PY -m world_model.v2.tray_rep.robust.smoke {checks,test1,test2,test3}` (test2 takes about 10 min on a loaded host; it resumes).

## Smoke results (2026-10-05; plumbing only)

Outputs: `/common/users/ss5772/bimanual_data/tray_rep/robust/smoke/` (`smoke_summary.json`, jsonl per run, `report/smoke/*.md|png`).
- `checks` all pass: `Corruptor('trk')` equals `env.Tracker` bitwise; `off` equals the stored exact obs; a burst freezes the ball and
  yields all three phases; the tracker set grows with age (std 1.9, 3.5, 6.5, 12.7, 28.6 cm at ages 0, 2, 5, 10, 20 for B10f with the
  default sigma_a = 1, before `scale`); the Test-1 loop with `off` on `piper`/N0 reproduces `cl_eval_mj.evaluate(WMGreedy)` on 8 states
  (same states, 8 of 8 outcomes equal, min-distance difference 0.0); `run_state` is deterministic; `run2` records are identical with 1 or 2
  workers and on resume (rob and aci); no hardware module was imported.
- `test1`: both setups x both arms x `off`, `trk`, `trkB10s`, `trkB10f` on 6 savable + 3 hold-ok states; VAL oracle on 24 candidates;
  a Test-1 VAL run and the conformal tau. Phase counts show steady, dropout and reacquisition states.
- `test2`: all five filters + no filter, both setups, `trk` and `trkB10f`, 6 carries; one tune cell (`piper_trk`/Rc3k/s0/`trkB10f`,
  n_val 6) and the matched TEST runs; report.
- `test3`: sweep {0,3,5} for plain and rob on 6 carries, both setups; `budget` on 4 states; report.
The models are the real trained ones (`runs/piper*/..._b3000_s0`), but 6 carries say nothing about performance.

## Cost (CPU-seconds, measured on westeros at load average 25 to 60: pessimistic)

| unit | cost |
|---|---|
| Test 1 state-run (rebuild, filter, 50-tick fallback) | 0.25 to 0.6 s |
| Test 2 carry, no filter / plain / conf | 0.8 / 0.8 / 0.3 s |
| Test 2 carry, epi (3 models) / aci / rob | 2.3 / 3.7 / 3.8 s |
| VAL oracle state | 0.9 s (about 15 CPU-min for 1500) |
| budget state (tolerable delay + turnaround) | about 5 s |

| job | unit | CPU-h |
|---|---|---|
| Test 1, all (7 conditions + 3 VAL conditions, 12 models) | | about 18 |
| Test 2 tune, one cell (n_val 100) | | 1.5 |
| Test 2 TEST, one cell (6 runs x 600 carries) | | 2.0 |
| Test 3, one setup (5 latencies x 3 seeds, plain + rob) | | about 12 |
| Headline only (`piper_trk`, 2 arms, `trk` + `trkB10f`) | | about 62 |
| Everything (both setups, 3 Test-2 conditions) | | about 165 |

At 12 concurrent 2-CPU Slurm jobs: about 3 h (headline) and 7 h (everything).

## Caveats

- The tracker is a Gaussian model with ASSUMED values. It is not the point-cloud tracker. Occlusion by the arms or the rim and the
  one-sided-view bias are not modelled. The Gaussian set is the best case for a Kalman-based set.
- (iv) is a re-implementation from the abstract. (iii) uses value disagreement over the 3 seeds. Both are readings, not the authors' code.
- The obs ball velocity is a finite difference of noisy positions. Its noise is large next to its scale. The set perturbs velocity by the
  conditional shift only.
- The set centre is the observed (stale) ball, not an extrapolation. The latency bias is covered by the radius term.
- Matching is on the intervention rate per tick. Deviation size and task time are reported, not matched.
- "Matched" thresholds are searched to within 10% on 100 VAL carries. The TEST intervention rates can differ more. The report prints them.
- Test 1 `conf` uses V(z); plain uses V(f(z,u_nom)). At matched intervention they may be close to the same filter.
- The measured lead (onset minus t) is tens of ticks (smoke median 48), not 3.5. The task's 3.5 is a different quantity (taken as given).
  The derived budget uses 3.5 as given, so read it next to the measured D (oracle tolerable delay; smoke median 4 ticks on 4 states).
- Smoke only: 6 to 9 states, 6 carries. The full runs are the owner's.
