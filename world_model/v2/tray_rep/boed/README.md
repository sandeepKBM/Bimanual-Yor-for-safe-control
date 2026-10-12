# boed: does a certified-safe probe pay off on the tray + ball testbed?

Simulation and saved data only. CPU only. No GPU, no hardware, no network. Nothing outside `tray_rep/{boed,bridge}` was edited.
Python: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`. Run from `/common/users/ss5772/bimanual`.
Data and results: `/common/users/ss5772/bimanual_data/tray_rep/boed/`. The draft pre-registration is `PREREG_BOED_DRAFT.md` there. It is not frozen.

## The question

The ball's rolling resistance `mu_roll` is hidden. It is drawn per episode, log-uniform in [0.0005, 0.006] m, and it is not in the observation.
It decides 35 to 50% of the drops. A gentle tilt-and-hold of the tray reveals it: the ball rolls down the slope, and how fast it rolls tells us `mu_roll`.
Does a probe that is certified safe before it runs pay for itself in time and drops, against a reactive filter?
An earlier pendulum study (G3) found that probes rarely beat a reactive switch as a one-step rule. What is left to test is a probe that is early,
reusable, and priced by what it does to the carry. This directory is the harness for that test. It does not run the test.

## What is here

| file | what it does |
|---|---|
| `carry.py` | The carry plan with a speed knob (`I`) and re-planning. Speed 1 is the stored carry, bit for bit. |
| `model.py` | The analytic rolling-ball model, the 40-point belief over `mu_roll`, the probe likelihood. `calib.json` is its calibration to MuJoCo. |
| `estimator.py` | A Kalman filter for the ball from the tracker stream (3 mm noise, 3-tick delay, 30 Hz, dropouts). |
| `probe.py` | The probe controller (settle, pre, hold, balance), its certificate, the shadow run. |
| `policy.py` | The predicted-drop table, its monotone fit, the intensity rule, the intensity grid. |
| `episode.py` | One carry under one policy (P1 to P6, or a table cell). |
| `run.py` | One policy x setup x value seed over a split. Resumable. Own output files per shard. |
| `table.py` | Builds the predicted-drop-vs-intensity table from the VAL cell runs. |
| `report.py` | Tables, belief calibration, paired-bootstrap criteria. |
| `check_model.py` | Calibrates the analytic model, reports the analytic-vs-MuJoCo gap, checks the posterior. |
| `launch_local.py` | Local launcher and Slurm task-file writer. |
| `slurm/boed_array.sbatch` | CPU array script. No `--cpus-per-task` (the cluster rejects it). No temp-dir delete. |
| `smoke_checks.py` | Determinism, resume, equality with the bridge, certificate purity. |

The bridge (`../bridge/`) gained a path-progress metric. See "Path progress" below.

## How it works

**Setups.** `piper` = noise-free ball observation. `piper_trk` = the realistic tracker. `piper_trk` is the headline. The F3 filter is the bridge's:
the Rc 3k value (`Rc_full_b3000_s<seed>`), `tau` = the bridge's VAL choice for each setup (`bridge/main/chosen_params.json`: 0.05 for `piper`, 0.1 for `piper_trk`).
Value seeds 0 to 2 are the F3 model seed and also shift the tracker noise stream.

**Intensity.** The stored carry (P2) is three straight transports planned for an arm about seven times stronger than the Piper. The arm clips the
acceleration at 1.5 m/s^2, so the tray covers only about 30% of the planned path. Divide every duration of the plan by a speed factor `I` and you
get the same path at a different pace. The peak acceleration falls as `I^2`. The knob used by the policies is relative to the fastest plan the arm
can execute without clipping: `I_rel = 1` is that plan, `I_rel = 0.4` takes 2.5 times as long. Grid: 1, 0.75, 0.55, 0.4. Below `I = 1` each
transport re-aims at its waypoint from where the tray is when it starts, and a short touch-up closes any gap at the end. The carry ends when the tray
has reached the goal and stopped.

**Belief.** 40 log-spaced grid points, uniform prior in log. The likelihood uses the ball samples the tracker delivers during the probe: a sample is new
when the ball part of the observation changes, and it measures the ball three ticks ago. The model is `(1+k) a = g c_xy - (mu/r) g cos(theta) v_hat`,
integrated at 4 ms over the measured tray axis. The ball's start state is a nuisance with a Gaussian prior from the ball filter. It is integrated out
(two Gauss-Newton steps and a linearised marginal). The noise is the tracker noise plus the model gap (1 mm with the tracker, 2 mm without).

**Probe.** The tray stands still and tilts about y by 0.12 rad. The arm authority shapes it: the slew limit makes the ramp take 0.22 s. Phases:
`settle` (probe-on-trigger only: bring the ball to the centre), `pre` (8 ticks level: samples that fix the start state), `hold` (tilt until the tracked ball
has moved 2 cm or 0.56 s have passed), `balance` (a PD law on the tracked ball brings it back to the centre and stops it). The balance phase is not
decoration. A free ball on a level tray with a small `mu_roll` keeps rolling and leaves the tray. The likelihood window is `pre` + `hold`.

**Certificate.** Before the probe runs, the whole controller is run on a shadow env restored from the true state, with `mu_roll` set to the lowest value
the belief still allows (the 1% quantile). It passes only if the ball stays on the tray with at least 2 cm of edge distance all along, and ends in a state
the oracle library (`oracle.certify`) labels hold_ok or savable. The shadow reads the true MuJoCo state. A robot could not do that. It is a simulation-side
certificate, and it is the optimistic one.

**Policies.** The F3 filter is on in every carry phase.

| id | what it does |
|---|---|
| P1 conservative | intensity from the table at the lowest `mu` of the prior. Fastest grid intensity whose drop rate there is at most eps (2%); the slowest if none is. No probe. |
| P2 reactive | the stored carry plus F3. No probe. It is the bridge's F3@Rc3k, and the smoke checks that it matches carry for carry. |
| P3 probe-at-pickup | certify, probe at t = 0, update the belief, pick the fastest intensity whose predicted drop under the posterior is at most eps, carry. |
| P4 probe-on-trigger | the stored carry until F3's first intervention. Then brake (F3 on), certify, probe, update, re-pick the intensity for the rest of the path, continue. |
| P5 oracle-mu | the intensity chosen with the TRUE `mu`. The upper bound. |
| P6 population (extra) | the fastest intensity whose drop rate averaged over the prior is at most eps. The best single speed with no information. |

P6 is not in the brief. It is the control that matters. P2 is the stored carry: it never reaches the goal, and its pace-extrapolated time (about 11 s for the whole path) is
close to the fastest plan the arm can execute (about 9 s). P2 is not slow. It is unsafe. So "P3 faster than P2" asks P3 to run at the fast end of the grid while it holds a 2% drop target that P2 does
not hold. P3 against P6 isolates the probe: both pick a feasible speed from the same table at the same eps, and only P3 knows `mu`.

**Predicted-drop table.** `drop[I, mu]` = drop rate of the F3-filtered carry at intensity `I` with `mu_roll` forced to node `mu`, on VAL, with MuJoCo.
8 log-spaced nodes. The raw counts are smoothed with a two-way monotone fit (more speed, more drops; more `mu`, fewer drops). The analytic model is
not used for the table. It cannot run the filter. Between nodes the drop rate is interpolated in log `mu`. P4 scales the table by the remaining path
fraction (constant hazard).

**Time.** One tick is 20 ms. `t_complete_s` is the time at which the tray reaches the goal, probe and hold time included. If the plan ends short of
the goal (the stored carry always does), it is the pace-extrapolated time: total time divided by path progress. It is NaN if the ball dropped. Time
comparisons use the carries that neither policy dropped.

## Path progress (bridge)

New fields in every bridge record: `path_len_nom` (m), `path_len_done` (m), `progress` (0 to 1), `goal_tick` (first tick within 2 cm of the end of the nominal
path, -1 if never), `goal_reached`. Progress is the arc length the tray origin has actually travelled divided by the nominal arc length (the sum of the three
transports). The paths are straight and the tray does not backtrack, so this is distance along the path. `bridge/metrics.py` summarises them (mean
progress, goal rate, mean goal time); `bridge/report.py` adds two columns to its table. Old fields are unchanged. Old result files still read: the new columns
show `n/a`. Old runs would have to be rerun to get progress. The stored (nominal) carry covers 22 to 38% of its path, so the metric changes the picture of F2
(the slow-down is now priced) and of every filter that slows the carry.

## Run it (owner)

```
cd /common/users/ss5772/bimanual ; PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
L="$PY -m world_model.v2.tray_rep.boed.launch_local"
$L --stage check  --workers 4                       # gap and posterior checks (minutes)
$L --stage table  --dry-run                         # list the VAL cell jobs (192)
$L --stage table  --workers 4 --jobs 12             # VAL cells, then:
$L --stage assemble                                 # -> <out>/main/table_<setup>.json and .md; READ THE TABLE before the TEST run
$L --stage test   --workers 4 --jobs 12             # P1..P6, 600 carries, both setups, seeds 0-2
$L --stage report                                   # tables and the paired-bootstrap criteria
$L --stage smoke  --workers 4                       # the tiny version, tag smoke
$PY -m world_model.v2.tray_rep.boed.smoke_checks --n 6 --workers 2
```
Slurm: `$L --stage table --list-tasks`, `$L --stage test --list-tasks --shards 4`, then
`sbatch --array=0-<N-1>%12 world_model/v2/tray_rep/boed/slurm/boed_array.sbatch <table|test>`. Rerunning resumes.
One run by hand: `$PY -m world_model.v2.tray_rep.boed.run --policy P3 --setup piper_trk --seed 0 --split test --workers 4 --tag main`.
Output: `<out>/<tag>/<split>/<setup>/<policy>__s<seed>.{jsonl,json}` (one line per carry). The TEST run reads the table; the table is built on VAL only.

## What the checks show (VAL / offline; no TEST carry used)

`check_model` (`smoke/check_*.json`) and `smoke_checks` (`smoke/checks.json`, 18 checks, all pass; the bridge's own `smoke_checks` also pass with the new fields).

- **Analytic model vs MuJoCo.** MuJoCo's rolling friction resists less than `mu_roll * N`: the effective value is 0.6 to 0.87 of it, and above `mu` about 0.0036 the ball creeps instead of sticking.
  The closed form of `hidden.py` is 2.6 mm rms off the MuJoCo ball over the pre + hold window (up to 12 mm over a 0.3 s hold). After the fitted map in `calib.json` it is 0.68 mm rms (max 1.7 mm).
  The model is used for the belief only, never for the table.
- **Posterior.** With the tracker: sd of log `mu` 0.23 against 0.72 for the prior; the 90% and 50% intervals covered 0.975 and 0.525 (40 random `mu`). Noise-free: sd 0.09, coverage 0.925 and 0.50.
  A 0.12 rad tilt cannot start a ball above `mu` about 0.0036: there the probe says "sticky", not "how sticky".
- **The probe's own drop rate.** 0 of 100 at each of `mu` = 0.0005, 0.0007, 0.001, in each setup. Smallest edge distance 2.45 cm (tracker, `mu` 0.0005). Largest excursion of the ball 12.6 cm from the centre (the edge is at 15).
  The probe takes 1.4 to 1.8 s and leaves the ball 2 to 3 cm off-centre (95th percentile 4 cm). The margin at the smallest `mu` is thin. The certificate (2 cm) passed on all 40 P3 smoke probes.
- **P2 = bridge F3.** The BOED stored-carry policy reproduces the bridge's F3 run carry for carry: drop tick, interventions, progress (6 carries, both setups).
- **Determinism.** P1, P3, P4 and a table cell give identical records with 1 and 2 workers; a resumed run equals a fresh one. The certificate leaves the real env untouched.

## Smoke run (`smoke/`, tag `smoke`; plumbing, not evidence)

20 VAL carries per arm, both setups, value seed 0, a table of 4 `mu` nodes x 4 intensities x 5 carries. `smoke/table_val.md`, `smoke/primary_val_piper_trk.md`.
All six arms ran end to end in both setups. The numbers mean nothing (20 carries; a table of 5 per cell). Things the smoke did show:

1. P4 as first written dropped carries that P2 saves. Plain braking (`brake_action`) levels the tray but does not support the ball while the tray decelerates. Fixed: an anti-roll stop (gentle deceleration with the
   operator's tilt law), started only after F3 has been quiet for 10 ticks, then a settle phase (balance the ball to the centre) before the certificate. P4 still does worst on the smallest `mu`: the ball there cannot be settled.
2. A slow carry is not safer for a small `mu`. In the no-filter exploration (20 VAL carries per cell) a ball with `mu` = 0.0005 dropped in 70 to 95% of carries at every speed; at `mu` >= 0.004 none dropped at any speed.
   So the table is not forced to be monotone along speed (only along `mu`).
3. The stored carry's pace-extrapolated time (about 11 s) sits between the fastest feasible plan (I_rel 1: about 9 s) and I_rel 0.75 (about 11 s). A 2% target will usually pick slower speeds. See the pre-registration draft, section 3.

## Cost (CPU-seconds per carry; westeros at load 30 to 70, pessimistic)

| arm or job | s per carry |
|---|---|
| P1 | 5.6 |
| P2 | 1.3 |
| P6 | 5.2 |
| P3 (probe, certificate, slow carry) | 7.9 |
| P4 | 1.7 |
| P5 | 4.3 |
| table cell (I_rel 1 to 0.4) | 3.6 to 7.4, mean 5.2 |

Full scale: VAL table 7680 carries = 11 CPU-h; TEST 21600 carries = 26 CPU-h; total about 37 CPU-h: 45 min on 56 cores, about 9 h on 4, about 1.6 h as a Slurm array (12 tasks at a time, 2 CPUs each).
The F3 filter (torch, one thread, about 8 ms per tick) is most of it; a carry at I_rel 0.4 is 1500 ticks.

## Not done, not tested

- No full-scale run. The VAL table and the TEST run have not been made. The smoke table has 5 carries per cell.
- The Slurm array scripts were written, not submitted (no Slurm submission allowed here). The task files `slurm/boed_tasks_{table,test}.txt` were written by `--list-tasks` (192 and 144 tasks).
- `launch_local --stage test`, `--stage table` at scale; `--no-cert` (the certificate ablation) was not exercised.
- The probe on a real tracker or a real ball. The tracker figures are assumed. The probe uses true-state certification (a robot could not).
- The matched-risk variant (eps set to P2's drop rate), a second probe direction (probe along y), a probe of a different tilt, a posterior-aware balance law. None was tried.
- The value of the trigger itself as evidence about `mu` (P4 ignores it).
