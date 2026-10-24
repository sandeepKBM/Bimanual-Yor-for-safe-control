# house/ocs: object-centric state (OCS)

SIMULATION ONLY. This is the single world estimate the control path reads. Design by the owner (2026-10-12); `state.py` is the schema.

## Why

The control path used to read simulator ground truth in about a dozen places (see the privileged-input audit, 2026-10-12):
people, the tray/cup pose in the shield, the base pose, the wall map, grasp contacts, and payload masses.

The fix is not a dozen separate patches. It is one interface:
- **producers** turn sensors into a `WorldState`;
- **consumers** (shield, zonotope rows/base filter, planner, and the VLM-authored IR specs) read only that `WorldState`.

Ground truth becomes just another producer (`Source.TRUTH`), used for ablations. `WorldState.privileged()` lists every
ground-truth field, so "fully perception-based" is a property we can test.

## Schema (state.py)

| piece | content |
|---|---|
| `WorldState` | `t`, `robot`, `objects{oid: ObjectState}`, `relations`, `free`, `meta`; `world_pose()`, `privileged()`, `assert_honest()` |
| `RobotState` | estimated base pose + set uncertainty (`base_source`: odom/VIO vs truth), lift, encoder q/qd, gripper gap/effort |
| `ObjectState` | `oid`, `kind`, `shape` (ir Shape, usually `ZonoShape`), `pose` in the `parent` frame, `unc` (SET: ir.Unc = position zonotope + rotation bound), `vel`, `cov` (Gaussian, estimator side), `name` + `name_conf` (VLM), `params{name: Belief}`, `dynamics` key, `source`, `t_meas`, `visible`, `age` |
| `Belief` | an interval `[lo, hi]` (what safety trusts) + optional mean/std + source (vlm prior / identified / fixed prior) |
| `FreeSpace` | 2-D log-odds grid + last-observed time per cell (unknown is explicit) |
| `Relation` | `(a, on/in/held_by/near, b, conf, source)` |

**Uncertainty has two forms on purpose.**
- Estimators keep a Gaussian `cov`.
- The safety layer uses the set `unc`, which plugs straight into the IR's robust mode and the zonotope rows. The set must be
  **calibrated** (conformal on residuals), not k·sigma. Experiment E2 measures exactly that.

**Frames.**
- `pose` is in the frame of `parent`: `None` = world/map, `"robot"` = the base frame, or another oid (e.g. the cup in the tray frame).
- Carried objects therefore move with the (estimated) base or tray automatically.

**Time.** Each object carries `t_meas`. Consumers that need the state at control time call the dynamics models to predict
forward, which handles latency. Occluded objects stay in the state with `visible=False` and growing `unc`.

## Dynamics registry (to build: ocs/dynamics.py)

| key | model | uncertainty growth |
|---|---|---|
| `static` | pose constant | none (plus the perception set) |
| `attached` | rigid to the parent (tray ← hands FK, cup ← tray) | parent's set ⊕ the attachment set |
| `slip_on_support` | cup on tray: slides when the in-plane effective acceleration exceeds mu·g, with mu from `params["mu"]` (an interval → worst case) | reach set from tray_rep/sets/reach.py |
| `rolling_on_support` | ball on tray | tray_rep reach / BallKF model |
| `const_vel` | people, movable objects | velocity set × horizon, plus a residual bound (conformal) |

`rollout(ws, horizon, dt, robot_plan)` returns a `WorldState` trajectory. IR atoms already accept a leading time axis, so STL `G`/`F`
windows evaluate on it directly.

## SE(3)-anchored tracker (ocs/anchored.py)

A general producer for any rigid object, made of swappable parts:
- `ObjectModel`: the shape as a zonotope in the object's own frame, fitted once (`from_points`) or taken from a model; the parent; the
  dynamics key; params; the contact prior.
- `MeasurementSource`: `BlobPlaneSource` (wrist blob → support plane; the same numbers as `CupWristProducer` while visible). A
  FoundationPose or stereo-cluster source implements the same `reset` / `measure`.
- `Propagator`: `StaticProp`, or `SupportSlideProp` (Coulomb slide on the support, driven by odometry and FK only).
  - The hypotheses are μ ∈ {lo, hi} × disturbance directions, plus the μ_mid nominal.
  - Velocities persist across measurements.
- `AnchoredProducer`: steps the hypotheses each tick and conditions them on each measurement (re-anchors the positions; a hypothesis
  more than `k_fals·r` from the measurement restarts from the nominal velocity). It emits pose = nominal hypothesis and set = hull ⊕ r.

`perception_pipeline(..., cup_tracker="anchored")` uses it for the cup (`anchored.cup_on_tray`). The default `"wrist"` is unchanged.
The E2–E4 driver takes `--cup-tracker`.

## Consumers (to build: ocs/consumers.py)

- `to_ir_entities(ws)`: `{name: ir.Entity}`, whose frame/unc callables read the `WorldState`. **VLM-authored specs then evaluate directly on the OCS**, with no hand-built entity table.
- `to_people_state(ws)`: the `[(xy, vel)]` the shield expects, plus per-person inflation from `unc`.
- `to_obstacle_zonos(ws)`: inflated world-frame zonotopes for zono rows / base filter / ClearanceMap.
- `to_occ(ws)`: the shield's wall grid (plus a localisation margin from `robot.base_unc`).

## Experiments (first round)

- **E1, equivalence.** `TruthProducer` → OCS → consumers must reproduce today's shield inputs exactly, so the teleop carry regression stays bit-identical through the OCS path.
- **E2, per-object estimation + calibration.** Perception producers vs truth over recorded runs. Per object kind: pose error, and
  **set coverage**, i.e. the fraction of ticks the true pose lies in the reported `unc` set. The dangerous number is under-coverage.
- **E3, per-object prediction.** Roll out each dynamics model over 0.2/0.5/1/2 s vs truth. Fit per-kind conformal residual bounds on calibration seeds, then report coverage on held-out seeds.
- **E4, constraint status agreement.** Evaluate a VLM-authored spec, e.g.
  `G(inside(cup,tray_top,.02) ∧ upright(cup,.1) ∧ clear(cup,laptop,.1) ∧ clear(cup,person,.5))`, on OCS(perception) vs OCS(truth). Report the 3-valued status agreement per tick. **False-safe** (perception says true, truth says false) is the key safety number.

---

# Round 1 implementation (2026-10-12)

## Schema changes (all additive; nothing renamed or removed)
- `ObjectState.name_source: Source | None` -- provenance of `name`. `WorldState.privileged()` now also lists `<oid>.name` when it is `Source.TRUTH`
  (the ground-truth names used until the VLM grounding of house/lang is wired in).
- `RobotState.yaw: float | None` -- the estimator's exact (unwrapped) yaw; `base.R` carries the same rotation. Needed so the OCS path to the shield is bit-exact.
- Convention made explicit: `Unc.rot` of an object is a bound on the error of its body AXES (tilt). The planar base's yaw bound is **not** added to children's
  rotation bounds (it enters their position sets through the lever term `rot_base * |p_child|` in `consumers.world_unc`).

## Files
| file | content |
|---|---|
| `util.py` | geometry helpers (`base_pose`, `disc_zono`, `ball_zono`, `ellipse_zono`, `person_shape`, `cup_shape`, exact LP point-in-zonotope `in_zonotope`) |
| `views.py` | `SensorView` interface (what a producer reads): `RecView` (a `Recording`) and `LiveView` (a running `TeleopSim`); `ArmFK` (robot kinematics); `TruthView` (privileged) |
| `producers.py` | `TruthProducer`, `BaseProducer`, `TrayFKProducer`, `CupWristProducer`, `PeopleProducer`, `ObstacleZonoProducer`, `NamesProducer`, `Pipeline`, `Calib` |
| `dynamics.py` | registry (`static`, `attached`, `const_vel`, `slip_on_support`, `rolling_on_support`) + `rollout(ws, horizon, dt, robot_plan)` |
| `consumers.py` | `to_env_state`, `to_people_state`, `person_info`, `to_obstacle_zonos`, `to_occ`, `loc_margin`, `to_ir_entities`, `world_unc`, `ShieldView` (shield proxy) |
| `record.py` | records a run once (truth + sensor streams) -> `Recording` (`<OUT>/rec/*.pkl.gz`) |
| `evalcore.py`, `exp_e1.py`, `exp_e234.py` | the experiments |
| `tests/test_ocs.py` | schema, world_pose chain, privileged audit, consumer equivalence on a synthetic state, rollout monotonicity, slip vs `reach.ball_step` |

## What each producer really uses
| producer | reads | honest? |
|---|---|---|
| `BaseProducer` | `loc.PoseEstimator` (wheel odometry + ZED VIO; truth is only what its sensors measure), arm/lift encoders | honest (`Source.ODOM`, `Source.ENCODER`). Falls back to `Source.TRUTH` (flagged) if the view has no odometry. |
| `TrayFKProducer` | arm + lift encoders -> FK of both hands o the NOMINAL weld relpose (`ObjectFrame.nominal`, an engineering prior: the tray is where the hands' handle offsets put it) | honest (`Source.FK`); the grasp-offset prior is the error source E2 measures |
| `CupWristProducer` | wrist-camera RGB red-blob centroid (colour threshold, `WristCam.seg_cup_proprio`: no geom ids), FK camera pose, the FK tray plane | honest (`Source.WRIST`) for the in-tray-plane position. Orientation = upright-on-tray and z = on the tray surface are contact PRIORS (a blob centroid does not observe them); `mu` = ShieldCfg prior interval (`Source.PRIOR`). Per-side blob bias is a calibration constant fitted on seeds 0-2. |
| `PeopleProducer` | head-camera depth frames (noisy stereo-law depth, grazing mask, dropout), `people_track.DepthDetector` (background subtraction against the prior house model) + `MultiTracker` (KF); depth points are placed in the ESTIMATED map frame (odometry pose) | honest (`Source.TRACKER`). Two simulator shortcuts inherited from people_track: the background-subtraction re-cast uses the true camera pose; `Visibility` (line of sight) is evaluation only. |
| `ObstacleZonoProducer` | the flagship stereo pipeline result (SGBM -> self-filter -> RANSAC planes -> DBSCAN -> zonotopes) + its measured p95 stereo inflation as `unc`; cached pickle | honest (`Source.STEREO`); capture uses the true base pose of the twin (flagship/perception), noted |
| `NamesProducer` | ground-truth association (people: nearest truth person; obstacles: the pipeline's `gt_key`) | **privileged** (`name_source=TRUTH`, flagged by `privileged()`); replace by house/lang grounding when a VLM daemon is up |
| `TruthProducer` | the simulator | privileged (`Source.TRUTH`) by definition |

`ws.privileged()` of the perception pipeline = `['<person>.name', '<obstacle>.name']` only (the names).

## Experiments (outputs in `/data/local/ss5772/molmospaces/ocs/`, each a `.json` + `.md`)
- E1 `e1_equivalence`, E2 `e2_calibration`, E3 `e3_prediction`, E4 `e4_constraints`.

```
cd /common/users/ss5772/bimanual ; export PYTHONPATH=$PWD:$PWD/experiments/ab_smooth_vs_bangbang:$PWD/experiments/realtime OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 BIMANUAL_TELEOP_FAST=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python ; O=world_model.v2.house.ocs
$PY -m pytest -q world_model/v2/house/ocs/tests                                   # unit tests (CPU, ~10 s)
$PY -m $O.record carry --seeds 0 1 2 3 4 5 --procs 6                              # record h5 carry (shield on, walking people), ~2-4 min each
( flagship desk run first: python -m world_model.v2.house.flagship.simulate --side shield --seed 0 --start desk --perception cached --out <dir with perception/*.pkl> )
$PY -m $O.record flagship --run-dir /data/local/ss5772/molmospaces/ocs/flagship_desk
$PY -m $O.exp_e1 inputs ; $PY -m $O.exp_e1 regress --seeds 0 1 --procs 8 ; $PY -m $O.exp_e1 report --seeds 0 1
$PY -m $O.exp_e234 all --procs 7                                                  # E2 + E3 + E4, ~10 min on 7 cores
```

## Round-1 results (details in `/data/local/ss5772/molmospaces/ocs/*.md`)
Recordings: h5 carry, shield ON (truth people), `people_clock="time"` (walker moves, p_stand static), seeds 0-5 (calibration 0-2, held-out 3-5) + the flagship
desk-start run (cached stereo zonotopes). Held-out coverage < 0.90 is flagged `UNDER`.

- **E1 (equivalence).** The OCS inputs equal the raw MuJoCo numbers bit-for-bit (tray/cup EnvState, people_state, wall EDT, shield proxy poses; stereo
  zonotopes -> `to_obstacle_zonos` == the pipeline's inflated `Zi` exactly). `run_episode` through the OCS shield proxy equals the direct run on every
  non-timing field (raw and shield, seeds 0 and 1: 0 fields differ). Versus the 2026-10-09 `carry_9seeds_summary.json` the only difference is the later-added
  `flmpc_qp` instrumentation field (absent in the old file).
- **E2.** Split-conformal sets fitted on seeds 0-2 do **not** transfer for episode-level errors: odometry error is a per-episode bias (0.025-0.07 m), so a pooled
  radius under-covers on held-out seeds (0.685); the normalised `k*sigma` set (k >= 2) covers 0.998 and is what the producer uses. Persons tracked while line-of-sight:
  0.895 given a track (UNDER 0.90); while occluded / coasting: 0.654 given a track and a track exists only 46% of occluded ticks (a static person's coasting track drifts
  up to 0.56 m; walker 0.8 m). Tray FK (robot frame) 0.938, cup (robot frame) 0.959 after an affine blob-centroid correction. Engineering-prior sets under-cover badly
  (tray 0.69, cup 0.006 in the robot frame). Flagship: robot-side rotation bound (calibrated on h5) covers only 0.80 on the office rig; stereo sets contain 100% of
  the front surface of book/bottle/pencup, 58-89% of box/lamp/mug and 20-44% of monitor/desk (partial views; OBB corner samples are pessimistic for round objects).
- **E3.** See `e3_prediction.md`. Truth-state rollouts (dynamics-model error only) reach ~0.9-0.98 conformal coverage; perception rollouts inherit the estimation
  errors (static persons: conformal radius from calibration seeds covers 0.55-0.75; the analytic growing set covers ~1.0).
- **E4.** Truth is violated on 76% of h5 ticks (the spec's `clear(cup, person, 0.5)`). Instantaneous OCS(perception) false-safe = **0** of 6433 violated ticks (sets ignored it
  would be 24); but perception certifies `true` on only 2.3% of satisfied ticks (cup-in-tray margin 3 cm vs a 5 cm cup localisation set; person tracks missing/coasting).
  **Predictive G[0,1 s]: the `inside` atom has false-safe 18 / 507 = 3.5%** (constant-twist rollout does not foresee the cup sliding after braking). Flagship desk: 0 false-safe, 4 unknown.
