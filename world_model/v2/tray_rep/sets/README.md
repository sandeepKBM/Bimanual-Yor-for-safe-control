# sets: one constraint language, analytic ball reachability, filter F5

Simulation and saved data only. CPU only. No hardware, no network, no learned part.
Python: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`. Run from `/common/users/ss5772/bimanual`.
Nothing outside `sets/` was edited, except one registration line in `bridge/filters.py` (`make_filter` knows "F5").

## The question

Can analytic sets, built from geometry and written in one shared language, keep the ball on the tray without the learned
latent-HJ value? If so, a probe that shrinks the rolling-resistance set `[mu_lo, mu_hi]` loosens the constraint at once, and the
carry can go faster. This is v0. It is a tool for that test, not the test.

## Files

| file | what |
|---|---|
| `zono.py` | Zonotope (batchable), Minkowski sum, linear map, support function, half-space distance, polytope containment, interval hull, Girard reduction, sphere / ball / ellipsoid outer zonotopes, `Polytope` (H-rep, erosion). |
| `ir.py` | The constraint language: entities, relations, logic, time. Compiles to margin, rows, status. |
| `tray_spec.py` | The tray spec (dict form) and the two state adapters: a MuJoCo state, and a predicted trajectory. |
| `reach.py` | The ball model, the backup law, the reachable sets (numba). |
| `estimate.py` | Ball Kalman filter on the bridge observation stream (handles latency, hold, dropout). |
| `qp.py` | The small QP solver (Hildreth, numba). |
| `filter.py` | F5. |
| `validate_reach.py` | Coverage of MuJoCo ball paths by the predicted sets; tuning of the model-error bound on VAL. |
| `smoke.py` | F5 in the bridge harness: 20 VAL carries, both setups, determinism, cost. |
| `tests/` | 44 tests (`python -m pytest -q world_model/v2/tray_rep/sets/tests`, 10 s). |

## The IR (constraint language v0)

**Entity.** A name, a local convex shape (`Point`, `Sphere`, `BoxShape`, `ZonoShape`, or a `Region` = H-rep polytope), a frame provider
`state -> Pose(R, p)`, an uncertainty provider `state -> Unc(pos: Zonotope, rot: radians)`, and an optional `parent` entity name.
A new object is a new entity (a shape plus two functions). It is not new constraint code.

**Relations.** Each returns a signed distance. Positive means satisfied.

| relation | value |
|---|---|
| `Inside(A, B, margin)` | `min_i ( b_i - margin - h_A(n_i) )` over the facets of B. `h_A` is the support function of A in B's frame. If B is box-like, the outside value is the exact Euclidean distance. |
| `Upright(A, theta_max)` | `theta_max - angle(R_A z, up)` |
| `Clear(A, B, d)` | `sd(A (+) -B) - d`, with `sd` the signed distance of the origin to the Minkowski difference (zonotopes) |
| `Support(A, B)` | `Inside(COM(A), footprint of B's top face)`. Boxes only (v0). |
| `shrink(E, r)` | The entity with its polytope eroded by `r`. |

**Robust version.** Pass `robust=True`. Every set of A is replaced by its Minkowski sum with the uncertainty zonotope, plus a ball of radius
`rot * (reach of A + distance to B)` for the pose-angle bound. The robust margin is never above the nominal one.

**Logic and time.** `And` = min, `Or` = max, `Not` = minus, `G[lo,hi]` = min over the window, `F[lo,hi]` = max over the window (STL robustness).
If the providers return arrays with a leading time axis (a predicted trajectory), every atom returns one value per tick.

**Compile targets.**

1. `margin(spec, state, robust)`: the scalar robustness at time 0.
2. `rows(spec, state, jac, robust=True)`: linear rows `A u <= b`. `jac` holds the Jacobians of each entity's position (and of the tray axis x, y) with
   respect to the decision vector `u`. Rows exist for `And`, `G`, and the four atoms. `Or`, `F`, `Not` raise `NotConvex`.
   `Clear` uses a fixed separating direction (an inner approximation).
3. `status(spec, state)`: `true` if the robust margin is >= 0; `false` if the nominal margin is < 0; `unknown` otherwise.

**Declarative form** (`ir.from_dict`, plain dict, yaml-compatible). The tray spec:

```python
G[lo,H]( inside(ball, shrink(tray_top, r_shrink))  and  upright(tray, theta_max) )
# TRAY_SPEC_FULL = {"G": {"lo": "lo", "hi": "H", "of": {"and": [
#     {"inside": {"a": "ball", "b": {"shrink": {"of": "tray_top", "by": "r_shrink"}}}},
#     {"upright": {"a": "tray", "theta_max": "theta_max"}}]}}}
```

`TRAY_SPEC_PAPER` is the literal form from the task (`upright` outside the `G`). The filter uses the `FULL` form with `lo = 1`:
tick 0 is the present state, and no command can change it.

**Not tray-specific.** `tests/test_ir.py::test_two_box_stack_spec_uses_only_support_and_inside` builds a two-box stack spec from
`support`, `inside`, `upright`, evaluates margin, status and rows. No new relation code.

### Equivalence with the hand-coded margin (mandatory check)

The env defines `d = sdf_rect(ball centre in the tray frame)` (inside > 0, outside = minus the Euclidean overshoot) and
`l = clip(d, -0.04, 0.08) / 0.08`. The env drops the ball when its CENTRE leaves the tray. So the ball entity is a **point**, and the tray top is
the prism over `|x| <= 0.15, |y| <= 0.10`. Then `margin(inside(ball, tray_top)) = d` and `l = l_from_margin(d)`.

The task text shrinks the tray top by `r_ball` (0.03). That is a different, stricter rule: the ball BODY stays on the tray. It is the same atom with a
`Sphere(r_ball)` ball and no shrink, or a point ball and `shrink(tray_top, r_ball)`. The test checks both give the same number.
F5 defaults to the centre rule plus a 2 mm buffer (`r_shrink = 0.002`). Set `r_shrink=0.03` for the body rule.

Result (`tests/test_equiv.py`, 591 MuJoCo states restored from the `data_piper_full` TEST snapshots, 85 outside the tray, 23 within 2 cm of the edge):
`max |d_ir - d_env| = 4.4e-16`, `max |l_ir - l_env| = 6.9e-16`. Against the stored (float32) `l`: `3e-8`. 2000 random synthetic poses: below `1e-9`.

## Reachable sets (`reach.py`)

**Model.** The quasi-static 5/7 ball of `analytic.py`, in the tray frame, 50 Hz, four sub-steps:
`(1 + k) a = g_in - f v_hat` while moving; it sticks if `|g_in| <= f`. `g_in` is the in-plane effective gravity (tilt and tray acceleration),
`f = (mu_eff / r) N/m`. `mu_eff` is the calibrated map of MuJoCo's `mu_roll` (`boed/calib.json`: MuJoCo resists 0.6-0.87 of `mu_roll * N`).
The model matches the reference `qs_rollout` to 2.5 mm (mu = 0) and MuJoCo to 8 mm over a 0.6 s tilt (`tests/test_reach.py`).

**The plan.** Tick 0: the command `u` under test. Ticks 1 to H-1: the BACKUP law. It brakes the tray (`a = -v/tau_b`, capped at half the accel limit),
sets the tilt that cancels the braking acceleration, and tilts against the ball (`kd = 3 /s` on its velocity, `kp = 2 /s^2` on its position). The tilt
rate goes through the same clip and slew (10 rad/s^2) the env applies. With `kd = kp = 0` it is pure "level and brake", and the tray part of the plan
needs no ball estimate.

**The set.** `S_k = centre_k + Z_k` for ticks `k = 0..H`, in the tray frame.

- `centre_k`: the nonlinear rollout at the middle resistance.
- `Z_k` is a box (the two axes are independent) that adds:
  1. the initial uncertainty: Kalman covariance at `k_sigma` per axis, outer-approximated by a 16-gon zonotope, propagated through the error dynamics
     (frictionless double integrator; after `fb_delay` ticks the backup's feedback damps it);
  2. the model error: an acceleration box `eps_a` per axis, every tick;
  3. the resistance set: the hull of the rollouts at `mu_lo`, middle, `mu_hi`. The sign of the friction follows the velocity, and simulation takes it exactly;
  4. with `kd = kp = 0` only: a stop certificate. If friction holds the ball from some tick to the end, the velocity error becomes a bounded slide and is removed.
- Only the centre depends on `u`. Its Jacobian `d centre_k / d u` is a forward difference of the nonlinear rollout (so saturation and sticking are inside it).
  Cost: 7 rollouts of 60 ticks.

**Horizon.** `H = 60` ticks (1.2 s). In 737 VAL branches where the ball was moving (above 5 cm/s) after the first command, the backup stops it (below 2 cm/s for good)
at tick 13 (median), 26 (90th percentile), 32 (95th), 61 (99th). H = 60 covers 98.8% of them. Longer buys little and the sets grow with H.

**Initial set (estimate.py).** A Kalman filter on the position samples of obs. Prediction: the same ball model, driven by the tray axis and acceleration read from obs.
Delayed samples (3 ticks) are applied to the stored state of that tick, then replayed to now. With no sample the covariance grows. `sigma_a = 0.10` (tracker) or
`0.05` (noise-free). On 60 VAL carries it gives 3.5 mm position error and 0.031 m/s velocity error with the tracker, and `NEES = 1.6 (pos, 2 ideal), 2.8 (all, 4 ideal)`:
it is a little conservative, as intended. The velocity from a 3 mm, 30 Hz tracker is the weak part: sigma_v is about 0.04 m/s.

### Model-error bound and coverage (VAL only)

Method (`validate_reach.py`): run VAL carries with the nominal law and the estimator. Every 8th tick, copy the MuJoCo state and run the plan
(`u_nom`, then the backup, on the true ball state for the feedback) for 60 ticks. Score the true ball path against the predicted set.
`eps_a` was chosen on VAL episodes 0-79 (1500 branches per setup) as the smallest value on the grid that gives 95% trajectory coverage; episodes 80-159 are the held-out check.
`k_sigma` was raised from 2 to 2.5 because the tracker setup fell to 94% at 2.

Default parameters (`eps_a = 0.08 m/s^2`, `k_sigma = 2.5`), full 50-state-per-second trajectories, 60 ticks:

| setup | VAL 0-79 (tuning) | VAL 80-159 (held out) | step coverage | mean half-width at k = 10 / 30 / 60 |
|---|---|---|---|---|
| piper (noise-free) | 98.3% | 97.6% | 99.6% | 10 / 28 / 37 mm |
| piper_trk (tracker) | 98.3% | 97.4% | 99.7% | 34 / 61 / 64 mm |

"Trajectory coverage" = the ball stays inside the set at every tick while it is on the tray. The tray half-width is 100 mm (y) and 150 mm (x).
Files: `/common/users/ss5772/bimanual_data/tray_rep/sets/smoke/smoke_summary.json`, branch records in `smoke/coverage_branches/`.

What this is and is not: a calibrated, empirical bound. It is not a proof. The perturbation damping after `fb_delay` is a heuristic.
Coverage was checked with the backup feedback on the true ball state, not on the estimate (ideal sensor for the feedback only).

## The filter F5 (`filter.py`)

Each tick:

1. Estimate the ball (Kalman).
2. Predict the sets for the plan "`u_nom`, then the backup".
3. Evaluate the tray spec on them. Robust margin >= 0: output `u_nom` unchanged.
4. Else solve `min |D (u - u_nom)|^2` over the 6-D `u`, with the IR's linear rows (every tick of the horizon, support functions of the linearised sets), the Piper
   box (16-gon inside the accel disc) and the tilt-rate slew. `D` = 1/a_max, 1/w_max (the bridge's normalised deviation). The decision variable for the tilt is the realised rate.
5. Check the answer on the nonlinear prediction (0.5 mm slack, inside the 2 mm buffer). If it fails, re-linearise at the answer (up to 2 times).
6. Infeasible or still unsafe: apply the backup command.

**QP solver.** A dedicated Hildreth dual coordinate ascent in numba (`qp.py`), plus a row pruning step that drops rows which cannot bind inside the box.
The repo has a Goldfarb-Idnani solver (`experiments/realtime/rt_qp.py`); it is tied to the FL-MPC workspace and its numba cache guard, so it was not reused.
Checked against OSQP on 60 random problems (cost within 0.2%) and on the 9 failed solves of 4 recorded carries: 8 were truly infeasible (OSQP agrees), 1 was a slow
convergence (fixed by a looser residual test, since the nonlinear check follows).

**Status for BOED.** `F5SetFilter.status(obs_hist, u)` gives `true | unknown | false` for any command. `F5SetFilter.set_mu(lo, hi)` narrows the resistance set.
The filter never reads the hidden `mu_roll`.

**Bridge spec.** `F5` or `F5:H=60,k_sigma=2.5,mu_lo=0.0005,mu_hi=0.006,eps_scale=1,r_shrink=0.002,theta_max=0.3,kd=3,kp=2,fb_delay=5,tau_b=0.5,a_brake_frac=0.5,n_relin=2`.
Per-tick stats go into the record as `x_f5_*` (status counts, QP solved, backup used, infeasible, re-linearisations, ms).
Run: `$PY -m world_model.v2.tray_rep.bridge.run --filter "F5" --setup piper_trk --split val --n-eps 20`.

### Smoke (20 VAL carries, `smoke/smoke_summary.json`; NOT an evaluation)

| setup | F0 drops | F5 drops | F5 intervention | F5 mean deviation (norm) | progress F0 / F5 | status true / unknown | backup used / QP solved |
|---|---|---|---|---|---|---|---|
| piper | 6/20 | 0/20 | 9.1% | 0.054 | 0.30 / 0.33 | 91% / 9% | 184 / 150 |
| piper_trk | 6/20 | 0/20 | 15.0% | 0.104 | 0.30 / 0.37 | 85% / 15% | 299 / 256 |

20 carries, one seed. The drop rates mean nothing yet. Progress is path progress (bridge metric). Determinism: 1 worker equals 2 workers, bitwise on all fields except timing.
`bridge/smoke_checks.py` still passes all checks (run with `--n 6 --workers 2 --out <scratch>`).

**Cost (single core, loaded host).** Mean 2.4 ms per tick noise-free, 3.0 ms with the tracker; median carry 1.9 / 2.8 ms. Ticks where the answer is "no change" cost about 1.7 ms.
Ticks that solve a QP cost 5-30 ms, up to 38 ms measured. So the mean is under the 5 ms target and the worst tick is not. Where the time goes:
the three-valued margin evaluation through the IR (about 1 ms), 7 rollouts (0.3 ms), the QP (up to 10 ms when infeasible: 1000 sweeps before it gives up).
A cap of about 200 sweeps for the infeasible case, or an early infeasibility test, would bring the worst tick near 5 ms. Not done.

## Known weak points (read before trusting a number)

- **Conservative under the tracker.** The half-width of the set reaches 60 mm of a 100 mm tray half-width. Most of it is the velocity uncertainty of the tracker.
  Expect many `unknown` ticks and more backup use than with perfect sensing.
- The backup is used in about half of the interventions. It is stronger than "level and brake" because it tilts against the ball. With `kd = kp = 0` it is the plain
  version; that sets a tighter limit on ball speed (a low-resistance ball cannot be stopped by a level tray inside the tray).
- The tray's yaw is ignored (minimal-rotation frame). It adds a few mm; `eps_p0` and `eps_a` cover it.
- `mu_lo..mu_hi` defaults to the full prior range `[5e-4, 6e-3]`. A probe is what narrows it.
- The linearisation is in `u` only, around `u_nom`. The nonlinear check and re-linearisation are what keep it honest.
- Coverage is empirical (97-98% on VAL). It is not a guarantee. A guarantee needs a set-valued backup feedback, which v0 does not have.

## What this cannot represent

- **Non-convex shapes.** Every shape is convex (zonotope, ball, box, polytope). A C-shaped tray, a bowl, or a cup with a handle needs a union (an `Or`), and `rows` refuses `Or`.
  Ring or wall obstacles would need a branch per face.
- **Constrained zonotopes / intersections.** Not implemented. "A inside B" is a support-function test, which needs no intersection; "the free space between two sets" does.
- **Deformable objects.** No state for them; a rope or cloth has no finite pose.
- **Granular or liquid payloads.** Beans, water, sand: the payload is a distribution, not a rigid set. A bounding set is legal but very loose, and the dynamics (slosh, spill) are not in `reach.py`.
- **Dynamics other than the rolling ball.** `reach.py` is the only reachability model; a new payload needs its own model (the IR does not need changing).
- **Contact-rich or switching constraints** (grasp, slip). Not in scope for v0.
- `Support` is for boxes whose top face stays horizontal enough; it uses the horizontal footprint of the top face.
- `Clear` uses a direction grid and a few refinement steps; the separation is a lower bound, exact only as the grid gets dense.

## Suggested pre-registration criterion (the owner decides)

On the `piper_trk` setup, 600 TEST carries, 3 tracker seeds, F5 at one operating point chosen on VAL (grid over `k_sigma` x `eps_scale`; `H` and `mu` fixed):

- **Good enough:** drop rate <= F3-Rc3k's + 2 percentage points, with intervention rate <= 1.5 x F3-Rc3k's. (From `RESULT_bridge_trk.md`: F3 Rc3k drops 13.6%, intervenes 3.5%. So F5 drops <= 15.6% at <= 5.3% intervention.)
- **Report with it:** path progress (not only completion), mean normalised deviation, backup fraction, `unknown` fraction, ms per tick (mean and worst).
- Secondary, same setup: F5 vs F2 (hand rule, 1.7% drops at 67% intervention) and vs F4 (oracle, 1.0% at 8%) on the Pareto of drops against intervention and progress.
- The criterion F5 <= F3 + 2 pp at <= 1.5x intervention may be unreachable with this tracker: the set widths above say the sets are loose. If so, a null result here is informative.
  State beforehand which is the headline: tracker or noise-free.
- The BOED claim needs a second pre-registered test: F5 with `mu` narrowed to the true value +-10% against F5 with the full range, same carries. The expected effect is a lower intervention rate and higher progress at the same drop rate.
