# v5: does a JOINT/centralized cooperative MPC beat v4-C's hand-tuned margin?

Tests v4's Sec.6 "principled next step": replace v4-C's decentralized,
hand-tuned +0.06m ego-priority margin with a JOINT/centralized MPC over
BOTH arms' torques, where priority is meant to EMERGE from the ego's grasp
constraint pinning it, not from an asymmetric margin.

## 1. Formulation (`mpc.py`)

One QP per physics step (receding horizon), decision variable = both arms'
torques over `NC=4` piecewise-constant control blocks of `L=2` physics
steps each (`NP=8` prediction steps = 40ms look-ahead @ DT=5ms). Dynamics
are linearized by freezing each arm's mass matrix/bias (ego's WITH the
grasped box reflected in, via the same `payload_reflected_terms` C0-C3/v4
already use) at the current state; under that freeze the double-integrator
recursion is *exactly* affine in the decision variable (verified
numerically to 1e-16, `_sensitivity` in `mpc.py`), so the whole problem is
a genuine linear-inequality QP, solved by the SAME solver every prior
condition uses (`real_Cartpole/controller_core/constrained_box_qp.py`,
numba-accelerated dual ascent) -- no new solver written.

- **Objective**: quadratic tracking of the ego's grasp-hold pose (zero
  target velocity) + the intruder's own trackable nominal goal
  (`run_ab4.intruder_ctrl_nominal`, reused) at every predicted step, plus
  an inter-block jerk penalty and a small effort regularizer.
- **Collision constraint**: the `K_PAIRS=5` nearest sphere pairs (from
  `sim_common.all_pair_barriers`), linearized barrier `h_lin(t) >= 0` at
  EVERY predicted step (`t=1..8`) -- using the SAME `D_MIN` both arms
  already share symmetrically. **No asymmetric margin anywhere in this
  constraint.**
- **Ego grasp**: hard-ish linear inequality on the EE's predicted
  Cartesian translation + small-angle rotation (2 stages x 3 axes x 2
  signs), NOT a literal joint-space box (kept small/cheap: 24 rows,
  independent of horizon length).
- **Solve time** (measured, full 1200-step run): mean **16.8ms**, max
  **49ms**, m=64 constraint rows, n=36 decision vars. Physics step budget
  is 5ms (DT). **Not real-time-plausible** as configured -- 3-10x over
  budget, dominated by the reused dual-ascent solver's row-count scaling
  (a synthetic probe at m=66/n=36 measured 17-43ms even numba-accelerated;
  this solver's own docstring says it is "deliberately scoped to a small
  number of constraints," not this horizon-times-K row count). A separate,
  shared cost (`all_pair_barriers`, ~17ms/step, already flagged as a
  bottleneck in `COLLISION_FIDELITY.md`) sits on top of this and affects
  every condition here, not just MPC.

## 2. A critical, honest sub-finding: the grasp-cap/feasibility knife-edge

The FIRST grasp-slack value tried (`EE_SLACK_POS=0.015m`,
`EE_SLACK_ROT=0.03rad` -- "keep the grasp essentially rigid," the most
literal reading of "hard-ish constraint") is **catastrophic**:
`qp_infeasible_frac=0.89`, `min_h=-0.0354` (worse than A), box drift
**1.21m** / **180deg** (a full flip -- worse than A's 0.23m/87deg),
`rms_ee_jerk_ego=64,088` (166x A's). Once the ego is pinned this tightly,
the intruder alone cannot decelerate enough within an 8-step/40Nm horizon
to keep the shared barrier non-negative at every predicted step; the
constraint set goes infeasible, the dual-ascent solver's saturated
multipliers (this is the solver's own documented failure mode, not a new
bug) drive both arms' torque to bang between the +-40Nm box bounds, and
the result is worse than doing nothing -- **the same qualitative failure
v4 Sec.2 found for its rejected high-alpha decentralized attempt, now
reproduced in the CENTRALIZED formulation.** A 3-point manual probe found
`EE_SLACK_POS=0.05m`/`EE_SLACK_ROT=0.2rad` is on the feasible side of a
SHARP transition (0.4% infeasible vs. 68-72% just one probe point
tighter) -- this looser value is what the table below reports as "MPC."
**This is slack-tuning that was required, not optional** -- exactly the
kind of caveat this task asked to report honestly. Full tight-grasp run
preserved at `logs/v5_MPC_TIGHTGRASP_infeasible.npz` /
`logs/v5_summary_TIGHTGRASP_infeasible.json`.

## 3. Metrics (A | v4-C | MPC, slack-tuned)

| metric | A (adversarial) | v4-C (decentralized, hand-tuned) | MPC (joint, no margin, tuned slack) |
|---|---|---|---|
| min h [m] | -0.0309 | +0.0159 | **-0.0013** |
| min real dist [m] | 0.041 | 0.095 | 0.075 |
| collision (real contact) | **True** | **False** | **False** |
| box dropped (>5cm) | **True** | **False** | **True** (6.2cm, just over) |
| box drift trans max | 0.231 m | **0.016 m** | 0.062 m |
| box drift rot max | 87.5 deg | **1.5 deg** | 9.2 deg |
| ego RMS EE jerk | 385 | **25** | 268 |
| intruder RMS EE jerk | 152 | 438 | 208 |
| intruder progress (frac of A's depth) | 1.00 | 0.70 | **1.07** (barely yielded) |
| `hdot` sign changes/s | 12.3 | **0.68** | 19.8 (worst of all three) |
| QP infeasible frac | 0.0 | 0.0 | 0.0017 |
| solve time mean/max | n/a | n/a | 16.8ms / 49ms |

## 4. THE KEY ANSWER

**No -- the joint MPC does NOT cleanly beat v4-C, and it is NOT robust
without hand-tuning something.** Two honest findings, not one:

1. With the grasp treated as genuinely tight/rigid (no hand-tuned
   priority margin AND no grasp slack either), the joint formulation is
   **worse than doing nothing**: 89% infeasible, catastrophic jerk/drift,
   full box flip. Making the collision constraint hold at every one of 8
   predicted steps, symmetrically, with a rigidly-pinned ego, is simply
   over-constrained given the intruder's actuation/horizon budget --
   removing v4-C's asymmetric margin did not make the problem well-posed
   by itself.
2. Once the grasp cap is loosened enough to restore feasibility (0.17%
   infeasible, robust), the hoped-for mechanism ("grasp pins the ego, so
   the free intruder does the yielding") **did not emerge**: the
   intruder's progress-vs-A is 1.07 (it yielded LESS than in v4-C, barely
   at all), while the EGO absorbed most of the avoidance via its own
   motion -- exactly backwards from the thesis. Net result: real
   collision is avoided (`collision=False`, like v4-C) and min_h is far
   better than A's, but box drift (6.2cm) still crosses the same 5cm
   drop threshold v4-C cleared with an order of magnitude to spare
   (1.6cm), and chatter (`hdot` sign-changes/s=19.8) is the WORST of the
   three conditions, worse even than the adversarial baseline.

So: MPC lands strictly between A and v4-C on collision safety, and
strictly WORSE than v4-C on grasp retention and smoothness, using MORE
wall-clock per step (16.8ms mean, not real-time-plausible at DT=5ms) and
requiring its own manually-probed tuning (the grasp slack) to avoid
outright collapse. The joint/centralized reformulation removed the need
for an asymmetric collision margin, but replaced it with a different
fragile constant (grasp-slack tightness) sitting on an equally sharp
feasibility cliff -- it did not deliver the hoped-for free lunch.

## 5. Honest caveats

- Dynamics are linearized (frozen M/bias, double-integrator) over a short
  (40ms) horizon -- a real nonlinear-MPC solve was judged too fiddly for
  the time budget (per the task's own documented fallback); this
  approximation's error is presumably part of why min_h still dips
  slightly negative even in the feasible/tuned configuration.
- The reused `solve_constrained_box_qp` is explicitly documented as scoped
  for a small number of constraints; m=64 rows here (`K_PAIRS=5 * NP=8` +
  24 grasp rows) already pushes it to 10-50ms/solve even with its numba
  fast path -- this is a solver-choice limitation, not something a
  cleverer formulation of THIS same solver call would fix without either
  fewer rows or a different solver.
- Only one scenario/seed (same as v1-v4); the grasp-slack feasibility
  boundary was located by a 3-point manual probe, not a fine sweep, so
  the specific 0.05m/0.2rad value is a validated-but-single-point choice,
  same epistemic status v4 flagged for its own margin constant.
- `all_pair_barriers`'s ~17ms/step cost (flagged in
  `COLLISION_FIDELITY.md`) is shared infrastructure, not unique to MPC,
  but it directly eats into the same real-time budget the QP solve does.

## 6. Figures

`artifacts/ab5_distance_h.png`, `ab5_box_drift.png`, `ab5_ee_jerk.png`,
`ab5_intruder_progress.png`, `ab5_solve_time.png`, `ab5_summary_bars.png`,
`ab5_render_montage.png` (visually: box stays in-gripper for both v4-C and
MPC at "after," unlike A where it has visibly fallen onto the table on its
side; MPC's peak-intrusion frame shows a closer/more chaotic arm
approach than v4-C's).

## 7. Blockers

None -- full A / v4-C / MPC comparison (including the rejected tight-grasp
attempt) completed and logged.

## 8. Rollback

Delete `mpc.py`, `run_ab5.py`, `make_plots_v5.py`, `render_montage_v5.py`,
`RESULTS_v5.md`, `logs/v5_*.{npz,json}`, and `artifacts/ab5_*.png`. Nothing
under `scenes/`, `YOR/`, `hj_reachability/`, or `sim_common.py` was
touched (this experiment only imports `sim_common`/`run_ab4`, never edits
them).
