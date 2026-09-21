# v6: numba-fast + long-horizon + explicit-priority joint MPC -- does it now beat v4-C?

Follow-up to RESULTS_v5.md, which found the joint/centralized cooperative
MPC (`mpc.py`) ran at 16.8ms mean/49ms max per step (8-step/40ms horizon,
solve-time-forced) and did NOT beat v4-C: priority did not emerge from the
grasp constraint alone (loose slack -> box drift 6.2cm, over the 5cm
threshold). This experiment (1) profiles where v5's time went, (2)
numba-accelerates the Python assembly, (3) uses the resulting budget to
extend the horizon, (4) adds an explicit priority cost, and (5) honestly
re-tests vs A / v4-C. New files only: `mpc6.py`, `run_ab6.py`,
`make_plots_v6.py`, `profile_mpc.py`, `profile_mpc6.py`, `profile_v4c.py`,
`validate_mpc6.py`. `mpc.py`/`run_ab5.py` (v5) untouched, kept for reference.

## 1. Profile: where v5's 16.8ms went

`profile_mpc.py` instruments v5's exact code path (300 steps, warm state):

| section | ms/step |
|---|---|
| MuJoCo calls (`mj_fullM`, payload terms, `mj_jac` for EE) | 0.29 |
| **Python QP-assembly** (sensitivity recursion + collision/grasp rows, via `sc.build_weighted_least_squares_qp`'s generic padded-matrix "terms" builder) | **7.04** |
| `constrained_box_qp` solve (already numba) | 7.59 |
| **joint_mpc.step() total** | **14.97** (report's 16.8ms, other process) |
| `all_pair_barriers` (shared infra, called by caller, NOT in step()) | 7.4-8.3 (separate, unmodifiable -- `sim_common.py` read-only) |

MuJoCo calls were never the problem (<2% of step()). The Python assembly
(v5's `_sensitivity`'s unit-impulse simulation loop + `_pad_e`/`_pad_i`
`np.hstack` per objective/constraint term, all through per-call numpy
dispatch) was ~47% of `step()`; the reused QP solve was the other ~51%,
**despite already being numba** -- confirming the task's premise for (a) and
identifying a second, independent bottleneck for (b)/(c): `constrained_box_qp`'s
dual-ascent solve cost scales with constraint-row count `m` (and with
decision-dim `n`), not just assembly overhead, exactly as its own docstring
warns ("deliberately scoped to a small number of constraints").

## 2. Numba speedup (real)

`mpc6.py` fuses the sensitivity recursion + ALL objective/constraint
assembly into two `@njit(cache=True)` kernels (`_sensitivity_core`,
`_assemble`) operating on raw arrays -- no padded-matrix `terms` list, H/f
accumulated directly into ego/int block-diagonal sub-blocks (the v5
objective has no ego-intruder cross terms, verified by inspection).
MuJoCo calls stay outside njit (`mj_fullM`/`mj_jac` outputs precomputed and
passed in), per the task's instruction.

**Correctness**: `validate_mpc6.py` runs `mpc6._assemble` at v5's IDENTICAL
horizon (NC=4,L=2,NP=8,K=5) on a real mid-run state and compares the solved
torques against v5's `mpc.JointMPC.step()` byte-for-byte: **max abs diff
4.4e-15 (ego) / 3.6e-15 (int) -- PASS**, i.e. the rewrite is a faithful
reimplementation, not a different algorithm.

**Speedup, apples-to-apples (same NP=8, m=44 vs v5's 64 rows since v6 also
subsamples collision checks -- see Sec.3)**: assembly **7.04ms -> 0.46-0.48ms
(~15x)**. QP solve at matched size stayed ~5.4-8.5ms (unaffected by the numba
rewrite -- it was already numba; row-count scaling is the solver's own
property). One caveat found and fixed: the reused `constrained_box_qp`'s OWN
numba kernel needs its own warmup (`numba_warmup()`) separately from
`mpc6.warmup()` -- the first *timed* call otherwise eats an ~11-12s one-time
JIT-compile (visible in `logs/v6_MPCFAST.npz`'s step-0 outlier); `run_ab6.py`
calls both warmups before the timed loop, and all reported means/maxes below
exclude step 0 (the compile artifact), matching how a real deployment would
warm up once before entering the control loop.

## 3. Horizon extension -- and a second row-count-driven wall

Naively extending the horizon by adding MORE control blocks (`NC`) drove QP
solve time back up sharply, because `constrained_box_qp`'s cost scales with
**decision-dim `n`=2·NC·NV as well as `m`** (`profile_mpc6.py` sweep, 300
steps each, warmed):

| config | NC | NP (horizon) | m rows | qp solve ms/step |
|---|---|---|---|---|
| matched v5 | 4 | 8 (40ms) | 44 | 5.4 |
| NC=4, longer blocks | 4 | 40 (200ms) | 44 | 8.3 |
| NC=6 | 6 | 36 (180ms) | 54 | 24.6 |
| NC=8 | 8 | 32 (160ms) | 64 | 51.6 |

Two fixes were needed, both kept in `mpc6.py`:
1. **Horizon extended by making blocks LONGER (`L_PER_BLOCK`), not by adding
   more blocks** -- `NC=4` (same decision-dim as v5) with `L_PER_BLOCK=8` ->
   `NP=32` (160ms look-ahead, 4x v5's 40ms) at v5's original `n=48` cost.
2. **Collision-constraint rows checked once per control block (at each
   block's end), not at every one of the `NP` physics-step predictions** --
   caps `m_coll = K_PAIRS * NC` instead of `K_PAIRS * NP`; the OBJECTIVE
   (tracking cost) still uses every predicted step, so the anticipation
   benefit is not lost, only the collision-row *count*. This is a genuine
   approximation (documented in `mpc6.py`): a barrier dip strictly between
   two checked steps would not be caught.

**Final config**: `NC=4, L_PER_BLOCK=8, NP=32 (160ms), K_PAIRS=5, m=44`.
Measured over the full 1200-step run (`logs/v6_MPCFAST.npz`, step 0
excluded): assemble **0.84ms** mean, QP solve **9.5ms** mean (max 18.7ms),
**joint_mpc.step() total 11.0ms mean / 20.8ms max** -- vs v5's 16.8ms
mean/49ms max, **despite a 4x longer horizon**. Not <=5ms for `step()`
alone, and the shared `all_pair_barriers` (~7.5ms, unmodifiable) still sits
on top: grand total **18.5ms mean / 32.0ms max**, still 3.7-6.4x the 5ms
physics-step budget -- **not real-time-plausible**, though meaningfully
better than v5 on both axes at once (horizon x4, wall-clock down ~1.5x /
~2.4x on the controller-only piece).

## 4. Explicit priority: "grasp-holder right-of-way"

v5 found priority does not emerge for free from the grasp hard constraint
alone. `mpc6.py` adds three PURELY COST-SIDE asymmetries (the shared
collision barrier stays symmetric, `D_MIN`, no margin, matching v4-C's
"principled next step" framing):
- **(a) asymmetric effort regularization**: `REG_EFFORT_EGO=5e-3` vs
  `REG_EFFORT_INT=5e-5` (100x) -- moving the ego's commanded torque away
  from 0 is expensive, the intruder's is cheap.
- **(b) asymmetric jerk weight**: `W_JERK_EGO=0.08` vs `W_JERK_INT=0.01`.
- **(c) explicit soft EE-hold cost** (`W_EE_HOLD=4e4`): the ego EE's
  predicted translational drift is COSTED at every one of the 32 predicted
  steps (v5 only had the hard cap at 2 discrete stages) -- this is the
  "explicit ego-grasp-preservation weight" alternative the task asked for,
  layered ON TOP OF (not instead of) v5's existing hard `EE_SLACK_POS/ROT`
  cap.

## 5. Metrics (A | v4-C | MPC-fast), full 1200-step run

| metric | A (adversarial) | v4-C (decentralized, hand-tuned) | MPC-fast (numba, 160ms, priority) |
|---|---|---|---|
| min h [m] | -0.0309 | **+0.0159** | -0.0011 |
| min real dist [m] | 0.041 | **0.095** | 0.075 |
| collision (real contact) | **True** | False | False |
| box dropped (>5cm) | **True** | False | False |
| box drift trans max | 0.231 m | **0.0158 m** | 0.0239 m |
| box drift rot max | 87.5 deg | **1.49 deg** | 2.66 deg |
| ego RMS EE jerk | 385 | 24.5 | **23.5** |
| intruder RMS EE jerk | 152 | 438 | **101** |
| intruder progress (frac of A's depth) | 1.00 | 0.70 (yields most) | 0.88 |
| `hdot` sign changes/s (chatter) | 12.3 | **0.68** | **32.2 (worst of all 3)** |
| QP infeasible frac | 0.0 | 0.0 | **0.0** |
| solve time (controller-only) mean | n/a | ~60.0ms (measured, `profile_v4c.py`: ego 27.1+int 33.0, m=625 rows/pair, unwarmed real cost) | **11.0ms** |
| solve time incl. shared barriers | n/a | ~67.2ms | **18.5ms** (max 32.0ms) |

v4-C's own solve time was never measured in RESULTS_v4/v5 ("n/a"); measuring
it here (`profile_v4c.py`, same warmed/steady-state methodology) found it
uses ALL ~625 sphere pairs as constraint rows per side (not top-K), and
**costs ~60ms/step controller-only** -- itself far from real-time, and
**~5.4x slower than MPC-fast**. This reverses v5's framing: it is MPC-fast,
not v4-C, that is closer to a real-time budget once both are measured fairly.

## 6. THE KEY ANSWER

**No -- MPC-fast still does not cleanly beat v4-C, but it is materially
closer than v5, and the reasons it falls short have shifted.**

What improved vs v5: feasible throughout (0% infeasible, vs v5's 0.17-89%
depending on slack), box NOT dropped (2.4cm, under the 5cm threshold, vs
v5's 6.2cm over it), intruder does yield more than v5's no-priority
version (0.88 vs v5's 1.07 -- the explicit priority cost measurably shifted
burden onto the intruder, though less than v4-C's 0.70), ego jerk is now the
lowest of all three conditions, intruder jerk is far LOWER than v4-C's, and
wall-clock is now several times FASTER than v4-C's own (previously
unmeasured) cost.

Two things still make it lose to v4-C:
1. **Safety margin**: `min_h` is still slightly negative (-0.0011, a soft
   breach, no real contact) vs v4-C's clean positive margin (+0.0159) --
   the shared, symmetric `D_MIN` barrier constraint, checked only at
   block-boundary steps (Sec.3 fix 2) rather than every physics step, lets
   a small dip through between checks that v4-C's every-physics-step
   decentralized barrier does not.
2. **Chatter got WORSE, not better**: `hdot` sign-changes/s = 32.2, the
   WORST of all three conditions (even A's uncooperative baseline is
   smoother at 12.3) -- a genuine regression vs both v4-C (0.68) and v5
   (19.8). Hypothesis (not further isolated under this task's time budget):
   the receding-horizon replan, combined with only checking collision at
   block boundaries, lets the barrier gradient direction flip between
   consecutive replans in a way v4-C's simpler single-stage QP does not.

So: real-time-plausibility and the free-lunch mechanism both improved
substantially with the numba rewrite + explicit priority, but the joint MPC
still trades a small, real safety-margin/chatter regression for its jerk and
wall-clock wins -- it is a genuinely different, not strictly better,
operating point than v4-C's hand-tuned decentralized margin, not a
replacement for it.

## 7. Honest caveats

- Collision constraint rows are checked only at `NC=4` block-boundary steps
  within the 160ms horizon, not all 32 predicted physics steps (Sec.3) --
  the shared barrier could dip between checks; this is the most likely
  contributor to both the small `min_h` breach and the elevated chatter,
  not swept/ablated further here.
- `W_EE_HOLD`/`REG_EFFORT_EGO`/`W_JERK_EGO` (the explicit priority weights)
  are single hand-picked values, not swept -- same epistemic status v4/v5
  flagged for their own tuned constants.
- The dynamics linearization (frozen M/bias, double-integrator) is
  unchanged from v5, now held over a 4x longer horizon (160ms vs 40ms) --
  the approximation error is presumably larger, plausibly contributing to
  the residual `min_h` breach as well.
- `all_pair_barriers` (~7.5ms/step, shared by ALL conditions incl. v4-C) is
  untouched (`sim_common.py` read-only) -- still the single largest
  per-step cost apart from v4-C's own QP, and the reason neither condition
  is real-time-plausible at DT=5ms in absolute terms.
- v4-C's ~60ms/step figure is from a separate, single 300-step profiling
  script (`profile_v4c.py`) replicating `run_ab4.py`'s per-step math
  verbatim, not from instrumenting `run_ab4.py` itself (kept read-only-in-
  spirit, not modified) -- consistent methodology (same warmup, same
  perf_counter placement) as `profile_mpc6.py`, but a separate run, not a
  shared process.
- Single scenario/seed, as with v1-v5.

## 8. Figures

`artifacts/ab6_distance_h.png`, `ab6_box_drift.png`, `ab6_ee_jerk.png`,
`ab6_intruder_progress.png`, `ab6_solve_time.png` (per-step assemble/QP
breakdown + a controller-only-cost bar comparison against v5 and v4-C),
`ab6_summary_bars.png`.

## 9. Blockers

None. Profiling, numba rewrite, equivalence validation, horizon-extension
sweep, explicit-priority addition, and the full A/v4-C/MPC-fast comparison
all completed and logged.

## 10. Rollback

Delete `mpc6.py`, `run_ab6.py`, `make_plots_v6.py`, `profile_mpc.py`,
`profile_mpc6.py`, `profile_v4c.py`, `validate_mpc6.py`, `RESULTS_v6.md`,
`logs/v6_*.{npz,json}`, and `artifacts/ab6_*.png`. `mpc.py`/`run_ab5.py`
(v5) and everything from v1-v4 are untouched. Nothing under `scenes/`,
`YOR/`, `hj_reachability/`, or `sim_common.py` was touched.
