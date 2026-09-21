# v4: Does a COOPERATIVE intruder dissolve the collision-vs-grasp trade-off?

Tests RESULTS_v3.md/COLLISION_FIDELITY.md's root-cause hypothesis on the real,
tight YOR geometry (`scenes/scene_yor_canonical.xml`, arms 0.185m apart): the
grasping ego (LEFT arm) alone cannot both avoid the intruder AND hold the
grasp; if the free intruder (RIGHT arm, nothing to protect) also yields, does
the ego keep the grasp?

Ego is FIXED across all three conditions to the v3 "C2" controller (smooth
HOCBF-QP, ALPHA1=ALPHA2=30, `sim_common.EgoController` + the corrected
faithful ~25-sphere/arm proxy) -- only the INTRUDER model changes. New code:
`run_ab4.py` (+`sim_common.disable_right_arm_position_servos`,
`make_plots_v4.py`, `render_montage_v4.py`).

## 1. Conditions

- **A -- adversarial baseline** (= v3's C2 row, byte-for-byte): intruder is
  the existing open-loop position-servo sweep (`sc.intruder_ctrl`, ramp +
  5Hz oscillation + ramp), unresponsive to the ego. Reproduced here as a
  same-pipeline sanity check before trusting B/C: `min_h=-0.0309`,
  `box_dropped=True` (drift 23.1cm) -- matches COLLISION_FIDELITY.md's
  corrected-proxy quick-recheck (`-0.0309`, `23.1cm`) exactly.
- **B -- cooperative symmetric**: intruder becomes torque-controlled and runs
  its OWN HOCBF-QP filter against the same faithful proxy, SAME class-K gains
  as the ego (ALPHA1=ALPHA2=30) -- both arms yield with equal authority.
- **C -- cooperative, ego-priority**: identical to B except the intruder adds
  a static extra margin (0.06m) to its OWN barrier value only (ego's barrier/
  gains untouched) -- the intruder starts backing off at a larger true gap
  than the ego does, i.e. it does most of the yielding.

Decentralized-reactive HOCBF-QP: each side solves its OWN box-QP each step,
treating the OTHER arm's currently-measured acceleration (`data.qacc` from
the previous step) as a fixed disturbance -- NOT a jointly-optimized/
centralized plan. The intruder's QP is the ego's mirror-derived with a sign
flip (v_int enters `hdot=n.(v_ego-v_int)` with a minus sign, so its
controllable term is `+n@Ji@Minv_int` vs. the ego's `-n@Je@Minv_ego`);
verified against a direct numeric check of the psi2 inequality before
trusting the full run.

## 2. A confound found and fixed before trusting B/C

Naively reusing `sc.intruder_ctrl` (which includes a 5Hz oscillation) as the
cooperative intruder's own torque-tracking target was WRONG: an isolated
check (computed-torque PD toward `sc.intruder_ctrl(t)`, no collision QP at
all) showed tracking error diverging past 1.4 rad within 1.5s, independent of
any avoidance interaction -- the 5Hz oscillation is an ADVERSARIAL-only stress
element (sim_common.py's own comment: chosen "near/above the ego task-PD's
natural frequency ... what makes bang-bang chatter"), tuned assuming a stiff
MuJoCo position servo, not a finite-gain/torque-limited computed-torque PD.
Fixed by giving the cooperative intruder a *trackable* nominal goal
(`intruder_ctrl_nominal` in `run_ab4.py`): identical ramp-in/hold/ramp-out
timeline and RIGHT_HOME_CTRL<->RIGHT_INTRUDE_CTRL endpoints, oscillation
dropped, plus a feedforward-velocity PD term (the original regulation-only PD
`Kp*(q_des-q) - Kd*qdot` has no way to track a moving reference). Isolated
re-check: <=0.16 rad transient error, ~0.04 rad stable steady-state offset.
Condition A is untouched (still the original oscillating trajectory).

A second attempt at C used a HIGHER intruder class-K gain (ALPHA=90, 3x)
instead of an extra margin, at the SAME +-40Nm torque budget -- this
**backfired**: `qp_infeasible_frac` jumped to 34%(ego)/11%(int), min_h got
*worse* (-0.052, worse than A), box drift catastrophic (1.15m). An aggressive
class-K gain without matching actuation headroom is brittle under
saturation -- a real, reported negative result, not a bug (see Sec. 4).
Switching to a margin-based asymmetry (same ALPHA=30 as B) fixed this.

## 3. Metrics (A | B | C)

| metric | A (adversarial) | B (cooperative symmetric) | C (cooperative, ego-priority) |
|---|---|---|---|
| min h [m] | -0.0309 | -0.0158 | **+0.0159** |
| min real dist [m] | 0.041 | 0.048 | 0.095 |
| collision (real contact) | **True** | **False** | **False** |
| box dropped (>5cm or table) | **True** | **True** | **False** |
| box drift trans max | 0.231 m | 0.079 m | **0.016 m** |
| box drift rot max | 87.5 deg | 42.3 deg | **1.5 deg** |
| ego RMS EE jerk | 385 | 466 | **25** |
| intruder RMS EE jerk | 152 | 620 | 438 |
| intruder progress (frac of A's intrusion depth) | 1.00 (ref.) | 0.99 | **0.70** |
| `hdot` sign changes/s (chatter proxy) | 12.3 | 5.7 | **0.68** |
| QP infeasible frac (ego / int) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |

## 4. THE KEY ANSWER

**Yes -- condition C reaches the quadrant nothing hit in v3 (collision-free
AND grasp-retained), and cleanly.** `min_h=+0.0159` (positive: the full
faithful-proxy safety margin held, not just "no MuJoCo contact"), real
min-distance 9.5cm (2.3x A's), box drift 1.6cm/1.5deg (both far under the
5cm/none-specified drop thresholds, an order of magnitude better than A), ego
jerk actually LOWER than the adversarial baseline (25 vs 385 rms -- the ego
barely has to react at all), and the intruder still closes ~70% of the gap
toward its own nominal target (11.3cm vs its 22.9cm starting separation, vs.
A's 6.2cm) -- it yields, it does not give up.

**B (symmetric, same authority) gets close but not clean**: no real contact
(collision=False) and both drift metrics improve sharply vs. A (drift
23.1cm->7.9cm, rotation 87.5->42.3deg), but `min_h` still dips slightly
negative (-0.0158, a soft-margin breach even though the real geometry never
touches) and box drift (7.9cm) still exceeds the 5cm drop threshold --
`box_dropped=True`. Equal yielding authority is a big improvement, not a
full fix.

So the answer is nuanced by which asymmetry: symmetric cooperation (B)
meaningfully shrinks the trade-off without eliminating it; giving the free
arm an explicit head start via an extra safety margin (C) eliminates it
cleanly in this scenario. This is NOT a free asymmetry, though -- see caveats.

## 5. Deadlock / oscillation

No deadlock: the intruder keeps moving toward (and substantially reaches) its
goal in both B (99% of A's depth) and C (70%). No sustained chatter either --
`hdot` sign-changes/s *decreases* monotonically A(12.3) > B(5.7) > C(0.68),
i.e. cooperation makes the interaction smoother, not more oscillatory, in
this scenario; the one place chatter/instability DID appear was the rejected
high-alpha C variant (Sec. 2), confirming this is a real risk of naive
decentralized double-hard-CBF composition in general, just not the specific
mechanism this scenario's final C uses.

## 6. Honest caveats

- This is decentralized-reactive cooperation (each side solves its own QP
  against a one-step-stale estimate of the other's acceleration), not a
  joint/centralized plan. It happened to compose cleanly here; the rejected
  high-alpha C attempt (Sec. 2/4) shows decentralized composition of two
  hard(-ish) CBF filters is NOT automatically safe -- a mismatched
  gain/torque-budget pairing pushed QP infeasibility to 34%/11% and made
  things worse than doing nothing. A joint/centralized QP (single combined
  psi2>=0 constraint solved over BOTH arms' torques at once) would remove
  this composition risk and is the principled next step if this needs to be
  robust rather than scenario-tuned.
- C's advantage comes from a hand-set 0.06m extra margin, not a
  first-principles priority mechanism (e.g. control-barrier priority/
  right-of-way weighting, or a negotiated split of the required avoidance
  budget) -- it was chosen after the higher-alpha attempt failed, i.e. it is
  a validated-but-single-value choice, not a swept/robust one.
- Every torque-command path here (ego AND intruder) still ends in the same
  low-pass + rate-limit filter as v1-v3's C2, which is known (COLLISION_
  FIDELITY.md) to let the "hard" QP's guarantee leak slightly -- consistent
  with B's `min_h` going marginally negative despite `qp_infeasible_frac=0`
  every step for both arms.
- The intruder's nominal goal was deliberately de-tuned (5Hz oscillation
  dropped, Sec. 2) to make it torque-trackable at all -- this makes the
  cooperative intruder's "goal" easier to reach than the adversarial one's,
  by design (the oscillation was an adversarial-only stress element, not
  part of the intrusion goal itself), but it does mean A and B/C are not
  driving the intruder toward byte-identical references.
- Single scenario/seed, as with v1-v3 -- no sweep over intrusion angle,
  speed, or box mass here.

## 7. Figures

`artifacts/ab4_distance_h.png`, `ab4_box_drift.png`, `ab4_ee_jerk.png`,
`ab4_intruder_progress.png`, `ab4_summary_bars.png`,
`ab4_render_montage.png` (visually confirms: box visibly slips/tumbles in A
and B by "after", stays gripped in C).

## 8. Blockers

None. Full A/B/C run + all figures completed.

## 9. Rollback

Delete `run_ab4.py`, `make_plots_v4.py`, `render_montage_v4.py`,
`RESULTS_v4.md`, `logs/v4_{A,B,C}.npz`, `logs/v4_summary.json`, and
`artifacts/ab4_*.png`; revert `sim_common.py`'s
`disable_right_arm_position_servos` addition (purely additive, does not
affect `run_ab.py`/C0-C3 behavior). Nothing under `scenes/`, `YOR/`, or
`hj_reachability/` was touched.
