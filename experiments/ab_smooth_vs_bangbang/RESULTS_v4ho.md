# v4-HO: higher-order smoothness on v4-C's per-step decentralized HOCBF-QP

## 1. What was added

v4-C's smoothness was first-order only (low-pass + rate-limit `TorqueCommandFilter`
AFTER each arm's QP). Added, inside BOTH arms' per-step QPs (`run_ab4ho.py`,
new file, `sc.solve_constrained_box_qp`/`sc.build_weighted_least_squares_qp`
unmodified, `run_ab4.py`/`sim_common.py` untouched):

1. **Slew-rate CONSTRAINT** (hard): `|tau_t - tau_{t-1}| <= RHO_SLEW=20Nm`
   per joint, on the QP's raw decision variable, applied for free by
   tightening the box bounds each step (`lower/upper = clip(tau_prev +- rho,
   +-TAU_MAX)`) -- the existing `psi2>=0` safety row (a_ineq/b_ineq) is
   byte-identical to `run_ab4.py`'s C, so the hard safety guarantee is
   unaffected.
2. **Snap PENALTY** (soft): `W_SNAP=0.02 * ||tau_t - 2*tau_{t-1} + tau_{t-2}||^2`,
   added as a second weighted-least-squares term via the existing
   multi-term `build_weighted_least_squares_qp` interface -- no new solver
   code. Requires 2 extra torque vectors of state per arm (`tau_prev`,
   `tau_prev2`), skipped for the first 1-2 steps (no history). No horizon
   MPC; still one box+linear-inequality QP per arm per step.

## 2. Metrics (A | v4-C | v4-C+HO)

| metric | A (adversarial) | v4-C (baseline) | v4-C+HO |
|---|---|---|---|
| min h [m] | -0.0309 | +0.01586 | +0.01445 |
| collision | True | False | False |
| box dropped | True | False | False |
| box drift trans max | 0.231 m | 0.01582 m | 0.01582 m |
| ego RMS/max EE jerk | 385 / 4840 | 24.54 / 355.9 | **24.54 / 355.9** |
| intruder RMS/max EE jerk | 152 / 3091 | 438.2 / 5628.3 | **433.2 / 5518.4** |
| `hdot` sign-changes/s (chatter) | 12.26 | 0.681 | **0.227 (3.0x lower)** |
| intruder progress vs A | 1.00 | 0.696 | 0.696 |
| QP infeasible frac (ego/int) | 0/0 | 0/0 | 0/0 |
| isolated QP assemble+solve (ego+int) | n/a | 57.44 ms/step | 57.72 ms/step (+0.5%) |

C reproduces RESULTS_v4.md exactly (min_h=0.0159, drift 1.58cm, chatter 0.68,
ego jerk 24.5) -- confirms the harness. Isolated timing (`profile_v4ho.py`,
numba-warmed, excludes the shared `all_pair_barriers` call, identical in both)
isolates exactly what the slew+snap addition costs.

## 3. Does HO help?

**Chatter: yes, substantially** -- 0.68 -> 0.227 sign-changes/s, a further
3x reduction on top of v4-C's already-lowest-of-A/B/C number. **Jerk:
flat-to-slightly-better** -- ego jerk unchanged to 4 significant figures
(24.543->24.545 RMS; the ego was already barely reacting under C), intruder
jerk improves marginally (438.2->433.2 RMS, 5628->5518 max). **Safety/grasp:
fully preserved** -- collision-free, box-retained (drift unchanged to
5 decimal places, since the hard psi2>=0 row is untouched), min_h stays
comfortably positive (0.0159->0.0145, a 1.4mm reduction in margin, not a
regression into unsafety). **Intruder progress: unchanged** (0.696 vs 0.696).
**Real-time: no meaningful extra cost** -- +0.28ms/step (+0.5%) for the QP
solve itself. Note v4-C's *existing* isolated QP cost (57ms/step, dominated
by the Python loop assembling ~625 keypoint-pair rows twice, not by
`solve_constrained_box_qp` itself) already exceeds the 5ms physics DT -- a
pre-existing v4-C characteristic (see RESULTS_v6.md's MPC comparison), NOT
something HO worsens; HO's own marginal cost is negligible.

**Net: HO is a clean, free win here** -- lower chatter, flat-or-better jerk,
unchanged safety/grasp, negligible extra compute.

## 4. Figures

`artifacts/ab4ho_hdot_chatter.png`, `ab4ho_ee_jerk.png`, `ab4ho_box_drift.png`.

## 5. Caveats

- `RHO_SLEW=20Nm` and `W_SNAP=0.02` are single hand-picked values, not swept.
- Single scenario/seed, same as v4.
- The "avg_step_time_ms" field inside `logs/v4ho_summary.json` mixes
  whole-run wall time (A/C, via reusing `run_ab4.run_condition` unmodified)
  with an isolated QP-only window (C_HO) -- NOT apples-to-apples; use
  `profile_v4ho.py`'s numbers (Sec. 2 table) for the real solve-time
  comparison instead.
- HO state (`tau_prev`/`tau_prev2`) is the QP's raw pre-filter output, not
  the post-filter command; the downstream `TorqueCommandFilter` is
  byte-identical to v4-C in both rows.

## 6. Blockers

None.

## 7. Rollback

Delete `run_ab4ho.py`, `profile_v4ho.py`, `make_plots_v4ho.py`,
`RESULTS_v4ho.md`, `logs/v4ho_{A,C,C_HO}.npz`, `logs/v4ho_summary.json`, and
`artifacts/ab4ho_*.png`. Nothing under `run_ab4.py`, `sim_common.py`,
`scenes/`, `YOR/`, or `hj_reachability/` was touched.
