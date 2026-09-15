# Comprehensive validation — FL-MPC vs V4-C* (N=50, final full-fidelity model)

Run 2026-09-15 on the FINAL model: proper gripper housing+jaws, arm↔arm, arm
self-collision, arm↔table/floor, right-arm↔cup, arm↔cube (held-cube runtime toggle),
plus the FL-MPC ego_hold release (left arm moves away, no spring-back). Hard geometry:
crossing, 3 cubes, V_EE_MAX_MS=0.28, cup_jitter 0.04. 50 seeds × 2 controllers, full
episodes. Harness: comprehensive_eval.py (raw rows in results_comprehensive_n50.json).

| controller | n | collision | spill | fling mean / worst (m/s) | h_min worst | jerk_rms mean | placed mean |
|---|---|---|---|---|---|---|---|
| **flmpc** | 50 | **0.0%** | 26.0% | 0.99 / 2.01 | **+0.008** | **2612** | 3.34 |
| **v4c**   | 50 | 58.0% | 96.0% | 7.48 / 77.04 | −0.075 | 12825 | 2.86 |

## Reading
- **Collision 0/50 vs 58%** — FL-MPC never collides across 50 randomized seeds, on the
  stricter full-collision-fidelity model. The graceful slack-QP fallback holds.
- **~5× smoother** (jerk 2612 vs 12825) — the "smooth vs bang-bang" thesis, quantified.
- **Fling bounded** (worst 2.0 m/s = graceful-fallback-limited infeasibility spike) vs
  v4c's **77 m/s** unbounded blow-up (no fallback). The 3 NaN/Inf QACC warnings in the
  log are v4c episodes exploding, not FL-MPC.
- **Clearance never negative** for FL-MPC (+0.008 worst) vs v4c penetration (−0.075).
- Spill 26% (FL-MPC) is with ego_hold released (~10cm dodge sloshes the cup); still far
  below v4c's 96% (its violent motion spills nearly always). Spill remains geometry-
  blind → the WM channel's job, orthogonal to the collision-safety result.

## Caveats
- jerk_rms is a RELATIVE proxy (triple finite-diff of the right-EE path), meaningful
  flmpc-vs-v4c, not an absolute physical jerk.
- Sim-only; not a formal guarantee (anticipatory HOCBF, not a certified CBF/HJ value).
