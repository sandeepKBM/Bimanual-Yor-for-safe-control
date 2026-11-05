# Head-to-head baselines for FL-MPC (the RIGHT comparison set)

> Status (2026-11-05): HISTORY: Sept 2026 baseline set for the FL-MPC cup-collect comparison (Piper-era arms). Not current: current claims are in docs/paper/claims_2026-11-02.md; FL-MPC now lives in world_model/v2/flmpc/.

Our problem class: **CENTRALIZED, cooperative control of ONE bimanual robot we fully command** —
avoid arm-arm/self collision WHILE holding a cup upright (orientation-manifold constraint), SMOOTHLY,
in real time. Baselines must be the SAME class. NeHMO/HaMMaR (decentralized, adversarial, multi-robot)
are therefore RELATED-WORK POSITIONING, not head-to-head. (See comparison_results.md, comparison-framing memory.)

Two deep-research sweeps (reactive-control family + optimization/MPC family), 2026-09-16, produced this.

## RUNNABLE head-to-head baselines (port to our MuJoCo Piper), priority order
| # | baseline | family | constraint style | axis it stresses | port effort | code |
|---|---|---|---|---|---|---|
| 1 | **MuJoCo MPC (MJPC)** | sampling/iLQG MPC | collision+cup = COST terms | ALL at once; the fair "strong real-time MPC without our HOCBF/FL structure" foil | **LOW** (native MuJoCo) | Apache-2.0, DeepMind |
| 2 | **CollisionIK** (Rakita) | reactive IK | orientation+collision = SOFT weighted; explicit velocity/accel/**jerk** min | smoothness + collision + constraint (soft) — the kinematic no-dynamics foil | LOW–MOD (MuJoCo port exists) | MIT |
| 3 | **TSID / Stack-of-Tasks** | whole-body QP (torque) | cup-upright as a HARD inequality; self-collision inequalities | CONSTRAINT (hard) at dynamics level; best Piper torque match | MOD (Pinocchio, no MuJoCo demo) | BSD-2 |
| 4 | **cuRobo (MPC mode)** | GPU MPPI | constraints = COST (authors note: no guarantee) | smoothness (min-jerk B-spline) + real-time (500 Hz); contrast to our HARD rows | MOD (CUDA, runs alongside MuJoCo) | Apache-2.0 |
| 5 | **OSCBF** (already ported) | CBF-QP filter | hard CBF; collision-only by default | the "collision-only filter BREAKS the cup constraint" anchor (40 deg tilt) | DONE | open |
| 6 | **APF** | potential field | none | floor baseline; local-minima failure motivates a principled method | trivial | reimpl |

**Primary recommendation:** run FL-MPC head-to-head vs **#1 MJPC (first & hardest)**, **#2 CollisionIK (soft-constraint foil)**, **#3 TSID (hard-constraint foil)**; keep #5 OSCBF (done) + #6 APF (floor). Honest bar (from the MPC sweep): *if FL-MPC does not clearly beat a tuned MJPC on collisions-at-equal-smoothness, the method claim is weak* — so MJPC is the decisive experiment.

Design note: IK family (CollisionIK/RangedIK/RelaxedIK) treats orientation+collision as SOFT (tradeable → a spill-failure mode to exploit); TSID/OSCBF/cuRobo differ on hard-vs-cost. Pair ONE soft peer (CollisionIK) vs ONE hard peer (TSID or OSCBF) to cover both framings of the cup constraint.

## POSITIONED-ONLY (related work; cite, don't benchmark)
- **NeHMO** (2507.13940), **HaMMaR** (2511.03591): decentralized adversarial learned-HJ (DeepReach); HaMMaR is the closest (adds cup manifold C(x)=0) but wrong regime. No public code. `[verified vs PDF]`.
- **WAITR / Serving Time** (ROAHM): real-time certifiably-safe non-spill transport, single arm, HAS code — the strongest non-spill anchor.
- **DART** (ICRA 2026): centralized dual-arm tray-tilt in MuJoCo, 500 Hz — closest task-level analog; code availability unconfirmed → RUNNABLE-if-code else positioned.
- **Nikou et al.** centralized cooperative NMPC (collision+singularity+orientation, proofs): closest FORMULATION match; assumes rigid closed-chain grasp (we don't); no code.
- **Bimanual GPU-MuJoCo MPPI / PRISM**: centralized bimanual sampling MPC with jerk-projected smoothness; code unconfirmed. RangedIK, RMPflow, mc_rtc, Crocoddyl/OCS2, ALTRO, TrajOpt/CHOMP/STOMP, GOMP-FIT: positioned across the two families.

## HJ-class evidence (our own reduced-scope repro, task #5)
Since NeHMO/HaMMaR have no code, we reproduce the MECHANISM on our 12-D two-arm case (DeepReach port
~90% done in hj_reachability/): F1 = analytic-BRT saturated safety filter (zero GPU) to measure jerk;
F2 = trained supervised-bootstrap value (westeros prep -> ilab GPU) deployed as filter + V>eps QP.
Purpose: turn "HJ safety control is bang-bang / non-smooth" from asserted -> measured, as positioning evidence.
