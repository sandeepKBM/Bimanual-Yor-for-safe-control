# Comparison results — FL-MPC vs baselines on the CONSTRAINED problem

Our problem is the **constrained** one: cooperative bimanual arm-arm collision avoidance
**while holding a cup upright** (a task/manifold constraint). The comparison lives on THIS
problem, not a stripped collision-only case. Thesis: *collision avoidance alone is already
solved (ours and OSCBF both 0/50); the contribution is doing it under the constraint,
cooperatively, and smoothly.*

All numbers below are MEASURED in-repo unless marked `[positioned]` (paper numbers, unverified).

## Headline: a collision-only filter (OSCBF) BREAKS our constraint
OSCBF (Morton & Pavone, IROS 2025), the public CBF-QP safety filter, ported faithfully into
OUR MuJoCo on the real Piper (kinematics verified 0.000 mm; scratch port at
`/common/users/ss5772/oscbf_baseline/piper_port/`). Cup carried through the crossing, 10 seeds:

| condition | arm-arm collisions | **cup tilt max / mean** |
|---|---|---|
| no filter | 9/10 | 74.6° / 72.8° |
| **OSCBF collision-only (EXP-A)** | **0/10** | **43.1° / 40.2°** ← breaks the constraint (7× past 5.7°) |
| **OSCBF + cup-upright CBF (EXP-B)** | **0/10** | **9.3° / 7.6°** |

EXP-A: OSCBF makes the *arms* safe but the *cup* unsafe — a collision-only filter is
structurally blind to the constraint. EXP-B: add the constraint and it holds (~8°) — proving
the 40° is the missing constraint, not the reach. (Technical note: the naive `cos(tilt)`
barrier is degenerate at upright — zero gradient — so 4 linear rows are needed.)

## Collision is NOT the differentiator (honest)
On the pure arm-arm crossing (no cup), 50 seeds:
- **OSCBF**: 0/50 collisions, worst clearance +0.51 cm, ~500 Hz (reactive filter).
- **FL-MPC (ours)**: 0/50 collisions (full task benchmark), ~25 Hz (160 ms-horizon MPC).

Both solve collision. OSCBF is ~20× faster **by construction** (a 1-step filter, not a horizon
MPC) — we do NOT claim to avoid collisions better. Our edge is the constraint + cooperation +
smoothness below.

## Smoothness — our jerk-regularized QP vs bang-bang (C1/C2, existing)
| controller | RMS EE jerk | grasped object |
|---|---|---|
| C1 bang-bang (hard-switch HJ-style) | **8280** | **dropped** (102° rot) |
| C2 smooth HOCBF-QP | **139** (~60× smoother) | held |

(FL-MPC vs v4-C\* on the full task: jerk 2612 vs 12825, ~5×. Note: an FL-MPC `W_JERK=0`
weight ablation did NOT separate — the realized closed-loop jerk is dominated by replanning —
so the smoothness claim rests on the bang-bang *controller* comparison C1/C2, which is stronger.)

## Cooperation — joint QP vs decentralized
- **FL-MPC** (joint QP, both arms co-optimized): **0/50 collision**.
- **v4-C\*** (decentralized reactive HOCBF-QP): **58% collision**.

The joint cooperative formulation lets the QP distribute avoidance across both arms.
(NOTE: a `W_POS_INT=1e6/5000` "unilateral" weight ablation was ABANDONED — any weight high
enough to make the intruder non-cooperative ill-conditions the QP → numerical blow-up / >10-min
episodes. A clean unilateral ablation needs a structural change, not a weight; the joint-vs-
decentralized result above is the cooperation evidence.)

## Literature peers (positioned, code not run)
- **HaMMaR** (2511.03591) `[positioned]`: the closest — a manifold-constrained HJ value +
  2-arm manipulation (cup-upright); the one prior work that also carries the constraint.
  Adversarial/unilateral; ~100 ms.
- **NeHMO** (2507.13940) `[positioned]`: neural-HJR multi-arm; headline is scaling (2→5 arms),
  out of our fixed-2-arm regime; adversarial/unilateral; ~36 ms.

## Summary matrix
| method | avoids collision | holds cup (constraint) | cooperative | smooth | real-time |
|---|---|---|---|---|---|
| OSCBF collision-only | ✓ 0/50 | ✗ 40° tilt | ✗ | reactive | ✓ ~500 Hz |
| OSCBF + cup-CBF | ✓ | ✓ 8° | ✗ | reactive | ✓ |
| bang-bang (C1) | ✓ | ✗ dropped | — | ✗ 60× jerk | — |
| v4-C\* | ✗ 58% | (task) | decentralized | mid | ✓ |
| **FL-MPC (ours)** | ✓ 0/50 | ✓ | ✓ joint | ✓ | ✓ ~25 Hz |

## Caveats (must carry into the paper)
- Cup is kinematically pinned (no true finger grasp), so **cup tilt == wrist tilt** and tilt is
  the clean discriminator; **spill is confounded** (loose-ball ejection artifact) — do not use it.
- OSCBF is decentralized/reactive (≈ v4-C\* class), faster by construction; its guarantee is a
  standard CBF, NOT a learned-HJ certificate — so it does NOT substitute for NeHMO/HaMMaR.
- Sim-only; our safety is an anticipatory HOCBF heuristic (collision rows slackened on
  infeasibility), NOT a formal CBF/HJ certificate.
- HaMMaR/NeHMO numbers are summarizer-extracted and UNVERIFIED — re-check PDFs before citing.
