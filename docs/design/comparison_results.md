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

## Cooperation — CENTRALIZED joint QP (FL-MPC) vs DECENTRALIZED (v4-C\*)
Architecture is the axis here, and the two are easy to conflate — be explicit:
- **FL-MPC (ours, flagship) = CENTRALIZED / cooperative.** ONE joint QP solves for
  BOTH arms' accelerations together (decision var a=[a_left; a_right]); shared collision
  rows couple them, so the QP choreographs both arms to distribute the avoidance. **0/50 collision.**
- **v4-C\* (ours, baseline) = DECENTRALIZED / reactive.** Each arm solves its OWN
  HOCBF-QP, treating the other arm as a moving obstacle. On the final model it is **collide-or-freeze,
  NOT retunable** (grid below): 60-90% collision when moving, or frozen at 0% when the keep-out is large.
- **NeHMO & HaMMaR (literature) = DECENTRALIZED + certified.** Each arm plans alone against
  a WORST-CASE (adversarial, HJ-Isaacs) neighbor — decentralization forces the learned HJ
  certificate. NOT cooperative.

So the "cooperative" row means CENTRALIZED-joint (only FL-MPC). Decentralization is the papers'
(and v4-C*'s) property, NOT ours — it buys scale + tolerance of uncontrolled/human neighbors at
the cost of worst-case conservativeness. For a single bimanual robot (two arms, one controller we
fully command) centralized cooperation is the matched choice; decentralization solves a multi-
robot/human-shared problem we do not have.

### v4-C\* retune on the final model (task #34) — NOT RETUNABLE (collide-or-freeze)
Swept the decentralized baseline's keep-out margin D_MIN x class-K gain ALPHA, 10 seeds each, final model:
| D_MIN / ALPHA | coll% | qp_infeasible | placed | jerk_rms | fling_worst |
|---|---|---|---|---|---|
| 0.03 / 30 (orig) | 60% | 0.04 | 2.80 | 19094 | 77 |
| 0.09 / 60 | 80% | 0.11 | 1.40 | 20735 | 49 |
| 0.12 / 60 | 70% | 0.11 | 1.60 | 13805 | 22 |
| 0.15 / 60 | 90% | 0.23 | 1.80 | 13384 | 18 |
| 0.12 / 30 | 60% | 0.11 | 2.00 | 33970 | 95 |
| 0.18 / 60 | **0%** | **1.00** | 1.70 | **12.9** | 0.3 |

No config is BOTH collision-safe AND task-completing. Every moving config collides 60-90% (and ALPHA=60
DEstabilizes -> flinging, jerk up to 34k). The only 0% (D_MIN=0.18) has the QP 100% INFEASIBLE -> the arm
FREEZES (near-zero jerk 12.9, placed 1.7 < baseline 2.8): zero collisions purely by not moving. So v4-C*
transitions directly from COLLIDING to FROZEN with no safe+capable operating point -- the decentralized
reactive HOCBF is ill-conditioned for the final geometry (same collide-or-freeze dichotomy as decentralized
worst-case Option B). This is the FAIR v4-C* result: not a tunable single number but structurally NO viable
operating point -- which strengthens the thesis (centralized cooperative FL-MPC, 0/50 + task-completing +
smooth, is the only method that threads the needle). Prior "58%" was one point of this collide regime.
(NOTE: a `W_POS_INT=1e6/5000` "unilateral" weight ablation was ABANDONED — any weight high
enough to make the intruder non-cooperative ill-conditions the QP → numerical blow-up / >10-min
episodes. A clean unilateral ablation needs a structural change, not a weight; the joint-vs-
decentralized result above is the cooperation evidence.)

## HJ safety filter — MEASURED reduced-scope repro of NeHMO/HaMMaR method class [F1, analytic value, no training]
Deployed the analytic single-integrator HJ fail-safe control u* = ubar*grad(clearance)/||.|| (the exact
NeHMO/HaMMaR boundary-avoidance form, NO trained net) on the crossing, RIGHT arm = ego. Metrics via the
IDENTICAL jerk/SPARC harness as the FL-MPC 2612 number. 4 seeds:
| variant | RMS EE jerk | SPARC | collision | cup tilt max |
|---|---|---|---|---|
| FL-MPC (ours) | 2612 | ~0 | 0/50 | held (~6 deg) |
| HJ raw (bang-bang) | 9437 | -12.6 | 4/4 | 99.7 (dropped) |
| HJ + eps-deadzone (blend) | 2809 | -12.6 | 0/4 | 58 (still bad) |

Three findings, all honest, all sharpen the thesis:
1. **Bang-bang CONFIRMED** for the raw fail-safe control: jerk 9437 = 3.6x FL-MPC, above the 8280 bang-bang
   reference; SPARC -12.6 vs ~0. (Theory: Hamiltonian linear in u -> saturated argmax -> switching.)
2. **BUT smoothness alone is a WEAK claim** — an eps-deadzone (i.e. making it CBF-QP-like) collapses jerk
   3.4x to 2809 ~ FL-MPC AND reaches 0/4 collision. "HJ is jerky" holds only for the RAW bang-bang form;
   blending fixes it. DO NOT sell smoothness as the sole differentiator.
3. **The ROBUST differentiators F1 exposes instead:**
   (a) **Constraint-blindness:** arm-arm HJ drops/tilts the cup 58-100 deg even when smooth — same failure as
       OSCBF-collision-only (40 deg). Baking the cup constraint into the QP is the real edge, independent of jerk.
   (b) **Second-order realizability:** the kinematic single-integrator HJ law assumes velocity is set instantly;
       on the torque-controlled arm it chatters and PENETRATES (4/4 collide, h_min -0.06). Our feedback-linearized
       2nd-order MPC with an anticipatory horizon does not. F2 (trained time-to-collision BRT with lookahead)
       tests whether a proper value fixes the penetration or if it is fundamental.
Caveat: analytic signed-distance value, no lookahead, arm-arm only. F2 (trained DeepReach BRT) is the faithful test.
Files: hj_reachability/analytic_safety_filter.py (F1).

## Literature peers `[verified vs PDF, fetched 2026-09-16]` (their code not run in-repo)
Both = **DECENTRALIZED, adversarial (HJ-Isaacs min-max), self-supervised DeepReach** value
functions. Constraint is HAND-SPECIFIED (not learned): HaMMaR writes it as smooth equality
manifolds C(x)=0, e.g. cup-up = orientation_z(x)=[0,0,1]^T. Hamiltonian H=max_u min_d <gradV, f(x,u,d)>
(neighbor = worst-case disturbance d). Value learned via DeepReach loss = boundary term |V-l| at t=T
+ lambda*|HJ PDE residual|; no demos, no RL.
- **HaMMaR** (2511.03591v2): the CLOSEST peer — the one prior work that also carries a task
  constraint (manifold-constrained HJR). Cup-holding, 2xUR5 (sim + real UR5 hardware):
  **82% success / 2% collision / 0.10+-0.09 s per step (~10 Hz) / path 7.7+-3.1 rad**; degrades
  to 5 arms (57/100). Real UR5: 100% over 5 trials. Enforces C(x)=0 as an EQUALITY (cup exactly
  upright) vs our inequality cone (+-~6 deg tube). Compute (~100 ms) is ~2.5x SLOWER than our ~40 ms.
- **NeHMO** (2507.13940): headline is SCALING (2->3->5 arms, 12/18/30-DoF) + a decentralized
  receding-horizon opt that infers each neighbor's worst-case action. Dual-UR5:
  **92% success / 2% collision / 36+-35 ms (~25 Hz) / path 8.5+-4.6 rad**; 5 arms 66%/8%. No task
  constraint (pure collision avoidance) — out of our constrained regime, but the scaling bar we concede.
  Compute (~36 ms) is on par with ours; runs on GPU-trained net, we run model-only on CPU.
Note: path length (rad) is an EFFICIENCY proxy, NOT smoothness — neither paper reports jerk; the
HJ-optimal policy is ~bang-bang at the boundary (cf. our C1/C2: bang-bang = 60x jerk, drops cup).

## Summary matrix
| method | architecture | avoids collision | holds cup (constraint) | cooperative | smooth | real-time |
|---|---|---|---|---|---|---|
| OSCBF collision-only | decentralized reactive | ✓ 0/50 | ✗ 40° tilt | ✗ | reactive | ✓ ~500 Hz |
| OSCBF + cup-CBF | decentralized reactive | ✓ | ✓ 8° | ✗ | reactive | ✓ |
| bang-bang (C1) | — | ✓ | ✗ dropped | — | ✗ 60× jerk | — |
| v4-C\* | **decentralized** QP | ✗ collide-or-freeze\* | (task) | ✗ | mid | ✓ |
| NeHMO `[verif]` | **decentralized** HJ-Isaacs | ✓ 2% | — (no constraint) | ✗ adversarial | ✗ (bang-bang) | ✓ ~25 Hz |
| HaMMaR `[verif]` | **decentralized** HJ-Isaacs | ✓ 2% | ✓ (C(x)=0 equality) | ✗ adversarial | ✗ (bang-bang) | ~10 Hz |
| **FL-MPC (ours)** | **CENTRALIZED joint** | ✓ 0/50 | ✓ (cone tube) | ✓ joint | ✓ jerk-reg | ✓ ~25 Hz |

\* v4-C\* retuned (D_MIN x ALPHA grid, task #34): NO config both safe and task-completing — collides 60-90%
when moving, or (D_MIN=0.18) freezes at 100% QP-infeasible with 0% collision. Not retunable on the final model.

## Caveats (must carry into the paper)
- Cup is kinematically pinned (no true finger grasp), so **cup tilt == wrist tilt** and tilt is
  the clean discriminator; **spill is confounded** (loose-ball ejection artifact) — do not use it.
- OSCBF is decentralized/reactive (≈ v4-C\* class), faster by construction; its guarantee is a
  standard CBF, NOT a learned-HJ certificate — so it does NOT substitute for NeHMO/HaMMaR.
- Sim-only; our safety is an anticipatory HOCBF heuristic (collision rows slackened on
  infeasibility), NOT a formal CBF/HJ certificate.
- HaMMaR/NeHMO numbers + method facts (decentralized, adversarial HJ-Isaacs, DeepReach,
  hand-specified C(x)=0) VERIFIED against the arXiv HTML PDFs (fetched 2026-09-16). Still not run
  in-repo / not head-to-head on our benchmark — they are POSITIONED on shared axes, different
  robot (UR5) and task, so do not claim same-benchmark superiority.
- Architecture is the sharp axis: ours (FL-MPC) is the ONLY centralized/cooperative method; all
  baselines + both papers are decentralized. This is matched-to-problem (one bimanual robot we
  command), not a universal-dominance claim.
