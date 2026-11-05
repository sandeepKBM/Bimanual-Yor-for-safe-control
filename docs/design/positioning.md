# FL-MPC — positioning & rationale (the conceptual spine)

> Status (2026-11-05): HISTORY: Sept 2026 positioning of the FL-MPC cup-collect work. Not current; claims and novelty statements belong to the user (docs/paper/claims_2026-11-02.md).

Durable record of the WHY behind the method and its comparison. Measured numbers live in
`comparison_results.md`; the baseline lineup in `baselines.md`. This file is the argument.

## 1. Problem class (what we actually solve)
CENTRALIZED, cooperative control of ONE bimanual robot **we fully command**: avoid arm-arm/self
collision WHILE holding a cup upright (a task-manifold / orientation constraint), SMOOTHLY (low jerk),
in real time. This is a single-robot, both-arms-commanded problem — NOT decentralized multi-robot,
NOT adversarial.

## 2. Architecture — one MPC vs their layered stack
NeHMO/HaMMaR are a 3-layer stack; we fuse the top two into one online solve.

| layer | NeHMO / HaMMaR | Ours (FL-MPC) |
|---|---|---|
| high | offline-TRAINED HJ value V_theta (safety + manifold certificate, GPU, per-system) | thin task layer (which cube; left-arm "move away") — just goals |
| mid | online LOCAL trajectory optimizer: min goal-cost s.t. V>=eps (value as a fence) | **ONE QP fusing task + collision (HOCBF rows) + cup-upright (hard rows) + jerk penalty + torque box**, 160 ms horizon, re-solved every step |
| low | robot's stock velocity SERVO (tracks; hidden filtering/damping) | feedback linearization tau = M(q)a + c: EXACT computed torque, q_ddot = a (transparent, no smoothing) |

Key: we collapse their high+mid into one QP. The cup constraint is baked in as **QP rows**, not a
pre-trained value manifold — so no offline training, retargetable by editing numbers. Because FL is
exact, the smoothness we plan is the smoothness that executes (measured at the COMMAND level, not
flattered by a stiff industrial servo like their UR5).

## 3. Planning or control? -> model-predictive CONTROL with embedded planning
MPC lives on the planning/control line: it solves a short-horizon trajectory optimization each cycle
(planning flavor) but executes only the first action and re-solves (control). Solve-a-trajectory ONCE
= planning; solve-it-repeatedly-closed-loop = control. We do the latter. We fold what is usually a
separate planning stage INTO the control loop — no plan-then-track handoff. The heart is control; the
low level (FL) is control; only the thin task sequencer is (trivial) planning.

## 4. The infinite-horizon HJ guarantee: real theorem, approximate in deployment
The "safe for all future time under worst-case" guarantee holds EXACTLY only under: (1) exact value
V=V*, (2) exact dynamics model, (3) correct disturbance bound D, (4) the safe control applied exactly
at the boundary, continuous-time, realizable, (5) full exact state, (6) the certified quantity is the
one you care about. In a LEARNED, DISCRETE, MODEL-APPROXIMATE, PLANNER-WRAPPED, PROXY-CONSTRAINED
deployment (what NeHMO/HaMMaR actually are) every assumption is only approximate, so the guarantee
reduces to an **approximate safety prior with an empirical eps-margin**. Concrete breakages:
- value is a neural approx (L1 ~0.01), worst near the boundary where it matters;
- model gap: they train a kinematic single-integrator — **F1 showed the single-integrator law PENETRATES
  on the 2nd-order torque arm**;
- disturbance bound voids if a real neighbor exceeds ubar;
- discrete + planner-wrapped: they use V>=eps as a constraint, and eps>0 is their own admission the
  V=0 boundary isn't trusted;
- proxy: HaMMaR certifies cup-axis-within-X-deg (geometry), NOT no-spill (slosh).
**Residual real benefit (not decorative):** the value encodes infinite-horizon info -> inevitable-
collision / TRAP detection, which our finite (160 ms) horizon genuinely lacks. We concede this; we
mitigate with the anticipatory HOCBF horizon + graceful fallback but do NOT formally guarantee it.

## 5. Manifold constraint vs our hard QP rows (first principles)
The "cup upright" condition is HAND-SPECIFIED in both approaches (not learned). HaMMaR writes it as a
smooth EQUALITY manifold C(x)=orientation_z=[0,0,1]^T and bakes it into a trained HJ value offline
(DeepReach: self-supervised on the HJ PDE residual). We write the SAME condition as 4 linear INEQUALITY
rows (an effective-gravity cone / +-~6 deg tube) in a live QP — no training. (A single cos(tilt) barrier
is degenerate at upright, zero gradient, hence 4 facets.) Theirs = certified but offline+conservative;
ours = heuristic but cheap, exact-linear over the horizon, retargetable.

## 6. HJ safety control is bang-bang (measured, F1)
Control-affine dynamics + convex control set -> Hamiltonian LINEAR in u -> argmax on the boundary of U
-> saturated, switching control -> high jerk (Pontryagin). NOT because V is jagged (a neural V is
smooth) — because the argmax saturates. F1 measured: raw HJ jerk 9437 vs FL-MPC 2612 (SPARC -12.6 vs
~0), drops the cup, AND penetrates (4/4) on the 2nd-order arm. An eps-deadzone recovers smoothness
(2809 ~ FL-MPC) and 0/4 collision but stays cup-blind (58-100 deg tilt). Lesson: smoothness ALONE is a
weak differentiator (a deadzone/CBF-QP fixes it); the robust differentiators are constraint-baked-in
and second-order realizability. See comparison_results.md.

## 7. Honest strategic assessment (no overclaiming)
Our CORE METHOD is not novel: feedback-linearized MPC + HOCBF collision rows + cooperative dual-arm QP
+ jerk regularization are all established. Our contribution is the COMBINATION on this specific problem
+ the empirical study = an application/systems (workshop) contribution, NOT a new theory. NeHMO/HaMMaR
have a genuinely more novel core (manifold-constrained HJ reachability learning) + an approximate formal
certificate + real hardware — but they solve a DIFFERENT problem (decentralized, adversarial, multi-
robot). We occupy a MATCHED niche, not a dominance claim. Differentiators that survive scrutiny:
(a) constraint baked in (collision-only filters — OSCBF, arm-arm HJ — drop the cup 40-100 deg);
(b) second-order realizability (kinematic HJ penetrates on a torque arm);
(c) cooperation (decentralization FREEZES here — Option B: worst-case tube blows up to 0.5-5 m >> the
    0.1-0.4 m arm separation -> near-100% QP-infeasible -> brake/freeze);
(d) smoothness we OPTIMIZE and MEASURE (they never report jerk/SPARC).
Frame the paper as "for a bimanual robot you command, a training-free cooperative smooth constrained QP
is the right simple tool, and here is why the decentralized-HJ machinery is unnecessary AND constraint-
blind here" — honest and defensible.

**GUARDRAIL — the constraint is load-bearing; never pitch this as arm-arm collision avoidance.** For
JUST arm-arm collision (drop [C] cup + [D] jerk) we have NO contribution: it is a solved, crowded problem
(OSCBF 0/50 == FL-MPC 0/50, measured; also HJ / CBF-in-MPC / reactive all ~0), our combination is not
novel there ([A]+[B]+[E] alone is covered by OSCBF + cooperative dual-arm MPC), AND decentralized methods
are arguably BETTER for pure collision (scale + certificate; cooperation buys ~nothing when one arm can
just yield). The cup/orientation constraint is what makes the problem non-trivial (a collision-only filter
BREAKS it), forces genuine cooperation (both yield WHILE holding upright), and makes the integration novel.
Pitch = CONSTRAINED cooperative bimanual manipulation; collision avoidance is table-stakes, not the claim.

## 8. The differentiated next step: a LEARNED slosh/spill constraint
Everyone — them AND us — uses a GEOMETRIC tilt proxy for "don't spill." Real spill is slosh dynamics
(acceleration, jerk, fill level), which NOBODY models. A learned world model that predicts ACTUAL spill,
used as a constraint inside the cooperative smooth MPC, attacks an unoccupied gap and is genuinely novel.
Builds on the WM-requirement ladder (analytic spill ceiling ~0.80-0.86 AUROC; evidence a compact slosh
latent is needed). Principle: **learn where analytics FAIL (spill), not where they suffice (collision/
tilt).** Adopting DeepReach to re-learn the collision safe set would just make us a follower on their turf.
The F2 HJ training is EVIDENCE (measuring the HJ class on our task), not a pivot.

**Concrete slosh-model recipe (lit hunt 2026-09-16):** Do NOT use a full slosh-PINN in the loop — every
slosh/free-surface PINN found is either fixed-excitation (NOT conditioned on container motion = our control
var) or a ~6 Hz / thousands-of-DOF particle surrogate: too slow/high-D for a 200 Hz constraint.
- IN THE LOOP: a spherical/planar PENDULUM (or trammel) analog of the 1st slosh mode — 2-4 states, forced by
  cup acceleration, analytic Jacobians, differentiable at 200 Hz. Spill constraint = surface-tilt <=
  theta_max(fill, geometry). Precedent: Muchacho pendulum-in-QP (IROS 2022, arXiv 2210.12614); Biagiotti
  input-shaping (2017/18); Rossi constraint-based (RA-L 2018).
- OPTIONAL learned RESIDUAL (neural-ODE or SINDy correction) on the pendulum to extend past its ~15 deg
  linear-validity box (the exact limit SFRRT attacks). Still tiny + differentiable. For a learned in-loop
  model prefer Koopman/SINDy over a PINN (explicit, control-oriented, differentiable).
- OFFLINE ONLY: PINN / neural-SPH / VOF-CFD to generate (cup-motion -> spill) data and IDENTIFY the pendulum
  parameters (a spacecraft EMM-via-PINN paper does exactly this, IFAC 2025).
**Scoop check (this direction):** NO published work puts a learned/physics-informed slosh model inside an MPC
CONSTRAINT for a carried open container -> appears NOVEL. In-loop precedents are ANALYTIC pendulum-in-QP
(Muchacho/Biagiotti/Rossi); the one learned model (SFRRT, Abderezaei/Roncone IROS 2024, arXiv 2408.00215) is
an OFFLINE transformer classifier + RRT* (beats the pendulum tilt-box but non-differentiable, not in a
controller). MUST position against: Muchacho 2210.12614/2402.05197, Biagiotti 2017/18, Rossi 2018, SFRRT 2408.00215.

## 9. Related work & novelty (prior-art hunt, 2026-09-16)
Our 5 distinguishing ingredients: [A] centralized cooperative bimanual (both arms co-optimized, one
problem); [B] collision as (HO)CBF constraints INSIDE the MPC; [C] hard object-orientation-manifold
(upright / effective-gravity cone); [D] explicit jerk/smoothness objective; [E] feedback-linearized
single unified QP-MPC (accel decision var, tau=Ma+c).

**SCOOP VERDICT: partially anticipated in pieces, NOT scooped as an integrated method.** No surveyed
paper unifies [A]+[B]+[C]+[D]+[E]; every closest one misses >=2. Nearest neighbors + what they miss:
1. **OSCBF** (arXiv 2503.06736) — closest FL+CBF-QP; INSTANTANEOUS filter (no MPC horizon), single arm,
   no upright cone. Already in our benchmarks ("OSCBF breaks our cup constraint").
2. **"Keep it Upright"** (Heins & Schoellig, RA-L 2023, arXiv 2305.17484; utiasDSL/upright) — CLOSEST
   orientation-in-MPC: waiter's-problem balance (effective-gravity/friction-cone HARD constraint) +
   collision in one MPC. But SINGLE arm, non-prehensile balance, plain-distance (no HOCBF), not FL-torque.
3. **DART** (arXiv 2604.17833, 2026) — CONCURRENT bimanual object-transport MPC; but BILEVEL (tilt-NMPC +
   impedance-QP), non-prehensile sliding, ZERO collision/CBF. Highest residual risk (concurrent+bimanual);
   differentiate on unified-QP + HOCBF + prehensile.
4. **Nikou/Verginis** (arXiv 1705.01426, MED 2017) — cooperative-NMPC lineage (co-grasped object to a
   pose); no HOCBF, no upright cone, no FL single QP, no jerk.
5. **Zeng/Zhang/Sreenath MPC-CBF** (arXiv 2007.11718, ACC 2021) — the HOCBF-in-MPC framework ANCESTOR we
   instantiate. The user-flagged arXiv 2405.12408 (CBF-guided MPC, adaptive decay rates, single UR5,
   obstacle avoidance) sits in this cluster — establishes CBF-in-MPC as prior art; its adaptive-decay
   idea is complementary/adoptable for our HOCBF rows.

**TIGHTEST DEFENSIBLE "FIRST":** the first training-free, centralized feedback-linearized MPC that
co-optimizes BOTH arms of a bimanual robot in a SINGLE real-time QP jointly enforcing (i) anticipatory
HOCBF arm-arm/self-collision constraints and (ii) a hard object-upright (effective-gravity-cone)
orientation-manifold constraint, with a jerk objective, torque box, and graceful slack-QP fallback.
**Conservative fallback:** first to combine HOCBF collision + object-orientation-manifold hard constraint
inside one FL bimanual MPC-QP (that intersection is unoccupied).
**DO NOT claim novelty on any ingredient in isolation** — HOCBF-in-MPC (Zeng/Sreenath), FL-MPC for arms,
upright-in-MPC (Keep it Upright), cooperative dual-arm NMPC (Nikou), CBF-in-FL-QP (OSCBF) are each owned.
The novelty is strictly the INTEGRATION. CBF-in-MPC is UNAMBIGUOUSLY established prior art on two levels
our claim must sit ABOVE: (1) framework = Zeng/Zhang/Sreenath ACC 2021 (2007.11718); (2) manipulator +
training-free + real-time (+ adaptive decay rates as QP decision vars) = Liu et al. 2024 (2405.12408, single
UR5, external-obstacle avoidance). Therefore explicitly do NOT claim any of: "first CBF-in-MPC", "first
CBF-MPC for manipulators", "first training-free real-time CBF-MPC on an arm", "first adaptive-decay CBF-MPC".
Liu 2405.12408's adaptive-decay-rate idea is COMPLEMENTARY/ADOPTABLE for our HOCBF rows (could ease the
QP-infeasibility cliff) — cite as adoptable, it is their contribution not ours.
Must-cite-and-differentiate: OSCBF (2503.06736), Keep it Upright (2305.17484), DART (2604.17833),
Nikou/Verginis (1705.01426), Zeng-Sreenath (2007.11718), Liu et al. (2405.12408).
