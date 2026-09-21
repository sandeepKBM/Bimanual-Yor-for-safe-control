# Direction: sim-synthesized -> real-validated -> active-refined safe set as a teleop filter

A candidate BIGGER paper beyond the FL-MPC workshop. Prior-art hunt 2026-09-18. Distinct from the
FL-MPC controller work; reuses our HJ/DeepReach safe-set (task #5) + teleop panel + FL-MPC/CBF filter.

## The pipeline
- **A. Synthesize** the safe operating bounds FROM SIM, on demand, given a scenario — a level set /
  reachable safe set / CBF-defined region (scenario-conditioned).
- **B. Validate** that boundary with TARGETED real-arm experiments (sim-to-real validation of the certificate).
- **C. Actively refine** — concentrate real trials on the level-set boundary regions of HIGHEST epistemic
  uncertainty / largest sim-to-real gap.
- **D. Deploy** the resulting rigorously-defined, validated boundary as a real-time TELEOP shared-control filter.

## OUR instantiation (refined 2026-09-18) — MPC-defined bounds + LLM orchestrator + multi-object
- **Stage A is NOT HJ reachability — it is OUR FL-MPC.** The operating bounds ARE the FL-MPC hard-constraint
  set for the given object (collision HOCBF + object-specific constraint e.g. cup-upright cone + torque box);
  the QP feasible region IS the "level set." No HJ value to train — the controller already defines the bounds.
- **An LLM is a HIGH-LEVEL orchestrator that PROPOSES a bound then PROVES it via multi-fidelity experiments.**
  It (1) proposes a bound (a hypothesis about the operating boundary for the object), (2) designs a validation
  campaign to rigorously prove/refute it, (3) allocates each experiment to SIM or REAL based on the UNCERTAINTY
  OF THAT INFORMATION SOURCE in that region (cheap sim where sim is trustworthy; expensive real where the
  sim-to-real gap or safety stakes are high) = multi-fidelity active experimental design / value-of-information.
  Generalizes across a few dozen OBJECTS (LLM reasons about object properties -> which constraints apply:
  open cup->upright; sealed box->none). Output per object = a validated operating-bound boundary -> teleop filter.
- **Two rigor anchors:** (i) rigor comes from the VALIDATION, not the LLM — LLM proposes, the multi-fidelity
  campaign proves; LLM stays OUTSIDE the safety-critical path. (ii) "prove" = STATISTICAL (conformal/PAC,
  confidence 1-delta) on a finite sim+real sample, NOT a formal proof — claim "statistically certified", not "proven".
- **LLM value-add vs plain multi-fidelity BO (a mature area):** proposing good bounds across many objects from
  world knowledge + driving MF allocation in natural language. The MF sim-vs-real machinery itself is not novel.

## LLM-orchestration prior art, failure mode, framing (hunt 2026-09-18)
SCOOP: UNCLAIMED — no LLM-orchestrated sim+real loop deriving/validating a per-object OPERATING-BOUND for a teleop
filter. Closest neighbors:
- **Bajcsy et al., "Updating Robot Safety Representations Online from NL Feedback" (arXiv 2409.14580, 2024)** —
  CLOSEST: language infers context constraints EXPLICITLY for SPILLS + FRAGILE objects (our domain) and updates an
  HJ safety controller online. But language-DRIVEN (not experiment-orchestrated), no sim-then-real validation,
  autonomous control not teleop. MUST-CITE.
- RoboGate (2603.22126, 2026): boundary-focused sim->real validation (our step C) but NO LLM. "someone will combine".
- DrEureka (RSS24): LLM configures sim (reward+DR) for real transfer; exports a POLICY not a safe set.
- Semantically Safe Manipulation (2410.15185): VLM->constraints->CBF at teleop rate; inference only, no cert.
- LLM->constraint priors (single-task, unvalidated): VoxPoser, Language-to-Rewards (LLM->MPC params), Code-as-Policies,
  Text2Reward, Safety-Chip. Field consensus (Safety Guardrails RAL26, BarrierBench 2511.09363): LLM must NEVER be the
  last line of safety; a verifier / physical test gates it (~50% of LLM constraints need correction).

CRITICAL FAILURE MODE + design rule: the dangerous case is a TOO-LOOSE LLM constraint (filled cup called "sealed"
-> filter permits tilt -> spill), because real validation samples INSIDE the claimed-safe region and may never probe
the truly-unsafe region the LLM wrongly excluded. Mitigation: derive the bound CONSERVATIVELY from MPC feasibility
(NOT the LLM); LLM only picks constraint TYPE + targets validation; treat any LLM "no constraint needed" as requiring
the strongest disproof. LLM stays OUTSIDE the certified path.

FRAMING (recommended): certification is the HEADLINE; LLM orchestration is the SCALABILITY enabler (dozens of objects
without per-object human spec-writing) + a FALSIFICATION CASE STUDY. Co-equal "LLM synthesizes constraints" framing
DILUTES (reviewers discount the wrapper); "LLM scales it + here is the loop CATCHING a deliberately-wrong LLM bound"
STRENGTHENS. Highest-value single experiment: demonstrate the loop FALSIFYING an intentionally wrong LLM constraint.
Tightest LLM-piece novelty: {LLM constraint-template selection} x {MPC-derived safe set as exported artifact} x
{LLM-directed real-arm boundary validation} x {teleop-filter consumer} = unclaimed composition.
Effort: template-selection LOW (days); LLM->MPC bound MED (1-2 wk); LLM-targeted real validation MED-HIGH (novel/risky,
hardware-bound); keeping LLM out of certified path MED. Hard gate (as whole direction): needs a physical arm.

## Multi-fidelity / VOI layer — the ACTUAL prover (addendum 2026-09-18)
"Route sim-vs-real by source uncertainty" is a SOLVED, named class: multi-fidelity / multi-information-source BO +
Bayesian experimental design (8+ yrs). Allocator NOT novel: Kandasamy BOCA/MF-GP-UCB (ICML17/AISTATS16), Takeno
MF-MES (ICML20; info-gain-per-cost acquisition = the routing rule), Poloczek misoKG (NeurIPS17; per-region
source-bias-vs-reality = the sim2real gap), Lindley/Rainforth EIG Bayesian experimental design, Marco "Virtual vs
Real" (ICRA17; allocate cheap sim vs expensive real under a gap model — direct robotics precedent), Rai/Antonova
(bipedal). Recent: MF hybrid RL via info-gain (2509.14848), guided MF-BO w/ digital twins (2509.17952).

DIVISION OF LABOR (honest + stronger): **MF-VOI is the ENGINE doing the rigorous work** (routes sim/real, quantifies
uncertainty reduction, guarantees efficient allocation) — off-the-shelf, LLM-FREE. If the paper implies the LLM
decides routing = HYPE (a Knowledge-Gradient/entropy-search acquisition makes that call). LLM does 3 real,
non-substitutable jobs: (1) HYPOTHESIS/BOUND PROPOSAL (defines WHAT the loop tests; VOI is agnostic about what);
(2) informative PRIOR + per-object sim-trust/discrepancy model (rigid sealed box -> sim~=real; liquid slosh/
deformable -> distrust sim -> spend real trials) — measurable IFF it beats an uninformative prior (fewer real
trials); (3) SCALING across the object catalog. This puts the LLM OUTSIDE the safety-critical path: the MF loop +
real trials certify; a wrong/loose LLM bound surfaces as high posterior uncertainty the allocator spends real trials
to refute (= the "who checks the LLM" answer).

NOVELTY seam: [A] LLM proposes bound+template+sim-trust prior -> [B] MF-VOI certifies (routes sim/real) -> [C]
per-object teleop filter. [B] alone = textbook (do NOT claim). [B]-for-boundary-certification (vs scalar optimization)
= modest twist (level-set estimation / failure-boundary; ROBOGATE 2603.22126, active local-validity 2406.07474).
[A]+[B]+[C] + LLM-seeded discrepancy prior feeding a principled MF-VOI certifier = the unclaimed seam.

RECOMMENDED HEADLINE: "certified per-object teleop bounds via multi-fidelity VOI, with an LLM supplying the
hypothesis-bound, constraint template, and informative sim-trust prior that make certification scale to a catalog."
Carrying empirical claims: (a) LLM-seeded prior reduces #real trials vs uninformative; (b) MF-VOI loop CATCHES a
deliberately-wrong LLM bound (falsification demo); (c) certified bounds -> safe teleop filter. Feasibility: MF-VOI
layer MED/low-risk (BoTorch multi-fidelity KG/MES, Emukit -> misoKG/MF-MES). New work = MPC-sim<->real-arm
discrepancy model + level-set (boundary) acquisition. Cost = real-arm trials (which the layer minimizes).
- **Unification:** the uncertain boundary region Stage C should target is exactly the SPILL/TILT constraint
  (our known geometric-proxy weakness) -> this MERGES with the learned-slosh direction (positioning.md §8).
- **HONEST caveat on the LLM layer:** the rigorous core (actively certify a boundary on real hardware ->
  teleop filter) stands on its own WITHOUT an LLM. The LLM buys multi-object generalization + autonomy but is
  ORCHESTRATION, not the safety contribution, and sits in a crowded/hype-heavy area (AI-scientist / autonomous
  labs). Keep the certified boundary as the claim; the LLM is a wrapper. (LLM-orchestration prior-art hunt pending.)

## Per-stage prior art + maturity
- **A (mature/off-the-shelf):** DeepReach (Bansal-Tomlin ICRA21); Parameter-Conditioned Reachable Sets
  (Borquez/Bansal ICRA23, arXiv 2209.14976) = scenario-conditioned on-demand; RBN = Reachability Barrier
  Networks (arXiv 2505.11755, 2025, value IS a differentiable CBF); HJRNO neural-operator BRT (2504.19989).
- **B (mature GP/ensemble; conformal is new):** Fisac "General Safety Framework" GP+HJ maximal safe set
  (arXiv 1705.01292, TAC19; Akametalu CDC14) — closest classical A+B(+C) ancestor; Wang/Borquez/Bansal
  ensemble reachable set validated on TurtleBot (arXiv 2403.05771, 2024). Conformal-prediction safe sets
  (2511.07899 2025; 2604.18482, 2602.03799 2026) = distribution-free statistical validation.
- **C (machinery mature; certification framing is the novel/risky part):** SafeOpt (Sui/Krause ICML15),
  StageOpt (ICML18), GOOSE (NeurIPS19), Transductive Active Learning/ARLET (ICML24), ActSafe (ICLR25).
  ALL expand a safe set to OPTIMIZE a function / learn dynamics — NONE to CERTIFY a reachable-set boundary
  that feeds a teleop filter. RoboGate (2603.22126, 2026) = boundary-focused hardware failure discovery,
  closest in spirit, a possible C building block.
- **D (mature/off-the-shelf):** OSCBF (2503.06736, already in our benchmarks); SPARK teleop safety kit
  (2502.03132, 2026, MuJoCo->Unitree G1); Failure-Aware BIMANUAL teleop via conservative-Q filter
  (2602.01092, 2026); passivity/CBF teleop filters (2102.08630).

## Scoop verdict
The full A->B->C->D loop for a manipulator/bimanual arm does NOT exist as an integrated system. Closest:
Fisac19 (A+B+weak-C, sim, no teleop); Wang24 (A+B on hardware, no active refine, no teleop); SPARK/OSCBF
(D + sim2real, but hand-designed safe set, no active refine); SafeOpt-family (C, but for BO/RL not
certification). NO system does the C->D handoff (actively-validated reachable boundary -> teleop filter).

## Tightest defensible contribution
Active, uncertainty-guided experimental design that selects real-robot trials ON the boundary of a
scenario-conditioned SIM-SYNTHESIZED reachable/CBF safe set, to VALIDATE and locally CORRECT it where the
sim-to-real epistemic gap is largest, deployed as a BIMANUAL teleop shared-control filter.
**Do NOT claim** in isolation: learned reachability, GP/conformal safe sets, safe BO, CBF teleop filters,
value-fn-as-filter — all pre-existing. Novelty = the integration + Stage-C-FOR-CERTIFICATION.

## Feasibility (for us)
- A + D = weeks; we are ~half-built (F2 DeepReach safe-set = A; FL-MPC/CBF filter + teleop panel = D).
- B = weeks (empirical); statistical-certification framing (conformal) is extra.
- C + integration = the multi-month core (boundary epistemic-uncertainty estimate + boundary-focused
  acquisition + high-D scalability; bimanual 14+ DoF hits DeepReach's curse of dimensionality -> reduced/
  task-space/latent safe set or per-arm decomposition, cf. NeHMO 2507.13940).
- **HARD GATE for us: Stages B & C REQUIRE a physical arm.** Sim-only today. Real bimanual-Piper
  experiments are the true prerequisite, not the algorithms. Choose "statistically validated" (conformal)
  over "formally verified" (NN-CBF verification, 2511.06341/2501.15189) as the defensible rigor claim.

## Recency (2025-2026) — field moving AROUND this, not onto it
learned-reachability-as-CBF (RBN, HJRNO, NeHMO); conformal safe-set validation (new hot axis); teleop
safety kits (SPARK, bimanual failure-aware teleop); certified sim2real (Simulation-Gap Functions 2603.20672,
Verifiable Foundation Models 2606.23754); RoboGate boundary-focused failure discovery. Ingredients now
off-the-shelf -> feasibility up, integration-claim window narrowing.

## Must-cite: DeepReach/PCR/RBN (A), Fisac 1705.01292 + Wang 2403.05771 + conformal reachability (B),
SafeOpt/StageOpt/ActSafe/ARLET/RoboGate (C), OSCBF/SPARK/2602.01092 (D).
