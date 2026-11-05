# Literature: probing skills + online belief estimation (2026-11-03)

> Status (2026-11-05): CURRENT literature reference (shallow search, 11-03). Probing and belief shrinking are adopted mechanisms, not our claim.

Three web searches (Sonnet agents, abstract pages only, not full texts). Opus re-checked 22 arXiv ids against their titles (all
match). "Not found" below means not found in a shallow search, not "does not exist".

## Read in full first
- **Zhang, Zhou, Vasudevan, Provably-Safe Online System Identification, RSS 2025, arXiv 2504.21486** (code: roahmlab.github.io/OnlineSafeSysID).
  Exciting trajectories inside torque/state/collision limits + robust ID giving over-approximating payload inertial bounds under
  bounded noise. Closest to our sound set + safe probe. Single arm, payload only, no contents, no info-gain choice (objective unread).

## Sound set (safety side)
- Lopez, Slotine, How, Robust Adaptive CBFs, arXiv 2003.10028: set-membership ID, safe for every parameter in the shrinking set.
- Lu et al., Robust Adaptive MPC, arXiv 1911.00865: recursive feasibility with online set updates (linear).
- Li et al. (Wierman), uncertainty sets via set membership, ICML 2024, arXiv 2309.14648: contraction rates (linear).

## Belief + active probe choice (efficiency side, no guarantees)
- Kruzliak ... Hoffmann, IROS 2024, arXiv 2404.07344: Bayesian net over mass/volume/stiffness, expected info gain picks
  squeeze/poke/lift/weigh, stop when nothing left to learn; data released. Strong coded baseline.
- Jin ... Yuan, U-GRAPH, ICRA 2025, arXiv 2502.02663: BNN CoM estimate from F/T, uncertainty picks the next lift/reorient. Template
  for the edge/tilt probe.
- Tremblay ... Dudek, arXiv 2312.00215: differentiable Bayesian filter + info-gathering MPC for mass / toppling height.
- Noseworthy, Shaw ... Roy, ICRA 2024: amortized belief over mass/CoM/friction for heavy-object lifting vs particle belief.
- Schneider ... Peters, IROS 2022, arXiv 2210.12806; Memmel ... Gupta, ASID, arXiv 2404.12308: info-gain / Fisher probe choice.
- Yu ... Liu, QOED, arXiv 2605.12084: pick identifiable parameter directions first (e.g. mass x lever before CoM).
- Vahs ... Fan, arXiv 2601.02948: particle belief (SVGD) + conformal (probabilistic) safety while learning.
- Safe exploration, probabilistic only: SafeOpt-MC 1602.04450, GoSafe 2105.13281, Turchetta 1606.04753.

## Contents / liquid probes
- Guo ... Yuan, arXiv 2307.15646: tilt + shake container, F/T + GelSight -> contents mass/volume/particles.
- Huang ... Yuan, RSS 2022, arXiv 2205.08771: oscillation physics (decay -> viscosity, frequency -> height); re-read from joint torques?
- Lai, Gao, Lam, IROS 2024, arXiv 2404.06904: GPT-4V reads F/T plots after shaking (69 -> 86% liquid ID).

## Two-arm / cooperative payload ID (passive)
- Wong ... Campolo, arXiv 2605.22021: dual-arm box, online mass/CoM from wrenches + SOCP friction constraint. No probing.
- Zhang ... Lynch, arXiv 2502.04600 (mocobots); Fan ... Murphey, CDC 2017, arXiv 1709.01555.

## LLM/VLM side (must cite)
- PhysCaP, Lin ... Sun, arXiv 2608.21031: LLM planner + prioritizer pick proprioceptive mass/stiffness probes; no safety gate. Closest.
- Mavrogiannis ... Aloimonos, ICRA 2025, arXiv 2409.15505: LLM writes probe programs over a perception-action API.
- Phys2Real, Wang ... Schwager, arXiv 2510.11689: VLM CoM prior fused with interaction estimate (inverse variance) = our current setup.
- Murphy, Model Discovery Agent, arXiv 2608.09696: LLM proposes models, VOI picks experiments, SMC updates (not robots).
- Tru-POMDP 2506.02860; Xiao ... Hsu 2607.06501; InfoSeeker 2510.01531: LLM hypotheses + info-gathering, no physics, no safety.
- LLMPhy, AISTATS 2026, arXiv 2411.08027: LLM proposes simulator parameters from video (passive).
- MaD Physics, arXiv 2605.10820: Gemini models weak at choosing measurements under a budget (negative result for LLM probe choice).

## How we use this
Probing / belief shrinking is a means, not a claim (claims_2026-11-02.md, 11-02; user 11-03: "we adopt their mechanisms").
Adopt: Zhang/Vasudevan for sound set + safe excitation; Lopez RaCBF for set-membership updates; Kruzliak / U-GRAPH / QOED for
choosing probes; Guo/Yuan and Huang/Yuan for tilt/shake contents probes. Judge any probe work only by whether it lets more actions
be admitted and the task finish (claims 1-3).
