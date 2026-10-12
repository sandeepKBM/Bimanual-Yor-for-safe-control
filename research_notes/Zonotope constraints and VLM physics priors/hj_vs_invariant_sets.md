# When is HJ reachability / a BRT actually needed for infinite-horizon safety (vs cheaper invariant-set / set-based methods)?

Research date: 2026-10-08. Web research only.

**Evidence labels**
- **VERIFIED**: I checked it against the primary text myself. This means the arXiv abstract pulled through the arXiv API, the abstract from the Semantic Scholar or Crossref API, or PDF text I extracted with `pdftotext` and grepped.
- **VERIFIED\***: from the primary full text, but read through the web-fetch extraction model. Quotes are near-verbatim, not checked character by character.
- **SUMMARY**: from a search snippet or secondary page, or my own synthesis across sources. Treat it as lower-confidence.

I checked every arXiv ID cited below against the arXiv API on 2026-10-08. All titles and authors match.

**Venue caveat.** The following venues are *not* stated in the arXiv comment or journal-ref fields. They come from memory or secondary pages, so treat them as SUMMARY-level even when the line's content is VERIFIED:
- DeepReach, ICRA 2021
- ARMTD, RSS 2020
- SPARROWS, RSS 2024
- Wabersich & Zeilinger, Automatica 2021
- Bajcsy et al., CDC 2019
- Borquez et al., ICRA 2023
- Chen et al. backup CBF, CDC 2021
- Herbert et al., CDC 2019 (arXiv says only "submitted to CDC 2019")
- Bird et al., Automatica 2023
- Kousik et al., IJRR 2020
- Borquez et al. 2312.15347, T-RO
- Hsu et al., Annual Review year

Venues that *are* stated in the arXiv metadata:
- ICRA 2023 for 2209.12336
- L4DC 2024 for 2312.08604
- ICRA 2025 for 2404.00814
- IROS 2022 for 2204.12507
- CDC 2021 for 2104.02808
- RSS 2025 for 2502.00935
- T-RO 2024 / IROS 2023 for 2211.14361
- L4DC 2023 for 2212.03228
- TAC for 1611.00122 and 1901.01780
- IJRR 2020 for 2012.03390
- ACC 2024 for 2310.06921
- OJ-CSYS for 2407.09645
- Annu. Rev. CRAS for 2309.05837

---

## Objective: is the claim right? ("HJ/HJB works when our analytical constraint is arbitrary and needs to be solved fast, and we need a BRT for infinite-horizon planning guarantees")

### Takeaway
The claim is mostly wrong, with one true core.
- **True core:** HJ does accept arbitrary (non-convex, even non-analytic or learned) failure sets and nonlinear or adversarial dynamics. It returns the *maximal*, least-conservative safe set together with the optimal safe policy.
- **"Solved fast": wrong.** HJ is slow *offline*, with cost exponential in state dimension. It is only fast *online* as a lookup of a precomputed value function.
- **"Need a BRT for infinite horizon": wrong as a necessity.** Any control-invariant subset gives all-time safety. Backup sets, MPC terminal sets, valid CBFs and braking fail-safes all qualify. The converged BRT (the viability kernel / maximal control-invariant set) is just the *least conservative* such set.
- Learned BRTs (DeepReach, neural operators, latent HJ) lose the deterministic guarantee. At best they keep a probabilistic one.

### Cited Findings
**Part C, "need a BRT for infinite-horizon guarantees":**
- (VERIFIED\*) The Hsu–Hu–Fisac survey defines a control-invariant set Ω as one from which "the robot can prevent the system from *ever* leaving the set by continually applying policy π". It says many methods "forego the maximal safe set and instead seek a smaller safe set Ω ⊂ Ω\* that can produce a reliable, if more restrictive, safety filter". The extreme case it gives is that the set of collision-free *at-rest* configurations is a valid safe set, but it causes the "freezing robot problem". — [Hsu, Hu, Fisac 2023, Annu. Rev. CRAS, arXiv:2309.05837](https://arxiv.org/abs/2309.05837)
- (VERIFIED) Wabersich et al.:
  - Predictive safety filters (PSFs) "return an implicit representation of a control invariant set … defined by the feasibility of the optimization problem". This set "can closely approximate the maximal control invariant set … with a sufficiently large time horizon".
  - The terminal set only needs to be control invariant: "A trivial choice for S_trm is any equilibrium point … Since such terminal equality constraints can be rather conservative, control Lyapunov techniques are commonly used … less restrictive invariant terminal set".
  - Source: [Wabersich, Taylor, Choi, Sreenath, Tomlin, Ames, Zeilinger, "Data-Driven Safety Filters: HJ Reachability, CBFs, and Predictive Methods for Uncertain Systems", IEEE CSM 43(5):137–177, 2023 (NSF PAR copy)](https://par.nsf.gov/biblio/10473415-data-driven-safety-filters-hamilton-jacobi-reachability-control-barrier-functions-predictive-methods-uncertain-systems)
- (VERIFIED) A finite-time BRT is *not* by itself an infinite-horizon certificate. In the backup-CBF comparison, HJ is run to a large t: "For a sufficiently large t, lim_{t→∞} h(x,t) is an inner approximation of the maximum control invariant set." — [Chen, Jankovic, Santillo, Ames, "Backup CBFs: Formulation and Comparative Study", CDC 2021, arXiv:2104.11332](https://arxiv.org/abs/2104.11332)

**Part B, "solved fast":**
- (VERIFIED) In the same paper's Table I, on the 3D Dubins car, HJ takes 37.3 s offline and 0.5 ms online. The backup CBF needs no offline step and takes 1.4 ms of integration plus a 3.6 ms QP online. On a 16D quadrotor, HJ and SOS are "NA"; the backup CBF runs at 4.92 ms + 28.33 ms. The authors put HJ's limit at "maximum dimension 4-5" and SOS at "around 8-10". — [arXiv:2104.11332](https://arxiv.org/abs/2104.11332)
- (VERIFIED\*) The Hsu survey states that grid HJ's cost "is exponential in the state dimension, which limits its applicability to systems with no more than six continuous state variables". — [arXiv:2309.05837](https://arxiv.org/abs/2309.05837)
- Note the two sources disagree on the ceiling (4–5 vs ≤6 states). The real figure depends on grid resolution and GPU toolboxes.

**Part A, "arbitrary constraint":**
- (VERIFIED) "in theory, HJ reachability can synthesize safe controllers for nonlinear systems and nonconvex constraints, in practice, it has been limited to hand-engineered collision-avoidance constraints modeled via low-dimensional state-space representations". The same paper extends HJ to failures "hard—if not impossible—to write down by hand", such as spilling a bag, through a latent-space classifier. — [Nakamura, Peters, Bajcsy, "Latent Safety Filters", RSS 2025, arXiv:2502.00935](https://arxiv.org/abs/2502.00935)
- (VERIFIED) The pros/cons figure in Wabersich et al. reads:

  | Method | Pros | Cons |
  |---|---|---|
  | HJ | "Maximal safe set", "Safety certification of systems" | "Curse of dimensionality", "Indirect synthesis of filter" |
  | Predictive control | "Scalable to large-scale systems", predictive decision making, approximate optimality | "Complexity of robust design", "Heavy online computation" |
  | CBF | "Ease of implementation", "Smooth safety filtering" | "CBF synthesis", myopic / instantaneous decision making, suboptimal filtering |

  — [Wabersich et al. 2023](https://par.nsf.gov/biblio/10473415-data-driven-safety-filters-hamilton-jacobi-reachability-control-barrier-functions-predictive-methods-uncertain-systems)

### Inferences
- A corrected version of the claim: *"Use HJ when the dynamics are nonlinear and low-dimensional (or decomposable), the failure set is arbitrary or non-analytic, there is bounded adversarial disturbance, and we want the least-conservative safe set. Pay the cost offline, then query online. For infinite-horizon safety, any control-invariant set suffices. HJ just gives the biggest one."*
- "Arbitrary constraint" does not distinguish HJ from the alternatives:
  - Rollout-based filters (backup CBF, gatekeeper, model predictive shielding, PSFs) only need to *evaluate* the constraint along a predicted trajectory. That works for any computable constraint.
  - Polynomial and hybrid zonotopes represent non-convex sets.
  - What *is* distinctive about HJ is the backward, maximal, game-theoretic set computation (see the section on arbitrary constraints below).

### Gaps
- No source I found gives a universal "HJ needed iff …" theorem. The answer is a trade-off, synthesized from the surveys.

---

## Q1. Does infinite-horizon safety require the full BRT / viability kernel, or does any control-invariant subset suffice? (backup CBFs, MPC terminal sets, CBFs, safety-filter surveys). What is the trade-off between conservatism, computation and guarantee?

### Takeaway
Any control-invariant subset of the constraint set suffices. That covers implicit sets (backup-policy flows, MPC feasibility sets) as well as explicit ones (valid CBF superlevel sets, braking or equilibrium terminal sets). The maximal set (viability kernel = complement of the infinite-horizon BRT) only buys *less conservatism*, meaning fewer and later interventions and a larger feasible region. It costs exponential offline compute and, as usually implemented, a bang-bang filter.

The middle ground has a finite-horizon check plus a small invariant terminal or backup set. That gives all-time safety at online cost, and it approaches the maximal set as the backup policy and horizon improve.

### Cited Findings
**General principle**
- (VERIFIED\*) Rollout and backup filters give a "*recursive safety* guarantee: as long as the fallback can maintain safety from the initial state, it will be able to maintain safety from every subsequent one". Forward-reachable-set (FRS) filters check that "the FRS under the fallback policy eventually becomes *fully contained* in the terminal safe set without previously *intersecting* the failure set". No infinite-horizon computation is needed, because the terminal set is invariant. — [Hsu, Hu, Fisac 2023](https://arxiv.org/abs/2309.05837)
- (VERIFIED\*) The same survey names the downsides. Precomputed fallback policies "can result in conservative filtering and diminished task performance". A poorly built fallback "will need to be used often … triggering more frequent interventions". CBFs "no longer encode or approximate the maximal safe set Ω\*". It frames safety filters as a modular "safety monitor / fallback policy / intervention scheme". — [arXiv:2309.05837](https://arxiv.org/abs/2309.05837)

**Backup CBFs and backup sets (Ames group)**
- (VERIFIED) "Identifying a backup control law that stabilizes the system is in fact sufficient to exploit … set-invariance conditions … one only needs to be able to numerically integrate the closed-loop dynamics of the system over a finite horizon under this backup law." The paper studies "the relationship between the optimality of the backup law and how conservative the resulting safety filter is". It demonstrates the method on a Segway, an industrial manipulator, a quadrotor and a lower-body exoskeleton. — [Gurriet, Mote, Singletary, Nilsson, Feron, Ames, "A Scalable Safety Critical Control Framework for Nonlinear Systems", IEEE Access 8, 2020, DOI 10.1109/ACCESS.2020.3025248](https://doi.org/10.1109/ACCESS.2020.3025248) (abstract via Semantic Scholar API)
- (VERIFIED) The backup CBF "guarantees the feasibility of the CBF QP via an implicitly defined control invariant set … evaluated online by forward integrating the dynamics under the backup policy". Other findings from the same paper:
  - The resulting set is "typically suboptimal in the sense that it is not the maximum control invariant set".
  - On a 3D Dubins car, set size ranks HJ largest, backup CBF smaller, and a 4th-order SOS polynomial smallest. With an aggressive LQR backup, the backup-CBF set is "almost identical to the one from Hamilton Jacobi".
  - The online QP is about 10× slower than an explicit-CBF QP, because the CBF condition is enforced at every τ in [t, t+T].
  - Source: [Chen, Jankovic, Santillo, Ames, arXiv:2104.11332](https://arxiv.org/abs/2104.11332)

**MPC with a terminal invariant set**
- (VERIFIED) Title, authors and venue confirmed via Crossref: [Mayne, Rawlings, Rao, Scokaert, "Constrained model predictive control: Stability and optimality", Automatica, June 2000, DOI 10.1016/S0005-1098(99)00214-9](https://doi.org/10.1016/S0005-1098(99)00214-9).
- (SUMMARY) That paper's content is classical: a control-invariant terminal set plus a terminal constraint gives recursive feasibility, which means constraint satisfaction for all time.
- (VERIFIED) Wabersich et al. state the terminal-constraint mechanism directly: the terminal set S_trm is "a discrete-time control invariant set … and thus can be kept safe for all time". Because it is invariant, "we can construct a feasible candidate sequence" at the next step. They list alternatives to equilibrium terminal sets: periodic safe orbits, discrete-time CBFs, and adaptive terminal-set enlargement. — [Wabersich et al. 2023](https://par.nsf.gov/biblio/10473415-data-driven-safety-filters-hamilton-jacobi-reachability-control-barrier-functions-predictive-methods-uncertain-systems)
- (VERIFIED) The predictive safety filter "receives the proposed control input and decides … if it can be safely applied … Safety is thereby established by a continuously updated safety policy, which is based on a model predictive control formulation". — [Wabersich & Zeilinger, "A predictive safety filter for learning-based control of constrained nonlinear dynamical systems", Automatica 2021, arXiv:1812.05506](https://arxiv.org/abs/1812.05506)
- (VERIFIED) gatekeeper builds safe trajectories "by numerically forward propagating the system over a (short) finite horizon". It proves "tracking such a trajectory ensures the system remains safe for all future time, i.e., beyond the finite horizon", under nonlinear dynamics, bounded disturbances, input constraints and partial environment knowledge. — [Agrawal, Chen, Panagou, "gatekeeper", IEEE T-RO 2024 / IROS 2023, arXiv:2211.14361](https://arxiv.org/abs/2211.14361)

**HJ and MPC hybrids**
- (VERIFIED) An HJ escape maneuver used as a persistent MPC constraint gave a real-time (100 Hz) safety controller on a full-scale steer-by-wire car against an adversarial human driver. — [Leung, Schmerling, Zhang, Chen, Talbot, Gerdes, Pavone, IJRR 39(10-11), 2020, arXiv:2012.03390](https://arxiv.org/abs/2012.03390)
- (VERIFIED) An MPC+HJ framework "guarantees recursive feasibility for the MPC controller, and it is scalable to high-dimensional systems". It is shown on a 4D Dubins car and a 6-DoF Kuka iiwa. This is an RSS 2025 *workshop* paper. — [Wang, Jordana, Righetti, Bansal, arXiv:2506.23346](https://arxiv.org/abs/2506.23346)

**CBFs**
- (VERIFIED\*) "Finding a valid CBF for a system of interest is often a nontrivial problem … an open research challenge." — [Hsu et al. 2023](https://arxiv.org/abs/2309.05837)
- (VERIFIED) "Synthesizing a valid CBF that is not overly conservative in the presence of input constraints is a notorious challenge." refineCBF updates a candidate or backup CBF with DP reachability, and each iteration is "provably at least as safe as the prior iteration and converges to a valid CBF". — [Tonkens & Herbert, IROS 2022, arXiv:2204.12507](https://arxiv.org/abs/2204.12507)
- (VERIFIED) HJ's online implementation "can be overly conservative and/or rely on chattering bang-bang control". The Control Barrier-Value Function (CBVF) unifies HJ and CBFs and gives a QP-based controller. — [Choi, Lee, Sreenath, Tomlin, Herbert, CDC 2021, arXiv:2104.02808](https://arxiv.org/abs/2104.02808)
- (VERIFIED) HJ least-restrictive filters "can exhibit abrupt switching and bang-bang behavior". The paper characterizes "the maximal set of control inputs that ensures safety" and builds filters that trade performance, computation and smoothness. — [Borquez, Chakraborty, Wang, Bansal, T-RO, arXiv:2312.15347](https://arxiv.org/abs/2312.15347)
- (VERIFIED) A 2025 set-based CBF turns a convex control-invariant set (polytopic, zonotopic or MPC-based) into a tunable barrier via its Minkowski functional, posed as a single QP. Its worst-case execution time is 28.04 µs on a motor drive. The paper notes that classic set-based filters "scale favorably … but typically intervene only at the set boundary". — [Wabersich, Berkel, Gruber, Reimann, arXiv:2507.07805](https://arxiv.org/abs/2507.07805)

### Inferences
**Trade-off summary** (SUMMARY, my synthesis):

| Method | Conservatism | Offline cost | Online cost | Guarantee |
|---|---|---|---|---|
| (a) Grid HJ, converged | lowest (maximal set) | exponential, ≲4–6 states | lookup, sub-ms | deterministic, up to grid error |
| (b) Backup CBF / gatekeeper / model predictive shielding | set shrinks with a poor backup policy or short horizon; near-maximal with a good backup | none | ms-scale rollouts plus QP, scales to 16D+ | deterministic, given the model |
| (c) MPC / PSF with terminal invariant set | approaches maximal as the horizon grows | small (terminal set only) | heavy NLP | recursive feasibility, needs a solver that returns a feasible solution |
| (d) Hand or SOS CBF | can be very conservative; validity under input bounds is hard | — | QP, µs–ms | valid only if the CBF is truly valid |

- A braking-to-rest or equilibrium terminal set is the cheapest valid invariant set for manipulators. ARMTD and SPARROWS use exactly this (Q3).

### Gaps
- No quantitative cross-method benchmark of safe-set volume against dimension beyond the 3D/16D comparison in Chen et al. 2021.
- The Gurriet IEEE Access full text was not retrieved (403). Only the abstract is cited.

---

## Q2. Is HJ "fast"? Grid PDE curse of dimensionality; what do DeepReach and successors, neural operators (HJRNO 2504.19989), warm-starting and parameter-conditioning buy, and what guarantees do they lose?

### Takeaway
Grid HJ is not fast. It is exponential in state dimension and practical to roughly 4–6 states unless the system decomposes. It is fast only online, as a precomputed lookup (about 0.5 ms).

Each acceleration trades away something:
- **Warm-starting** keeps a conservative guarantee but gives only about 1.6× speedup.
- **Decomposition** is exact, but only for special structure.
- **Hopf formulas** are dimension-robust, but only for linear (time-varying) dynamics and convex settings. The Koopman lift adds approximation.
- **Parameter-conditioning** moves the cost offline and adds dimensions.
- **DeepReach-style PINNs** scale to 9–40D, but the deterministic guarantee drops to *probabilistic* (scenario or conformal) or *none*.
- **Neural operators (HJRNO)** are 300–866× faster at inference, but were shown only on 3–4D, are trained on grid ground truth, and come with *no* guarantee.

### Cited Findings
**The baseline cost**
- (VERIFIED) "The main challenge is addressing its exponential computational complexity with respect to the number of state variables." The paper also introduces a GPU-parallelized Level Set Toolbox. — [Bansal, Chen, Herbert, Tomlin, "HJ Reachability: A Brief Overview and Recent Advances", CDC 2017, arXiv:1709.07523](https://arxiv.org/abs/1709.07523)
- (VERIFIED) Dimension ceilings: 4–5 states per [arXiv:2104.11332](https://arxiv.org/abs/2104.11332); "no more than six" per [arXiv:2309.05837](https://arxiv.org/abs/2309.05837) (VERIFIED\*). Online lookup is 0.5 ms vs 37.3 s offline for the 3D Dubins car ([arXiv:2104.11332](https://arxiv.org/abs/2104.11332)).
- (VERIFIED) Grid methods "face challenges with … high-dimensional state spaces … 'curse of dimensionality'". Mitigations listed are "state decompositions, warm starting, or approximating solutions with neural networks". Non-HJ alternatives are SOS and set operations on "polytopes, ellipsoids, and zonotopes". — [Wabersich et al. 2023](https://par.nsf.gov/biblio/10473415-data-driven-safety-filters-hamilton-jacobi-reachability-control-barrier-functions-predictive-methods-uncertain-systems)

**Structure-exploiting exact methods**
- (VERIFIED) Decomposition computes BRSs and BRTs "without incurring additional approximation errors" for a class of coupled nonlinear systems, "orders of magnitude faster". It is shown on a 6D acrobatic quadrotor and a 10D near-hover quadrotor. It is "slightly conservative" with adversarial disturbances. — [Chen, Herbert, Vashishtha, Bansal, Tomlin, IEEE TAC, arXiv:1611.00122](https://arxiv.org/abs/1611.00122)
- (VERIFIED) Hopf formulas need no grid. For convex, positively 1-homogeneous Hamiltonians they "seem to obtain methods which are polynomial in the dimension", at "10^-4 to 10^-8 seconds per evaluation on a laptop". — [Darbon & Osher 2016, arXiv:1605.01799](https://arxiv.org/abs/1605.01799)
- (VERIFIED) Chow et al. extend this to state-dependent and non-convex Hamiltonians, but via *conjectured* Lax/Hopf principles with local-minima risk. — [Chow, Darbon, Osher, Yin 2017, arXiv:1704.02524](https://arxiv.org/abs/1704.02524)
- (VERIFIED) The Hopf formula "avoids the curse of dimensionality … but is restricted to linear time-varying systems". A Koopman lift extends it to nonlinear systems approximately, shown on a 10D glycolysis model. — [Sharpless, Shinde, Kim, Chow, Herbert, arXiv:2303.11590](https://arxiv.org/abs/2303.11590)
- (VERIFIED) A spectral Koopman-Hopf method gives upper and lower *bounds* on the value function and BRS for adversarial nonlinear systems. — [Sutavani & Vaidya 2026, arXiv:2609.06878](https://arxiv.org/abs/2609.06878)

**Online-update tricks**
- (VERIFIED) Warm-starting "will result in guaranteed conservative solutions" and is "exact" for many common formulations. It was 1.6× faster than standard and 6.2× faster than untuned discounted reachability, including on a 10D quadcopter. — [Herbert, Ghosh, Bansal, Tomlin, CDC 2019, arXiv:1903.07715](https://arxiv.org/abs/1903.07715)
- (VERIFIED) Parameter-conditioned reachable sets treat expected changes as "virtual states" solved offline with high-dimensional (DeepReach-style) methods. Online the system "can simply query", enabling "a real-time update of the safety assurances". The approach assumes the changes can be parameterized. — [Borquez, Nakamura, Bansal, ICRA 2023, arXiv:2209.14976](https://arxiv.org/abs/2209.14976)
- (VERIFIED) A "real-time safety analysis method based on HJ reachability" provides guarantees in a priori unknown environments sensed online. — [Bajcsy, Bansal, Bronstein, Tolani, Tomlin, CDC 2019, arXiv:1905.00532](https://arxiv.org/abs/1905.00532)

**DeepReach and successors**
- (VERIFIED) DeepReach uses a sinusoidal-network PDE solver whose cost "do[es] not scale directly with the state dimension, but rather with the complexity of the underlying reachable tube". It was shown on 9D and 10D problems. — [Bansal & Tomlin, ICRA 2021, arXiv:2011.02082](https://arxiv.org/abs/2011.02082)
- (VERIFIED) "Neural networks can make errors and thus the computed solution may not be safe." The fix is an error bound plus scenario optimization, giving a *probabilistically* safe tube. — [Lin & Bansal, ICRA 2023, arXiv:2209.12336](https://arxiv.org/abs/2209.12336)
- (VERIFIED) Robust scenario optimization and conformal prediction are proved equivalent. There is a "direct trade-off between resilience to outlier errors … and the strength of the probabilistic safety guarantee". — [Lin & Bansal, L4DC 2024, arXiv:2312.08604](https://arxiv.org/abs/2312.08604)
- (VERIFIED) "The accuracy of the learned solution decreases with system complexity", partly from soft boundary conditions. ExactBC imposes them exactly. — [Singh, Feng, Bansal, ICRA 2025, arXiv:2404.00814](https://arxiv.org/abs/2404.00814)
- (VERIFIED) PINN residual training has "unstable training dynamics and suboptimal solutions". MPC-generated supervision at collocation points improves this, shown on 2D, 7D, 13D and 40D systems. — [Feng, Qiu, Bansal 2025, arXiv:2505.03830](https://arxiv.org/abs/2505.03830)

**RL-based HJ approximation**
- (VERIFIED) A survey of learning the HJ value jointly with RL policies to scale beyond grids. — [Ganai, Gao, Herbert, IEEE OJ-CSYS, arXiv:2407.09645](https://arxiv.org/abs/2407.09645)
- (SUMMARY, search snippet) The discounted safety Bellman equation V = (1−γ)ℓ + γ·min{ℓ, V′} makes the safety backup a contraction, so RL can approximate it. — [Fisac, Lugovoy, Rubies-Royo, Ghosh, Tomlin, ICRA 2019 (Princeton record)](https://collaborate.princeton.edu/en/publications/bridging-hamilton-jacobi-safety-analysis-and-reinforcement-learni/)
- (VERIFIED) That discounting "alter[s] the original reachability semantics". The paper proposes a semantics-preserving non-additive Bellman operator whose fixed point exactly matches HJ. — [El-Hajj, Solanki, van Beers, de Visser, van Kampen 2026, arXiv:2607.07893](https://arxiv.org/abs/2607.07893)
- (VERIFIED) Related work: [Solanki et al., arXiv:2601.08050](https://arxiv.org/abs/2601.08050). Its v2 title is "Unifying Hamilton-Jacobi Reachability and Reinforcement Learning"; v1 had a different title.
- (VERIFIED) Learning-based safety filters "lack formal safety guarantees". This paper soundly verifies Q-function self-consistency and fixes "zero-sublevel-set shrinkage". — [Li, Hu, Yang, Liu 2025, arXiv:2506.15693](https://arxiv.org/abs/2506.15693)
- (VERIFIED) In ISAACS, the learned safety policy "does not intrinsically guarantee safety". The guarantee comes from a forward-reachability rollout shield, shown on a 5D race car. — [Hsu, Nguyen, Fisac, L4DC 2023, arXiv:2212.03228](https://arxiv.org/abs/2212.03228)
- (VERIFIED\*) "Neural network critics … the corresponding safety filters come with no guarantees." — [Hsu et al. 2023](https://arxiv.org/abs/2309.05837)

**Neural operators (HJRNO)**
- (VERIFIED) HJRNO "learns a mapping between value functions, enabling fast inference with strong generalization across different obstacle shapes and system configurations". It is pitched as a "foundation model approach". — [Li & Chen, arXiv:2504.19989](https://arxiv.org/abs/2504.19989)
- (VERIFIED\*) Details from its full text:
  - Systems tested are only Air3D (3D) and a dynamic Dubins car (4D).
  - Inference takes 0.003–0.005 s, against about 0.9 s (single obstacle) and 2.6 s (indoor) for the grid solver.
  - Test relative L2 error is 0.002–0.037.
  - Ground truth comes from the `hj_reachability` grid solver, with 100–300 training samples on a 50×50 grid.
  - There is no formal or probabilistic guarantee. Training data grows with the hyperparameter grid.
  - Source: [arXiv:2504.19989 (HTML v2)](https://arxiv.org/html/2504.19989v2)

### Inferences
- "Fast HJ" in the online sense means precompute, then look up. That is fast only if the constraint and dynamics are fixed, or parameterized ahead of time.
- If the constraint is arbitrary *and* changes at runtime (new obstacle geometry, a new task constraint), grid HJ needs recomputation. The fast options are:
  - neural operators or DeepReach: no or probabilistic guarantee;
  - warm-start or local updates: modest speedup, low-dim only;
  - Hopf: linear dynamics only.
- HJRNO's training data is still produced by a grid solver. Its dimensional reach is therefore bounded by what the grid can do. It amortizes the cost; it does not remove the curse. (Inferred from its 3D/4D experiments and grid ground truth.)
- For linear or feedback-linearized dynamics (a double integrator per joint), Hopf-type HJ and zonotope reachability are both cheap and exact-ish. That removes much of the reason to use grid HJ there. (SUMMARY / inference.)

### Gaps
- No published wall-clock data for DeepReach training on >10D systems was retrieved here. Those papers report accuracy, not a speed table in the abstract.
- I found no deterministic (non-probabilistic) verification of high-dimensional neural BRTs. Li et al. 2025 verifies self-consistency only on standard low-dimensional benchmarks.
- HJRNO hardware and timing details were reported vaguely in the extracted text.

---

## Q3. For ARBITRARY (non-convex, nonlinear) constraints / failure sets, what can zonotope-family set-based reachability (Althoff/CORA, Kochdumper, Vasudevan ARMTD/ARMOUR/SPARROWS) handle, where does it get too conservative, and what can HJ handle that it cannot?

### Takeaway
**What zonotope methods do well:**
- Zonotope-family methods compute *forward* over-approximations. Their cost is polynomial in dimension.
- Non-convexity is handled through richer representations:
  - polynomial / sparse polynomial zonotopes and constrained polynomial zonotopes (non-convex, closed under many operations);
  - hybrid zonotopes (unions of 2^N convex pieces at linear representation cost).
- For robots they verify a *fixed* policy or *parameterized trajectory family* in real time (ARMTD, ARMOUR, SPARROWS, RTD). They reach infinite-horizon safety via a braking fail-safe, which is an invariant terminal set.

**Where they get conservative:**
- the wrapping effect under nonlinear dynamics;
- coarse body over-approximations (zonotope swept volumes in clutter);
- the trajectory parameterization itself;
- *backward*, game-theoretic synthesis, because zonotopes are not closed under the Minkowski difference.

**What HJ does that they cannot (in low dimension):** the maximal backward safe set and optimal safe policy for nonlinear dynamics with adversarial disturbance, with arbitrary failure-set shape.

### Cited Findings
**Representations**
- (VERIFIED) Sparse polynomial zonotopes "can represent non-convex sets and are generalizations of zonotopes, polytopes, and Taylor models". Operations have "polynomial complexity w.r.t. the dimension", and "the wrapping effect is substantially reduced" for nonlinear reachability. — [Kochdumper & Althoff, IEEE TAC 2020, arXiv:1901.01780](https://arxiv.org/abs/1901.01780)
- (VERIFIED) Constrained polynomial zonotopes are a non-convex representation "closed under linear map, Minkowski sum, Cartesian product, convex hull, intersection, union, and quadratic as well as higher-order maps", with at most polynomial complexity in representation size. — [Kochdumper & Althoff 2020, arXiv:2005.08849](https://arxiv.org/abs/2005.08849)
- (VERIFIED) A hybrid zonotope is "equivalent to the union of 2^N constrained zonotopes … through the addition of N binary zonotope factors". It gives exact forward reachable sets for linear mixed-logical-dynamical (MLD) hybrid systems, with "only linear growth in the complexity". It is closed under linear map, Minkowski sum, generalized intersection and halfspace intersection. — [Bird, Pangborn, Jain, Koeln, arXiv:2106.14831](https://arxiv.org/abs/2106.14831) (SUMMARY: published in Automatica 2023)

**Backward reachability with zonotopes**
- (VERIFIED) "Backward reachability analysis is a two-player game and involves Minkowski difference operations, but zonotopes are not closed under such operations." The paper therefore *under-approximates* backward reachable sets, for *uncertain linear systems* only. — [Yang & Ozay 2021, arXiv:2107.01724](https://arxiv.org/abs/2107.01724)
- (VERIFIED) Hybrid zonotopes give exact *one-step* backward reachable sets for ReLU neural-network feedback systems, with MILP certification for finite-time avoidance. — [Zhang, Zhang, Xu 2023, arXiv:2303.10513](https://arxiv.org/abs/2303.10513)
- (VERIFIED) For nonlinear plants this becomes an *over-approximation*, with "a refinement procedure … to alleviate the potential conservatism". — [Zhang, Zhang, Xu, ACC 2024, arXiv:2310.06921](https://arxiv.org/abs/2310.06921)

**Safety shields and robot planners**
- (VERIFIED) A safe-RL shield uses parameterized polynomial-zonotope reachability plus mixed-integer action projection. It "can efficiently handle input constraints and dynamic obstacles, eases incorporation of the spatial robot dimensions … guarantees robust safety despite process noise and measurement errors, and is well suited for high-dimensional systems". — [Kochdumper, Krasowski, Wang, Bak, Althoff, arXiv:2210.10691](https://arxiv.org/abs/2210.10691)
- (VERIFIED) RTD computes an offline forward reachable set of parameterized trajectories. "Persistent feasibility is achieved by prescribing a minimum sensor horizon and a minimum duration for the planned trajectories." It was run on a Segway and a Rover. — [Kousik, Vaskov, Bu, Johnson-Roberson, Vasudevan, arXiv:1809.06746](https://arxiv.org/abs/1809.06746) (SUMMARY: IJRR 2020)
- (VERIFIED) In ARMTD, "Each trajectory includes a fail-safe maneuver (braking to a stop)". Per-joint reachable sets are computed offline and intersected with obstacles at runtime to give "provably-conservative collision-avoidance constraints". On a 6-DoF arm it "never crashes". — [Holmes, Kousik, Zhang, Raz, Barbalata, Johnson-Roberson, Vasudevan, RSS 2020, arXiv:2002.01591](https://arxiv.org/abs/2002.01591)
- (VERIFIED) ARMOUR adds a robust tracking controller with bounded error under mass and inertia uncertainty, and a recursive Newton–Euler over "a continuum of desired trajectories". It is real-time and validated on hardware. — [Michaux et al., arXiv:2301.13308](https://arxiv.org/abs/2301.13308)
- (VERIFIED) SPARROWS builds a sphere-based reachable set with an exact signed distance function. It is "significantly less conservative than previous approaches". — [Michaux, Li, Chen, Chen, Zhang, Vasudevan, RSS 2024, arXiv:2402.08857](https://arxiv.org/abs/2402.08857)
- (VERIFIED\*) SPARROWS numbers, 7-DoF arm (full text, Tables III and VI):
  - Planning time with 10 obstacles: 0.14±0.07 s for SPARROWS vs 0.18±0.08 s for zonotope-based ARMTD. With 40 obstacles: 0.20 s vs 0.45 s.
  - Random-obstacle success at the higher acceleration bound (π/6): 87 / 62 / 40 % for SPARROWS vs 56 / 17 / 0 % for ARMTD, at 10 / 20 / 40 obstacles.
  - Every trajectory ends with q̇ = 0 (braking). Obstacles are assumed static.
  - Source: [arXiv:2402.08857 HTML](https://arxiv.org/html/2402.08857v1)

**Survey context and HJ comparison**
- (VERIFIED) Wabersich et al. list zonotopes, with polytopes and ellipsoids, as set-operation routes to *approximate* the maximal control-invariant set without solving the HJ PDE. — [Wabersich et al. 2023](https://par.nsf.gov/biblio/10473415-data-driven-safety-filters-hamilton-jacobi-reachability-control-barrier-functions-predictive-methods-uncertain-systems)
- (VERIFIED) HJ "can handle general nonlinear system dynamics, bounded adversarial system disturbances, and state and input constraints". — [Lin & Bansal, arXiv:2209.12336](https://arxiv.org/abs/2209.12336); same claim in [DeepReach, arXiv:2011.02082](https://arxiv.org/abs/2011.02082)
- (VERIFIED) "in theory … nonconvex constraints". — [Nakamura et al., arXiv:2502.00935](https://arxiv.org/abs/2502.00935)
- (SUMMARY) The canonical review of set-propagation techniques (zonotopes, support functions; linear, nonlinear and hybrid systems) is Althoff, Frehse, Girard, "Set Propagation Techniques for Reachability Analysis", Annu. Rev. CRAS 4:369–395, 2021. — [TUM portal record](https://portal.fis.tum.de/en/publications/set-propagation-techniques-for-reachability-analysis/). Full text was blocked (Anubis or 403), so none of its specific claims are cited.

### Inferences
**What set-based methods handle well:**
- Linear or feedback-linearized dynamics, where zonotope reachability is exact up to disturbance bounds and scales to high dimension.
- Non-convex *obstacles*, via polynomial or hybrid zonotopes, unions, or sphere decompositions with signed distance functions.
- Verifying a *given* policy or trajectory family online in tens to hundreds of ms.

**Where they become too conservative:**
- Long horizons under nonlinear dynamics (wrapping).
- Dense clutter, if the body over-approximation is coarse: zonotope ARMTD drops to 0% success at 40 obstacles while spheres reach 40%.
- Anything outside the chosen trajectory parameterization.
- Backward or game synthesis beyond linear systems.

**What HJ uniquely gives (low-dimensional only):**
- The *maximal* safe set and *optimal* safe feedback policy for nonlinear dynamics under adversarial disturbance.
- Failure sets given by any margin function ℓ(x), including learned classifiers in a latent space.

### Gaps
- No direct head-to-head on the *same* benchmark between zonotope set-propagation and grid HJ was found in this pass.
- The Althoff/Frehse/Girard review could not be read. No sourced numbers on the wrapping effect or zonotope dimension limits for nonlinear systems; the scaling claim relies on the Kochdumper abstracts.
- SPARROWS figures came through the fetch extractor and are not cross-checked character by character against the tables.

---

## Q4. When is HJ the right tool? (nonlinear dynamics, games / disturbances, learned latent dynamics, need for the maximal safe set)

### Takeaway
**HJ is the right tool when all of these hold:**
- the state is low-dimensional (≲5–6 states) or decomposable, or the dynamics are linear enough for Hopf;
- the dynamics are nonlinear and/or there is a bounded adversary or disturbance;
- the least-conservative (maximal) safe set matters, and no good backup policy is available to make a backup CBF or PSF near-maximal;
- you can precompute offline.

**It is also the natural formalism for failures that cannot be written down**, using latent-space HJ with a learned failure classifier. That case comes with an explicit loss of formal guarantees and known smoothness pathologies.

**Prefer invariant-set or predictive methods when** the dimension is high, the constraint changes online, or a natural fail-safe exists (braking or stopping).

### Cited Findings
**Strengths**
- (VERIFIED) HJ "explicitly characterizes the maximal control invariant set in X before deploying the controller, which makes it attractive for the verification of safety-critical systems". By contrast, PSFs avoid "an explicit pre-computation of an optimal control policy, thereby enabling scalability". — [Wabersich et al. 2023](https://par.nsf.gov/biblio/10473415-data-driven-safety-filters-hamilton-jacobi-reachability-control-barrier-functions-predictive-methods-uncertain-systems)
- (VERIFIED) Formal treatment of "bounded adversarial system disturbances" is a headline HJ strength. — [arXiv:2209.12336](https://arxiv.org/abs/2209.12336), [arXiv:2011.02082](https://arxiv.org/abs/2011.02082)

**Latent and learned settings**
- (VERIFIED) Latent Safety Filters do HJ in a world model's latent space. "Constraint specification is then transformed into a classification problem in the latent space." The method protects a Franka against spilling a bag and toppling objects. — [Nakamura, Peters, Bajcsy, RSS 2025, arXiv:2502.00935](https://arxiv.org/abs/2502.00935)
- (VERIFIED\*) Limitations stated in the same paper's full text:
  - The value is approximated with DDPG on the discounted latent Bellman equation, because exact computation is "intractable due to the dimensionality of the latent embedding".
  - "our practical implementation currently lacks formal assurances due to the combination of possible errors when learning the world model, failure classifier, and resulting HJ value function".
  - "The safety filter can only protect against outcomes that the world model can predict."
  - There is a "Need to recompute a safety value function for any new additional constraints."
  - Source: [arXiv:2502.00935 HTML](https://arxiv.org/html/2502.00935v3)
- (VERIFIED) Classifier margins make latent HJ values "saturated … [with] discontinuous jumps". The authors prove "the value function's Lipschitz constant scales linearly with the margin function's Lipschitz constant". RL values trained only on safety-policy data are "inaccurate … for nominal policy actions, precisely where CBF filtering needs them". LatentCBF fixes both and doubles task completion compared with switching filters. — [Nakamura, Bishop, Man, Johnson, Manchester 2025, arXiv:2511.18606](https://arxiv.org/abs/2511.18606)

**Middle ground**
- (VERIFIED) refineCBF uses DP reachability to *refine* a backup or expert CBF in the loop. — [Tonkens & Herbert, arXiv:2204.12507](https://arxiv.org/abs/2204.12507)
- (VERIFIED) CBVF merges HJ with CBF-QP to get a smooth, robust online filter. — [Choi et al., arXiv:2104.02808](https://arxiv.org/abs/2104.02808)
- (VERIFIED) HJ supervised by MPC scales learned reachability to 40D. — [Feng et al., arXiv:2505.03830](https://arxiv.org/abs/2505.03830)

### Inferences (decision rule, SUMMARY)
**Use HJ (grid) when** the state is ≤ about 5 dimensions (or decomposable), the dynamics are nonlinear, there is a disturbance or adversary, the maximal set is needed, and the constraints are fixed or parameterizable. Examples: relative-state pursuit-evasion, a 3–4D reduced model of the dangerous subsystem.

**Use learned or latent HJ when** the failure is only observable or learnable (spill, topple). Accept *no* deterministic guarantee. Add probabilistic verification (scenario/conformal, Lin & Bansal) or a model-based rollout shield (ISAACS-style). Smooth the margins if the value will be used as a CBF (LatentCBF).

**Use a backup CBF, gatekeeper or PSF with a terminal invariant set when** the system is high-dimensional, the constraint is arbitrary but *evaluable* along rollouts, and a natural fail-safe exists (brake to stop, hover, return to an equilibrium). This gives infinite-horizon safety at online cost, and is near-maximal with a good backup.

**Use zonotope or sphere set-based reachability when** the dynamics are linear or feedback-linearized and you must *verify a planned trajectory family* against geometric obstacles in real time with a hard guarantee (ARMTD, ARMOUR, SPARROWS). Accept the parameterization conservatism.

**Project-specific inferences for this repo** (bimanual FL-MPC; labelled inference, not literature):
- FL-MPC's feedback-linearized double integrator gives linear joint-space dynamics. Even so, two 6-DoF arms give 24 states, so grid HJ on the full state is infeasible (24 ≫ 6).
- The literature-standard way to get the infinite-horizon guarantee the claim asks for is a braking terminal constraint (q̇ = 0 at horizon end) in the existing QP, as in ARMTD and SPARROWS. Because the system is centralized and cooperative, both arms' plans are known. A joint at-rest configuration with arm–arm clearance, held with the torque box satisfied, is then a control-invariant set as long as the environment is static.
- Caveat: invariance of "at rest" requires the gravity-holding torque to fit inside the torque box. On the real Piper the feed-forward cap is about ±8 Nm, against about 7.4 Nm of j1/j2 gravity (repo memory notes). The terminal set therefore only *barely* exists on hardware, unlike in sim (TAU_MAX = 100).
- Spill is geometry-blind, so latent HJ is the right formalism for it. The literature says plainly that this carries no formal guarantee (Nakamura 2025), which matches the repo's existing latent-HJ framing.

### Gaps
- No source quantifies the "maximal set is worth it" threshold, i.e. how much task performance a maximal vs a backup-induced set buys in manipulation.
- No source evaluates HJ vs backup-CBF/PSF specifically on bimanual or multi-arm manipulators.
- The project-specific points above are inferences from the cited mechanisms and repo notes. They are not literature findings.
