# Safety Filtering under Tracking Loss, Occlusion, Sensor Dropout and Measurement Latency (2023-2026)

Scope: safety filters and safe control under intermittent, delayed or occluded observations, and how sensing delay relates to the reaction time (lead time) left before failure. Searched 2026-10-05.

Verification tags:
- **VERIFIED-ABS**: title, authors, date and abstract read from the arXiv API or arXiv abs page.
- **VERIFIED-HTML**: details read from the paper's arXiv HTML full text. The fetch tool extracted them, so equations are quoted as it returned them.
- **VERIFIED-PDF**: PDF text extracted locally and read.
- **SUMMARY**: from a search snippet or a third-party page only.

Dates are first arXiv submission dates. Excluded as already known: ReKep 2409.01652, SafeBimanual 2508.18268, UNISafe 2505.00779, Cao & Bansal 2609.34300.

---

## Q1. Which safety filters explicitly model measurement latency and show how it eats the safe reaction window? Does any derive a "required perception latency" from the dynamics?

### Takeaway
Yes, but almost entirely outside manipulation and outside learned or latent models.

Five lines of work make the effect of latency explicit:
- **Closed-form latency bounds:** the RADAR quadrotor paper (2608.01855) derives `v_max = max(0, sqrt(2aR) - a·t_end)`, so each second of latency costs `a` m/s of safe speed. This extends Falanga et al. 2019.
- **Per-state maximum tolerable delay:** delay-resilient shields (2307.02164) compute a "controllability" value for each state, defined as the largest delay under which that state is still safe.
- **Disturbance tolerance that shrinks with delay:** CBF placement in networked control (2603.29792) proves that the admissible disturbance shrinks as delay grows. A filter at the plant, which sees fresh measurements, dominates a delayed remote predictive CBF.
- **Reachable sets inflated by staleness:** SARA-shield HRC (2604.15221) grows the human reachable set by `Δt·v_h,max` for the stale interval of a 65.86 ms vision pipeline.
- **Discretization or sampling gaps:** a one-step expanded BRT (2609.17904) and sample-and-hold CBFs (2304.08685).

Input-delay CBFs with predictors (ECBF, DaCBF, ICBF, PCBF) are the mature theory. Measurement delay maps onto them through the same predictor construction. I found no paper that derives a required perception latency for a learned or latent-world-model safety filter, or for a manipulation task.

### Cited Findings

**Latency-to-reaction-window derivations (required latency)**

- **Mandaokar & Rinner, "RADAR Perception for Dynamic Obstacle Avoidance onboard small-scale Quadrotor UAVs", arXiv 2608.01855, 2026-08-03.** VERIFIED-ABS + VERIFIED-HTML.
  - *Sensing problem:* sensing range and end-to-end sensing-to-command latency against fast thrown obstacles.
  - *Method:* derives "latency and spatial bounds that relate sensing range, relative speed, and control delay, yielding sufficient conditions for successful avoidance". From the HTML:
    - `v_max(R,a,t_end) = max(0, √(2aR) − a·t_end)`
    - `d_crit = v_rel(t_end + t_evade)`, with `t_evade = √(2R_sum/a)`
    - Worked example: R = 4 m, a = 25 m/s², t_end = 15 ms gives v_max = 13.77 m/s and d_crit = 3.69 m. Safety fails when d_crit exceeds the perception range.
    - Latency budget: t_det ≈ 12.34 ms plus t_avoid ≈ 1.31 ms, so ≈ 13.65 ms end to end on a Raspberry Pi 4B.
    - Tracker is interacting multiple models (IMM); controller is a CBF that outputs evasive accelerations.
  - *Learned/WM:* no; the authors stress "without relying on GPU or deep learning".
  - *Experiments:* 300 ball throws (3 sizes, light/dark) plus 90 in smoke; 20 onboard flights.
  - *Does NOT:* handle occlusion ("we only consider detection in direct LoS"), multiple obstacles, or learned models; flights are indoor only. [arXiv](https://arxiv.org/abs/2608.01855), [HTML](https://arxiv.org/html/2608.01855)

- **Mandaokar & Rinner, "Distributionally Robust Acceleration Control Barrier Filter for Efficient UAV Obstacle Avoidance", arXiv 2512.00462, 2025-11-29, accepted IEEE RA-L.** VERIFIED-ABS.
  - *Method:* second-order CBF as linear half-spaces on commanded acceleration. "Latency, actuator limits, and obstacle accelerations are handled through an effective clearance that considers dynamics and delay." Adds Cantelli tightening and a DR-CVaR early trigger.
  - *Experiments:* simulation.
  - *Does NOT:* use learned models or address occlusion. [arXiv](https://arxiv.org/abs/2512.00462)

- **Foundation, one line: Falanga, Kim & Scaramuzza, "How Fast is Too Fast? The Role of Perception Latency in High-Speed Sense and Avoid", RA-L 2019.** SUMMARY. Derives the maximum speed to avoid an obstacle as a function of perception latency, sensing range and maximum acceleration. [RPG PDF](https://rpg.ifi.uzh.ch/docs/RAL19_Falanga.pdf), [ResearchGate](https://www.researchgate.net/publication/330945969_How_Fast_is_Too_Fast_The_Role_of_Perception_Latency_in_High-Speed_Sense_and_Avoid)

- **Foundation, one line: ISO/TS 15066 speed-and-separation monitoring.** SUMMARY. The protective separation distance explicitly includes robot-system reaction time (sensing plus processing latency), stopping distance and measurement uncertainty. NIST implementation notes illustrate it: a human at 2000 mm/s reported 0.5 s late means about 1000 mm less separation than indicated. [NIST](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=914783)

**Delay as worst-case hidden state (shields, discrete)**

- **Cano Córdoba, Palmisano, Fränzle, Bloem & Könighofer, "Safety Shielding under Delayed Observation", arXiv 2307.02164, 2023-07-05, ICAPS 2023.** VERIFIED-ABS + VERIFIED-PDF.
  - *Sensing problem:* worst-case observation delay of δ steps.
  - *Method:* solves the safety game under delay, which "amounts to pre-deciding actions δ steps in advance". The shield sees only a proper prefix of the run and keeps a buffer of the δ latent actions.
  - *Maximum tolerable delay per state:* the **controllability value** ϕc(s) is "the maximum delay for which s is controllable". It is computed by solving the game for δ = 1…δmax, giving a per-state tolerable-latency map (worst case O(|S|^δmax)). A robustness heuristic ϕr gives almost the same interference rates at lower cost.
  - *Experiments:* CARLA car-on-car intersection and pedestrian crosswalk, with Δt = 0.5 s per step. Pedestrians move up to 1 m per step. "The larger the delay … the earlier the shield initiates braking." No collisions in the tested scenarios, although the game "occasionally" entered empty-strategy states from discretization.
  - *Does NOT:* use continuous dynamics, learned models or perception; the abstraction is hand-built. [arXiv](https://arxiv.org/abs/2307.02164)

**Delay degrades the CBF safety margin (continuous / discrete CBF theory)**

- **Beger, Chen & Hirche, "Where to Put Safety? Control Barrier Function Placement in Networked Control Systems", arXiv 2603.29792, 2026-03-31, accepted L-CSS.** VERIFIED-ABS + VERIFIED-HTML.
  - *Sensing problem:* network delay between the plant and a remote MPC. Delay is split into a known constant part τ̂ and an unknown residual r_k ≤ r̄.
  - *Result:* "local CBFs provide higher disturbance tolerance due to access to fresh state measurements".
    - Remote MPC-CBF tolerance: `w̄_r = Δ^rem/G(τ̂,l) − d_u`, with `G = S_{τ̂+l} + (1−γ)S_{τ̂+l−1}` and S_r a sum of Lipschitz powers. It therefore shrinks roughly geometrically with delay and horizon.
    - Proposition 3: `w̄_r(τ̂>0) < w̄_r(0) ≤ w̄_l`.
    - Proposes a combined local-plus-predictive architecture.
  - *Experiments:* simulated planar 3-DoF manipulator, MPC at 20 Hz, local loop at 200 Hz.
    - Remote MPC-CBF safe rate drops from 92% to 64% as delay goes from 3+0 to 5+1 steps.
    - Local CBF is 96-100% safe but with about 0.02 mm clearance (late intervention).
    - Combined architecture: 100% safe with 2-4 mm clearance.
  - *Does NOT:* include hardware, perception, occlusion or learned models. [arXiv](https://arxiv.org/abs/2603.29792), [HTML](https://arxiv.org/html/2603.29792)

- **Borquez, "Timely Activation of Safety Filters via One-Step Reachability Expansion", arXiv 2609.17904, 2026-09-15.** VERIFIED-ABS.
  - *Problem:* discrete sampling lets the system "jump into the unsafe BRT between updates".
  - *Method:* a one-step expanded BRT as the activation set for a least-restrictive HJ filter, computed with standard continuous-time solvers.
  - *Does NOT (per abstract):* model latency longer than one step, or learned models. [arXiv](https://arxiv.org/abs/2609.17904)

- **Bahati, Ong & Ames, "Sample-and-Hold Safety with Control Barrier Functions", arXiv 2304.08685, 2023-04-18, ACC 2024 submission.** VERIFIED-ABS.
  - *Method:* modulates the sampled input so that sampling causes no safe-set violation. Characterizes sampling-frequency requirements and event-trigger conditions.
  - *Experiments:* adaptive-cruise-control simulation.
  - *Does NOT:* handle measurement delay or dropout. [arXiv](https://arxiv.org/abs/2304.08685)

- **Kim, Das, Kim, Ames, Burdick & Sloth, "Minimizing Conservatism in Safety-Critical Control for Input-Delayed Systems via Adaptive Delay Estimation", arXiv 2411.17277, 2024-11-26, ECC 2025.** VERIFIED-ABS.
  - *Method:* delay-adaptive CBFs (DaCBF). A disturbance observer bounds the state-prediction error, and the maximum delay-estimation error bound is shrunk online so the prediction-error bound is monotonically non-increasing.
  - *Experiments:* automated connected truck.
  - *Does NOT:* address perception or occlusion. [arXiv](https://arxiv.org/abs/2411.17277)

- **Kiss, Das, Molnar & Ames, "Integral Control Barrier Functions with Input Delay: Prediction, Feasibility, and Robustness", arXiv 2603.24566, 2026-03-25.** VERIFIED-ABS.
  - *Method:* predictor feedback embedded in integral CBFs, with a closed-form feasibility condition for joint state and input constraints. The abstract notes that delay makes controllers "respond too late, and with excessively large corrective actions".
  - *Does NOT (per abstract):* address measurement dropout, occlusion or learning. [arXiv](https://arxiv.org/abs/2603.24566)

- **Paredes Salazar, Usevitch & Goel, "Predictive Control Barrier Functions for Discrete-Time Linear Systems with Unmodeled Delays", arXiv 2510.01059, 2025-10-01, ACC 2026 submission.** VERIFIED-ABS. Extends the prediction horizon so that unknown relative degree from input delay becomes relative degree one. Experiments: delayed double integrator and bicopter, in simulation. [arXiv](https://arxiv.org/abs/2510.01059)

- **Pati, Hwang & Yong, "Control Barrier Functions for Linear Continuous-Time Input-Delay Systems with Limited-Horizon Previewable Disturbances", arXiv 2403.04243, 2024-03-07, ACC 2024.** VERIFIED-ABS. Limited-preview CBF for input-delay systems with recursive feasibility under input bounds. [arXiv](https://arxiv.org/abs/2403.04243)

- **Foundations, one line each:**
  - Molnar, Kiss, Ames & Orosz, "Safety-Critical Control with Input Delay in Dynamic Environment", arXiv 2112.08445 (2021; accepted IEEE TCST). Environmental CBFs predict both robot and environment over the delay interval; Segway hardware. VERIFIED-ABS. [arXiv](https://arxiv.org/abs/2112.08445)
  - Kiss, Molnar, Ames & Orosz, "Control Barrier Functionals: Safety-critical Control for Time Delay Systems", arXiv 2206.08409 (2022). VERIFIED-ABS. [arXiv](https://arxiv.org/abs/2206.08409)
  - "Control Barrier Functions for Sampled-Data Systems with Input Delays", arXiv 2005.06418 (2020). SUMMARY; title from a search listing only. [arXiv](https://arxiv.org/pdf/2005.06418)

**Latency-aware safety with learned perception (closest to "learned")**

- **Saviolo, Picello, Mao, Verma & Loianno, "Reactive Collision Avoidance for Safe Agile Navigation", arXiv 2409.11962, 2024-09-18.** VERIFIED-ABS.
  - *Method:* NMPC with adaptive CBFs. A neural network refines noisy RGB-D depth, and points with minimum time-to-collision are prioritized. The abstract frames separate perception, planning and control as causing "compounded errors and delays".
  - *Does NOT (per abstract):* derive a latency bound. [arXiv](https://arxiv.org/abs/2409.11962)

- **Thumm, Frei, Ni, Althoff & Pavone, "Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees", arXiv 2604.15221, 2026-04-16.** VERIFIED-ABS + VERIFIED-HTML.
  - *Latency accounting:*
    - Median pipeline latency 65.86 ms on an RTX 5090: 2D pose 13.20 ms, OOD detector 36.39 ms.
    - Between prediction timesteps, SARA shield extends conformal set radii by `Δr = Δt·v_h,max` with the ISO maximum human speed v_h,max = 2.0 m/s.
  - *Results:* conformal sets 7.6× smaller in volume than the ISO 13855 model; probability of dangerous failure per hour (PFHD) bounded at 9.5e-7 with 99.999% confidence on the test distribution.
  - *Hardware:* Franka with a RealSense 435i camera.
  - *Learned:* yes. Learned pose estimation and motion prediction, with conformal calibration. Not a world model.
  - Occlusion and dropout are covered in Q2. [arXiv](https://arxiv.org/abs/2604.15221), [HTML](https://arxiv.org/html/2604.15221)

**Failure-prediction lead time (the "lead time" side of the timing relation; not safety filters)**

- **Römer, Kobras, Worbis & Schoellig, "Failure Prediction at Runtime for Generative Robot Policies" (FIPER), arXiv 2510.09459, 2025-10-10, NeurIPS 2025.** VERIFIED-ABS. Two signals: OOD detection via random network distillation (RND) in the policy embedding space, and action-chunk entropy, both calibrated with conformal prediction. Predicts failure before it happens. Does not relate perception latency to that lead time. [arXiv](https://arxiv.org/abs/2510.09459)

- **Zhang et al., "Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents", arXiv 2606.23085, 2026-06-22.** VERIFIED-ABS. World-model latents with functional conformal prediction; LIBERO-Long and ManiSkill-Long. A search snippet says it is invoked once per action-chunk boundary (SUMMARY). [arXiv](https://arxiv.org/abs/2606.23085)

- **Park et al., "Hide-and-Seek in Trajectories", arXiv 2605.30834, 2026-05-29, NeurIPS 2026.** VERIFIED-ABS: VLA failure localization. SUMMARY: per a search snippet, it raises alarms "slightly before the annotated failure onset". [arXiv](https://arxiv.org/abs/2605.30834)

- **Chen, "AEGIS: A Backup Reflex for Physical AI", arXiv 2606.06660, 2026-06-04.** VERIFIED-ABS. An activation probe detects high-risk steps "while there is still time to act" and escalates to a stronger policy. Recovers 10.1% of lost LIBERO-Spatial trajectories, versus 4.6% for blind escalation. [arXiv](https://arxiv.org/abs/2606.06660)

### Inferences
- **The latency budget is linear in the closing speed and the decel authority.** Every latency-aware result above reduces to the same shape: the needed clearance is about `v·t_latency + v²/(2a)`. Equivalently, `dv_max/dt_latency = −a` (2608.01855). In the RADAR example, 10 ms of latency costs 0.25 m/s of safe speed at a = 25 m/s². Applied to arm-arm safety, the required clearance at a closing speed is about `v_close·(t_sense + t_compute + t_actuate) + v_close²/(2·a_brake)`, with a_brake bounded by the torque limits. This is my extrapolation; no paper does it for manipulators.
- **The SARA-shield inflation gives a concrete HRC number.** 65.86 ms × 2.0 m/s ≈ 0.13 m of extra reachable-set radius from pipeline latency alone (my arithmetic from VERIFIED-HTML values).
- **The CBF-placement result supports putting the hard safety layer in the lowest-latency loop.** That means the proprioceptive or joint-state loop, with perception-driven or predictive safety layered above it. The delayed remote predictive filter loses disturbance tolerance roughly geometrically in delay plus horizon, through Lipschitz sums (2603.29792).
- **Delay shields give a per-state "tolerable latency" map that could be ported to a value function.** For a reachability value V, the analogue is the largest k such that the k-step expanded BRT (generalizing 2609.17904's one-step expansion) still excludes the current belief set. That is an inference, not a published method.
- **Lead time and latency are reported separately.** Failure-prediction papers (FIPER, Foresight, AEGIS, Hide-and-Seek) report detection lead time. None of them subtracts perception or monitor latency from it to give net actionable time.

### Gaps
- No paper derives a required perception or observation latency for a learned latent safety filter (DINO-WM / Dreamer-style latent HJ), or for any manipulation task. Latent filters report inference time only. A search snippet cites a maximum filter inference of 19.3 ms, enough for 50 Hz, but it was unclear which paper this came from (possibly 2509.14758; SUMMARY, unverified).
- No HJ-reachability paper with explicit state or measurement delay (2023-2026) was found. The searches returned only the discrete-sampling expansion (2609.17904).
- Measurement (sensing) delay, as distinct from input delay, is rarely treated separately in 2023-2026 CBF papers. The predictor approach covers both, but no paper isolates "camera latency" from actuation delay in a manipulator safety filter.
- A 2024 "latency cascade" SSM study reporting total latency below 140 ms surfaced only in a snippet with no identifiable primary source (SUMMARY, not used).

---

## Q2. Which methods handle occlusion or temporary loss of a tracked object (occlusion-aware reachability, phantom or hidden agents, propagating possible states during loss), and sensor dropout?

### Takeaway
Occlusion-aware safety is mature in driving and mobile navigation. The standard recipe:
- Spawn phantom or hidden agents at occlusion boundaries.
- Grow their forward reachable sets at maximum speed until they are observed.
- Plan or filter against those sets.
- Optionally keep a contingency (backup) plan.

The strongest recent safety-filter instance is **OcclusionCBF (2609.06342)**: backup CBFs over reachable occupancy, with recursive feasibility, hardware tests and about 2-5 ms per update.

Belief-space CBFs carry the belief through measurement loss:
- BCBF (2309.06499)
- Reach-avoid POMDP plus BCBF with conformal guarantees (2603.10572)
- RFS-BCBF (2607.15016), which explicitly handles objects entering and leaving view

Dropout and packet loss are handled formally only in stochastic linear or polynomial networked settings (Akbarzadeh/Lavaei line).

Track loss of a *previously seen* object is rarely modeled as its own case. OcclusionCBF does not model it. Sequential reasoning (Nyberg/Tumova line, HC-MTS) and Thumm et al.'s replace-with-prediction are the closest.

### Cited Findings

**Occlusion-aware reachability / hidden agents (driving & navigation)**

- **Kim, Park, Wada, Atanasov, Koga & Panagou, "OcclusionCBF: Backup Control Barrier Functions for Safe Navigation Among Hidden Dynamic Obstacles", arXiv 2609.06342, 2026-09-06.** VERIFIED-ABS + VERIFIED-HTML.
  - *Problem:* "robots navigating under occlusion may enter states from which no admissible input can avoid a dynamic obstacle once it becomes visible."
  - *Method:*
    - Occluded regions are enclosed in convex polytopes whose facets expand by `r_col + v̄_o·s`. The term b_jl(t) carries motion since the last observation.
    - A backup policy rollout is certified against this collision-inflated occupancy and a verified terminal set, giving affine QP constraints.
    - Proves recursive feasibility and collision avoidance "for every hidden-obstacle motion covered by the occupancy prediction".
  - *Experiments:*
    - 100 randomized trials per density with 10-50 moving discs, plus MetaUrban.
    - Unicycle success 86% vs below 70% for baselines.
    - Hardware: TurtleBot 4 vs TurtleBot 3 blind crossing; plain CBF-QP collides while OcclusionCBF intervenes before detection.
    - 1.6-4.5 ms per update vs 48-186 ms for MPC baselines.
  - *Does NOT:*
    - model sensing latency;
    - model track loss of a previously seen obstacle as a distinct case;
    - handle robustness to occupancy-prediction errors;
    - give inter-sample guarantees ("enforces only the sampled constraints");
    - use learned models.
  - [arXiv](https://arxiv.org/abs/2609.06342), [HTML](https://arxiv.org/html/2609.06342)

- **Nyberg, Gautier & Tumova, "Hope for the Best, Prepare for the Worst: Occlusion-Aware Contingency Planning for Autonomous Vehicles", arXiv 2607.03155, 2026-07-03.** VERIFIED-ABS.
  - *Method:* reachability "to sequentially determine the possible states of hidden traffic participants", combined with a tree planner "capable of reasoning over future observations and the absence thereof". Guarantees collision avoidance with possible hidden participants while being less conservative.
  - *Experiments:* one challenging simulated scenario.
  - *Does NOT:* include hardware, latency or learned models. [arXiv](https://arxiv.org/abs/2607.03155)
  - Precursor (SUMMARY, arXiv ID not verified): Sánchez, Nyberg, Pek, Tumova & Törngren, "Foresee the Unseen: Sequential Reasoning about Hidden Obstacles for Safe Driving" (IV 2022). Uses a point-mass hidden-obstacle state set refined by reachability and previous observations. [TU Delft portal](https://research.tudelft.nl/en/publications/foresee-the-unseen-sequential-reasoning-about-hidden-obstacles-fo/)

- **Zheng, Yang, Zheng, Peng, Wang & Ma, "Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Driving", arXiv 2502.06359, 2025-02-10.** VERIFIED-ABS.
  - *Method:* forward reachable sets of phantom vehicles give risk-aware dynamic velocity bounds. These feed a biconvex NLP with spatiotemporal barrier constraints that optimizes an exploration trajectory and a fallback trajectory, decomposed with consensus ADMM.
  - *Experiments:* simulation plus real-world occluded intersections.
  - *Does NOT:* model latency or learned models. [arXiv](https://arxiv.org/abs/2502.06359)

- **Park, Choi, Chin, Lee & Baek, "Occlusion-aware Risk Assessment and Driving Strategy for Autonomous Vehicles Using Simplified Reachability Quantification", arXiv 2306.07004, 2023-06-12, RA-L 2023.** VERIFIED-ABS.
  - *Method:* phantom-agent reachability with a simple state distribution gives a risk-based speed limit.
  - *Results:* collision rate down by up to 6.14×, intersection traversal time up by 1.48×, under 5 ms constant-time computation, in simulation.
  - *Does NOT:* give a formal guarantee; it is risk-based. [arXiv](https://arxiv.org/abs/2306.07004)

- **van der Ploeg, Nyberg, Gaspar Sánchez, Silvas & van de Wouw, "Overcoming the Fear of the Dark: Occlusion-Aware Model-Predictive Planning for Automated Vehicles Using Risk Fields", arXiv 2309.15501, 2023-09-27, T-ITS submission.** VERIFIED-ABS (truncated). Risk-field MPC that accounts for occluded entities; soft or risk formulation rather than hard guarantees. [arXiv](https://arxiv.org/abs/2309.15501)

- **Tan, Lei & Ukkusuri, "Worst-Case Hidden-Vehicle Trajectory Search in Spatiotemporal Occlusion Regions" (HC-MTS), arXiv 2609.20480, 2026-09-17, ECCV 2026 SDAD workshop (oral, best paper).** VERIFIED-ABS.
  - *Method:* finite hidden-state modes, each "certified by a backward witness satisfying multi-frame visibility, occupancy, semantic-map support, and class-specific kinematic constraints". A bilevel minimax search finds the worst history-consistent hidden-vehicle trajectory.
  - *Experiments:* eight Waymo Open Motion scenarios.
  - *Does NOT:* act as a runtime safety filter; it is a stress-test search. [arXiv](https://arxiv.org/abs/2609.20480)

- **Jia, Shu, Gan & Ding, "Learning Occlusion-aware Decision-making from Agent Interaction via Active Perception" (Pad-AI), arXiv 2409.17618, 2024-09-26, IEEE IV.** VERIFIED-ABS. Learned RL over semantic motion primitives with prediction-integrated "risk-aware learning and security guarantees". The guarantee claim is not specified in the abstract. Closed-loop simulation. [arXiv](https://arxiv.org/abs/2409.17618)

- **Zhou, Shaikh, Chaubey, Haggerty, Koga, Panagou & Atanasov, "Control Strategies for Pursuit-Evasion Under Occlusion Using Visibility and Safety Barrier Functions", arXiv 2411.01321, 2024-11-02.** VERIFIED-ABS.
  - *Problem:* prevents track loss by keeping the evader in the field of view despite line-of-sight obstructions.
  - *Method:* the FoV SDF is a visibility CBF and the obstacle SDF a safety CBF, using generalized gradients. A kinodynamic planner supplies the reference.
  - *Experiments:* CARLA and real robots with onboard sensing.
  - *Does NOT:* guarantee safety *after* the target is lost. [arXiv](https://arxiv.org/abs/2411.01321)

- **Yuan, Cao, Lee, Moon & Scherer, "SCOPE: Field-of-View-Aware Path Planning in Unknown Space via Safety-Volume Certification", arXiv 2608.04420, 2026-08-05.** VERIFIED-ABS. Requires "the complete robot-inflated volume of an intended motion to be observed and verified free before execution", using a certified graph and observation obligations. Static unknown space only, as far as the abstract states. [arXiv](https://arxiv.org/abs/2608.04420)

- **Foundations, one line each:**
  - Orzechowski, Meyer & Lauer, "Tackling Occlusions & Limited Sensor Range with Set-based Safety Verification", arXiv 1807.01262 (2018). Critical sensing-field edges each represent a hidden obstacle. VERIFIED-ABS. [arXiv](https://arxiv.org/abs/1807.01262)
  - Zhang & Fisac, "Safe Occlusion-aware Autonomous Driving via Game-Theoretic Active Perception", arXiv 2105.08169 (RSS 2021). Hybrid zero-sum game against an initially hidden pursuer. VERIFIED-ABS. [arXiv](https://arxiv.org/abs/2105.08169)
  - Firoozi, Mir, Camps & Schwager, "OA-MPC: Occlusion-Aware MPC for Guaranteed Safe Robot Navigation with Unseen Dynamic Obstacles", arXiv 2211.09156 (2022). FRS of potential occluded agents inside NMPC. VERIFIED-ABS. [arXiv](https://arxiv.org/abs/2211.09156)

**Belief propagation during loss / partial observability (CBF in belief space)**

- **Vahs, Pek & Tumova, "Belief Control Barrier Functions for Risk-aware Control", arXiv 2309.06499, 2023-09-12, RA-L.** VERIFIED-ABS. BCBFs over EKF beliefs, so the robot stays in the safe region "with desired confidence". Demonstrated on a quadrotor "exposed to external disturbances and varying sensing conditions"; control up to 1 kHz. [arXiv](https://arxiv.org/abs/2309.06499)

- **Vahs, Verhagen & Tumova, "Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control", arXiv 2603.10572, 2026-03-11.** VERIFIED-ABS.
  - *Method:* a layered certificate architecture in belief space.
    - Belief CLFs make information gathering a Lyapunov problem; they are learned with RL.
    - BCBFs use conformal prediction for probabilistic finite-horizon safety.
    - Synthesis reduces to lightweight QPs.
  - *Learned:* yes, via RL for the BCLF; not a world model. [arXiv](https://arxiv.org/abs/2603.10572)

- **Han, Chen, Cai, Torroba, Stenius, Jensfelt, Alonso-Mora & Tumova, "Risk-Aware Belief Control Barrier Functions over Random Finite Sets", arXiv 2607.15016, 2026-07-16.** VERIFIED-ABS.
  - *Problem:* "an unknown and time-varying number of moving objects from noisy, incomplete measurements". The search snippet adds that limited range and occlusion make objects enter and leave view.
  - *Method:* SMC-PHD particle belief, a nonsmooth BCBF, forward invariance under continuous prediction, and "an explicit condition under which discrete updates preserve safety". This is the closest formal treatment of belief propagation across measurement gaps.
  - *Experiments:* simulation plus real underwater tests. [arXiv](https://arxiv.org/abs/2607.15016)

- **Enwerem, Baras & Belta, "Risk-Constrained Belief-Space Optimization for Safe Control under Latent Uncertainty", arXiv 2604.03868, 2026-04-04, CDC 2026.** VERIFIED-ABS.
  - *Method:* belief over an unknown time-invariant parameter, such as "unobserved geometry"; CVaR-constrained belief-space MPPI with a probabilistic safety guarantee.
  - *Does NOT:* handle a time-varying hidden state or dropout. [arXiv](https://arxiv.org/abs/2604.03868)

**Dropout, packet loss, degraded or attacked sensors**

- **Akbarzadeh, Soudjani & Lavaei, "Safety Barrier Certificates for Stochastic Control Systems with Wireless Communication Networks", arXiv 2309.05570, 2023-09-11.** VERIFIED-ABS. Control barrier certificates with packet loss and end-to-end delays, giving a lower bound on safety probability. [arXiv](https://arxiv.org/abs/2309.05570)

- **Akbarzadeh, Mamduhi & Lavaei, "Safety Controller Synthesis for Stochastic Networked Systems under Communication Constraints", arXiv 2507.15031, 2025-07-20.** VERIFIED-ABS. Constant uplink delay plus Bernoulli loss on both channels, handled through an augmented state; discrete-time stochastic linear systems. [arXiv](https://arxiv.org/abs/2507.15031)

- **Akbarzadeh, Ashoori, Mamduhi & Lavaei, "Safe Packetized Control for Stochastic Constrained Networked Systems", arXiv 2606.24148, 2026-06-23.** VERIFIED-ABS. Polynomial stochastic systems; a buffer at the plant stores packetized safe control sequences against downlink loss. [arXiv](https://arxiv.org/abs/2606.24148)

- **Tan, Sundar, Bruzzone, Ong, Lunardi, Andreoni, Tabuada & Ames, "Secure Safety Filter: Towards Safe Flight Control under Sensor Attacks", arXiv 2505.06845, 2025-05-11, IROS 2025 submission.** VERIFIED-ABS. A secure state reconstructor generates plausible states and feeds a CBF filter, with bounded noise and reduced-order models. Software-in-the-loop plus drone hardware. [arXiv](https://arxiv.org/abs/2505.06845)

- **Zongo & Wei, "Runtime Safety Filtering for Learned Small UAS Separation Policies under GNSS Degradation", arXiv 2607.10014, 2026-07-10, DASC 2026.** VERIFIED-ABS.
  - *Method:* estimate the worst-case state consistent with bounded observation uncertainty, then compare a DT-CBF action filter against observation filtering (feeding the worst-case state to the policy).
  - *Result:* the action filter gave "negligible" improvement, while observation filtering cut near mid-air collisions by 90%.
  - *Learned:* yes (RL policy). [arXiv](https://arxiv.org/abs/2607.10014)

- **Yun & Azizan, "ATOM-CBF: Adaptive Safe Perception-Based Control under Out-of-Distribution Measurements", arXiv 2511.08741, 2025-11-11.** VERIFIED-ABS. OOD-aware adaptive perception-error margin inside the CBF; F1Tenth with LiDAR and a quadruped with RGB, in simulation. [arXiv](https://arxiv.org/abs/2511.08741)

- **Thumm et al., 2604.15221** (see Q1). VERIFIED-HTML.
  - *Handling pose dropout:* "If the pose is classified as OOD, we discard the estimated 3D pose and replace it by the first entry of the current motion prediction buffer." A validity flag tracks real vs predicted poses, and N_req recent valid poses are needed before new predictions are accepted. OOD handling cuts invalid predictions by 24% at a 1.9% accuracy cost.
  - *Does NOT:* give coverage guarantees for the reused predictions, or handle humans entering or leaving ("left to future work"). [HTML](https://arxiv.org/html/2604.15221)

- **Ong & Ames, "Intermittent Safety Filters for Event-Triggered Safety Maneuvers with Application to Satellite Orbit Transfers", arXiv 2304.08684, 2023-04-18, CDC 2023 submission.** VERIFIED-ABS. **Caution: despite the name, this is intermittent *actuation*** (event-triggered switching of the safety filter), not intermittent observation. Satellite simulation only. [arXiv](https://arxiv.org/abs/2304.08684)

- **Foundation, one line: Sinopoli et al., "Kalman Filtering with Intermittent Observations" (2004).** SUMMARY. Critical arrival-probability threshold for bounded error covariance. [PDF](https://people.eecs.berkeley.edu/~jordan/papers/sinopoli-etal.pdf)

**Learned / world-model side (observation gaps, delays)**

- **Kim, Nakamura & Bajcsy, "How Well Do Latent World Models Understand Partially Observable Safety Constraints?" (arXiv title; v1 title "What You Don't Know Can Hurt You: How Well do Latent Safety Filters Understand Partially Observable Safety Constraints?"), arXiv 2510.06492, 2025-10-07.** VERIFIED-ABS + VERIFIED-HTML.
  - *Failure modes defined:*
    - **Estimation gap:** current observations do not reveal the safety quantity.
    - **Prediction gap:** "failures are observable once they occur but cannot be reliably anticipated".
  - *Diagnostics:* a mutual-information safety-observability measure; for wax melting, RGB scores 0.221 vs IR 0.801. Also a 16-step rollout predictability test.
  - *Hardware tasks (Franka FR3):*
    - Wax melting (RGB+IR): privileged IR supervision raised the lift rate from 15% to 100%.
    - Rice pouring from an opaque bottle (RGB+GelSight): conformal risk calibration gave about 14% spill on filled bottles; multimodal allowed 60% execution on empty bottles vs 18% for RGB.
  - *Does NOT:* model occlusion as such, track loss, or latency. Its stated limitations: "how best to inject the missing safety information remains unclear". [arXiv](https://arxiv.org/abs/2510.06492), [HTML](https://arxiv.org/html/2510.06492)

- **Nakamura, Peters & Bajcsy, "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis", arXiv 2502.00935, 2025-02-02, RSS 2025.** VERIFIED-ABS. The base latent safety filter (latent HJ in a generative WM). Assumes the latent captures safety-relevant state; no latency or dropout model. [arXiv](https://arxiv.org/abs/2502.00935)

- **Nakamura, Bishop, Man, Johnson, Manchester & Bajcsy, "How to Train Your Latent Control Barrier Function", arXiv 2511.18606, 2025-11-23.** VERIFIED-ABS. Shows that current latent HJ value functions are "fundamentally incompatible" with smooth CBF-QP filtering and proposes fixes. Not about observation gaps. [arXiv](https://arxiv.org/abs/2511.18606)

- **Karamzade, Kim, Lanier, Corsi & Fox, "Model-Based Reinforcement Learning under Random Observation Delays", arXiv 2509.20869, 2025-09-25.** VERIFIED-ABS.
  - *Problem:* random, out-of-sequence sensor delays in POMDPs.
  - *Method:* model-based belief filtering inside Dreamer; shows that frame stacking is insufficient.
  - *Learned:* latent WM, yes. *Safety:* none. [arXiv](https://arxiv.org/abs/2509.20869)

- **Mednikov & Gal, "Decoupled Delay Compensation: Enhancing Pre-trained MARL Policies via Learned Dynamics Filtering", arXiv 2605.26286, 2026-05-25.** VERIFIED-ABS. Learned gated transition model plus Kalman layer to estimate current states under stale observations and packet loss. No safety. [arXiv](https://arxiv.org/abs/2605.26286)

- **Mohammed et al., "Tracking the Unseen: An Occlusion-Robust Framework for Target Tracking Under Full and Long-Term Occlusion", arXiv 2609.17427, 2026-09-15.** VERIFIED-ABS. YOLOv11n, Kalman prediction during occlusion, and re-ID after reappearance. Perception only; no safety. [arXiv](https://arxiv.org/abs/2609.17427)

### Inferences
- **The common recipe for keeping safety through loss is the same everywhere.** During the gap, the hidden object's set is propagated with worst-case bounded dynamics, so it grows at roughly `v_max·Δt`. Reappearance shrinks it. The controller certifies a backup maneuver against the grown set. This appears in OcclusionCBF, OA-MPC, Nyberg/Tumova, SARA's `Δt·v_h,max`, and the delay shields' ε-ball. Track loss and latency are therefore the same mathematical object: an age-of-information-dependent set inflation.
- **Belief-space CBFs are the probabilistic analogue.** RFS-BCBF (2607.15016) is the only one found with an explicit condition for safety across discrete measurement updates.
- **Observation-side filtering may beat action-side filtering for learned policies under degraded sensing.** The evidence is 2607.10014's 90% vs negligible result. That is one paper with one domain (sUAS separation), so it is weak evidence.
- **No latent-world-model safety filter propagates a reachable or belief set during observation loss.** The Bajcsy-lab results (2510.06492) diagnose the observability gap but do not supply a loss-time propagation rule.

### Gaps
- Track loss of a *previously tracked* object, followed by re-acquisition with a possible identity mismatch, is not a distinct case in any safety filter found. OcclusionCBF explicitly does not model it, per its HTML.
- No safety filter combining occlusion reasoning with explicit sensing latency was found. OcclusionCBF does not model latency; the RADAR paper does not handle occlusion.
- Koschi & Althoff-style set-based prediction of occluded traffic participants (pre-2023) was not re-verified here.

---

## Q3. Is there MANIPULATION work where the manipulated or carried object is occluded by the robot's own arms or grippers, and safety is kept during the occlusion?

### Takeaway
I found **no paper that maintains a formal safety guarantee for a grasped or carried object while the robot's own arm or gripper occludes it.** The nearest neighbours each cover only part of the problem:
- **CURA-PPO (2602.01731, ICRA 2026):** the pushed object occludes the onboard sensor, so hidden obstacles become a collision risk. It uses learned RL, risk/uncertainty and confidence maps, with no formal guarantee.
- **Visibility-aware mobile grasping (2605.02487):** handles unobserved obstacles plus target visibility during grasping.
- **Bekris-lab occlusion-aware manipulation (2205.11719, 2022):** conservative occluded volume, with a safety guarantee, for static clutter.
- **Self-occlusion avoidance:** visibility maximization (2202.12557, 2022) and probabilistic occlusion-free IBVS CBFs (2309.03476). These prevent occlusion rather than keep safety during it.
- **Rice-pouring prediction gap (2510.06492):** the closest match to "contents of a carried container are unobservable", but it concerns contents, not arm occlusion.

Active-perception bimanual work (AV-ALOHA, MAAP) and visuo-haptic tracking (V-HOP) fight arm or gripper occlusion with no safety layer.

### Cited Findings

- **Hwang, Yang, Jeong, Yoon & Yoon, "Uncertainty-Aware Non-Prehensile Manipulation with Mobile Manipulators under Object-Induced Occlusion" (CURA-PPO), arXiv 2602.01731, 2026-02-02, ICRA 2026.** VERIFIED-ABS.
  - *Problem:* "the manipulated object occludes the sensor's field of view, creating occluded regions that can lead to collisions."
  - *Method:* RL predicts collision possibility as a distribution, from which risk and uncertainty are extracted; uncertainty drives active perception, and confidence maps encode observation reliability.
  - *Result:* up to 3× higher success than baselines.
  - *Learned:* yes (PPO). *Does NOT:* give a formal safety guarantee or involve a grasped or carried object; it is pushing. [arXiv](https://arxiv.org/abs/2602.01731)

- **Hu, Xiao, Hsu & Zhang, "Visibility-Aware Mobile Grasping in Dynamic Environments", arXiv 2605.02487, 2026-05-04.** VERIFIED-ABS.
  - *Problem:* two visibility constraints. Incomplete observations create collision risk along the whole-body trajectory, and occlusions hinder target pose refinement.
  - *Method:* a velocity-aware gaze policy, a subgoal policy that maintains target visibility, and a real-time whole-body replanner.
  - *Does NOT (per abstract):* include a formal safety certificate, or address occlusion of an already-grasped object. [arXiv](https://arxiv.org/abs/2605.02487)

- **Miao, Wang & Bekris, "Safe, Occlusion-Aware Manipulation for Online Object Reconstruction in Confined Spaces", arXiv 2205.11719, 2022-05-24 (foundation).** VERIFIED-ABS. Framework "that ensures safety with completeness guarantees". Per the search snippet (SUMMARY), a conservative volume containing occluded space is used for collision checking. Static scenes, simulation. [arXiv](https://arxiv.org/abs/2205.11719)

- **He, Newbury, Tran, Haviland, Burgess-Limerick, Kulić, Corke & Cosgun, "Visibility Maximization Controller for Robotic Manipulation", arXiv 2202.12557, 2022-02-25 (foundation).** VERIFIED-ABS. "Occlusions caused by a robot's own body"; line-of-sight visibility as a *soft* constraint in a reactive QP. Reports a trade-off between occlusion rate and task metrics. No safety guarantee during occlusion. [arXiv](https://arxiv.org/abs/2202.12557)

- **Zhang, Yang & Luo, "Occlusion-Free Image Based Visual Servoing using Probabilistic Control Barrier Certificates", arXiv 2309.03476, 2023-09-07, IFAC WC 2023.** VERIFIED-ABS. Chance-constrained CBF keeps feature points in the FoV and unoccluded by obstacles with a given probability, inside MPC. Simulation only. Prevents occlusion; does not keep safety once features are lost. [arXiv](https://arxiv.org/abs/2309.03476)

- **Kim, Nakamura & Bajcsy, 2510.06492 (see Q2).** VERIFIED-HTML. Rice pouring with an opaque bottle on a Franka. The safety-relevant state (fill level) is hidden, giving a "prediction gap" handled by conformal shielding thresholds. The closest manipulation analogue to hidden carried-container contents. No arm occlusion and no latency model.

- **Zheng, Yu, Peng, Shi, Li, Zhang, Li, Wang, Lu & Jia, "Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation", arXiv 2606.18960, 2026-06-17, CoRL 2026.** VERIFIED-ABS. Targets "frequent end-effector occlusions and rapid wrist-camera motion" in action-conditioned video WMs using a surfel-indexed memory (W-VMem). Learned WM; no safety. [arXiv](https://arxiv.org/abs/2606.18960)

- **Huang, Yuan, Kim, Pradhan, Chen, Li & Hermans, "Out of Sight, Still in Mind: Reasoning and Planning about Unobserved Objects with Video Tracking Enabled Memory Models" (DOOM/LOOM), arXiv 2309.15278, 2023-09-26, ICRA 2024.** VERIFIED-ABS. Object-permanence memory for occluded objects in multi-object manipulation planning; simulation and real experiments. No safety. [arXiv](https://arxiv.org/abs/2309.15278)

- **Li, Jia, Akbulut, Xiang, Konidaris & Sridhar, "V-HOP: Visuo-Haptic 6D Object Pose Tracking", arXiv 2502.17434, 2025-02-24, RSS 2025.** VERIFIED-ABS. Fuses haptics with vision for in-hand pose tracking that stays coherent when vision is degraded; works across gripper embodiments. Perception only; no safety. [arXiv](https://arxiv.org/abs/2502.17434)

- **Nagahama, Wan, Hu & Harada, "Bimanual Regrasp Planning and Control for Active Reduction of Object Pose Uncertainty", arXiv 2503.22240, 2025-03-28.** VERIFIED-ABS. Reduces grasped-object pose uncertainty "without using a fixture or a camera" through three orthogonal flat-pad grasps. Relevant as camera-free pose certainty; no safety. [arXiv](https://arxiv.org/abs/2503.22240)

- **Active perception against arm occlusion in bimanual setups (no safety):**
  - Chuang et al., "Active Vision Might Be All You Need: Exploring Active Vision in Bimanual Robotic Manipulation" (AV-ALOHA), arXiv 2409.17435, 2024-09-26. VERIFIED-ABS. [arXiv](https://arxiv.org/abs/2409.17435)
  - Chen et al., "MAAP: Multi-Agent Active Perception for Collaborative Manipulation", arXiv 2609.21929, 2026-09-18. Every arm's wrist camera serves as a moving viewpoint. Dual-arm placement: 14/20 for MAAP+RAIL vs 0/20 for fixed-view ACT. VERIFIED-ABS. [arXiv](https://arxiv.org/abs/2609.21929)

- **"PGRF: A robot manipulation keypoint tracking framework integrating manipulator posture and grasp information" (ACM DL, 10.1145/3778886.3778900).** SUMMARY. Title from search only; the page returned HTTP 403, so the content is unverified. The title suggests it uses arm posture and grasp state to keep keypoint tracking through gripper occlusion. [ACM](https://dl.acm.org/doi/full/10.1145/3778886.3778900)

### Inferences
- **For a rigidly grasped object, self-occlusion is mostly not a geometric safety problem.** Its pose is available from proprioception and forward kinematics through the grasp, provided the grasp has not slipped. What occlusion actually hides is (a) grasp slip and (b) contents, such as liquid or beads. Both are better modeled as a hidden-state belief propagated from proprioception and touch than as a phantom-agent reachable set. This matches 2510.06492's estimation and prediction gaps and V-HOP / 2503.22240's non-visual pose sources.
- **The OcclusionCBF-style recipe would transfer to occluded *free* objects** (another arm's target, a human hand) on the manipulation side, by growing the occupancy at the object's maximum velocity during occlusion. No manipulation paper does this with a backup-CBF certificate.

### Gaps
- No manipulation safety filter (CBF, HJ or latent) was found that maintains guarantees while a carried object, or another arm's object, is occluded by the robot's own links.
- No bimanual paper treats one arm occluding the other arm's object as a safety-relevant observation gap.
- The PGRF content could not be read (403).

---

## Q4. Have point-tracking failures in VLM-keypoint pipelines (ReKep's largest failure source) been addressed by any follow-up with safety guarantees?

### Takeaway
**No follow-up was found that gives safety guarantees under keypoint or point-tracking failure.** Follow-ups take three routes, none with guarantees under tracking loss:
- **Work around tracking:** OmniManip swaps keypoint tracking for 6D object pose tracking; Code-as-Monitor uses simplified "constraint elements"; SuperPose-type pipelines combine FoundationPose and SAM2 for tracking-loss recovery.
- **Add detection or monitoring without guarantees:** CaM, closed-loop multi-agent verification.
- **Add CBF safety layers that assume perception is correct:** VLSA/AEGIS, semantic safety filters.

ReKep's own failure breakdown (48.7% point tracking, 17.9% keypoint proposal) is SUMMARY here because it was already known; it came from a third-party summary snippet.

### Cited Findings
- ReKep failure breakdown: point tracking 48.7%, keypoint proposal 17.9%, "primarily due to occlusions during manipulation". SUMMARY, via a third-party snippet. [EmergentMind](https://www.emergentmind.com/topics/relational-keypoint-constraints-rekep), [ReKep arXiv](https://arxiv.org/abs/2409.01652)

- **Zhou, Su, Chi, Zhang, Wang, Huang, Sheng & Wang, "Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection", arXiv 2412.04455, 2024-12-05, CVPR 2025.** VERIFIED-ABS.
  - *Method:* VLM-generated code monitors spatio-temporal constraints. "Constraint elements" abstract entities into compact geometric elements, which "simplifies tracking".
  - *Result:* 28.7% higher success and 31.8% shorter execution time under severe disturbances.
  - *Does NOT:* give formal guarantees or explicitly handle track loss. [arXiv](https://arxiv.org/abs/2412.04455)

- **Pan, Zhang, Wu, Zhao, Gao & Dong, "OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints", arXiv 2501.03841, 2025-01-07.** VERIFIED-ABS (truncated).
  - *Method:* object canonical-space interaction primitives. Per the HF and alphaXiv snippets (SUMMARY), it has a dual closed loop whose low-level execution loop uses "6D pose tracking", replacing point tracking.
  - *Does NOT:* give safety guarantees. [arXiv](https://arxiv.org/abs/2501.03841)

- **Zhang, Ni, Mohapatra, Liu, Zhang & Zhu, "EmboAlign: Aligning Video Generation with Compositional Constraints for Zero-Shot Manipulation", arXiv 2603.05757, 2026-03-05.** VERIFIED-ABS. Explicitly cites "cumulative errors from imperfect depth estimation and keypoint tracking"; uses VLM constraints for rollout selection. No guarantees. [arXiv](https://arxiv.org/abs/2603.05757)

- **Deng et al., "SuperPose: Improved 6D Pose Estimation with Robust Tracking and Mask-Free Initialization", arXiv 2409.19986, 2024-09-30.** VERIFIED-ABS. FoundationPose plus SAM2 plus LightGlue to fix "tracking loss"; industrial. Perception only. [arXiv](https://arxiv.org/abs/2409.19986)

- **Hu et al., "VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer" (AEGIS), arXiv 2512.11891, 2025-12-09, IROS 2026.** VERIFIED-ABS. CBF safety layer on VLAs; SafeLIBERO benchmark. The abstract does not address perception or tracking uncertainty. [arXiv](https://arxiv.org/abs/2512.11891)

- **Brunke, Zhang, Römer, Naimer, Staykov, Zhou & Schoellig, "Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards", arXiv 2410.15185, 2024-10-19, RA-L 10(5) 2025.** VERIFIED-ABS. Semantic map plus LLM-inferred unsafe conditions (e.g., "rotating a cup of water is unsafe"), certified with CBFs. The abstract does not state handling of occlusion or tracking loss. [arXiv](https://arxiv.org/abs/2410.15185)

- **He et al., "A Closed-Loop Multi-Agent Framework for Robust Multi-Robot Manipulation", arXiv 2607.06990, 2026-07-08, RSS 2026.** VERIFIED-ABS. A Verification Agent monitors execution; LLM-based, with no formal guarantees. [arXiv](https://arxiv.org/abs/2607.06990)

- **Lips, Moletta, Welle, Kragic & wyffels, "On the Generalization Capabilities, Design Choices and Limitations of Keypoint Imitation Learning", arXiv 2605.26649, 2026-05-26.** VERIFIED-ABS. Over 2000 real rollouts; KIL reaches 75% success. A limitations study; no safety. [arXiv](https://arxiv.org/abs/2605.26649)

### Inferences
- **The gap is open.** Nothing found couples a keypoint tracker's confidence or visibility flag to a safety filter that falls back to a reachable or belief set of the lost keypoint. Thumm et al.'s OOD-to-predicted-pose replacement (2604.15221) plus set inflation by `Δt·v_max` is the nearest transferable template, from HRC.
- **The perception-robust CBF line is the natural theory to transplant**, but none of it has been applied to VLM keypoints. That line includes ATOM-CBF (2511.08741), Secure Safety Filter (2505.06845) and measurement-robust CBFs.

### Gaps
- No ReKep follow-up with formal safety under tracking failure was found. The search covered CaM, OmniManip, EmboAlign, VLSA and semantic safety filters.
- OmniManip's 6D-pose-tracking claim is SUMMARY (from snippets), not read in full text.
- Measurement-Robust CBFs (Dean, Taylor, Cosner, Ames, ~2020) appeared only as a search-result title, so they were not verified here (SUMMARY). [ResearchGate](https://www.researchgate.net/publication/351463646_Measurement-Robust_Control_Barrier_Functions_Certainty_in_Safety_with_Uncertainty_in_State)
