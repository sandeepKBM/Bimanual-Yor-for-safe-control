# Learned and latent safety filters robust to observation noise, partial observability and state-estimation error (2023-2026)

Provenance and labels (read first). Web research only, 2026-10-05. No robot, device or private service was contacted.
- **VERIFIED**: I fetched the arXiv abstract page (and, where marked "+HTML", the arXiv HTML full text) in this session. WebFetch returns a model-written digest of the page, so the numbers quoted from HTML are as reliable as that digest.
- **SUMMARY**: the claim comes only from a search-result snippet or a third-party listing. Treat it as unconfirmed.
- **PRIOR-NOTE**: read in an earlier session of this project and not re-read now. Sources: `research_notes/Coverage versus caution latent safety/latent_hj_safety_filters.md`, and the memory file `latent-hj-literature-2026.md`.
- The brief listed some papers as already known: UNISafe, AnySafe, Santos, Tabbara 2608.00315, Dean, Cosner, perception contracts, ATOM-CBF, Lin & Bansal, Cao & Bansal, LatentCBF, ReKep and SafeBimanual. They appear here only as anchors for follow-up papers.
- Warning: one arXiv ID and date pair did not match, noted inline (2609.22108).

## Q1. Which papers compute HJ reachability or safety values in belief space, or for POMDPs (including learned or neural versions)?

### Takeaway
Belief-space safety values exist, but almost all model uncertainty about other agents' intent or about task and physical parameters. None model sensor noise in the robot's own state. The learned belief-space values are:
- Li/Tang/Azizan 2609.15915, a learned safety value over the information space;
- Hu 2606.02562, a neural belief-space filter verified with conformal prediction;
- Deception Game 2309.01267, adversarial RL over joint physical and learning dynamics.

The closest work on observation or POMDP uncertainty is not neural:
- Vahs/Verhagen/Tumova 2603.10572, belief CBFs plus learned belief CLFs;
- Jahanshahi & Chen 2608.13819, an analytic CBVF tightened by a conformal bound on the estimation error;
- RoVer-CoRe 2511.14755, grid HJ over a bounded estimation-error set. Its authors say outright that the true POMDP reformulation is future work.

### Cited Findings
- **Vahs, Verhagen, Tumova, "Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control"**, arXiv 2603.10572, v1 11 Mar 2026, v2 5 May 2026. VERIFIED (abstract level).
  - Uncertainty: POMDP with non-Gaussian beliefs (the abstract digest gives belief dimension >10^4).
  - Robustness: a "layered, certificate-based control architecture that operates directly in belief space". Belief CLFs for information gathering are learned with RL. Belief CBFs give probabilistic finite-horizon safety via conformal prediction. Everything reduces to real-time QPs.
  - Images or point clouds: none mentioned.
  - Experiments: simulation plus a real space-robotics platform, compared against constrained-POMDP solvers.
  - Does NOT: learn the safety value as a neural HJ function (the learned part is the information-gathering CLF), or use images or latent world models.
  - [arXiv](https://arxiv.org/abs/2603.10572)
- **Li, Tang, Azizan, "Safe Meta-Reinforcement Learning via Information Space Reachability"**, arXiv 2609.15915, 14 Sep 2026. VERIFIED.
  - Uncertainty: the agent's belief over the underlying task. The "information space" is physical state plus that belief.
  - Robustness: a safety value equal to "the probability of the agent avoiding unsafe regions indefinitely". It satisfies a self-consistency condition and a Bellman equation, so it can be learned with meta-RL, and it is used for both safety filtering and constrained policy optimization.
  - Images: none mentioned.
  - Experiments: "meta-RL benchmarks" only.
  - Does NOT: model noise in the physical-state observation (the belief is over the task, not the state); no hardware.
  - [arXiv](https://arxiv.org/abs/2609.15915)
- **Haimin Hu, "Permissive Safety Through Trusted Inference: Verifiable Belief-Space Neural Safety Filters for Assured Interactive Robotics"**, arXiv 2606.02562, 1 Jun 2026, WAFR 2026. VERIFIED (abstract).
  - Uncertainty: human preferences, goals, competency and willingness to cooperate, plus runtime inference errors, in a high-dimensional belief space.
  - Robustness: a neural belief-space safety filter, certified by conformal prediction only on the region "where inference is expected to be reliable". This certified a "significantly more permissive" filter than a standard conformal baseline.
  - Experiments: a simulated human-vehicle interaction benchmark.
  - Does NOT: perception or sensor noise on the robot's own state; images; hardware.
  - [arXiv](https://arxiv.org/abs/2606.02562)
- **Hu, Zhang, Nakamura, Bajcsy, Fisac, "Deception Game: Closing the Safety-Learning Loop in Interactive Robot Autonomy"**, arXiv 2309.01267, CoRL 2023 (v1 3 Sep 2023, v2 1 Nov 2023). VERIFIED.
  - Uncertainty: the robot's evolving belief, under both Bayesian belief propagation and "implicit learning through large pre-trained neural trajectory predictors".
  - Robustness: adversarial RL safety analysis over the joint physical and learning dynamics.
  - Does NOT: model observation noise or perception error; the abstract mentions no images.
  - [arXiv](https://arxiv.org/abs/2309.01267)
- **Hu, Isele, Bae, Fisac, shielding-aware dual control (active uncertainty reduction) with an HJ safety shield**, arXiv 2302.00171, IJRR. PRIOR-NOTE (owner opened the abstract).
  - The uncertainty is other drivers' intent, and the robot gathers information on purpose.
  - Not a perception-noise paper.
  - [arXiv](https://arxiv.org/abs/2302.00171)
- **Jahanshahi & Mo Chen (SFU), "Control Barrier-Value Functions under Partial Observability: Safety Guarantees via Conformal Prediction"**, arXiv 2608.13819, 13 Aug 2026. VERIFIED.
  - Uncertainty: estimation error under partial observability, plus bounded disturbances.
  - Robustness: they "apply conformal prediction to the estimation error and obtain an error bound at a user-chosen miscoverage level", tighten the CBVF (HJ + CBF) safety analysis with that bound, and get finite-horizon probabilistic guarantees for the true state, enforced by a QP filter.
  - Not learned or neural.
  - Experiments: a simulated, partially observable obstacle-avoidance case.
  - Does NOT: compute a belief-space value; learn a value; use images.
  - The prior note records their admission that "Larger bounds enlarge ... and shrink the certified safe set."
  - [arXiv](https://arxiv.org/abs/2608.13819)
- **Lin, Pinto, Bansal, "Robust Verification of Controllers under State Uncertainty via Hamilton-Jacobi Reachability Analysis" (RoVer-CoRe)**, arXiv 2511.14755, v1 18 Nov 2025, v2 15 Apr 2026, L4DC 2026. VERIFIED (+HTML).
  - Uncertainty: the estimate is x̂ = x + e with e in an element-wise bounded set. In the aircraft case the bounds come from "histograms of the prediction errors" of a camera DNN on X-Plane runway images.
  - Robustness: controller, observation function and estimator are concatenated into one closed loop, and a robust HJ value is computed with the grid-based `hj_reachability` toolbox on a 101^3 or 100^3 grid.
  - Experiments, both simulation only:
    - aircraft taxiing with a DNN camera estimator;
    - NN-based lunar-rover navigation whose visual-odometry error grows with time.
  - Does NOT: handle partial observability. The authors write: "Relaxing this assumption fundamentally alters the structure of the underlying differential game, requiring a reformulation of the value function to accommodate partial observability, which is a direction we leave for future work".
  - Does NOT: use neural values (scaling is deferred to "learning-based reachability tools"). It is a verifier, not a filter.
  - This is the direct follow-up to Lin & Bansal 2312.08604.
  - [arXiv](https://arxiv.org/abs/2511.14755); [HTML](https://arxiv.org/html/2511.14755)
- **Vahs, Pek, Tumova, "Belief Control Barrier Functions for Risk-aware Control"**, arXiv 2309.06499. SUMMARY.
  - Uses EKF beliefs, runs at up to 1 kHz, tested on a quadrotor under disturbances and varying sensing conditions.
  - Not learned.
  - [ar5iv](https://ar5iv.arxiv.org/html/2309.06499)
- Follow-ups to belief CBFs, both SUMMARY:
  - Vahs & Tumova, "Risk-aware Control for Robots with Non-Gaussian Belief Spaces", 2309.12857 — [arXiv](https://arxiv.org/html/2309.12857v2)
  - Han, Vahs, Tumova, "Risk-Aware Robot Control in Dynamic Environments Using Belief Control Barrier Functions", 2504.04097 (sample-based beliefs) — [arXiv](https://arxiv.org/abs/2504.04097v1)
  - Caution: an ID I recalled from memory for the belief-CBF paper, 2304.04583, resolves to an unrelated combinatorics paper. Do not use it.
- **Vahs, Choi, Schmid, Tumova, Fan, "Parameter-Robust MPPI for Safe Online Learning of Unknown Parameters"**, arXiv 2601.02948, 6 Jan 2026. VERIFIED.
  - Uncertainty: a particle belief over physical parameters, maintained with Stein variational gradient descent.
  - Robustness: conformal prediction evaluates the safety constraints, and a safety backup trajectory is optimized in parallel with the nominal one.
  - Experiments: simulation and hardware.
  - Parameter uncertainty only, not observation noise.
  - [arXiv](https://arxiv.org/abs/2601.02948)
- Older foundations, one line each:
  - Fridovich-Keil/Bajcsy et al. cast human-motion prediction as HJ reachability over the joint human state and model-parameter belief (1910.13369, SUMMARY) — [arXiv](https://arxiv.org/abs/1910.13369)
  - Borquez/Nakamura/Bansal's parameter-conditioned DeepReach is queried online with POINT estimates, without uncertainty sets (2209.14976, PRIOR-NOTE) — [arXiv](https://arxiv.org/abs/2209.14976)

### Inferences
- Nobody, as far as this search found, has computed a **neural HJ or DeepReach value over a belief over the robot's own physical state**, as opposed to a belief over a task or another agent's intent, with perception-induced observation noise.
- RoVer-CoRe names exactly this reformulation as open. The belief-space neural safety values (2609.15915, 2606.02562, 2309.01267) all put the uncertainty in task, intent or human parameters.
- In practice the "belief" is either an explicit filter (EKF, particles, SVGD) or implicit (recurrent or meta-RL state). Neither line validates against learned-perception error from images or point clouds.

### Gaps
- I did not find the experiment details of Deception Game (sim vs real); the abstract does not give them.
- I did not verify the Carr et al. line on shielding under partial observability (AAAI 2023). A search returned an AAAI link without a title.
- I did not find a learned-value POMDP reach-avoid paper from the Herbert, Tomlin or Ames groups in 2023-2026. Searches surfaced nothing; this is not proof of absence.

## Q2. Which latent world-model safety filters explicitly handle the gap between the estimated latent and the true state? Which are robust to noisy or delayed observations?

### Takeaway
Only Kim/Nakamura/Bajcsy 2510.06492 explicitly separates the **estimation gap** (the current latent lacks the safety-relevant state) from the **prediction gap**. Its fixes are privileged multimodal supervision and conformal risk calibration, tested on Franka hardware.

Every other calibrated latent filter bounds the **prediction or dynamics error and treats the encoder's current latent as truth**:
- Cao & Bansal 2609.34300;
- SLS² 2606.15594;
- UNISafe (epistemic out-of-distribution uncertainty).

I found **no latent world-model safety filter that is evaluated or certified under observation noise or delay**. The only noise-robustness work for world models is outside safety filtering (WISER 2512.01119).

### Cited Findings
- **Kim, Nakamura, Bajcsy, "How Well Do Latent World Models Understand Partially Observable Safety Constraints?"**, arXiv 2510.06492. VERIFIED (abstract + project page).
  - v1 7 Oct 2025 had the title "What You Don't Know Can Hurt You: How Well do Latent Safety Filters Understand Partially Observable Safety Constraints?"; v2 is dated 6 Jun 2026.
  - Uncertainty:
    - **Estimation gap**: safety-critical information is invisible in the current observations, for example temperature.
    - **Prediction gap**: failures become observable only after they happen.
  - Diagnostics: a mutual-information measure of "safety observability", plus rollout-based predictability. The MI estimate predicts the filter's myopic behaviour better than classification accuracy does.
  - Robustness:
    - "privileged multimodal supervision": train the world model to also predict thermal or tactile signals, then deploy with RGB only;
    - "conformal risk calibration" of the intervention threshold, to control false-safe rates.
  - Modalities: RGB, RGB+thermal, RGB+tactile.
  - Experiments: Franka Research 3 hardware, with two tasks:
    - wax melting (thermal constraint);
    - rice pouring from an opaque bottle (hidden contents, spill).
  - Results:
    - the RGB-only filter failed to prevent overheating;
    - with privileged thermal supervision the RGB-only controller succeeded in all 20 hardware trials;
    - the multimodal models tilted empty bottles safely and stayed conservative with filled ones.
  - Does NOT: address sensor noise, observation delay or corruption; the project page discusses none.
  - Does NOT give formal guarantees. The prior note says the conformal fix works "albeit with increased conservativeness".
  - [arXiv](https://arxiv.org/abs/2510.06492); [project](https://cmu-intentlab.github.io/multisafe/); [HTML v1](https://arxiv.org/html/2510.06492)
- **Cao & Bansal 2609.34300, "When World Models Lie"** (known paper).
  - Its follow-up-relevant detail: Adaptive Conformal Inference builds online uncertainty sets "from discrepancies between predicted and observation-inferred latent states". The observation-inferred latent serves as ground truth, so this covers the prediction gap, not the estimation gap. SUMMARY (search snippet).
  - PRIOR-NOTE: "reactive and does not provide formal safety guarantees"; assumes exchangeability.
  - [arXiv](https://arxiv.org/abs/2609.34300)
- **Nath, Srinivasan, Yin, Jiang, Fang, Glen Chou, "Pixels to Proofs: Probabilistically-Safe Latent World Model Control via Parallel Conformal Robust MPC" (SLS²)**, arXiv 2606.15594, 14 Jun 2026. VERIFIED (+HTML).
  - Uncertainty: latent dynamics prediction error, e_t = s^z_{t+1} − f(s^z_t, u_t), conformally calibrated on held-out transitions.
  - Robustness: GPU system-level-synthesis robust MPC tubes, a learned and conformalized latent constraint checker, and an "in-domain latent region" constraint.
  - Guarantee: with probability ≥1−δ the state stays in the safe set over the horizon, "under exchangeability of calibration and test prediction errors".
  - Images: RGB at 224×224.
  - Experiments:
    - MuJoCo: Reacher, OGBench Cube, Push-T, **bimanual rope**;
    - **hardware: a bimanual rope task with two 7-DoF KUKA iiwa arms**.
  - Does NOT: quantify the current-latent or encoder estimation error. The frozen encoder's latent is assumed known.
  - Does NOT use an HJ value; safety comes from MPC tubes.
  - Stated limitation: Push-T is hard because the planner relies on Euclidean latent goal distance.
  - [arXiv](https://arxiv.org/abs/2606.15594); [HTML](https://arxiv.org/html/2606.15594)
- **UNISafe 2505.00779** (known). It handles out-of-distribution observations and actions through epistemic uncertainty in an augmented latent+uncertainty state. It does not model estimation error of an in-distribution latent (PRIOR-NOTE).
  - Known follow-ups in the same line:
    - AnySafe 2509.19555;
    - LatentCBF 2511.18606;
    - Kim et al. above.
  - [arXiv](https://arxiv.org/abs/2505.00779)
- **Tabbara/Sibai line**, the known 2608.00315 plus its neighbours:
  - "Designing Latent Safety Filters using Pre-Trained Vision Models", arXiv 2509.14758. SUMMARY, title only. [arXiv](https://www.arxiv.org/pdf/2509.14758)
  - CrossSafe 2609.28984 (Tabbara, Yang, Sibai, 24 Sep 2026): cross-embodiment latent HJ filters over five bimanual robots. PRIOR-NOTE. [arXiv](https://arxiv.org/abs/2609.28984)
  - Neither is known to model observation noise.
- **Zollicoffer, Chopra, Yan, Ma, Eaton, Riedl, "World Model Robustness via Surprise Recognition" (WISER)**, arXiv 2512.01119, 30 Nov 2025. VERIFIED. A CVPR 2026 Findings listing exists (SUMMARY).
  - Uncertainty: noisy or out-of-distribution sensor failures (single or multiple faulty sensors).
  - Robustness: the world model's surprise drives rejection sampling that suppresses unreliable inputs.
  - Experiments: CARLA and Safety Gymnasium, with Cosmos and DreamerV3.
  - Does NOT: build a safety filter or an HJ or CBF value; it makes the WM-based RL agent robust.
  - [arXiv](https://arxiv.org/abs/2512.01119); [CVF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zollicoffer_World_Model_Robustness_via_Surprise_Recognition_CVPRF_2026_paper.pdf)
  - A search snippet listed perturbation types (Gaussian noise, glare, jitter, latency/lag) and attacks that test whether "early corruption persists in the recurrent latent state and whether clean observations can re-anchor the belief". I could not attribute the snippet with certainty to WISER or to **ARB4WM: An Adversarial Robustness Benchmark for World Models in Continuous Control** (arXiv 2606.16605). SUMMARY. [ARB4WM](https://arxiv.org/pdf/2606.16605)
- **GUARDIAN 2602.06026** (details in Q3) is the only filter here built for noisy or adversarially perturbed observations. Its HJ value is a low-dimensional grid, not latent. VERIFIED. [arXiv](https://arxiv.org/abs/2602.06026)

### Inferences
- The pattern holds across 2025-2026 latent filters: **calibrate the dynamics or prediction error, trust the encoder**. That covers Cao & Bansal, SLS² and, in effect, AnySafe and UNISafe.
- The estimation gap is so far fixed with *more information* (privileged or multimodal supervision), not with *robustness* (a worst case over an encoder-error set).
- A robust latent HJ value over a calibrated encoder-error ball, V_rob(z) = min over z' in B(z, q̂) of V(z'), looks unoccupied. RoVer-CoRe and GUARDIAN apply that construction only to low-dimensional grid HJ with a physical state estimate.
- Kim et al.'s rice-pouring task (opaque bottle, hidden contents) is the closest published analogue to an unobservable cup-contents spill. SLS² and CrossSafe are the closest bimanual neighbours.

### Gaps
- I found no paper evaluating a latent safety filter under **observation delay or latency**, and none under injected image noise; the results returned only general world-model robustness work.
- I found no paper certifying latent filters against encoder error on out-of-distribution images. Absence in my search is not proof of absence.
- I did not read the full SLS² paper for its quantitative hardware numbers.

## Q3. Which learned CBF or safety-critic methods certify safety under bounded or probabilistic observation error from a learned perception model (images or point clouds)?

### Takeaway
The certified methods follow one recipe: **bound the learned perception error, then robustify a mostly hand-designed or grid-computed barrier or HJ value.** Four papers follow it:

| Paper | How the perception error is bounded | Barrier or value it robustifies |
|---|---|---|
| Yang/Pappas/Mangharam/Lindemann 2304.00194 | conformal prediction | measurement-robust CBF |
| GUARDIAN 2602.06026 | neural-network verification (CROWN) | worst-case HJ over the certified state set |
| Jahanshahi & Chen 2608.13819 | conformal prediction | CBVF |
| Mei/Dixit/…/Majumdar 2403.08185 | conformal prediction on occupancy | geometric planner |

Certified **learned** barriers or values that are *also* robust to perception error are essentially absent:
- ORN-CBF (2509.16614) learns observation-conditioned neural CBFs but does not model sensor error;
- ACoFi (2604.18482) and Sibai et al. calibrate model error, not observation error.

### Cited Findings
- **Yang, Pappas, Mangharam, Lindemann, "Safe Perception-Based Control under Stochastic Sensor Uncertainty using Conformal Prediction"**, arXiv 2304.00194, CDC 2023 (v1 1 Apr 2023, v2 25 Aug 2023). VERIFIED.
  - Uncertainty: state-estimation error of a learned perception map under arbitrary stochastic sensor noise.
  - Robustness: conformal prediction gives state regions that contain the true state with high probability. These feed measurement-robust CBFs plus self-triggered sampled-data control. The authors call it the first framework with probabilistic guarantees in this setting.
  - Hardware: a LiDAR-equipped F1/10th car.
  - Does NOT: learn the CBF (the barrier is designed by hand); handle distribution shift beyond exchangeability.
  - Direct follow-up to Dean 2010.16001 and Cosner 2104.14030.
  - [arXiv](https://arxiv.org/abs/2304.00194)
- **Zhang, Hoxha, Fainekos, Panagou, "Conformal Prediction in the Loop: Risk-Aware Control Barrier Functions for Stochastic Systems With Data-Driven State Estimators"**, IEEE L-CSS 2025. SUMMARY.
  - The arXiv ID was not found; do not cite one.
  - [ResearchGate](https://www.researchgate.net/publication/391927005_Conformal_Prediction_in_the_Loop_Risk-Aware_Control_Barrier_Functions_for_Stochastic_Systems_with_Data-Driven_State_Estimators)
- **Rober, Rose, How (MIT), "GUARDIAN: Safety Filtering for Systems with Perception Models Under Adversarial Attack"**, arXiv 2602.06026, v1 5 Feb 2026, v2 14 Sep 2026. VERIFIED (+HTML). Peer-reviewed venue not confirmed.
  - Uncertainty: an ℓ∞-bounded perturbation ‖Δy‖∞ ≤ ε on the sensor input of an NN state estimator.
  - Robustness: CROWN bounds the estimate inside [x̲, x̄], which is inflated by the estimator error. The HJ filter evaluates Φ(X̄, u) = min over s in X̄ and d in D of V(f(s, u, d)) on a precomputed grid value.
  - Experiments, all simulation:
    - TaxiNet (2D) with 8×16 downsampled X-Plane images, robust to PGD at ε = 0.021, at 0.032 s per step;
    - a 4D range-measurement localization task, with a "state-dependent vulnerability analysis";
    - a scalar comparison where MR-CBF, R-CBF and R-CBF-QP baselines failed under large estimation errors.
  - Limitations (verbatim): "the scalability limitations of HJ reachability are well known ... exponential complexity in the state dimension"; Assumption 2 "must be checked at runtime".
  - Does NOT: use a learned or neural value; use a latent space; run on hardware.
  - [arXiv](https://arxiv.org/abs/2602.06026); [HTML](https://arxiv.org/html/2602.06026)
- **Mei, Dixit, Booker, Zhou, Storey-Matsutani, Ren, Shorinwa, Majumdar, "Perceive With Confidence: Statistical Safety Assurances for Navigation with Learning-Based Perception"**, arXiv 2403.08185 (v1 13 Mar 2024, v3 17 Apr 2025). VERIFIED. CoRL 2024 is from memory, not verified.
  - Uncertainty: errors of pre-trained object-detection and scene-completion models in unseen environments.
  - Robustness: conformal calibration of the perceived occupancy, plus a safe planner, gives "end-to-end statistical assurance on safety in unseen environments".
  - Simulation (quadruped, indoor): 70% fewer obstacle misdetections and 100% safety.
  - Hardware: a 40% empirical safety improvement over baselines and 93.3% fewer misdetections; the safety gap widens to 46.7% at higher speeds.
  - Does NOT: use a learned HJ value or CBF; couple to dynamics-aware reachability.
  - [arXiv](https://arxiv.org/abs/2403.08185)
- **Derajić, Bernhard, Hönig, "ORN-CBF: Learning Observation-conditioned Residual Neural Control Barrier Functions via Hypernetworks"**, arXiv 2509.16614 (v1 20 Sep 2025, v3 8 Mar 2026). VERIFIED (abstract).
  - What it learns: an observation-conditioned neural CBF derived from HJ reachability, which "approximately recover[s] the maximal safe sets". Predicted safe sets never intersect observed failure sets, and a hypernetwork does the observation conditioning.
  - Experiments: a ground robot and a quadcopter, in simulation and on hardware. It improves success rate and out-of-domain generalization.
  - Does NOT: model sensor noise or estimation error. The sensor modality is not given in the abstract.
  - [arXiv](https://arxiv.org/abs/2509.16614)
- **Tong, Dawson, Fan, "Enforcing safety for vision-based controllers via Control Barrier Functions and Neural Radiance Fields"**, arXiv 2209.12266, ICRA 2023. VERIFIED.
  - Method: a NeRF renders the next image ("single-step visual foresight") so a CBF can be evaluated on visual feedback.
  - Experiments: real-time simulation.
  - Does NOT: quantify the error of the NeRF predictions.
  - [arXiv](https://arxiv.org/abs/2209.12266)
- **Tabbara & Sibai, "Learning Ensembles of Vision-based Safety Control Filters"**, arXiv 2412.02029, 2 Dec 2024. VERIFIED.
  - Uncertainty: out-of-distribution visual observations.
  - Robustness: ensembles of diverse pre-trained vision backbones, evaluated by state and control classification accuracy on DeepAccident.
  - Does NOT: run closed-loop experiments; give guarantees.
  - [arXiv](https://arxiv.org/abs/2412.02029)
  - Related, title only (SUMMARY): "Pre-Trained Vision Models as Perception Backbones for Safety Filters in Autonomous Driving", 2410.22585. [arXiv](https://arxiv.org/pdf/2410.22585)
- **Huriot, Tabbara, Sibai, "Safe Control using Learned Safety Filters and Adaptive Conformal Inference" (ACoFi)**, arXiv 2604.18482, 20 Apr 2026, L4DC 2026. VERIFIED.
  - Method: applies ACI to the range of learned HJ safety values for the nominal action, giving a soft guarantee that the error rate is "asymptotically upper bounded".
  - Experiments: Dubins car and Safety Gymnasium.
  - The uncertainty is learned-model error, **not observation error**.
  - [arXiv](https://arxiv.org/abs/2604.18482)
- PRIOR-NOTE, adjacent:
  - **Mirzaeedodangeh, Shekhtman, Matni, Lindemann, "Robust Conformal CBF and CLF Controllers via Iterative Policy Updates"**, 2606.15366: conformal bounds in CBFs are invalidated by the distribution shift the filter itself induces, and iterative policy updates fix it. [arXiv](https://arxiv.org/abs/2606.15366)
  - **Sinha, Schmerling, Pavone, "Closing the Loop on Runtime Monitors with Fallback-Safe MPC"**, 2309.08603: a conformally calibrated perception monitor with a perception-independent fallback. [arXiv](https://arxiv.org/abs/2309.08603)
- **Miao, Darir, Sayan Mitra, "Correcting Learning-based Perception for Safety"**, arXiv 2609.22108. VERIFIED (abstract).
  - Date mismatch: the digest says v1 was submitted 16 Aug 2026, which conflicts with a 2609 ID (Sep 2026). Check it.
  - Method: offline "preimages of perception contracts" characterize the error sets of the ML estimator, and runtime risk heuristics select a corrected state.
  - Perception models: YOLO and LaneNet.
  - Experiments: adaptive cruise control in simulation across weather and road conditions. It corrected 73% of 45 safety-violating scenarios at a 2.8% average completion-time cost.
  - Direct follow-up to perception contracts 2309.13515.
  - [arXiv](https://arxiv.org/abs/2609.22108)
- **RL safety critics under observation attack: Liu, Guo, Cen, Zhang, Tan, Li, Zhao, "On the Robustness of Safe Reinforcement Learning under Observational Perturbations"**, arXiv 2205.14691, ICLR 2023. SUMMARY.
  - Optimal safe-RL solutions are "not robust and safe against carefully designed observational perturbations".
  - They propose maximum-cost and maximum-reward attackers. The maximum-reward attack is strong and stealthy.
  - They also propose robust adversarial training.
  - [arXiv](https://arxiv.org/pdf/2205.14691); [ICLR](https://iclr.cc/virtual/2023/poster/11925)
- Search hits on conformal or uncertainty-aware CBFs, title only (SUMMARY). Their perception-relevance is unconfirmed.
  - "CPED-NCBFs: A Conformal Prediction for Expert Demonstration-based Neural CBFs", 2507.15022 — [arXiv](https://arxiv.org/html/2507.15022)
  - "Learning to Adapt Control Barrier Functions Under Epistemic and Aleatoric Uncertainty", 2504.03038 — [arXiv](https://arxiv.org/html/2504.03038)
  - "Safety-Critical Control under Uncertainty via Adaptive Conformal Quantile Prediction Intervals", 2609.23280 — [arXiv](https://arxiv.org/pdf/2609.23280)
  - "Differentiable Optimization Layered Safety-Critical Control for Risk-Aware Navigation via Conformal Prediction", 2605.16327 — [arXiv](https://arxiv.org/pdf/2605.16327)
  - "Statistical Guarantees in Data-Driven Nonlinear Control: Conformal Robustness for Stability and Safety", 2506.06228 — [arXiv](https://arxiv.org/pdf/2506.06228)
  - "Perception-Integrated Safety Critical Control via Analytic Collision Cone Barrier Functions on 3D Gaussian Splatting", 2509.14421 — [arXiv](https://arxiv.org/html/2509.14421)
- False positive: Zhou, Zhang, Luo, "Safety-Critical Control with Uncertainty Quantification using Adaptive Conformal Prediction", 2407.03569 (ACC 2024). VERIFIED.
  - It handles motion and dynamics noise with no learned perception, so it does not answer this question.
  - [arXiv](https://arxiv.org/abs/2407.03569)

### Inferences
- With learned perception, the certified pieces never coincide:
  - the *perception error* is certified, by conformal prediction or neural-network verification;
  - the *barrier or value* stays analytic or grid-based.
- The two combinations that would answer this question are missing:
  - a learned value plus certified observation error;
  - a latent value plus certified encoder error.
- Point clouds and LiDAR appear only in:
  - Yang et al. (LiDAR F1/10th);
  - Perceive-with-Confidence (occupancy from detection and scene completion).
- No certified method found operates on 3D point clouds with a learned value.
- Conformal guarantees in this whole line depend on exchangeability, which the filter's own intervention breaks (2606.15366, PRIOR-NOTE). That caveat transfers directly to any calibrated latent filter.

### Gaps
- I found no paper that certifies a **learned** CBF or HJ value against **image** perturbations end to end. GUARDIAN certifies images only through a grid value on a 2D TaxiNet state.
- I did not confirm the sensor modality of ORN-CBF.
- I did not find the arXiv ID of the Panagou-group L-CSS paper.

## Q4. Has anyone shown, and measured, that perception errors at critical moments (near the failure boundary) break a learned safety value?

### Takeaway
Partly. The Bansal group's vision-controller line shows and measures, with reachability, that perception errors cause *closed-loop system-level* failures that depend on the state:
- Chakraborty & Bansal 2211.02736;
- Chakraborty/Gupta/Bansal 2410.21736;
- Gupta et al. 2309.13475.

Corso et al. 2205.10677 quantify a risk function per perception error. GUARDIAN reports a state-dependent vulnerability analysis. Liu et al. show adversarial observations break safe-RL policies.

The **target is always a learned controller or a hand-designed filter, not a learned safety value**. I found no measured study of how learned or latent HJ, CBF or critic values degrade with perception error *as a function of distance to the failure boundary*.

### Cited Findings
- **Chakraborty & Bansal, "Discovering Closed-Loop Failures of Vision-Based Controllers via Reachability Analysis"**, arXiv 2211.02736, RA-L 8(5) 2023, 2692-2699 (v1 Nov 2022, v4 Apr 2024). VERIFIED.
  - Method: casts "the problem of finding closed-loop vision failures as a Hamilton-Jacobi reachability problem". A BRT over simulator-generated images yields the failing states and the visual inputs that trigger them.
  - Experiments: RGB NN controllers for indoor navigation and for aircraft taxiing.
  - The abstract page does **not** state the "more catastrophic near a wall" claim.
  - [arXiv](https://arxiv.org/abs/2211.02736)
- **Chakraborty, Gupta, Bansal, "Enhancing Safety and Robustness of Vision-Based Controllers via Reachability Analysis"**, arXiv 2410.21736 (29-30 Oct 2024). VERIFIED.
  - Method: Neural Reachable Tubes stress-test vision controllers. The failures found feed a runtime failure classifier with a fallback, and are added to retraining data.
  - Experiments: vision-guided aircraft taxiing in simulation.
  - It reports beating "methods relying on prediction error or uncertainty quantification". In other words, perception-error magnitude alone is a poor predictor of system-level failure.
  - [arXiv](https://arxiv.org/abs/2410.21736)
- A search snippet tied to this line states: "The same prediction error by a visual policy can be much more catastrophic near a wall compared to an empty hallway", and "Not all perception errors translate to meaningful system-level failures". SUMMARY; the exact source page (2410.21736 or 2309.13475) was not confirmed.
  - [2410.21736 HTML](https://arxiv.org/html/2410.21736)
  - [Gupta, Chakraborty, Bansal, "Detecting and Mitigating System-Level Anomalies of Vision-Based Controllers", 2309.13475](https://arxiv.org/html/2309.13475)
- **Corso, Katz, Innes, Du, Ramamoorthy, Kochenderfer, "Risk-Driven Design of Perception Systems"**, arXiv 2205.10677 (May/Oct 2022; outside the date window, foundational). VERIFIED.
  - Method: "formulate a risk function to quantify the effect of a given perceptual error on overall safety", then use it in risk-weighted training losses and targeted data generation.
  - Result: 37% lower collision risk in aircraft collision avoidance.
  - The abstract does not say how the risk is computed.
  - [arXiv](https://arxiv.org/abs/2205.10677)
- **GUARDIAN 2602.06026**: the 4D range-measurement case includes a "state-dependent vulnerability analysis", and the baselines (MR-CBF, R-CBF, R-CBF-QP) "failed under large estimation errors" where GUARDIAN stayed safe. VERIFIED (HTML digest; I did not see the figures). [HTML](https://arxiv.org/html/2602.06026)
- **RoVer-CoRe 2511.14755** sweeps perception-error bound coverage from 0 to 1 in the aircraft case. The resulting verified safe set shows how much of the state space survives each error level. VERIFIED (HTML digest). [HTML](https://arxiv.org/html/2511.14755)
- **Liu et al. 2205.14691 (ICLR 2023)**: carefully designed observation perturbations make optimal safe-RL policies unsafe, and the reward-maximizing attack does it stealthily. SUMMARY. [arXiv](https://arxiv.org/pdf/2205.14691)
- **Kim et al. 2510.06492**: when safety features are unobservable, latent safety filters become *myopic*, and accuracy-style metrics miss this, while MI-based observability predicts it. This is the closest learned-latent-value evidence, but it concerns information absence, not noise. SUMMARY (snippet from the arXiv HTML) and VERIFIED (project page). [HTML](https://arxiv.org/html/2510.06492)
- **Miao/Darir/Mitra 2609.22108**: 45 perception-induced safety-violating ACC scenarios, 73% corrected. It targets errors at the perception model's operational boundary, not at the safe-set boundary. VERIFIED (abstract). [arXiv](https://arxiv.org/abs/2609.22108)

### Inferences
- The idea that perception error matters most near the failure boundary is the founding motivation of several lines:
  - reachability-based vision falsification;
  - measurement-robust CBFs, where the required margin scales with the barrier gradient;
  - risk-driven perception design.
- Only controller-level failure rates have been measured, not the corruption of a *learned safety value* itself near its zero level set.
- An experiment on our side would sit in an open slot. It would plot the false-safe rate of V(encoder(obs + noise)), binned by the true distance to the boundary, for a learned or latent value. This inference rests on what my search did not find.

### Gaps
- I did not confirm which paper contains the "near a wall vs empty hallway" sentence.
- I did not read the Corso et al. full text, so I cannot say whether its risk function is an HJ or DP value.
- I found no quantitative study of adversarial or noise sensitivity of DeepReach or latent HJ values near the zero level set. The search for it returned only world-model robustness papers, WISER and ARB4WM, with no safety value.
