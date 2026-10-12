# Perception Error Bounds to Robust Safety Constraints (reachability / CBF / MPC / runtime monitors), 2023-2026

Legend. **[VERIFIED]** means I read the claim on the paper's arXiv abstract page (or the PMLR page for the L4DC 2025 paper) during this session (2026-10-05). I read abstracts, not full PDFs, so implementation details beyond the abstract are not claimed. **[SUMMARY]** means the claim comes only from a search-engine snippet or a third-party page. "Does NOT" fields are my inferences from what the abstract states and leaves out. Treat them as "not claimed in the abstract", not as proven absences. Dates are arXiv v1 submission dates. Venues are given only when the arXiv comment field or a proceedings page states them.

ID corrections found while verifying (useful to the report writer):
- arXiv 2502.14149 is "PitVQA++" (surgical VQA). It is NOT the latent-safety-filter paper, so do not cite it as one. — [arXiv 2502.14149](https://arxiv.org/abs/2502.14149) [VERIFIED]
- arXiv 2410.12019 is by Deglurkar, Shen, Muthali, Pavone, Margineantu, Karkus, Ivanovic and Tomlin, not the Bansal group. — [arXiv 2410.12019](https://arxiv.org/abs/2410.12019) [VERIFIED]
- arXiv 2410.15185 ("Semantically Safe Robot Manipulation") is by Brunke, ..., Schoellig. — [arXiv 2410.15185](https://arxiv.org/abs/2410.15185) [VERIFIED]
- Search snippets gave wrong dates: the Deterministic World Model paper is Dec 2025, not Dec 2024 ([2512.08991](https://arxiv.org/abs/2512.08991)), and the Stochastic World Models paper is Sep 2026, not Aug 2025 ([2609.38120](https://arxiv.org/abs/2609.38120)). [VERIFIED]
- Glen Chou's RSS 2026 paper is listed on the RSS site as "Seeing is Believing: Certified Perception-Based Control from Learned Visual Representations via System Level Synthesis". On arXiv it is titled "VISION-SLS: ...". It is the same paper (arXiv 2604.24894). — [RSS program](https://roboticsconference.org/program/papers/172/) [SUMMARY]; [arXiv 2604.24894](https://arxiv.org/abs/2604.24894) [VERIFIED]

---

## Q1. What has followed measurement-robust CBFs (Dean 2020, Cosner 2021) and perception contracts (2023)? Extensions to vision/point clouds, learned dynamics, manipulation?

### Takeaway
The MR-CBF line has moved in three directions:
- **Calibrated or adaptive error margins:** CP in Yang et al. 2023 and Zhang et al. 2025; OOD-adaptive margins in ATOM-CBF; online-adapted R-CBF parameters in Das/Ames 2025.
- **Distributionally robust sensor-level CBFs on raw LiDAR/point clouds:** Long/Atanasov 2024, including a 6-DoF xArm manipulator in 2025.
- **Learned robust margins:** the NMR-CBF of Rober/How 2026.

The perception-contract line (Mitra, UIUC) has produced:
- data/requirement-guided refinement (DaRePC);
- Lyapunov perception contracts (L4DC 2025);
- abstract rendering of Gaussian-splat scenes;
- runtime correction via contract preimages (2026);
- an ellipsoid-mixture inverse-contract variant from another group (ACC 2026).

Manipulation is still rare. I found five pieces of manipulation work:
- Chou 2022 (17D manipulation task);
- the Atanasov xArm6 DR-CDF barrier (2025);
- UNISafe on a Franka (2025);
- CORD-SLS for rope and cloth with CP-calibrated perception tubes (2026);
- CSymPlan on a Franka FR3 (2026), which treats measurement uncertainty as a bounded disturbance after feedback linearization.

None of these is bimanual or cooperative arm-arm.

### Cited Findings

#### Foundations (one line each)
- Dean, Taylor, Cosner, Recht, Ames, "Guaranteeing Safety of Learned Perception Modules via Measurement-Robust Control Barrier Functions", arXiv 2010.16001 (2020-10-30). Defines the MR-CBF and uses it to guide perception sampling and quantify tolerable learned-perception error. Simulated Segway. — [arXiv](https://arxiv.org/abs/2010.16001) [VERIFIED]
- Cosner, Singletary, Taylor, Molnar, Bouman, Ames, "Measurement-Robust Control Barrier Functions: Certainty in Safety with Uncertainty in State", arXiv 2104.14030 (2021-04-28). Combines MR-CBF with backup sets. Hardware Segway with onboard stereo-vision state estimation. — [arXiv](https://arxiv.org/abs/2104.14030) [VERIFIED]
- Lindemann, Robey, Jiang, Das, Tu, Matni, "Learning Robust Output Control Barrier Functions from Safe Expert Demonstrations", arXiv 2111.09971 (2021-11-18; journal). ROCBFs are learned from demonstrations, given a state estimator with error bounds that are "e.g., estimated from data". — [arXiv](https://arxiv.org/abs/2111.09971) [VERIFIED]
- Cosner, Jimenez Rodriguez, Molnar, Ubellacker, Yue, Ames, Bouman, "Self-Supervised Online Learning for Safety-Critical Control using Stereo Vision", arXiv 2203.01404 (ICRA 2022). Learns the stereo-vision uncertainty self-supervised, adapts it online, and robustifies CBFs with it. Quadruped hardware. — [arXiv](https://arxiv.org/abs/2203.01404) [VERIFIED]
- Chou, Ozay, Berenson, "Safe Output Feedback Motion Planning from Images via Learned Perception Modules and Contraction Theory", arXiv 2206.06553 (WAFR 2022). Learns perception from RGB(-D) plus a high-probability upper bound on perception error, valid in a "trusted domain". A contraction-based controller/observer gives a tracking-error tube that enters a sampling-based planner. Tested on a 4D car, a 6D planar quadrotor and a **17D manipulation task**. — [arXiv](https://arxiv.org/abs/2206.06553) [VERIFIED]

#### MR-CBF follow-ups, 2023-2026
- **"Safe Perception-Based Control under Stochastic Sensor Uncertainty using Conformal Prediction"**, Shuo Yang, George J. Pappas, Rahul Mangharam, Lars Lindemann. arXiv 2304.00194 (2023-04-01), CDC 2023. — [arXiv](https://arxiv.org/abs/2304.00194) [VERIFIED]
  - *Modality:* LiDAR-based learned perception map.
  - *Error set:* CP "state estimation regions".
  - *Into safety:* a sampled-data MR-CBF controller that uses self-triggered control to avoid stochastic calculus.
  - *Guarantee:* probabilistic safety, which the authors call the first in this setting.
  - *Experiments:* F1/10 car.
  - *Does NOT:* learn dynamics (known model) or work in a latent space; the state is low-dimensional.
- **"Conformal Prediction in the Loop: Risk-Aware Control Barrier Functions for Stochastic Systems With Data-Driven State Estimators"**, Junhui Zhang, Bardh Hoxha, Georgios Fainekos, Dimitra Panagou. IEEE Control Systems Letters, 2025. Embeds CP bounds from learned state estimators into risk-aware CBFs. I found no arXiv ID. — [dblp/search snippet](https://dblp.uni-trier.de/pid/34/4314.html) [SUMMARY]
- **"ATOM-CBF: Adaptive Safe Perception-Based Control under Out-of-Distribution Measurements"**, Kai S. Yun, Navid Azizan. arXiv 2511.08741 (2025-11-11). Already known; included for linkage. — [arXiv](https://arxiv.org/abs/2511.08741) [VERIFIED]
  - *Error set:* an OoD-aware adaptive perception error margin from epistemic uncertainty, with no ground-truth labels.
  - *Into safety:* the margin enters the CBF filter, so conservatism adapts in real time.
  - *Experiments:* simulations only: F1Tenth with LiDAR scans and a quadruped with RGB images.
  - *Venue:* L4DC 2026 (PMLR v331). — [PMLR](https://proceedings.mlr.press/v331/yun26a.html) [SUMMARY]
- **"Learning-Based Measurement-Robust Control Barrier Functions for Obstacle Avoidance under State Estimation Error"**, Nicholas Rober, Yixuan Jia, Jonathan P. How. arXiv 2608.20467 (2026-08-20). — [arXiv](https://arxiv.org/abs/2608.20467) [VERIFIED]
  - *Error set:* state-estimation (odometry) error.
  - *Into safety:* two new forms.
    - **DMR-CBF** adds an inner worst-case optimization over drift-dynamics uncertainty to the CBF condition.
    - **NMR-CBF** replaces that inner optimization with a learned term. It is pretrained under DMR-CBF supervision and fine-tuned through differentiable trajectory rollouts.
  - *Guarantee:* theoretical analysis for DMR-CBF only. The abstract says the NMR-CBF achieves "empirical safety comparable".
  - *Experiments:* planar double integrator, 12D quadrotor, and a Unitree Go2 on hardware.
  - *Does NOT:* use images or point clouds directly; the perception error is odometry error.
- **"Safe Navigation under State Uncertainty: Online Adaptation for Robust Control Barrier Functions"**, Ersin Das, Rahal Nanayakkara, Xiao Tan, Ryan M. Bena, Joel W. Burdick, Paulo Tabuada, Aaron D. Ames. arXiv 2508.19159 (2025-08-26). — [arXiv](https://arxiv.org/abs/2508.19159) [VERIFIED]
  - *Into safety:* online optimization-based adaptation of R-CBF parameters reduces conservatism and infeasibility under estimation error. Several safety constraints are merged into one numerical CBF via Poisson's equation.
  - *Experiments:* tracked-vehicle hardware.
  - *Does NOT:* learn the perception bound (the abstract does not describe how the error set is obtained).
- **"Sensor-Based Distributionally Robust Control for Safe Robot Navigation in Dynamic Environments"**, Kehan Long, Yinzhuang Yi, Zhirui Dai, Sylvia Herbert, Jorge Cortés, Nikolay Atanasov. arXiv 2405.18251 (2024-05-28). — [arXiv](https://arxiv.org/abs/2405.18251) [VERIFIED]
  - *Into safety:* a DR-CBF that "directly integrates noisy sensor measurements and state estimates" into the constraint.
  - *Experiments:* differential-drive robots in simulation and on hardware.
- **"Neural Configuration-Space Barriers for Manipulation Planning and Control"**, Kehan Long, Ki Myung Brian Lee, Nikola Raicevic, Niyas Attasseri, Melvin Leok, Nikolay Atanasov. arXiv 2503.04929 (2025-03-06). **Manipulation.** — [arXiv](https://arxiv.org/abs/2503.04929) [VERIFIED]
  - *Barrier:* a learned configuration-space distance function (CDF) barrier.
  - *Into safety:* a distributionally robust CDF-barrier formulation that accounts for neural-model error and point-cloud sensor noise "without assuming a known underlying distribution".
  - *Experiments:* UFactory xArm6 in simulation and hardware, "relying only on onboard point-cloud observations".
  - *Does NOT:* handle bimanual arms or learned dynamics.
- **"Formation-Aware Adaptive Conformalized Perception for Safe Leader-Follower Multi-Robot Systems"**, Richie R. Suganda, Bin Hu. arXiv 2603.08958 (2026-03-09). — [arXiv](https://arxiv.org/abs/2603.08958) [VERIFIED]
  - *Error set:* Risk-Aware Mondrian CP gives formation-conditioned (heteroscedastic) quantiles of onboard-vision relative-state error.
  - *Into safety:* a "Formation-Aware Conformal CBF-QP" with a smooth margin that keeps the leader in the camera field of view.
  - *Guarantee:* finite-sample probabilistic.
  - *Experiments:* Gazebo.
- **"Robust Control Barrier Functions for Safe Control Under Uncertainty Using Extended State Observer and Output Measurement"**, Jinfeng Chen, Zhiqiang Gao, Qin Lin. arXiv 2308.13943 (2023-08-26). An extended state observer gives unified bounds on state and disturbance estimation error, which robustify the CBF. Simulated adaptive cruise control and Segway. No learned perception. — [arXiv](https://arxiv.org/abs/2308.13943) [VERIFIED]
- **"Sensor-Based Safety-Critical Control Using an Incremental Control Barrier Function Formulation via Reduced-Order Approximate Models"**, Johannes Autenrieb, Hyo-Sang Shin. arXiv 2410.08096 (ACC 2025). Introduces "measurement-robust incremental CBFs" that substitute model terms with sensor measurements. Simulations: a first-order system with sensor bias and a hypersonic glide vehicle. No learned perception. — [arXiv](https://arxiv.org/abs/2410.08096) [VERIFIED]
- **"Safe Output-Feedback Adaptive Optimal Control of Affine Nonlinear Systems"**, Ogri, Qureshi, Bell, Makumi, Kamalapurkar. arXiv 2510.20081 (2025-10-22). CBFs robustified for state-estimation error from DNN-based adaptive observers, with a Lyapunov-based guarantee. Simulation only. — [arXiv](https://arxiv.org/abs/2510.20081) [VERIFIED]
- **"Credibility-Aware Learning and Control for Safe USV Navigation under Perception Uncertainty"**, Yuhang Zhang et al. arXiv 2605.26974 (2026-05-26). — [arXiv](https://arxiv.org/abs/2605.26974) [VERIFIED]
  - *Error set:* a trust factor computed from the gap between filter covariance and empirical error.
  - *Into safety:* CoReCBF-QP expands the collision geometry according to the uncertainty.

#### Perception-contract follow-ups (Mitra group, UIUC, and others)
- **"Assuring Safety of Vision-Based Swarm Formation Control"**, Chiao Hsieh, Yubin Koh, Yangge Li, Sayan Mitra. arXiv 2210.00982 (2022-10; submitted to ACC 2024). Perception contracts are built from sampled data across states, environments and weather. They give error bounds that grow with inter-agent distance and are matched to a log-polar quantizer for a quantized-consensus convergence proof. — [arXiv](https://arxiv.org/abs/2210.00982) [VERIFIED]
- **"Learning-based Inverse Perception Contracts and Applications"**, Dawei Sun, Benjamin C. Yang, Sayan Mitra. arXiv 2309.13515 (2023-09-24). This is the known "perception contracts 2023" item; ICRA 2024 per search snippet. The IPC learns a set that contains the ground truth with high probability, and a control algorithm uses it for vision-based quadcopter landing. — [arXiv](https://arxiv.org/abs/2309.13515) [VERIFIED]; venue [search snippet](https://mitras.ece.illinois.edu/pubs.html) [SUMMARY]
- **"Refining Perception Contracts: Case Studies in Vision-based Safe Auto-landing"**, Yangge Li, Benjamin C. Yang, Yixuan Jia, Daniel Zhuang, Sayan Mitra. arXiv 2311.08652 (2023-11-15). — [arXiv](https://arxiv.org/abs/2311.08652) [VERIFIED]
  - *Method:* DaRePC, a data- and requirement-guided refinement of contracts.
  - *Systems:* 6D and 12D flight-control systems with multi-stage ML perception (aircraft landing; drone gate passing).
  - *Finding:* it discovers unsafe conditions such as "low-horizon sun".
- **"Lyapunov Perception Contracts for Operating Design Domains"**, Yangge Li, Chenxi Ji, Jai Anchalia, Yixuan Jia, Benjamin C. Yang, Daniel Zhuang, Sayan Mitra. L4DC 2025, PMLR 283:1053-1065. — [PMLR](https://proceedings.mlr.press/v283/li25c.html) [VERIFIED]
  - *Contract:* LPCs are specifications for DNN state estimators, synthesized from data plus controller/plant models. They assure **closed-loop stability**, not set-invariance safety.
  - *Use:* find operating design domains in finitely parameterized environments.
  - *Experiments:* visual auto-landing with simulation and Google Earth imagery.
- **"Abstract Rendering: Computing All that is Seen in Gaussian Splat Scenes"**, Yangge Li, Chenxi Ji, Xiangru Zhong, Huan Zhang, Sayan Mitra. arXiv 2503.00308 (2025-03-01). Computes the set of all images rendered from a continuous range of camera poses as constraints on the image matrix, using linear bound propagation through Gaussian-splat rendering (including relational abstractions for matrix inversion and depth sorting). Motivated by formal verification of vision-based systems. — [arXiv](https://arxiv.org/abs/2503.00308) [VERIFIED]
- **"Correcting Learning-based Perception for Safety"**, Yan Miao, Hussein Darir, Sayan Mitra. arXiv 2609.22108 (listed 2026-08-16). — [arXiv](https://arxiv.org/abs/2609.22108) [VERIFIED]
  - *Method:* offline preimages of perception contracts characterize ML state-estimation uncertainty; at runtime a risk heuristic picks the state that drives control.
  - *Results:* in 45 unsafe vision-based adaptive-cruise scenarios using YOLO and LaneNet, safety was preserved in 73%. The 27% failures occur where the preimages are "not fully conformant". Completion time rose 2.8% on average.
  - *Guarantee:* empirical only.
- **"Gaussian Mixture-Based Inverse Perception Contract for Uncertainty-Aware Robot Navigation"**, Bingyao Du, Joonkyung Kim, Yiwei Lyu. arXiv 2603.04329 (ACC 2026). — [arXiv](https://arxiv.org/abs/2603.04329) [VERIFIED]
  - *Error set:* GM-IPC extends the IPC from a single ellipsoid to **unions of ellipsoids** from Gaussian mixture models, so multi-modal and non-convex error sets are captured.
  - *Training losses:* inclusion, distribution matching, and an empty-space penalty.
  - *Use:* real-time safe navigation planners.
  - *Guarantee:* probabilistic.
- **"Formal Verification Techniques for Vision-Based Autonomous Systems – A Survey"**, Mitra, Păsăreanu, Prabhakar, Seshia, Mangal, Li, Watson, Gopinath, Yu (2024). A survey of perception contracts and related methods. — [NSF PAR](https://par.nsf.gov/biblio/10642945-formal-verification-techniques-nbsp-vision-based-autonomous-systems-survey) [SUMMARY]

#### Robust MPC / SLS with learned perception error bounds (Chou line)
- **"VISION-SLS: Safe Perception-Based Control from Learned Visual Representations via System Level Synthesis"**, Antoine P. Leeman, Shuyu Zhan, Melanie N. Zeilinger, Glen Chou. arXiv 2604.24894 (2026-04-27), RSS 2026. — [arXiv](https://arxiv.org/abs/2604.24894) [VERIFIED]
  - *Modality:* RGB images of at least 512x512, via pretrained visual features.
  - *Error set:* a learned low-dimensional observation map with **state-dependent error bounds**, "empirically-calibrated".
  - *Into safety:* a causal affine time-varying output-feedback SLS policy gives robust constraint satisfaction. It is solved by sequential convex programming with Riccati recursions.
  - *Experiments:* 4D car, 10D quadrotor, a 59D humanoid under partial observability, and a hardware ground vehicle.
  - *Does NOT:* use a latent world model (the dynamics are a known nonlinear model per the abstract) or do manipulation.
- **"Robustness without Wrinkles: Parallel Simulation and Robust MPC for Certified Deformable Manipulation"** (CORD-SLS), Wei-Chen Li, Jeffrey Fang, Sasanka Polisetti, Yuexi Song, Glen Chou. arXiv 2606.14188 (2026-06-12). **Manipulation.** — [arXiv](https://arxiv.org/abs/2606.14188) [VERIFIED]
  - *Error set:* CP is used to "calibrate visual-feedback and perception-error bounds for MPC".
  - *Into safety:* the bounds produce reachable tubes inside a GPU-parallel output-feedback robust MPC that plans through a differentiable contact-smoothed simulator.
  - *Guarantee:* high-probability safe control.
  - *Experiments:* rope and cloth (obstacle avoidance, routing, folding, smoothing) in simulation and on hardware, with millisecond-speed planning.
  - *Does NOT:* use a latent world model (the model is a physics simulator).

#### Manipulation-specific (perception uncertainty to safety)
- **"CSymPlan: Certified Symbolic Planning and Control for High-DOF Manipulators"**, Aditya Narendra et al. arXiv 2608.22983 (2026-08-24). — [arXiv](https://arxiv.org/abs/2608.22983) [VERIFIED]
  - *Approach:* feedback linearization reduces the manipulator to a sampled, perturbed operational-space **double integrator**. Torque-realization error, model mismatch and **measurement uncertainty** are treated as bounded disturbances in an abstraction-based reach-avoid synthesis.
  - *Variants:* offline policy tables, and online synthesis with pFaces from "changing task and perception information".
  - *Experiments:* "perception-driven Franka FR3 experiments" with zero safety violations; the robot holds or stops when no certified action exists.
  - *Does NOT:* learn the perception bound (it is assumed bounded).
- **"Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees"**, Jakob Thumm, Marian Frei, Tianle Ni, Matthias Althoff, Marco Pavone. arXiv 2604.15221 (2026-04-16). — [arXiv](https://arxiv.org/abs/2604.15221) [VERIFIED]
  - *Error set:* CP bounds on vision-based 3D human pose and motion prediction. A gradient-based OOD monitor replaces OOD poses with past predictions.
  - *Into safety:* the CP sets plug directly into the reachability-based **SARA shield**.
  - *Results:* CP sets are 7.6x smaller in volume than model-based ones; probability of dangerous failure per hour is bounded at 9.5e-7 with 99.999% confidence "under our test distribution". The authors call this "necessary but not sufficient" for ISO 13849 PL d.
- **"Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures"** (UNISafe), Junwon Seo, Kensuke Nakamura, Andrea Bajcsy. arXiv 2505.00779 (CoRL 2025). Covered in Q2. Vision-based Franka experiments in simulation and on hardware. — [arXiv](https://arxiv.org/abs/2505.00779) [VERIFIED]
- **"Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards"**, Lukas Brunke et al. arXiv 2410.15185 (2024-10-19). — [arXiv](https://arxiv.org/abs/2410.15185) [VERIFIED]
  - *Pipeline:* a 3D semantic map plus LLM reasoning infers unsafe conditions, including "moving a cup of water above a laptop" and "rotating a cup of water". A control-barrier certification then filters inputs.
  - *Experiments:* teleoperation and diffusion policies in a real kitchen.
  - *Does NOT:* bound perception error.
- **"Safety filtering of robotic manipulation under environment uncertainty: a computational approach"**, Johansson, Lindmark, Wiberg, Servin. arXiv 2509.12674 (2025-09-16). A physics-simulation safety filter with sparse re-evaluation for a **simulated bimanual** task with uncertain mass and friction. Uses generalized factors of safety for grasp stability and actuator limits. Uncertainty is in world parameters, not perception. — [arXiv](https://arxiv.org/abs/2509.12674) [VERIFIED]
- **"SPIRIT: Perceptive Shared Autonomy for Robust Robotic Manipulation under Deep Learning Uncertainty"**, Jongseok Lee et al. arXiv 2603.05111 (2026-03-05). Neural-tangent-kernel uncertainty from point-cloud registration switches between autonomy and haptic teleoperation in aerial manipulation. A switching rule, not a CBF or reachability set. — [arXiv](https://arxiv.org/abs/2603.05111) [VERIFIED]

#### Object-pose uncertainty sets (perception side, usable by manipulation filters)
- **"Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation"**, Heng Yang, Marco Pavone. arXiv 2303.12246 (CVPR 2023). Conformal keypoint sets (circles or ellipses) propagate to a Pose UnceRtainty SEt (PURSE) with the same coverage. RANSAG plus an SDP give worst-case error bounds. No safety filter. — [arXiv](https://arxiv.org/abs/2303.12246) [VERIFIED]
- **"Uncertainty Quantification of Set-Membership Estimation in Control and Perception: Revisiting the Minimum Enclosing Ellipsoid"**, Yukai Tang, Jean-Bernard Lasserre, Heng Yang. arXiv 2311.15962 (L4DC 2024 oral). Computes minimum enclosing ellipsoids of set-membership estimates via an SOS hierarchy with pruning and Chebyshev-center enhancements. Applied to system identification and object pose estimation. No safety filter. — [arXiv](https://arxiv.org/abs/2311.15962) [VERIFIED]
- **"Uncertainty Quantification for Visual Object Pose Estimation: S-Lemma Ellipsoidal Bounds"**, Lorenzo Shaikewitz, Charis Georgiou, Luca Carlone. arXiv 2511.21666 (IEEE T-RO). Starting from high-probability keypoint pixel bounds, SLUE (an S-lemma convex program) gives a single ellipsoid that contains the true pose with high probability; an SOS hierarchy converges to the minimum-volume ellipsoid. Motivated by "robust control and planning", but no controller is shown in the abstract. — [arXiv](https://arxiv.org/abs/2511.21666) [VERIFIED]
- **"Correct-by-Construction Vision-based Pose Estimation using Geometric Generative Models"**, Ulices Santa Cruz, Mahmoud Elfar, Yasser Shoukry. arXiv 2601.17556 (2026-01-24). Certified error bounds for NN pose estimators of planar targets (signs, runway markings), trained with a physics-derived generative model; extended to cluttered scenes via NN reachability. — [arXiv](https://arxiv.org/abs/2601.17556) [VERIFIED]

### Inferences
- The "error bound to margin" step has become routine. It takes four forms: a Lipschitz-type MR-CBF margin; a CP quantile added to the CBF condition; a DR/Wasserstein-type tightening; or a tube in robust MPC/SLS. The current frontier is **how to obtain the bound**: state-dependent or heteroscedastic (VISION-SLS, Mondrian CP, the Ruchkin/Ivanov state-dependent CP), adaptive under OOD (ATOM-CBF), or learned (NMR-CBF).
- CSymPlan (FL to a double integrator, measurement error as a bounded disturbance) and CORD-SLS (CP-calibrated perception tubes in robust MPC) are the closest architectural analogues to an FL-MPC whose QP constraints would be tightened by perception error. None is bimanual or arm-arm cooperative. This is an inference from the abstracts.
- Perception contracts so far target aircraft and drone landing, ACC and formation control. I found no perception-contract work on manipulation. The only manipulation-flavoured contract-adjacent work is the pose-set literature (PURSE, SLUE, MEE), which stops at the uncertainty set and never closes the loop into a filter.

### Gaps
- I could not find an arXiv ID for Zhang/Hoxha/Fainekos/Panagou (L-CSS 2025). The arXiv HTML search returned no hit for the title phrase.
- I found no 2023-2026 paper that uses object-pose CP sets (PURSE, SLUE) inside a manipulation CBF or MPC constraint, for example on cup tilt or grasp. This may be a real gap or a search miss.
- No arm-arm or bimanual perception-robust CBF paper turned up.

---

## Q2. Which papers use conformal prediction to build state-estimate / perception regions and plug them into reachability or CBFs? Especially combined with LEARNED reachability values.

### Takeaway
CP perception sets now feed:
- MR-CBFs (Yang et al. 2023; Zhang et al. 2025; Suganda & Hu 2026);
- HJ/CBVF safety in estimator space (Jahanshahi & Chen 2026);
- symbolic reachability with state-dependent CP (Ruchkin/Ivanov group, 2025);
- MDP shields (Scarbro et al., RV 2025);
- planners (Perceive with Confidence, 2024);
- robust-MPC tubes (CORD-SLS 2026);
- a reachability shield for HRC (Thumm et al. 2026).

For **learned** reachability values, CP or ACI has been applied mostly to the *value function's own error*: Lin & Bansal 2023, Tabbara/Sibai 2025, ACoFi 2026. Two papers go further and combine a calibrated set with a learned latent HJ value:
- **UNISafe** (CoRL 2025) calibrates an epistemic-uncertainty OOD threshold with CP and runs reachability in the augmented latent and uncertainty space.
- **Cao & Bansal 2026** ("When World Models Lie") builds ACI sets from the gap between the world model's predicted latent state and the latent inferred from observations, then takes the minimum of the learned HJ value over that set.

### Cited Findings

#### CP perception or state sets into CBF / MR-CBF / shields
- **Yang, Pappas, Mangharam, Lindemann**, arXiv 2304.00194 (CDC 2023). CP state-estimation regions feed a sampled-data MR-CBF on an F1/10 car with LiDAR. See Q1. — [arXiv](https://arxiv.org/abs/2304.00194) [VERIFIED]
- **Zhang, Hoxha, Fainekos, Panagou** (L-CSS 2025). CP bounds from data-driven state estimators feed risk-aware CBFs. — [snippet](https://dblp.uni-trier.de/pid/34/4314.html) [SUMMARY]
- **Suganda & Hu**, arXiv 2603.08958 (2026). Mondrian CP feeds a conformal CBF-QP for field-of-view visibility. — [arXiv](https://arxiv.org/abs/2603.08958) [VERIFIED]
- **"Differentiable Optimization Layered Safety-Critical Control for Risk-Aware Navigation via Conformal Prediction"**, Jinyang Dong, Shizhen Wu, Yongchun Fang. arXiv 2605.16327 (2026-05-05). CP turns sensor noise into "risk-aware obstacle ellipsoids" around an elliptical robot. Nested differentiable optimization layers build the CBFs, which enter a QP. Numerical simulations only. — [arXiv](https://arxiv.org/abs/2605.16327) [VERIFIED]
- **"Conformal Safety Shielding for Imperfect-Perception Agents"**, William Scarbro, Calum Imrie, Sinem Getir Yaman, Kavan Fatehi, Corina S. Pasareanu, Radu Calinescu, Ravi Mangal. arXiv 2506.17275 (RV 2025). — [arXiv](https://arxiv.org/abs/2506.17275) [VERIFIED]
  - *Error set:* CP sets of state estimates from high-dimensional perception.
  - *Shield rule:* an action is allowed only if it is safe for all estimates in the set.
  - *Guarantees:* local safety, plus a proven global bound on the probability of reaching unsafe states.
  - *Experiments:* TaxiNet airplane taxiing.
  - *Does NOT:* handle continuous control; it is a discrete MDP.
- **"Perceive With Confidence: Statistical Safety Assurances for Navigation with Learning-Based Perception"**, Zhiting Mei, Anushri Dixit, Meghan Booker, Emily Zhou, Mariko Storey-Matsutani, Allen Z. Ren, Ola Shorinwa, Anirudha Majumdar. arXiv 2403.08185 (2024-03-13). — [arXiv](https://arxiv.org/abs/2403.08185) [VERIFIED]
  - *Modality:* pre-trained models for object detection and scene completion (occupancy).
  - *Error set:* a novel CP calibration that is robust to state distribution shift.
  - *Into safety:* combined with "any safe planner" for end-to-end statistical safety.
  - *Results:* quadruped hardware in unseen indoor static environments; 70% fewer misdetections than uncalibrated models in simulation; 100% safety.
  - *Does NOT:* handle dynamic scenes or manipulation.

#### CP perception bounds into reachability / verification (Ruchkin, Ivanov, Tran)
- **"State-Dependent Conformal Perception Bounds for Neuro-Symbolic Verification of Autonomous Systems"**, Thomas Waite, Yuang Geng, Trevor Turnquist, Ivan Ruchkin, Radoslav Ivanov. arXiv 2502.21308 (NeuS 2025). Perception error bounds are made heteroskedastic in the *state* rather than in time, and are synthesized by gradient-free optimization to give an optimal interface between perception and control verification. Tested on mountain car. — [arXiv](https://arxiv.org/abs/2502.21308) [VERIFIED]
- **"Statistical-Symbolic Verification of Perception-Based Autonomous Systems using State-Dependent Conformal Prediction"**, Yuang Geng, Thomas Waite, Trevor Turnquist, Radoslav Ivanov, Ivan Ruchkin. arXiv 2512.02893 (2025-12-02). The journal-style extension. — [arXiv](https://arxiv.org/abs/2512.02893) [VERIFIED]
  - *Error set:* state-dependent CP with a genetic-algorithm state-space partition.
  - *Into verification:* a branch-merging hybrid-system reachability algorithm that trades uncertainty for scalability.
  - *Results:* less conservative than the state of the art on two case studies.
  - *Does NOT:* use a CBF, an online filter, or learned dynamics.
- **"Bridging Dimensions: Confident Reachability for High-Dimensional Controllers"**, Yuang Geng, Jake Brandon Baldauf, Souradeep Dutta, Chao Huang, Ivan Ruchkin. arXiv 2311.04843 (2023-11-08). Image controllers are distilled into low-dimensional controllers; their reachable sets are inflated by statistical approximation errors (trajectory-based or action-based). Three OpenAI Gym benchmarks. — [arXiv](https://arxiv.org/abs/2311.04843) [VERIFIED]
- **"Deterministic World Models for Closed-loop Reachability Analysis of End-to-End Vision-based Control"**, Yuang Geng, Zhongzheng Zhang, Chengzhen Jiang, Yanru Li, Xinyang Wang, Zhuoyang Zhou, Hoang-Dung Tran, Ivan Ruchkin. arXiv 2512.08991 (2025-12-08). See Q4. A latent-free state-to-image decoder enables closed-loop reachability, and **CP inflates the reachable sets** by a trajectory-tube deviation bound. — [arXiv](https://arxiv.org/abs/2512.08991) [VERIFIED]
- **"Control Barrier–Value Functions under Partial Observability: Safety Guarantees via Conformal Prediction"**, Niloofar Jahanshahi, Mo Chen. arXiv 2608.13819 (2026-08-13). — [arXiv](https://arxiv.org/abs/2608.13819) [VERIFIED]
  - *Error set:* CP on estimator error at a chosen miscoverage level.
  - *Into safety:* the bound is built into **estimator-space** CBVF (HJ plus CBF) safety analysis, then a QP filter that is robust to bounded disturbance.
  - *Guarantee:* finite-horizon probabilistic safety for the true state.
  - *Experiments:* a partially observable obstacle-avoidance case study.
  - *Does NOT:* claim a learned value function or images in the abstract; it is likely grid or low-dimensional (inference).
- **CORD-SLS**, arXiv 2606.14188. CP-calibrated perception-error bounds become robust-MPC reachable tubes for rope and cloth. See Q1. — [arXiv](https://arxiv.org/abs/2606.14188) [VERIFIED]
- **Thumm et al.**, arXiv 2604.15221. CP human-motion sets feed the SARA reachability shield. See Q1. — [arXiv](https://arxiv.org/abs/2604.15221) [VERIFIED]

#### CP / ACI with LEARNED reachability values (value error, world-model error)
- **Lin & Bansal**, "Verification of Neural Reachable Tubes via Scenario Optimization and Conformal Prediction", arXiv 2312.08604 (L4DC 2024). Already known. Gives probabilistic guarantees for DeepReach-style neural tubes. — [arXiv](https://arxiv.org/abs/2312.08604) [VERIFIED]
- **"Statistically Assuring Safety of Control Systems using Ensembles of Safety Filters and Conformal Prediction"**, Ihab Tabbara, Yuxuan Yang, Hussein Sibai. arXiv 2511.07899 (2025-11-11). CP bounds the gap between a learned HJ value and the realized safety return of the learned safe policy, and calibrates the switch from the nominal controller to the HJ safe policy. Also studies ensembles of HJ values. — [arXiv](https://arxiv.org/abs/2511.07899) [VERIFIED]
- **"Safe Control using Learned Safety Filters and Adaptive Conformal Inference"** (ACoFi), Sacha Huriot, Ihab Tabbara, Hussein Sibai. arXiv 2604.18482 (L4DC 2026). — [arXiv](https://arxiv.org/abs/2604.18482) [VERIFIED]
  - *Method:* a learned HJ safety filter plus ACI; the switching criterion adapts to observed prediction errors.
  - *Guarantee:* the asymptotic miscoverage rate is at most a user parameter. The authors call this a "soft safety guarantee rather than a hard" one.
  - *Experiments:* Dubins car and Safety Gymnasium; beats a fixed threshold, especially under OOD.
- **"Learning Ensembles of Vision-based Safety Control Filters"**, Ihab Tabbara, Hussein Sibai. arXiv 2412.02029 (2024-12-02). An empirical study of ensembles of vision-backbone safety filters on the DeepAccident dataset. No formal bound. — [arXiv](https://arxiv.org/abs/2412.02029) [VERIFIED]
- **"Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures"** (UNISafe), Junwon Seo, Kensuke Nakamura, Andrea Bajcsy. arXiv 2505.00779 (CoRL 2025). — [arXiv](https://arxiv.org/abs/2505.00779) [VERIFIED]
  - *Modality:* high-dimensional observations through a generative world model.
  - *Error set:* world-model **epistemic uncertainty**, with the OOD threshold **calibrated via CP**.
  - *Into safety:* HJ reachability in an **augmented latent + uncertainty** state space, giving a latent safety filter that avoids known and unseen hazards.
  - *Experiments:* vision-based control with a **Franka manipulator**, in simulation and on hardware.
  - *Does NOT:* bound the value error itself or give a hard guarantee (inference).
- **"When World Models Lie: Adaptive Safety Analysis Under Wrong Imaginations"**, John Cao, Somil Bansal. arXiv 2609.34300 (2026-09-28). — [arXiv](https://arxiv.org/abs/2609.34300) [VERIFIED]
  - *Error set:* **ACI** sets built online from the gap between predicted and observation-inferred latent states, i.e. directly observed world-model error.
  - *Into safety:* safety is evaluated pessimistically by **minimizing the learned HJ value over the set**.
  - *Guarantee:* finite-time coverage of the adaptive radius.
  - *Experiments:* simulation and hardware; fewer failures than state-of-the-art latent filters.
  - *Why it matters:* the abstract argues that ensemble disagreement and value-consistency residuals "can remain small" even when predictions deviate.
- **"How Safe Am I Given What I See? Calibrated Prediction of Safety Chances for Image-Controlled Autonomy"**, Zhenjiang Mao, Carson Sobolewski, Ivan Ruchkin. arXiv 2308.12252 (L4DC 2024). Pipelines based on generative world models with CP-calibrated **safety-chance predictions**. Racing car and cartpole. A monitor, not a filter. — [arXiv](https://arxiv.org/abs/2308.12252) [VERIFIED]
- **"Zero-shot Safety Prediction for Autonomous Robots with Foundation World Models"**, Mao, Dai, Geng, Ruchkin. arXiv 2404.00462 (ICRA 2024 workshop). An LLM-based causal latent surrogate for safety prediction. No CP in the abstract. — [arXiv](https://arxiv.org/abs/2404.00462) [VERIFIED]

#### Adjacent CP-in-control papers (not perception, but often cited alongside)
- **Zhou, Zhang, Luo**, "Safety-Critical Control with Uncertainty Quantification using Adaptive Conformal Prediction", arXiv 2407.03569 (ACC 2024). Already known. ACP on *motion noise* feeds a probabilistic CBF inside MPC. — [arXiv](https://arxiv.org/abs/2407.03569) [VERIFIED]
- **ACQR**: Zhou, Zhang, Lyu, Luo, "Safety-Critical Control under Uncertainty via Adaptive Conformal Quantile Prediction Intervals", arXiv 2609.23280 (CDC 2026). Already known; a direct follow-up from the same group. ACP plus conformal quantile regression gives asymmetric, state-dependent intervals that feed a probabilistic CBF inside MPC. Simulation only. — [arXiv](https://arxiv.org/abs/2609.23280) [VERIFIED]
- **"Conformal Reachability for Safe Control in Unknown Environments"**, Xinhang Ma, Junlin Wu, Yiannis Kantaros, Yevgeniy Vorobeychik. arXiv 2602.03799 (2026). CP intervals on **unknown dynamics** at each step feed reachability, and policies are trained to maximize the provably safe horizon. Seven settings including cartpole, lane following, drone and navigation. — [arXiv](https://arxiv.org/abs/2602.03799) [VERIFIED]
- **"Statistical Guarantees in Data-Driven Nonlinear Control: Conformal Robustness for Stability and Safety"**, Ting-Wei Hsu, Hiroyasu Tsukamoto. arXiv 2506.06228 (2025). CR-CLF and CR-CBF robustified against *learned-model* prediction error. — [arXiv](https://arxiv.org/abs/2506.06228) [VERIFIED]
- **"CP-NCBF"**, Tayal, Singh, Jagtap, Kolathaya. arXiv 2503.17395 (2025). Split CP verifies neural CBFs. — [arXiv](https://arxiv.org/abs/2503.17395) [VERIFIED]
- **"Safe Probabilistic Planning for Human-Robot Interaction using Conformal Risk Control"**, Gonzales, Mizuta, Leung, Ratliff. arXiv 2603.10392 (2026). Conformal risk control on CBF safety values under human prediction error. — [arXiv](https://arxiv.org/abs/2603.10392) [VERIFIED]
- **"Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization"**, Han Wang, Chao Ning. arXiv 2510.16376 (NeurIPS 2025). Feedback-based CP with a joint risk constraint for shrinking-horizon trajectory optimization. Prediction, not perception. **Distinct from the Zhang/Panagou L-CSS paper with a similar title.** — [arXiv](https://arxiv.org/abs/2510.16376) [VERIFIED]
- **Survey**: Lindemann, Zhao, Yu, Pappas, Deshmukh, "Formal Verification and Control with Conformal Prediction", arXiv 2409.00536 (2024). — [arXiv](https://arxiv.org/abs/2409.00536) [VERIFIED]

### Inferences
- Two distinct "error sets" are calibrated in the learned-value line:
  - **value or certificate error**: Lin & Bansal, Tabbara/Sibai, ACoFi, CP-NCBF;
  - **state or latent error**: Yang, Jahanshahi & Chen, UNISafe, Cao & Bansal.

  Only Cao & Bansal 2026 does the specific "perception/WM error set, then min of the learned HJ value over the set" operation. That is the closest prior art for a perception-robust latent HJ shield. UNISafe is the closest that also does manipulation (Franka).
- Every CP guarantee here is marginal or finite-sample under exchangeability. The ACI variants give long-run or asymptotic miscoverage bounds; ACoFi explicitly calls its guarantee "soft". None gives a hard per-episode safety guarantee for a learned value. This is an inference from the abstracts' wording.
- State-dependent or heteroscedastic CP (Waite 2025, Geng 2025, Suganda 2026, ACQR 2026, VISION-SLS's state-dependent bounds) is the main lever against the conservatism of a global quantile.

### Gaps
- I did not find a paper that uses a CP *perception* set (from images or object pose) inside a **learned** HJ value for **manipulation** with hard constraints. UNISafe calibrates an OOD threshold, not a state set. Cao & Bansal's hardware platform is not named in the abstract.
- I could not confirm whether Jahanshahi & Chen's CBVF is learned or grid-based. The abstract does not say.

---

## Q3. Which papers use zonotopes, constrained zonotopes or ellipsoids for perception uncertainty in robot safety (runtime monitoring, set-based MPC), and how do they scale to high-dimensional or learned systems?

### Takeaway
Set-based representations are used in four ways:
1. **Wrapping a learned perception module as bounded or set-valued measurement noise in a low-dimensional state space.** Examples are the constrained-zonotope observer plus LP robust MPC with a CNN camera (Niknejad et al. 2025), CCG obstacle estimates converted to CBFs (Matias & Silvestre 2026), and constrained/hybrid-zonotope fusion of V2X perception (Narri et al. 2025).
2. **Bounding sensor error in runtime monitors.** Examples are zonotope abstract domains for RLola (2026) and PZR on a 5-DoF MuJoCo arm (2026).
3. **Ellipsoids for pose or obstacle uncertainty.** Examples are SLUE, MEE, PURSE, GM-IPC and CP ellipsoids into CBFs.
4. **Zonotope or hybrid-zonotope propagation through CNNs.** This is shown only on small examples.

None propagates zonotopes through a learned latent world model. Scaling is handled either by staying low-dimensional or by order reduction (PZR, RLola).

### Cited Findings
- **"Robust Model Predictive Control Design for Autonomous Vehicles with Perception-based Observers"**, Nariman Niknejad, Gokul S. Sankar, Bahare Kiumarsi, Hamidreza Modares. arXiv 2509.05201 (2025-09-05). — [arXiv](https://arxiv.org/abs/2509.05201) [VERIFIED]
  - *Modality:* camera with a CNN perception module.
  - *Error set:* **constrained-zonotope** set-based estimation that captures biased, heavy-tailed (non-Gaussian) perception error.
  - *Into safety:* robust MPC reformulated as an **LP**, with a Minkowski-Lyapunov cost and contractive zonotopic invariant sets; the terminal set comes from an ellipsoidal approximation.
  - *Guarantee:* closed-loop stability and bounded estimation error.
  - *Experiments:* an omnidirectional mobile robot on hardware under ROS2.
  - *Does NOT:* model obstacles or collisions in the abstract; the dynamics are linear (inference from the LP form).
- **"Predictive Zonotope Reduction: Precise Runtime Monitoring under Uncertainty"** (PZR), Vladimir Krsmanovic, Florian Kohn, Bernd Finkbeiner, Milan Simovic. arXiv 2609.03699 (2026-09-03). Already known. — [arXiv](https://arxiv.org/abs/2609.03699) [VERIFIED]
  - *Method:* reducer selection is posed as optimal control and solved by beam-search MPC, then distilled into a small neural policy.
  - *Setting:* RLola monitoring of a **5-DoF robotic arm in MuJoCo** with ISO 5725 sensor uncertainty, running on a Raspberry Pi 5.
  - *Result:* fewer false positives than static reduction.
  - *Does NOT:* include a controller or filter; it is monitoring only.
- **"Cutting Corners on Uncertainty: Zonotope Abstractions for Stream-based Runtime Monitoring"**, Bernd Finkbeiner, Martin Fränzle, Florian Kohn, Paul Kröger. arXiv 2601.11358 (2026-01-16). The direct precursor to PZR. — [arXiv](https://arxiv.org/abs/2601.11358) [VERIFIED]
  - *Method:* zonotopes as an abstract domain for Robust-Lola; affine arithmetic with slack variables tracks sensor calibration and measurement error.
  - *Result:* a sound, bounded-memory over-approximation; several reduction strategies are compared.
- **"Safe Navigation under Uncertain Obstacle Dynamics using Control Barrier Functions and Constrained Convex Generators"**, Hugo Matias, Daniel Silvestre. arXiv 2601.07715 (2026-01-12). — [arXiv](https://arxiv.org/abs/2601.07715) [VERIFIED]
  - *Error set:* guaranteed set-valued **CCG** estimates of obstacle states, propagated over the sampling interval.
  - *Into safety:* the sets are converted into CBFs via a convex program (validity from the Implicit Function Theorem), merged, and enforced in a CBF-QP. Minkowski sums give rigid-body agents of arbitrary shape.
- **"Shared Situational Awareness Using Hybrid Zonotopes with Confidence Metric"**, Vandana Narri, Jonah J. Glunt, Joshua A. Robbins, Jonas Mårtensson, Herschel C. Pangborn, Karl H. Johansson. arXiv 2512.11493 (2025-12-12). Constrained-zonotope set estimation computes a per-sensor confidence metric that handles inconsistent measurements such as false positives. Sets are fused with **hybrid zonotopes** for V2X pedestrian perception, in simulation and real experiments. Perception fusion, not a filter. — [arXiv](https://arxiv.org/abs/2512.11493) [VERIFIED]
- **"Safe Multi-Robot Coordination via VLM-LLM Reasoning and Reachability Analysis"**, Mohamed Dwedar, Ahmad Hafez, Alexander Jesser, Amr Alanwar. arXiv 2609.27816 (listed 2026-08-18). VLM semantics are mapped to "conservative metric geometric constraints (inflated obstacle sets, safe corridors)". A robot-specific **zonotope reachability gate** propagates reachable tubes to check obstacle avoidance, corridor containment and inter-robot separation before approving commands. Online experiments with a quadruped and a camera-less vehicle. The inflation is not calibrated to a perception error bound in the abstract. — [arXiv](https://arxiv.org/abs/2609.27816) [VERIFIED]
- **"Efficient Reachability Analysis for Convolutional Neural Networks Using Hybrid Zonotopes"**, Yuhao Zhang, Xiangru Xu. arXiv 2503.10840 (ACC 2025). Hybrid-zonotope set propagation through CNNs, with network reduction, for verifying perception or control networks. "Numerical examples" only. — [arXiv](https://arxiv.org/abs/2503.10840) [VERIFIED]
- **Ellipsoids for perception uncertainty:** — [2511.21666](https://arxiv.org/abs/2511.21666); [2311.15962](https://arxiv.org/abs/2311.15962); [2303.12246](https://arxiv.org/abs/2303.12246); [2603.04329](https://arxiv.org/abs/2603.04329); [2605.16327](https://arxiv.org/abs/2605.16327) [VERIFIED]
  - SLUE (pose ellipsoid from keypoint bounds; T-RO);
  - MEE of set-membership pose estimates (L4DC 2024);
  - PURSE (CP keypoints to a non-convex pose set, then worst-case bounds; CVPR 2023);
  - GM-IPC (unions of ellipsoids; ACC 2026);
  - Dong et al. (CP obstacle ellipsoids into CBF; 2026).
- **"CSymPlan"**, arXiv 2608.22983 (2026). A bounded measurement-uncertainty disturbance enters an abstraction-based (symbolic) reach-avoid synthesis on an FL double-integrator model of a Franka FR3. This is set-based in spirit, though the abstract does not name the set representation. — [arXiv](https://arxiv.org/abs/2608.22983) [VERIFIED]

### Inferences
- **How they scale:** every set-based perception-safety method above works on a **low-dimensional estimated state**: robot pose, obstacle states, human keypoints, arm joint readings. The learned perception module is treated as a black-box sensor with a set-valued error. Set propagation through learned networks (hybrid zonotopes for CNNs; abstract rendering for Gaussian splats) remains small-scale. I found none that propagates zonotopes through a learned latent dynamics or world model.
- Zonotope order growth is the recognized bottleneck for online use. PZR and RLola address it with learned or heuristic reduction, and CCG and constrained zonotopes trade exactness against LP/QP size. For a 12-DoF bimanual joint-space state, the zonotope order grows with the generators added per perception update. A reduction scheme like PZR's would be needed for real-time QP tightening; this is an inference.

### Gaps
- I found no Althoff/CORA 2023-2026 paper that propagates a **learned-perception error set** (rather than sensor-noise intervals) into a reachability shield, except Thumm et al. 2026 (CP into SARA). Targeted searches for CORA plus perception returned only older online-verification work.
- I found no ellipsoidal or zonotopic perception-uncertainty work for bimanual manipulation.

---

## Q4. Closed-loop verification of vision-based controllers with perception error models (Katz et al., NNV, Hsieh et al.): what do they do, and does any of it touch manipulation?

### Takeaway
Four families exist:
1. **Generative surrogates of the camera, verified with NN or reachability tools.** Examples are Katz 2021 (GAN), Cai/Fan/Bak 2024 (cGAN composition and unrolling), Geng/Tran/Ruchkin 2025 (deterministic state-to-image world model plus CP inflation), Akinwande/Kochenderfer/Barrett 2026 (stochastic world models; 80%+ of the RGB benchmark resolved), and Estornell/Everett 2025 (geometric perturbation maps from state to image).
2. **Perception abstractions or contracts.** Examples are Hsieh 2021, DaRePC 2023, LPC 2025, abstract rendering 2025, and Pasareanu's confusion-matrix abstractions.
3. **Statistical inflation of low-dimensional reachability.** Bridging Dimensions and state-dependent CP.
4. **HJ-based failure discovery.** Chakraborty & Bansal.

Benchmarks are all aircraft taxi or landing, ACC, CARLA braking, Gym, or indoor navigation. **None of the closed-loop vision-verification papers I found targets manipulation.** The manipulation-adjacent exceptions sit outside this verification line: PZR's arm monitor, Chou 2022, CORD-SLS and UNISafe.

### Cited Findings
- **Foundation:** Katz, Corso, Strong, Kochenderfer, "Verification of Image-based Neural Network Controllers Using Generative Models", arXiv 2105.07091 (DASC 2021). A GAN maps state to image and is concatenated with the controller for low-dimensional closed-loop verification. Aircraft taxi. — [arXiv](https://arxiv.org/abs/2105.07091) [VERIFIED]
- **Foundation:** Hsieh, Joshi, Misailovic, Mitra, "Verifying Controllers with Convolutional Neural Network-based Perception: A Case for Intelligible, Safe, and Precise Abstractions", arXiv 2111.05534 (submitted to HSCC 2022). Infers perception abstractions from requirements and data, then verifies with CBMC. Lane keeping and an agricultural robot. — [arXiv](https://arxiv.org/abs/2111.05534) [VERIFIED]
- **"Efficient Determination of Safety Requirements for Perception Systems"**, Sydney M. Katz, Anthony L. Corso, Esen Yel, Mykel J. Kochenderfer. arXiv 2307.01371 (2023-07-03; submitted to DASC 2023). Uses "smoothing bandits" (Gaussian processes plus threshold bandits) to find the set of *safe perception performance characteristics* from a black-box closed-loop simulator. Vision-based aircraft collision avoidance. This is the reverse direction: system safety to a requirement on perception error. — [arXiv](https://arxiv.org/abs/2307.01371) [VERIFIED]
- Corso, Katz, Kochenderfer, "Risk-Driven Design of Perception Systems" (NeurIPS 2022). Reports 37% lower collision risk on vision-based detect-and-avoid. — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/40739b3bb584c117b3e2f418d17f63a1-Abstract.html) [SUMMARY]
- **"Scalable Surrogate Verification of Image-based Neural Network Control Systems using Composition and Unrolling"**, Feiyang Cai, Chuchu Fan, Stanley Bak. arXiv 2405.18554 (AAAI-25). cGAN surrogate; composing dynamics, cGAN and controller removes one-step overapproximation, and unrolling several control steps reduces multi-step error, using existing NN verifiers. — [arXiv](https://arxiv.org/abs/2405.18554) [VERIFIED]
- **"Deterministic World Models for Closed-loop Reachability Analysis of End-to-End Vision-based Control"**, Geng, Zhang, Jiang, Li, Wang, Zhou, Hoang-Dung Tran, Ruchkin. Tran being NNV's lead developer is my background knowledge, not verified this session. arXiv 2512.08991 (2025-12-08). — [arXiv](https://arxiv.org/abs/2512.08991) [VERIFIED]
  - *Model:* a latent-free decoder from state to image, trained with a saliency-reconstruction loss plus a control-consistency loss.
  - *Verification:* closed-loop reachability, with **CP inflation** to transfer the surrogate's guarantee to the real system with high probability.
  - *Experiments:* CARLA braking, CartPole, MountainCar, Pendulum; tighter than cGAN baselines.
- **"Stochastic World Models for Verifying Vision-Based Neural Feedback Systems"**, I. Samuel Akinwande, Mykel J. Kochenderfer, Clark Barrett. arXiv 2609.38120 (2026-09-29). — [arXiv](https://arxiv.org/abs/2609.38120) [VERIFIED]
  - *Model:* stochastic world models with physically grounded latents, built from operations that standard verifiers can bound.
  - *Verification:* falsification, adaptive refinement, symbolic and backward analyses.
  - *Results:* resolves the whole state space of an emergency-braking benchmark where the state of the art left 38% unresolved; resolves over 80% of the RGB version, where no prior verification results existed. The surrogate reproduces held-out frames better than GANs with up to 130x as many parameters.
- **"Verification of Visual Controllers via Compositional Geometric Transformations"**, Alexander Estornell, Leonard Jung, Michael Everett. arXiv 2507.04523 (2025-07-06). Models uncertain observations as **geometric perturbations**, not Lp pixel balls, to build a boundable mapping from state to image. Gives sound outer-approximations of reachable sets on benchmark control environments. — [arXiv](https://arxiv.org/abs/2507.04523) [VERIFIED]
- **"Closed-loop Analysis of Vision-based Autonomous Systems: A Case Study"**, Corina S. Pasareanu, Ravi Mangal, Divya Gopinath, Sinem Getir Yaman, Calum Imrie, Radu Calinescu, Huafeng Yu. arXiv 2302.04634 (2023-02-06). Replaces the camera and DNN with a **probabilistic abstraction from confusion matrices** for probabilistic model checking, and uses local DNN analyses as run-time guards. TaxiNet. The follow-up is conformal shielding ([2506.17275](https://arxiv.org/abs/2506.17275)). — [arXiv](https://arxiv.org/abs/2302.04634) [VERIFIED]
- **"Discovering Closed-Loop Failures of Vision-Based Controllers via Reachability Analysis"**, Kaustav Chakraborty, Somil Bansal. arXiv 2211.02736 (2022-11-04). HJ BRT with simulation finds visual inputs that cause closed-loop failure. — [arXiv](https://arxiv.org/abs/2211.02736) [VERIFIED]
- **"Enhancing Safety and Robustness of Vision-Based Controllers via Reachability Analysis"**, Kaustav Chakraborty, Aryaman Gupta, Somil Bansal. arXiv 2410.21736 (2024-10-29). — [arXiv](https://arxiv.org/abs/2410.21736) [VERIFIED]
  - *Method:* neural reachable tubes mine failures. Two uses follow: an online learned failure-monitor classifier that triggers a fallback controller, and offline retraining on augmented failure data.
  - *Does NOT:* give a perception error bound.
- **Bridging Dimensions**, arXiv 2311.04843, and **state-dependent CP verification**, arXiv 2502.21308 and 2512.02893. See Q2. — [2311.04843](https://arxiv.org/abs/2311.04843); [2512.02893](https://arxiv.org/abs/2512.02893) [VERIFIED]
- **"System-Level Analysis of Module Uncertainty Quantification in the Autonomy Pipeline"**, Sampada Deglurkar, Haotian Shen, Anish Muthali, Marco Pavone, Dragos Margineantu, Peter Karkus, Boris Ivanovic, Claire J. Tomlin. arXiv 2410.12019 (CDC 2024; TCST version). Derives probabilistic specifications on a module's uncertainty measure from a system specification, and input-output robustness to compare designs. AV and aircraft runway-incursion systems. — [arXiv](https://arxiv.org/abs/2410.12019) [VERIFIED]
- **Perception-contract closed-loop work:** — [2311.08652](https://arxiv.org/abs/2311.08652); [PMLR](https://proceedings.mlr.press/v283/li25c.html); [2503.00308](https://arxiv.org/abs/2503.00308) [VERIFIED]
  - DaRePC, arXiv 2311.08652;
  - LPC, L4DC 2025;
  - Abstract Rendering, arXiv 2503.00308.
- Santa Cruz, Elfar, Shoukry, arXiv 2601.17556 (2026). Certified pose estimators via geometric generative models plus NN reachability. Follows NNLander-VeriF (arXiv 2203.15841, 2022). — [2601.17556](https://arxiv.org/abs/2601.17556) [VERIFIED]; [2203.15841](https://arxiv.org/pdf/2203.15841) [SUMMARY]
- Mitra et al. 2024 survey of formal verification for vision-based systems. — [NSF PAR](https://par.nsf.gov/biblio/10642945) [SUMMARY]

### Inferences
- The field is moving from GAN surrogates toward **world-model surrogates** (DWM 2025; stochastic WMs 2026). These are small, verifier-friendly decoders from state to image, statistically inflated with CP to cover the real camera. It is the mirror image of the latent-safety-filter line: they keep a physical low-dimensional state and learn the image, whereas latent HJ learns the state from the image.
- Manipulation is absent from this verification literature. The likely reasons are the high-dimensional contact state and object state that is not directly observable. The pose-set papers (PURSE, SLUE) could serve as the "perception contract" for manipulation, but nobody has closed that loop in the papers found. This is an inference.

### Gaps
- I did not find a dedicated 2023-2026 **NNV** paper on closed-loop camera verification. Tran (NNV) is a co-author of DWM 2512.08991, but the abstract does not say NNV is the reachability engine.
- I did not verify the venue of Pasareanu et al. 2302.04634 or Chakraborty & Bansal 2211.02736. The arXiv comment fields I read did not state them.
- arXiv API access was rate-limited (HTTP 429) mid-session. Discovery then relied on WebSearch, the arxiv.org HTML search and abstract pages, so some 2025-2026 papers may have been missed, especially non-arXiv L4DC, CDC and HSCC papers.
