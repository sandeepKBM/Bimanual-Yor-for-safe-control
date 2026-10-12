# Safe manipulation under perception uncertainty for payload-dynamics hazards (spill / slide / roll / topple), 2023-2026

Scope note for the report writer. Search date 2026-10-05. Method: arXiv API keyword and author queries (titles, authors, IDs and first-submission dates taken from the arXiv API record), plus full-text reads of the arXiv HTML/PDF for the most relevant papers, plus web search for non-arXiv venue papers.
Tags: **VERIFIED** = read on the paper's arXiv abstract record, arXiv HTML/PDF full text, or the publisher/institutional record. **SUMMARY** = from a search-engine snippet or third-party page only.
Dates are arXiv first-submission dates (v1). Papers the caller already knows (UNISafe 2505.00779, AnySafe 2509.19555, ReKep 2409.01652, SafeBimanual 2508.18268, Cao & Bansal 2609.34300, LatentSafe 2502.00935, LatentCBF 2511.18606, Kim et al. 2510.06492) are only cross-referenced, not re-reviewed.

## Q1. Non-prehensile / tray / plate transport with safety constraints (friction cone, ZMP, slosh models): who does it, and does anyone perceive a free object on the tray (e.g. a rolling ball)?

### Takeaway
There is a dense, mature 2023-2026 literature on the "waiter problem". It covers sticking/friction-cone and tipping constraints inside MPC/QP/OCP, and pendulum-type slosh models with tilt or free-surface spill limits. Almost all of it assumes the payload's pose and parameters are known, or handles only parametric uncertainty (friction, inertia). None of these papers perceives the payload online in closed loop with a formal safety constraint. No paper was found that perceives a free rolling object (ball) on a carried tray and enforces a safety constraint on it. The closest items are older or not safety-framed (STORM ball-on-tray, CyberRunner labyrinth).

### Cited Findings

#### Rigid-object waiter problem (slide / topple), model-based constraints
- **Keep it Upright: Model Predictive Control for Nonprehensile Object Transportation with Obstacle Avoidance on a Mobile Manipulator.** Adam Heins, Angela P. Schoellig. arXiv 2305.17484, 2023-05-27 (RA-L 2023 per lab PDF, SUMMARY). [VERIFIED abstract]
  - Payload/failure: up to seven balanced and stacked objects, including a tall bottle; slide and topple.
  - Mechanism: whole-body constrained MPC with balancing (contact) constraints plus obstacle avoidance. Plans with "minimum statically-feasible friction coefficients" for robustness to frictional uncertainty.
  - Hardware: velocity-controlled mobile manipulator; EE speeds and accelerations up to 2.0 m/s and 7.9 m/s²; dodges a thrown ball while balancing a bottle. Single arm.
  - Does NOT: perceive the balanced objects or model perception uncertainty; no learned component.
  - [Source](https://arxiv.org/abs/2305.17484)
- **Robust Nonprehensile Object Transportation with Uncertain Inertial Parameters.** Adam Heins, Angela P. Schoellig. arXiv 2411.07079, 2024-11-11. [VERIFIED full text]
  - Mechanism: sticking constraints (no slide, no tip, no contact loss via the contact wrench cone) made robust to worst-case CoM over a polyhedral set. A post-hoc SDP (moment relaxation) verifies the plan against any realizable inertia on a bounding shape.
  - Perception: the object pose on the tray is assumed known; only inertial-parameter uncertainty is treated. Hardware: Ridgeback base + UR10 (Vicon for the base).
  - Results: a 56 cm box with an offset mass is transported 3/3 runs with robust constraints, while baselines drop it.
  - Limitations: the planner uses soft constraints ("robust sticking is not guaranteed but is heavily encouraged"); inertia uncertainty is ignored while planning and checked only post hoc (~2.2 s per check).
  - [Source](https://arxiv.org/html/2411.07079v3)
- **Robotic Nonprehensile Object Transportation with a Hanging Tray.** Heins, Schoellig. arXiv 2606.10039, 2026-06-08. [VERIFIED abstract]
  - Approach: a rope-suspended tray acts as a 3D pendulum. It reduces sliding and slosh while only a 3-DOF mobile base is actuated.
  - Perception: computer vision is used only to find people with a raised hand (visual servoing to them), not to perceive the payload.
  - Does NOT: model perception uncertainty or give a formal guarantee.
  - [Source](https://arxiv.org/abs/2606.10039)
- **Non-Prehensile Object Transportation via Model Predictive Non-Sliding Manipulation Control.** M. Selvaggio, A. Garg, F. Ruggiero et al. IEEE TCST 2023. [SUMMARY, search snippet]
  - MPC enforces sticking contacts with the parametrized contact forces kept in a conservatively approximated friction cone, plus joint position, velocity and torque limits.
  - Simulation only, on a torque-controlled 7-DoF KUKA iiwa.
  - [Source](https://core.ac.uk/works/151760597); [PDF](https://www.iris.unina.it/retrieve/39335e44-b012-40d2-ac3c-6ebaa62b93ac/tcst2023.pdf)
- **A non-prehensile object transportation framework with adaptive tilting based on quadratic programming.** R. Subburaman, M. Selvaggio, F. Ruggiero. IEEE RA-L 8(6):3581-3588, 2023. [SUMMARY]
  - An operational-space QP adaptively tilts the tray to prevent slippage.
  - [Source](https://www.iris.unina.it/handle/11588/895198)
- **Time-Optimized Trajectory Planning for Non-Prehensile Object Transportation in 3D.** Lingyun Chen, Haoyu Yu, Abdeldjallil Naceri, Abdalla Swikir, Sami Haddadin. arXiv 2408.16420, 2024-08-29. [VERIFIED abstract]
  - Object-stability model plus tray-rotation-aware time-optimal planning along straight 3D lines; Franka Panda.
  - No perception or uncertainty.
  - [Source](https://arxiv.org/abs/2408.16420)
- **Hearing the Slide: Acoustic-Guided Constraint Learning for Fast Non-Prehensile Transport.** Yuemin Mao, Bardienus P. Duisterhof, Moonyoung Lee, Jeffrey Ichnowski. arXiv 2506.09169, 2025-06-10. [VERIFIED full text]
  - Mechanism: a contact microphone under the tray trains a NN mapping tray motion to a dynamically conditioned friction coefficient. This coefficient is a hard friction constraint in an SQP time-optimal planner on a UR5e.
  - Results: object displacement falls by 53.9% (PLA model) and 86.0% (more conservative glass model) on average. Transport time rises by 12.6% and 27.0%.
  - Sensing: the learned model appears to be used offline in planning; the paper does not state online acoustic feedback. Slide ground truth is measured post hoc with a top-down camera and a grid sticker.
  - Does NOT: give calibrated uncertainty on the learned friction or perceive the object online.
  - [Source](https://arxiv.org/html/2506.09169v1)
- **Towards Multi-Object Nonprehensile Transportation via Shared Teleoperation: A Framework Based on Virtual Object MPC.** Xinyang Fan, Zhaoyang Chen, Shu Xin, Yi Ren, Zainan Jiang, Fenglei Ni, Hong Liu. arXiv 2604.06932, 2026-04-08. [VERIFIED abstract]
  - Mechanism: the human controls position, the robot controls tray orientation; a virtual-object (VO) constraint simplification feeds an MPC trajectory smoother.
  - Results: nine objects at up to 2.4 m/s²; sliding distance −72.45%; tip-overs 0% vs 13.9% for the baseline.
  - Gap: how the objects are perceived is not stated in the abstract, and the HTML full text returned 404.
  - [Source](https://arxiv.org/abs/2604.06932)
- **Dynamic Non-Prehensile Object Transport via Model-Predictive Reinforcement Learning (CV-MPC).** Neel Jawale, Byron Boots, Balakumar Sundaralingam, Mohak Bhardwaj. arXiv 2412.00086, 2024-11-27. [VERIFIED full text]
  - Mechanism: an ensemble of value functions is learned offline from 50-100 demos and combined risk-aversely (log-sum-exp) inside an MPC that also enforces friction-cone, joint-limit and collision constraints.
  - Perception: the paper says it considers "a perception free setting". Slip labels come from a human watching tray markings.
  - Results: Franka Panda, single arm; success = object moves <2 cm; 88.33% on convex shapes and ≥80% on household objects.
  - Uncertainty: covers data coverage (epistemic, via the ensemble), not perception.
  - [Source](https://arxiv.org/html/2412.00086)
- **Contact-Guided Exploration for Non-Prehensile Locomanipulation with Multi-Critic RL.** Tolomei, Mittal, Angelini, Garabini, Salaris, Hutter. arXiv 2608.28140, 2026-08-28. [VERIFIED abstract]
  - Non-prehensile pushing/transport of a chair by a quadrupedal manipulator.
  - Not a payload-on-tray safety method.
  - [Source](https://arxiv.org/abs/2608.28140)
- **Before the Tipping Point: Force-Guided Active Perception for Shape-Agnostic Estimation of 3D Centers of Mass.** Steven M. Hyland, Jing Xiao, Cagdas D. Onal. arXiv 2609.12894, 2026-09-11. [VERIFIED abstract]
  - Estimates mass, CoM height and toppling angle (<5% error) from a sub-critical tipping push using a 6-axis F/T sensor. A safety margin keeps the object below critical tipping.
  - This is topple-safety via force perception, not vision; no transport.
  - [Source](https://arxiv.org/abs/2609.12894)

#### Liquid / slosh-aware transport (spill)
- **Model Predictive Control for Non-Prehensile Robotic Manipulation of Liquid Containers with Sloshing Boundedness Guarantees.** G. Medico, R. Di Leva, G. Palli, M. Carricato, L. Biagiotti. IEEE/ASME AIM 2026, DOI 10.1109/AIM65483.2026.11658331 (not found on arXiv). [VERIFIED, institutional record abstract]
  - Model: jerk-controlled triple integrator for the robot; nonlinear equivalent spherical-pendulum slosh model under 6D EE motion; multiple containers.
  - Constraints: a friction-cone constraint per container, plus a closed-form free-surface-elevation estimate that bounds spillage.
  - Solver and hardware: SQP at 50 Hz planner / 500 Hz robot; UR5e.
  - Does NOT: perceive the liquid. The slosh state is model-predicted.
  - [Source](https://cris.unibo.it/handle/11585/1081652)
  - A separate snippet adds a "constraint dominance analysis": the friction cone subsumes tipping and torsional slip for typical containers. [SUMMARY, search snippet only]
- **Time-Optimal Transport of Loosely Placed Liquid Filled Cups along Prescribed Paths.** Klaus Zauner, Hubert Gattringer, Andreas Mueller. arXiv 2510.25255, 2025-10-29. [VERIFIED abstract]
  - Slosh dynamics are embedded in a direct-multiple-shooting OCP for a cup loose on a tray, minimizing sloshing to avoid spillage.
  - No perception.
  - [Source](https://arxiv.org/abs/2510.25255)
- **Optimal Feed-Forward Control for Robotic Transportation of Solid and Liquid Materials via Nonprehensile Grasp.** Luigi Biagiotti, Davide Chiaravalli, Riccardo Zanella, Claudio Melchiorri. arXiv 2306.14212, 2023-06-25. [VERIFIED abstract]
  - A feed-forward smoothing plus EE re-orientation plug-in prevents sliding and sloshing under velocity/acceleration limits.
  - Mocap tracks the operator's hand (the teleop input), not the payload.
  - [Source](https://arxiv.org/abs/2306.14212)
- **Geometric Slosh-Free Tracking for Robotic Manipulators.** Jon Arrizabalaga, Lukas Pries, Riddhiman Laha, Runkang Li, Sami Haddadin, Markus Ryll. arXiv 2402.05197, 2024-02-07. [VERIFIED abstract]
  - A virtual-quadrotor (differentially flat) slosh-free reference orientation feeds a QP resolved-acceleration controller; Franka; real time.
  - No pendulum model, no perception.
  - [Source](https://arxiv.org/abs/2402.05197)
- **Emergency Stopping for Liquid-manipulating Robots.** Samuli Hynninen, Ville Kyrki. arXiv 2604.16667, 2026-04-17. [VERIFIED full text]
  - Mechanism: an MPC computes time-optimal spill-free stopping trajectories as a plug-and-play safety layer. It uses a spherical-pendulum model with bilinear vertical-acceleration coupling and a tilt bound (5°).
  - Results (Franka): 0.87 s stop with 5.3° max slosh, vs 0.73 s and 22.3° for a no-slosh-model baseline.
  - Perception: hazard detection is assumed external; the slosh state is propagated open-loop with no liquid sensing.
  - Uncertainty: not modelled. Only a ±50% rod-length robustness test.
  - [Source](https://arxiv.org/html/2604.16667)
- **Clutter-Aware Spill-Free Liquid Transport via Learned Dynamics.** Ava Abderezaei, Anuj Pasricha, Alex Klausenstock, Alessandro Roncone. arXiv 2408.00215, 2024-08-01. [VERIFIED abstract]
  - Mechanism: RRT* with a container-aware informed sampler, time parameterization, and a transformer that classifies trajectories as spill-free or not. This is a learned spill classifier, a kind of learned safety certificate.
  - Real-world: various container shapes and fill levels; feasible space at least 3× larger than prior work.
  - Gap: whether fill level is perceived or given is not stated in the abstract.
  - [Source](https://arxiv.org/abs/2408.00215)
- **Fetch My Beer: Synthetic-to-real Hierarchical Policy for Smooth Pick-and-place.** Yingyue Li, Chenyangguang Zhang, Ruida Zhang, Bowen Fu, Guangyao Zhai, Xiangyang Ji. arXiv 2609.18119, 2026-09-16. [VERIFIED abstract]
  - Safety comes from the data: a VLM filters unstable grasp poses and fluid simulation validates transport trajectories. A latent-diffusion controller then executes smooth motions.
  - No runtime safety filter or uncertainty model.
  - [Source](https://arxiv.org/abs/2609.18119)
- **Shared Telemanipulation with VR controllers in an anti slosh scenario.** Max Grobbel, Balint Varga, Sören Hohmann. arXiv 2309.07714, 2023-09-14. [VERIFIED abstract]
  - NMPC teleoperation with slosh suppression on a UR5e.
  - Follow-up: **Towards Universal Shared Control in Teleoperation Without Haptic Feedback**, Grobbel, Schneider, Hohmann, arXiv 2506.23624, 2025-06-30 [VERIFIED abstract]. Collision-free plus slosh-suppressing optimization at 13 ms latency.
  - [Source](https://arxiv.org/abs/2309.07714); [Source](https://arxiv.org/abs/2506.23624)
- **Shared Autonomy Control for Slosh-Free Teleoperation.** R. I. Cabral Muchacho, S. Bien, R. Laha, A. Naceri, L. F. C. Figueredo, S. Haddadin. IROS 2023. [SUMMARY]
  - [Source](https://dblp1.uni-trier.de/pers/l/Laha:Riddhiman.html)
- Foundations, one line each:
  - Muchacho et al., "A Solution to Slosh-free Robot Trajectory Optimization", arXiv 2210.12614 (IROS 2022). Spherical-pendulum point mass, QP. [VERIFIED abstract] [Source](https://arxiv.org/abs/2210.12614)
  - Selvaggio et al., shared-control teleoperation for tray transport with haptic cues, IEEE T-RO (pre-2023). [SUMMARY] [Source](https://www.iris.unina.it/handle/11588/854570)
  - Dual-arm slosh-free liquid transfer via expanded impedance control (J. Vibration and Control 27(23-24), 2021). [SUMMARY] [Source](https://avesis.gazi.edu.tr/yayin/07742454-9ffa-49ae-b349-d5910077923b/an-expanded-impedance-control-scheme-for-slosh-free-liquid-transfer-by-a-dual-arm-cooperative-robot)

#### Free / rolling object on a tray or plate with perception (closest hits)
- **STORM** (Bhardwaj et al.). arXiv 2104.13542, 2021-04-28, pre-2023 foundation. [VERIFIED abstract]
  - GPU sampling-based joint-space MPC "can tightly integrate perception... learned cost functions from raw sensor data".
  - A ball-on-tray balancing demo tracks the ball with a RealSense at 30 Hz, median error 3.9 cm. [SUMMARY, search snippet; attribution to STORM is likely but unconfirmed]
  - No safety constraint on the ball.
  - [Source](https://arxiv.org/abs/2104.13542)
- **CyberRunner** ("Sample-Efficient Learning to Solve a Real-World Labyrinth Game Using Data-Augmented Model-Based RL"). Thomas Bi, Raffaello D'Andrea. arXiv 2312.09906, 2023-12-15. [VERIFIED abstract]
  - A camera-observed rolling ball on a tilting board, controlled by model-based RL (learned world model), 5 h of real data.
  - Falling into holes is a failure, but there is no explicit safety filter, no uncertainty-aware perception, and no manipulator.
  - [Source](https://arxiv.org/abs/2312.09906)
- **PRISM: Projection-Integrated Sampling-Based MPC with Bayesian Cost Tuning for Bimanual Manipulation.** Alinjar Dan, Iryna Hurova, Karl Kruusamäe, Arun Kumar Singh. arXiv 2608.25666, 2026-08-26. [VERIFIED abstract]
  - Tasks include "obstacle-constrained ball transport" and "tray transport" (PerAct² variants). A GPU physics simulator serves as an online world model; dual UR5e sim-to-real.
  - Gap: ball perception and safety constraints on the ball are not described in the abstract.
  - [Source](https://arxiv.org/abs/2608.25666)
- **Adaptive Contact-Implicit MPC with Online Residual Learning.** Wei-Cheng Huang, Alp Aydinoglu, Wanxin Jin, Michael Posa. arXiv 2310.09893, 2023-10-15. [VERIFIED abstract]
  - Rolls unknown objects on a table with residual-learned contact MPC on hardware. A vision-based ball estimate (PointGrey + Hough) is reported in a snippet. [SUMMARY for the vision detail]
  - Not tray transport, no safety constraint.
  - [Source](https://arxiv.org/abs/2310.09893)

### Inferences
- The waiter/slosh literature encodes payload safety as hard model constraints: friction cone, contact-wrench cone or ZMP-equivalent, and pendulum tilt or free-surface bounds. These live inside MPC/QP/OCP. This is structurally the same "safety in the QP constraints" pattern as an FL-MPC with a cup-tilt row.
- Uncertainty, when treated, is parametric: friction (Heins 2023), inertia/CoM (Heins 2024), friction learned from sound (Mao 2025), and data coverage (Jawale 2024). It is never the perceived payload state.
- The rolling-ball-on-carried-tray case with online perception plus a safety guarantee appears to be an open niche. The nearest items (STORM 2021, CyberRunner 2023) use vision for a ball but have no safety layer.
- Slosh state is almost always predicted open-loop from a pendulum model, not sensed (Hynninen 2026 states this explicitly). Spill safety therefore inherits model error with no perception-based correction.

### Gaps
- Fan et al. 2604.06932: how the objects are perceived (HTML 404). PRISM 2608.25666: how the ball is perceived and whether there are ball constraints. Abderezaei 2408.00215: whether fill level is sensed.
- Selvaggio TCST 2023, Subburaman RA-L 2023 and Muchacho IROS 2023 are not on arXiv (or were not found there). Their details are SUMMARY only.
- No Toussaint- or Kemp-group 2023-2026 waiter-type paper was found (arXiv abstract search for "waiter" with Toussaint returned nothing). No Tedrake, Bohg or Fei-Fei paper on payload-spill safety was found.
- No paper using vision-based slosh estimation in closed loop with a safety constraint was found. Biagiotti-group PEN6D work reportedly validates slosh models against camera surface measurements (search snippet only), but offline.

## Q2. Learned world models / latent safety values for spill, slide and topple hazards: who does it beyond the known set, and does anyone model perception uncertainty of the payload?

### Takeaway
Beyond the known Bajcsy/Bansal line (LatentSafe, UNISafe, AnySafe, LatentCBF, Kim et al., Cao & Bansal), learned safety for payload hazards is thin. It consists of:
- a WM-based safety critic for real-world RL (FARL);
- learned spill classifiers (Abderezaei 2024);
- ensemble value functions inside MPC for the waiter task (Jawale 2024);
- VLM/LLM-inferred constraints feeding a CBF or HJ value (Brunke 2024; Santos 2024);
- VLM code monitors (Code-as-Monitor 2024).

None models perception uncertainty of the payload state explicitly. The only uncertainty modelled is world-model epistemic uncertainty (UNISafe, known) or conformal scores on WM uncertainty (Ward 2026, a bimanual failure detector).

### Cited Findings
- **Failure-Aware RL (FARL): Reliable Offline-to-Online RL with Self-Recovery for Real-World Manipulation.** Huanyu Li, Kun Lei, Sheng Zang, Kaizhe Hu, Yongyuan Liang, Bo An, Xiaoli Li, Huazhe Xu. arXiv 2601.07821, 2026-01-12. [VERIFIED full text]
  - Mechanism: a world-model-based safety critic, a latent dynamics model with a constraint head, predicts H-step discounted constraint-violation probability. Above a threshold ε_safe, control switches to an offline-trained recovery policy.
  - Benchmark: the abstract motivates with "spilling water or breaking fragile glass", but FailureBench actually contains boundary push, bounded soccer (a rolling ball), fragile push-wall and obstructed push (vase). There is no liquid-spill task.
  - Perception: RealSense D435 with YOLOv8 plus colour filtering for state; perception uncertainty is not modelled.
  - Hardware and results: Franka, single arm; IR failures −73.1%, performance +11.3%.
  - [Source](https://arxiv.org/html/2601.07821); [abstract](https://arxiv.org/abs/2601.07821)
- **Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards.** Lukas Brunke, Yanni Zhang, Ralf Römer, Jack Naimer, Nikola Staykov, Siqi Zhou, Angela P. Schoellig. arXiv 2410.15185, 2024-10-19 (RA-L 2025 per lab PDF, SUMMARY). [VERIFIED full text]
  - Perception: RGB-D (Femto Bolt) frames plus visual-inertial SLAM; segmentation; CLIP-embedded object point clouds fused into an open-vocabulary object-level map.
  - Constraints: an LLM infers semantic constraints: spatial "(laptop, above)", behavioural "caution", and pose "constrained rotation" for "a cup of water". These become CBF constraints, including an EE-orientation-error bound to keep the cup upright.
  - Hardware and results: real Franka FR3 with teleop and diffusion policies; the rotation constraint reduces rotations by 75.39%.
  - Assumptions: static scenes; manipulated-object and robot geometry known.
  - Does NOT: model perception or map uncertainty in the CBF, or use any slosh dynamics (spill is proxied by kinematic orientation and keep-out constraints).
  - [Source](https://arxiv.org/abs/2410.15185)
- **Updating Robot Safety Representations Online from Natural Language Feedback.** Leonardo Santos, Zirui Li, Lasse Peters, Somil Bansal, Andrea Bajcsy. arXiv 2409.14580, 2024-09-22. [VERIFIED abstract]
  - A VLM interprets language plus images to update constraints, including context-dependent ones like "spills" and fragile objects. A Hamilton-Jacobi reachability safety controller is warm-start-updated online.
  - The abstract does not describe payload-dynamics modelling or perception-uncertainty modelling.
  - [Source](https://arxiv.org/abs/2409.14580)
- **Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection.** Enshen Zhou, Qi Su, Cheng Chi, Zhizheng Zhang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, He Wang. arXiv 2412.04455, 2024-12-05. [VERIFIED abstract]
  - VLM-generated code checks spatio-temporal constraints over "constraint elements" (compact geometric abstractions of tracked parts), for both reactive and proactive failure detection.
  - Results: +28.7% success under disturbances.
  - Does NOT: give a dynamics-aware safety value or calibrated perception uncertainty. Whether it includes spill examples was not verified.
  - [Source](https://arxiv.org/abs/2412.04455)
- **Clutter-Aware Spill-Free Liquid Transport via Learned Dynamics** (arXiv 2408.00215): a learned transformer spill classifier on trajectories; see Q1. [VERIFIED abstract] [Source](https://arxiv.org/abs/2408.00215)
- **CV-MPC** (arXiv 2412.00086): an ensemble value function as a learned, risk-averse slip "value" inside an MPC; explicitly perception-free; see Q1. [VERIFIED full text] [Source](https://arxiv.org/html/2412.00086)
- **Robotic Multiphase Interaction: Manipulating Coupled Liquid and Solid Dynamics with a World Model.** Yixuan Feng, Peng Wang. arXiv 2609.21448, 2026-09-18. [VERIFIED abstract]
  - Setting: a water-filled sponge in an SPH porous-flow simulation. An action-conditioned WM predicts the retained-water state (>60% lower error) and selects among diffusion/flow policy proposals.
  - Spill is cited as a safety risk, but there is no safety filter or reachability, and it is simulation only.
  - [Source](https://arxiv.org/abs/2609.21448)
- **OopsieVerse: A Safety Benchmark with Damage-Aware Simulation for Robot Manipulation.** Arnav Balaji, Arpit Bahety, Sriniket Ambatipudi, Daniel Lam, Junhong Xu, Roberto Martín-Martín. arXiv 2606.31993, 2026-06-30. [VERIFIED abstract]
  - DamageSim turns contact forces, temperature and "liquid interactions" into mechanical, thermal and fluid damage signals in OmniGibson and RoboCasa.
  - Used for damage-conditioned imitation learning and RL. A benchmark and training-signal source, not a safety filter.
  - [Source](https://arxiv.org/abs/2606.31993)
- **CrossSafe: Towards Cross-Embodiment Latent Safety Filters.** Ihab Tabbara, Yuxuan Yang, Hussein Sibai. arXiv 2609.28984, 2026-09-24. [VERIFIED abstract]
  - HJ reachability in a morphology-aware latent space, shared across five bimanual embodiments and five tasks.
  - Constraints are whole-body collision avoidance only; no payload hazards.
  - [Source](https://arxiv.org/abs/2609.28984)
- **Foundational World Models Accurately Detect Bimanual Manipulator Failures.** Isaac R. Ward, Michelle Ho, Houjun Liu, Aaron Feldman, Joseph Vincent, Liam Kruse, Sean Cheong, Duncan Eddy et al. arXiv 2603.06987, 2026-03-07. [VERIFIED abstract]
  - A probabilistic WM in the Cosmos-tokenizer latent space; its uncertainty serves as a conformal non-conformity score for runtime failure monitoring.
  - Data: Push-T and a new Bimanual Cable Manipulation dataset.
  - Detection only, with no avoidance or safety filter; cable task, not a payload.
  - [Source](https://arxiv.org/abs/2603.06987)
- **V-MORALS: Visual Morse Graph-Aided Estimation of Regions of Attraction in a Learned Latent Space.** Faiz Aladin, Ashwin Balasubramanian, Lars Lindemann, Daniel Seita. arXiv 2602.23524, 2026-02-26. [VERIFIED abstract]
  - Estimates regions of attraction from image trajectories in a learned latent space.
  - Generic systems, not payload-specific.
  - [Source](https://arxiv.org/abs/2602.23524)
- **Known set, cross-reference only** (confirmed to exist via the arXiv API this session):
  - LatentSafe 2502.00935. Its abstract cites a Franka "spilling the contents of a bag or toppling cluttered objects". [SUMMARY snippet] [Source](https://arxiv.org/abs/2502.00935)
  - LatentCBF = "How to Train Your Latent Control Barrier Function: Smooth Safety Filtering Under Hard-to-Model Constraints", Nakamura, Bishop, Man, Johnson, Manchester, Bajcsy, arXiv 2511.18606. [Source](https://arxiv.org/abs/2511.18606)
  - Kim, Nakamura, Bajcsy, "How Well Do Latent World Models Understand Partially Observable Safety Constraints?", arXiv 2510.06492. [Source](https://arxiv.org/abs/2510.06492)
  - Cao & Bansal, "When World Models Lie: Adaptive Safety Analysis Under Wrong Imaginations", arXiv 2609.34300. [Source](https://arxiv.org/abs/2609.34300)

### Inferences
- The Bajcsy/Bansal latent-safety line remains the only family doing reachability-style safety values on payload hazards (spill, topple) from pixels. Outside it, learned payload safety is either a critic or classifier without reachability semantics (FARL, Abderezaei) or a value ensemble used as a cost (CV-MPC).
- "Perception uncertainty of the payload" is consistently unmodelled. The closest proxies are WM epistemic uncertainty with conformal calibration (UNISafe, known; Ward 2026). These capture model or OOD uncertainty, not "where is the ball / how full is the cup" state uncertainty.
- Semantic or VLM pipelines (Brunke, Santos, Code-as-Monitor) convert perception into constraints but treat perception outputs as exact. Brunke's spill handling is purely kinematic (orientation bound plus keep-out region), with no slosh or payload dynamics.

### Gaps
- Not verified whether Code-as-Monitor contains spill or topple examples.
- No 2023-2026 paper was found that trains a latent safety value specifically for slide-off or roll-off of a free object on a carried tray, or that conditions a safety value on calibrated payload-state uncertainty.

## Q3. Bimanual carrying / co-manipulation safety: who handles payload dynamics, not just arm-arm collision?

### Takeaway
Bimanual payload-dynamics work is sparse and mostly unsafe-by-construction:
- SteadyTray: humanoid two-hand tray, RL, no safety layer.
- DART: dual-arm tray object regulation; MuJoCo only; no friction or topple constraints.
- PRISM: dual-UR5e tray and ball transport with a physics-sim world model.
- Paro et al.: bimanual tray transport handled as a rigid closed-chain constraint only.

Johansson et al. 2025 is the only bimanual safety filter found for a payload hazard (grasp slip / drop under uncertain mass and friction). It is simulation-only and has no perception. Bimanual latent safety (CrossSafe) and bimanual WM monitors (Ward) address collision or failure detection, not payload dynamics.

### Cited Findings
- **SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual RL.** Anlun Huang, Zhenyu Wu, Soofiyan Atar, Yuheng Zhi, Michael Yip. arXiv 2603.10306, 2026-03-11. [VERIFIED full text]
  - Bimanual: tray held in both hands ("bimanual loco-manipulation"; left- and right-EE alignment rewards).
  - Perception: the student policy observes object position and quaternion in the camera frame with a 32-step history. The teacher additionally gets privileged tray pose and object velocities.
  - Payloads: coffee cup, water-filled wine glass, surgical tools, sealed food container, on a Unitree G1 zero-shot. Failure = tilt, drop or topple.
  - Results: 96.9% velocity-tracking success and 74.5% push robustness in simulation.
  - Does NOT: use a CBF, safety value or perception-uncertainty model (robustness only via domain randomization and observation delay).
  - [Source](https://arxiv.org/html/2603.10306)
- **DART: Learning-Enhanced Model Predictive Control for Dual-Arm Non-Prehensile Manipulation.** Autrio Das, Shreya Bollimuntha, Madala Venkata Renu Jeevesh, Keshab Patra, Tashmoy Ghosh, Nagamanikandan Govindan, Arun Kumar Singh, K Madhava Krishna. arXiv 2604.17833, 2026-04-20. [VERIFIED full text]
  - Task: two xArm7s regulate an object's pose on a tray by tilting it. The MPC compares a physics model (Stribeck friction), online RLS regression, and an RL-learned dynamics model.
  - Simulation only (MuJoCo) with ground-truth object state.
  - Constraints are only velocity, input and rate bounds: no friction-cone, stay-on-tray or topple constraint. It claims to be the "first framework for non-prehensile dual-arm manipulation of objects on a tray".
  - [Source](https://arxiv.org/html/2604.17833)
- **PRISM** (arXiv 2608.25666, dual UR5e): bimanual tray transport and obstacle-constrained ball transport with a physics simulator as the online world model; see Q1. [VERIFIED abstract] [Source](https://arxiv.org/abs/2608.25666)
- **Fast Coordinated Bimanual Motion Planning With Hard Constraints.** Borna Paro, Luka Petrović, Ivan Marković. arXiv 2608.20946, 2026-08-21. [VERIFIED abstract]
  - Enforces a fixed relative EE transform (closed chain) continuously with a leader-follower IK parameterization, 19.4× faster planning; real bimanual Kinova Gen3 tray transport.
  - Does NOT: model payload dynamics (slide or spill) or perception.
  - [Source](https://arxiv.org/abs/2608.20946)
- **Safety filtering of robotic manipulation under environment uncertainty: a computational approach.** Anna Johansson, Daniel Lindmark, Viktor Wiberg, Martin Servin. arXiv 2509.12674, 2025-09-16. [VERIFIED full text]
  - Task and hazards: two ViperX 300 S arms hand over a box; grasp slip, drop and actuator overload.
  - Mechanism: a physics-based safety filter (AGX Dynamics) rolls out with nominal parameters, then sparsely re-evaluates critical transitions over uncertain mass and friction. Safety is scored with "generalized factors of safety" (contact FOS from Coulomb friction, motor FOS from torque limits). Safe probing actions shrink the parameter distributions.
  - Perception: none; parameters are given as truncated Gaussians.
  - Simulation only; the authors note that reliance on a well-calibrated simulator is a limitation.
  - [Source](https://arxiv.org/html/2509.12674v1)
- **Multi-Quadruped Cooperative Object Transport: Learning Decentralized Pinch-Lift-Move.** Bikram Pandit, Aayam Kumar Shrestha, Alan Fern. arXiv 2509.14342, 2025-09-17. [VERIFIED abstract]
  - Decentralized cooperative transport through contact forces only, scaling to 2-10 robots.
  - No spill or topple safety layer.
  - [Source](https://arxiv.org/abs/2509.14342)
- **CrossSafe** (2609.28984) and **Ward et al.** (2603.06987): bimanual latent HJ (collision only) and a bimanual WM failure monitor (detection only); see Q2. [VERIFIED abstracts] [Source](https://arxiv.org/abs/2609.28984); [Source](https://arxiv.org/abs/2603.06987)
- **Human-robot collaborative transport.**
  - Yang & Mavrogiannis, "Implicit Communication in Human-Robot Collaborative Transport", arXiv 2502.03346, 2025-02-05. [Source](https://arxiv.org/abs/2502.03346)
  - Yang & Mavrogiannis, "Learning from Humans for Proactive Assistance in Human-Robot Collaborative Transport", arXiv 2609.25351, 2026-09-21. [Source](https://arxiv.org/abs/2609.25351)
  - Both concern coordinating with a human on a large or heavy object; the abstract excerpts show no payload spill or slide modelling. [VERIFIED abstract excerpt]
- **Older foundation, one line.** Agravante et al., "Collaborative Human-Humanoid Carrying Using Vision and Haptic Sensing" (ICRA 2014). Visual servoing plus impedance control keeps a ball from falling off a jointly carried board. [SUMMARY] [Source](https://hal-lirmm.ccsd.cnrs.fr/lirmm-00950727)
- **Known, cross-reference only:** SafeBimanual 2508.18268 (caller already has it).

### Inferences
- No bimanual paper combines all of: perceived payload state, a payload-dynamics model, and a formal safety constraint or value. SteadyTray has perception but no safety. Johansson has safety but no perception. DART and PRISM have payload dynamics but no safety guarantees.
- Bimanual safety work splits cleanly: arm-arm or whole-body collision (CrossSafe, SafeBimanual, Paro) versus payload hazards (Johansson, SteadyTray). The combination, a centralized cooperative bimanual controller with payload (spill/roll) constraints, looks unoccupied in 2023-2026.

### Gaps
- Could not verify whether PRISM's tray and ball tasks include explicit "stay on tray" constraints or perception.
- No 2023-2026 bimanual slosh or liquid-transport paper on arXiv was found. Dual-arm slosh work found is pre-2023 (J. Vibration and Control 2021), and the Medico et al. AIM 2026 paper is single-arm.

## Q4. Point-cloud or keypoint perception uncertainty combined with reachability or safety values in manipulation: does anything exist?

### Takeaway
Only fragments exist:
- calibrated pose-uncertainty sets from keypoints (Yang & Pavone 2023), with no downstream safety controller;
- belief-space CVaR-constrained MPPI under pose uncertainty for contact-rich stowing (Enwerem et al. 2026), the closest "perception uncertainty + safety constraint in manipulation";
- conformal calibration of contact-aware motion uncertainty (Marques et al. 2026);
- point-cloud semantic maps feeding CBFs without uncertainty (Brunke 2024).

Nothing was found that combines point-cloud or keypoint payload-perception uncertainty with an HJ or CBF safety value for spill, slide or topple.

### Cited Findings
- **Risk-Constrained Belief-Space Optimization for Safe Control under Latent Uncertainty.** Clinton Enwerem, John S. Baras, Calin Belta. arXiv 2604.03868, 2026-04-04. [VERIFIED abstract]
  - Mechanism: belief-space MPPI with a CVaR objective plus a CVaR constraint on a trajectory safety margin, under an unknown time-invariant parameter (e.g., unobserved geometry or pose). Provides a probabilistic safety guarantee.
  - Experiments: MuJoCo vision-guided dexterous stowing with "pose uncertainty exceeding prescribed lateral clearance"; 82% success.
  - Insertion/stowing, not payload transport.
  - [Source](https://arxiv.org/abs/2604.03868)
- **Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation.** Heng Yang, Marco Pavone. arXiv 2303.12246, 2023-03-22. [VERIFIED abstract]
  - Conformal keypoint prediction sets are propagated to a Pose UnceRtainty SEt (PURSE) with guaranteed coverage, plus worst-case error bounds via SDP. LineMOD-Occlusion.
  - Provides the perception-side uncertainty set only; no safety controller.
  - [Source](https://arxiv.org/abs/2303.12246)
- **Particle-Based Conformal Prediction for Contact-Aware Uncertainty Calibration in Stratified Configuration Spaces (CaPTURe).** Luís Marques, Kristian Popov, Dmitry Berenson. arXiv 2608.09166, 2026-08-10. [VERIFIED abstract]
  - Conformal, geometry-aware prediction regions for future configurations under contact; examples include navigating clutter and part insertion.
  - Motion-model uncertainty, not perception.
  - [Source](https://arxiv.org/abs/2608.09166)
- **Runtime Monitoring of Perception-Based Autonomous Systems via Embedding Temporal Logic.** Parv Kapoor, Abigail Hammer, Ashish Kapoor, Karen Leung, Eunsuk Kang. arXiv 2605.12651, 2026-05-12. [VERIFIED abstract]
  - Temporal-logic predicates over learned embeddings with conformal calibration, evaluated in manipulation environments.
  - Monitoring only, not control.
  - [Source](https://arxiv.org/abs/2605.12651)
- **ARMOUR / "Can't Touch This: Real-Time, Safe Motion Planning and Control for Manipulators Under Uncertainty."** Jonathan Michaux, Patrick Holmes, Bohao Zhang, Che Chen, Baiyue Wang, Shrey Sahgal, Tiancheng Zhang, Sidhartha Dey et al. arXiv 2301.13308, 2023-01-30. [VERIFIED abstract]
  - Reachability-based, provably safe receding-horizon planning plus a robust tracking controller under uncertain mass and inertia of carried objects (payload-parametric uncertainty). The safety target is collision avoidance and joint limits.
  - No perception uncertainty and no payload-motion hazard.
  - [Source](https://arxiv.org/abs/2301.13308)
- **Brunke et al. 2410.15185** (Q2): a point-cloud object-level semantic map feeds CBFs; perception and map uncertainty are not modelled; static scenes. [VERIFIED full text] [Source](https://arxiv.org/abs/2410.15185)
- **Johansson et al. 2509.12674** (Q3): uncertainty-aware (mass, friction) physics safety filter with probing; no perception. [VERIFIED full text] [Source](https://arxiv.org/html/2509.12674v1)
- **Perception-driven CBFs from 3D Gaussian splats (navigation, not manipulation payloads), for analogy only.**
  - SAFER-Splat: Timothy Chen, Aiden Swann, Javier Yu, Ola Shorinwa, Riku Murai, Monroe Kennedy, Mac Schwager. arXiv 2409.09868, 2024-09-15. [VERIFIED title/authors] [Source](https://arxiv.org/abs/2409.09868)
  - Tscholl, Nakka, Gunter. arXiv 2509.14421, 2025-09-17: 3DGS collision-cone CBF. [VERIFIED title/authors] [Source](https://arxiv.org/abs/2509.14421)
  - SplatCtrl: Siddarth Jain, Ho Jin Choi. arXiv 2607.08948, 2026-07-09: Gaussian scene representation plus reactive manipulator control. [VERIFIED title/authors; content not read] [Source](https://arxiv.org/abs/2607.08948)
- **Known, cross-reference only:**
  - UNISafe 2505.00779: WM epistemic uncertainty with a conformal threshold plus reachability in an augmented latent-uncertainty space. [Source](https://arxiv.org/abs/2505.00779)
  - Kim et al. 2510.06492: partial observability of safety constraints in latent WMs. [Source](https://arxiv.org/abs/2510.06492)

### Inferences
- The building blocks exist separately but have not been assembled for payload hazards:
  - calibrated perception sets (PURSE-style conformal keypoints);
  - risk- or belief-constrained MPC (Enwerem CVaR-MPPI);
  - robust payload constraints (Heins robust sticking constraints);
  - latent HJ with uncertainty (UNISafe).
- A method that propagates calibrated payload-state uncertainty (e.g., ball or cup pose from point clouds or keypoints, or fill level) into a robust friction-cone, tilt or HJ safety constraint would appear novel as of 2026-10-05, based on this search.

### Gaps
- Measurement-robust CBFs (Dean et al., CoRL 2020; Cosner et al. 2021) are the relevant pre-2023 foundations for perception-error-robust CBFs. They were not re-verified this session, so no arXiv IDs are given.
- Not checked: conformal or robust CBF manipulation papers indexed only in IEEE Xplore (non-arXiv). Coverage of RA-L/T-RO-only papers is limited to what web search surfaced.
- Could not confirm the content of SplatCtrl 2607.08948 (title and authors only).
