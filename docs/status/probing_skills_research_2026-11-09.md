# Probing skills for hidden-contents carry: literature notes (2026-11-09)

Scope: web reading only (no hardware, no code run). Question set from the 11-09 brief: which probing /
interactive-perception skills estimate mass, COM, contents that move, slosh, and friction before or during a carry,
with joint torques as the main sensor, and which methods choose the probe.
Probing, belief shrinking and active learning below are mechanisms from the cited papers, not ours.
Access key: OPENED = I read the page/abstract/body text returned by the fetch tool. ABSTRACT = only the abstract/listing
was readable. SNIPPET = only a search-result summary; not opened (do not cite numbers from these without checking).
The fetch tool summarises pages with a small model, so numbers quoted below are as returned by it; recheck any number
before it enters a doc or a claim.

## 1. Table of papers

| # | Paper (authors, venue, year) | URL | Access | Measures | Sensors | Reported accuracy (source) | Applicability to us |
|---|---|---|---|---|---|---|---|
| 1 | Provably-Safe, Online System Identification. Zhang, Zhou, Vasudevan. arXiv 2504.21486, Apr 2025 (no venue listed) | https://arxiv.org/abs/2504.21486 | OPENED (first 100k of 138k chars) | EE payload mass, first moments (COM), inertia, as interval bounds | Joint pos/vel, joint torque; torque noise bounded at 2.5%, other-robot-parameter uncertainty 5% (paper's assumptions) | No numeric error metrics. Qualitative: intervals contained true values, tighter than random excitation. Kinova Gen3, five dumbbells 1.81-3.63 kg, initial mass interval [1.2, 5.2] kg, 5 repeats each | HIGH for (a),(b): joint-torque only, torque/joint limits and collision enforced over the parameter interval during the probe. Rigid payload only; single arm |
| 2 | A Learning-Based Approach for Estimating Inertial Properties of Unknown Objects from Encoder Discrepancies. Lao, Han, Ma, Chirikjian. arXiv 2104.09858, 2021 (venue not stated) | https://arxiv.org/abs/2104.09858 | OPENED | Mass and COM from steady-state poses; learned joint-torque model then weighted least squares | Encoders only (+ AprilTag camera for COM reference frame) | Mass MAE 3.72 g (NMAE 4.75%), COM MAE 8.8 mm (NMAE 11.94%), avg of 4 objects, 1000 repeats of 64-sample estimates, 4-DOF OpenMANIPULATOR-X (paper) | MEDIUM: shows static multi-pose torque -> mass+COM; small masses, 4-DOF, needs training on known objects; no sliding contents |
| 3 | Physically-Consistent Parameter Identification of Robots in Contact. arXiv 2409.09850, 2024 | https://arxiv.org/pdf/2409.09850 | SNIPPET | Whole-body inertial parameters (legged), LMI-constrained, joint torque only | Joint torque | not read | LOW: legged floating base; only relevant as a physical-consistency (LMI) constraint on the fit |
| 4 | Hand-designed excitation for low-cost arm (CRANE-X7). arXiv 2605.15949v2, 2026 | https://arxiv.org/pdf/2605.15949 | SNIPPET | Base-parameter identification with single/adjacent-joint primitives | Joint torque | not read | LOW-MEDIUM: simple excitation primitives are cheap to build; not opened |
| 5 | Learning Object Properties Using Robot Proprioception via Differentiable Robot-Object Interaction. Chen, Liu, Ma, Eastman, Rus, Randle, Ivanov, Matusik. arXiv 2410.03920 (v2 Mar 2025); MIT news says ICRA 2025 (not confirmed on the page) | https://arxiv.org/abs/2410.03920 | OPENED | Mass of fixed-joint object; mass of a sphere moving inside a closed container shaken by the gripper; elastic moduli | Joint encoders only (current-control mode); object initial position known | Fixed objects: 0.056 vs 0.050 kg, 0.022 vs 0.020, 0.020 vs 0.023, 0.008 vs 0.010 kg, one trial each. Sphere in container: 0.018 identified vs 0.012 kg measured. 2.5-2.7 s per ID on an M1 Max. Repeats/variance not reported | MEDIUM for (c): only published sliding-contents-in-closed-container proprioceptive result found, but one trial, 12 g sphere, single joint, OpenManipulator-X. Multi-joint, slosh, granular listed as future work by the authors |
| 6 | Interactive Learning of Physical Object Properties Through Robot Manipulation and Database of Object Measurements. Kruzliak, Hartvich, Patni, Rustler, Behrens, Abu-Dakka, Mikolajczyk, Kyrki, Hoffmann. arXiv 2404.07344, Apr 2024 | https://arxiv.org/abs/2404.07344 | OPENED (v1 html) | Category, material, elasticity, density, volume, mass (weighing) | RealSense D435, microphone, joint torque (weighing), gripper force/displacement | Weighing: mass +-6 g for payloads up to 1 kg; density sd 223 kg/m^3; squeeze elasticity sd 10 kPa. 17 objects x 5 runs; beat random selection on category entropy. No safety handling in paper | HIGH for question 3 (probe selection by expected information gain over a Bayesian network, stop when no action has positive EIG); LOW for the numbers (household objects < 1 kg) |
| 7 | Learning to Double Guess (U-GRAPH). Jin, Mo, Yuan. arXiv 2502.02663v2, Feb 2025 (no venue on page) | https://arxiv.org/abs/2502.02663 | OPENED | 3D COM of arbitrary graspable objects, two grasp orientations, BNN uncertainty guides second orientation | 6-axis F/T on UR5e wrist (WSG-50 gripper) | 12 real objects x 5 grasp configs; per-axis mean errors only in a table (my average of its U-GRAPH column ~7.3 mm, my calculation; abstract says 1.47 cm, unexplained mismatch). Mustard bottle 5-6 mm in-distribution, 12-16 mm out-of-distribution. ~150 h training data, 204 grasps | MEDIUM: active choice of 2nd orientation is the idea; needs F/T and a trained BNN, small objects, no moving contents. "Information gain" is learned via ActiveNet, not computed |
| 8 | Before the Tipping Point: Force-Guided Active Perception for Shape-Agnostic Estimation of 3D Centers of Mass. Hyland, Xiao, Onal. arXiv 2609.12894, Sep 2026 | https://arxiv.org/abs/2609.12894 | OPENED (abstract + html experiment section) | Mass, COM height, toppling angle from one sub-critical push-and-retract tip | 6-axis wrist F/T + AprilTag camera, known pivot edge; ABB IRB120 | Box: mass 0.658 true vs 0.641 est (2.6%), COM height 14.624 vs 14.594 cm (0.2%). Abstract: relative error below ~5.0% on 4 objects. Trial counts per object not stated. Cycle 5-32 s. Safety: push stops at F_safe = eta*F_max; higher eta (0.65) truncates and raises error | MEDIUM for (b) COM height of the crate if a force-limited tip about an edge is acceptable. Needs F/T (we have none), rigid objects; paper says moving contents not addressed |
| 9 | Predicting Center of Mass by Iterative Pushing for Object Transportation and Manipulation. Hyland, Xiao, Onal. IROS 2023 | https://par.nsf.gov/servlets/purl/10535080 | OPENED (first 2 pages via pdftotext) | Planar (x,y) COM line of action by pushing a payload on a surface | OptiTrack payload pose (can be replaced by F/T per authors) | Relative error 4.95% sim, 5.46% experiment; converges <30 s sim, ~20 s experiment (abstract) | LOW: small mobile pusher, planar, isotropic friction assumed |
| 10 | Industrial Dual-Arm Box Handling via Online Inertial Estimation and Convex Wrench Optimization (title as in the search listing; the abs page returned "Dual-Arm Box Manipulation ..."). Wong, Yang, Lee, Campolo. arXiv 2605.22021 (v2 Oct 2026); comments: submitted to Robotics and Autonomous Systems | https://arxiv.org/abs/2605.22021 | ABSTRACT (PDF too large for the fetch tool) | Box mass and COM online from measured contact wrenches, then friction-feasible SOCP wrench split | Per the search snippet, F/T sensors on dual Kinova arms (not confirmed in the opened abstract) | No numbers in the abstract | MEDIUM: closest bimanual setting (mass + COM during the lift); needs contact wrenches, which we lack; we would have to rebuild wrench from joint torques |
| 11 | Huang, Guo, Yuan. Understanding Dynamic Tactile Sensing for Liquid Property Estimation. RSS 2022 | https://www.roboticsproceedings.org/rss18/p072.html | ABSTRACT | Liquid type, volume, concentration from oscillation frequency/decay after a perturbation of a bottle | GelSight tactile sensor | 100% classification of 3 liquids; volume "high precision", no number in abstract | LOW: needs tactile sensing; useful as the physics (frequency and decay set height and viscosity) |
| 12 | Emergency Stopping for Liquid-manipulating Robots. Hynninen, Kyrki. arXiv 2604.16667, Apr 2026 (preprint) | https://arxiv.org/abs/2604.16667 | OPENED (html) | Not an estimator. Spherical-pendulum slosh model inside MPC; rod length from the first-mode natural frequency for a cylindrical container (e.g. 80 mm diameter, 100 mm fill gives 21 mm rod) | None for liquid; slosh angle from video only for evaluation | Real Franka, 5 deg slosh limit: exact rod length max 5.3 deg, no-slosh-model baseline 22.3 deg, +50% rod length 8.1 deg, -50% rod length 1.8 deg (paper). Sim: average max violation 0.29 deg | MEDIUM for (c): shows what slosh parameter the planner needs and its sensitivity to error. Open container, cylinder, single fill |
| 13 | On System Identification of Space Manipulator Systems Including Their Fuel Sloshing Effects. IEEE RA-L 2023, DOI 10.1109/LRA.2023.3252341 (authors not confirmed) and Rackl & Lampariello, ICMIC 2014 (pendulum slosh, joint pos/torque) | https://unpaywall.org/10.1109%2FLRA.2023.3252341 | SNIPPET | Slosh parameters (3D mass-spring-damper; pendulum equivalent) plus base mass from joint torques, space robots | Joint torque, joint position, GNC | not read | LOW-MEDIUM: precedent for identifying slosh parameters from joint torque; free-floating, locked-arm identification experiment |
| 14 | Interactive object recognition (Sinapov, Stoytchev group; IJRR 2011 preprint; Bergquist IROS 2009 workshop). | https://home.engineering.iastate.edu/alexs/papers/IJRR_2011/IJRR_2011_preprint.pdf | NOT ACCESSED (server returned "Request Rejected"); SNIPPET only | Object recognition from lift/shake/drop/crush/push proprioception + audio | Joint torque/proprioception, microphone | Snippet: >98% recognition with multiple behaviors, 50 household objects (unverified) | LOW: classification of known objects, not parameters |
| 15 | Efficiently Computable Safety Bounds for Gaussian Processes in Active Learning. Tebbe, Zimmer, Steland, Lange-Hegermann, Mies. AISTATS 2024, arXiv 2402.18260 | https://arxiv.org/abs/2402.18260 | ABSTRACT | Safe active learning: GP safety probability checked along a continuous exploration trajectory | n/a | No numbers in abstract; engine example | LOW-MEDIUM for question 3: generic safe-exploration machinery; no manipulator payload example found |
| 16 | Estimation and Exploitation of Objects' Inertial Parameters in Robotic Grasping and Manipulation: A Survey. Mavrakis, Stolkin. arXiv 1911.04397 (to appear RAS) | https://arxiv.org/abs/1911.04397 | ABSTRACT (PDF text not decodable) | Survey: visual, exploratory, fixed-object estimation | n/a | n/a | Use as the index of earlier methods (tilting, pushing, lifting on several sides); I could not read it |
| 17 | Towards Learning to Perceive and Reason About Liquids. Schenck, Fox. ISER 2016, arXiv 1608.00887; Detection and Tracking of Liquids arXiv 1606.06266 | https://arxiv.org/abs/1608.00887 | SNIPPET | Liquid detection/tracking from RGB during pouring | RGB | not read | LOW: open containers during pouring; our contents are hidden |
| 18 | Pouring by Feel (Piacenza, Lee, Isler, arXiv 2310.18473) | https://arxiv.org/abs/2310.18473 | SNIPPET | Poured weight from joint torque; authors say pouring by proprioception alone not shown before | Joint torque | not read | LOW: source-container weighing |

Not found in this search: a paper that estimates liquid mass or fill level from joint torques during a shake on a fixed-base arm;
a paper that estimates a sliding solid block's mass or position inside a crate during a bimanual lift; a paper on per-mug fill
in a tray; a published friction-of-contents probe. Treat these as gaps, not as proof of absence.

## 2. Skills grouped by what they measure (question 1 and 2)

- Total mass: weighing by joint torque (Kruzliak, +-6 g up to 1 kg; Lao, 3.7 g MAE small objects). Excitation-trajectory
  regressor identification also yields mass (Zhang, no numeric error given). Source for the +-6 g: paper 6.
- COM (3D): static multi-pose torque with a learned or analytic model (Lao, 8.8 mm MAE); regressor first moments (Zhang, interval,
  no number); F/T-based rotation of grasp (U-GRAPH, ~7 mm by my average); tip test (Hyland 2026, <5% on COM height, needs F/T);
  push line-of-action (Hyland 2023, planar, ~5%). Only Zhang and Lao avoid an F/T sensor.
- Inertia: Zhang regressor (10 parameters per link, as intervals). No accuracy numbers read.
- Contents that move: shake + differentiable simulation (Chen, one trial); oscillation frequency/decay (Huang, tactile);
  slosh pendulum parameters from geometry (Hynninen, Rackl/Lampariello snippet). All different sensors or settings from ours.
- Friction of contents: none found. Hyland 2026 assumes a single Coulomb coefficient for the object-surface contact only.
- Articulation (lids): not covered by what I opened.
- Which-probe selection: expected information gain over a Bayesian network (Kruzliak); learned ActiveNet over grasp orientations
  (U-GRAPH); GP safe active learning (Tebbe); excitation planned under constraints (Zhang).
- Time costs read: identification phase 7.5 s (4 trajectories x 3 s, 1.5 s planning each; Zhang); 2.5-2.7 s compute for mass by
  differentiable sim (Chen); 5-32 s per push-retract cycle (Hyland 2026); ~20 s pushing (Hyland 2023).
- Risk of the probe itself: Zhang bounds torque, joint limits, collisions and caps payload under 4 kg; Hyland 2026 bounds the push
  force; Kruzliak, Chen, Lao, U-GRAPH state no safety treatment in what I read.

## 3. Probe selection and safety during probing (question 3)

- Kruzliak (paper 6): choose the next action with highest expected information gain over a Bayesian network of properties; each
  action used at most once (five actions); stop when expected gain is not positive. Does not address safety.
- Zhang (paper 1): safety is in the trajectory optimisation: constraints are enforced for all parameters in the current interval,
  and the interval shrinks as data arrives. This is the closest match to a checker that bounds the probe before the load is known.
  The 4 kg payload cap and the 2.5%/5% noise and uncertainty bounds are the paper's ASSUMED values.
- Tebbe (paper 15): safety probability over a whole continuous exploration path under a GP; abstract only.
- Hyland 2026 (paper 8): force-fraction margin eta*F_max; shows the trade-off (lower eta gives better estimate, more risk).
- U-GRAPH (paper 7): no safety; selection by learned error prediction rather than formal information gain.

## 4. Mapping to our probe needs (question 4)

(a) Total mass up to the payload limit without stopping early. Candidate: Zhang-style interval identification, where the pre-check
    is replaced by torque/limit constraints evaluated over the current mass interval (start wide, shrink). The adaptive-step fix
    being built is the same idea at one degree of freedom (vertical lift). Zhang's 1.2-5.2 kg initial interval and 4 kg cap are
    for a Kinova, not NERO.
(b) Horizontal COM: static torque at two or more tray tilts or arm-pair force splits gives a moment balance. This derivation is
    mine (standard statics), not taken from a paper I opened; Lao uses a learned torque model for the same idea. Vertical COM
    height needs a tilt or tip (Hyland 2026 uses F/T) or the dynamic regressor first moments (Zhang).
(c) Sliding contents and slosh. Only Chen (shake, one trial, 12 g) addresses moving contents proprioceptively. For slosh,
    Hynninen needs a rod length from fill and geometry; per-mug hidden fill is not obtainable from those papers.
    A torque time series after a small tray tilt or acceleration step could show decay and frequency; no source for doing this with
    joint torques was found, so any use is ASSUMED/unvalidated.

## 5. Ranked shortlist (for us)

Bound labels follow AGENTS.md: MEASURED, MODEL, ASSUMED. "ASSUMED" items must be raised to the user before a claim rests on them.

1. Interval-based excitation identification, Zhang et al. 2504.21486.
   Build: end-effector regressor for the object+weld on each arm (or the closed chain); excitation trajectories with the NERO
   torque/limit/collision rows; interval update from torque residuals. Sensors: joint pos/vel (MODEL: sim exact), joint
   torque. Torque-noise bound: ASSUMED (paper uses 2.5%; NERO current-to-torque noise not MEASURED). Gives mass, first moments,
   inertia for rigid contents. Open: a 7.5 s four-trajectory phase is longer than poslift; closed-chain coupling of two arms;
   whether a locked sliding block breaks the rigid assumption (it will, once it slides).
2. Quasi-static multi-pose torque probe (extension of poslift; idea from Lao 2104.09858, statics from standard mechanics).
   Build: hold the tray at 2-3 tilts/yaw offsets, read steady torques, solve for mass and horizontal COM; watch torque drift
   with time or pose as a "contents move" flag. Sensors: joint torque (noise ASSUMED). Open: how far the tilt can go before
   a block slides or a mug slips (friction ASSUMED); no published accuracy for 7-DoF at 0.5-5 kg.
3. Probe-selection layer, Kruzliak et al. 2404.07344 (EIG over a Bayesian network), with Zhang-style constraint rows as the
   safety filter. Build: discrete action set (poslift, tilt, shake, regressor), noise model per action, EIG stop rule.
   Per-action noise values: ASSUMED until measured in sim; paper's +-6 g is for <=1 kg household payloads.
   Open: the guarantee type of an admission that rests on a sampled belief (sampled, not proof).
4. Shake-based proprioceptive ID of moving contents, Chen et al. 2410.03920.
   Build: differentiable or sampled-simulation fit of a sliding-mass model to shake joint trajectories (our MuJoCo model is the
   simulator, so sim-to-real mismatch is zero in sim by construction; that must be stated). Evidence base is one trial; treat as
   exploratory. Sensors: joint pos (current control). Open: identifiability of block mass vs. friction; shake amplitude vs.
   payload limit; the paper's own limits (known initial pose, local minima).
5. Slosh/contents-aware acceleration limit, Hynninen & Kyrki 2604.16667 (pendulum, rod length from natural frequency), with
   frequency read from a probe tap. Build: slosh model parameters from a vision fill estimate or a torque ring-down; the paper
   uses cylinder geometry and an open container. Rod-length error of +50% broke the 5 deg limit (8.1 deg real), so the
   parameter bound matters. Open: whether per-mug fill can be seen by the ZED/wrist cameras.
6. F/T-free tip test for COM height of the crate, after Hyland et al. 2609.12894.
   Build: force-limited tilt about a table edge using joint-torque-derived force. Requires replacing the wrist F/T with
   estimated end-effector force (ASSUMED accuracy). Risk: the crate can topple; the paper's eta margin trades error for safety.
   Open: not applicable to tray-of-mugs; moving contents are outside the paper's rigid model.

## 6. Open questions

- None of the opened papers reports results at our scale (0.5-5 kg per object, 3 kg per arm, two arms). Reported masses:
  1.8-3.6 kg (Zhang), < 1 kg (Kruzliak), < 0.7 kg (Hyland), grams (Lao, Chen).
- Which of our unmeasurable phenomena (COM height, slide, slosh, per-mug fill) need a measurement at all before the checker admits
  the carry, versus a bound that covers them? That is a decision for the user under the 11-08 rule (LLM names phenomena, skills
  supply numbers).
- NERO joint-torque noise and current-to-torque map are not known; every torque-based accuracy above depends on it.
- Abstract-only or snippet-only entries (rows 3, 4, 10, 11, 13-18) should be opened before use. Row 14's host rejected the fetch.
- Paper 7's abstract (1.47 cm) and table average (~7.3 mm, my calculation) disagree; check the paper before citing.
