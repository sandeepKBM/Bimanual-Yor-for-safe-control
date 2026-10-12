# Literature check: robust spatio-temporal semantics, STL-HJ, Cao & Bansal, PRISM

Date: 2026-10-05. Web research only. Nothing was run, connected or committed.

Tags. **VERIFIED** = I read the abs page, or the raw full text (PDF run through pdftotext). **SUMMARY** = a search snippet, or an HTML page that the fetch tool summarised for me. Treat SUMMARY as a lead, not a fact. One HTML summary of PRISM said "no motion capture"; the raw PDF says OptiTrack. So summaries can be wrong.

arXiv was not rate-limited. The Springer page for Chen et al. blocked me, and Semantic Scholar returned HTTP 429.

## Short verdict

The three parts are each taken. The combination is not, as far as I can find. Section 5 has the detail.

## Q1. STL monitoring under perception uncertainty

**Conformal-prediction STL runtime verification.** Your memory is right.
- Lindemann, Qin, Deshmukh, Pappas, "Conformal Prediction for STL Runtime Verification", arXiv 2211.01539, first submitted 3 Nov 2022. VERIFIED (abs page). I believe it is ICCPS 2023; the abs page did not say, so that part is from memory.
- It predicts future states from the observed trajectory, builds conformal prediction regions, and bounds the chance that the future trajectory violates an STL formula. Tests: F-16 and a car. Inputs are trajectories, not point clouds. VERIFIED (abs page).
- Zhao, Hoxha, Fainekos, Deshmukh, Lindemann, "Robust Conformal Prediction for STL Runtime Verification under Distribution Shift", arXiv 2311.09482, 16 Nov 2023. Franka in Isaac simulation. VERIFIED (abs page).
- Zhao, Zhu, Hoxha, Fainekos, Deshmukh, Lindemann, "Distributionally Robust Predictive Runtime Verification under Spatio-Temporal Logic Specifications", arXiv 2504.02964, 3 Apr 2025. It uses STREL (spatio-temporal reach and escape logic) for multi-agent systems. The abstract says nothing of perception or three-valued logic. VERIFIED (abs page).
- Cairoli et al. have a "Conformal Quantitative Predictive Monitoring of STL ..." paper, arXiv 2211.02375. Title is cut off in my snippet and I did not check authors. SUMMARY.

**Perception-aware conformal STL monitors (2026). These are the closest to us on the perception side.**
- Hoxha, Schön, Okamoto, Lindemann, Fainekos, "Vision-Based Runtime Monitoring under Varying Specifications using Semantic Latent Representations", arXiv 2605.13923, 13 May 2026. Past-time STL. A network predicts the robustness of basic atoms from images. Conformal calibration gives a radius, and subtracting it gives a valid lower bound on the robustness of every formula in the fragment. SUMMARY (html). The summary also says: no three-valued verdicts, no delay or dropout, predicates are scalar functions (clearance distances, margins) with no spatial operators, data are simulated overhead images and Waymo bird's-eye renders, and the guarantee needs exchangeable episodes. Abstract fields are VERIFIED (abs page).
- Kapoor, Hammer, Kapoor, Leung, Kang, "Runtime Monitoring of Perception-Based Autonomous Systems via Embedding Temporal Logic", arXiv 2605.12651, 12 May 2026. Predicates are distances in a learned embedding space (CLIP, DINOv2, world models), with a conformal threshold. The abs page confirms the conformal calibration. VERIFIED (abs page). The html summary says: two-valued semantics, 2D images only, no point clouds, no HJ, no hardware, and tracker failure and delay are named as not addressed. SUMMARY (html).

**Interval robustness and three-valued semantics. This is the closest on the logic side.**
- Baird, Harapanahalli, Coogan, "Interval Signal Temporal Logic from Natural Inclusion Functions", arXiv 2309.10686, 19 Sep 2023 (v2 Dec 2023). Predicates are replaced by interval inclusion functions; min and max by their minimal inclusion versions. The result is an interval robustness plus a three-valued logical semantics. The interval over-approximates the robustness under every realisation of the uncertainty. Demos: offline monitoring of an uncertain hardware trace, and robust online control. VERIFIED (abs page); the three-valued detail is SUMMARY (search snippet).
- Finkbeiner, Fränzle, Kohn, Kröger, "A Truly Robust Signal Temporal Logic: Monitoring Safety Properties of Interacting Cyber-Physical Systems under Uncertain Observation", Algorithms 15(4):126, 2022. Claims determinate truth values where plain interval evaluation stays inconclusive, using affine arithmetic and SMT. No arXiv ID found. SUMMARY (snippet).
- Deshmukh, Donzé, Ghosh, Jin, Juniwal, Seshia, "Robust Online Monitoring of Signal Temporal Logic", arXiv 1506.08234; RV 2015 best paper; Formal Methods in System Design 51(1):5-30, 2017. Robust satisfaction intervals (RoSI) for partial traces: while the future is unseen, the robustness is an interval, and the verdict can stay undecided. SUMMARY (snippet). This is the standard source for "unknown until enough data".
- Dong et al., "Quantitative Predictive Monitoring and Control for Safe Human-Machine Interaction", arXiv 2412.13365, 17 Dec 2024. STL-U gives a robustness-degree interval over uncertain predictions. Case studies: diabetes and driving. VERIFIED (abs page).
- Krasowski, Dietrich, Gezer, Skjetne, Sørensen, Arcak, "pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis", arXiv 2511.00934, 2 Nov 2025 (v3 22 Jul 2026). Bounds on robustness from PAC-bounded reachable sets, propagated through the operators. Quadrotor and maritime. VERIFIED (abs page).
- Dietrich et al., "Staying on Spec: Real-Time Monitoring under Uncertainty with a Maritime Case Study", arXiv 2608.02811, 3 Aug 2026. Data-driven reachable sets for spec evaluation. The abs page says nothing about dropout or three-valued logic. VERIFIED (abs page).

**Probabilistic STL.**
- Sadigh and Kapoor, "Safe Control under Uncertainty with Probabilistic Signal Temporal Logic", RSS 2016, arXiv 1510.07313 (arXiv title "Safe Control under Uncertainty"). Predicates hold with a stated probability; the logic adapts as beliefs update. Quadrotors and cars. SUMMARY (snippet).
- Yan et al., "Online Monitoring and Risk Assessment of Non-Cooperative UAVs via STL-Aware Adaptive Fusion Kalman Filtering", arXiv 2607.26527, 29 Jul 2026. Kalman filters plus STL, with probabilistic reachable sets. Whether the set grows in measurement gaps is not stated in the abstract. VERIFIED (abs page) for title and date.

**Geometric predicates from point clouds or detections.**
- PerceMon: Balakrishnan, Deshmukh, Hoxha, Yamaguchi, Fainekos, arXiv 2108.08289, 17 Aug 2021. Timed Quality Temporal Logic with spatial operators, monitoring object detectors and trackers in CARLA and ROS. The abstract does not say how it handles uncertainty. VERIFIED (abs page).
- SpaTiaL has robustness for geometric predicates. NL2SpaTiaL and Differentiable SpaTiaL build on it. You already know these.
- I found nothing that gives a SpaTiaL predicate (distance, containment, direction) an interval or inclusion-function semantics over a tracker's uncertainty set. This is a negative search result, not proof. It was a handful of searches.
- Perception-error propagation exists on the vision side: conformal keypoint sets pushed through to pose sets (arXiv 2303.12246, CVPR 2023). SUMMARY (snippet). It is not tied to logic.

**With learned values or HJ.** I found none that combines perception uncertainty, STL and a learned HJ value. Section 2 lists the HJ-STL papers. None of them has a perception set.

## Q2. STL meets Hamilton-Jacobi reachability

**The paper.** Chen, Tam, Livingston, Pavone, "Signal Temporal Logic Meets Reachability: Connections and Applications", Workshop on the Algorithmic Foundations of Robotics (WAFR) 2018; Springer proceedings (Algorithmic Foundations of Robotics XIII), pp. 581-601. Your memory (Chen, ~2018) is right. I found no arXiv ID and could not open the Springer page. The venue and authors are SUMMARY (search snippet and a bibliography entry in So et al.). I did not read its content. Do not describe its technical content from me.

**The connection, as stated by a paper that cites it.** So, Sharpless, Herbert, Fan, "Value Functions for Temporal Logic: Optimal Policies and Safety Filters", arXiv 2605.01051, 1 May 2026. SUMMARY (html) for the following. It says HJ reachability values for reach, avoid and reach-avoid have the same objectives as the quantitative semantics of basic temporal-logic predicates: reach is F, avoid is G, reach-avoid is until (U). It cites Chen et al. 2018 for this, along with Donzé & Maler 2010 and Fainekos & Pappas 2009. It decomposes a nested formula into a graph of values, proves optimality of non-Markovian policies against the robustness score for nested U, G and G-U formulas, and shows the Q function works as a safety filter. The html summary reports a state-space treatment with no vision and no perception uncertainty. I did not check whether its experiments use learned Q functions. Stated limits: no general conjunction rule, safety guarantee needs bounded until, and some formulas (e.g. FG p1 and GF p2) fall outside the fragment. Abstract fields VERIFIED (abs page).

**Other STL/temporal-logic plus HJ work.**
- Hirsch, Sharpless, Lee, Herbert, "Extending and Unifying the Fundamental Tasks of Hamilton-Jacobi Reachability Analysis", arXiv 2608.18060, 18 Aug 2026. Generalised reach-avoid; abstract says it covers composite tasks "including ones from timed temporal logic". VERIFIED (abs page).
- Jiang, Arfvidsson, He, Chen, Johansson, "Guaranteed Completion of Complex Tasks via Temporal Logic Trees and Hamilton-Jacobi Reachability", arXiv 2404.08334, 12 Apr 2024. Grid HJ values for temporal logic trees. VERIFIED (abs page).
- Verhagen, Lindemann, Tumova, "Robust STL Control Synthesis under Maximal Disturbance Sets", arXiv 2404.05535, 8 Apr 2024. Disturbance-robust STL semantics with an HJ-based algorithm. Disturbance is on the dynamics, not on perception. VERIFIED (abs page).
- Choi, Pant, Nam, Hellmann, Nune, Hwang, "Signal Temporal Logic Evaluation and Synthesis Using Deep Reachability Analysis and Layered Control Architecture", arXiv 2602.23313, 26 Feb 2026. A neural (DeepReach-style) backward reachable tube for STL feasibility. The abstract does not use robustness as a margin and has no perception uncertainty. VERIFIED (abs page).
- Tabbara, Yang, Sibai, "Towards General Language-Conditioned Latent Safety Filters", arXiv 2608.00315, 31 Jul 2026. An HJ actor and critic conditioned on language-specified constraints. The abs page does not say STL is used. VERIFIED (abs page).

**Learned or latent follow-ups that use STL robustness as the HJ margin.** I did not find one. The latent-filter line (Nakamura et al. and descendants) learns l(z) as a classifier. Cao & Bansal state this in Section II (VERIFIED, full text). The nearest learned item, So et al., is in state space.

**Consequence for us.** "Robustness is the HJ target; G and F are avoid and reach" is established. Do not claim it. Cite Chen et al. and So et al.

## Q3. Cao & Bansal, arXiv 2609.34300 (read in full from the raw PDF)

Title: "When World Models Lie: Adaptive Safety Analysis Under Wrong Imaginations". John Cao and Somil Bansal (Stanford). Submitted 28 Sep 2026. All claims in this section are VERIFIED (full text) unless marked as my inference.

**How the set is built.**
- World model: z_t is drawn from encoder E(. | ẑ_t, o_t); ẑ_t comes from learned dynamics f(. | z_{t-1}, a_{t-1}) (Eq. 1). The encoder takes the predicted latent and the current observation.
- Score: s_t = e_t = ||z_t - ẑ_t||_2 (Eq. 9). It is the gap between the predicted latent and the encoder's inferred latent. The encoder output is the reference; the paper never questions it.
- Set: U_{t+1} = { z : ||z - ẑ_{t+1}||_2 <= min(q_{t+1}, q_max) } (Eq. 10). It is an L2 ball around the next predicted latent.
- Radius: q_{t+1} is a quantile of the stored errors, with the level moved by adaptive conformal inference: α_{t+1} = α_t + λ(α - err_t), err_t = 1[e_t > q_t] (Eq. 6).
- Cap: q_max comes from trajectory-level conformal prediction on a calibration set (the (1 - α_traj) quantile of each trajectory's errors, then the (1 - α_cal) quantile across trajectories).
- Start: q_1 = infinity, so the radius equals q_max at the start. The paper's words: "maximum pessimism is held at the start, when data is scarce."
- Use: V̄(ẑ_{t+1}) = min of the learned HJ value V over U_{t+1} (Eq. 8), solved by projected gradient descent. The filter then switches to the safety policy if V̄ is below a threshold δ (Eq. 4).
- The margin l is a learned classifier on failure-labelled observations, as in latent filters.

**Set over latents or states.** Latents only. The set is a ball of latent codes. The paper says it "does not imply that every latent state considered ... is physically realizable". There is no state or tracker set.

**Missing or late observations.**
- There is no mechanism for dropout, occlusion or noise. The word "occlusion" does not appear in the text.
- Delay appears only as a test probe: the world model is conditioned on o_{t-L}. L is 0, 2 or 4 in simulation, and 10 on hardware at 15 Hz (about 0.67 s). The method has no delay model. It works because stale input raises the prediction error.
- My inference from Algorithm 1: lines 2-4 need o_t to compute e_t. With no observation, there is no new error, so the buffer cannot update. Nothing in the method grows the radius with time since the last fix. Only q_max at the start, and ACI after an error is seen, widen the set.
- Related: they never inject sensor noise. The test shifts are a forceful throw, a time-step change (temporal abstraction), and staleness.

**Guarantees.**
- Proposition 1 is the standard ACI result of Gibbs & Candès. It holds with no exchangeability.
- Theorem 1: with probability >= 1 - α_cal over new trajectories, the fraction of steps where e_t exceeds the radius is <= α + α_traj + r_T. It assumes exchangeability between new and calibration trajectories.
- It bounds how often the latent prediction error exceeds the radius. It does not bound safety, and it does not say the set contains the true latent state. The paper says plainly: "reactive and does not provide formal safety guarantees".

**Experiments.**
1. Slippery Dubins car (images). 1500 nominal training trajectories. The slippery zone scales velocity by 1.5.
2. ManiSkill Franka: pick up a globe, place it on a shelf, avoid a vase. 3000 scripted trajectories (1286 failed, 1714 safe). Nominal policy is a diffusion policy. 100 episodes per method for natural errors; 400 per method for shifts (time step 0.05/0.1/0.2 s; staleness L = 0/2/4). q_max = 2.0, α = 0.2, λ = 0.05, δ = 0.15.
3. Hardware, real Franka Panda at 15 Hz. A dustpan throws a tomato into a bin; failure means the tomato lands outside. Dreamer world model trained on 500 teleoperated trajectories (350 success, 150 failure). Cameras: side, top-down and wrist, plus proprioception. q_max = 4.5, α = 0.2, λ = 0.05, δ = 0.2.
4. Hardware test design: **open-loop replays**, not live closed-loop control. 20 success and 20 failure replays per category (in-distribution, forceful throw, staleness L = 10). The paper reports 480 runs, which fits 4 methods x 120 (my arithmetic). The metric is failed trajectories rescued and successes preserved.
5. Baselines: unfiltered, LatentSafe (no adaptation), UNISafe, ACoFi.
6. Results as written: on hardware under staleness, the strongest baseline had "50% fewer rescues" than theirs. In simulation the method had the fewest failures under natural errors, and was safer than LatentSafe and ACoFi under shifts. UNISafe had the lowest failure rate under shifts but many incomplete episodes. I did not extract exact per-method counts from the figures.
7. Cost: 39.59 ms per step with their solver, against 17-18 ms for baselines, on an RTX 5090.
8. Neither perception noise nor a tracker appears anywhere. The cameras feed a Dreamer encoder.

**Every limitation they state.**
1. The method is reactive and gives no formal safety guarantee.
2. Solving Eq. 8 costs extra compute.
3. Theorem 1 assumes exchangeability, which safety filtering can break "if recovery actions are absent from the calibration set".
Also stated in the results: the filter raises incompletion, because the nominal policy was not trained to resume after an intervention. They do not mention encoder error.

**Exactly how our tracker set differs.**

| Point | Cao & Bansal | Ours |
|---|---|---|
| Set lives over | latent codes (L2 ball) | tracker state / geometric parameters |
| Source of radius | past gaps between predicted latent and encoder latent | tracker covariance and fit residual, calibrated |
| Trust in perception | encoder is the reference | the tracker is the object under test |
| No observation | no update; radius cannot grow by design | set grows with time since last fix |
| Verdict | none; V̄ below δ gives a filter switch | true / false / unknown |
| Guarantee | miscoverage rate of the prediction error | worst case over a set that is meant to contain the truth (to be shown) |
| Margin l | learned classifier | analytic robustness of fitted primitives |

The shared core: take the minimum of a learned HJ value over a calibrated set, and use conformal calibration to size it. That core is theirs.
Where we do not differ: an ACI-style radius is a fair option for ours as well. It is a cite, not an invention.
A trap: if our value takes the tracked ball state directly (a low-dimensional latent), then "minimum over the ellipsoid" is the usual measurement-robust barrier construction. See the earlier report's "Where the difference is thin" section.

## Q4. PRISM, arXiv 2608.25666 (read in full from the raw PDF)

Title: "PRISM: Projection-Integrated Sampling-Based MPC with Bayesian Cost Tuning for Bimanual Manipulation". Dan, Hurova, Kruusamäe, Singh. Submitted 26 Aug 2026. VERIFIED (full text) throughout.

- **The ball.** It is a free, unattached sphere. Two UR5e arms with Robotiq grippers press on it from opposite sides and lift it by contact force alone. A cost keeps both end-effectors at the same yz point and at a fixed offset l_ball. The figure caption says the arms "maintain the unattached ball". Task: lift it, then move it to a target while avoiding obstacles (a central obstacle in the figure). In the move phase, ball-manipulator collisions are dropped from the cost so the grippers can touch it.
- **How the ball is perceived.** In simulation, from the simulator state. On the real robot, OptiTrack motion capture. The mocap PC publishes object poses (obstacles, ball, tray) over ROS2, and the control laptop writes them into MuJoCo mocap bodies. So the ball pose is ground-truth-grade. There is no camera, depth or point cloud. There is no model of tracker noise, delay or dropout.
- **How obstacles are handled.** Known geometry in the MuJoCo world, updated by the same mocap feed. A "novel obstacle placement" test is a MuJoCo Franka comparison, 20 trials, PRISM 60% against 20% for a learned policy. The authors warn it is not a benchmark.
- **Constraints.** Hard: joint position, velocity, acceleration and jerk limits, enforced by a QP that projects each sampled trajectory (ADMM solver). Soft: collision is a weighted cost term, and cost weights come from offline Bayesian optimisation. Phase changes trigger when pose costs fall under thresholds.
- **Safety guarantees.** None for collision or spill. The only "guarantee" is kinematic feasibility of samples. Safety is not claimed in the abstract.
- **Real robot.** Dual UR5e, 10-16 Hz on a laptop GPU, sim-to-real shown qualitatively: the tray task (Fig. 1) and the ball task (Fig. 6) in sim and real. I found no real-robot success rates or trial counts. All numbers are simulation. Examples: Bayesian optimisation lifts success from 75% to 100% (ball), 60% to 95% (handover), 45% to 90% (tray), 45% to 80% (box).
- **Stated limits.** The cost structure and phase split are hand-designed. MuJoCo must be accurate; deformables, fluids and highly dynamic contacts may open a sim-to-real gap.
- **Relevance.** PRISM is a planner. It is not a perception-robust filter. It gives a citable example of squeezed-ball bimanual transport with a mocap ball. It is a weak baseline for our claim.

## 5. Is "robust three-valued spatio-temporal semantics over growing tracker sets, used as a learned HJ target" open?

**Mostly open, as a combination. Thin as a theory claim.**

Taken, with the owner:
1. Interval robustness and three-valued semantics over uncertain signals: Baird et al. 2309.10686; Finkbeiner et al. 2022; Deshmukh et al. RoSI 1506.08234. The set-growth idea in the sense of "unknown until data arrives" is RoSI.
2. Conformal bounds on STL robustness from perception: Hoxha et al. 2605.13923; Lindemann et al. 2211.01539 and follow-ups.
3. Robustness equals HJ reach/avoid values: Chen et al. 2018; So et al. 2605.01051; Hirsch et al. 2608.18060. HJ plus disturbance-robust STL: Verhagen et al. 2404.05535.
4. Min of a learned HJ value over a calibrated set: Cao & Bansal 2609.34300 (latent set, no dropout).
5. Geometric predicates with robustness: SpaTiaL.

Not found (my search, not exhaustive):
- SpaTiaL-style geometric predicates (fitted primitives) lifted to inclusion functions over a tracker's set.
- A set that grows with time since last fix and yields an explicit unknown verdict, inside a perception pipeline.
- That robust margin used as l(x) for a learned or latent HJ value, with a measured effect.
- Latent-filter papers I read learn l as a classifier. None uses analytic spatial robustness as l.

What would make a reviewer say "obvious":
- Each ingredient is cited above. Say clearly that the contribution is the composition and the evidence.
- Three-valued is near free from I-STL. Do not sell it as a main novelty.
- Claim the set-growth and the boundary measurement (false-safe rate against distance to the boundary under noise, delay and dropout), which the earlier report also found open.

## 6. What to cite

Core, by claim.
- Spatial predicates: Pek et al. 2023 (SpaTiaL); NL2SpaTiaL 2512.13670; Differentiable SpaTiaL 2604.02643.
- Interval and three-valued semantics: Baird, Harapanahalli, Coogan 2309.10686; Deshmukh et al. 1506.08234 (RoSI); Finkbeiner et al., Algorithms 15(4):126, 2022.
- Conformal STL monitoring: Lindemann, Qin, Deshmukh, Pappas 2211.01539; Zhao et al. 2311.09482; Hoxha et al. 2605.13923; Kapoor et al. 2605.12651 (embedding predicates, as contrast).
- STL and HJ: Chen, Tam, Livingston, Pavone, WAFR 2018; So et al. 2605.01051; Hirsch et al. 2608.18060; Verhagen et al. 2404.05535.
- Learned HJ over a set: Cao & Bansal 2609.34300 (as the closest operation); UNISafe 2505.00779; Kim et al. 2510.06492.
- Probabilistic STL, for completeness: Sadigh & Kapoor, 1510.07313.
- Optional context: PerceMon 2108.08289; pacSTL 2511.00934; PRISM 2608.25666 (only as a bimanual mocap-ball planner with no safety claim).

Cautions.
- Do not describe the technical content of Chen et al. 2018 beyond what So et al. say about it. I did not read it.
- Do not quote the numbers from the first HTML summary of Cao & Bansal ("70-80% success"). I did not confirm them in the figure text.
- Cao & Bansal's hardware test is open-loop replay. Say so if we contrast it with live closed-loop results.
- Search was a few dozen queries. A paper could exist that I missed. Do one more pass (Google Scholar, forward citations of 2309.10686 and of Cao & Bansal) before any "first" claim.
