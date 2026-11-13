# Literature notes (moved from the session memory, 2026-11-05)

> Status (2026-11-05): CURRENT reference. Moved verbatim from Claude memory notes (USER 11-05: memory should not hold papers).
> Notes marked HISTORY belong to dropped directions (latent HJ, pan demo). Positioning lines are the user's to decide.

## Lit map: VLM-written constraints 10-09

_Deep-research map (10-09) of LLM/VLM-written analytical constraints for manipulation: 14 works, where the evidence stops; LTL/STL + 2025-26 coverage thin_

Full report: research_notes/vlm_llm_analytic_constraints_lit_2026-10-09.md (workflow run wf_e42e0c70-90c, 108 agents).
The user asked for an unbiased map with no comparison to our work.

- **Coverage:** 14 works.
  - Six were verified 3-0: KUDA, GeoManip, OmniManip, ReKep, CoPa, MOKA.
  - Eight have low confidence (checked once): NL2SpaTiaL, SafeBimanual, UniDiffGrasp, SAFER, OWL-TAMP, Brunke et al.
    (semantically safe), VoxPoser, L2R.
- **Bimanual:**
  - ReKep: 3 tasks x 10; auto 12/30, human-annotated 18/30.
  - SafeBimanual: RoboTwin 8 sim tasks + real Galaxea R1. The LLM toggles 5 author-written safety costs per stage.
  - UniDiffGrasp: Baxter, grasp success 0.767.
  - SAFER: 1 hardware run.
  - No work has the model write closed-chain, load-sharing or timing constraints.
- **Safety:** only Brunke has formal guarantees (CBF-QP, 45 Hz, 0% violations, FR3). This holds relative to the LLM's
  constraint set (60% precision, 99% recall). CBF forms are author-designed. Set-based constraints are author-designed
  only (superquadrics, polygon).
- **Uncertainty:** no explicit uncertainty modelling found (friction, mass, perception bounds).
- **Trials:** real trials are 5-10 per task. There is no shared benchmark.
- **Gaps:** LTL/STL works were not examined (Lang2LTL, NL2TL, AutoTAMP, Safety Chip, PRoC3S …). 2025-26 coverage is
  limited by the search.

**How to apply:** cite as a coverage-limited map. Run a gap-fill round (LTL/STL, CBF, 2025-26 bimanual, uncertainty)
before relying on any absence claim. Related: [[direction-analytic-vlm-boed]], [[latent-hj-literature-2026]].

## Lit verdicts 10-12: VLM priors in safety filters

_Verified 10-12 related-work verdicts for the "Confidently Wrong" paper. The 5 novelty points are new or partial; must-cite PSS, Brunke, Phys2Real; full table + BibTeX in research_notes_

Process: 3 Haiku searches → a Sonnet verification on primary pages, mostly abstract level. Output: research_notes/lit_vlm_priors_safety_filters_2026-10-12.md (table a-g + BibTeX).

**Verdicts** ("new" = not found, not proven absent):
- **P1** (VLM physical prior inside a hard real-time filter, with the measured safety cost): the COMBINATION is new.
  - PHYSICIAN (NeurIPS'26 WiML poster): VLM friction/mass + a CBF layer, but non-real-time and uncalibrated.
  - Phys2Real (ICRA'26): centre-of-mass prior; notes the VLM "may be confidently wrong"; no filter.
  - Peng (Humanoids'25): overestimated VLM friction makes the policy aggressive.
- **P2** (widen-only rule): new AS A STATED RULE.
  - Close in spirit: PSS (arXiv 2609.34356, Kim/Fradi/Dai/Ostaszewski/Schmidhuber) does not shrink the calibrated radius from the VLM spread.
  - AlphaAdj (Chen & Chandra 2603.21142) caps the VLM's alpha by geometry.
- **P3** (semantic + state uncertainty in one real-time filter): PARTIAL. PSS does conformal VLM event prediction + a 50 Hz backup CBF (quadruped, sim).
- **P4** (margin jump → infeasible → brake → secondary hazard): new as a measured failure. General CBF infeasibility: Xiao ECC'23, Rousseas 2505.05502.
- **P5** (bimanual): the combination is new.
  - GF-VLA (engrXiv, dual-arm LLM-CBF) has no uncertainty.
  - SafeBimanual has no hard filter.
  - Johansson 2509.12674 has no VLM.

**Must cite/differentiate:** PSS, Brunke (RA-L'25), Phys2Real.

**Corrections from verification:**
- 2603.06921 = CN-CBF (not a pedestrian-deadlock paper).
- 2505.05502 = Rousseas feasibility analysis.
- Phys2Real's prior is centre of mass only.

The paper draft v2 (docs/archive/paper/draft_2026-10-12/main.pdf) now carries this; v1 is saved as main_v1.tex.

Related: [[lit-vlm-analytic-constraints-2026-10-09]], [[object-centric-state]].

## Borquez hybrid HJ verdict 10-21

_2026-10-21 full-text read of Borquez/Peng/Bansal 2609.17430 hybrid HJ (hLRF + hBRAT); low risk to T3/T4, must cite; Demo C still not built_

Read full text 2026-10-21 (arXiv v1, 15 Sep 2026; IJRR claim from earlier note not re-checked).
- Covers: hybrid HJI-VI over modes with controlled/forced transitions + resets; hLRF = least-restrictive filter
  (forward invariance of safe set only); hBRAT = reach-avoid tube, "safety and task completion" = reach goal G
  within fixed horizon T under bounded additive disturbance. Quadruped hardware, 4-D grid [70,70,40,5].
- Does NOT cover: partial observability / belief (full state via mocap); parameter sets that shrink online;
  counting nominal-recovery cycles; online recomputation. Their OWN future work lists "incremental or online
  reachability as new information arrives" and high-dim scaling (grid is exponential) — that is our setting.
- Strongest attack it enables: "an hBRAT with grasp modes already gives completion; T3 is redundant." Answer:
  hBRAT needs an offline global grid on full state with a known world; ours is 14+ DoF, set-valued belief
  changing online, certified per segment. Position it as the full-information, low-dim ideal we approximate.
- Status of Risk 3: Demo C (box turnover w/ joint limit, NERO 7-DoF, support-assisted hand transition) has no
  code; only assets_nero/nero_arm.xml + world_model/v2/nero/urdf_to_mjcf.py (10-09). Related: [[thesis-plan-rev2-2026-10-15]],
  [[lab-7dof-agilex-deferred]].

## ARMTD verdict 10-21

_ARMTD (Holmes..Vasudevan, RSS 2020, UMich) full text read 10-21: the "precomputed sets, cabinet" paper user recalled; Fetch arm (not UR5e), zonotope JRS (not HJB), braking fail-safe, 16/100 safe stops = our livelock gap_

ARMTD = Autonomous Reachability-based Manipulator Trajectory Design, RSS 2020, roboticsproceedings.org/rss16/p100.pdf,
code github.com/ramvasudevan/arm_planning. User remembered it as "HJB sets precomputed on ur5e picking from a cupboard".
- Fetch 6-DOF arm (hardware fig: low shelf -> high shelf of a cabinet). Zonotope joint reachable sets (JRS) of PARAMETERIZED
  trajectories computed OFFLINE; online each receding-horizon step (t_plan 0.5 s, t_f 1 s) intersect with obstacles ->
  conservative constraints -> optimize the trajectory parameters. Every plan ends in a fail-safe = BRAKING TO A STOP; if no
  solution in t_plan, keep executing the previous plan (incl. its brake).
- Static obstacles, known (conservative) obstacle estimates; no belief/parameter uncertainty; no liveness/completion claim.
- Random obstacles: ARMTD 84/100 goals, 0 crashes, 16/100 STOPPED SAFELY without a new plan; CHOMP 82/100, 18 crashes.
- Relevance: (1) it is the receding-horizon "plan next while executing, keep last certified plan" pattern we adopt -> cite as
  prior; (2) its fail-safe is stop-type and its 16% safe stops are exactly the stall our task-ready backup + T3 count target;
  (3) its offline JRS = a library of certified motion families (full-state, offline, no LOOK). Follow-ups ARMOUR/WAITR/SPARROWS
  not re-read. Related: [[borquez-hybrid-hj-verdict]], [[tamp-replan-2026-10-21]].

## ARMOUR+WAITR verdict 10-21

_ARMOUR (T-RO 2025) + WAITR (2023) full text read 10-21: set-based mass intervals + torque/contact-wrench constraints, braking fail-safe, single Kinova arm; WAITR future work = contact changes + force closure (our C3/Demo C); no ROAHM follow-up found_

ROAHM lab (Vasudevan, UMich), both Kinova Gen3 7-DoF single arm, receding horizon t_p 0.5 s, every plan ends BRAKING to a stop.
- ARMOUR arXiv 2301.13308 (T-RO 2025): interval uncertainty on link/payload inertial params (sim +-3% link mass; hw 2 lb dumbbell
  +-3%), robust tracking controller with uniform error bound, polynomial-zonotope RNEA -> input (torque) reachable sets ->
  torque/joint-limit/collision constraints for ALL params. Random: 92/100 goals, 0 crashes, 8 safe stops; Hard: straight-line
  HLP 0/7 goals (0 collisions), RRT* HLP 6/7.
- WAITR arXiv 2309.03111: tray rigidly held (force closure ASSUMED), unsecured object on it; constraints for all params in
  [Delta]: separation (f_z), linear slip (friction cone, mu_s), tip (CoP in circular contact radius). mu is MEASURED and fixed
  (0.36, 0.60), not a set. Sim 100: WAITR+GB-HLP 91 goals / 9 safe stops / 0 violations; ARMOUR 64/30/6 violations.
  FUTURE WORK (verbatim intent): handle contact-state changes (slip) and constraints guaranteeing force closure on objects.
- Overlap with us: "constraint over all params in an interval set, multilinear wrench, Kinova" is THEIRS -> C3 must be pitched as
  grasp-hold (force closure) + mode switch (1 vs 2 hands, set-down) + friction LOWER BOUND as a set, not "robust wrench
  constraints". Cup-on-tray S3 rows are WAITR-like -> cite. Not covered by them: task-ready backups, completion bound, belief
  shrink/LOOK, bimanual, mode switches. Repo list 10-21 shows no grasp/force-closure follow-up; related: OnlineSafeSysID
  (RSS 2025, = 2504.21486 in prior_art), crows (conformal reachable sets, ICRA 2025).
Related: [[armtd-verdict-2026-10-21]], [[borquez-hybrid-hj-verdict]].

## Karagulle WSTL verdict 10-22

_Ruya Karagulle (UMich, Ozay) safe preference learning via weighted STL: low overlap, cite, optional reuse as a preference score over checker-admitted candidates_

Read 10-22 (3 Sonnet agents, full PDFs). Dissertation "Integrating Formal Methods and Human Feedback for Safe and Personalized
Control Synthesis" (Scholar 2026) NOT found public (Deep Blue 403).
- WSTL origin = Mehdipour/Vasile/Belta L-CSS 2021 (arXiv 2010.00752), NOT Karagulle: positive weights never flip robustness sign.
- Karagulle papers: HSCC 2023 (early), RA-L 2024 (arXiv 2311.02099, pairwise prefs, 8-person pilot), HSCC 2024 (Bayesian active
  queries over a weight grid, 11 participants, 79% agreement), 4C IFAC 2025 (prefs + demos, PyTeLo MILP, hard r>0, n=4 pilot),
  arXiv 2511.08502 (ACC 2026; pruning + log-transform -> exact MILP; F1 test 81.7 vs 82.5 RS = no test gain), arXiv 2609.22464
  (CDC 2026, rank-capacity of WSTL). Also FORMATS 2022, a 2025 patent app (Toyota), unverified 2026 reward-machine paper.
- Her guarantee = semantics soundness on KNOWN deterministic trajectories; no uncertainty/belief sets/manipulation/planner/VLM.
  LLM->STL is only her future work (watch: overlaps VLM-written specs, not checkers/T1-T4).
- Verdict: overlap LOW. Cite Mehdipour 2021 + RA-L 2024 / HSCC 2024 / 4C as "preferences learned inside a hard spec envelope,
  known noise-free plant". Optional reuse in shared autonomy: preference/VLM score as soft cost over C1/C2/C3-admitted candidates
  (her soundness argument transfers: ranking can't override admission); active pairwise queries ~ C4. Her log-MILP trick does NOT
  help C3 (corner evaluation is simpler). Not built; don't claim.
Related: [[lit-vlm-analytic-constraints-2026-10-09]], [[armour-waitr-verdict-2026-10-21]].

## Lit: short-horizon cert 10-26

_Lit verdict 10-26 (Sonnet + Codex luna, independent, agree): short certified horizon + fail-safe stop + replan is standard (ARMTD, Althoff/Pek); our e_v*t disc not found; \"stop always admitted\" = passive safety; PDDLStream = static deterministic, SS-Replan defers motion to next replan_

Two independent passes (Claude Sonnet agent + Codex gpt-5.6-luna with web search), 2026-10-26; they agree on all core points.
- Short horizon + fail-safe + replan is STANDARD: ARMTD (Holmes/Kousik RSS 2020: 1.0 s plan, replan 0.5 s, braking fail-safe; obstacles STATIC per plan); RTD (Kousik IJRR 2020, static); Althoff & Magdici T-IV 2016, Magdici & Althoff ITSC 2016 (fail-safe to standstill), SPOT (Koschi & Althoff IV 2017, reachability under traffic rules, dropped on violation), Pereira & Althoff T-ASE 2018 (human arm), Pek & Althoff T-RO 2021 (5 s fail-safe, 0.25 s steps, pedestrian may not react; "invariably safe" terminal) [Codex-only numbers].
- Person models in lit: worst-case reachable sets (Althoff) or probabilistic/confidence-aware (Fisac RSS 2018, Bajcsy ICRA 2019: 2 s prediction, 50-300 ms replans, ~0.23 s/plan). Our constant-velocity + e_v*t disc NOT found. No special rule for a stationary person found (a gap).
- Safety at rest: "stop always admitted" = PASSIVE SAFETY (Fraichard ICRA 2007; Bouraine, Fraichard, Salhi AuRo 2012, also limited-FoV) -- say so in the paper; ARMTD/Pek are stricter (fail-safe itself collision-free).
- Freezing robot problem: Trautman & Krause IROS 2010; remedy in lit = interaction-aware prediction; no guaranteed-safe planner proves anti-freezing. RSS (Shalev-Shwartz 2017) = responsibility, not a planner.
- PDDLStream (ICAPS 2020): deterministic, initial state (no belief), stream facts "cannot be changed" -> static-scene certification; no execution/replanning. SS-Replan (ICRA 2020): determinize + replan after each observation, DEFERS motion streams not needed before the next replan (= our receding window). RH-TAMP (RAS 2021), reactive kinodynamic TAMP (IJRR 2022), SIPP (full space-time route, needs known obstacle trajectories), T-TAMPEST (AAAI 2025).
Reports: scratchpad codex_horizon.out / codex_pddl.out (session 0fdf97c2).
Related: [[armtd-verdict-2026-10-21]], [[planner-plans-safely-2026-10-26]].

## van Wijk recurrence BR-CBF 2610.17337

_van Wijk, Ong, Ames, Burdick arXiv 2610.17337 (12 Oct 2026) \"Filtering for Liveness: Recurrence via Backup Reachability CBFs\" — the prior art for \"backup ends where the task continues\"; abstract only read_

arXiv 2610.17337 [eess.SY, cs.RO], v1 12 Oct 2026, 6 pages, no venue listed. Safety filter for RECURRENCE (liveness): the system must
return to a given set within a max time gap; backup-reachability CBF + timer switches to a reaching controller when the nominal would miss
the deadline; allows temporary departures from the set. Demo: drone racing with a beacon-visibility recurrence constraint.
Per the user's 10-26 decision, this is why "backup ends where the task continues" alone is not new ([[scope-pan-demo-2026-10-26]]).
Full HTML read 2026-10-26 (via fetch summary): theorems guarantee RECURRENCE only (return to fixed, known set C0 within T_b; Thm 4 switched
controller), NOT task completion; nominal kept 'when possible' (QP objective, no theorem); task completion shown only in the 1 sim example
(planar quadrotor, 2 gates, 7 beacons, backups fired twice, no baselines/hardware). Exact dynamics, NO parameter uncertainty; one backup
controller in theory, several per-beacon via max/combinatorial CBF in the example (details omitted). Our gap: parameter-dependent task-ready
set, uncertainty + range updates, choice among certified recoveries, completion-or-refusal claim.

## AgentChord 2605.11951

_AgentChord (Xu et al., arXiv 2605.11951, RSS 2026) — LLM-agent task graph with pre-compiled recovery branches; main baseline; no physical guarantees; read 2026-10-26 (first 100k chars)_

"From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation", Sheng Xu ... Guiliang Liu,
arXiv 2605.11951 (12 May 2026), RSS 2026, CC BY-NC-SA. Composer (nominal task graph) / arranger (recovery branches) / conductor (execution);
all methods use GPT-5 via API. Motions = atomic skill library + optimization solvers (EmbodiChain); physical checks only collision/reachability
via solver constraints, NO force/torque/grasp-stability checks. Recovery filter = forward-progress rule (drop branches whose cost-to-go exceeds
the failure state's) — closest overlap with our task-ready idea, heuristic, cite it. Monitors detect failure AFTER it happens (gripper aperture
"most reliable"). No completeness/optimality claim; admits MLLMs miss rare failures. Hardware AgileX CobotMagic (two 6-DoF arms, fixed base);
tasks: dual-arm pour, rearrange, handover, towel fold. Code release not stated (project page shengxu.net/AgentChord); appendix not read.
USER framing 10-26: we build on AgentChord as the next line, but as safe kinodynamic planning + control, not trusting the LLM for completeness
([[scope-pan-demo-2026-10-26]]).
USER 10-27: the second pillar is TASK-PRESERVING RECOVERY (certified return to a parameter-dependent task-ready set, recovery chosen among
certified actions). AgentChord's analogue = heuristic forward-progress rule (no physics); van Wijk's = return to a fixed known set (no uncertainty).
11-04 code read (clone /data/local/ss5772/agentchord/src @74fe7dd, EmbodiChain): CobotMagic arms use OPWSolver (analytic 6-DoF IK); also
srs_solver (7-DoF, Warp, for DexForce W1), pinocchio/pink/pytorch/differential solvers. Planner = MotionGenerator: EEF/joint interpolation
-> IK per waypoint (seeded from previous) -> TOPP-RA (only registered planner); NO collision-aware path planning, no closed-chain coupling
(per-arm control_part). Grasps = antipodal sampler on the mesh + gripper-object collision filter, scored by approach angle/opening/centroid
distance -- no reach/IK in the ranking (IK fails later = action fails). Our NERO port uses CarryEpisode._add_path, not their solvers.
11-04 SRS solver vs ours (NERO fitted into their W1 DH template, FK fit 2.3e-7; 200 random in-limit q/arm, ilab4): ours 200/200 both arms;
theirs as written 149/140 of 200; two bugs: reach test |SW| < d_bs+d_ew (=0.408 m; should be |d_se-d_ew|=0.04) and NO 2*pi angle wrap
(valid solutions rejected as out of limits). With both fixed: 200/200. They find ~1/6 the solutions/pose (p50 ~25 vs 128-174); pos err 1.8e-7 vs 3e-8.
Robot: CobotMagic 2x6-DoF, bases (0.233, +-0.300, 0) = 0.60 m apart, upright; tasks per-arm (separate trajectories), NO closed-chain task.
11-09 results read (arxiv.org/html/2605.11951 v1; Tables I-II and Sec. IV-A..C, V verbatim; Table III/V and App. D/E via summary only; App. B/C
not read). Success %, AgentChord vs best baseline, 20 trials/task. Real (Table II): fold towel 55 (CaM 50), coffee tray 60 (CaM 60, tie),
dual-arm pour 80 (CaM 70), handover 85 (CaM 75), rearrange 90 (ReKep 95: only loss), single-arm pour 95 (CaM 90); avg 77.5. Sim (Table I,
object dropped with prob p per atomic action): 95-100 on every task, lowest dual-arm pour p=0.10 (95). Failures: injected drops/pushes only;
no hidden contents, payload/torque limits or moving base. Their reasons: towel thin/deformable; general = reasoning limits, IK infeasible
in recovery, perception/monitor errors; limitations: "may still miss rare, compound, or non-hand-induced failures".
Ours: bases 0.185 m apart, tilted 45 deg outward, shoulders 0.38 m apart at 0.77 m. Script: scratchpad stack/srs_vs_agentchord.py, srs_diag.py.

## Lit: program synthesis for adaptation 10-28

_Lit check 10-28 on LLM program synthesis + CEGIS with a parameter-set certificate for online adaptation (pan demo); He Zhu (Rutgers) overlap and collaboration fit_

Idea (owner proposal, NOT yet a user scope decision): LLM proposes skill-DSL programs (loops allowed), the parameter-set
certificate admits / returns a counterexample parameter (CEGIS), re-synthesis while the pan rests (passively safe), loop
termination via ranking (each scoop removes >= delta). Belief update fixed; synthesize actions only. Fits C2/C3 of [[scope-pan-demo-2026-10-26]].

Verified by owner (abstracts):
- He Zhu (Rutgers): VRL PLDI'19 (CEGIS + verified program shield; state set, no LLM); ReGuS PLDI'24 (Cui/Wang/Qiu/Zhu;
  programs with loops/conditionals/procedures for manipulation, REWARD-guided, no certificate); VEL-inf (Springer 2025,
  pp 79-103; infinite-horizon reach-avoid via finite subtasks with pre/postconditions, verified chaining = completion,
  no ranking functions, no param sets). Also VELM CAV'24, RoboScribe OOPSLA'25, Formal-LLM (co-author). He gave a Rutgers talk
  2025-11-21 "From reasoning to acting: a program synthesis approach to RL". Strong collaboration fit.
- INPROVF arXiv 2503.13660 (Meng, Zhou, Weinberger, Kress-Gazit): LLM repairs GR(1) robot controllers after assumption
  violations, formal check + feedback loop; discrete, no param sets, no termination.
- Yang/Bhatt/Ward/Hu/Biswas/Topcu arXiv 2410.14865: LLM Python robot programs -> automata -> model-checked safety;
  composition theorem; no physical params, no completion, no runtime re-synthesis.
Subagent-only (unverified): Ravanbakhsh & Sankaranarayanan EMSOFT'16 robust CEGIS for CLFs (param/disturbance set + reach);
LOOPY FMCAD'24 and LIFT (LLM proposes ranking functions/loop bounds, sound checker) for C programs; LLM-Modulo; Jha'23 LLM+SMT CEGIS.

Verdict: each piece exists separately; the combination (LLM proposer + certificate over a set-valued physical-parameter belief
returning parameter counterexamples + loop termination/completion + online re-synthesis, in force-aware manipulation) not found.
Not searched yet: Kress-Gazit runtime revision beyond INPROVF, probabilistic termination/supermartingales, parametric MDP synthesis.

## PRISM 2608.25666 11-01

_PRISM (arXiv 2608.25666, Tartu) bimanual sampling-MPC — code MIT, but planner carries the object as a mocap (no grasp/slip/payload), JAX/MJX+CUDA, no box-lift branch; port stopped pending user_

PRISM: Dan, Hurova, Kruusamae, Singh (U. Tartu), arXiv 2608.25666 (v1 26 Aug 2026), page sites.google.com/view/prismbimanual.
Code: github.com/AKS-Lab-Univertsity-of-Tartu/bimanual_manipulation, MIT (LICENSE "Copyright (c) Alinjar Dan (2025)"), last commit
2025-11-27 (ccffb41). Read-only clone: /data/local/ss5772/ext_untrusted/prism_2026-11-01/main (552 MB, never run).
Verified from code (11-01):
- Branches main / pass_cube / push_ball / tray_multiple_targets: NO box-lift branch published.
- Rollout (mjx_planner.py ~610-625): joint VELOCITIES set directly (kinematic); tray = mocap body placed at gripper midpoint - 10 cm.
  No grasp force, mass, slip or lid -> cannot predict our hard-world failure (claw twist under hidden load with rolled grasps).
- Planner is ROS2 + UR5e 6-DoF hard-coded joint names; real_demo uses ur_rtde (hardware: never run). Config: batch 500, 15 steps, dt 0.1.
- Needs jax 0.5.3 + mujoco-mjx 3.3.1 + CUDA 12 (our env: mujoco 3.9.0, no jax/mjx). GPUs off-limits under the 11-01 compute rules.
Sandeep (relayed 11-01) asked to adopt it as proposer/controller under our certificate; I stopped before porting (his "stop if >2 days /
not available" rule) and reported. Related: [[demo-set-2026-11-01]], [[feedback-compute-rules-2026-11-01]].

## Lit: VLM assumption + safe test 11-02

_Lit scan 11-02 (shallow, Sonnet agent, arXiv abstract pages opened) — VLM proposes physical assumption + safe test + certificate over belief and tracking bound: combination NOT found; closest = Zhang/Zhou/Vasudevan provably-safe online SysID 2504.21486_

USER 11-02 framing: VLM supplies the assumption (fill/mass/tilt tolerance); robot tests it safely; tilt comes from imperfect
tracking, so the certificate must hold over belief AND a tracking-noise bound. My split: world claims (mass/fill/COM) are safely
testable by a probe admitted on the BROAD prior; tolerance claims (tilt-to-spill) are not -> derive from geometry (tan = 2(H-h)/D).
Abstract-level only (full texts not read):
- Closest competitor: Zhang, Zhou, Vasudevan, Provably-Safe Online System Identification, arXiv 2504.21486 (RSS 2025 listing):
  safe exciting trajectories + guaranteed interval payload bounds that tighten. No VLM, no tilt/spill, no task certificate.
- Heins & Schoellig 2411.07079 (RA-L 2025): nonprehensile transport robust to ALL realizable inertial params (moment relaxations);
  no active test, no tracking-error bound. Also 2305.17484 Keep it Upright.
- VLM hypothesis + interaction, no safety: PhysMem 2602.20323 (Guibas), CoRAL 2605.02600 (RSS 2026), DeliGrasp 2403.07832.
- Spill: 2408.00215 learned spill-free transport (fill level only a test condition). No certificate over a fill interval found.
- Unverified snippets: Corah & Michael ICRA 2017 safe cooperative lifting mass estimation; mocobots 2502.04600.
Related: [[lit-vlm-priors-safety-filters-2026-10-12]], [[armour-waitr-verdict-2026-10-21]], [[direction-analytic-vlm-boed]].

## Lit: VLM operates a simulator 11-02

_Lit scan 11-02 (abstracts only, Sonnet agent) — VLM operates a simulator to test assumptions; closest = Prompting with the Future 2506.13761, AntiGrounding 2506.12374, Real2Sim-BT 2601.08454, RigPI 2606.25212; combo with conditional cert + safe real probe not found_

USER 11-02 idea: give the (frontier/open) VLM simulator access to run tests instead of guessing; Jetson Thor on board.
My verdict to user: fits pillar 1 (stronger proposer), soundness unaffected (Lemma 1); sim tests show "IF theta THEN safe", not
that theta is true -> safe real probe still needed; VLM must never grade its own tests; add baseline "VLM + sim, no certificate".
Abstract-level findings (verified = arXiv page opened):
- LLM uses sim as tool: Mind's Eye 2210.05359 (MuJoCo, ICLR 2023 per snippets); LLMPhy 2411.08027 (AISTATS 2026, LLM writes sim
  programs, fits mass/friction from reconstruction error; notes non-identifiability).
- VLM + digital twin select/validate actions: Prompting with the Future 2506.13761; AntiGrounding 2506.12374 (twin-fidelity caveat).
- VLM prior + measured ID: RigPI 2606.25212 (IROS 2027, VLM bounds + differentiable sim + F/T); Real2Sim-BT 2601.08454 (VLM names
  params, behavior tree probes them -- closest to safe probe); Phys2Real 2510.11689 (ICRA 2026).
- Sim->real certificates: Betting 2608.21572 (anytime-valid, few real trials + sim bank); SPiDR 2509.18648; RVSG 2508.02338.
- Unverified: ~30% first-pass LLM sim code executable (secondary summary). No Jetson VLM+MuJoCo loop found.
Missing in all: conditional claim on an assumption box + safe real probe residual + hidden contents (fill/COM, tilt/spill).
USER 11-02 correction: the CLAIM is belief-state planning + task-preserving recovery; probes (incl. VLM-written probe programs) only
narrow the belief in real time -> probe/SysID papers are components to cite, not competitors; probe enters claim 1 only via its
measurement-noise bound (theta* stays in the narrowed set); narrowing keeps old certificates valid (subset).
Related: [[lit-vlm-assumption-safe-test-2026-11-02]], [[jetson-thor-specs-budget]], [[agentchord-probe-setup-2026-10-27]].

## Lit: probing + belief 11-03

_Lit scan 11-03 — mechanisms to ADOPT for probing / belief shrinking (not our claim); read Vasudevan 2504.21486 first_

Full list: docs/status/lit_probing_belief_2026-11-03.md (contents-armside2 worktree). 22 arXiv ids title-checked by Opus.
Key: Zhang/Vasudevan 2504.21486 (sound bounds + safe excitation, code) = closest; Lopez RaCBF 2003.10028 (set-membership);
Kruzliak 2404.07344 (EIG probe choice + stop, coded baseline); U-GRAPH 2502.02663 (active CoM); PhysCaP 2608.21031 (LLM picks probes, no gate);
Phys2Real 2510.11689 (VLM prior fusion = our current setup); MaD Physics 2605.10820 (LLMs weak at choosing measurements).
These are components we adopt (user 11-03); no novelty is claimed here. See [[feedback-belief-mechanisms-adopted]].
Related: [[probe-reask-2026-11-03]], [[lit-vlm-assumption-safe-test-2026-11-02]].

## Lit: grasp + bimanual planners 11-04

_11-04 web survey (3 Sonnet agents, shallow) of arm-aware grasp selection + bimanual closed-chain planners + usable libraries; lead-verified arXiv titles_

USER 11-04 asked for a survey after the AgentChord IK miss (right j7 5-10 deg past +90 deg on fixed-orientation carry moves).
Lead-verified (arXiv API): Cohn/Shaw/Simchowitz/Tedrake 2309.08770 (bimanual planning on analytic-IK + psi parametrization, IRIS/GCS in Drake,
kinematic only, no paper repo; demo page cohnt.github.io/Bimanual-Web); Garg/Cohn/Tedrake 2411.18913 (undistorted parametrized C-space GCS);
Haghshenas/Hansson/Norrlof 2303.07039 (convex TOPP for cooperative arms, wrench limits); Lertkultanon/Pham 1705.02573 (certified-complete bimanual regrasp);
Wan/Harada/Kanehiro 1812.03274 (single vs dual arm preparatory regrasp); Cai et al. 2509.19261 (grasp scored by manipulability + joint-limit proximity);
Ghalamzan et al. 1707.08147 (grasp chosen for the post-grasp motion; human-in-loop); Reuleaux 1710.01328 (reachability maps, ROS1).
Vahrenkamp KIT PDFs exist (2012 Grasp-RRT / bimanual grasp+motion; 2011a bimanual grasp manipulability) - Simox C++ archived.
Not verified by lead: Dio/Graichen/Volz IROS 2024 cooperative TOPP third-order (FAU cris page); Wu et al. Robotica 2022 dual redundant regrasp bridging.
Libraries: OMPL 2.0.1 pip cp312 (constrained spaces), Drake 1.58 pip (GCS needs Mosek typically), pin 4.1.0 pip; cuRobo/GraspGen need GPU;
learned grasp nets ignore the arm. No ready code found for arm-aware grasp selection or bimanual closed-chain planning on MuJoCo -> build on our SRS IK.
TOPP-RA already does joint-torque constraints (agent claim "no planner enforces torque" ignores it). Related: [[planner-decisions-2026-11-03]], [[stack-architecture-2026-11-03]].

## Latent-HJ literature 2026

_HISTORY (superseded, 11-05 audit): Verified 2025-26 papers beyond Fisac-discounted latent HJ; LatentCBF (margin smoothing), Kim et al. (estimation gap), value-constrained MPC terminal constraint_

Literature search 2026-09-28 (Sonnet agent). The owner opened the arXiv abstracts of the top 5 to confirm title, authors and dates.
- **2511.18606 LatentCBF** (Nakamura, Bishop, Man, Johnson, Manchester, Bajcsy, Nov 2025)
  - Diagnosis: latent values saturate and jump because the margin l is classifier-like.
  - Fixes: a gradient penalty on l, and critic data mixed from nominal + safety policies.
  - Unverified full-text claim: L_V ≤ L_l·max{1,(1−γ)/(1−γL_f)}.
- **2510.06492** (Kim, Nakamura, Bajcsy): "estimation gap" (safety quantity unobservable, i.e. our cup contents)
  vs "prediction gap". It gives an MI observability diagnostic, privileged supervision for estimation gaps, and conformal
  calibration for prediction gaps. This explains why our learned-latent value collapsed while the physics latent worked.
- **2604.23863** (Wang, Nguyen, Jordana, Righetti, Bansal, Apr 2026): a learned HJ value as an MPC TERMINAL constraint,
  with recursive feasibility, on a Flexiv Rizon arm. Closest architecture to FL-MPC + value rows.
- **2606.16022 λ-Reachability** (Chen, Li, Sun, Liu, Jun 2026): a geometric-horizon multi-step safety Bellman target;
  a drop-in replacement for the one-step Fisac target.
- **2609.14087** (Feng, Sahin, Thorup, Bansal, Sep 2026): gradient-free supervised discrete-time neural HJ with a
  windowed curriculum; benchmarks up to 80-D.
- Others the agent opened but I did NOT re-verify: RBN 2505.11755; ExactBC 2404.00814; Lin & Bansal conformal 2312.08604;
  UNISafe 2505.00779; ISAACS 2212.03228; FISOR 2401.10700; Li et al. Lipschitz reach-avoid 2408.07866.
- Novelty hunt 2026-09-28; the owner opened these abstracts:
  - **2609.15915** (Li, Tang, Azizan, Sep 14 2026): a safety value over the INFORMATION space (state + belief), with
    a Bellman equation, for meta-RL. => "Belief-conditioned amortized safety value" is TAKEN; don't claim it.
  - **2209.14976** (Borquez, Nakamura, Bansal): parameter-conditioned DeepReach, queried online with POINT estimates;
    no uncertainty sets, no guarantees.
  - **2609.28984 CrossSafe** (Tabbara, Yang, Sibai, Sep 24 2026): cross-embodiment latent HJ filters over FIVE
    BIMANUAL robots. The closest neighbour to any "bimanual latent HJ" claim.
  - **2302.00171** (Hu, Isele, Bae, Fisac, IJRR): dual control (active uncertainty reduction) + HJ safety shield. The
    uncertainty is other agents' intent (driving); the robot probes itself.
  - Agent-ranked surviving gaps, abstract-level only (read ACoFi 2604.18482 and Vahs 2603.10572 / 2601.02948 in full
    before claiming):
    1. Few-shot recalibration of a learned safety margin from k SAFE (gentle) probes, valid at AGGRESSIVE deployment:
       a one-sided probe-to-deployment shift that the safety constraint itself induces.
    2. A calibrated posterior-to-tube guarantee when the posterior is learned or misspecified.
  - **2509.12674** (Johansson, Lindmark, Wiberg, Servin, Sep 2025, owner-verified): a physics-sim safety filter for
    SIMULATED BIMANUAL manipulation under uncertain mass/friction, with "targeted uncertainty reduction through probing
    actions". It focuses on grasp stability; the abstract doesn't specify how probes are selected; no vision.
    A likely reviewer citation against any probing claim.
  - G1/G2/G3 agent verdicts:
    - decision-focused BOED is largely SOLVED (Hübotter et al. transductive active learning; Action-BED 2606.23662):
      use it and cite it, don't claim it;
    - test-time observability-gap detection is PARTIAL;
    - "predict whether the BRT will shrink (nominal / probe / neither), calibrated against realized shrinkage" is
      OPEN (~55%). Full-text read needed first: Hu 2606.02562, Vahs 2603.10572, Li 2609.15915.
  - Both agents converged independently: SAFE probing gives only ONE-SIDED (censored) information ("no failure at
    excitation a" certifies nothing beyond a). That links G3 and G4. SafeOpt's Lipschitz expansion is the classic
    GP-bandit answer to check against.
  - VLM / shared-autonomy neighbours, owner-verified 2026-09-28:
    - **2409.14580** (Santos, Li, Peters, Bansal, Bajcsy, ICRA 2025): language + VLM feedback updates HJ constraints
      online, human → robot only; the robot never requests.
    - **2510.11689 Phys2Real** (Wang, ..., Schwager, ICRA 2026): VLM prior over physical parameters (COM) fused with
      online estimates; ensemble uncertainty; no safety or guarantee.
    - **2602.01092** (Zhou et al., Feb 2026): bimanual teleop guided by an offline-RL conservative value; no HJ, no probes.
  - **Consolidated gap ranking 2026-09-28:**
    1. **G3:** predict before acting whether safe-set uncertainty shrinks under nominal motion / a probe / neither,
       calibrated against realized shrinkage (~55% open).
    2. **G4 + G3b merged:** a calibrated per-object θ credible set (VLM prior + safe probes) valid under the shift the
       safety constraint itself induces (gentle probes → aggressive use; known → novel objects). The tube-containment step
       is only a corollary if V is monotone in θ.
    3. **G2b:** probe REQUESTS to a noisy human operator that stay inside the tube with high probability, maximizing
       information per unit operator effort (~50%). Best treated as G3's probe arm.
  - For unobservable fill, a correctly calibrated VLM set covers roughly the full range, so the guarantee is valid but
    useless. The VLM's value is on OBSERVABLE θ.
  - Jetson Thor VLM latency is UNDOCUMENTED: budget seconds, and use the VLM as a per-object prior.
  - **Full-text read 2026-09-28** (PDFs + extracted text in /common/users/ss5772/bimanual_data/lit/pdfs/; report in
    lit/fulltext_G3.md; the owner grep-verified 6 key quotes). G3 is NOT killed (~70%).
    - Components exist separately:
      - continue/probe/fallback exists only as structure (Johansson; its probes were "emulated" by hand-editing the
        distribution);
      - PRMPPI switches to robust reactively and notes "mass becomes unobservable in the upright pole position" without
        addressing it;
      - SafeOpt's expander is an optimistic pre-probe test in a bandit setting with no nominal-route data source;
      - Hu BeliefSF only suggests falling back "whenever inference quality is low", and its guarantees are
        "distributional rather than worst-case".
    - Most likely cited against G3: Hu 2606.02562 + Deception Game 2309.01267; for the probe half, SafeOpt/StageOpt.
    - Censoring lemma: SafeOpt expansion from a censored-only seed (ℓ = h) expands nothing. The THREAT is any
      observable near-miss margin; the thesis must justify that none is observable.
    - Must beat: PRMPPI posterior-CP + reactive switch; implicit V(x,b); SafeOpt expander score; θ-EIG/variance;
      reactive monitors; ACoFi-style V calibration.
    - Not yet read: GoSafe / safe BO with binary feedback (the biggest censoring threat), staircase/Bruceton
      threshold estimation, Bajcsy 2021 learning-aware HJ.
  - **Full-text read PART 2, 2026-09-28** (lit/fulltext_G3_part2.md; the owner grep-verified the Dual-gatekeeper and
    SABRE quotes). G3 narrowed; ~50% that the whole combination is unpublished.
    - **TOP THREAT: Dual-gatekeeper** (Naveed, Singh, Agrawal, Panagou, arXiv 2604.15507, Apr 2026; v1 2510.06351).
      Budget-constrained safe dual control: predict θ-uncertainty shrinkage of candidate probe trajectories before
      acting; commit only if safe and within budget; else stay robust. Future work: "execution-aware" prediction.
    - **SABRE** (Bennett, Misra, Kallus, 2210.14492): binary safety feedback; a "region of disagreement" ≈ our U_t;
      "we cannot easily extrapolate safety beyond the observed states".
    - **ISE** (Bottero et al., NeurIPS 2022, 2212.04914): information about the safety indicator.
    - Distinctions:
      - D1 (value-level target) FALLS as novelty;
      - D3 (censoring lemma) FALLS: known and trivial per sensitivity testing / SABRE; keep it as a proposition;
      - D2 (nominal/probe/none split) survives weakly;
      - D4 SURVIVES, narrower: a distribution-free one-sided calibrated forecast of VALUE-level shrinkage per source that
        holds when executed ≠ planned. It's exactly Dual-gatekeeper's future work, so the scoop risk is high.
    - Still unchecked: EWOC (escalation with overdose control, a classical bounded-failure threshold search).
- **Bottom line:** nothing recovers information that is absent from the latent. Only privileged supervision, an
  explicit physics state (ours), or uncertainty augmentation do.

Related: [[latent-hj-flmpc-result]], [[wm-requirement-ladder]].
- **RSS-claim lit check (2026-09-29, Sonnet agent; the OWNER HAS NOT verified these quotes):**
  - Mechanism (a), "the latent HJ value is exploited by its own max over actions", is KNOWN qualitatively:
    - UNISafe Sec 3 ("optimistic safety estimates of uncertain OOD scenarios");
    - Cao & Bansal "When World Models Lie: Adaptive Safety Analysis Under Wrong Imaginations" arXiv 2609.34300 (Sep 28 2026, CONCURRENT; OWNER-VERIFIED 09-29: "the resulting value function can inherit its errors and produce overconfident safety estimates"; fix = Adaptive Conformal Inference uncertainty sets from predicted-vs-observed latent error + pessimistic safety evaluation, deployment-time only, NO new data; sim + hardware) => it is a MANDATORY baseline; our distinction = fix the model/value with targeted data instead of pessimism, measured as less conservatism at equal safety;
    - LatentSafe's incomplete action-coverage ablation;
    - LatentCBF (critic OOD actions; mixed data);
    - FISOR / V-OCBF (OOD-action max in offline HJ);
    - RAMBO/MOPO in MBRL.
  - The fix (c) is structurally DAgger-for-models (Ross & Bagnell 2012). VAML/objective-mismatch only reweight data.
  - NOT FOUND: (b) "the greedy fallback is worse than holding still" measured, and (c) VALUE-ARGMAX-directed
    counterfactual data for safety values.
  - Sharpest reframing: "the max is an adversary against the model (winner's curse); the min with ℓ makes the error
    matter only near the boundary, collapsing V to ℓ."
  - Must beat: in-data-restricted max (FISOR/IQL), UNISafe and Cao-Bansal pessimism, DAgger-style fallback rollouts,
    RAMBO adversarial training, ensemble-uncertainty-directed branches. Replicate on a second WM type.
  - Most cited against: UNISafe (mechanism), Ross & Bagnell (fix).
- **SafeBimanual (Deng et al., arXiv 2508.18268, Aug 2025; read from the abstract page only, 2026-10-05):**
  - Test-time trajectory optimization over ANY pre-trained bimanual diffusion policy: hand-designed safety costs
    (tearing avoidance, arm-object collision) applied as guidance during denoising.
  - A VLM schedules which costs apply, from keypoints and pairwise relations.
  - RoboTwin sim, 8 tasks: +13.7% success, -18.8% unsafe interactions. Real robot, 4 tasks: +32.5% success.
  - **Relevance:** "a VLM picks the safety constraints" is TAKEN for bimanual. Our VLM role must differ: a calibrated
    prior over hidden physical params / failure family for a learned reachability value, refined by probes, never
    certifying.
  - Their hazards are geometric/kinematic (our FL-MPC analytic layer). They show no dynamics- or hidden-state hazards
    (spill, a rolling payload), which is our gap. Their safety is soft guidance, not hard constraints or reachability.
  - **Tasks** (from the HTML via the fetch summarizer, not PDF-checked):
    - RoboTwin sim: Dual Bottles Pick (easy/hard), Block Handover, Blocks Stack, Pour Water, Pick Apple Handover,
      Dual Shoes Place, Diverse Bottles Pick. DP average: SR 20.7 → 36.0%, DR 45.6 → 20.9%.
    - Real (Galaxea-R1, two 7-DoF arms, ZED head cam + D435i wrist cams; iDP3): Pass Banana, Pour Water, Wipe Bowl,
      Place Fruits, Prepare Breakfast (pour + place + carry loaded plate).
  - VLM = GPT-4o, per task stage, outputs a binary mask over the 5 keypoint costs (object collision, behaviour
    alignment, poking, tearing, gripper collision).
  - "Spill" in Pour Water is handled only by a geometric alignment cost.
  - **Bridge:** their "carry loaded plate" stage is exactly where keypoint costs can't see a rolling or shifting
    payload, so it is a natural comparison scenario for tray + ball.
- **HJRNO (Li & Chen, SFU, arXiv 2504.19989, Apr 2025; from the search summary only):**
  - A Fourier Neural Operator learns the map (obstacle/failure-set function + system config) → HJ value function on a
    grid, with ~1e-3 s inference, and generalizes across obstacle shapes and dynamics configs.
  - Related: arXiv 2603.17022 (FNO for the HJI solution operator under varying obstacles, contingency planning).
  - **Owner view:**
    - grid FNO = low-dim state only. For our latent value you'd use a DeepONet-style branch (object / margin
      descriptor) + trunk (latent z), which ≈ a conditioned value.
    - An operator speeds up the SOLVE; it does not fix the WM's coverage hole, so it gives a "fast wrong answer"
      without counterfactual data.
    - Fits as: VLM descriptor → operator → value, refined by probes; it also helps real time (value rows 102 ms).
- **User Q 2026-10-05, tracking quality:** owner plan.
  - Measure the tracker error model offline.
  - Online, use an innovation / NIS check against WM predictions.
  - Train on the measured noise model (the tray_rep tracker params).
  - Set the V threshold by outcome-conformal (oc_eval) under the real tracker noise.
  - On dropout, fall back to the conservative policy.
  - Sweep latency 0/1/3/5 ticks to get a tracker latency requirement (the lead is only 70 ms).
- **LIT CHECK 2026-10-05: ReKep + set-based constraints** (8 searches + 3 page fetches; quotes from summarizers).
  - **ReKep** (Huang et al., CoRL 2024, 2409.01652):
    - GPT-4o writes Python costs over 3D keypoints (DINOv2 + SAM + k-means), tracked at 20 Hz; bimanual on two
      Frankas (garment fold, pack shoes);
    - NO keypoint uncertainty, NO guarantees, rigid-grasp assumption, no payload dynamics;
    - point tracking is the LARGEST failure source (occlusion), shown in Fig. 4; ~49% per a secondary summary, not
      verified.
  - **VLM / language → failure set for HJ is TAKEN:**
    - Santos, Li, Peters, Bansal, Bajcsy 2409.14580 (VLM updates the failure set, HJ warm-start online);
    - AnySafe 2509.19555 (Agrawal, Seo, Nakamura, Tian, Bajcsy: latent filter conditioned on a constraint image,
      conformal-calibrated similarity);
    - Tabbara, Yang, Sibai 2608.00315 (language-conditioned HJ actor/critic, Jul 2026);
    - 2509.14758 (latent filters on pre-trained vision models).
  - VLA + CBF semantic layers: VLSA/AEGIS 2512.11891; a graph-fused VLA dual-robot CBF (engrXiv 6733);
    2607.01378; 2607.29569; VLPSA 2609.22462.
  - Perception-uncertainty sets: measurement-robust CBFs (Dean 2010.16001; Cosner 2104.14030), perception contracts
    2309.13515, adaptive conformal CBF 2407.03569 / 2609.23280, ATOM-CBF 2511.08741, predictive zonotope runtime
    monitoring 2609.03699.
  - **Gap (as far as this check shows):** VLM-named parts grounded to point-cloud segments + set-valued perception
    uncertainty + a learned reachability value for a carried-payload DYNAMICS hazard (bimanual) is not combined
    anywhere.
  - **Sharp version:** "perception error at the moments that matter (track loss near the edge) breaks latent safety
    values; calibrated set-valued robust margins restore safety at a measured cost". Testable on tray+ball point
    clouds.
- **DEEP LIT PASS DONE 2026-10-05:** bimanual/reports/Perception robust latent safety filters.md (228 lines; 4
  researchers, notes in research_notes/Perception robust latent safety filters/).
  - **Verdict: HALF OPEN.**
    - TAKEN: the min of a learned latent HJ value over a calibrated set (Cao & Bansal 2609.34300, ACI over the
      predicted-vs-encoded latent gap); robust margins from perception error with the conservatism cost measured
      (RoVer-CoRe 2511.14755, GUARDIAN 2602.06026, Jahanshahi & Chen 2608.13819; grid/analytic values only).
    - OPEN: (1) the false-safe rate of a learned/latent value vs distance to the boundary under tracker noise /
      delay / dropout; (2) a set built from the TRACKER's uncertainty that grows during track loss (all latent filters
      trust the encoder; Kim et al. 2510.06492 name the "estimation gap"); (3) any perception-robust safety for a free
      payload on a carried tray; (4) a required perception latency for a latent filter.
  - **Thin spot:** if the value takes the tracked ball state directly, min-over-ellipsoid = MR-CBF/GUARDIAN on a
    learned value. Defences: the boundary-resolved measurement + set growth during loss, or a learned point-cloud
    encoder.
  - Experiments proposed: the boundary false-safe map (kill if near ≈ far); 5 filters at equal conservatism incl.
    Cao-Bansal ACI (kill if ACI ≈ tracker set during dropout); a latency budget from the lead time.
  - TODO: read the Cao & Bansal FULL text before claiming any difference.
  - **PRISM 2608.25666 (owner-checked abstract):** bimanual sampling MPC with a GPU physics sim as the online world
    model. PerAct² variants incl. obstacle-constrained BALL TRANSPORT and tray transport; dual UR5e sim-to-real.
    Safety = joint kinematic bounds only; perception not stated. Cite as the nearest bimanual ball/tray task; not
    perception-robust, no reachability.
  - Correction for the report: the latent value-row cost is 102 ms (MASTER §1b), not the old 190 ms memory figure.
- **LIT CHECK 2 (2026-10-05): reports/lit_spatial_robust_semantics_and_caobansal_2026-10-05.md.** Cao & Bansal and
  PRISM were read from the full PDF text.
  - **Cao & Bansal 2609.34300:**
    - the set = an L2 ball in LATENT space around the predicted latent;
    - radius = the ACI quantile of the predicted-vs-encoded latent gap, capped by a trajectory-level conformal q_max;
    - it trusts the encoder; NO dropout / occlusion mechanism (staleness only as a test probe o_{t-L});
    - Theorem 1 bounds the latent-error exceedance, not safety;
    - Franka 15 Hz tomato on a dustpan, open-loop replays, no injected perception noise.
  - **PRISM 2608.25666:** the ball is squeezed between two UR5e grippers and tracked by OptiTrack mocap over ROS2. No
    camera / point cloud / noise model. Kinematic hard constraints, collision soft, no guarantees, real runs
    qualitative.
  - **Ingredients already taken:** conformal STL runtime verification (Lindemann, Qin, Deshmukh, Pappas, 2211.01539);
    interval / three-valued STL (Baird, Harapanahalli, Coogan 2309.10686; Finkbeiner 2022; Deshmukh 1506.08234);
    conformal robustness lower bounds from images (Hoxha 2605.13923); STL↔HJ (Chen, Tam, Livingston, Pavone, WAFR
    2018; So, Sharpless, Herbert, Fan 2605.01051).
  - **Open:** the COMPOSITION (interval / three-valued SpaTiaL over tracker sets that grow during track loss, used as
    a learned HJ target) + the measured false-safe rate near the boundary.
  - Do forward-citation passes on 2309.10686 and 2609.34300 before any "first" claim.
- **The user's "VLM prior + arm learns the sim-to-real gap" paper = Phys2Real** (arXiv 2510.11689; Wang, Tian, Swann,
  Shorinwa, Jiajun Wu, Mac Schwager; ICRA 2026). It is from Schwager's MSL, NOT Bohg's IPRL.
  - A GPT-5 centre-of-mass prior (self-reported uncertainty) is fused by inverse variance with an RMA-style
    10-model ensemble estimate; the policy is conditioned on the parameters. xArm, T-block / hammer push.
  - No safety constraints, no sets; VLM calibration is future work.
  - Open niche (agent): calibrated VLM prior → set (conformal → box / zonotope) as the initial uncertainty set in
    robust constraints, shrunk online while acting safely.
  - Related: Scan-Materialize-Simulate 2505.14938 (open-loop VLM physics); RigPI 2606.25212 (a VLM-bounded search box
    + differentiable sim); DrEureka 2406.01967; ASID 2404.12308; PhysTwin 2503.17973.
  - Notes: research_notes/Zonotope constraints and VLM physics priors/vlm_priors_sim2real.md.
- **Polynomial annihilation (PA) / split points (2026-10-08; notes: research_notes/Zonotope constraints and VLM physics
  priors/polynomial_annihilation_splitpoints.md):**
  - PA (Archibald, Gelb & Yoon, SINUM 2005) = a local jump / kink detector on scattered samples. Used in UQ
    (discontinuities of simulator outputs; Gorodetsky & Marzouk 1402.2845 = PA + SVM active learning), DG shock
    detection and image reconstruction. NO use found on point clouds, robotics or safe sets.
  - **Agent pushback:** split on ZONOTOPE OVER-COVERAGE / concavity / fit error, not raw geometric edges (a box has
    edges but is already a zonotope; a bowl has none but must be split; zonotopes are centrally symmetric, so wedges
    waste volume).
  - PA's best uses: kinks in 1D cut-plane profiles (prune CoACD), active-constraint switches in clearance / safe-set
    functions, black-box dynamics. Weak on noisy single-view clouds.
  - VLM role: say where precision matters + give coarse part seeds; geometry places the exact cuts.
  - Close 2026 work: Dwedar 2609.27816 (VLM → inflated obstacle sets + zonotope reachability check),
    CompCPZ 2608.17717 (constrained polynomial zonotopes along the language parse tree).
  - Split rule to reuse: Althoff, Stursberg & Buss CDC 2008, generator-split criterion.
  - Proposed test: CoACD vs VLM parts vs VLM + PA refinement → extra volume, false clearance violations, QP time.
- **Zonotopes from point clouds (2026-10-08; notes: .../zonotopes_pointclouds_manipulation.md):**
  - ARMTD 2002.01591 / ARMOUR 2301.13308 / SPARROWS 2402.08857 / CROWS 2410.09924 use zonotopes for the ROBOT's swept
    volume; their OBSTACLES are given (mocap, hand-placed boxes), never point clouds. SPLANNING 2409.16915 uses a
    Gaussian splat.
  - Point cloud → convex sets in manipulation = superquadrics / ellipsoids / learned distance fields, not zonotopes.
    CoACD needs a mesh (bad for live partial clouds). GCS / IRIS give free-space polytopes as linear rows.
  - **The closest precedent to beat:** Brunke et al. 2410.15185 (Schoellig lab).
    - An LLM gives (object, relation) pairs, each turned into a superquadric fitted to the point cloud (upward stretch
      for "above"); 12 relations.
    - CBF filter at 45 Hz on a Franka. Cup of water "constrained rotation" exists but is only a SOFT cost.
    - Constrains the end-effector position only.
  - Also VLSA / AEGIS 2512.11891 (VLM picks the obstacle → ellipsoid → QP, end effector only) and VLPSA 2609.22462
    (Poisson safety, full body + grasped object).
  - **Gap (provisional):** language / VLM- or perception-specified ZONOTOPE sets → per-step support-function rows in a
    feedback-linearised MPC; nothing bimanual.
- **HJ vs invariant sets (2026-10-08; notes: .../hj_vs_invariant_sets.md; 39 IDs checked):** the user's claim is
  MOSTLY WRONG.
  - HJ accepts arbitrary failure sets, but backup-CBF / gatekeeper / MPC filters also accept any evaluable constraint.
    Only HJ gives the MAXIMAL set + the optimal safe policy.
  - It is NOT fast: grid HJ is limited to ~4-6 states (2104.11332, 2309.05837); fast only as a lookup of a
    precomputed value.
  - A BRT is NOT needed for infinite-horizon safety: any control-invariant set works (backup set, MPC terminal set,
    CBF, brake-to-stop as in ARMTD / SPARROWS). With a good backup, the backup-CBF set ≈ the HJ set.
  - Fast HJ variants trade away guarantees (DeepReach probabilistic, HJRNO none).
  - Zonotope methods: polynomial cost; conservative under the wrapping effect; ARMTD 0% vs SPARROWS 40% at 40
    obstacles; backward reachability only under-approximated (Yang & Ozay 2107.01724).
  - Repo implication: 2 arms = 24 states → grid HJ infeasible. Standard route: a brake-to-stop terminal constraint in
    the FL-MPC QP. BUT on the real Piper, holding at rest needs ~7.4 of 8 Nm, so that terminal set barely exists on
    hardware.
- **WAITR = the CLOSEST TECHNICAL RIVAL** (owner-checked abstract, 2026-10-08): "Serving Time: Real-Time, Safe Motion
  Planning and Control for Manipulation of Unsecured Objects", Brei, Michaux, Zhang, Holmes, Vasudevan, arXiv
  2309.03111.
  - Reachability over-approximates the CONTACT WRENCH on an unsecured object, with uncertainty in arm dynamics,
    object dynamics and friction. Certifiably safe, real time; single arm; sim + a real demo.
  - No VLM prior, no online shrinking, not bimanual. Object perception is not stated (likely given).
  - Our novelty must sit on: VLM prior → calibrated set + online shrinking (BOED) + bimanual closed chain + perceived
    point-cloud geometry + a ROLLING payload (no static contact).
- **REPORT: reports/Zonotope constraints and VLM physics priors.md** (126 lines).
  - Language → zonotopes is already partly taken (CompCPZ 2608.17717, Dwedar 2609.27816; abstract-level only).
  - Experiment B needs a synthetic VLM prior in sim (rolling resistance is not visible in a render).
  - The Piper ff cap is ±8 Nm, but a position-offset encoding moves the limit to the motor's peak torque (unknown,
    hardware-only).

## Pan-lift skills web scan 11-01

_HISTORY (superseded, 11-05 audit): Web scan 11-01 — no public pan/pot lift-with-lid or rim-clamp skill; reuse mink RelativeFrameTask (in our pinned 1.2.0) for a coupled two-arm lift_

Sonnet web scan 11-01 (standard search, mostly snippets, nothing cloned). Found NO public pan/pot-lift-with-lid skill, no rim-clamp
skill handling torsional slip, no coupled-arm RoboTwin task.
- mink `RelativeFrameTask` (claw-to-claw relative pose in one QP) — present in our pinned mink 1.2.0
  (bimanual_data/pylibs/mink120/mink/tasks/relative_frame_task.py, checked). Kinematic only: does not fix the right arm's ~10 deg sag
  (PD tracking) and gives no internal-force control.
- DuoBench Carry-Pot (github.com/RobotControlStack/duobench, arXiv 2606.11901, MuJoCo, Apache-2.0): 2 Frankas lift pot by side
  handles; VR-teleop demos, no coupled controller, no lid; ACT 44% / pi0.5 21%.
- arXiv 2511.21264 (MJX sampling MPC, 2x UR5e, tray): relative-velocity + EE-distance costs, no code. arXiv 2608.20946: tray with
  hard relative-transform constraint, no code.
- Lid, rim clamp: nothing reusable; ours to build.
Related: [[pan-skills-rebuild-2026-10-30]], [[contact-model-2026-10-30]].


## 2026-11-12: existing probing skills (web search, Sonnet research agent; links fetched unless "search only")
- robot_payload_id (Pfaff et al. 2025, arXiv 2503.00370, github.com/nepfaff/robot_payload_id, MIT): excitation trajectories + SDP physically consistent fit of the payload's 10 inertial
  parameters (+ reflected inertia, friction); Drake, single arm; torque-only use not stated in the README.
- OnlineSafeSysID (roahmlab, RSS 2025, arXiv 2504.21486): payload inertial parameters as guaranteed intervals, exciting trajectories kept inside torque limits; Kinova + ROS 2 +
  RAPTOR + HSL; single arm.
- ShahramKhorshidi/system_identification (ICRA 2025, arXiv 2409.09850, BSD-3): LMI/SDP or log-Cholesky physically consistent fit; Pinocchio; LMI needs MOSEK.
- Nadeau, Giamou, Kelly 2022 (arXiv 2203.00830): fast physically consistent payload ID for cobots; estimator code not found.
- VLM-chosen probing: Adami et al. 2026 (arXiv 2601.08454, VLM writes behavior trees of sensing primitives, real Franka), Phys2Real (arXiv 2510.11689, VLM prior fused
  with interaction data; search only). DeliGrasp (arXiv 2403.07832) has the LLM give numeric mass/friction (search only).
- Not found: a MuJoCo/robosuite/ManiSkill/Isaac skill library with weigh/shake/tilt probes; safe incremental-lift (hefting) code; cooperative bimanual payload ID code.
- Claude on embodied benchmarks (same day, Sonnet agent): ERQA Claude 3.5 Sonnet 35.5% vs GPT-4o 47.0% (arXiv 2503.20020); BenchLM ERQA board Opus 4.5 46.8%,
  Opus 4.6 51.6% vs Gemini 3 Pro 70.5% (mostly vendor-reported); no Claude 5.x physical/embodied numbers found.
