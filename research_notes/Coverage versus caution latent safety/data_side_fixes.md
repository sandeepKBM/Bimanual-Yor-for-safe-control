# Data-side fixes for world-model / learned-dynamics exploitation in control and safety (2010 – Oct 2026)

Scope: counterfactual augmentation, DAgger-for-models, directed vs random data collection, predator-prey (falsifier) loops. Latent-HJ-specific papers (LatentSafe, UNISafe, Cao & Bansal, LatentCBF) and pure offline-RL theory are excluded except as one-line pointers.

Method and verification status:
- Quotes tagged "[full text]" were extracted by me with `pdftotext` from PDFs saved in `/common/users/ss5772/bimanual_data/lit/pdfs/` (files named `arxiv_<id>.txt`; ISAACS is `isaacs_hsu23a.txt`). Quotes tagged "UNVERIFIED" come from search snippets or abstracts only. I could not open Corso et al. beyond a snippet.
- Closeness labels: SCOOPS (already publishes the candidate claim), PARTIAL (publishes a component of it), BACKGROUND (context only).
- Flags: [RECENT 2025-26] = recent; [CONCURRENT] = posted within about 3 months of 2026-10-01.

**Headline (adversarial).** I found NO paper that SCOOPS the full candidate claim: off-policy counterfactual action coverage, not pessimism and not on-policy aggregation, fixes a learned safety value, and the selection rule matters little. Components are published separately, and several are close:
- (i) "Pessimism / conservative model use does not fix coverage" is stated outright by RECON (2609.33336, concurrent) and RENEW (2607.14180, concurrent), but in imitation / offline MBRL, not safety values.
- (ii) "On-policy data alone misses what the planner will pick; you also need a separate exploration distribution" is the core of Ross & Bagnell 2012.
- (iii) "Off-expert action support gap fixed by action-coverage expansion" is WorldSync (2608.24885, concurrent), for world models used as policy simulators.
- (iv) "A model-exploiting adversary exposes counterfactual action failures" is GALILEO/AWRM (2206.04890), by reweighting, not by new data.
- (v) "Directed vs uniform sampling for learned HJ safety values" is Dallas et al. 2604.13325, and it finds directed WINS significantly. This is the most direct contrary evidence to "selection rule matters little".

Not found anywhere: a head-to-head of on-policy DAgger rollouts vs off-policy counterfactual branches for a learned safety VALUE or filter. That specific comparison, plus a null result on selection rule, appears open.

---

## Q1. DAgger-for-models, agnostic system ID, and on-policy vs exploratory model data: what do they show about on-policy data missing the actions a planner's max will pick?

### Takeaway
Ross & Bagnell (2012) formalize the exact failure ("controller synthesis picks a policy very different from the data-collection policy, so the model is bad where the controller goes"). Their fix is NOT on-policy data alone. It mixes on-policy data with samples from a separate state-action exploration distribution, and the guarantee is relative to how well that distribution covers the relevant policy. Later empirical work (Nagabandi 2018, Lambert 2020) supports "on-policy aggregation helps, but on-policy data alone does not give complete dynamics coverage".

### Cited Findings
- **Ross & Bagnell, "Agnostic System Identification for Model-Based RL" (ICML 2012; arXiv 1203.1007). Label: PARTIAL, high.** Local copy: `arxiv_1203.1007.pdf`. [full text]
  - Problem statement: "This problem arises as the policy resulting from controller synthesis is often very different from the 'exploration' policy used to collect data." And: "Train: model is fit based on samples near the desired trajectory ... Test: learned policy ends up in new regions where model is bad, leading to poor control performance." (Fig. 1 caption)
  - Why coverage is needed: they note open-loop ID and persistent excitation "attempt to sufficiently 'cover' the state-action space" but need the realizable-model assumption. Their iterative method (DAgger for MBRL) "iterates between 1) collecting new data about the system by executing a good policy under the current model, as well as by sampling from a given exploration distribution, and 2) updating the model".
  - Dependence on the exploration distribution: "our approach uses a state-action exploration distribution to sample transitions and allows us to guarantee small regret against any policy with a state-action distribution close to this exploration distribution." And: "If the exploration distribution is close to that of a near-optimal policy, then our approach guarantees near-optimal performance, provided a good model of data exists."
  - Batch baseline: "Due to the mismatch in train/test distributions, uniform exploration is often the best option with this approach" (Batch), at a cost that scales with MDP size.
  - Helicopter results: compared nu_t (noise around the desired trajectory), nu_e (expert controller), nu_en (expert plus noise). "In both cases, for any choice of nu, DAgger outperforms Batch significantly" and "DAgger is more robust to the choice of nu, as it always obtains good performance given enough iterations, whereas Batch obtains good performance with only one choice of [nu]". Relevance: the aggregation loop makes the choice of exploration distribution matter less, which echoes our "selection rule matters little". But it is a smooth-control, linear-model-class setting, not a max-over-actions safety value.
  - Gap versus our claim: no safety objective, no max over actions, no counterfactual-branch comparison.
- **Nagabandi et al., "Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning" (ICRA 2018; arXiv 1708.02596). Label: BACKGROUND/PARTIAL (weak).** [full text]
  - "This on-policy data aggregation (i.e., reinforcement learning) improves performance by mitigating the mismatch between the data's state-action distribution and the model-based controller's distribution [40]."
  - Ablation: "training using mostly the aggregated on-policy rollouts significantly improves performance" over initial random data.
  - Note: this favours DAgger over random-only data, with random being non-directed whole-trajectory data. It says nothing about off-policy branches.
- **Lambert et al., "Objective Mismatch in Model-based RL" (L4DC 2020; arXiv 2002.04523). Label: BACKGROUND.** [full text]
  - "Fig. 5b shows that on-policy data will not lead to a complete dynamics understanding because the grid validation data rapidly diverges."
  - Supports "on-policy data is not full coverage". Also shows model likelihood is a poor proxy for control reward.
- **Berger et al., "Biased Dreams: Limitations to Epistemic Uncertainty Quantification in Latent Dynamics Models" (RLJ 2026; arXiv 2604.25416). [RECENT 2025-26] Label: BACKGROUND.** [full text of the summary page]
  - "ensemble disagreement captures local epistemic uncertainty, [but] it does not reliably reflect global compounding model error accumulated over prolonged RSSM latent rollouts." Also: "an attractor behavior that draws rollouts toward well-supported latent regions, where uncertainty diminishes despite increasing discrepancies from the true environment dynamics."
  - Relevance: explains why ensemble-uncertainty-directed collection need not beat random in latent models.
- **Bhamidipaty et al., "Imperfect World Models are Exploitable" (arXiv 2605.15960, May 2026). [RECENT 2025-26] Label: BACKGROUND.** [full text of abstract]
  - "exploitation is essentially unavoidable on large policy sets", and they "derive a safe horizon within which it can be avoided". Pure theory, no data-collection prescription.
- **Frauenknecht et al., "All Models are Wrong, Knowing Where is Useful" (arXiv 2606.01363, May 2026). [RECENT 2025-26] Label: BACKGROUND.** Position paper: "inaccuracies of the learned dynamics model are typically exploited by the agent". It advocates uncertainty-aware handling (pessimism), not coverage.
- **COPlanner (Wang et al., arXiv 2310.07220; ICLR 2024). Label: BACKGROUND.** [full text of abstract] Combines "conservative model rollouts and optimistic environment exploration". It is a hybrid of pessimism for rollouts plus uncertainty-directed real data. No random-vs-directed ablation was checked.

### Inferences
- The Ross & Bagnell result supports half of our claim: on-policy aggregation alone is insufficient and a separate exploration distribution is needed. It does NOT contradict "the selection rule matters little" and weakly supports it, since DAgger made results robust to the choice of nu.
- Our finding that DAgger of the fallback only reaches 10-23% while counterfactual branches reach 1-3% is an instance of the same phenomenon. The novelty would be in applying it to a max-over-actions safety value (argmax V picks actions the on-policy distribution never contains).
- Reviewers will cite Ross & Bagnell as prior art for "need exploration distribution beyond on-policy". The framing "caution cannot substitute for coverage" is not in that paper.

### Gaps
- I did not find a paper that quantifies DAgger-for-models vs off-policy branching for a learned value or safety filter.
- I did not verify the MBPO "branched rollouts" mechanism beyond known background. It uses model rollouts from real states, not real branches.
- Not read: PILCO-style data collection (Deisenroth & Rasmussen 2011) and PETS (Chua 2018) in detail. They are not cited as evidence here.

---

## Q2. Counterfactual / branch / recovery data augmentation: do any fix a learned safety value or filter better than pessimism?

### Takeaway
Counterfactual-action data is known to fix model/value failures in general RL (CoDA/MoCoDA, WorldSync, GALILEO). Recovery or violation data helps safe RL (Recovery RL, PROCO, HG-DAgger). None of them compares a coverage fix against a pessimism fix for a learned safety VALUE. The explicit "conservatism is not enough, data acquisition is the missing piece" argument appears only in RECON and RENEW (both concurrent, both outside safety values).

### Cited Findings
- **RECON, "Beyond Conservatism: Recoverability-Conditioned Exploration for Model-Based Imitation Learning" (Chen et al., arXiv 2609.33336, 27 Sep 2026). [CONCURRENT] Label: PARTIAL, closest on the "caution vs coverage" thesis.** Local copy: `arxiv_2609.33336.pdf`. [full text]
  - Abstract: "Conservative MBIL mitigates model exploitation during policy optimization, but when real-environment interactions are collected by the same conservative policy, uncertain regions around the expert distribution remain insufficiently sampled."
  - Fig. 1 caption: "Conservative model use is not sufficient for model acquisition."
  - Closing: "Conservative model use determines how an imperfect world model should guide policy optimization, but not where limited interaction should be spent to improve that model."
  - Formal claim: "conservatism controls which model predictions are used, but nominal collection controls recovery error only where the recovery occupancy is covered."
  - Differences from us: model-based imitation learning, not safety values. Uses an explorer policy (on-policy-style), not off-policy counterfactual branches. DMC / MetaWorld / PointMaze, 3 seeds. This is the single nearest statement of "caution cannot substitute for coverage" and must be cited.
- **RENEW (Bhamidipaty, Kochenderfer, Ramamoorthy, arXiv 2607.14180, Jul 2026; RLC 2026 workshop). [CONCURRENT] Label: PARTIAL.** Local copy. [full text]
  - "Prior work addresses this either by collecting more expert demonstrations ... or by conservative algorithms that avoid uncertain regions, which limits generalization." And: "All three approaches [MOPO, MOReL, COMBO] trade coverage for robustness".
  - Repairs exploitation by preference labels over imagined rollouts, not new real data.
  - Appendix G.2 (relevant to selection rule): "At K = 2 ... naive outperforms RENEW on both environments. The state-action spaces are small ... so uniform sampling provides adequate coverage, and ensemble disagreement is a noisier estimator of model error in continuous spaces."
- **WorldSync / WorldEcho (Chen et al., "Do Robotic World Models Really Follow Actions?", arXiv 2608.24885, Aug 2026). [CONCURRENT] Label: PARTIAL (world models, not safety values).** Local copy. [full text]
  - "expert-only AC-WMs follow demonstrated actions reasonably well but exhibit two characteristic failure modes under off-expert actions. They either produce visually plausible yet overly optimistic futures that deviate from the conditioned actions, or lose visual integrity".
  - "Expert demonstrations cover only a narrow subset of feasible action consequences, leaving the off-expert support gap".
  - Fix: "multi-task simulated trajectories that span expert behavior, local perturbations, cross-state replays, policy rollouts, and broad feasible actions". The training set mixes several action distributions. I only skimmed the Table 2 ablation and did not confirm which component of the coverage mix matters.
  - Nearest published statement of "off-policy counterfactual action coverage fixes the world model", but without safety values or DAgger comparison.
- **GALILEO / AWRM (Chen et al., "Adversarial Counterfactual Environment Model Learning", NeurIPS 2023; arXiv 2206.04890). Label: PARTIAL on mechanism.** [full text of abstract]
  - "this approach may catastrophically fail to predict counterfactual action effects due to the selection bias of behavior policies during data collection."
  - Fix: "AWRM incorporates an adversarial policy that exploits the model to generate a data distribution that weakens the model's prediction accuracy, and subsequently, the model is learned under this adversarial data distribution."
  - Caveat: it reweights existing offline data (importance-weight style); no new real branches are collected. It is the closest published analogue to our "value-directed" idea (a model-exploiting adversary), and it is not compared against uniform branch collection in the way I verified.
- **CoDA (Pitis, Creager, Garg, NeurIPS 2020; arXiv 2007.02863) and MoCoDA (Pitis et al., NeurIPS 2022; arXiv 2210.11287). Label: PARTIAL (general RL).** [full text]
  - MoCoDA abstract: "a locally factored dynamics model provably generalizes out-of-distribution to unseen states and actions" and "train an offline RL agent to solve an out-of-distribution robotics manipulation task on which standard offline RL algorithms fail".
  - On random counterfactuals: "Another option is to sample random state-actions from S x A (RAND). While this provides coverage ... there is no guarantee that our locally factorized model generalizes well in RAND." They use model-generated counterfactuals, so they depend on the model's own generalization; our branches are re-labelled by the true simulator.
- **Recovery RL (Thananjeyan et al., RA-L 2021; arXiv 2010.15920). Label: BACKGROUND/PARTIAL (weak).** [full text of abstract]
  - "(1) leveraging offline data to learn about constraint violating zones before policy learning and (2) separating the goals of improving task performance and constraint satisfaction across two policies". Reports 2-20x better violation/success trade-off in simulation. This is safety-critic data (violation zones) plus a recovery policy, not a counterfactual-action fix for a model-based value, and it has no pessimism-vs-coverage ablation.
- **HG-DAgger (Kelly et al., ICRA 2019; arXiv 1810.02890). Label: BACKGROUND.** [full text of abstract] "learns a safety threshold for a model-uncertainty-based risk metric"; "demonstrate improved performance over both DAgger and behavioral cloning". Imitation, not model or value learning.
- **DART (Laskey et al., CoRL 2017; arXiv 1703.09327). Label: PARTIAL for the on- vs off-policy contrast, but for imitation policies.** [full text of abstract]
  - "We propose an off-policy approach that injects noise into the supervisor's policy while demonstrating. This forces the supervisor to demonstrate how to recover from errors."
  - "For high dimensional tasks like Humanoid, DART can be up to 3x faster in computation time and only decreases the supervisor's cumulative reward by 5% during training, whereas DAgger executes policies that have 80% less cumulative reward than the supervisor."
  - Closest published precedent for "off-policy perturbation instead of on-policy aggregation". It is about training-time safety and compute, and the headline claim is about policies, not models or values.
- **PROCO (Xue et al., arXiv 2605.01356, May 2026). [RECENT 2025-26] Label: BACKGROUND/PARTIAL.** [full text of abstract] "performs model-based rollouts to synthesize diverse counterfactual unsafe samples, supporting reliable feasibility identification". Counterfactual unsafe data generated by the learned model plus an LLM-derived cost, in offline safe RL. No true-simulator relabelling and no pessimism-vs-coverage ablation was checked.
- **ExORL (Yarats et al., arXiv 2201.13425; ICLR-W 2022). Label: BACKGROUND (offline-RL empirical, not theory).** [full text] "exploratory data allows vanilla off-policy RL algorithms, without any offline-specific modifications, to outperform or match state-of-the-art offline RL algorithms ... data generation is as important as algorithmic advances for offline RL".
- **Counterexample-Guided Data Augmentation (Dreossi et al., IJCAI 2018; arXiv 1805.06962). Label: BACKGROUND.** Vision; see predator-prey section.
- **Pessimism camp (BACKGROUND):** MOPO, MOReL, COMBO, RAMBO-RL (arXiv 2204.12581: "alternate between optimising the policy and adversarially optimising the model" in order to be conservative). All trade coverage for robustness (RENEW's wording above).

### Inferences
- The thesis "pessimism is not a substitute for coverage" is now in the literature (RECON, RENEW, ExORL), but only for offline MBRL / imitation. For safety VALUES specifically I found no direct statement. The sharp novelty gap is the measured statement for a safety value: pessimism (ensemble / CB / in-data max) fails, off-policy counterfactual branches fix it.
- A reviewer would map "off-policy counterfactual branches" onto DART (off-policy perturbation vs DAgger), CoDA/MoCoDA (counterfactual augmentation), and WorldSync (action coverage expansion). We should differentiate on: (a) true-simulator relabelling of branches, (b) a max-over-actions safety value as the exploited object, (c) a pre-registered null result on selection rules.
- The concurrency risk is real: RECON, RENEW, WorldSync were all posted within 2 months.

### Gaps
- No paper found that applies counterfactual branch data to a learned HJ / safety value and compares to pessimism.
- Did not verify the safety-critical driving scenario-generation literature in detail (LD-Scene, SEAL, ScenePilot, etc.). Search results only; those papers target policy robustness and benchmarks, not learned safety values (UNVERIFIED).
- "Learning from interventions" beyond HG-DAgger was not explored.

---

## Q3. Active learning of dynamics / safety boundaries: do they compare against RANDOM, and did random turn out competitive?

### Takeaway
Mixed. Papers proposing active methods usually beat random sampling, especially when the sampler uses ground-truth structure (boundary / verifier / recoverability) rather than model self-reported uncertainty. Several papers show random is competitive in low-dimensional or well-regularized settings or that the benefit is problem-dependent. This is mostly outside the exact setting of branch selection for a safety value.

### Cited Findings
Directed beats random / uniform:
- **Dallas, Lew, Talbot, DeCastro, Bansal, Subosits, "Boundary Sampling to Learn Predictive Safety Filters via Pontryagin's Maximum Principle" (arXiv 2604.13325, Apr 2026). [RECENT 2025-26] Label: PARTIAL, MOST CONTRARY to "selection rule matters little" in a safety-value setting.** [full text]
  - "Boundary trajectories that barely avoid constraint violation are informative, but rare under typically-used uniform sampling, motivating a principled approach".
  - Learned HJ / control barrier value function. "Across all sample and model sizes, PMP statistically significantly outperforms uniform sampling." At 30k epochs, "failure rate statistically significantly decreases from 0.66 to 0.3 (2-layer) and from 0.50 to 0.45 (3-layer). As we train for longer the gap narrows (achieving 0.08 failure rate)."
  - Caveats: four seeds, one-sided t-test, true dynamics known (the sampler uses PMP on the known model), gap narrows with training. States are sampled; actions are not counterfactually branched.
- **Deglurkar, Smith, Li, Tomlin, "Active Calibration of Reachable Sets Using Approximate Pick-to-Learn" (arXiv 2603.24894). [RECENT 2025-26] Label: PARTIAL.** [full text] "boundary and random active learning require a similar number of samples ... even if the Boundary and Random experiments yield a similar probabilistic guarantee due to using a similar number of samples, the final reachable set learned via the boundary sampling strategy tends to have a better FPR and FNR." Its main baselines are two other calibration methods, not a generic random sampler.
- **OpAx (Sukhija et al., "Optimistic Active Exploration of Dynamical Systems", NeurIPS 2023; arXiv 2306.12371). Label: BACKGROUND.** [full text] "For both, active exploration reduces epistemic uncertainty faster compared to random exploration." Pendulum, 10 seeds. Measures uncertainty, not safety.
- **ActSafe (As et al., ICLR 2025; arXiv 2410.09486). Label: BACKGROUND.** [full text] Compares to UNIFORM ("samples actions uniformly at random"), OPTIMISTIC, and GREEDY baselines on a sparse-reward task; reports "ActSafe outperforms the baselines". This is exploration-for-task, with pessimism used for safety.
- **RECON (Q2).** Ablation: "Random collection improves generic model coverage but spends too little interaction on task-relevant deviations"; uncertainty-only collection "can preserve early learning, but degrades later performance". Directed (gated) > random and > ungated uncertainty. [full text]
- **ExORL (Q2).** "comparing the different exploration algorithms, we see similar performance across each of the three representative algorithms (ICM, Proto, and APS) with each of them being much stronger than random." Directed methods are similar to one another but better than a random policy. Random here means a random-action policy, weaker than random branch points.

Random competitive / small benefit:
- **Chatzikiriakos, Jamieson, Iannelli, "High Effort, Low Gain: Fundamental Limits of Active Learning for Linear Dynamical Systems" (arXiv 2509.11907; AISTATS 2026). [RECENT 2025-26] Label: BACKGROUND.** [full text] "our analysis also uncovers cases where the benefit of experiment design is small even in an oracle setup, indicating that the usefulness of experiment design needs to be evaluated on a problem-specific level." Linear systems; random isotropic Gaussian excitation can be near-optimal.
- **RENEW Appendix G.2** (Q2): uniform sampling adequate in small state-action spaces; ensemble disagreement noisier in continuous spaces.
- **Berger et al. 2604.25416** (Q1): ensemble disagreement unreliable for compounding latent error, which undercuts uncertainty-directed selection in latent models.

The "random is a strong baseline" active-learning literature (supervised classification):
- **Mittal, Tatarchenko, Cicek, Brox, "Parting with Illusions about Deep Active Learning" (arXiv 1912.05361).** [full text of abstract] "Active learning methods are quite sensitive w.r.t. changes in the training procedure like data augmentation. They improve by a large-margin when integrated with semi-supervised learning, but barely perform better than the random baseline." Label: BACKGROUND.
- **Munjal, Hayat, Hayat, Sourati, Khan, "Towards Robust and Reproducible Active Learning Using Neural Networks" (CVPR 2022; arXiv 2002.09564).** [full text of abstract] "under identical experimental settings, different types of AL algorithms (uncertainty based, diversity based, and committee based) produce an inconsistent gain over random sampling baseline" and "under strong regularization, AL methods show marginal or no advantage over the random sampling baseline under a variety of experimental conditions." Label: BACKGROUND.
- **Luth, Bungert, Klein, Jaeger, "Navigating the Pitfalls of Active Learning Evaluation" (NeurIPS 2023; arXiv 2301.10625).** [full text] Abstract: "today's AL literature presents an inconsistent and contradictory landscape". Body: "this simple protocol lets even our random-query baseline exceed the originally reported performance of recent AL methods"; "Munjal et al. [44] demonstrate how adequate HP tuning on a validation set allows a random query baseline to outperform current QMs under their originally proposed HP settings." They also report some methods (BADGE) performed "consistently better or on par with Random". Label: BACKGROUND.
- **Luth et al., "Finally Outshining the Random Baseline" (TMLR 2026; arXiv 2601.13677). [RECENT 2025-26]** [full text of abstract] "existing AL methods are unable to consistently outperform improved random sampling baselines adapted to 3D data". Label: BACKGROUND.
- **Other (UNVERIFIED, search snippet only):** Wagenmaker & Jamieson 2020 (COLT) active excitation; Corso et al. JAIR 2021 survey of black-box safety validation reports uniform random search as the baseline for falsification (arXiv 2005.02979; I did not see a statement of random competitiveness).

### Inferences
- The AL-reproducibility papers are a good citation for the pattern "directed selection gain disappears under a well-tuned baseline". They are classification papers, so they are only an analogy for branch selection in a dynamics/value setting.
- Contrast: gains from directed collection tend to show up when the selector uses ground-truth-confirmed information (PMP boundary, recoverability, verifier counterexamples) and not when it uses the model's own uncertainty (Berger et al.; RENEW's K=2 case). That is consistent with our killed arm using model-reliance as the selection signal. It does not by itself explain why true-confirmed selection would also have failed; we did not test that.
- Our null result is thus consistent with part of the literature (AL-null, linear-systems "small benefit", RENEW small-space), and in tension with Dallas et al. (boundary > uniform, significant) and RECON (directed > random). Quote both sides.

### Gaps
- No paper found that tests directed vs random branch selection for learning a safety value from branched real data.
- ISE (arXiv 2212.04914) and SABRE (arXiv 2210.14492) were checked locally and do not include a random-sampling baseline in the form we need (ISE compares to other safe-BO acquisitions; SABRE to a naive baseline). SafeOpt (Sui et al. ICML 2015) not checked for random baselines. Chen/Herbert/Bansal "learning-aware safety" not read.

---

## Q4. Has anyone measured that the greedy fallback of a learned safety value is worse than doing nothing / holding still?

### Takeaway
I found no clean measurement of "greedy argmax fallback of a learned value is worse than holding still / zero action". Closest are statements of fallback overreaction or filter-induced failures, mostly as caveats rather than controlled comparisons.

### Cited Findings
- **Gameplay Filters (Nguyen, Hsu, Yu, Tan, Fisac, CoRL 2024; arXiv 2405.00846).** [full text] "reward-based RL baselines and ISAACS with avoid-only objective fail almost immediately by overreacting and flipping over. Reach-avoid policies behave more robustly by bringing the robot to a stable stance." And: "We observe that ISAACS control actor is strictly better than [the task policy] and comparable to [the fixed-pose policy]." Label: PARTIAL (weak). It is a learned fallback worse than stable standing for a specific objective, in a quadruped, with the adversary in the loop. Not about data coverage.
- **ISAACS (Hsu, Nguyen, Fisac, L4DC 2023).** [full text] "Using the ISAACS model directly performed well for the most part, but it lacked theoretical guarantees and the safety rate was indeed nonzero." Also: ISAACS critic "1.4% false-safe rate ... contrasts with a more conservative 20.4% false-unsafe rate". Label: BACKGROUND.
- **UNISafe (Seo et al., arXiv 2505.00779; latent-HJ specific, pointer only).** [full text] "the safety filter may behave unpredictably, executing random or overconfident actions or even exacerbating unsafe situations." A prose caveat, not a measurement. Label: BACKGROUND.
- **Verifiable Safety Q-Filters (Li, Hu, Yang, Liu, arXiv 2506.15693).** [full text] The pretrained Q-network "violates constraint satisfaction". Cites "zero-sublevel-set shrinkage". Label: BACKGROUND.

### Inferences
- A controlled "argmax-V fallback vs hold-still / zero-action vs unfiltered" comparison on a nominal-data value would be a novel, citable contribution if it is pre-registered, because the literature does not present it as a measured failure.

### Gaps
- Searches for "worse than unfiltered / no filter" returned nothing measured. I did not search driving or aviation runtime-assurance literature for filter-increases-failure reports.

---

## Predator-prey training loops for learned safety values and certificates (added scope)

### Takeaway
Predator-prey (falsifier or adversary vs value/model) is established in CEGIS-style certificate learning and in adversarial-RL safety synthesis (ISAACS, Gameplay Filters, RARL, MADR). Evidence at matched budget vs random is mostly indirect: it is shown against domain randomization or classical augmentation, rarely against an identically budgeted random-counterexample arm. Over-conservatism is reported often (shrinkage, false-unsafe rate, extra interventions). The distinguishing feature of these loops is that the falsifier's output is confirmed by a true simulator, verifier, or true-dynamics labelling, which our killed value-directed arm did not require. No paper compares the two selection signals directly.

### Cited Findings
Do they beat random at matched budget?
- **RARL (Pinto, Davidson, Sukthankar, Gupta, ICML 2017; arXiv 1703.02702). Label: BACKGROUND.** [full text] "If the adversary is kept fixed, the protagonist could learn to overfit to its adversarial actions. Therefore, instead of using either a random or a fixed-adversary, we advocate generating the adversarial actions using a learned policy". The paper connects this to hard-example mining. The reported gains are against baseline RL and varied-parameter tests, not a matched-budget random-disturbance data arm that I verified.
- **Gameplay Filters (CoRL 2024).** [full text] "The employed [policy] can withstand 150 N from all directions, but the non-game-theoretic counterpart (RARL+DR) is vulnerable to the tugging from the left and can only withstand 43 N. This suggests that DR struggles to capture the worst-case realization of disturbances in a bounded class. This arises from its inherent nature: as the dimension of the disturbance input increases, the likelihood of the random policy simulating the worst-case disturbance decreases exponentially." Also: "domain randomization test is far less challenging (all safety methods excel)". Label: PARTIAL. It is the strongest published statement that random disturbance coverage misses worst-case actions in safety, but it is for disturbance, not for the agent's own counterfactual actions, and not matched-budget.
- **ISAACS (L4DC 2023).** [full text] "ISAACS outperforms single-agent RL (even with DR) by a large margin. DR optimizes against the average among all possible disturbances and is less robust to worst-case realizations." Label: PARTIAL (weak). Matched training budgets were not verified.
- **Counterexample-Guided Data Augmentation (Dreossi, Ghosh, Yue, Keutzer, Sangiovanni-Vincentelli, Seshia, IJCAI 2018; arXiv 1805.06962). Label: PARTIAL for "selection rule matters little".** [full text] Counterexample-guided augmentation: "The average precision and recall improvements for uniform random are 14.43% and 14.56%, for low-discrepancy Halton 16.05% and 14.57%, for cross-entropy 16.11% and 6.18%, and for random with distance constraint 14.95% and 14.26%", versus 4.91% / 4.46% for standard augmentation. So different counterexample generators performed similarly, while all beat classical augmentation. They also report that error-table-directed sampling found 329 misclassifications in 10k iterations vs 103 for pure uniform random. Vision (object detection), not a safety value. Important nuance: all arms there were confirmed misclassifications (true labels), which is exactly the confirmed-counterexample property.
- **Verification-Aided Learning of NN Barrier Functions (Chen et al., arXiv 2403.07308, 2024).** [full text] "CEGIS has been widely applied to NN certificate function synthesis ... Although the counterexamples are believed to be informative and useful for the learner, there is often no guarantee that CEGIS will find a feasible solution even if there exists one." They note "the counterexamples generated by the verifiers are often scarce" and so augment by sampling around them and running PGD. Label: BACKGROUND/PARTIAL (weak). No random-vs-counterexample ablation that I verified.
- **Boundary Sampling via PMP (Dallas et al. 2026).** See Q3: directed boundary samples beat uniform for a learned HJ value (statistically significant, gap narrows). Label: PARTIAL.
- **MADR (Teoh et al., arXiv 2510.18845; updated Aug 2026).** [full text of abstract] MPC-guided adversarial supervision for DeepReach "significantly outperforms state-of-the-art baselines" (Vanilla DeepReach, grid DP, ISAACS, sampling MPC). Label: BACKGROUND. The baseline is uniform self-supervised sampling, plus other methods, at matched iterations per the text I skimmed.
- **Corso et al. survey of black-box safety validation (JAIR 2021; arXiv 2005.02979).** UNVERIFIED beyond the snippet "uniform random search is used as a baseline assessment method for falsification algorithms".

Over-conservatism / collapse:
- **Verifiable Safety Q-Filters (arXiv 2506.15693).** [full text] The paper proposes "a multiplicative Q-network structure to mitigate zero-sublevel-set shrinkage issues" and the conclusion says it alleviates "zero-sublevel-set shrinkage common in prior methods" during counterexample-guided fine-tuning. Label: PARTIAL. This is direct evidence that CEGIS-style loops can shrink the safe set (over-conservatism).
- **ISAACS.** [full text] "This contrasts with a more conservative 20.4% false-unsafe rate".
- **Gameplay Filters.** [full text] "Hsu et al. [38] observe safety overrides 5 times as frequent as those of a least-restrictive HJ filter" for robust rollout filters; they argue worst-case is not intrinsically more conservative. They also say a "suboptimal (not-truly-worst-case) disturbance model may lead the gameplay rollout to erroneously" misjudge safety.
- **CEGDA oscillation / overfitting claim:** UNVERIFIED (search snippet says "adding multiple adversarial samples might produce overfitting or 'oscillations'"; I did not find the word "oscillations" in the extracted text, so do not quote it).

RAMBO-style adversarial model training (pessimism, not data):
- **RAMBO-RL (Rigter, Lacerda, Hawes, NeurIPS 2022; arXiv 2204.12581).** [full text] "alternate between optimising the policy and adversarially optimising the model." It trains an adversarial model to minimise the value, a form of pessimism. Label: BACKGROUND. It is not about safety certificates.
- **GALILEO / AWRM (2206.04890).** An adversarial policy exploits the model and the model is trained on the resulting distribution (offline reweighting). Label: PARTIAL (see Q2). Closest published analogue to a model-exploiting predator.

Is a falsifier that CONFIRMS in the true simulator different from aiming data where the value relies on the model?
- In ISAACS, Gameplay Filters, RARL and MADR, the adversary acts in the true (training) simulator, so every adversarial sample is labelled by true dynamics. Evidence: Gameplay Filters, "co-training a safety controller and a worst-case disturbance through simulated self-play".
- In neural-certificate CEGIS (Chen et al. 2403.07308), counterexamples come from a verifier or PGD on the true condition.
- In GALILEO the adversary exploits the learned MODEL; there is no confirmation step. That makes it closer to our value-directed arm.
- No paper that I found runs both selection signals (model-reliance-directed vs true-confirmed falsifier) at matched budget against random. In that sense our killed arm fits the "model-directed, unconfirmed" family; our counterfactual branches re-labelled by true rollouts fit the "confirmed" family, which makes sense of why selection among confirmed branches matters little.

### Inferences
- There is a plausible reading that the data-side benefit comes from the confirmed relabelling of off-distribution actions (coverage), and the predator's selection adds a second-order gain. CEGDA's finding (different confirmed counterexample generators all similar, all better than baseline augmentation) is the closest published instance of that pattern. This is inference, in a vision setting.
- A reviewer from the CEGIS community will say "this is CEGIS". The defence is the explicit pre-registered null: the predator adds nothing over random confirmed branches at 3-10k branches. We do not know if this holds at smaller budgets.
- Over-conservatism is a documented risk of falsifier loops (shrinkage). Our branch data should be checked for set shrinkage or false-unsafe growth before claiming the fix is free.

### Gaps
- No matched-budget random-counterexample ablation found in neural CBF/Lyapunov CEGIS papers I read (2403.07308, 2506.15693, 2506.13011). I did not read Dawson et al. 2021 (robust neural Lyapunov-barrier), Chang et al. 2019 (neural Lyapunov control), or Zhang et al. 2602.05311 in full.
- Asymmetric self-play for safety data (e.g., Sukhbaatar et al.) was not covered.
- Bayesian-optimization falsification (multi-fidelity BO, arXiv 2409.08097) was seen only via search snippet; its "more efficient than ... baseline methods" claim is UNVERIFIED and I did not check whether random search was among the baselines.

---

## Summary table of closeness labels

| Paper (ID / venue, year) | Authors | Label | Why |
|---|---|---|---|
| Agnostic System ID (1203.1007, ICML 2012) | Ross, Bagnell | PARTIAL (high) | On-policy alone not enough; needs exploration distribution; DAgger-MB robust to the choice of nu |
| RECON (2609.33336, Sep 2026) [CONCURRENT] | Chen et al. | PARTIAL (high) | "Conservative model use is not sufficient for model acquisition"; random vs directed ablation |
| RENEW (2607.14180, Jul 2026) [CONCURRENT] | Bhamidipaty, Kochenderfer, Ramamoorthy | PARTIAL | Pessimism trades coverage; uniform adequate in small spaces |
| WorldSync (2608.24885, Aug 2026) [CONCURRENT] | Chen et al. | PARTIAL | Off-expert action support gap; coverage expansion |
| GALILEO/AWRM (2206.04890, NeurIPS 2023) | Chen et al. | PARTIAL | Counterfactual-action failure from selection bias; model-exploiting adversary; reweighting |
| DART (1703.09327, CoRL 2017) | Laskey et al. | PARTIAL | Off-policy perturbation vs on-policy DAgger (imitation) |
| PMP boundary sampling (2604.13325, Apr 2026) | Dallas, Lew, ..., Bansal | PARTIAL (contrary) | Directed beats uniform for a learned HJ safety value |
| Active calibration of reachable sets (2603.24894) | Deglurkar et al. | PARTIAL | Boundary vs random: similar sample counts, better FPR/FNR |
| Gameplay Filters (2405.00846, CoRL 2024) | Nguyen, Hsu, Yu, Tan, Fisac | PARTIAL | Random disturbances miss worst case; fallback overreaction |
| ISAACS (L4DC 2023) | Hsu, Nguyen, Fisac | PARTIAL (weak) | Predator-prey safety; DR less robust |
| CEGDA (1805.06962, IJCAI 2018) | Dreossi et al. | PARTIAL | Counterexample generators similar; beat classical augmentation |
| CoDA / MoCoDA (2007.02863; 2210.11287) | Pitis et al. | PARTIAL | Counterfactual augmentation fixes OOD actions in offline RL |
| Recovery RL (2010.15920, RA-L 2021) | Thananjeyan et al. | BACKGROUND/PARTIAL | Violation/recovery data for safe RL |
| ExORL (2201.13425) | Yarats et al. | BACKGROUND | Data matters as much as algorithm |
| Nagabandi 2018 (1708.02596); Lambert 2020 (2002.04523) | | BACKGROUND | On-policy aggregation helps; on-policy not complete coverage |
| HG-DAgger (1810.02890) | Kelly et al. | BACKGROUND | Interactive IL |
| Verifiable Q-filters (2506.15693) | Li, Hu, Yang, Liu | BACKGROUND/PARTIAL | Shrinkage in CEGIS fine-tuning |
| Verification-aided NCBF (2403.07308) | Chen et al. | BACKGROUND | CEGIS caveats |
| OpAx (2306.12371); ActSafe (2410.09486) | Sukhija et al.; As et al. | BACKGROUND | Directed > random on uncertainty / task |
| High Effort, Low Gain (2509.11907) | Chatzikiriakos, Jamieson, Iannelli | BACKGROUND | Benefit of experiment design can be small |
| Munjal 2022 (2002.09564); Mittal 2019 (1912.05361); Luth 2023 (2301.10625); Luth 2026 (2601.13677) | | BACKGROUND | Random is a strong AL baseline |
| Bhamidipaty 2026 (2605.15960); Berger 2026 (2604.25416); Frauenknecht 2026 (2606.01363); COPlanner (2310.07220); RAMBO-RL (2204.12581); PROCO (2605.01356) | | BACKGROUND | Exploitation theory; uncertainty limits; pessimism; model-generated counterfactuals |
| RARL (1703.02702) | Pinto et al. | BACKGROUND | Learned adversary vs random/fixed |

Overall: NO SCOOPS found. Highest risk papers for a "this is already known" reviewer comment: Ross & Bagnell 2012, RECON, RENEW, WorldSync, GALILEO, DART, CEGDA.

Local PDFs newly saved to `/common/users/ss5772/bimanual_data/lit/pdfs/` (arxiv_<id>.pdf and .txt): 1912.05361, 2002.09564, 2301.10625, 2605.15960, 2210.11287, 2506.15693, 2402.05279, 2601.21297, 2511.07899, 2411.16959, 2410.13979, 2603.24894, 2606.01363, 2603.23679, 1810.02890, 2310.07220, 2609.33336, 1810.12162, 2003.06082, 2405.00846, 2604.13192, 2403.07308, 1805.06962, 1703.02702, 2506.13011, 2605.07757, 1703.09327, 2201.13425, 2005.02979, 1708.02596, 2601.13677, 2509.11907, 2604.13325, 2510.18845, 2608.24885, 2608.04653, 2206.04890, isaacs_hsu23a.pdf/.txt. (`arxiv_1905.13122` in that folder is NOT Neural Lyapunov Control; I used a wrong ID, ignore it.)

No hardware, robot, or live-service connections were made; this was web reading, local PDF text extraction, and no experiments.
