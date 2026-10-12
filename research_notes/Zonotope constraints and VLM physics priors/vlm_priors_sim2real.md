# VLM/LLM-derived physical priors followed by online learning of the sim-to-real gap

Researched 2026-10-08. Web research only. No robot, device or private service was contacted.
Tags: **VERIFIED** = read on the arXiv abstract, arXiv HTML, arXiv API record or the project page. **SUMMARY** = search snippet or a secondary summary only.
Coverage: about 30 searches and fetches. Sources checked: arXiv, the arXiv API, the IPRL publications page and project pages. Google Scholar was **not** checked.

## Q1. Which paper is "a VLM-based way to find a prior, then the arm learns the sim-to-real gap" (believed to be from the Bohg lab), and which Bohg-lab papers come closest?

### Takeaway
The paper is almost certainly **Phys2Real** (arXiv **2510.11689**, ICRA 2026). It comes from Stanford, but **not from Bohg's IPRL**. The authors are Mac Schwager's MSL group with Jiajun Wu. I found no Bohg-authored paper from 2023 to 2026 that uses a VLM or LLM for physical priors and then adapts on the real robot. Bohg co-publishes often with Schwager, which may explain the mix-up.

### Cited Findings
**Rank 1: Phys2Real (best match)**
- **VERIFIED.** Title: "Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation." Authors: Maggie Wang, Stephen Tian, Aiden Swann, Ola Shorinwa, Jiajun Wu, Mac Schwager. v1 posted 13 Oct 2025, v2 posted 6 Mar 2026. Comments field: "Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026." — [arXiv abs 2510.11689](https://arxiv.org/abs/2510.11689)
- **VERIFIED.** Affiliations: all Stanford except Ola Shorinwa (Princeton). The project page does not mention Bohg or IPRL. MSL lists it as "wang2026phys2real." Code: github.com/phys2real/phys2real. — [Project page](https://phys2real.github.io/); [MSL bibliography](https://msl.stanford.edu/bibliography/wang2026phys2real.html)
- **VERIFIED (abstract).** Phys2Real is a real-to-sim-to-real RL pipeline with three parts: (1) 3D Gaussian-splat geometric reconstruction, (2) VLM-inferred prior distributions over physical parameters, and (3) online estimation of physical parameters from interaction data. The policies are conditioned on interpretable physical parameters. VLM predictions are refined with online estimates using ensemble-based uncertainty quantification. — [arXiv abs](https://arxiv.org/abs/2510.11689)
- **VERIFIED (HTML v2).** The prior comes from **GPT-5**, which estimates the vertical centre of mass from V=5 views, queried Q=5 times per image. The prior mean is the mean across prompts. σ_vlm is the **mean of the VLM's self-reported uncertainties**, not the spread of its answers, because the spread gives poor uncertainty ("the VLM may be confidently wrong"). — [arXiv HTML v2](https://arxiv.org/html/2510.11689v2)
- **VERIFIED (HTML v2).** Online part: an RMA-style (Rapid Motor Adaptation) ensemble of M=10 adaptation models predicts the parameter from a 10-step observation-action history. Ensemble variance gives the epistemic uncertainty. Each member outputs aleatoric uncertainty, trained with a Gaussian NLL loss. The fusion is inverse-variance weighting: θ̂ = (θ_vlm/σ²_vlm + θ_rma/σ²_rma)/(1/σ²_vlm + 1/σ²_rma). This is the BLUE under an independence assumption. — [arXiv HTML v2](https://arxiv.org/html/2510.11689v2)
- **SUMMARY (search extract of the paper).** Early in an episode the fused estimate leans on the VLM prior. As contact information builds up, the ensemble's epistemic uncertainty shrinks and the estimate converges toward the true parameter. — [arXiv PDF](https://arxiv.org/pdf/2510.11689)
- **VERIFIED.** Hardware and tasks: a 6-DoF UFactory xArm with a cylindrical pusher does planar pushing. Actions are x-y position deltas. Tasks are a T-block with its centre of mass at the top or bottom, and a hammer with off-centre mass. — [arXiv HTML v2](https://arxiv.org/html/2510.11689v2)
- **VERIFIED.** Results against domain randomization (DR): bottom-weighted T-block 100% vs 79% success; top-weighted T-block 57% vs 23%; hammer 15% faster. The ablations say both the VLM prior and the interaction information are needed. — [arXiv abs](https://arxiv.org/abs/2510.11689). **Small discrepancy:** the project page gives the DR baseline on the top-weighted T-block as **24%**, not 23%. — [Project page](https://phys2real.github.io/)
- **VERIFIED.** Stated limitations:
  - Scaling to many parameters is hard.
  - VLM uncertainty could be calibrated with "conformal prediction or next-token prediction probabilities" (future work).
  - The mesh pipeline suits roughly symmetric objects.
  - The VLM prior matters less in contact-rich tasks that give a dense adaptation signal.
  - There are **no safety constraints, safe exploration or uncertainty sets**.
  — [arXiv HTML v2](https://arxiv.org/html/2510.11689v2)

**Rank 2: Scan, Materialize, Simulate (SMS). Stanford ASL, VLM prior but no online learning**
- **VERIFIED.** "Scan, Materialize, Simulate: A Generalizable Framework for Physically Grounded Robot Planning." Authors: Amine Elhafsi, Daniel Morton, Marco Pavone (Stanford and NVIDIA). arXiv 2505.14938, posted 20 May 2025. Phys2Real cites it as "Elhafsi et al. (2026)": VLM estimates used to initialize simulation parameters, with no online refinement. — [arXiv API record / HTML](https://arxiv.org/html/2505.14938v1); [Phys2Real HTML](https://arxiv.org/html/2510.11689v2)
- **VERIFIED.** Pipeline: Gaussian splatting, then a VLM query per object for material class, density, friction and restitution, then physics simulation for action optimization. It is **open-loop**. The paper says: "we query material properties, optimize actions in simulation, and execute them without incorporating feedback." It names system identification and online parameter updates as future work. The VLM estimates get no uncertainty handling. Tasks: billiards on a Franka FR3, and quadrotor landing. — [arXiv HTML](https://arxiv.org/html/2505.14938v1). **Conflict:** a search snippet says GPT-4, while the HTML extraction says GPT-4o. — [arXiv PDF](https://arxiv.org/pdf/2505.14938)
- **VERIFIED (arXiv listing).** Daniel Morton has co-authored with Bohg: "Deformable Cargo Transport in Microgravity with Astrobee" (2505.01630; Morton, Antonova, Coltin, Pavone, Bohg). — [arXiv API query au:Bohg](http://export.arxiv.org/api/query?search_query=au:Bohg+AND+abs:simulation&sortBy=submittedDate&sortOrder=descending&max_results=40)

**Rank 3 and below: Bohg/IPRL papers. None uses a VLM physical prior with online gap learning.**
- **VERIFIED (IPRL publications page, 2024 to 2026).** None of the listed papers matches. The nearest are Sim-to-Real ones: SimToolReal, Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration, and Get a Grip. VLM-related ones include Scaffolding Dexterous Manipulation with VLMs, Breaking Lock-In and CHORUS. Physics-related ones are DexMOTS and Causal-PIK. — [IPRL publications](https://iprl.stanford.edu/publications)
- **VERIFIED (arXiv API listing; gists only = SUMMARY for content).** These are the closest Bohg papers:
  - "Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration" (2504.12609; Lum, Lee, Liu, Bohg; CoRL 2025). A real-to-sim-to-real pipeline from one RGB-D human video. **No VLM physical prior.**
  - "Scaffolding Dexterous Manipulation with Vision-Language Models" (2506.19212; de Bakker, Hejna, Lum, …, Bohg, Sadigh). The VLM gives keypoints and 3D trajectory scaffolds for sim RL, **not physical parameters.**
  - "SimToolReal" (2602.16863). Procedural tools and sim-to-real RL, no VLM prior.
  — [arXiv API query](http://export.arxiv.org/api/query?search_query=au:Bohg+AND+abs:simulation&sortBy=submittedDate&sortOrder=descending&max_results=40)
- **VERIFIED (arXiv listing).** Bohg's older real-to-sim work estimates physical parameters **without** VLMs: "A Bayesian Treatment of Real-to-Sim for Deformable Object Manipulation" (2112.05068; Antonova, Yang, Sundaresan, Fox, Ramos, Bohg) and "DiffCloud" (2204.03139). — [arXiv API query au:Bohg physical](http://export.arxiv.org/api/query?search_query=au:Bohg+AND+abs:physical&sortBy=submittedDate&sortOrder=descending&max_results=40)
- **VERIFIED (arXiv listing).** Bohg and Schwager have co-authored "Get a Grip" (2410.23701) and "Breaking Lock-In" (2604.23121). — [arXiv API query](http://export.arxiv.org/api/query?search_query=au:Bohg+AND+abs:simulation&sortBy=submittedDate&sortOrder=descending&max_results=40)

### Inferences
- Phys2Real matches every element of the description: a VLM prior, an arm interacting, the sim-to-real gap closed online, and Stanford. The lab attribution is the only mismatch. The user probably merged "Stanford manipulation" with Bohg. The shared Stanford ecosystem makes this easy: Bohg–Schwager co-authorships, Jiajun Wu, and Morton's link to Bohg.
- Nuance on "the arm learns the gap": in Phys2Real the online part is RMA-style **inference** of a physical parameter by an adaptation ensemble, fused with the prior. The paper describes it as trained like RMA, which points to training in simulation. It is not described as gradient updates or residual-dynamics learning on the real robot. So it estimates a low-dimensional parameter online. It does not learn a general residual model.
- Ranking: (1) Phys2Real is an exact functional match. (2) SMS has the VLM prior but no online learning. (3) Among Bohg papers, Human2Sim2Robot (real-to-sim-to-real) and the 2021 Bayesian real-to-sim work have the right structure but no VLM prior.

### Gaps
- Google Scholar was not checked; it usually blocks automated fetches. The arXiv API searches on "Bohg" with "physical" or "simulation" in the abstract may miss papers whose abstracts use other wording. A Bohg workshop paper or non-arXiv paper on this topic cannot be ruled out.
- I could not confirm whether Phys2Real's adaptation ensemble is trained only in simulation. That is inferred from the "RMA" framing.

## Q2. Survey of related non-Bohg work: prior source, what adapts online, robot and task, safety and uncertainty

### Takeaway
The field splits into three kinds of work. **VLM/LLM priors used open-loop:** PhysObjects, SMS, NeRF2Physics, PhysQuantAgent, DrEureka. **VLM-seeded system identification that is offline or one-off:** RigPI, Adami et al., Agentic Real2Sim. **Online or residual gap learning with no VLM prior:** ASID, TRANSIC, RialTo, PhysTwin, residual physics. Only Phys2Real joins a VLM prior to online adaptation with explicit uncertainty fusion.

### Cited Findings
**A. Physically grounded VLMs and LLM or VLM property estimation (prior only)**
- **PhysObjects / "Physically Grounded Vision-Language Models for Robotic Manipulation"** (arXiv 2309.02561; Jensen Gao, Bidipta Sarkar, Fei Xia, Ted Xiao, Jiajun Wu, Brian Ichter, Anirudha Majumdar, Dorsa Sadigh; ICRA 2024). **VERIFIED.** The prior comes from fine-tuning a VLM on crowd-sourced and automated physical-concept annotations of household objects. It plugs into an LLM planner and raises real-robot task success. **Nothing adapts online. No safety constraints.** — [arXiv API record](http://export.arxiv.org/api/query?id_list=2309.02561). That the VLM is InstructBLIP is **SUMMARY**, taken from Phys2Real's related-work section. — [Phys2Real HTML](https://arxiv.org/html/2510.11689v2)
- **NeRF2Physics / "Physical Property Understanding from Language-Embedded Feature Fields"** (2404.04242; Zhai, Shen, Chen, Wang, Wang, Wang, Guan, Shenlong Wang; CVPR 2024). **VERIFIED.** An LLM proposes candidate materials. Zero-shot kernel regression over a CLIP-embedded point cloud then gives dense mass, friction and hardness. No robot adaptation and no safety. — [arXiv API record](http://export.arxiv.org/api/query?id_list=2404.04242)
- **PhysQuantAgent** (2603.16958; Yokomizo, Miyanishi, Gang, Kurita, Inoue, Iwasawa; March 2026). **VERIFIED.** VLM mass estimation from RGB-D video using three visual-prompting aids (detection, scale estimation, cross-section generation), plus a new benchmark, VisPhysQuant. Perception only. — [arXiv API record](http://export.arxiv.org/api/query?id_list=2603.16958)
- **ViTacPhys** (2608.21355; Aug 2026). **VERIFIED.** Estimates mass, friction class and stiffness from human visuo-tactile demonstrations, using vision-language semantic priors. Grasping success is 95.0% on in-distribution objects and 83.4% on out-of-distribution objects. No online sim-to-real gap learning. — [arXiv API record](http://export.arxiv.org/api/query?id_list=2608.21355)

**B. LLM-guided sim-to-real configuration (zero-shot)**
- **DrEureka** (2406.01967; Ma, Liang, Wang, Wang, Zhu, Fan, Bastani, Jayaraman; RSS 2024). **VERIFIED.** An LLM writes the reward function and the DR distributions.
  - **RAPP** (Reward-Aware Physics Prior): sweep each parameter in sim and keep the values where the initial policy still meets a success criterion. The min and max feasible values become the bounds.
  - The LLM then picks which parameters to randomize and chooses ranges within the RAPP bounds.
  - The "safety" part is only a **safety instruction in the reward prompt** ("steady and stable… minimal action rate").
  - It is **fully zero-shot, with no real-world feedback loop.**
  - Robots: Unitree Go1, a LEAP hand doing cube rotation, and a yoga-ball balancing task.
  — [arXiv abs](https://arxiv.org/abs/2406.01967); [arXiv HTML](https://arxiv.org/html/2406.01967)

**C. VLM-seeded system identification or real-to-sim (offline or one-off; not continual online)**
- **RigPI: "Dynamic Parameter Identification of Rigid Body via VLM-Seeded Differentiable Simulation"** (2606.25212; Xincheng He, Rongrong Zhang, Wei Jiang, Wenqiang Xu; v1 23 Jun 2026). **VERIFIED.**
  - The VLM provides "informed initialization and a **constrained search space**."
  - F/T and motion data then refine the inertial and friction parameters through a two-stage optimization in differentiable simulation. This avoids physically implausible solutions.
  - Tested in real-robot experiments on revolute and prismatic objects.
  - Comments field, verbatim: "Accepted to … IROS 2027". This is likely a typo for 2026, so treat the venue as unconfirmed.
  - No safety constraints are mentioned.
  — [arXiv abs](https://arxiv.org/abs/2606.25212)
- **"Real2Sim via Active Perception with Behavior Trees Automatically Generated by VLMs"** (2601.08454; Adami, Zudaire, Carli, Falco; Padova, ABB, Poliba; Jan 2026). **VERIFIED.**
  - GPT-4o or Gemini 1.5 Pro works out which physical parameters are missing. It then generates a behavior tree (BT) of sensing primitives. The VLM outputs a plan, not parameter values.
  - A Franka Panda measures mass by quasi-static lifting (external wrench from joint torques). It finds height with move-until-contact and estimates friction by pushing.
  - The results are written into MuJoCo XML.
  - The BT acts as a "**deterministic safety filter**": kinematic-feasibility checks, torque thresholds and resource locks abort anomalous VLM commands.
  - There is **no formal uncertainty quantification**, and no sim-to-real policy transfer or online adaptation.
  — [arXiv HTML v2](https://arxiv.org/html/2601.08454v2)
- **Agentic Real2Sim** (2607.19190; Guanxiong Chen + 26 co-authors; July 2026). **VERIFIED (abstract).** Vision-language agents turn real interaction recordings into runnable simulations, including inferring physical parameters. Covers rigid, deformable and humanoid scenes, and supports policy fine-tuning on the converted episodes. The abstract describes no online adaptation and no safety. — [arXiv abs](https://arxiv.org/abs/2607.19190)
- **SMS** (above): VLM prior, open-loop. — [arXiv HTML](https://arxiv.org/html/2505.14938v1)

**D. Active system identification, digital twins and residual or online gap learning (no VLM prior)**
- **ASID: "Active Exploration for System Identification in Robotic Manipulation"** (2404.12308; Memmel, Wagenmaker, Zhu, Yin, Fox, Gupta; Apr 2024). **VERIFIED.** Starts from an initial, possibly inaccurate simulator. It designs exploration policies that collect informative real-world data, then identifies parameters such as articulation and mass and transfers the result. The prior is a simulator, not a VLM. No safety constraints in the abstract. — [arXiv API record](http://export.arxiv.org/api/query?id_list=2404.12308)
- **PhysTwin** (2503.17973; Jiang, Hsu, Zhang, Yu, Shenlong Wang, Yunzhu Li; Mar 2025). **VERIFIED.** Inverse physics from sparse interaction videos reconstructs the geometry and physical parameters of **deformable** objects, which then support model-based planning. Offline, no VLM prior, no safety. — [arXiv API record](http://export.arxiv.org/api/query?id_list=2503.17973)
- **RialTo: "Reconciling Reality through Simulation"** (2403.03949; Torne, Simeonov, Li, Chan, Chen, Gupta, Agrawal). **VERIFIED.** Builds a digital twin from a little real data and fine-tunes imitation policies with RL in it. Reports more than 67% robustness gains. No VLM physical prior. — [arXiv API record](http://export.arxiv.org/api/query?id_list=2403.03949)
- **TRANSIC** (2405.10315; Jiang, Wang, Zhang, Jiajun Wu, Fei-Fei; CoRL 2024). **VERIFIED.** Residual policies learned from **human online corrections** close the sim-to-real gap in contact-rich furniture assembly. No VLM prior. — [arXiv API record](http://export.arxiv.org/api/query?id_list=2405.10315)
- **Robot Learning from a Physical World Model** (2511.07416; Mao, …, Guibas, Guizilini, Zhou, Yue Wang; Nov 2025). **VERIFIED.** Video generation is grounded through physical world reconstruction, followed by object-centric residual RL. Zero-shot, with no real robot data. — [arXiv API record](http://export.arxiv.org/api/query?id_list=2511.07416)
- **Contact-Aware Neural Dynamics** (2601.12796; Jing, …, Abbeel, Xiaolong Wang, Sha Yi; Jan 2026). **VERIFIED.** Implicit sim-to-real alignment: a tactile, contact-aware neural model refines simulated states from real observations. No VLM. — [arXiv abs](https://arxiv.org/abs/2601.12796)
- **Sim-to-Real of Soft Robots with Learned Residual Physics** (2402.01086, ETH SRL). **SUMMARY.** A residual force network learns the gap between the simulator and the real soft robot from sparse markers. — [arXiv HTML](https://arxiv.org/html/2402.01086v1)
- **Bayesian sim calibration** (NPDR, BayesSim). **SUMMARY.** Posterior inference over simulator parameters from "coarse prior ranges" plus a little real data. The priors are hand-set, not from a VLM. — [NPDR, PMLR](https://proceedings.mlr.press/v164/muratore22a/muratore22a.pdf); [BayesSim](https://www.alphaxiv.org/overview/1906.01728v1)

### Inferences
- **Online adaptation from a VLM prior:** Phys2Real is the only one, through inverse-variance fusion. SMS states it as future work. RigPI and Adami et al. are one-shot system identification seeded or orchestrated by a VLM. DrEureka is zero-shot.
- **Form of the prior:** a point estimate (SMS); a mean plus a self-reported σ (Phys2Real); a bounded search box (RigPI); a box from sim feasibility, then chosen by an LLM (DrEureka); a plan of what to measure (Adami). RigPI's VLM box and DrEureka's RAPP bounds are already **set-valued priors**. That is the natural hook for zonotope or interval uncertainty sets.
- The Bayesian real-to-sim literature, including Bohg's own 2021 work, takes coarse hand-set prior ranges. Swapping in a VLM prior is an obvious and so far apparently unclaimed change. Phys2Real does the Gaussian, non-Bayesian-filter version of it.

### Gaps
- Some venues are unconfirmed in the arXiv metadata I read: ASID (possibly ICLR 2024), PhysTwin (possibly ICCV 2025), RialTo (possibly RSS 2024). These are not verified here.
- I did not read the full papers for ASID, PhysTwin, TRANSIC or PhysObjects, so method details beyond the abstracts are not verified.
- PhysGen and URDFormer / Digital Cousins (image-to-sim assets) were not checked.

## Q3. Does any work combine a VLM/LLM physical prior with safe exploration or set-based / uncertainty-aware constraints while learning the gap?

### Takeaway
**I found none.** No paper joins a VLM/LLM physical prior to formal safe exploration (safe BO, CBF or robust MPC, reachability or zonotope sets, chance constraints) while learning the sim-to-real gap online. The closest pieces are separate:
- Phys2Real: uncertainty-aware fusion, but no constraints.
- RigPI: a VLM-bounded parameter box, but offline and with no safety.
- Adami et al.: heuristic torque-threshold and feasibility aborts, but no uncertainty model.
- DrEureka: a safety instruction in the reward plus feasible DR bounds, but zero-shot.

Safe-learning methods do use priors and sets, but those priors are hand-set or meta-learned, never from a VLM.

### Cited Findings
- **VERIFIED.** Phys2Real discusses no explicit safety constraints, safe-exploration protocol or uncertainty sets. It lists better VLM calibration (conformal prediction or token probabilities) as future work. — [arXiv HTML v2](https://arxiv.org/html/2510.11689v2)
- **VERIFIED.** In RigPI, the VLM gives a "constrained search space" for identification only, with no control-safety use. — [arXiv abs](https://arxiv.org/abs/2606.25212)
- **VERIFIED.** In Adami et al., the BT is a "deterministic safety filter" with torque thresholds and kinematic-feasibility checks. There is "no formal uncertainty quantification or bounds"; safety comes from reactive aborts. — [arXiv HTML v2](https://arxiv.org/html/2601.08454v2)
- **VERIFIED.** DrEureka's safety is a prompt instruction in reward design. RAPP bounds the DR ranges by sim feasibility. There is no real-world adaptation. — [arXiv HTML](https://arxiv.org/html/2406.01967)
- **VERIFIED.** SMS says outright that it does not handle VLM-estimate uncertainty and is open-loop. — [arXiv HTML](https://arxiv.org/html/2505.14938v1)
- **SUMMARY.** "Meta-Learning Priors for Safe Bayesian Optimization" (Rothfuss et al., CoRL 2022, arXiv 2210.00762) meta-learns safety-compliant GP priors from offline data, not from language. — [arXiv HTML](https://arxiv.org/html/2210.00762v3); [PMLR](https://proceedings.mlr.press/v205/rothfuss23a.html)
- **VERIFIED (abstract).** Safe Stochastic Explorer (2602.00868; Shinde, Hirsch, Yip, Herbert; Jan 2026) learns unknown safety functions online with GPs and gives probabilistic bounds on violations, including safe interaction with unknown objects. **No VLM prior.** — [arXiv abs](https://arxiv.org/abs/2602.00868)
- **SUMMARY.** "Data-Driven Reachability Analysis Using Matrix Zonotopes" (Alanwar et al., L4DC 2021, arXiv 2011.08472) over-approximates reachable sets of unknown systems from noisy data. Later extensions add partial model knowledge or bounding-set assumptions. — [arXiv PDF](https://arxiv.org/pdf/2011.08472); [L4DC listing](https://mlanthology.org/l4dc/2021/alanwar2021l4dc-datadriven)
- **SUMMARY.** "Neural Barriers" (arXiv 2610.05542) combines high-order robust adaptive CBFs with hybrid online residual learning and online conformal prediction of uncertainty. **No VLM prior.** — [arXiv HTML](https://arxiv.org/html/2610.05542v1)
- **SUMMARY.** A 2025 survey of safe contact-rich learning says shields, CBFs and reachability "typically rely on conservative uncertainty sets." It also says current VLM-based models "typically have weak uncertainty estimation." — [arXiv 2512.11908](https://arxiv.org/pdf/2512.11908)

### Inferences
- **Open niche:** take a VLM physical prior, calibrate it (for example with conformal prediction, which Phys2Real names as future work) into a **set** (an interval box or zonotope over mass, CoM and friction). Use that set as the initial uncertainty set in robust or set-based constraints (zonotope reachability, robust or adaptive CBF, tube MPC). Then **shrink the set online** with interaction data through set-membership or Bayesian updates while acting safely. I found no paper that does this full loop. Phys2Real's own admission that "the VLM may be confidently wrong" is the motivation: a VLM-derived set must be calibrated or inflated before it can serve as a hard-constraint set.
- RigPI's VLM box and DrEureka's RAPP feasible ranges are existing precedents for VLM-derived **set** priors. Neither is used for control safety.

### Gaps
- Search coverage is limited, at roughly 30 queries and fetches, and Google Scholar was not checked. A 2026 workshop paper or preprint that combines the two cannot be ruled out. It should be re-checked before any novelty claim.
- I did not verify the zonotope-reachability literature beyond snippets. That area is presumably covered by the sibling "zonotope constraints" research thread.
