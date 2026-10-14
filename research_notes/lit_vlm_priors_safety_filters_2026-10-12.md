# Literature check: VLM priors + hard real-time safety filters (2026-10-12)

Method: arXiv abstract pages for all IDs; arXiv HTML full text for PSS, Feng, Phys2Real, Peng, Real2Sim-BT, Tabbara, AlphaAdj, 2609.27816; PDF text for Brunke (HTML 404). Items marked [abs] were judged from the abstract / landing page only. Read-only, no downloads of code/data, no logins, no hardware or live-service contact (public web pages only).

Legend: Y = yes, P = partial, N = no, ? = could not determine from what was read. (a) models/calibrates uncertainty in LLM/VLM outputs; (b) VLM-derived PHYSICAL parameters; (c) hard real-time safety filter; (d) studies failures when VLM/LLM output is wrong; (e) restricts how a language prior may change the constraint set; (f) bimanual; (g) margin dynamics / infeasibility as a failure mode.

## Existence / corrections
All items EXIST except where noted:
- **#16 arXiv 2603.06921 is NOT a "CBF deadlock with pedestrians" paper.** It is "CN-CBF: Composite Neural Control Barrier Function for Robot Navigation in Dynamic Environments" (Derajic, Bernhard, Hoenig). Not about margins/infeasibility. Do not cite for that purpose. I did not find the intended paper.
- #16 arXiv 2505.05502 is "Feasibility Analysis and Constraint Selection in Optimization-Based Controllers" (Rousseas, Lee, Dimarogonas, Panagou), not literally "constraint relaxation / catastrophic infeasibility", but it is on-topic for QP infeasibility.
- #16 Xiao/Cassandras/Belta 2303.09403 = "Learning Feasibility Constraints for Control Barrier Functions", ECC 2023.
- #5 Safety Chip: arXiv page lists NO venue; ICRA 2024 not confirmed from the page.
- #9 Dean et al.: arXiv page lists no venue; "CoRL 2020" not confirmed from the page (I believe it is correct but did not verify).
- #13 GF-VLA is on engrXiv (DOI 10.31224/6733, posted 2026-04-01), not arXiv; authors Gu, Cheng, Gao.
- #6 Phys2Real: VLM prior is ONLY center-of-mass along one axis (not friction/mass).
- #3 Feng: arXiv 2511.05889 has no venue listed (preprint).

## Table

| # | Paper (URL) | Exists | a | b | c | d | e | f | g | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Predictive Semantic Safety (https://arxiv.org/abs/2609.34356) | Y | P | N | Y | N | P | N | P | Split conformal calibrates position error of the whole "observation-to-motion prediction procedure" (VLM+motion model), delta=0.05, N_cal=1000. Filter+policy 50 Hz, OSQP, VLM requests 1 s on an A100. No VLM-failure sensitivity study; invalid prediction -> whole-plane occupancy fallback. "Hypothesis spread does not justify reducing the calibrated radius" and joint update accepted only if identities, interval, membership and full filter feasibility pass (gating, not formal widen-only). QP "without slack variables"; feasibility is a design condition, not studied as failure. Go1 quadruped, MuJoCo only. |
| 2 | Semantically Safe Robot Manipulation (https://arxiv.org/abs/2410.15185) | Y | N | N | Y | N | P | N | P | LLM labels objects -> semantic constraints + "level of caution" selecting class-K functions/weights; filter at 45 Hz on Franka FR3; cup-of-water case reduces EE angular speed 75%. No uncertainty, no VLM-wrong study. Pose constraint "softened ... less prone to infeasibility". Single arm. |
| 3 | From Words to Safety (https://arxiv.org/abs/2511.05889) | Y | N | N | Y | P | P | N | P | LLM fills template fields (buffer, v_max); parser right 52/60; limitations: failures "propagate into incorrect safety set definitions" but not analysed. Filter 2-10 ms, "remain real-time". Output limited to predefined templates/3 categories. "If Vsafe<=0, no safe backup exists"; over-conservative segmentation -> "robot freezing". TurtleBot 2, navigation only. |
| 4 | Plug in the Safety Chip (https://arxiv.org/abs/2309.09919) [abs] | Y | N | N | N | N | N/A | N | N | LTL constraint module prunes LLM-agent actions; symbolic/plan level. |
| 5 | Safety Guardrails for LLM-Enabled Robots / RoboGuard (https://arxiv.org/abs/2503.07885) [abs] | Y | N | N | N | Y | P | N | N | Root-of-trust LLM grounds predefined rules into LTL specs; control synthesis "minimally overriding user preferences"; unsafe plans 92% -> <3% under jailbreaks. Plan level, not control rate. |
| 6 | Phys2Real (https://arxiv.org/abs/2510.11689) | Y | P | Y | N | P | N | N | N | Inverse-variance fusion (Eq. 4) of VLM prior (self-reported sigma) and ensemble online estimate. Paper says VLM spread "can cause poor uncertainty estimates (i.e., the VLM may be confidently wrong)"; conformal prediction only as future work. VLM gives CoM along stem only; friction/mass future work. No safety filter; RL pushing, ICRA 2026. |
| 7 | Friction-Aware Safety Locomotion (https://arxiv.org/abs/2409.09845) | Y | N | Y | N | P | N | N | N | VLM+RAG friction estimate fed to RL policy; "When the predicted CoF exceeds the actual value, the policy becomes overly aggressive"; no uncertainty; no CBF; VLM ~5 s, policy 50 Hz. |
| 8 | Real2Sim via Active Perception w/ VLM Behavior Trees (https://arxiv.org/abs/2601.08454) | Y | N | P | N | P | P | N | N | VLM chooses which parameters to measure; values come from contact probing, not VLM priors. BT leaf predicates act as "deterministic safety filter" against VLM hallucination, no CBF, no rates. Single Franka. |
| 9 | Measurement-Robust CBFs (https://arxiv.org/abs/2010.16001) [abs] | Y | N | N | P | N/A | N | N | N | CBF robust to perception-model error; Segway sim; no language. |
| 10 | Uncertainty-aware Latent Safety Filters (https://arxiv.org/abs/2505.00779) [abs] | Y | P | N | N | Y | N | N | N | Conformal-calibrated threshold on world-model epistemic uncertainty; HJ in augmented state; "overconfidently misclassifying risky OOD situations as safe". Not language, no real-time claim. CoRL 2025. |
| 11 | Adaptive Shielding w/ HJ Reachability (https://proceedings.mlr.press/v283/lu25a.html) [abs] | Y | N | N | P | P | P | N | N | Rough dynamics -> local model mismatch -> shield becomes more conservative; TurtleBot 2. L4DC 2025, PMLR 283:1257-1270. |
| 12 | KnowNo (https://arxiv.org/abs/2307.01928) | Y | Y | N | N | Y | N | N | N | Conformal prediction over LLM planner choices; LLMs "often produce confident answers that are wrong"; asks for help. Plan level. CoRL 2023 oral. |
| 13 | Graph-Fused VLA ... Dual-Robot Control via CBFs (https://engrxiv.org/preprint/view/6733) [abs] | Y | N | N | ? | N | N | Y | N | Dual-arm; LLM flags unsafe spatial relations; CBF enforces semantic + geometric constraints. Page reports no calibration, rates or failure analysis. |
| 14 | General Language-Conditioned Latent Safety Filters (https://arxiv.org/abs/2608.00315) | Y | N | N | P | P | P | N | P | 20 Hz; "current VLMs are not yet sufficiently reliable to serve as failure functions"; fixed constraint templates; "does not guarantee satisfaction of language-specified constraints"; if no feasible safe action, falls back to mean actor action; filter can stall arm. Single manipulator. |
| 15 | Dynamic CBF Regulation w/ VLMs (AlphaAdj) (https://arxiv.org/abs/2603.21142) | Y | N | N | Y | P | P | N | N | VLM risk score -> CBF alpha; controller 60 Hz, VLM ~2 Hz / 695 ms; staleness gate, clipping to [0,1]; alpha_final = min(alpha_vlm, alpha_cap) so geometry caps permissiveness (one-directional cap). "Further calibration" is future work; no wrong-VLM experiments; no feasibility analysis of alpha jumps. Ground robot. |
| 16a | CN-CBF (https://arxiv.org/abs/2603.06921) | Y (wrong topic) | N | N | N | N | N | N | N | Neural CBF navigation; NOT the deadlock paper. |
| 16b | Feasibility Analysis & Constraint Selection (https://arxiv.org/abs/2505.05502) | Y | N | N | N | N | N | N | Y | Necessary/sufficient feasibility conditions for linear constraints; constraint selection. |
| 16c | Learning Feasibility Constraints for CBFs (https://arxiv.org/abs/2303.09403) | Y | N | N | N | N | N | N | Y | QP feasibility hard for high relative degree; learns extra feasibility HOCBF. ECC 2023. |
| 17 | The Safety Filter: A Unified View (https://arxiv.org/abs/2309.05837) | Y | N | N | Y | N | N | N | P | Survey, Annual Review of Control, Robotics, and Autonomous Systems. Background only. |

## Additional relevant papers found (item 18)
| Paper (URL) | a | b | c | d | e | f | g | Evidence |
|---|---|---|---|---|---|---|---|---|
| PHYSICIAN: Verification of Physical Grounding in VLA Models (https://neurips.cc/virtual/2026/163700) [poster page only; no arXiv found] | N | Y | P | P | N | N | N | Gemini VLM estimates friction/mass; sim vs video-world-model divergence as "empirical uncertainty signal"; CBF layer filters actions; success 12%->89% under deceptive low friction; verification <550 ms (so not a hard-RT loop); no calibration; AlphaX 4WD. Single author (Abdulsalam), NeurIPS 2026 WiML poster. CLOSEST on P1. |
| Safe Multi-Robot Coordination via VLM-LLM Reasoning and Reachability (https://arxiv.org/abs/2609.27816) | N | N | P | P | P | N | N | Detection confidence + reprojection error inflate obstacle radius (r_perc); zonotope reachability gate; 30 Hz decisions; no calibration/conformal; missing confidence = max uncertainty; quadruped + vehicle. Semantic + state uncertainty in one gate (relevant to P3). |
| CBVF under Partial Observability via Conformal Prediction (https://arxiv.org/abs/2608.13819) [abs] | N | N | Y | N | N | N | N | Conformal bound on state-estimator error -> QP safety filter; no language. |
| Safety filtering of robotic manipulation under environment uncertainty (https://arxiv.org/abs/2509.12674) [abs] | N | N | N | P | N | Y | N | Physics-simulation safety filter with uncertain mass/friction, probing actions, simulated bimanual task; not real-time claimed; no VLM. |
| SafeBimanual (https://arxiv.org/abs/2508.18268) [abs] | N | N | N | N | P | Y | N | VLM selects safety cost functions to guide diffusion for bimanual policies (test-time guidance, not a hard filter). |
| SAFER, Safety Aware Task Planning via LLMs (https://arxiv.org/abs/2503.15707) [abs] | N | N | P | P | N | N | N | Multi-LLM safety agent + CBFs for multi-robot task planning. |

Not found in any search: a paper with VLM-elicited friction prior + conformal uncertainty sets + hard-RT filter; any explicit "language prior may only widen" rule; any treatment of VLM/perception-driven margin jumps as a QP-infeasibility failure mode. Search was finite (web search is not exhaustive); absence is not proof.

## BibTeX (CONFIRMED papers)
```bibtex
@article{kim2026pss,
  title   = {Predictive Semantic Safety: From Visual Physical Reasoning to Safety-Critical Control},
  author  = {Kim, Taekyung and Fradi, Salem and Dai, Yanning and Ostaszewski, Mateusz and Schmidhuber, J{\"u}rgen},
  journal = {arXiv preprint arXiv:2609.34356},
  year    = {2026}
}
@article{brunke2025semantically,
  title   = {Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards},
  author  = {Brunke, Lukas and Zhang, Yanni and R{\"o}mer, Ralf and Naimer, Jack and Staykov, Nikola and Zhou, Siqi and Schoellig, Angela P.},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {10}, number = {5}, pages = {4810--4817}, year = {2025}
}
@article{feng2025words,
  title   = {From Words to Safety: Language-Conditioned Safety Filtering for Robot Navigation},
  author  = {Feng, Zeyuan and Zhang, Haimingyue and Bansal, Somil},
  journal = {arXiv preprint arXiv:2511.05889},
  year    = {2025}
}
@article{yang2023safetychip,
  title   = {Plug in the Safety Chip: Enforcing Constraints for {LLM}-driven Robot Agents},
  author  = {Yang, Ziyi and Raman, Shreyas S. and Shah, Ankit and Tellex, Stefanie},
  journal = {arXiv preprint arXiv:2309.09919},
  year    = {2023},
  note    = {Venue (ICRA 2024) not confirmed from arXiv page}
}
@article{ravichandran2025roboguard,
  title   = {Safety Guardrails for {LLM}-Enabled Robots},
  author  = {Ravichandran, Zachary and Robey, Alexander and Kumar, Vijay and Pappas, George J. and Hassani, Hamed},
  journal = {arXiv preprint arXiv:2503.07885},
  year    = {2025}
}
@inproceedings{wang2026phys2real,
  title     = {Phys2Real: Fusing {VLM} Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation},
  author    = {Wang, Maggie and Tian, Stephen and Swann, Aiden and Shorinwa, Ola and Wu, Jiajun and Schwager, Mac},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
  note      = {arXiv:2510.11689}
}
@inproceedings{peng2025friction,
  title     = {Friction-Aware Safety Locomotion for Wheeled-legged Robots using Vision Language Models and Reinforcement Learning},
  author    = {Peng, Bo and Baek, Donghoon and Wang, Qijie and Ramos, Joao},
  booktitle = {IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year      = {2025},
  note      = {arXiv:2409.09845}
}
@article{adami2026real2sim,
  title   = {Real2Sim via Active Perception with Behavior Trees Automatically Generated by {VLMs}},
  author  = {Adami, Alessandro and Zudaire, Sebastian and Carli, Ruggero and Falco, Pietro},
  journal = {arXiv preprint arXiv:2601.08454},
  year    = {2026}
}
@article{dean2020mrcbf,
  title   = {Guaranteeing Safety of Learned Perception Modules via Measurement-Robust Control Barrier Functions},
  author  = {Dean, Sarah and Taylor, Andrew J. and Cosner, Ryan K. and Recht, Benjamin and Ames, Aaron D.},
  journal = {arXiv preprint arXiv:2010.16001},
  year    = {2020},
  note    = {CoRL 2020 per requester; not on arXiv page}
}
@inproceedings{seo2025uncertainty,
  title     = {Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures},
  author    = {Seo, Junwon and Nakamura, Kensuke and Bajcsy, Andrea},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2025},
  note      = {arXiv:2505.00779}
}
@inproceedings{lu2025adaptive,
  title     = {Safe Learning in the Real World via Adaptive Shielding with {Hamilton-Jacobi} Reachability},
  author    = {Lu, Michael and Gosain, Jashanraj and Sang, Luna and Chen, Mo},
  booktitle = {Proceedings of the 7th Annual Learning for Dynamics \& Control Conference},
  series    = {PMLR}, volume = {283}, pages = {1257--1270}, year = {2025}
}
@inproceedings{ren2023knowno,
  title     = {Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners},
  author    = {Ren, Allen Z. and Dixit, Anushri and Bodrova, Alexandra and Singh, Sumeet and Tu, Stephen and Brown, Noah and Xu, Peng and Takayama, Leila and Xia, Fei and Varley, Jake and Xu, Zhenjia and Sadigh, Dorsa and Zeng, Andy and Majumdar, Anirudha},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2023},
  note      = {arXiv:2307.01928}
}
@misc{gu2026gfvla,
  title        = {Graph-Fused Vision-Language-Action Models for Semantically Safe Dual-Robot Control via Control Barrier Functions},
  author       = {Gu, Jiajun and Cheng, Weihao and Gao, Longsen},
  year         = {2026},
  howpublished = {engrXiv preprint 6733},
  doi          = {10.31224/6733}
}
@article{tabbara2026language,
  title   = {Towards General Language-Conditioned Latent Safety Filters},
  author  = {Tabbara, Ihab and Yang, Yuxuan and Sibai, Hussein},
  journal = {arXiv preprint arXiv:2608.00315},
  year    = {2026}
}
@article{chen2026alphaadj,
  title   = {Dynamic Control Barrier Function Regulation with Vision-Language Models for Safe, Adaptive, and Realtime Visual Navigation},
  author  = {Chen, Jeffrey and Chandra, Rohan},
  journal = {arXiv preprint arXiv:2603.21142},
  year    = {2026}
}
@article{rousseas2025feasibility,
  title   = {Feasibility Analysis and Constraint Selection in Optimization-Based Controllers},
  author  = {Rousseas, Panagiotis and Lee, Haejoon and Dimarogonas, Dimos V. and Panagou, Dimitra},
  journal = {arXiv preprint arXiv:2505.05502},
  year    = {2025}
}
@inproceedings{xiao2023feasibility,
  title     = {Learning Feasibility Constraints for Control Barrier Functions},
  author    = {Xiao, Wei and Cassandras, Christos G. and Belta, Calin A.},
  booktitle = {European Control Conference (ECC)},
  year      = {2023},
  note      = {arXiv:2303.09403}
}
@article{hsu2023safetyfilter,
  title   = {The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems},
  author  = {Hsu, Kai-Chieh and Hu, Haimin and Fisac, Jaime Fern{\'a}ndez},
  journal = {Annual Review of Control, Robotics, and Autonomous Systems},
  year    = {2024},
  note    = {arXiv:2309.05837}
}
@misc{abdulsalam2026physician,
  title        = {{PHYSICIAN}: Verification of Physical Grounding in Vision-Language-Action Models},
  author       = {Abdulsalam, Nafisat},
  year         = {2026},
  howpublished = {NeurIPS 2026 poster (WiML); https://neurips.cc/virtual/2026/163700}
}
@article{dwedar2026multirobot,
  title   = {Safe Multi-Robot Coordination via {VLM-LLM} Reasoning and Reachability Analysis},
  author  = {Dwedar, Mohamed and Hafez, Ahmad and Jesser, Alexander and Alanwar, Amr},
  journal = {arXiv preprint arXiv:2609.27816},
  year    = {2026}
}
@article{jahanshahi2026cbvf,
  title   = {Control Barrier--Value Functions under Partial Observability: Safety Guarantees via Conformal Prediction},
  author  = {Jahanshahi, Niloofar and Chen, Mo},
  journal = {arXiv preprint arXiv:2608.13819},
  year    = {2026}
}
@article{johansson2025safetyfiltering,
  title   = {Safety filtering of robotic manipulation under environment uncertainty: a computational approach},
  author  = {Johansson, Anna and Lindmark, Daniel and Wiberg, Viktor and Servin, Martin},
  journal = {arXiv preprint arXiv:2509.12674},
  year    = {2025}
}
@article{deng2025safebimanual,
  title   = {{SafeBimanual}: Diffusion-based Trajectory Optimization for Safe Bimanual Manipulation},
  author  = {Deng, Haoyuan and Guo, Wenkai and Wang, Qianzhun and Wu, Zhenyu and Wang, Ziwei},
  journal = {arXiv preprint arXiv:2508.18268},
  year    = {2025}
}
@article{khan2025safer,
  title   = {Safety Aware Task Planning via Large Language Models in Robotics},
  author  = {Khan, Azal Ahmad and Andrev, Michael and Murtaza, Muhammad Ali and Aguilera, Sergio and Zhang, Rui and Ding, Jie and Hutchinson, Seth and Anwar, Ali},
  journal = {arXiv preprint arXiv:2503.15707},
  year    = {2025}
}
```
