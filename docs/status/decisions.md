# Decision ledger: user rulings (newest first)

The one list of the user's rulings. A ruling here overrides any doc that disagrees. Add new rulings at the top, with the date, the
ruling in one or two sentences, and the file that records it. Do not delete rows: mark a replaced ruling "replaced by <date>".
Seeded 2026-11-05 from `docs/paper/claims_2026-11-02.md` §1 (rows marked "claims §1") and the code and status files named.
Earlier rulings (before 10-15) are history in `docs/design/MASTER.md`.

## Open (the user has not ruled)

| Since | Question | Where |
|---|---|---|
| 11-02 | Plan-time robust grasp (Holladay et al.) as a baseline, now that the grasp is out of scope. | claims §8 item 2 |
| 11-02 | Hardware facts: NERO joint-torque read and its noise; XL430 squeeze force. | claims §8 item 3; `docs/status/probe_limits_2026-11-05.md` "Open" |
| 11-02 | Revise and approve `proposal_2026-10-27.md` for hidden contents, the weld and the four claims. | claims §8 item 4 |
| 11-05 | Restate the headline, claims 1-3, Lemma 1 and A1-A5 for the 11-05 problem (claims §3). | claims §3 |
| 11-05 | Probe target: which hidden parameters and object priors the demo uses; lift-off detection. | `docs/status/probe_limits_2026-11-05.md` "Open" |
| 11-01 | Theta ranges and plan menu for the pot (section 3 of the demos file; grip-force parts are obsolete since 11-02). | `docs/paper/demos_2026-11-01.md` §3-4 |

## Rulings

| Date | Ruling | Source |
|---|---|---|
| 11-05 | **Plans executed in a run are authored by the Codex GPT models, not by us.** The VLM/LLM proposer writes the high-level plan (segments, goals, recoveries, fallback); our low-level planner and checker work it out and verify it. Hand-written task plans (carry.PLANS, the closed-chain goal offsets, stored backups) are test fixtures only. | this ledger |
| 11-05 | **Torque limit = the MJCF numbers** (actuator forcerange, 100 Nm on every arm joint): "use mjcf numbers ofc that is real". Replaces the 16 Nm MIT-range default of 11-03 (kept behind a switch for old runs). Payload rating 3 kg (datasheet) is unchanged. | this ledger; `world_model/v2/nero/config.py` |
| 11-05 | **Try all three demos (pot, tray, box)** to find where the method works and where it does not, before choosing the main one. | this ledger |
| 11-05 | **A plan is a sequence of segments.** Within each segment we show the guarantees and the recoveries. (Reconciles 10-26 "short horizon + safe stop" with 11-03 "plan once, then track": the plan is made once, as segments, each with its certificate and backup.) | this ledger |
| 11-05 | Memory holds no papers: literature notes live in `docs/status/literature_notes.md`. | this ledger |
| 11-05 | **Problem restated.** The VLM is the orchestrator: belief range B, high-level plans (task steps, recoveries for B, a fallback). A low-level planner works the plans out; a checker verifies every plan before execution. Probes are safe by design and run without the checker (ASSUMED); the probe is truth (ASSUMED). Rules 1-4. Replaces the 11-02 problem statement and Θ_0 = hull(Θ_prior ∪ Θ_VLM). | claims §1, §3 |
| 11-05 | Establish and write down what the arms can probe within their limits. | `docs/status/probe_limits_2026-11-05.md` (opening USER quote) |
| 11-05 | Count the payload rating in the unsafe set U and run the same worlds for both methods (ours vs AgentChord-style). | `world_model/v2/nero/contents/same_worlds.py:2` |
| 11-05 | Test the probe-ladder "free" variants 1, 2 and 4 separately. | `world_model/v2/nero/contents/probe_variants.py:1` |
| 11-04 | Shared autonomy is no longer part of the thesis problem. | claims §1 |
| 11-03 | Retire the scripted carry plan for a bimanual kinodynamic planner. Owner picks D1-D7 under "do them": plan once then track; closed-chain path + TOPP-RA + FL-MPC; analytic SRS IK; plan against the prior box (sampled); proposer gives task graph + goals only; tray and box first, then the pot. | `docs/status/2026-11-03_handoff_notes.md` "User decisions 11-03 (planner)" |
| 11-03 | Recovery planning is a later, separate algorithm. | owner memory `recovery-planner-later-2026-11-03` (not yet in a repo doc) |
| 11-03 | The NYU gripper is the only gripper ("remove it entirely"); the AgileX gripper is not our hardware. Rigid weld for every grasp. XL430 PWM cap active. | `world_model/v2/nero/config.py` (GRIPPER comment); `2026-11-03_handoff_notes.md` "Done 11-03", "AgileX removal" |
| 11-03 | Cooperative impedance runs inside FL-MPC (`ctl="coopimp_safe"`, with table and joint rules); squeeze target 0 N; goal tolerance 3 cm. | `2026-11-03_handoff_notes.md` "Done 11-03" |
| 11-03 | Probing, belief shrinking and active learning are mechanisms adopted from others' work, not our claim. Replies: do the task, report what it does, what it shows, open questions; no unasked claim or novelty statements. | `AGENTS.md` "Replies" |
| 11-02 | **The grasp is not our problem.** No claw rows; the claw twist cap and the breakable weld are dropped. Failures that count come from how the arms carry the load. Hardware gripper = YOR's (XL430). | claims §1 |
| 11-02 | **Probing is a means, not a claim.** Soundness needs every narrowing to be sound. Claim 5 dropped. | claims §1 |
| 11-01 | Grasp = rigid weld once the claw closes (ASSUMED); grasp work stopped. Focus: hidden contents (tray of mugs, closed box onto a shelf, the pot). | claims §1 |
| 11-01 | Main demo = covered pot with two side handles; pan = candidate. **Reopened** (see Open). | claims §1, §8 item 1 |
| 11-01 | The arms are AgileX NERO 7-DoF, not Piper. Limits live only in `world_model/v2/nero/config.py`. | `AGENTS.md` "Robot fidelity" |
| 11-01 | Compute rules: never touch jenga; westeros CPU only, at most 12 processes; no westeros GPUs; no Slurm submits. 11-03 overrides: GPUs and Slurm for the VLM tests, ilab4 CPU for the AgentChord rerun. Current status: ask the user (`AGENTS.md` "Parallel sweeps" still suggests `cdx-agent --cpu-job`). | owner memory `feedback-compute-rules-2026-11-01` (not in a repo doc) |
| 10-29 | Bounds and assumptions rule: label every bound MEASURED / MODEL / ASSUMED; never shrink the world set to model sensing; same-world comparisons. | `AGENTS.md` "Bounds and assumptions" |
| 10-27 | Framing: safe kinodynamic planning and control as AgentChord's successor; second pillar = task-preserving recovery. | claims §1 |
| 10-26 | Claim narrowed to a parameter-dependent task-ready set; recovery = any action that passes admission. "A way back always exists" is a lemma, not a claim. | claims §1 |
| 10-26 | Base only as a recovery option and slow transport; manipulation is the core; people avoidance is borrowed. | claims §1 |
| 10-26 | Proposer = open-weight model only. | claims §1 |
| 10-26 | Baselines: AgentChord, estimate-then-act, plan-time robust grasp, fixed-target backup. | claims §1 |
| 10-26 | The planner plans safely: certified short receding horizon ending in a safe stop, then replan. | claims §1 |
| 10-26 | Parallel sweeps: size the pool from idle cores; pin threads to 1. | `AGENTS.md` "Parallel sweeps" |
| 10-24 | No reset of the belief to its original expectation. No Shield fallback. | claims §1 |
| 10-23 | No fairness assumption on people. | claims §1 |
| 10-15 | Learned world model / latent HJ dropped. Direction = classical planning + formal checkers + VLM proposals. | claims §1 |
