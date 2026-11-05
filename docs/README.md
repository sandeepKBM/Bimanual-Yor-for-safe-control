# docs/ index

The one list of every document under `docs/`, with its status. Built 2026-11-05. Keep it up to date when you add, supersede or
archive a doc. If two docs disagree: user rulings in `docs/status/decisions.md` win, then `docs/paper/claims_2026-11-02.md`, then the
code for facts about the code. Simulation only; hardware rules are in `AGENTS.md`.

## Read first

1. `AGENTS.md` (repo root): how to operate here; robot limits point to `world_model/v2/nero/config.py`.
2. `docs/paper/claims_2026-11-02.md`: the four claims; §3 = the problem statement (user, 11-05); §8 = open decisions.
3. `docs/status/decisions.md`: user rulings, newest first, and the questions still open.
4. `docs/status/2026-11-03_handoff_notes.md`: planner decisions D1-D7 and the next-step order.
5. `docs/status/checker_design_2026-11-05.md`: the checker design (not built; build after planner step 2).
6. `docs/paper/proposal_2026-10-27.md`: the formulation basis (needs revision; read its status banner first).

## Current work, in one paragraph

Hidden-contents carry in `world_model/v2/nero/contents/`: a tray of 2/4/6 mugs (hidden fill), a closed box onto a shelf (hidden mass
and centre of mass) and the pot. The grasp is a rigid weld (ASSUMED) with the NYU gripper. Probe ladder: `contents/probe.py`
(narrows the total mass only); limits: `contents/probe_limits.py`. Planner stack: `world_model/v2/stack/` (ScriptedPlanner is the
default; ClosedChainPlanner is opt-in via `--planner closed_chain` or `BIMANUAL_PLANNER`). Same-world comparison, ours vs the
AgentChord-style executor: `contents/same_worlds.py`, raw output in `/data/local/ss5772/contents/same_worlds*_2026-11-05/` (no doc
yet). AgentChord's own dual-arm pour task cannot run on this stack today: it needs one object per arm, a pour skill and a liquid model.
The main demo for the decisive test is open (claims §8 item 1).

## All documents

Status: **current** = use it; **superseded by X** = use X instead; **history** = record only, do not act on it.

### docs/paper

| Path | Purpose | Status | Date |
|---|---|---|---|
| `paper/claims_2026-11-02.md` | claims, evidence, scope, open decisions; §3 problem statement | **current** (headline, claims 1-3, Lemma 1, A1-A5 not yet restated for §3: user) | 11-02, §3 11-05 |
| `paper/demos_2026-11-01.md` | demo set and test-world method | current for the test-world method; main demo pending (claims §8 item 1); claim-5 and claw parts obsolete | 11-01 |
| `paper/proposal_2026-10-27.md` | thesis proposal: formulation, assumptions, baselines, metrics, references | current as formulation basis; **needs revision** for 11-01..11-05 (banner lists the superseded parts) | 10-27 |
| `archive/paper/claims_2026-10-27.md` | five claims, pan demo | superseded by `claims_2026-11-02.md` | 10-27 |
| `archive/paper/claims_2026-10-26.md` | claims C1-C6, Demo C | superseded by `claims_2026-11-02.md` | 10-26 |
| `paper/theorems_T1_T2_draft_2026-10-16.md` | T1 soundness, T2 monotonicity proofs (base rows) | history; proof techniques reusable for claim 1 / Lemma 1 | 10-16 |
| `paper/theorems_T3_T4_draft_2026-10-16.md` | T3 completion, T4 recovery existence (base carry) | history; proof techniques reusable for claims 2 / 3 | 10-16 |
| `paper/experiment_plan_2026-10-16.md` | experiments E1-E7 | superseded by `claims_2026-11-02.md` | 10-16 |
| `paper/YOR_Thesis_Full_Idea_and_Demonstration_Plan_2026-10-15.docx` | earlier thesis statement, demos A-F | superseded by `claims_2026-11-02.md` | 10-15 |
| `paper/Bimanual_Thesis_Literature_Journey_2026-10-13.docx` | literature narrative | history | 10-13 |
| `archive/paper/draft_2026-10-12/main.tex`, `main_v1.tex`, `main_v2.tex` | paper draft "Confidently Wrong" (VLM priors, perception) | history | 10-12 |
| `archive/paper/figs/zono_explainer.py`, `.png` | zonotope explainer figure | history | 10-12 |

### docs/status

| Path | Purpose | Status | Date |
|---|---|---|---|
| `status/decisions.md` | ledger of user rulings and open questions | **current** | 11-05 |
| `status/2026-11-03_handoff_notes.md` | planner decisions D1-D7, next steps, open questions | **current** (banner lists updates since 11-03) | 11-03 |
| `status/checker_design_2026-11-05.md` | task-agnostic checker design; Rules 1-4 as checker calls | **current**; design only, not built | 11-05 |
| `status/probe_limits_2026-11-05.md` | what the NERO arms can probe from joint torques | **current** (MODEL; conditional on an ASSUMED torque-read error) | 11-05 |
| `status/lit_probing_belief_2026-11-03.md` | literature on probing and online belief estimation | current (shallow search; adopted mechanisms, not a claim) | 11-03 |
| `archive/status/2026-11-03_handoff_auto.md` | machine-generated handoff snapshot | history | 11-03 |
| `status/coffee_tray_approach_2026-11-02/` (`.tex`, `.pdf`, `agentchord_plan.dot/.pdf`) | coffee-tray status note, AgentChord plan graph | history (scripted plan retired 11-03) | 11-02 |
| `status/pan_skills_rebuild_2026-10-30.md` | pan skills rebuild (mink, RoboTwin) | history | 10-30 |
| `status/pan_constraint_layer_vs_vizcoast_2026-10-28.md` | pan: our constraint layer vs VIZ-COAST | history (10-28 headline superseded inside) | 10-28 |
| `status/pan_grasp_two_arm_2026-10-28.md` | pan: one claw vs two, grasp proposer | history (grasp out of scope) | 10-28 |
| `status/2026-10-22_owner_notes.md` | owner notes, TAMP/gatekeeper | superseded by `claims_2026-11-02.md` | 10-22 |
| `archive/status/2026-10-22_handoff_auto.md` | machine handoff snapshot | history | 10-22 |
| `archive/status/2026-10-21_owner_notes.md` | owner notes, TAMP/gatekeeper | superseded by `claims_2026-11-02.md` | 10-21 |
| `archive/status/2026-10-21_handoff_auto.md` | machine handoff snapshot | history | 10-21 |
| `status/2026-10-17_tamp_status.md` | TAMP layer status | superseded by `claims_2026-11-02.md` and the 11-03 handoff notes | 10-17 |
| `archive/status/2026-10-17_handoff_auto.md` | machine handoff snapshot | history | 10-17 |
| `status/2026-10-16_report.md` | report and plan (10-16 frame) | superseded by `claims_2026-11-02.md` | 10-16 |
| `status/finetune_study_2026-10-16.md` | VLM fine-tuning study | history (VLM fine-tune not active) | 10-16 |
| `status/prior_art_2026-10-15.md` | prior-art web check | history (10-15 framing); rows still the prior-art list; claims §7 is newer | 10-15 |
| `status/observations_2026-10-15_pick_vlm.md` | live VLM pick test | history (small n) | 10-15 |
| `status/observations_2026-10-12.md` | lab notebook, teleop/perception/language | history | 10-12 |
| `status/molmospaces_integration_plan.md` | house-scale scene plan | history | 10-05 |
| `status/mpc_fl_abstraction_options.md` | FL-MPC constraint-abstraction options | history (FL-MPC moved to `world_model/v2/flmpc/` 11-03) | 10-05 |
| `status/pin_cup_fix_2026-10-08.md` | pin_cup carrier fix | history | 10-08 |
| `status/stale_inventory_2026-10-08.md` | stale-data inventory after the pin_cup fix | history | 10-08 |
| `status/vdirect_f_handoff_2026-10-01.md` | learned-WM falsifier handoff | history (learned WM dropped 10-15) | 10-01 |

### docs/design (all Sept-Oct; none current)

| Path | Purpose | Status | Date |
|---|---|---|---|
| `design/MASTER.md` | old single source of truth (cooperative safety + learned WM) | superseded by `claims_2026-11-02.md`; history | 09-29 |
| `design/world_facts_interface_2026-10-22.md` | world-facts interface proposal | superseded; idea used in `world_model/v2/stack/` | 10-22 |
| `design/teleop_safeset_direction.md` | teleop safe-set direction | superseded (learned HJ dropped 10-15, teleop 11-04) | 09-18 |
| `design/baselines.md` | FL-MPC cup-collect baselines | history | 09-16 |
| `design/comparison_results.md` | FL-MPC cup-collect comparison results | history | 09-21 |
| `design/positioning.md` | FL-MPC cup-collect positioning | history | 09-21 |
| `design/comparison_plan.md` | comparison plan vs NeHMO/HaMMaR | history | 09-14 |
| `design/collision_fidelity.md` | collision audit of the old cup-collect model | history | 09-14 |
| `design/mpc_formulation.md` | mpc7 / JointMPCv7 formulation | superseded (mpc7 defunct; FL-MPC) | 09-08 |
| `design/method.md` | Piper-era method | superseded by `MASTER.md`, itself superseded | 09-06 |
| `design/state_spaces.md` | state spaces for the HJ / learned-WM layers | superseded (learned WM dropped 10-15) | 09-06 |

### docs/runbooks and top level

| Path | Purpose | Status | Date |
|---|---|---|---|
| `runbooks/slurm.md` | sbatch rule: generous `--time`, self-release | current as a general rule; who may submit where: `AGENTS.md` + user's compute rules | 09-06 |
| `runbooks/slurm_rules_vendor_skill.md` | vendor Slurm rules copy | superseded for node choice; history | 09-06 |
| `offload.md` | Box uploader + Slurm gateway adapter | reference; network actions need the user's approval | 09-21 |
| `README.md` | this index | current | 11-05 |
