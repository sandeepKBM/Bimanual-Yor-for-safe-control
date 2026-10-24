# VLM fine-tuning study (option 2), 2026-10-16

Two Sonnet agents did this study. The owner spot-checked the Cosmos training doc, the Unsloth VRAM figure, and the sim smoke records.
The VLM is a PROPOSER: tuning it improves completeness (fewer vetoes and probes), never safety.

## Recipes
- **Cosmos 3: full fine-tune only for the reasoner.**
  - cosmos-framework docs/training.md says LoRA is "VFM only".
  - The Edge reasoner is "only 2B" and the recipes target 4-8x H100/GB200. Not a first target.
- **Qwen3.5-9B: bf16 LoRA at about 22 GB** (Unsloth docs, which also confirm vision SFT). It fits one A100-40GB.
  - ms-swift supports Qwen3-VL LoRA and multimodal GRPO.
- **Distillation precedent:** a teacher's checker-filtered traces, then SFT of a student. LocalNav 2606.27871 and Embodied-Reasoner 2503.21696; abstracts only.

## Sim ground truth (agent smoke: /data/local/ss5772/ft_study/gen_smoke.py, smoke_out/)
- 21 local houses (val 0-19, 541). About 150 hand-sized free assets across 50 categories (904 instances). About 1.5 s/sample/process.
- **Masses are explicit but per-category constants and often wrong** (verified):
  - Book_13: 2.51 kg for 20.7x28x2.5 cm.
  - Mug 0.07 kg; basketball 3.6 kg; candle 0.5 kg.
- Contents, fragile and handle are absent, and every object is empty.
- **So: do NOT train mass on sim labels.**
- Correction: the `pick/props.py:4` note "h5 has no free bodies" holds only for the TeleopSim pick scene. Compiling val_5 gives 40 free joints.

## Spec-writing data (the VLM's job in the pipeline)
- **Today's grader is weak.** spec_eval only checks compile + nominal margin > 0 + 2-3 hard-coded must-violate states.
- **Better grader:** gold specs generated from the scene entity table; score = sign agreement of candidate vs gold margins over many sampled states, plus compile.
- **Then:** rejection-sampled SFT; add GRPO only if SFT plateaus (precedent: SafeGen-LLM 2602.24235).

## PDDLStream
- Not in the repo. caelan/pddlstream is GPL-3 and Py3 + Fast Downward; no code changes since 2023, so vendor and pin it.
- **SS-Replan is the closest template:** a detect action, belief streams and plan-tail reuse. Its beliefs are particles with a probability threshold.
- **TAMPURA is not a drop-in:** hand-written abstraction, LAO*, 20-200 s per step.
- **Delta we add:**
  - set-valued belief objects;
  - for-all test streams (the checkers);
  - the backup-to-task-ready test;
  - probe value;
  - a replanning loop with set updates;
  - a VLM-spec -> problem compiler and validator.

## Recommendation
1. Fine-tune for spec writing first: Qwen3.5-9B LoRA on graded gold-spec data.
2. Physical-property priors from real-image data: PhysObjects (Gao et al. 2023); availability and license still to verify. Not sim masses.
3. Cosmos stays an off-the-shelf comparison.

## Teacher eval, 2026-10-16 (80 held-out records: 64 arms + 16 naming; unseen categories and houses 10-13)
Data: /data/local/ss5772/molmospaces/teacher_eval/*_eval_set.jsonl. Scored by `vlm/codex_eval.py --score`.

| model | parse | arms vs rule | naming | fragile / handle vs table | mass log-IoU vs table | p50 |
|---|---|---|---|---|---|---|
| Codex gpt-5.6-luna (high) | 1.00 | 0.61 | 0.44 | 0.91 / 0.97 | 0.42 | 5.1 s |
| Cosmos3-Nano (thinking on, TP=1) | 1.00 | 0.63 | 0.63 | 0.98 / 0.98 | 0.26 | 2.5 s |
| Cosmos3-Edge (thinking on) | 0.23 | 0.17 | 0.44 | — | 0.20 | 10.7 s (max 105) |

**Arms disagreements:** all Codex ones were rule = 2 → model = 1, mostly thin books. The models say "pinch the edge". The rule only checks
top-down footprint grasps, so the RULE is the weak point (same gap as rim/handle grasps; this belongs to Checker 3's grasp model).

**Caveats:**
- The table (mass, fragile, handle) is a hand ASSUMPTION, so these columns show agreement, not accuracy.
- Naming fails on small or distant objects.
- n = 80.

**Ops lesson:** `ssh host "setsid nohup srun ... &"` BLOCKED until the remote step ended (1 h idle Nano). Background the LOCAL ssh instead.
