# Live VLM one-arm-or-two planner on the pick props (2026-10-15) [V: owner read planner_table_live_default.json]

- Setup:
  - Qwen3.5-9B daemon (job 519036), `pick.smoke_planner --backend live --planner vlm+veto`. User-approved VLM use.
  - 5 head-camera proposals.
  - Input: the ego image crop + the measured extent.
- **Result:** the VLM agreed with the physical rule on 3/5. It was confidently wrong on BOTH bimanual objects; the physics veto caught both.
  - crate 29x20x20 cm (true 1.2 kg): "Box fits single arm; right side is closer", mass 0.5 kg.
  - basket 33x21x18 cm: "Box is too narrow for two hands", mass 0.5 kg.
  - Grounding: the bottle was labelled "robot_arm"; the cube was labelled "cup_obj".
  - Mass guesses (0.1–0.5 kg) under-estimate the crate. Under the VLM rule they can only raise the estimate, so they were ignored.
- Latency p50 4.2 s per object (worker thread, cached per object). 0 fallbacks.
- Outputs: /data/local/ss5772/molmospaces/pick/live_vlm/ (planner_table_live_default.json, crops, run.log).
- Meaning: more evidence for the VLM rule (the VLM proposes, the certificate admits). n = 5, a single scene, so this is anecdotal, not a rate.

## 2026-10-16 — Cosmos 3 Edge (4B) on the same 5-object pick test
Setup:
- Served by NVIDIA's `vllm/vllm-openai:cosmos3` image via singularity on rlab1, RTX 4500 Ada 24 GB (`vlm/slurm/cosmos3_serve.sh`).
- The pip plugin cannot load Edge (`cosmos3_edge` is not registered).
- Prompt v2: usable jaw 6 cm, "label may be wrong", mass `[lo, hi]`, contents/fragile/handle.
- Parser takes the LAST JSON object. A truncated reply (finish=length) counts as no answer.
- Harness: `--wait-s 900 --vlm-timeout 300`.

| run | mug 6x5x10 | cube 6x5x7 | crate 29x20x20 | bottle 4x4x22 | basket 33x21x18 | agree | p50 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B (10-15, prompt v1) | 1 ✔ | 1 ✔ | **1 ✘ veto** | 1 ✔ | **1 ✘ veto** | 3/5 | 4.2 s |
| Edge thinking OFF | **2 ✘ veto** | **2 ✘ veto** | 2 ✔ | **2 ✘ veto** | 2 ✔ | 2/5 | 1.8 s |
| Edge thinking ON, 2048 tok (v1 parser) | no answer (cut) | 1 ✔ | 2 ✔ | no answer (cut) | fragment parsed, 1 ✘ veto, mass 210–540 (grams as kg) | — | 17 s |
| Edge thinking ON, 4096 tok | **2 ✘ veto** | 1 ✔ | 2 ✔ (mass [0.75, 1.7]) | no answer (97 s) | 2 ✔ | 3/5 | 8.9 s |

Findings:
- Thinking OFF is degenerate: the same answer ("2 arms, small, 0.3–0.5 kg") for every object.
- Thinking ON gets both large objects right, which Qwen did not. But it misjudged the mug as 2 arms, and it rambles. One object had no answer even at 4096 tokens.
- Labels in thinking mode came back "unknown". Thinking-mode labelling was truncated or unparsed.
- **Parser hazard (fixed):** the greedy `{.*}` grabbed a JSON fragment from unfinished reasoning. It was read as an answer, with grams treated as kg. The check caught it.
  - Fix: parse the last object only, and treat a truncated reply as no answer.
- **Veto logic gap (open, Checker 3/4):** on the crate, the VLM's widened mass hi = 1.7 kg exceeds even the 2-arm limit of 1.5 kg (ASSUMED payload), so the 2-arm plan was vetoed.
  - The code then fell back to the rule's 2-arm plan, which uses its own narrower mass prior.
  - Under "VLM may only widen", nothing is admissible here. The correct output is probe/ask (e.g. a lift test), not the rule plan.
  - The true crate mass is 1.2 kg, so the VLM range [0.75, 1.7] does contain the truth.
- n = 5 objects, 1 scene, 1 seed: examples, not rates.

### 2026-10-16 (later) — Cosmos 3 Nano (16B), thinking requested ON, 4096 budget
Setup:
- rlab2 A100-40GB x2 (TP=2), shared from telecom's hold; image vLLM 0.23.1.
- TP on PCIe needed `NCCL_P2P_DISABLE=1` AND `--disable-custom-all-reduce`.
- Same prompt v2 and harness.

Results:
- **5/5 agree with the rule**, 1 veto, p50 **1.1 s**.
- mug, cube, bottle → 1 arm. crate, basket → 2 arms.
- Labels are generic ("cylinder", "desk_part").
- Basket: mass [1.5, 2.5] kg. The hi value exceeds the 2-arm limit, so the 2-arm plan was vetoed and the rule plan ran. Same open veto gap as Edge.
- The thinking length was not logged (the daemon strips reasoning). 1.1 s suggests little reasoning.

Ranking so far on this 5-object test (examples, not rates): Nano 5/5 > Qwen 3/5 = Edge-thinking 3/5 > Edge-no-think 2/5 (degenerate).
