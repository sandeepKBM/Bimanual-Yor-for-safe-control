# Cup-task safety-stack eval harness (SCAFFOLD + SMOKE)

**See `RESULTS_DOES_WM_HELP.md` for the current (fixed) SPILL-scenario
8-seed/config result with the REAL trained WM monitor -- it supersedes the
collision-only smoke table below, which never exercised a real spill
(`spilled=False` everywhere) and is kept only as scaffold history.**

Answers: does adding a semantic (world-model) monitor on top of the
geometric HJ-reachability safety layer help on the bimanual cup task? This
harness is config-driven so the placeholder modules below can be swapped for
the real trained ones with **no changes to `run_eval.py`**.

## Files

- `safety_module.py` — `GeometricSafety` / `SemanticSafety` Protocols +
  placeholder implementations (`NoGeometricSafety`,
  `AnalyticHOCBFQPGeometricSafety`, `NoSemanticSafety`, `OracleSpillMonitor`).
- `scene_setup.py` — loads `scenes/scene_yor_cup.xml`, physically grasps the
  cup+8 balls with the LEFT arm (teleport-to-fingers technique, generalized
  from `sim_common.init_grasped_state`'s box grasp — **no existing code in
  this repo grasps the cup**; `collection/collect_cup.py` scripts the cup's
  pose directly and leaves it ungrasped by design), plus the task-PD
  controller and arm-arm contact/drift helpers.
- `run_eval.py` — CLI, 4 configs, episode loop, per-episode JSON + combined
  `results/all_episodes.csv`.
- `report.py` — aggregates to `results/summary.{csv,json}` +
  `../artifacts/eval_compare.png`.

## Scenario

LEFT arm grasps the cup near `sim_common.LEFT_GRASP_Q` (verified in
`collection/collect_cup.py`'s own comments to sit "inside/against this
scene's cup rim" — i.e. a plausible cup-grasp pose already, reused as-is).
RIGHT arm runs `sim_common.intruder_ctrl` (the same scripted, non-cooperative
sweep as `experiments/ab_smooth_vs_bangbang`'s C0–C3), targeting
`RIGHT_INTRUDE_CTRL`, which was IK-solved specifically against a left arm at
`LEFT_GRASP_Q` — so intrusion severity is directly comparable to the
well-tested box-task numbers. Episode length default 1.6 s (320 steps @ 5 ms)
— covers the intruder's ramp-in + part of its oscillating hold.

## Interfaces (how the real models plug in)

**Geometric** (`GeometricSafety.correct(GeometricInputs) -> GeometricResult`):
placeholder is byte-for-byte `run_ab.py`'s C2 law (analytic arm-arm distance
`h(q)` over the ~625-pair keypoint proxy → HOCBF psi1/psi2 rows → box-QP),
reusing `sim_common`'s own building blocks, not reimplemented. **Real swap:**
a trained DeepReach network gives `V(q)` (+ time-derivative) on
`x_HJ=[q_L,q_R]`; replace `h_p`/`hdot_p` (and the `ALPHA1*h_p` term inside
`psi1_p`) in `AnalyticHOCBFQPGeometricSafety.correct` with `V`/`dV/dt` — the
constraint-row algebra and QP call are unchanged. The cooperative
(both-arms-yield) extension from `RESULTS_v4.md` condition C is the next
step after that.

**Semantic** (`SemanticSafety.assess(SemanticObservation) -> SemanticVerdict`):
placeholder is an ORACLE (`OracleSpillMonitor`) using
`cup_task_utils.count_spilled` + a linear ground-truth extrapolation of each
ball's velocity to flag imminent spill. **Real swap:** a trained DreamerV3
failure head on `z_t = RSSM(o_{<=t})`, `o_t = {image, proprio(26D), action}`
per `docs/design/state_spaces.md` S3 — **never** object pose or ground-truth
flags (the oracle placeholder intentionally breaks this rule; the real
monitor's `assess()` must read only the restricted `SemanticObservation`
fields). `run_eval.py`'s only semantic *intervention* right now is "freeze"
(zero the position-attraction term, pure velocity damping) when `veto=True`
— a stand-in for a real abort/replan action.

## Configs

`none` (both off) / `geometric_only` / `semantic_only` / `both` — see
`CONFIGS` dict in `run_eval.py`.

## Smoke run (3 episodes/config, 1.6 s, `MUJOCO_GL=egl`, `safe_env`)

| config | success% | collision% | spill% | grasp-dropped% | mean min h [m] | mean RMS EE jerk | mean solve [ms] |
|---|---|---|---|---|---|---|---|
| none | 0 | 100 | 0 | 0 | -0.047 | 334 | 0.0 |
| geometric_only | 0 | 100 | 0 | 100 | -0.029 | 541 | 41.2 |
| semantic_only | 0 | 100 | 0 | 0 | -0.047 | 334 | 0.0 |
| both | 0 | 100 | 0 | 100 | -0.029 | 541 | 40.6 |

Reproduces `RESULTS_v3/v4.md`'s known collision-vs-grasp trade-off on the cup
task: the solo-ego HOCBF-QP shrinks the safety-margin violation
(-0.047→-0.029 m) but the resulting evasive motion drops the grasp (0%→100%).
Collision stays 100% throughout — expected: this intruder is the same
non-cooperative sweep RESULTS_v3 showed no solo-ego filter fully stops; a
cooperative intruder (RESULTS_v4 condition C) is what closed that gap there.
`semantic_only`/`both` are numerically identical to `none`/`geometric_only`
here because the oracle's veto never fires in this short/mild smoke window
(no ball's extrapolated position crosses the cup boundary in time) — **the
plumbing was separately verified to work**: an earlier probe run (nonzero
`cup_jitter_m`, since reverted as a diversity source — see `run_eval.py`
comment, it has a ball-placement side effect) showed the oracle correctly
detecting an already-spilled state, firing `veto=True` for the whole episode,
and the resulting freeze avoided the arm-arm collision seen in `none`
(`collision: False` vs `True`, same seed). `semantic_precision`/`recall` are
NaN in the checked-in smoke run for the same reason (no positive labels to
score against); the metric itself is computed in `run_eval.py`'s
`_semantic_precision_recall` and is populated whenever any spill event
occurs within the monitor's horizon.

`artifacts/eval_compare.png` renders (4-config x 8-panel bar grid: success,
collision, spill, grasp-dropped rates; mean min-h, RMS jerk, solve time,
veto fraction).

## What's needed to swap placeholders for real models

1. Geometric: a trained DeepReach `V(q)` checkpoint + a loader that gives
   `V`/`grad V`/`dV/dt` at a joint config; wire into
   `AnalyticHOCBFQPGeometricSafety` (or a new class) per the swap-in note
   above.
2. Semantic: a trained WM (DreamerV3 RSSM + failure head) checkpoint +
   inference wrapper taking `{image, proprio(26D), action}`; implement
   `SemanticSafety.assess` reading only `SemanticObservation`'s restricted
   fields (currently `model`/`data` are passed through only for the oracle —
   a real implementation should ignore them).
3. To get non-degenerate semantic precision/recall and to test the
   cooperative-intruder regime, lengthen `--duration-s` (e.g. 3–6 s, full
   oscillating hold) and/or fix `cup_jitter_m`'s ball-recentering so it can
   be used as an episode-diversity knob again.

## Blockers

None. All 4 configs run end-to-end; table + plot generated.

## Rollback

Delete `eval/` and `artifacts/eval_compare.png`. Nothing outside `eval/` /
`artifacts/` was touched; `sim_common.py`, `cup_task_utils.py`,
`scene_yor_cup.xml` are untouched (import-only).
