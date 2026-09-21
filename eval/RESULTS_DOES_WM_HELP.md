# Does the WM help? -- SPILL scenario (fixed), 8 seeds/config

## What changed

**Scenario** (`eval/run_eval.py`, `eval/scene_setup.py`): the old scenario
was collision-only (`spilled=False` everywhere), so the WM's actual job
never got exercised. New scenario: RIGHT arm no longer scripted-intrudes
(idle, so collision isn't decisive); LEFT arm executes a fast **tilt**
transfer of the grasped cup (quintic ramp, 70 deg over 1.6 s, via the EE's
rotational Jacobian pinv) -- a semantic failure (spill) the geometric
arm-arm-distance layer cannot see.

**Grasp bug found + fixed**: instrumenting contacts (`ncon`) through
`init_cup_grasped_state`'s own settle sequence showed the finger-pad grasp
never touches the cup wall at all (pads end up ~3 cm from every wall
segment, i.e. dead-center in the empty cup) -- the cup was never physically
carried by the arm; any EE motion left it frozen in place (verified: cup
position bit-for-bit unchanged across a 100 deg EE tilt). Fixed by
kinematically re-pinning the cup to the EE every step
(`scene_setup.reattach_cup_rigidly`) -- same technique
`collection/collect_cup.py`'s own validated spill motions use. Only the cup
body is pinned; the 8 balls stay fully free and are what actually spill.
`cup_dropped`/`cup_drift` are therefore ~0 always now (pinned by
construction) and no longer meaningful.

**Veto wiring fixed**: the semantic veto's `freeze` correctly zeroed the
PD's position term, but the transfer *schedule* was still indexed by
wall-clock time, so a released veto found the target had marched ahead
while the arm sat frozen -> a violent catch-up snap (measured: rms jerk
~1300 vs ~17 baseline, and it still spilled 3-5/8 balls). Fixed by giving
the transfer its own clock that only advances on unvetoed steps -- an
actual hold, not pause-then-lurch. Oracle horizon bumped 0.15s->0.35s
(needed lead time given rolling-ball momentum can't be undone by freezing
the arm alone). `semantic_only`/`both` run the REAL trained WM monitor
(`world_model/wm_semantic_monitor.py`'s `RealWMSpillMonitor`,
`wm_cup_full/best.pt` + `monitor_head_cup_full.pt`, threshold 0.5) via the
pre-existing `world_model/run_wm_eval_harness.py` monkeypatch.

## Results (8 seeds/config, 4.5s episodes)

| config | success% | spill% (headline) | collision% | mean min h [m] | veto frac |
|---|---|---|---|---|---|
| none | 0 | **100** | 0 | 0.135 | 0.0 |
| geometric_only | 0 | **100** | 0 | 0.135 | 0.0 |
| semantic_only | 100 | **0** | 0 | 0.086 | 1.0 |
| both | 100 | **0** | 0 | 0.086 | 1.0 |

**Yes -- the WM helps, cleanly.** `geometric_only` spills exactly as often
as `none` (geometry is structurally blind to spill); `semantic_only`/`both`
prevent it in all 8/8 seeds using the real trained monitor.

**Honest caveat**: `semantic_veto_frac=1.0` -- both the tuned oracle and the
real WM monitor veto essentially the *entire* episode, not a sharp
just-in-time trigger. This is a coarse but functional signal, not evidence
of fine-grained risk discrimination; a real evaluation would also check the
monitor doesn't veto a genuinely safe/gentle transfer (not tested here).

Figure: `artifacts/does_wm_help_spill.png`. Data:
`eval/results/real_wm/{batch_a,sem_b0,sem_b1,both_b0,both_b1}/all_episodes.csv`
(merged to `eval/results/real_wm/all_episodes.csv`), `summary.{csv,json}`
via `eval/report_spill.py`.

## Blockers
None outstanding.

## Rollback
Revert `eval/run_eval.py`, `eval/scene_setup.py`, `eval/safety_module.py`
(horizon default only); delete `eval/report_spill.py`,
`eval/results/real_wm/`, `artifacts/does_wm_help_spill.png`. Nothing outside
`eval/`/`artifacts/` touched.

## Demo video (HUD, prevented-failure + workaround)

`artifacts/safety_demo.mp4` (1280x720, 40fps, 21.0s, h264/yuv420p, 4.8MB),
built by `scripts/make_safety_demo.py` (new; only orchestrates/renders --
no edits to eval/, world_model/, or experiments/). 4 segments, real-time
playback, rendered offscreen (`MUJOCO_GL=egl`), env `mujoco_ur5e`:

1. **Spill baseline** (config `none`): the real trained WM monitor
   (`world_model/wm_semantic_monitor.RealWMSpillMonitor`, `wm_cup_full/
   best.pt` + `monitor_head_cup_full.pt`) runs in SHADOW (drives the HUD's
   P(spill) meter only, never wired to control) -- meter climbs to 1.00, no
   intervention, cup spills (8/8 balls) -> "SPILLED".
2. **Spill WM-protected** (config `semantic_only`): the SAME real monitor
   now drives the hold veto (identical wiring to `world_model/
   run_wm_eval_harness.py`'s monkeypatch) -- meter climbs, "⚠ SPILL RISK →
   HOLDING" banner fires on veto, transfer holds, 0 spilled -> "SAFE".
3. **Collision adversarial** (`run_ab4.py` condition A): arm-arm margin `h`
   meter (real per-step `all_pair_barriers` value) goes negative ->
   "COLLISION" (+ box dropped), no cooperation banner (none engages).
4. **Collision cooperative** (`run_ab4.py` condition C): same real `h`
   meter; "⚠ COLLISION → COOPERATIVE AVOID" banner fires while h stays
   below a visualization threshold (0.05m, chosen by inspecting the
   logged h(t), not a certified value); min_h stays positive (+0.016) ->
   "SAFE, GRASP HELD".

All 4 outcomes reproduced this run's own console log exactly matching prior
verified results (this doc's table + `experiments/ab_smooth_vs_bangbang/
logs/v4_summary.json`): baseline spilled 8/8, WM 0/8, adversarial
collision=True/box_dropped=True, cooperative collision=False/
box_dropped=False. Faked/placeholder elements: none of the plotted signals
(P(spill), veto, h) are synthetic -- only the banner's h-threshold (0.05m)
and the video's frame stride (every 5 physics steps, for real-time fps) are
chosen for visualization, not measured. `run_ab4.run_condition`'s own
`np.savez` side effect is redirected to a tempdir for this script's calls
(module global swap, not an edit to `run_ab4.py`) so no files outside
`artifacts/` were written. No blockers.
