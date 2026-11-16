# Cup-task world-model eval: does it model our object, and does it help?

Checkpoint: `world_model/logdir/wm_cup_full/best.pt` (2000 offline steps on
`collection/data/cup_yor_full`, 200 episodes: `success_gentle` / `spill_shake`
/ `spill_tilt` / `collision_intrusion`, 50 each). Held-out split = last 20
episodes (`episode_000180`..`episode_000199`, same 4-way mix, 5 each).
`image` = ZED workspace cam (`zed_cam`), duplicated into both halves of the
6-channel field (`collection/cameras.py`'s `YorCupCameraRenderer` stores the
same ZED frame into `rgb_head` AND `rgb_wrist_right`) -- so despite the
"head+wrist" field names, the WM was trained on the real ZED view, twice.

## 1. WM eval (`world_model/eval_wm.py`, `world_model/eval_wm_zoom.py`)

Whole-frame metrics (held-out 20 episodes, `--prefix-len 8 --rollout-len 20`,
resampled 3x to confirm stability):

| metric | value |
|---|---|
| reconstruction image MSE | 0.00054 |
| reconstruction proprio MSE | 0.0043 |
| reconstruction KL | 2.25 |
| open-loop imagined image MSE (20 imagined steps) | 0.0006 |
| open-loop imagined proprio MSE | 0.0094-0.0101 |

These numbers look excellent (loss fell from ~40 -> ~15 during training, and
eval MSE is near the reconstruction floor even 20 imagined steps out). **But
this is misleading taken at face value** -- see the honest filmstrip read
below.

### Honest filmstrip read: does it capture cup tipping / ball scatter?

`eval_wm.py`'s own filmstrip (`artifacts/wm_eval_cup_full.png`, and 3
resamples) looks visually STATIC at the default 64px/frame display size --
true/recon/imagined rows all look like an unchanging table+arms scene. This
is NOT because nothing is happening; it's because the ZED frame is dominated
by static background (sky, table, robot arms) and the cup+8 balls occupy a
handful of pixels, invisible at thumbnail scale. A zoomed, cropped,
specific-episode investigation (`world_model/eval_wm_zoom.py`, table-region
crop only) tells a different, mixed story on the two spill mechanisms in the
data:

- **`spill_tilt` (episode_000182, `artifacts/wm_eval_spilltilt_zoom.png`):**
  the cup visibly rotates/tips over the true trajectory (upright ->
  progressively rotated -> lying on its side by t=157). The imagined rollout
  **does track this** -- the predicted cup also rotates in the right
  direction, though the magnitude is underestimated/blurred relative to
  ground truth (a "ghost" outline grows in the error row as the true cup
  diverges further from upright than the prediction). **Verdict: partially
  modeled** -- gross object-tipping motion is captured, magnitude is blurry.
- **`spill_shake` (episode_000181, `artifacts/wm_eval_spillshake_zoom.png`):**
  the true frames clearly show 8 small ball dots scattering outward from the
  cup as the episode progresses (confirmed directly from the raw ZED array,
  `/tmp/spill_zed_frames.png`). The **teacher-forced reconstruction** (full
  observation access every step, not even open-loop imagination --
  `artifacts/wm_teacherforced_ballcheck.png`) reconstructs the cup
  correctly but **shows essentially no balls at any timestep**, even though
  the true frames do. The open-loop imagined rollout is the same or worse.
  **Verdict: NOT modeled** -- the decoder/encoder appear to treat the
  1-3px ball blobs as noise and regress toward the (ball-free) mean, because
  plain per-pixel MSE loss on a 64x64 frame gives a handful of ball pixels
  negligible weight against the dominant static background.

**Net read:** the very-low whole-frame MSE numbers above are real but
misleading -- they are dominated by the (correctly, trivially) reconstructed
static background. The WM models the cup's own gross pose/tilt reasonably
(spill_tilt) but does **not** faithfully model the ball-scatter event itself
(spill_shake) at the pixel level. This matters directly for the monitor head
below: if it works, it is likely reading cup-tilt/shake proprio+motion
correlates of the perturbation, not actually "seeing" spilled balls (which
are occluded/invisible from ZED once inside the cup anyway, per the task
brief, and now also shown to be under-modeled once they scatter outside it).

Filmstrips: `artifacts/wm_eval_cup_full.png` (whole-frame, default sample),
`artifacts/wm_eval_spillshake_zoom.png`, `artifacts/wm_eval_spilltilt_zoom.png`,
`artifacts/wm_teacherforced_ballcheck.png` (teacher-forced ball-visibility
check).

## 2. Monitor head (`world_model/monitor_head.py`)

TODO (fill after run2 completes): precision/recall/ROC-AUC for spill,
collision on the held-out split, using the REAL per-step spill label
(`--spill-signal per_step`, i.e. `collect_cup.py`'s `dropped[t]` flag) rather
than the old constant-per-episode broadcast placeholder.

Bug found + fixed en route: `monitor_head.py`'s `roc_curve()` already had a
numpy>=2.0 compat shim (`getattr(np, "trapezoid", None) or getattr(np,
"trapz", None)`) but the first run crashed with `AttributeError: module
'numpy' has no attribute 'trapz'` inside a garbled traceback (mismatched
frame/line info, consistent with a stale `__pycache__` .pyc). Rerun with
`python -B` (no bytecode cache) to rule this out.

## 3. Eval-harness comparison (`world_model/run_wm_eval_harness.py`)

TODO: run with `--wm-checkpoint world_model/logdir/wm_cup_full/best.pt
--head-checkpoint world_model/logdir/wm_cup_full/monitor_head_cup_full.pt`,
all 4 configs, compare against the oracle-based smoke table in
`eval/README.md`.

## 4. Verdict

TODO after sections 2-3.
