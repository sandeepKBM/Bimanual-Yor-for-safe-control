# k = achieved / planned object acceleration (measurement, 2026-11-17)

`measure_k.py` measures, on the motions Rule 1 certifies, how the object's achieved acceleration in the MuJoCo sim compares with the planned one.
Sim only, no checker or planner file is edited. Tag MEASURED, statistic "max over n (not a bound)", noise model "sim, deterministic per world".
Leaves out: real-robot tracking, sensing, worlds outside the cells.

## Run

```
PYTHONPATH=<worktree> BIMANUAL_NO_NETWORK=1 BIMANUAL_PLANNER=closed_chain \
python measure_k.py --task box --cells 0.5-1.0,1.0-1.5,... --n 20 --procs P --out DIR --a 0.5
```
Threads are pinned to 1 and `BIMANUAL_PLAN_CACHE=DIR/plan_cache` is set by the script. `Pool(P, maxtasksperchild=3)`. Restartable: an episode whose
`DIR/eps/<task>_<lo>-<hi>_a<a>_<i>.json` exists is skipped (each file is written atomically, tmp + rename). `episodes.jsonl` and `TABLE.json` are rebuilt from `eps/` at the end
(`--table-only` rebuilds them without running). Seed of episode i in a cell: `1117000000 + round(100 * lo) * 1000 + i` (recorded in every record; the same worlds for every `a`).

## What one episode does

1. World = `carry_cert.sample_world(task, cell, rng(seed))`; `CarryEpisode(w, a_plan=a, planner="closed_chain")`; `ep.set_belief(cell centre, a_plan=a)`; hold 0.4 s.
2. `executor.run_segments(ep, plan, rec, True)` with `certs=None` (the planner's STORED backups, no Rule 3 / 4). Each segment: the nominal run (snapshots every 0.25 s inside the
   move, t = 0 after the first step, and the end), then the counterfactual family: restore each snapshot and run the stored backup
   (stop -> retrace of this and earlier moves at A_BACKUP -> hold -> weld off -> retreat -> hold). This is the same recovery from the same snapshots as
   `run_segments(interrupt=...)` at that branch time. A nominal U violation does not stop the run here (the executor would stop after that segment); it is recorded in `nominal_viol`
   and `U`, and the plan is measured to its end.
3. A step hook records the object body's linear velocity (world frame, free-joint qvel) every 5 ms with the phase and the active `Piece` (`pdd`).
   Achieved acceleration = (v_i - v_{i-w}) / (w * 5 ms), w = 1, 4, 10 (5, 20, 50 ms), per world axis and as the norm; windows never cross a restore.

## Definitions

- `k_plan` = |achieved| / a_ref, a_ref = max |Piece.pdd| of the active piece (its TOPP-RA cap: `a` for the nominal pieces, A_BACKUP for backup pieces; a stop uses the piece it interrupted).
- `k_a` = |achieved| / a, the per-axis box Rule 1's rows are certified on (`envelope.accel_corners(a)`).
- `*_axis` = max over the 3 world axes of the per-axis ratio; `*_norm` = norm of the acceleration vector over the same denominator. `abs_norm` is m/s^2.
- Phases: `nominal_piece`, `piece_boundary` (first 10 steps of a piece or the 10 steps after its end), `piece_end` (last 10 steps of a piece), `hold`, `stop`, `retrace_piece`, `post_release` (weld off / retreat; excluded from the kind-level max).
  The stored backup has no read, so its `hold` after the stop is the hold a read_then_retrace program would read in. The `release` segment is excluded from the kind-level max.
- TABLE.json rows: per (task, cell, a, kind in {nominal, backup}): `n`, `n_with_U_monitor` (episodes with any `viol_all` kind in that kind), `k_all_phases` (p50 / p95 / max over
  episodes of each episode's max over segments, branches and phases, at 5 / 20 / 50 ms), `phase_of_max` (the phase and segment of the episode with the largest k_plan_norm, plus a histogram
  over episodes), `per_phase` (the same statistics per phase).
- p50 / p95 / max are over n episodes: "max over n", never a bound.

## Not covered

The Rule 1 programs of `read_then_retrace` (bins at their own a_j) are not run: certs=None runs the stored backup only. Real-robot tracking, sensing, worlds outside the cells.
