# Library + LOOK integration (`gate_lib.py`, `sweep_lib.py`) -- SIMULATION ONLY

`LibGatekeeper(Gatekeeper)` overrides `_commit`, `gk_step`, `filter_base`, `counters` only. Base defaults are untouched; with `stall_s=inf` it reproduces the base golden counters bit for bit (test).

## Trigger
- Every commit (hold-u_op or candidate) -> `lib.add(tr, belief it was certified under, v0=tr.V[0], ...)` (grow xy 0.15 / p 0.3 / mu 0.10; `add_stride` thins it).
- After a gk step with no candidate commit (mode keep / regime 2): a streak timer runs. `lib.recover(last_bel, v0)` is called when the streak >= `stall_s` (ASSUMPTION 1.0 s) or on ENTERING regime 2, then at most every `rec_period_s` (0.5 s, ASSUMPTION) and never while a recovery trajectory is still running (unless in regime 2). A candidate that certifies pre-empts any recovery in progress.
- direct: entry trajectory re-based at now, mode `rec_direct` (`_cb` set for the monitor keep rule). `debug=True` re-runs check1/check2 under the current belief; a failure is counted (`n_rec_mismatch`), the entry refused (`strict=True` raises).
- move: legs committed one by one (mode `rec_move`), each re-certified under the then-current belief (abort -> `n_rec_move_abort`), then the entry found by containment (`n_rec_move_done`).
- look: see below. none: base behaviour.

## LOOK mechanics
- Tracker source only (truth people carry no observation age -> counted no-op `n_look_noop`).
- Only from rest: `|v0|_inf <= 1e-3`; otherwise `n_look_deferred`.
- Target = person whose track is a soft deficit (`person{j}:e_p|age|e_v` in `rc.look["reduce"]`), largest deficit ratio first, whose bearing is outside the camera half-FOV and within `max_glance_deg`. In view already -> `n_look_infov` (occlusion, a glance does not help); only pose/velocity deficits -> `n_look_nofix`.
- A `people_track.look.Looker` (all automatic triggers OFF) is forced active on that bearing. While active, `filter_base` replaces the operator twist with the Looker's in-place yaw (0,0,+-0.7); that twist goes through the SAME gk certification (hold -> check1+check2); if it is not committed the glance is cancelled (`n_look_blocked`). It ends when the person is in the FOV or after `max_glance_s`; Looker cooldown/duty applies. The next gk steps re-evaluate with the fresh belief; `n_look_resolved` = a commit within 3 s of a glance.

## Counters (for T3)
`n_rec_calls/direct/move/look/none/commit/mismatch`, `n_rec_move_abort/abandoned/done`, `n_look_glances/noop/deferred/nofix/infov/cooldown/blocked/resolved`, `n_regime2_entries`, `lib_size`, `lib_added/rejected/dominated/merged`, `lookup_ms_*`, `recover_ms_*`, `lib_add_ms_*` + all base counters.

## Limitations
- The library only helps when a rest-start (or same-twist) certified entry covers the CURRENT belief; entries are created only from the gatekeeper's own commits.
- A stay-put segment is often the only certified entry near a standing person (tests); recovery then gives no progress.
- MOVE connectors are straight line + rotation; aging near people refuses them. LOOK cannot fix pose (`e_xy`) deficits and cannot see through occlusion.
- `lib.add` with growth costs tens of ms per commit (see `lib_add_ms_*`); `falsified/monitor_failure` invalidation is NOT wired (a falsification only counts in `n_falsified`).
- stall_s, rec_period_s, REST_EPS are assumptions, not tuned.
- Run: `python -m world_model.v2.house.gatekeeper.sweep_lib --seeds 0-1 --workers 6 --max-s 45 --out <dir> [--baselines]`.
