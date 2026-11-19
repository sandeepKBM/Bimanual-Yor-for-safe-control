# Current state (2026-11-17)

What is built, wired into the run loop, and tested, and what is not. Simulation only (MuJoCo); there is no hardware code in this repo.
Paths are under `world_model/v2/` unless given in full. Each code fact was checked by grep in the `basestop-1116` worktree on 11-17.
Gate pass counts come from the owner's runs; the agent re-ran only `test_base_stop_rows`, `test_cover_gate`, `test_remaining_route_gate`
and `test_dependency_rules` (33 passed). Rulings are in `docs/status/decisions.md`; this file holds no rulings.

Branch: `contents-armside-2026-11-02` (worktree `.claude/worktrees/contents-armside2`). `basestop-1116` is merged into it on 11-17.
`world_model/v2/stack/`, `world_model/v2/flmpc/`, `docs/README.md` and the 11-02..11-17 docs are not on `master` and not on the main
checkout's branch (`feat/pan-2026-10-29`).

## 1. Built, wired, tested

"Wired" = called by the contents run path (`nero/contents/probe_select.py` -> `adapters/rule1_contents.run_cell` -> `stack/executor.py`).

| Item | Where | Wired | Gate |
|---|---|---|---|
| Fail-closed run path: a task not in `RULE1_TASKS` is refused as Rule 1 MISSING, never admitted in silence. `BIMANUAL_LEGACY_RULE1_SKIP=1` restores the old skip. | `nero/contents/probe_select.py:63` (`RULE1_TASKS`), `:403-409` | yes | none yet |
| Planned-path rows after static admission: collision (arm_arm, arm_world, obj_world must all be present, else MISSING), joint margin 0.05 rad (ASSUMED), reach, closed chain, tilt | `probe_select.planned_path_rows` (`:174`) | yes | none for the tilt row |
| Base-stop check of a `base_move` step: rows `base_stop.{recovery, in_route, at_rest, tip, torque, landing_hold, continuation, goal_support, next_step, open_content}`; program `base_stop_finish` (finish the planned route after the stop); swerve-shift correction; yaw scrub; drift corners; per-level MEASURED stop card | `stack/checks/base_stop.py`, `adapters/rule1_contents.py`, `stack/planners/base_route.py` | yes, through `rule1_contents.run_cell` | `stack/tests/test_base_stop_rows.py` (16), `adapters/tests/test_base_stop_card_gate.py` 11/11, `stack/tests/test_remaining_route_gate.py` 9/9, `adapters/tests/test_base_stop_gate.py` (2, slow). Base step alone, L1 and L2: 8/9 rows pass; `next_step` is MISSING only when the base step is checked alone (`rule1_contents.py:1021`, filled by `link_base_next`, `:1142`) |
| Rule 1 on `box_bookcase` | `RULE1_TASKS` includes it | yes | `adapters/tests/test_bookcase_gate.py` (11-18: default plan, tilt MEASURED n = 20, cell 1.0-1.5, a = 0.5: collision row ok (sampled); Rule 1 admits all four steps) |
| T4: Rule 1 refusals go back to the proposer, tagged FIXABLE / NOT_FIXABLE | `stack/checks/fixability.py`; merged in 0b512f2 | yes | `nero/contents/tests/test_t4_gate.py` |
| Rule 4 at run time: offline mock target proposer (no network) | `nero/contents/runtime_rules.py:35` `rule4_hook`, `adapters/rule4_contents.py`, `stack/checks/rule4.py` | yes (`probe_select.py:533`) | `stack/tests/test_rule4_gate.py` tests `rule4.py` on synthetic callables; no gate for the run-time hook |
| R_K witness: depth 1 (one recovery program, then in R or the continuation); guarantee sampled | `adapters/rule1_contents.py` landing rows | yes | — |
| Executor phases: program `read_then_retrace`, Rule 4 transfer, Rule 3 cover (continue / switch / fallback, fail-closed), wide table, path stop, snapshot logs, no-certificate cascade (A_MIN retrace, then A_BACKUP backup, then hold; each counted as uncertified), boundary read at each segment end | `stack/executor.py`, `stack/checks/cover.py`, `stack/checks/rule1.py:138-144` (wide table) | yes | `stack/tests/test_cover_gate.py` 6/6, `adapters/tests/test_wide_table_gate.py` 3/3. No gate for `read_then_retrace`, the cascade, `landed_safe` or a certificate mismatch |
| Tilt row measured per plan: `tilt_measure.measure_plan`, n = 20 (ASSUMED, `probe_select.TILT_N`), worlds from the belief cell, base move included | `nero/contents/tilt_measure.py:164`, `probe_select.py:171` | yes | none; its numbers predate vref and must be re-measured |
| vref: FL-MPC gets the planned reference velocity (d/dt of the position reference). `BIMANUAL_LEGACY_ZERO_VREF=1` restores zeros | `nero/contents/carry.py:666-672` | yes, default on | `nero/contents/tests/test_vref_gate.py` 2/2; legacy bit-identical to the pre-edit file |
| Numba GJK distance. `BIMANUAL_LEGACY_GJK_SCIPY=1` restores scipy | `nero/gjk_numba.py`, switch in `nero/clearance.py:19` | yes, default on | `nero/tests/test_gjk_numba_gate.py` 3/3 |
| Parallel base-stop sweeps (`ctx.procs`, spawn pool, `maxtasksperchild=3`) | `adapters/rule1_contents.py:713-715`, `:1084` | yes | serial vs 16 processes byte-identical; 213 s vs 33 s (owner run) |
| Release: the landing support is exempt from obj_world. `BIMANUAL_LEGACY_NO_SUPPORT_EXEMPT=1` restores the old monitor | `nero/contents/carry.py:136-140` | yes | `nero/contents/tests/test_support_exempt_gate.py` 4/4 |
| Safe = on a support AND claws released. Rule 1 refutes a hold landing (`landing.not_supported`) unless `BIMANUAL_LEGACY_HOLD_IS_SAFE=1` | `adapters/rule1_contents.py:749-753`; `carry.py` `landed_safe`; `stack/executor.py:270` | yes | no gate for `landed_safe` |
| Tilt-aware load-world collision row (USER 11-17; both ASSUMED values approved 11-17). `BIMANUAL_LEGACY_UNTILTED_COLLISION=1` = the untilted row | `adapters/collision_rows.py` (`TiltSpec`, `swing_twist_lb`, `_Scene`), `nero/contents/tilt_measure.py` (`RotTrack`), `probe_select.planned_path_rows` / `run_rule1` / `fixed_job`, `runtime_rules.rule4_hook(tilt=)`, `z3_oracle_contents` | yes | `adapters/tests/test_tilt_collision_gate.py` 10/10; legacy flag = pre-tilt module on 5 tasks x 2 a (owner, 11-18) |

**Tilt-aware load-world collision row (wired 11-18).** The planned-path rows MEASURE the load tilt of the plan first (swing, twist, pivot
drift per piece, max over n = 20 episodes: sampled). The collision row, Rule 1 and the run-time Rule 4 hook use that measurement. No
usable measurement = MISSING (fail closed). Recovery motions use the plan's envelope (ASSUMED, approved). A start-pose mismatch above
1e-6 rad = MISSING (ASSUMED, approved). Agent checks (sampled): bound slack min 2.7e-5 m over 254,466 pair checks; composition with the
swept term min 0.15 mm after the fix `if allow and dist < cap` (before the fix: -12.4 mm on place). The rigid-load model and the pivot at
the claw midpoint are MODEL. `z3_oracle_contents` treats a MISSING row as no information (never a learned block), so it cannot admit a
plan while the tilt is not passed to it (open finding 10). Box smokes (agent, worlds 2-3): the tilted row refutes the lift
(`shelf|box_lid` / `shelf|box_wall_w`), together with the J7 finding 9.

## 2. Base stop card

`nero/results/base_stop_card_2026-11-16/CARD.json`. MEASURED; statistic "max over n stops (not a bound)"; noise model: none (sim,
deterministic per world). Leaves out: route-tracking error before the branch, refused worlds, other plans and levels.

| Level | n stops | extra overshoot | lateral drift | settle |
|---|---|---|---|---|
| L1 | 162 | 2.57 mm / 0.0054 rad | 2.79 mm | 0.30 s |
| L2 | 108 | 6.33 mm / 0.0090 rad | 5.21 mm | 0.31 s |
| L3 | 0 | no card | — | — |

Rows that need the card are MISSING unless the card's level matches. L3 has no stops, so no card, so L3 is never admitted for a base step.
The tip row (`stack/checks/base_tip.py`) still reads the older G0 base card `nero/results/base_card_G0b_2026-11-09/CARD.json`
(`adapters/base_tip_model.py:28`).

## 3. Not built, or design only

| Item | State |
|---|---|
| Rule 1 rows certified at a × k (k = MEASURED achieved / planned acceleration; ruling 11-17) | not built. k is measured only (`nero/results/accel_k_2026-11-17/`, before vref) |
| R_K deeper than 1 | not built: `rule1.K = 3` and `rule1.NODE_CAP = 50` (ASSUMED, `stack/checks/rule1.py:26-27`) are only recorded as assumptions |
| COM along the claw axis in the checker | not used: `probe.split_position` (`nero/contents/probe.py:695`) is computed by `claw_guard`; "The checker does not use the split yet" (`probe.py:764`) |
| Speed envelope (control-rate speed modulator inside a certified bound) | draft design, not built: `docs/status/speed_envelope_design_2026-11-16.md` |
| Tilt-aware collision row | parked (section 1) |
| Hardware (claim 4) | no code |

## 4. Open findings (not rulings)

| # | Finding | Code check (11-17) |
|---|---|---|
| 1 | Runtime in_R requires a support, while Rule 1 accepts hold landings. | **Does not match the default code.** `rule1_contents.py:749-753` refutes a hold landing unless `BIMANUAL_LEGACY_HOLD_IS_SAFE=1`; `base_stop.landing_hold` (`:1102`) is a prefix row, and its landing is `set_down:goal`. Owner to restate or close. |
| 2 | R_K is depth 1; K and NODE_CAP are unused. | Confirmed: `rule1.py:26-27`, used only in assumption records (`rule1.py:182-183`, `rule4.py:81`). |
| 3 | `row.combine` lets a missing row through when the other rows pass. | Confirmed: `stack/checks/row.py:65-68`: `adm` drops `missing` rows and `admit = bool(adm) and all(r.ok for r in adm)`; the missing row only lowers the guarantee to "missing". Callers that test only `admit` admit. `probe_select` refuses on guarantee "missing" for the path rows. |
| 4 | Back-up starts measured 3-5x the planned acceleration before vref, so the Rule 1 stop rows must be re-measured. | Diagnosis in ledger 11-17; data `nero/results/accel_k_2026-11-17/`. Not re-measured with vref. |
| 5 | L3 has no stop card. | Confirmed: `CARD.json` `per_level` has L1 and L2 only. |
| 6 | coopimp still uses a zero vref. | **Not confirmed.** `CoopImpCtl.tau` (`nero/contents/coop_imp.py:174`) does not call FL-MPC; it takes the velocity reference from central differences of the joint reference (stencil `H_FD` 0.05 s ASSUMED, `coop_imp.py:196-206`, `:236`). Only the angular acceleration reference is 0 (`:234`). Owner to restate. |
| 7 | `RULE1_TASKS` belongs in TaskDef (`stack/tasks.py`). | Confirmed: it is in `probe_select.py:63`, not in `stack/tasks.py` (AGENTS.md: per-task facts live only in `tasks.py`). |
| 8 | The checker modelled the load untilted. Measured: 47/180 box worlds touched the shelf or came within 5 mm of it during the lift, and Rule 1 admitted that plan. | Addressed 11-18 by the tilt-aware row (section 1). `test_obj_world_gate[box-4]` (nominal carry, no checker) still records the 4.8 mm near-miss: it tests the carry, not the checker. |
| 9 | The box hold pose has right J7 = 1.5502 rad against the planned-path joint limit 1.5197 (q_hi - 0.05 margin, ASSUMED). The planned-path rows (run before Rule 1) refuse it, so no box run reaches Rule 1 or T4 today; the T4 loop gate stubs the path rows. | `nero/contents/probe_select.py:384-397`; the 11-16 J7 note in `stack/tests/test_motion_gate.py:5` |
| 10 | `z3_oracle_contents` (stage c) gets no tilt: a MISSING collision row is skipped as no information, so the oracle learns only `block_exact` and admits no plan while the tilt row is on. `z3_watcher_contents` still uses the untilted `_Scene`. | `adapters/z3_oracle_contents.py`, `adapters/z3_watcher_contents.py` |

## 5. Pending (not done)

- Run 3: `box_bookcase`, live Codex (`codex:gpt-5.6-sol:high`), cap 8 requests, fresh worlds.
- Re-measure tilt (per-plan tilt row) and k with vref on.
- Gate tests not yet written: run-time Rule 4 hook with a tilt, `read_then_retrace`, the no-certificate
  cascade, `landed_safe`, certificate mismatch.
- Next experiment (USER 11-17): fixed off-centre load; probe the COM along the claw axis (`claw_guard` `split_position`); compare
  COM-anywhere vs probed COM vs oracle COM on the same worlds.
- Backup-and-continue demo: after everything works (USER 11-17).

## 6. New in 11-16/11-17 (files)

| Kind | Path |
|---|---|
| check | `stack/checks/base_stop.py` |
| run-time rules | `nero/contents/runtime_rules.py` |
| distance | `nero/gjk_numba.py` |
| results | `nero/results/base_stop_card_2026-11-16/` (CARD.json, `measure_base_stop.py`, `before_fix_1116/`, `before_level_1117/`), `nero/results/accel_k_2026-11-17/` (`measure_k.py`, `box_a0.5/`), `nero/results/bookcase_codex_2026-11-16/` (run1 capped at 8; run2 records, plans, world 2), `assumed_sweep_2026-11-16/`, `tilt_table_2026-11-16/`, `speed_mono_2026-11-16/`, `z3_box_2026-11-16/`, `z3_stage0_2026-11-16/`, `box_place_video_2026-11-16/` |
| gates | `stack/tests/test_base_stop_rows.py`, `stack/tests/test_remaining_route_gate.py`, `adapters/tests/test_base_stop_card_gate.py`, `adapters/tests/test_base_stop_gate.py`, `adapters/tests/test_wide_table_gate.py`, `nero/contents/tests/test_vref_gate.py`, `nero/contents/tests/test_support_exempt_gate.py`, `nero/tests/test_gjk_numba_gate.py` |
