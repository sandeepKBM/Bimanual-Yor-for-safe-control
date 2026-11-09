# Golden fixture (migration step 0): today's checker / goal / U / R numbers

Frozen record of what the code at commit `cac058c0718bea5f6c64e36068d2466e9d2a0b15` (branch `contents-armside-2026-11-02`) returns, so the
unified `stack/` tasks / goal / checks / outcome refactor can show nothing changed. Data only: the generator calls the current code and edits
nothing. Every file is `{meta, data}`; `meta` holds git sha, date, python, env flags (`BIMANUAL_*`, thread pins, `MUJOCO_GL`). Floats are full precision.

The committed files were generated from a pristine export of `cac058c` (the live worktree's tracked files were being edited by other work while the
runs were going: `probe.py`, `same_worlds.py`, `carry.py`, `CALIB.json`; those edits change probe results for box i=0 and tray6 i=0). Run
`make_golden.py` on a clean checkout of the commit you want to freeze.

| file | what | counts |
|---|---|---|
| `admission.json` | `carry_cert.static_cell` dicts (action, a_admit, u_nom, rows incl. support_dest, torque_payload, open_content), `carry_cert.support_row`, `probe.c_admit_max`; planners scripted and closed_chain; worlds = `probe._world(task, probe._seed("e2e", task, i))`, i = 0..2, paired with cell = lightest / middle / heaviest of `carry_cert.cells(task)` | 5 tasks x 2 planners x 3 = 30 static cells, 30 support rows (each with planner-goal and default displacement), 10 c_admit_max |
| `probe_admission.json` | `probe.rung_torque_u(task, m)` on an m grid; `Probe.admissible(m_to)` for M_lo in {prior lo, prior mid} x 4 pushes (called unbound on a stand-in with `.task`, `.M_lo`); `interval_spec` + `static_interval` for 2 intervals per task and planner | 10 task/planner grids (8 or 9 m each), 80 admissible rows, 20 interval rows |
| `runtime_episodes.json` | per episode: record summary, and a log of `CarryEpisode._check` (inputs: t, tau ratio, claw wrench per side, object and mug tilt, z vs z0, contacts, lid; outputs: `ep.track["viol"]`, track scalars) and `in_R` (ok, why) every 400th control step (2 s) and at each first U entry, plus the end state | 8 episodes: ours (probe -> admit -> carry/refuse, closed_chain, default poslift probe, default CALIB) on box i=0, tray2, tray4, tray6, pot, box i=2; theirs-ungated (baseline_ungated) on box i=0, tray6 |
| `runtime_episodes_theirs_ac.json` | same logging for THEIR cached AgentChord graph (offline, no LLM) via `execute.same_world_trial`; this is where U is exercised (box i=0, 5: `claw_payload`; i=1: `obj_tilt`, all at `e03_transfer_over_shelf`; same as the 11-08 demos3_regen rows) | 5 episodes: box 0, 1, 5, tray6 0, pot 0 |
| `goal_success.json`, `goal_success_theirs_ac.json` | for each episode above: inputs and outputs of `execute.goal_check`, `success_ours`, the `ac_success` expression (inline in `run_world`; copied), probe outcome ("carried"/"refused_in_R"), and the `stack/eval/harness._contents_job` outcome expression (inline; copied; ours-carried only) | 8 + 5 |
| `pan.json` | pan Episode (pan corner world of `pan/tests/test_segments.py`: pinch, push; pot nominal of `PotUniverse`: all four segments): `in_R` margins (reach off v1/v2, reach on at start), per segment `success(seg)`, `track` margin/worst/viol (from `_monitor`), `pan_state`, and `_monitor` samples (inputs and margins); `cert/admit.certify` on 3 stored world_trials records (`nero/results/pan_cert_v2_2026-10-29/smokeB2/raw/{I03,V10,V11}.json`, sha256 stored; family (0.03, 0.01) as in those records) for each segment at conf 0.95 and 0.9 | 2 episodes (6 segment runs), 4 segments x 2 certify calls |
| `coffee_tray.json` | 5 stored `tray_check` (= `coffee_tray_check` output) dicts from `/data/local/ss5772/agentchord/runs/*/episodes.jsonl` (strict success, loose only, not level, capsule out, level and in but not moved), with a replay of the check on poses rebuilt from them | 5 of 219 records with a check |

## Not captured / caveats
- Coffee tray: ac_run records store only the `check_detail` dict, not the raw poses (tray, capsule, tray0). The replay rebuilds poses from `capsule_rel_tray` and `tray_dxy` (tray0 = identity; tray tilt = 0 if `tray_level` else 15 deg, ASSUMED), so list/`tray_xy_err` fields match to 1e-3 only. No U monitors exist there.
- U is hit only by THEIR plan on the heavy box. The ours episodes and the ungated ones had no U entry (box i=0 refused in R, the others carried). Ours box i=2 and tray6/pot theirs-ac are no-U baselines.
- Pan episodes are built with `observe=None` (no pose estimate): the pinch/push `success` flags are False there; the fixture freezes that behaviour as is.
- `pan_certify` uses 3 smoke records (family 0.03, 0.01); the default `backups.FAMILY` (0.06 ..) raises KeyError on them.
- Runtime logs depend on `probe.default_calib()` (`results/probe_calib_poslift_2026-11-09/CALIB.json`), so re-run only with that file unchanged.
- Sampling: the logs are every 400th `_check` call, not every step. First U entries are always logged.

## Regenerate
From the worktree root (or a clean export of the commit; set `GOLDEN_GIT_SHA` if there is no `.git`):
```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
$PY world_model/v2/stack/tests/golden/make_golden.py --only admission            # about 22 min
$PY world_model/v2/stack/tests/golden/make_golden.py --only probe_admission,coffee_tray   # about 9 min
$PY world_model/v2/stack/tests/golden/make_golden.py --only episodes --procs 5   # about 12 min
$PY world_model/v2/stack/tests/golden/make_golden.py --only pan                  # about 2 min
/data/local/ss5772/agentchord/venv_nero/bin/python world_model/v2/stack/tests/golden/make_golden.py --only episodes_ac --procs 3   # about 1.5 min (needs langchain_core)
```
The five commands are independent (run in parallel: about 22 min wall). The first run should have a warm `stack/planners/plan_cache` (run `closed_chain.static_path(task)` once for each task) to avoid concurrent cache writes.
Simulation only: no network, no robot, no LLM (AgentChord's plan is read from its cache dir `/data/local/ss5772/contents/agentchord_plans_sol_2026-11-05`).
Check of determinism: tray6 i=0 and box i=0 `ours` were re-run in fresh processes (with and without the logging wrapper) on the edited tree: identical, so the wrapper does not change the physics.
