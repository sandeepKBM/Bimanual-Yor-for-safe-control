# FL-MPC bitwise replay harness

Purpose: prove, bit for bit, that `FLMPC.step()` still gives the same outputs after a refactor
(for example the planned optional `extra_rows=None` hook, option A in
`docs/status/mpc_fl_abstraction_options.md`). It is the safety net that note calls "T0/T1".

Everything here is new files only. It edits no existing file. It runs in simulation on CPU. It never
imports `YOR/robot/*` or `YOR/extra/*`, and it opens no socket, ROS node or hardware connection.

## What it does, in plain English

1. **Record** (done once, on the *old* code). We run short simulations and, at many control steps,
   save everything `step()` reads and everything it produces:
   - the MuJoCo state (all small `MjData` arrays, including stale derived ones such as `qM`,
     `qfrc_bias`, `xmat`, `cdof`, which `step()` reads as they are);
   - every argument of `step()` (references, `pairs_all`, `table_rows`, payload terms, all kwargs);
   - the controller state before the call (warm starts, in-flight queue, `_cup_axis_local`, counters);
   - the full return value, every call to `solve_constrained_box_qp` (digest of inputs and outputs, in
     order), and the side effects on the controller object.
2. **Check** (run any time, on the *current* code). For each record it rebuilds the controller and
   `data`, calls `step()` again and compares with `np.array_equal` (NaN-aware). It compares:
   the return value, the **sequence of solver calls** (this is what catches a refactor that bypasses the
   module-global `sc.solve_constrained_box_qp` patch point), and all controller side effects.
   Any difference is a FAIL. A tolerance column (`--atol`, default 1e-9) is printed only for information.
3. **Selftest** proves the checker is not blind (see below).

Three controller classes are covered:
- `FLMPC` (reference Python path; flagship episodes and random states),
- `ValueFLMPC` (the path that swaps `mpc_fl.sc` for a solver proxy and appends value rows),
- `RTFLMPC` drop-in `step()` (numba realtime path, pure computation).
Not covered: `RTFLMPC.step_lean` / `PipelinedRTFLMPC` (they need the lean-dynamics wiring and a thread),
`FLMPCDecentralized`, `FastFLMPC`, and the v4c / run_ab4 controller (it never uses FL-MPC).

## Where the records are

`/common/users/ss5772/bimanual_data/flmpc_replay/` (kept well below 500 MB):
- `ep_<profile>_s<seed>/` : records from a short flagship-demo episode (profile `flagship` = demo CLI
  defaults with faked grasp and all control fixes; `ce` = the comprehensive_eval N=50 job config);
- `synthetic_s7/` : random robot states x option grid (cone modes, tilt box, leash, table rows, speed cap,
  `ego_hold`, `tau_max_vec`, delay, adaptive_yield, large `H_MARGIN` to force the relaxed / zero fallback),
  for all three classes;
- `<dir>/models/<sha>.mjb.gz` : the compiled MuJoCo model (about 64 MB raw each, stored once, gzip);
- `logs/`, `meta_*.json` : recorder logs.
Files are `rec_NNNNN.pkl.gz`. The `check` command finds them recursively.

## Commands (run from `/common/users/ss5772/bimanual`)

```
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
H=experiments/ab_smooth_vs_bangbang/replay

# 1. record (only needed to make NEW records; takes a long time on a loaded machine: the episode build
#    alone was ~9 min on westeros under load). 1 core per job; keep --jobs <= 4.
$PY $H/replay_cli.py record --out /common/users/ss5772/bimanual_data/flmpc_replay \
      --seeds 0 1 --profiles flagship ce --synthetic --jobs 3

# 2. check (the thing you run before/after a refactor; a few minutes, CPU)
$PY $H/replay_cli.py check --records /common/users/ss5772/bimanual_data/flmpc_replay --coverage

# 3. prove the checker catches regressions
$PY $H/replay_cli.py selftest --records /common/users/ss5772/bimanual_data/flmpc_replay
```
Pin threads (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1`); the scripts
set these by default. Do not run the episode recorder on a GPU node for no reason: it needs no GPU.

## How to use it around the refactor

1. **Before** touching `mpc_fl.py`: run `check`. It must print `RESULT: PASS`. (If not, the environment
   or the records are not usable as a baseline; stop and look.) Do not re-record after the refactor:
   records are the frozen "old behaviour".
2. Make the change (for example add `extra_rows=None` to `step()`).
3. **After**: run `check --coverage`. Required: `RESULT: PASS` on every class (FLMPC, ValueFLMPC,
   RTFLMPC). Look at the `[coverage]` lines: lines of `FLMPC.step` that no record executes are NOT
   protected, so do not edit them without adding a record that hits them.
4. If the hook is meant to replace the `ValueFLMPC` proxy, the ValueFLMPC records still must pass
   unchanged through the old proxy path. Add new records for the hook path separately (the T2 parity
   test in the options note), do not overwrite these.
5. A deliberate numerical change (new default) will FAIL. That is the point: re-record only with an
   explicit decision, into a NEW directory.

## What counts as a pass, and what can still slip through

- Pass = every record is bit-identical (`np.array_equal`, NaN-aware) on return value, solver-call
  sequence and controller side effects. The summary also reports how many compared fields are even
  byte-identical (distinguishes `-0.0` and NaN payloads).
- Proven only for the recorded situations. Unhit code is listed by `--coverage`. Behaviour that depends on
  things we did not record (a different model, different `sc` globals other than `TAU_MAX` and `DT`,
  `fast_qp` installed instead of the plain solver) is not tested; the record stores the solver
  implementation name for reference.
- `selftest` injects, by run-time monkeypatch only (no file is edited): a 1e-12 change to one row, a 1e-12
  change to tau, a private solver binding that bypasses `sc.solve_constrained_box_qp`, ValueFLMPC without
  its proxy, a dropped state update, a 1e-12 change in the fallback backstop, and a 1e-12 change in the
  RT step. Each must FAIL; the unmodified code must PASS before and after.

## Files

- `replay_core.py` record/replay/compare machinery (snapshots, solver tracer, Recorder, `replay_one`, `check_one`)
- `record_episode.py` records a short flagship / ce episode (wraps `FLMPC.step` and `__init__` at run time)
- `record_synthetic.py` records random-state branch-coverage cases (FLMPC, ValueFLMPC, RTFLMPC)
- `check.py` the checker (+ line coverage of `mpc_fl.py` / `flmpc_value.py` during replay)
- `selftest.py` regression-injection self-check
- `replay_cli.py` single entry point
