# Stage 0b probe fix: launch recipe (calibration + probe-only comparison)

SIMULATION ONLY, CPU only, offline (no Codex/LLM calls: the behind worlds replay the proposer's first plan from the run_n20_v2b cache).
Methods: `poslift` (default, unchanged), `poslift_ff` (hybrid force-step finish), `claw_guard` (per-claw load guard); code in `probe.py`.
New ASSUMED values (need user approval before any claim rests on them): `DM_FINE` 0.1 kg (env `PL_DM_FINE`), `FF_FREE_RATIO` 0.2
(`PL_FF_FREE_RATIO`), `FF_T_HOLD` 1.2 s (`PL_FF_T_HOLD`), claw_guard `EPS_CLAW` = eps_lo and `K_STEP_CLAW` = pooled K_STEP until step 1 measures them.

## 0. Freeze the code (once)

```bash
WT=/common/users/ss5772/bimanual/.claude/worktrees/demo-a2
OUT=/common/users/ss5772/bimanual_data/contents/probe_fix_2026-11-12
REV=$(git -C $WT rev-parse HEAD)            # the demo-0b-probe commit to run; `git -C $WT status --short` must show no change under world_model/
mkdir -p $OUT/code && git -C $WT archive $REV | tar -x -C $OUT/code && echo $REV > $OUT/CODE_REV
export PYTHONPATH=$OUT/code CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd $OUT/code
```

Size every `--procs P` from the cores idle NOW on the host (`nproc` minus `uptime` load; AGENTS.md "Parallel sweeps").
Every command resumes: rerun the same line after a crash and finished worlds are skipped.

## 1. Calibration (per method; n = 20 per task, box/tray6/pot = the kstep_calib_2026-11-09 worlds)

```bash
for M in poslift_ff claw_guard; do
  setsid nohup $PY -m world_model.v2.nero.contents.probe_liftoff calibrate --method $M --tasks box,tray6,pot --n 20 --procs P \
      --out $OUT/calib_$M > $OUT/calib_$M.log 2>&1 &
done
```

60 worlds per method (noise-free, eps 0, from the empty mass, no stop rule).  Output `$OUT/calib_<M>/CALIB.json` + `calibrate.jsonl`:
`eps` (EPS_LO, EPS_FREE), `per_task`, `K_STEP`, `FALSE_FREE`, `dL`, `JUMP` (height rungs), and `FF` (poslift_ff: finish-phase dL, dL minus the
commanded step, `FALSE_FREE_FF`, stop reasons, ceiling) or `CLAW` (claw_guard: `EPS_CLAW`, `K_STEP_CLAW`, `JUMP_CLAW`, residuals on the counter / free,
split offset vs sim-truth COM).  Check before step 2: `FALSE_FREE_FF.n_false` = 0, `per_task.*.n_error` = 0.

claw_guard's operating values come from its own CALIB only via `PL_CLAW_CALIB=$OUT/calib_claw_guard/CALIB.json` (or a reviewed copy at
`world_model/v2/nero/results/probe_calib_claw_guard_2026-11-12/CALIB.json`); without it EPS_CLAW = eps_lo and K_STEP_CLAW = pooled K_STEP (ASSUMED).

## 2. Probe-only comparison (100 worlds x 3 methods = 300 probes; no carry)

Worlds: box_behind / tray6_behind i = 0..19 (smoke.behind_world) + box / tray6 / pot i = 0..19 (calibration worlds; in-sample for the eps).

```bash
K=4                                          # shards; one per host or per process group
export PL_CLAW_CALIB=$OUT/calib_claw_guard/CALIB.json
for k in 0 1 2 3; do
  setsid nohup $PY -m world_model.v2.nero.contents.probe_compare run --out $OUT/compare --procs P --shard $k/$K \
      --calib poslift_ff=$OUT/calib_poslift_ff/CALIB.json,claw_guard=$OUT/calib_claw_guard/CALIB.json \
      > $OUT/compare_${k}of${K}.log 2>&1 &
done
# after all shards:
$PY -m world_model.v2.nero.contents.probe_compare summary --out $OUT/compare      # -> $OUT/compare/SUMMARY.md
```

Per record: Theta, truth inside (sampled), width, why, n_rungs / n_ff_rungs, last read L, u_read_max, peak_u (sim truth), every U kind during the probe,
probe_time_s, static verdict on Theta and on the true contents point (+ base level on the behind tasks), eps and their source, the probe summary
(poslift_ff: finish-phase record; claw_guard: per-claw cap, EPS_CLAW source, split reading).  SUMMARY.md also checks poslift's Theta against the source
run (run_n20_v2b) on the behind worlds.

Cost guide (smoke, westeros under load ~72): one behind world ~6 min wall per probe including the static verdicts and c_admit_max; calibration worlds less.

## Smoke (already run by the builder; not part of the run)

```bash
$PY -m world_model.v2.nero.contents.probe_compare run --out SCRATCH/smoke --sets behind --only box_behind:4,tray6_behind:0 --procs 6 --provisional-eps
```
`--provisional-eps` lets an uncalibrated method borrow poslift's eps; every record then says `PROVISIONAL` in `eps_source`.  Never use it for step 2.
