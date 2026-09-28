#!/bin/bash
# Owner (Opus) independent re-verification of the real-time fast path on a QUIET node:
#   (1) end-to-end equivalence gate (orig vs fast, must be bit-identical),
#   (2) clean wall-clock per-step profile, orig and fast.
# CPU-only; Slurm allocates 2 CPUs -> run exactly 2 single-threaded processes at a time.
# Runbook: no --cpus-per-task, generous --time, finite work, _DONE sentinel.
#SBATCH --job-name=bim_rt_verify
#SBATCH --nodelist=ilab1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/common/users/ss5772/bimanual/experiments/realtime/ilab_runs/verify_%j.out
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
REPO=/common/users/ss5772/bimanual
OUT=$REPO/experiments/realtime/ilab_runs
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd $REPO
echo "host=$(hostname) start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
# Stage 1: clean wall-clock profiles, orig and fast in parallel (2 allocated CPUs).
$PY experiments/realtime/profile_flmpc_step.py         --json-out $OUT/prof_orig_${SLURM_JOB_ID}.json > $OUT/prof_orig_${SLURM_JOB_ID}.txt 2>&1 &
$PY experiments/realtime/profile_flmpc_step.py --fast  --json-out $OUT/prof_fast_${SLURM_JOB_ID}.json > $OUT/prof_fast_${SLURM_JOB_ID}.txt 2>&1 &
wait
echo "profiles done $(date -u +%FT%TZ)"
# Stage 2: end-to-end equivalence gate (runs orig + fast episodes, compares every logged series).
$PY experiments/realtime/gate_equivalence.py --mode e2e --only both > $OUT/gate_e2e_${SLURM_JOB_ID}.txt 2>&1
echo "gate exit=$? done $(date -u +%FT%TZ)"
date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT/_DONE_${SLURM_JOB_ID}
