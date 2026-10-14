#!/bin/bash
# G4 clean timing of rt_flmpc vs the current fast path on a QUIET ilab node (owner submits;
# the agent does not submit Slurm jobs). CPU-only, simulation only.
# Slurm allocates 2 CPUs -> the py and rt measurement runs go SEQUENTIALLY, each pinned to the
# first allocated CPU (taskset), threads pinned to 1; the second CPU absorbs OS noise.
# Runbook: no --cpus-per-task, generous --time, finite work, _DONE sentinel.
#SBATCH --job-name=bim_rt_g4
#SBATCH --nodelist=ilab1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/common/users/ss5772/bimanual/experiments/realtime/ilab_runs/rt_g4_%j.out
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
REPO=/common/users/ss5772/bimanual
OUT=$REPO/experiments/realtime/ilab_runs/rt_g4_${SLURM_JOB_ID}
mkdir -p $OUT
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
CPU=$(taskset -pc $$ | awk -F': ' '{print $2}' | cut -d, -f1 | cut -d- -f1)
cd $REPO/experiments/realtime
echo "host=$(hostname) cpu=$CPU start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
# numba compile/cache-load time is measured by the first run (fresh cache dir) and excluded from both
export NUMBA_CACHE_DIR=$OUT/numba_cache
taskset -c $CPU $PY rt_gate_g4.py --impl rt --out $OUT > $OUT/g4_rt.txt 2>&1
echo "rt exit=$? $(date -u +%FT%TZ)"
taskset -c $CPU $PY rt_gate_g4.py --impl py --out $OUT > $OUT/g4_py.txt 2>&1
echo "py exit=$? $(date -u +%FT%TZ)"
date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT/_DONE
