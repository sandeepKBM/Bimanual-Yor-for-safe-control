#!/usr/bin/env bash
#SBATCH --job-name=vimap
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/al/vimap_logs/slurm_%x_%A_%a.out
# AL-0 value-of-information map: one task = one (I, arm authority, population, mu-set) cell = line of vimap_tasks.txt (80 VAL carries, closed loop, F5 + per-carry mu set).
# Submit:  mkdir -p /common/users/ss5772/bimanual_data/tray_rep/al/vimap_logs; sbatch --array=0-131 world_model/v2/tray_rep/al/slurm/vimap_array.sbatch   (N = lines - 1)
# Slurm gives 2 CPUs per job (no --cpus-per-task). Temp on the shared data disk, emptied with find -delete.
# SIMULATION / SAVED DATA ONLY: no hardware, no network, no GPU. Resumable: rerun = resume (by episode).
set -uo pipefail
export TMPDIR=/common/users/ss5772/bimanual_data/slurm_tmp/vimap_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $TMPDIR /common/users/ss5772/bimanual_data/tray_rep/al/vimap_logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
# fixed-physics defaults for the whole F5 stack (al.filt also passes calib_file=v2, eps_a=0 explicitly)
export BIMANUAL_MU_EFF_FILE=/common/users/ss5772/bimanual_data/tray_rep/calib_v2/mu_eff_calib.json BIMANUAL_F5_EPS_A=0.0
unset BIMANUAL_LEGACY_PIN
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
TASKS=${TASKS_FILE:-world_model/v2/tray_rep/al/slurm/vimap_tasks.txt}
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" $TASKS)
echo "[vimap] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} args=$ARGS $(date -u +%FT%TZ)"
$PY -m world_model.v2.tray_rep.al.run_vi $ARGS --workers 2
rc=$?
find "$TMPDIR" -delete 2>/dev/null
echo "EXIT $rc"
exit $rc
