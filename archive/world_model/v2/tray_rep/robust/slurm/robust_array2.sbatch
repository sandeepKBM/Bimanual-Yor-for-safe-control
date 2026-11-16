#!/usr/bin/env bash
#SBATCH --job-name=robust
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/robust/logs/slurm_%x_%A_%a.out
# Copy of robust_array.sbatch (frozen; its sha is in PREREG_ROBUST_FROZEN.md), changed for INFRASTRUCTURE only
# (deviation D1, robust/DEVIATIONS_ROBUST.md). Node /tmp filled up with leftover per-task folders (no temp-dir delete),
# so numba could not write its cache ("Disk quota exceeded"). The per-task temp folder now lives on the shared data disk
# and is emptied at the end with find -delete. Same arguments: $1 = stage. SIMULATION / SAVED DATA ONLY. Resumable.
set -uo pipefail
STAGE=${1:?stage name: oracle_val | test1 | budget | tune | test2 | test3}
R=/common/users/ss5772/bimanual_data/tray_rep
export TMPDIR=/common/users/ss5772/bimanual_data/slurm_tmp/robust_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $TMPDIR $R/robust/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/tray_rep/robust/slurm/robust_tasks_${STAGE}.txt)
echo "[robust] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} args=$ARGS $(date -u +%FT%TZ)"
$PY -m $ARGS --workers 2
rc=$?
find "$TMPDIR" -delete 2>/dev/null
echo "EXIT $rc"
exit $rc
