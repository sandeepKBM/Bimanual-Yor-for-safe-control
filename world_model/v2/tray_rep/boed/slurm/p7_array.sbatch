#!/usr/bin/env bash
#SBATCH --job-name=expC
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/boed_p7/logs_c/slurm_%x_%A_%a.out
# Experiment C (P7 / P7s on the frozen BOED protocol; prereg bimanual_data/tray_rep/boed_p7/PREREG_C_FROZEN.md). Copy of
# boed_array.sbatch with the temp folder on the shared data disk (node /tmp has a 2 GB quota), emptied with find -delete, and its own
# task file p7_tasks_test.txt. Output under boed_p7/ only; the frozen boed/ folder is read (table) and never written.
# SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: rerun = resume.
set -uo pipefail
export TMPDIR=/common/users/ss5772/bimanual_data/slurm_tmp/expC_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $TMPDIR
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/tray_rep/boed/slurm/p7_tasks_test.txt)
echo "[expC] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} args=$ARGS $(date -u +%FT%TZ)"
$PY -m world_model.v2.tray_rep.boed.run $ARGS --workers 2
rc=$?
find "$TMPDIR" -delete 2>/dev/null
echo "EXIT $rc"
exit $rc
