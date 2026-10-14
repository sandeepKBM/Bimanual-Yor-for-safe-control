#!/usr/bin/env bash
#SBATCH --job-name=expCP8
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/boed_p8/logs_test/slurm_%x_%A_%a.out
# Experiment C, arm P8 (addendum; prereg bimanual_data/tray_rep/boed_p8/PREREG_CP8_FROZEN.md): boed/p8.py, no probe, online belief,
# worst-case re-pick at transport boundaries. Temp folder on the shared data disk (node /tmp 2 GB quota), emptied with find -delete;
# own task file p8_tasks_test.txt. Output under boed_p8/ only; the frozen boed/ folder is read (table) and never written.
# SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: rerun = resume.
set -uo pipefail
export TMPDIR=/common/users/ss5772/bimanual_data/slurm_tmp/expCP8_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $TMPDIR
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/tray_rep/boed/slurm/p8_tasks_test.txt)
echo "[expCP8] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} args=$ARGS $(date -u +%FT%TZ)"
$PY -m world_model.v2.tray_rep.boed.p8 $ARGS --workers 2
rc=$?
find "$TMPDIR" -delete 2>/dev/null
echo "EXIT $rc"
exit $rc
