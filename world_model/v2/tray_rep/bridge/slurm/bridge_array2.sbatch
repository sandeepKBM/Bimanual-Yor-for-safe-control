#!/usr/bin/env bash
#SBATCH --job-name=bridge
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/bridge/logs/slurm_%x_%A_%a.out
# Copy of bridge_array.sbatch without --cpus-per-task (this cluster's unlimited partition rejects it; Slurm gives
# 2 CPUs per job by default) and without the temp-dir delete. Same arguments: $1 = stage (tune | test).
# SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: rerun = resume.
set -uo pipefail
STAGE=${1:?stage name: tune | test}
R=/common/users/ss5772/bimanual_data/tray_rep
export TMPDIR=/tmp/ss5772_bridge_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $R/bridge/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/tray_rep/bridge/slurm/bridge_tasks_${STAGE}.txt)
echo "[bridge] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} args=$ARGS $(date -u +%FT%TZ)"
$PY -m world_model.v2.tray_rep.bridge.run $ARGS --workers 2
rc=$?
echo "EXIT $rc"
exit $rc
