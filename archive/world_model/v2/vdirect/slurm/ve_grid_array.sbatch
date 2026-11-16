#!/usr/bin/env bash
#SBATCH --job-name=ve_grid
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-35%12
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_ext/logs/slurm_%x_%A_%a.out
# Seed extension (PREREG_ADDENDUM_E), stage 2: one array task = one cell of ve_tasks.txt (seeds 3-5; F/R/Rr at
# 300/1k/3k, U at 1k/3k, B0). Submit with --dependency=afterok:<ve_nom job> and --nice so the addendum-F grid runs first.
# SIMULATION ONLY: no hardware, no network. Resumable: a cell with _DONE is skipped.
set -uo pipefail
export VD_ROOT=/common/users/ss5772/bimanual_data/vdirect_ext VD_SRC_ROOT=/common/users/ss5772/bimanual_data/vdirect_ext
export TMPDIR=/tmp/ss5772_ve_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/vdirect/slurm/ve_tasks.txt)
TID=$(echo $ARGS | awk '{print $2"_b"$4"_s"$6}')
echo "[ve_grid] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} cell=$TID $(date -u +%FT%TZ)"
if [ -f $VD_ROOT/runs/$TID/_DONE ]; then echo "skip (done)"; rc=0
else $PY -m world_model.v2.vdirect.run_arm $ARGS; rc=$?; fi
echo "EXIT $TID $rc"
rm -rf $TMPDIR
exit $rc
