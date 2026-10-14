#!/usr/bin/env bash
#SBATCH --job-name=vd_grid
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-50%12
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_ilab/logs/slurm_%x_%A_%a.out
# WRITTEN BY THE vdirect KILL TEST, NOT SUBMITTED. Owner submits AFTER vd_setup_nom.sbatch finished:
#   sbatch --dependency=afterok:<stage1 job id> vd_grid_array.sbatch     (via the rlab1 gateway)
# One array task = one (arm, budget, seed) cell, from vd_tasks.txt (51 lines, ordered: budgets <= 3k first, then B0,
# 10k, 30k). %12 = the per-user GPU cap. Each cell <= ~1.5 h on an A4000-class GPU (U and B0 are the longest).
# SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: a cell with _DONE is skipped.
set -uo pipefail
export VD_ROOT=/common/users/ss5772/bimanual_data/vdirect_ilab
export TMPDIR=/tmp/ss5772_vd_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $VD_ROOT/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
[ -f $VD_ROOT/_DONE_STAGE1 ] || { echo "stage 1 not done"; exit 3; }
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/vdirect/slurm/vd_tasks.txt)
TID=$(echo $ARGS | awk '{print $2"_b"$4"_s"$6}')
echo "[vd_grid] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} cell=$TID gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
if [ -f $VD_ROOT/runs/$TID/_DONE ]; then echo "skip (done)"; rc=0
else $PY -m world_model.v2.vdirect.run_arm $ARGS; rc=$?; fi
echo "EXIT $TID $rc"
rm -rf $TMPDIR
exit $rc
