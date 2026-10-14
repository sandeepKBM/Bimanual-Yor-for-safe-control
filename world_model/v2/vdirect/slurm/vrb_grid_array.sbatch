#!/usr/bin/env bash
#SBATCH --job-name=vrb_grid
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-26%5
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_rb/logs/slurm_%x_%A_%a.out
# PREREG_ADDENDUM_RB grid: one array task = one (arm, budget, seed) cell of vrb_tasks.txt (27 lines: R 0-8, Rc 9-17,
# Rb 18-26). Submit 0-17 now and 18-26 with --dependency=afterok:<vrb_nom>. Reads vdirect_ilab/shared READ-ONLY via
# the vdirect_rb/shared symlink; writes only under VD_ROOT. SIMULATION / SAVED DATA ONLY: no hardware, no network.
# Resumable: a cell with _DONE is skipped.
set -uo pipefail
export VD_ROOT=/common/users/ss5772/bimanual_data/vdirect_rb VD_SRC_ROOT=/common/users/ss5772/bimanual_data/vdirect_rb
export TMPDIR=/tmp/ss5772_vrb_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $VD_ROOT/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
[ -f $VD_ROOT/shared/_DONE_SETUP ] || { echo "shared/ missing"; exit 3; }
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/vdirect/slurm/vrb_tasks.txt)
TID=$(echo $ARGS | awk '{print $2"_b"$4"_s"$6}')
echo "[vrb_grid] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} cell=$TID gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
if [ -f $VD_ROOT/runs/$TID/_DONE ]; then echo "skip (done)"; rc=0
else $PY -m world_model.v2.vdirect.run_rb $ARGS; rc=$?; fi
echo "EXIT $TID $rc"
rm -rf $TMPDIR
exit $rc
