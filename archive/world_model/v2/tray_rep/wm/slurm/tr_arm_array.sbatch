#!/usr/bin/env bash
#SBATCH --job-name=tr_arm
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-41%6
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/logs/slurm_%x_%A_%a.out
# tray_rep phase 1b: one array task = one (arm, budget, seed, obs-mode) cell = one line of tr_tasks.txt (42 lines:
# N0 0-5, R 6-23, Rc 24-41). Run tr_prep.sbatch first (data, oracle, norm, branches). To run only N0 first: --array=0-5.
# SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: a cell with _DONE is skipped.
set -uo pipefail
R=/common/users/ss5772/bimanual_data/tray_rep
export TMPDIR=/tmp/ss5772_tr_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $R/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba PYTHONPATH=/common/users/ss5772/bimanual
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/tray_rep/wm/slurm/tr_tasks.txt)
TID=$(echo $ARGS | awk '{print $2"_"$8"_b"$4"_s"$6}')
echo "[tr_arm] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} cell=$TID gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
if [ -f $R/runs/piper/$TID/_DONE ]; then echo "skip (done)"; rc=0
else $PY -m world_model.v2.tray_rep.wm.run_arm $ARGS --setup piper --workers 2; rc=$?; fi
echo "EXIT $TID $rc"
rm -rf $TMPDIR
exit $rc
