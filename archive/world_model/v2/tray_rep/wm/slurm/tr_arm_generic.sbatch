#!/usr/bin/env bash
#SBATCH --job-name=tr_arm_gen
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/logs/slurm_%x_%A_%a.out
# Generic tray_rep arm cell (owner, 2026-10-05). One array task = one line of $TASKS. Each line is a full run_arm
# argument list INCLUDING --setup (and --data when the data dir is not data_<setup>_<obs_mode>).
# Submit: sbatch --array=0-N%K --export=ALL,TASKS=<abs path> tr_arm_generic.sbatch
# SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: a cell with _DONE is skipped.
set -uo pipefail
R=/common/users/ss5772/bimanual_data/tray_rep
export TMPDIR=/tmp/ss5772_trg_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $R/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba PYTHONPATH=/common/users/ss5772/bimanual
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" ${TASKS:?set TASKS})
SETUP=$(echo $ARGS | sed -n 's/.*--setup \([^ ]*\).*/\1/p')
TID=$(echo $ARGS | awk '{for(i=1;i<=NF;i++){if($i=="--arm")a=$(i+1);if($i=="--budget")b=$(i+1);if($i=="--seed")s=$(i+1);if($i=="--obs-mode")m=$(i+1)}} END{print a"_"m"_b"b"_s"s}')
echo "[tr_arm_gen] host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0} setup=$SETUP cell=$TID gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
if [ -f $R/runs/$SETUP/$TID/_DONE ]; then echo "skip (done)"; rc=0
else $PY -m world_model.v2.tray_rep.wm.run_arm $ARGS --workers 2; rc=$?; fi
echo "EXIT $TID $rc"
exit $rc
