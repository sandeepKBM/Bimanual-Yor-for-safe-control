#!/usr/bin/env bash
#SBATCH --job-name=vrb_cons
#SBATCH --partition=unlimited
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --array=0-26%16
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_rb/logs/slurm_%x_%A_%a.out
# Conservatism arrays for the 27 RB-grid runs (cb_eval.py cons, unchanged; CPU only), one task per vrb_tasks.txt line.
# Submit with --dependency=afterany:<vrb_grid jobs>. Reads runs READ-ONLY; writes only $VD_ROOT/cb/cons/<run>.npz.
# SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: an output with ._DONE is skipped.
set -uo pipefail
export VD_ROOT=/common/users/ss5772/bimanual_data/vdirect_rb
export TMPDIR=/tmp/ss5772_vrbcons_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $VD_ROOT/cb/cons
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc MPLCONFIGDIR=$TMPDIR/mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/vdirect/slurm/vrb_tasks.txt)
TID=$(echo $ARGS | awk '{print $2"_b"$4"_s"$6}')
OUT=$VD_ROOT/cb/cons/$TID.npz
echo "[vrb_cons] host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0} run=$TID $(date -u +%FT%TZ)"
if [ -f $OUT._DONE ]; then echo "skip (done)"; rc=0
elif [ ! -f $VD_ROOT/runs/$TID/_DONE ]; then echo "run not done -> skipped"; rc=4
else $PY -m world_model.v2.vdirect.cb_eval cons --run $VD_ROOT/runs/$TID --out $OUT; rc=$?; fi
echo "EXIT $TID $rc"
rm -rf $TMPDIR
exit $rc
