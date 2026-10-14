#!/usr/bin/env bash
#SBATCH --job-name=vf_oc
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_f/logs/slurm_%x_%A_%a.out
# Outcome-conformal audit (PREREG_ADDENDUM_F, secondary): one array task = one finished run, from the list file in
# OC_LIST (lines "<run root> <run name>"). Submit with --array=0-<n-1>%k --export=ALL,OC_LIST=<file>:
#   vf_oc_grid_runs.txt (the 51 ilab grid runs; no dependency) and vf_oc_new_runs.txt (the 33 F-grid runs;
#   --dependency=afterany:<vf_grid job>). Reads runs READ-ONLY; writes only $VD_ROOT/oc/<run>.json.
# SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: an output with ._DONE is skipped.
set -uo pipefail
export VD_ROOT=/common/users/ss5772/bimanual_data/vdirect_f
export VD_OC_CACHE=$VD_ROOT/shared_oc
export TMPDIR=/tmp/ss5772_vfoc_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $VD_ROOT/logs $VD_ROOT/oc
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
read RROOT NAME <<< "$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" ${OC_LIST:?OC_LIST not set})"
OUT=$VD_ROOT/oc/$NAME.json
echo "[vf_oc] host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0} run=$RROOT/runs/$NAME gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
if [ -f $OUT._DONE ]; then echo "skip (done)"; rc=0
elif [ ! -f $RROOT/runs/$NAME/_DONE ]; then echo "run not done -> skipped"; rc=4
else $PY -m world_model.v2.vdirect.oc_eval --run $RROOT/runs/$NAME --out $OUT; rc=$?; fi
echo "EXIT $NAME $rc"
rm -rf $TMPDIR
exit $rc
