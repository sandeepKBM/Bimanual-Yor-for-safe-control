#!/usr/bin/env bash
#SBATCH --job-name=vrb_nom
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=6-8
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_rb/logs/slurm_%x_%A_%a.out
# PREREG_ADDENDUM_RB: NOM m0 pre-train + round-0 probe value for fresh seeds 6-8 (run_arm.py --arm NOM, unchanged).
# Array index = seed. SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: a run with _DONE is skipped.
set -uo pipefail
export VD_ROOT=/common/users/ss5772/bimanual_data/vdirect_rb VD_SRC_ROOT=/common/users/ss5772/bimanual_data/vdirect_rb
export TMPDIR=/tmp/ss5772_vrb_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $VD_ROOT/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
S=${SLURM_ARRAY_TASK_ID:-6}
echo "[vrb_nom] host=$(hostname) job=${SLURM_JOB_ID:-local} seed=$S gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
if [ -f $VD_ROOT/runs/nom_s${S}_m0/_DONE ]; then echo "skip (done)"; rc=0
else $PY -m world_model.v2.vdirect.run_arm --arm NOM --seed $S --member 0; rc=$?; fi
echo "EXIT nom_s${S}_m0 $rc"
rm -rf $TMPDIR
exit $rc
