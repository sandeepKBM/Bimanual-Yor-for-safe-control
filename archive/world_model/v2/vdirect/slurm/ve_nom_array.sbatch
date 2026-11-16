#!/usr/bin/env bash
#SBATCH --job-name=ve_nom
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_ext/logs/slurm_%x_%A_%a.out
# Seed extension (PREREG_ADDENDUM_E), stage 1: NOM pre-trains for seeds 3-5 x members 0-2 (member 0 also trains the
# round-0 probe value). shared/ is a symlink to the ilab grid's (read-only). SIMULATION ONLY. Resumable via _DONE.
set -uo pipefail
export VD_ROOT=/common/users/ss5772/bimanual_data/vdirect_ext VD_SRC_ROOT=/common/users/ss5772/bimanual_data/vdirect_ext
export TMPDIR=/tmp/ss5772_ve_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
i=${SLURM_ARRAY_TASK_ID:-0}; s=$(( 3 + i / 3 )); m=$(( i % 3 ))
echo "[ve_nom] host=$(hostname) task=$i seed=$s member=$m $(date -u +%FT%TZ)"
if [ -f $VD_ROOT/runs/nom_s${s}_m${m}/_DONE ]; then rc=0; echo "skip (done)"
else $PY -m world_model.v2.vdirect.run_arm --arm NOM --seed $s --member $m; rc=$?; fi
echo "EXIT NOM_s${s}_m${m} $rc"
rm -rf $TMPDIR
exit $rc
