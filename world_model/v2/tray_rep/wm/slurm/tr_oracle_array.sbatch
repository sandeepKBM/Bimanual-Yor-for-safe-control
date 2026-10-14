#!/usr/bin/env bash
#SBATCH --job-name=tr_oracle
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/logs/slurm_%x_%A_%a.out
# tray_rep oracle on TEST states, 16 shards. Needs data_piper_full merged. After all tasks:
#   $PY -m world_model.v2.tray_rep.oracle merge --data $R/data_piper_full --tag test_v0 --n-parts 16
#   mkdir -p $R/data_piper_tray_only/oracle && cp $R/data_piper_full/oracle/test_v0.* $R/data_piper_tray_only/oracle/
# SIMULATION / SAVED DATA ONLY.
set -uo pipefail
R=/common/users/ss5772/bimanual_data/tray_rep
export TMPDIR=/tmp/ss5772_tror_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $R/logs
export XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba PYTHONPATH=/common/users/ss5772/bimanual
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
$PY -m world_model.v2.tray_rep.oracle run --data $R/data_piper_full --split test --n 2000 --workers 2 --tag test_v0 --shard ${SLURM_ARRAY_TASK_ID}/16
rc=$?
rm -rf $TMPDIR
exit $rc
