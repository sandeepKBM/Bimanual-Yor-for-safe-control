#!/usr/bin/env bash
#SBATCH --job-name=tr_prep_step
#SBATCH --partition=unlimited
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/logs/slurm_%x_%j.out
# One tray_rep prep step on a dataset dir (owner, 2026-10-05). CPU only (Slurm gives 2 CPUs per job).
# Env: DATA (dataset dir), STEP (norm | branches | refs), SEEDS (for branches, e.g. 0), LAWS (default Rc),
# BUDGETS (default 3000). SIMULATION ONLY: no hardware, no network. Resumable (prep skips finished outputs).
set -uo pipefail
export TMPDIR=/tmp/ss5772_trp_${SLURM_JOB_ID:-local}; mkdir -p $TMPDIR
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
echo "[tr_prep_step] host=$(hostname) step=${STEP:?} data=${DATA:?} seeds=${SEEDS:-} $(date -u +%FT%TZ)"
case $STEP in
  norm)     $PY -m world_model.v2.tray_rep.wm.prep norm --data $DATA --workers 2; rc=$? ;;
  branches) $PY -m world_model.v2.tray_rep.wm.prep branches --data $DATA --laws ${LAWS:-Rc} --budgets ${BUDGETS:-3000} --seeds ${SEEDS:?} --workers 2; rc=$? ;;
  refs)     $PY -m world_model.v2.tray_rep.wm.prep refs --data $DATA; rc=$? ;;
  *) echo "unknown STEP"; rc=2 ;;
esac
echo "EXIT prep $STEP ${SEEDS:-} $rc"
exit $rc
