#!/usr/bin/env bash
#SBATCH --job-name=tr_prep
#SBATCH --partition=unlimited
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/logs/slurm_%x_%A.out
# tray_rep phase 1b CPU chain, in order: data (both obs modes) -> oracle on TEST -> norm + branches (R, Rc at 300/1000/3000,
# seeds 0-2) -> CL references. Slurm gives 2 CPUs per job, so WORKERS=2: use the westeros launcher (16 workers) when you can.
# For the oracle an array of shards is faster: see tr_oracle_array.sbatch (then `oracle merge`).
# SIMULATION / SAVED DATA ONLY. Every step is resumable.
set -uo pipefail
R=/common/users/ss5772/bimanual_data/tray_rep; WORKERS=${WORKERS:-2}
export TMPDIR=/tmp/ss5772_trprep_${SLURM_JOB_ID:-local}; mkdir -p $TMPDIR $R/logs
export XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba PYTHONPATH=/common/users/ss5772/bimanual
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python; M=world_model.v2.tray_rep
cd /common/users/ss5772/bimanual
for m in full tray_only; do
  $PY -m $M.gen_nominal --config piper --obs-mode $m --n-eps 3000 --workers $WORKERS || exit 1
  $PY -m $M.gen_nominal --config piper --obs-mode $m --n-eps 3000 --merge || exit 1
done
D_FULL=$R/data_piper_full; D_TO=$R/data_piper_tray_only
$PY -m $M.oracle run --data $D_FULL --split test --n 2000 --workers $WORKERS --tag test_v0 || exit 1
mkdir -p $D_TO/oracle && cp $D_FULL/oracle/test_v0.* $D_TO/oracle/
for D in $D_FULL $D_TO; do
  $PY -m $M.wm.prep norm --data $D --workers $WORKERS || exit 1
  $PY -m $M.wm.prep branches --data $D --laws R,Rc --budgets 300,1000,3000 --seeds 0,1,2 --workers $WORKERS || exit 1
  $PY -m $M.wm.prep refs --data $D || exit 1
done
date -u +%FT%TZ > $R/_DONE_prep
rm -rf $TMPDIR
