#!/usr/bin/env bash
#SBATCH --job-name=vrb_analyze
#SBATCH --partition=unlimited
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_rb/logs/slurm_%x_%j.out
# PREREG_ADDENDUM_RB analysis (analyze_rb.py), CPU. Submit with --dependency=afterany:<vrb_cons>. Writes
# vdirect_rb/analysis/{tables_rb.md, summary_rb.json} and the sentinel analysis/_DONE. SAVED DATA ONLY.
set -uo pipefail
export TMPDIR=/tmp/ss5772_vrban_${SLURM_JOB_ID:-local}; mkdir -p $TMPDIR
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc MPLCONFIGDIR=$TMPDIR/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
$PY -m world_model.v2.vdirect.analyze_rb; rc=$?
date -u +%FT%TZ > /common/users/ss5772/bimanual_data/vdirect_rb/analysis/_DONE
echo "EXIT analyze $rc"
rm -rf $TMPDIR
exit $rc
