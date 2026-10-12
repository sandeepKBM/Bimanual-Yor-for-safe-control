#!/usr/bin/env bash
#SBATCH --job-name=gen_pc
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/logs/slurm_%x_%A_%a.out
# Point-cloud (pc_est) dataset, camera rig $CAM, sharded: array task i of N runs gen_pc --shard i/N (owner, 2026-10-05).
# Merge afterwards (one quick local call with --merge). CPU only. SIMULATION ONLY: no hardware, no network. Resumable.
set -uo pipefail
R=/common/users/ss5772/bimanual_data/tray_rep
export TMPDIR=/tmp/ss5772_gpc_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-0}
echo "[gen_pc] host=$(hostname) shard=$I/$N cam=${CAM:?set CAM} $(date -u +%FT%TZ)"
$PY -m world_model.v2.tray_rep.perception.gen_pc --config piper --camera $CAM --n-eps 3000 --shard $I/$N \
    --out $R/data_piper_pc_${CAM//+/_} --workers 2; rc=$?
echo "EXIT gen_pc shard $I/$N $rc"
exit $rc
