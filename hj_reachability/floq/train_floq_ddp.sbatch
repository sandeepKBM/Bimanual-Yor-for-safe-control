#!/usr/bin/env bash
#SBATCH --job-name=bimanual_floq_train_ddp
#SBATCH --partition=unlimited
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:a4000:2
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# bimanual/hj_reachability/floq/train_floq_ddp.sbatch
#
# Multi-GPU (DDP, single node) relaunch of train_floq.sbatch. Same two
# phases (reuse episode dataset, then train), but training now runs via
# torchrun -> hj_reachability/floq/train_bimanual_floq_ddp.py, which
# DDP-wraps ur5e_benchmark.floq's FOUND model/losses (imported unmodified)
# and -- critically -- also SHARDS the val/test eval forward pass across
# ranks (not just training), since the vendor per-epoch full-split eval at
# eval_flow_samples=8 is what actually dominated wall clock in the
# single-GPU runs (272949/273001/273272), not the ~1.4k train batches/epoch.
# See train_bimanual_floq_ddp.py's module docstring for the full rationale
# and docs/runbooks/slurm.md / docs/runbooks/slurm_rules_vendor_skill.md for
# the node-selection reasoning.
#
# NODE PINNING NOTE: no --nodelist here on purpose. Live `scontrol show node`
# checks during setup (2026-09-06, ~16:20-16:57) showed the cluster's a4000
# capacity (rlab3, rlab4 -- the only two nodes with gpu:a4000 gres) swinging
# from 2-3 free on rlab4 down to fully allocated (0 free anywhere, including
# every other allowed ilab/rlab node except a single free 4500_ada on ilab3)
# within about 20 minutes, driven by other users' churn (several short-lived
# array-job tasks). Given that volatility, pinning a specific node risked
# pending on a node that looked free at write-time but wasn't by submit-time;
# `--gres=gpu:a4000:2` (no --nodelist) lets Slurm place the job on whichever
# of rlab3/rlab4 actually has 2 free a4000s the moment it schedules.
# world_size=2 (not 3+) both because it's a clean divisor of the configured
# batch_size=256 (effective_global_batch stays exactly 256) and because 2 was
# the largest count with any real chance of being free without a long pend
# given the observed churn -- see the mission report for the exact
# free/allocated numbers at each check.
#
# Batch size choice: global_batch (config: 256) is held constant across GPUs
# (per-GPU batch = 256 // world_size = 128 at world_size=2) rather than kept
# at 256/GPU -- this preserves the single-GPU run's optimization semantics
# (gradient noise scale, no LR retune needed) exactly, with no rounding
# remainder at world_size=2. See train_bimanual_floq_ddp.py's train_floq_ddp()
# log line for the exact
# resolved numbers at runtime.
#
# Time budget note (inherited from train_floq.sbatch, 2026-09-06): even with
# DDP, --time stays generous (this partition's "unlimited" default) rather
# than tuned down, per the repo's slurm-generous-time-autorelease rule --
# epochs stay at the configured 15 and the script exits (freeing the node)
# as soon as training finishes.

set -euo pipefail

REPO_ROOT="${BIMANUAL_ROOT:-/common/users/ss5772/bimanual}"
FLOQ_DIR="${REPO_ROOT}/hj_reachability/floq"
PY="${FLOQ_PY:-/common/users/ss5772/miniforge3/envs/vlaism-deepreach/bin/python}"
TORCHRUN="${FLOQ_TORCHRUN:-/common/users/ss5772/miniforge3/envs/vlaism-deepreach/bin/torchrun}"

DATASET_ROOT="${FLOQ_DATASET_ROOT:-${FLOQ_DIR}/datasets/bimanual_v1}"
CONFIG="${FLOQ_CONFIG:-${FLOQ_DIR}/configs/bimanual_floq_v1.yaml}"
OUTPUT_DIR="${FLOQ_OUTPUT_DIR:-${FLOQ_DIR}/outputs/runs/bimanual_floq_v1_ddp_${SLURM_JOB_ID:-local}}"
NUM_GPUS="${FLOQ_NUM_GPUS:-2}"
MASTER_PORT="${FLOQ_MASTER_PORT:-29513}"

export PYTHONUNBUFFERED=1

cd "${FLOQ_DIR}"

echo "[train_floq_ddp.sbatch] host: $(hostname)"
echo "[train_floq_ddp.sbatch] SLURM_JOB_ID: ${SLURM_JOB_ID:-<none>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[train_floq_ddp.sbatch] visible GPUs:"
  nvidia-smi -L || true
fi

# Reuse the already-built episode dataset instead of paying the rollout
# regeneration cost every run (see train_floq.sbatch's same guard).
if [[ -s "${DATASET_ROOT}/manifest.json" && -s "${DATASET_ROOT}/feature_schema.json" \
      && -s "${DATASET_ROOT}/train.csv" && -s "${DATASET_ROOT}/val.csv" && -s "${DATASET_ROOT}/test.csv" ]]; then
  echo "[train_floq_ddp.sbatch] reusing existing episode dataset -> ${DATASET_ROOT} (skipping regeneration)"
  cat "${DATASET_ROOT}/manifest.json"
else
  echo "[train_floq_ddp.sbatch] generating episode dataset -> ${DATASET_ROOT}"
  "${PY}" rollout_dataset.py \
    --output-root "${DATASET_ROOT}" \
    --num-episodes 4000 \
    --max-steps 60 \
    --seed 20260906
fi

echo "[train_floq_ddp.sbatch] training FLOQ (DDP, ${NUM_GPUS} GPUs) -> ${OUTPUT_DIR}"
"${TORCHRUN}" --standalone --nproc_per_node="${NUM_GPUS}" --master-port="${MASTER_PORT}" \
  train_bimanual_floq_ddp.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}"

# Training is finite (bimanual_floq_v1.yaml: epochs=15, max_train_batches=0
# i.e. all batches) so torchrun's worker processes exit on their own once
# done -- this script then ends and Slurm releases the node immediately.
# Sentinel below is belt-and-suspenders for scripts/offload/slurm_autorelease.sh
# (see docs/runbooks/slurm.md) in case a future change makes a rank hang.
mkdir -p "${OUTPUT_DIR}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUTPUT_DIR}/_DONE"
echo "[train_floq_ddp.sbatch] done, wrote ${OUTPUT_DIR}/_DONE"
