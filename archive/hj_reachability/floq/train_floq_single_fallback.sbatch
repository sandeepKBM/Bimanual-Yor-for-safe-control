#!/usr/bin/env bash
#SBATCH --job-name=bimanual_floq_train_single
#SBATCH --partition=unlimited
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --nodelist=rlab4
#SBATCH --gres=gpu:a4000:1
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# bimanual/hj_reachability/floq/train_floq_single_fallback.sbatch
#
# Single-GPU FALLBACK for the FLOQ relaunch (mission: "maximize GPU use via
# DDP"). We built and locally smoke-tested a working DDP path
# (train_bimanual_floq_ddp.py + train_floq_ddp.sbatch, 2x a4000 via torchrun
# --standalone) -- it is correct (see that file's docstring and the mission
# report), but is NOT runnable right now: live `scontrol show node` checks
# on 2026-09-06 ~16:20-17:05 showed every allowed ilab/rlab node fully
# GPU-allocated except a single free slot on ilab3 (gpu:4500_ada) -- the two
# a4000 nodes (rlab3, rlab4, the only nodes with gpu:a4000 gres) were both
# 8/8 allocated by other users' jobs. Submitting the 2-GPU DDP job (273343)
# confirmed this: Slurm's own backfill scheduler estimated StartTime ~16
# hours out (Reason=Priority, SchedNodeList=rlab4) -- not something a bounded
# "confirm it's alive and stepping, then return" mission should sit and wait
# on. Job 273343 was cancelled; this is the fallback per the mission's own
# instruction ("Fall back to max-single-GPU + the eval-cost fix... Reliability
# + correctness beat GPU count").
#
# This reuses train_bimanual_floq_ddp.py directly (not the old
# train_bimanual_floq.py) launched as plain `python` (no torchrun) --
# WORLD_SIZE/RANK/LOCAL_RANK are unset, so the script's `_dist_state()` takes
# its is_distributed=False branch and behaves as an ordinary single-process
# trainer, EXCEPT it still gets the two eval-cost fixes that help even at
# world_size=1: eval_flow_samples 8->2 in the config (training still uses
# flow_samples=1, unaffected) and skipping the per-epoch TEST-split eval
# (only run once at the end on the best checkpoint; best-checkpoint selection
# was always val_auroc-only, so this changes nothing about which checkpoint
# wins). No DDP-specific code paths execute, so nothing here depends on the
# DDP correctness argument -- this is exactly the single-GPU trainer, just
# with the eval bottleneck fix applied.
#
# If cluster capacity frees up later (`scontrol show node rlab3/rlab4` shows
# >=2 free gpu:a4000), prefer resubmitting train_floq_ddp.sbatch instead --
# it is the faster path once GPUs are actually available.

set -euo pipefail

REPO_ROOT="${BIMANUAL_ROOT:-/common/users/ss5772/bimanual}"
FLOQ_DIR="${REPO_ROOT}/hj_reachability/floq"
PY="${FLOQ_PY:-/common/users/ss5772/miniforge3/envs/vlaism-deepreach/bin/python}"

DATASET_ROOT="${FLOQ_DATASET_ROOT:-${FLOQ_DIR}/datasets/bimanual_v1}"
CONFIG="${FLOQ_CONFIG:-${FLOQ_DIR}/configs/bimanual_floq_v1.yaml}"
OUTPUT_DIR="${FLOQ_OUTPUT_DIR:-${FLOQ_DIR}/outputs/runs/bimanual_floq_v1_single_${SLURM_JOB_ID:-local}}"

export PYTHONUNBUFFERED=1

cd "${FLOQ_DIR}"

echo "[train_floq_single_fallback.sbatch] host: $(hostname)"
echo "[train_floq_single_fallback.sbatch] SLURM_JOB_ID: ${SLURM_JOB_ID:-<none>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[train_floq_single_fallback.sbatch] visible GPUs:"
  nvidia-smi -L || true
fi

if [[ -s "${DATASET_ROOT}/manifest.json" && -s "${DATASET_ROOT}/feature_schema.json" \
      && -s "${DATASET_ROOT}/train.csv" && -s "${DATASET_ROOT}/val.csv" && -s "${DATASET_ROOT}/test.csv" ]]; then
  echo "[train_floq_single_fallback.sbatch] reusing existing episode dataset -> ${DATASET_ROOT} (skipping regeneration)"
  cat "${DATASET_ROOT}/manifest.json"
else
  echo "[train_floq_single_fallback.sbatch] generating episode dataset -> ${DATASET_ROOT}"
  "${PY}" rollout_dataset.py \
    --output-root "${DATASET_ROOT}" \
    --num-episodes 4000 \
    --max-steps 60 \
    --seed 20260906
fi

echo "[train_floq_single_fallback.sbatch] training FLOQ (single GPU, eval-cost fix) -> ${OUTPUT_DIR}"
"${PY}" train_bimanual_floq_ddp.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUTPUT_DIR}/_DONE"
echo "[train_floq_single_fallback.sbatch] done, wrote ${OUTPUT_DIR}/_DONE"
