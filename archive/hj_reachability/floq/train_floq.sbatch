#!/usr/bin/env bash
#SBATCH --job-name=bimanual_floq_train
#SBATCH --partition=unlimited
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:a4000:1
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# bimanual/hj_reachability/floq/train_floq.sbatch
#
# FLOQ (ur5e_benchmark.floq, FOUND/unmodified) adapted to the bimanual
# arm-arm avoidance kinematic game. Two phases, both on the allocated node:
#   1) generate a real-scale episode dataset (rollout_dataset.py; CPU-only,
#      MuJoCo FK oracle -- ~20 min for 4000 episodes x <=60 steps, see
#      FLOQ_SETUP.md's timing note)
#   2) train_bimanual_floq.py -> ur5e_benchmark.floq.train.train_floq on that
#      dataset (GPU)
#
# Local smoke-tested first (20 episodes / 30 steps / 2 epochs, GPU 4 on
# westeros) -- see FLOQ_SETUP.md for the loss curve. This script is the
# larger, real run.
#
# Submit via a Slurm login node (e.g. rlab1) directly with sbatch, or from a
# workstation without local sbatch via
#   bash scripts/offload/submit_via_slurm_gateway.sh hj_reachability/floq/train_floq.sbatch
#
# Time budget note (2026-09-06, after job 273001): the FLOQ train loop calls
# evaluate_floq_model on the FULL val+test split every single epoch with
# eval_flow_samples=8 (vs flow_samples=1 for training) -- this eval pass, not
# the ~1.4k train batches/epoch, dominates wall clock (empirically: >6 min
# for just 20 train batches + one epoch's val/test eval on a free local GPU;
# job 273001 did not finish even epoch 1 of 15 in 5+ hours on an a4000). At
# that pace 15 epochs plausibly takes on the order of days, not the original
# 12h. --time is therefore set generously (partition "unlimited"'s own
# default) rather than tuned down -- epochs stay at the configured 15 and the
# script still exits (and the node deallocates) as soon as training finishes,
# it just won't be cut off first.

set -euo pipefail

REPO_ROOT="${BIMANUAL_ROOT:-/common/users/ss5772/bimanual}"
FLOQ_DIR="${REPO_ROOT}/hj_reachability/floq"
PY="${FLOQ_PY:-/common/users/ss5772/miniforge3/envs/vlaism-deepreach/bin/python}"

DATASET_ROOT="${FLOQ_DATASET_ROOT:-${FLOQ_DIR}/datasets/bimanual_v1}"
CONFIG="${FLOQ_CONFIG:-${FLOQ_DIR}/configs/bimanual_floq_v1.yaml}"
OUTPUT_DIR="${FLOQ_OUTPUT_DIR:-${FLOQ_DIR}/outputs/runs/bimanual_floq_v1_${SLURM_JOB_ID:-local}}"

export PYTHONUNBUFFERED=1

cd "${FLOQ_DIR}"

# Reuse an already-built episode dataset instead of paying the ~13min
# rollout_dataset.py regeneration cost every run (job 272949's failure mode:
# that regen alone ate into the old 12h budget). Only skip when all expected
# artifacts are present and non-trivial in size; otherwise (re)build.
if [[ -s "${DATASET_ROOT}/manifest.json" && -s "${DATASET_ROOT}/feature_schema.json" \
      && -s "${DATASET_ROOT}/train.csv" && -s "${DATASET_ROOT}/val.csv" && -s "${DATASET_ROOT}/test.csv" ]]; then
  echo "[train_floq.sbatch] reusing existing episode dataset -> ${DATASET_ROOT} (skipping regeneration)"
  cat "${DATASET_ROOT}/manifest.json"
else
  echo "[train_floq.sbatch] generating episode dataset -> ${DATASET_ROOT}"
  "${PY}" rollout_dataset.py \
    --output-root "${DATASET_ROOT}" \
    --num-episodes 4000 \
    --max-steps 60 \
    --seed 20260906
fi

echo "[train_floq.sbatch] training FLOQ -> ${OUTPUT_DIR}"
"${PY}" train_bimanual_floq.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0

# Training is finite (bimanual_floq_v1.yaml: epochs=15, max_train_batches=0
# i.e. all batches, no open-ended loop) so train_bimanual_floq.py exits on
# its own once done -- this script then ends and Slurm releases the node
# immediately, without waiting for --time. Sentinel below is
# belt-and-suspenders for scripts/offload/slurm_autorelease.sh (see
# docs/runbooks/slurm.md) in case a future change makes the command hang.
mkdir -p "${OUTPUT_DIR}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUTPUT_DIR}/_DONE"
echo "[train_floq.sbatch] done, wrote ${OUTPUT_DIR}/_DONE"
