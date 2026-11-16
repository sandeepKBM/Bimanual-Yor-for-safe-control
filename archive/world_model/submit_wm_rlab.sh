#!/usr/bin/env bash
# world_model/submit_wm_rlab.sh
#
# Submit a bimanual offline-world-model training Slurm job from
# arrakis/westeros (or any host without local sbatch), reusing the existing
# gateway machinery (scripts/offload/submit_via_slurm_gateway.sh ->
# scripts/offload/slurm_gateway.sh) verbatim -- only the target sbatch
# script + world-model-specific env vars are new. NOT executed by this task
# (scaffold only, mirrors collection/submit_collect_rlab.sh; wm_train.sbatch
# now calls the real world_model/train_wm.py entrypoint -- see its header).
#
# Example (NOT run by this task):
#   COLLECTION_DIR=/common/users/ss5772/bimanual/collection/data/run_001 \
#   DREAMER_FMT_DIR=/common/users/ss5772/bimanual/world_model/data/dreamer_fmt/train_001 \
#   LOGDIR=/common/users/ss5772/bimanual/world_model/logdir/wm_run_001 \
#   TRAIN_STEPS=2000 EVAL_HOLDOUT=5 \
#   bash world_model/submit_wm_rlab.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CONDA_ENV="${CONDA_ENV:-mujoco_ur5e}"
export COLLECTION_DIR="${COLLECTION_DIR:-${REPO_ROOT}/collection/data/run_001}"
export DREAMER_FMT_DIR="${DREAMER_FMT_DIR:-${REPO_ROOT}/world_model/data/dreamer_fmt/train_001}"
export CONFIG="${CONFIG:-${REPO_ROOT}/world_model/train_wm_config.yaml}"
export LOGDIR="${LOGDIR:-${REPO_ROOT}/world_model/logdir/wm_run_001}"
export TRAIN_STEPS="${TRAIN_STEPS:-2000}"
export EVAL_EVERY="${EVAL_EVERY:-50}"
export CKPT_EVERY="${CKPT_EVERY:-50}"
export EVAL_HOLDOUT="${EVAL_HOLDOUT:-5}"
export BIMANUAL_ROOT="${REPO_ROOT}"

bash "${REPO_ROOT}/scripts/offload/submit_via_slurm_gateway.sh" world_model/wm_train.sbatch
