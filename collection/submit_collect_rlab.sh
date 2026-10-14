#!/usr/bin/env bash
# collection/submit_collect_rlab.sh
#
# Submit a bimanual data-collection Slurm job from arrakis/westeros (or any
# host without local sbatch), reusing the existing gateway machinery
# (scripts/offload/submit_via_slurm_gateway.sh -> scripts/offload/
# slurm_gateway.sh) verbatim -- only the target sbatch script + collection-
# specific env vars are new. NOT executed by this task (scaffold only).
#
# Example:
#   NUM_EPISODES=500 NUM_STEPS=3000 FAMILY=arm_start_shift \
#   OUT_DIR=/common/users/ss5772/bimanual/collection/data/run_001 \
#   bash collection/submit_collect_rlab.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CONDA_ENV="${CONDA_ENV:-mujoco_ur5e}"
export NUM_EPISODES="${NUM_EPISODES:-500}"
export NUM_STEPS="${NUM_STEPS:-3000}"
export RENDER_SIZE="${RENDER_SIZE:-84}"
export RENDER_EVERY="${RENDER_EVERY:-1}"
export SEED_START="${SEED_START:-0}"
export FAMILY="${FAMILY:-arm_start_shift}"
export OUT_DIR="${OUT_DIR:-${REPO_ROOT}/collection/data/run_$(date +%Y%m%d_%H%M%S)}"
export BIMANUAL_ROOT="${REPO_ROOT}"

bash "${REPO_ROOT}/scripts/offload/submit_via_slurm_gateway.sh" collection/collect.sbatch
