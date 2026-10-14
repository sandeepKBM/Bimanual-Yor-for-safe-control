#!/usr/bin/env bash
#SBATCH --job-name=bimanual_collect
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#
# collection/collect.sbatch
#
# Scaffold, modeled on scripts/offload/slurm_box_episode_uploader.sbatch --
# single-pass Slurm job that runs collection/collect.py once. Submit via
# sbatch directly on a Slurm login node, or from a workstation without local
# sbatch via collection/submit_collect_rlab.sh. NOT executed by this task.

set -euo pipefail

REPO_ROOT="${BIMANUAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONDA_ENV="${CONDA_ENV:-mujoco_ur5e}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/collection/data/run_${SLURM_JOB_ID:-local}}"
NUM_EPISODES="${NUM_EPISODES:-500}"
NUM_STEPS="${NUM_STEPS:-3000}"
RENDER_SIZE="${RENDER_SIZE:-84}"
RENDER_EVERY="${RENDER_EVERY:-1}"
SEED_START="${SEED_START:-0}"
FAMILY="${FAMILY:-arm_start_shift}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
python -m collection.collect \
  --out-dir "${OUT_DIR}" \
  --num-episodes "${NUM_EPISODES}" \
  --num-steps "${NUM_STEPS}" \
  --render-size "${RENDER_SIZE}" \
  --render-every "${RENDER_EVERY}" \
  --seed-start "${SEED_START}" \
  --family "${FAMILY}"
