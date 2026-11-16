#!/usr/bin/env bash
#SBATCH --job-name=bimanual_hj_train
#SBATCH --partition=unlimited
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a4000:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# bimanual/hj_reachability/train.sbatch
#
# SCAFFOLD ONLY -- do NOT sbatch this yet. See hj_reachability/README.md
# "what's missing before real training" (BimanualArmArmAvoidance dynamics
# class must be added to the DeepReach trainer first).
#
# Submit via a Slurm login node (e.g. rlab1) directly with sbatch, or from a
# workstation without local sbatch via
#   bash hj_reachability/submit_rlab.sh
# which reuses scripts/offload/submit_via_slurm_gateway.sh +
# scripts/offload/slurm_gateway.sh (same gateway pattern as
# scripts/offload/slurm_box_episode_uploader.sbatch).

set -euo pipefail

DEEPREACH_ROOT="${DEEPREACH_ROOT:-/common/users/ss5772/DeepReach/deepreach}"
DEEPREACH_PY="${DEEPREACH_PY:-/common/users/ss5772/miniforge3/envs/vlaism-deepreach/bin/python}"
TRAIN_CONFIG="${TRAIN_CONFIG:-/common/users/ss5772/bimanual/hj_reachability/train_config.yaml}"

cd "${DEEPREACH_ROOT}"
"${DEEPREACH_PY}" run_experiment.py --config "${TRAIN_CONFIG}"
