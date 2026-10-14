#!/usr/bin/env bash
#SBATCH --job-name=bimanual_box_uploader
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# bimanual/scripts/offload/slurm_box_episode_uploader.sbatch
#
# Adapted from:
#   /common/users/ss5772/VLAism/ur5e_benchmark/scripts/slurm_box_episode_uploader.sbatch
#
# Single-pass Slurm job that runs the bimanual Box uploader wrapper once.
# Submit via sbatch directly on a Slurm login node (e.g. rlab1), or from a
# workstation without local sbatch via
# scripts/offload/submit_via_slurm_gateway.sh.

set -euo pipefail

REPO_ROOT="${BIMANUAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${SPOOL_ROOT:?SPOOL_ROOT is required}"
: "${RUN_NAME:?RUN_NAME is required}"

# Recommended but optional if LOCAL_REMOTE_ROOT is used.
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
# NOTE: collector-name is a closed enum in the reused CLI
# (openvla|openpi|ball|t5|auto|unknown) -- see bimanual/offload/config.sh.
COLLECTOR_NAME="${COLLECTOR_NAME:-unknown}"

export SOURCE_ROOT
export SPOOL_ROOT
export RUN_NAME
export COLLECTOR_NAME
export RCLONE_REMOTE
export LOCAL_REMOTE_ROOT="${LOCAL_REMOTE_ROOT:-}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-}"
export RCLONE_PASSWORD_COMMAND="${RCLONE_PASSWORD_COMMAND:-}"
export BW_LIMIT="${BW_LIMIT:-}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export BIMANUAL_ROOT="${REPO_ROOT}"
export UR5E_BENCHMARK_SRC="${UR5E_BENCHMARK_SRC:-/common/users/ss5772/VLAism/ur5e_benchmark/src}"

bash "${REPO_ROOT}/scripts/offload/run_box_episode_uploader.sh"
