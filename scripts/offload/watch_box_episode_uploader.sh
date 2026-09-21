#!/usr/bin/env bash
# bimanual/scripts/offload/watch_box_episode_uploader.sh
#
# Adapted from:
#   /common/users/ss5772/VLAism/ur5e_benchmark/scripts/watch_box_episode_uploader.sh
#
# WARNING: this is an infinite poll loop (`while true`). Do not run it from an
# interactive assistant session. Launch it deliberately (e.g. `nohup ... &`,
# `tmux`, or the Slurm sbatch wrapper) when you actually want continuous
# draining of SOURCE_ROOT into Box. See docs/offload.md for launch guidance.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLEEP_MINUTES="${SLEEP_MINUTES:-30}"

if ! [[ "${SLEEP_MINUTES}" =~ ^[0-9]+$ ]] || [[ "${SLEEP_MINUTES}" -lt 1 ]]; then
  echo "SLEEP_MINUTES must be a positive integer." >&2
  exit 2
fi

while true; do
  echo "[watch_box_uploader_bimanual] pass_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bash "${REPO_ROOT}/scripts/offload/run_box_episode_uploader.sh"
  echo "[watch_box_uploader_bimanual] pass_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep "$((SLEEP_MINUTES * 60))"
done
