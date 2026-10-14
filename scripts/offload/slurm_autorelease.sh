#!/usr/bin/env bash
# bimanual/scripts/offload/slurm_autorelease.sh
#
# Belt-and-suspenders auto-release watcher for the repo-wide Slurm rule
# (see docs/runbooks/slurm.md): every sbatch should have a GENEROUS --time
# ceiling but the training/job command itself should be finite so the batch
# script exits and Slurm releases the node the moment real work is done --
# never waiting for --time. This watcher is the fallback for a command that
# might hang despite that: it polls for a DONE sentinel file the sbatch
# script writes as its last step, and scancels the job the moment the
# sentinel appears (or once, optionally, a max wait elapses -- see
# --max-wait-seconds).
#
# Usage:
#   bash scripts/offload/slurm_autorelease.sh <job_id> <done_sentinel_path> \
#     [--poll-seconds N] [--max-wait-seconds N]
#
# Run this from wherever squeue/scancel are reachable directly, or source
# scripts/offload/slurm_gateway.sh first and export SLURM_AUTORELEASE_GATEWAY=1
# to go through the gateway helpers (for hosts with no local sbatch/squeue).
#
# Example (FLOQ run):
#   bash scripts/offload/slurm_autorelease.sh 272950 \
#     /common/users/ss5772/bimanual/hj_reachability/floq/outputs/runs/bimanual_floq_v1_272950/_DONE &
#
# Exit codes: 0 = sentinel seen, job released (or already gone).
#             1 = bad usage. 2 = max-wait elapsed without sentinel (job left running).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/offload/slurm_autorelease.sh <job_id> <done_sentinel_path> [--poll-seconds N] [--max-wait-seconds N]" >&2
  exit 1
fi

JOB_ID="$1"
SENTINEL="$2"
shift 2 || true

POLL_SECONDS=30
MAX_WAIT_SECONDS=0   # 0 = no limit; just keep polling until the sentinel appears.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll-seconds)
      POLL_SECONDS="$2"; shift 2 ;;
    --max-wait-seconds)
      MAX_WAIT_SECONDS="$2"; shift 2 ;;
    *)
      echo "[slurm_autorelease] unknown arg: $1" >&2
      exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

_squeue_has_job() {
  if [[ "${SLURM_AUTORELEASE_GATEWAY:-0}" == "1" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/scripts/offload/slurm_gateway.sh"
    slurm_gateway_squeue -h -j "${JOB_ID}" 2>/dev/null | grep -q "${JOB_ID}"
  else
    squeue -h -j "${JOB_ID}" 2>/dev/null | grep -q "${JOB_ID}"
  fi
}

_scancel_job() {
  echo "[slurm_autorelease] sentinel found at ${SENTINEL} -- scancelling job ${JOB_ID} to release the node now"
  if [[ "${SLURM_AUTORELEASE_GATEWAY:-0}" == "1" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/scripts/offload/slurm_gateway.sh"
    slurm_gateway_scancel "${JOB_ID}" || true
  else
    scancel "${JOB_ID}" || true
  fi
}

echo "[slurm_autorelease] watching job=${JOB_ID} sentinel=${SENTINEL} poll=${POLL_SECONDS}s max_wait=${MAX_WAIT_SECONDS}s(0=unbounded)"

elapsed=0
while true; do
  if [[ -e "${SENTINEL}" ]]; then
    _scancel_job
    exit 0
  fi
  if ! _squeue_has_job; then
    echo "[slurm_autorelease] job ${JOB_ID} is no longer in squeue (finished/cancelled elsewhere) -- nothing to do"
    exit 0
  fi
  if [[ "${MAX_WAIT_SECONDS}" -gt 0 && "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]]; then
    echo "[slurm_autorelease] max-wait-seconds elapsed with no sentinel -- job ${JOB_ID} left running, exiting watcher" >&2
    exit 2
  fi
  sleep "${POLL_SECONDS}"
  elapsed=$((elapsed + POLL_SECONDS))
done
