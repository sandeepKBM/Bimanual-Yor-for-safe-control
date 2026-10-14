#!/usr/bin/env bash
# bimanual/scripts/offload/submit_via_slurm_gateway.sh
#
# Submit a bimanual Slurm wrapper from arrakis/westeros (or any host without sbatch).
#
# Adapted from:
#   /common/users/ss5772/VLAism/ur5e_benchmark/scripts/submit_via_slurm_gateway.sh
# Only REPO_ROOT (bimanual project root) changed; gateway logic is unchanged.
#
# Example:
#   RUN_NAME=yor_bimanual \
#   SOURCE_ROOT=/common/users/ss5772/bimanual/episodes \
#   SPOOL_ROOT=/common/users/ss5772/bimanual/box_spool \
#   bash scripts/offload/submit_via_slurm_gateway.sh scripts/offload/slurm_box_episode_uploader.sbatch

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=bimanual/scripts/offload/slurm_gateway.sh
source "${REPO_ROOT}/scripts/offload/slurm_gateway.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/offload/submit_via_slurm_gateway.sh <submit_script.sh> [extra env exported by caller]" >&2
  exit 1
fi

SUBMIT_SCRIPT="$1"
shift || true

if [[ ! -f "${REPO_ROOT}/${SUBMIT_SCRIPT}" && ! -f "${SUBMIT_SCRIPT}" ]]; then
  echo "[submit_via_slurm_gateway] missing submit script: ${SUBMIT_SCRIPT}" >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/${SUBMIT_SCRIPT}" ]]; then
  SUBMIT_SCRIPT="${REPO_ROOT}/${SUBMIT_SCRIPT}"
fi

slurm_gateway_print_status
echo "[submit_via_slurm_gateway] script=${SUBMIT_SCRIPT}"

# Bug (found 2026-09-06, job 272949): this used to call
# `slurm_gateway_run "bash '${SUBMIT_SCRIPT}'"`, which on a host with no
# local sbatch (westeros/arrakis) ran the .sbatch file directly as a plain
# bash script on the gateway login node -- the #SBATCH lines are just
# comments to bash, so nothing was ever enqueued (squeue showed nothing;
# the job ran as a stray foreground process on rlab1 instead). Call
# sbatch itself, not bash, on the gateway host.
#
# Some clusters' partitions (confirmed: this one's "unlimited" partition)
# reject an explicit --cpus-per-task on sbatch ("Please do not specify the
# number of CPUs"); strip it here so callers don't have to hand-edit their
# .sbatch files per cluster.
SANITIZED_SCRIPT="${SUBMIT_SCRIPT}"
if grep -qE '^#SBATCH[[:space:]]+--cpus-per-task' "${SUBMIT_SCRIPT}"; then
  # mktemp *next to* the original script (shared NFS path under REPO_ROOT),
  # not $TMPDIR/tmp -- sbatch is invoked over ssh on the gateway host, which
  # does not share a local /tmp with this workstation.
  SANITIZED_SCRIPT="$(mktemp "$(dirname "${SUBMIT_SCRIPT}")/.$(basename "${SUBMIT_SCRIPT}").sanitized.XXXXXX")"
  grep -vE '^#SBATCH[[:space:]]+--cpus-per-task' "${SUBMIT_SCRIPT}" > "${SANITIZED_SCRIPT}"
  chmod +x "${SANITIZED_SCRIPT}"
  echo "[submit_via_slurm_gateway] dropped unsupported --cpus-per-task line -> ${SANITIZED_SCRIPT}"
fi

export REPO="${REPO_ROOT}"
slurm_gateway_sbatch "${SANITIZED_SCRIPT}"
