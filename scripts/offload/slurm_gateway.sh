#!/usr/bin/env bash
# bimanual/scripts/offload/slurm_gateway.sh
#
# Slurm gateway helpers for workstations without a local sbatch (e.g. arrakis, westeros).
#
# Adapted from:
#   /common/users/ss5772/VLAism/ur5e_benchmark/scripts/slurm_gateway.sh
# Only REPO/REPO_ROOT default was changed to point at the bimanual project;
# gateway host default (rlab1) and all logic are unchanged.
#
# Usage:
#   source scripts/offload/slurm_gateway.sh
#   slurm_gateway_squeue -u "$USER"
#   slurm_gateway_scancel 157427
#   slurm_gateway_run bash scripts/offload/slurm_box_episode_uploader.sbatch
#
# Env:
#   SLURM_GATEWAY_HOST   default: rlab1
#   SLURM_GATEWAY_USER   default: current USER
#   REPO                 default: /common/users/ss5772/bimanual (used by slurm_gateway_run)

set -euo pipefail

SLURM_GATEWAY_HOST="${SLURM_GATEWAY_HOST:-rlab1}"
SLURM_GATEWAY_USER="${SLURM_GATEWAY_USER:-${USER:-ss5772}}"

slurm_gateway_has_local() {
  command -v sbatch >/dev/null 2>&1
}

slurm_gateway_workstation_class() {
  local host
  host="$(hostname -s 2>/dev/null || hostname)"
  case "${host}" in
    arrakis|westeros)
      echo "workstation"
      ;;
    rlab*|ilab*)
      echo "slurm_login"
      ;;
    *)
      if slurm_gateway_has_local; then
        echo "slurm_login"
      else
        echo "workstation"
      fi
      ;;
  esac
}

slurm_gateway_remote() {
  ssh -o BatchMode=yes "${SLURM_GATEWAY_USER}@${SLURM_GATEWAY_HOST}" "$@"
}

slurm_gateway_exec() {
  if slurm_gateway_has_local; then
    "$@"
  else
    slurm_gateway_remote "$@"
  fi
}

slurm_gateway_squeue() {
  slurm_gateway_exec squeue "$@"
}

slurm_gateway_scancel() {
  slurm_gateway_exec scancel "$@"
}

slurm_gateway_sbatch() {
  slurm_gateway_exec sbatch "$@"
}

slurm_gateway_sacct() {
  slurm_gateway_exec sacct "$@"
}

slurm_gateway_sinfo() {
  slurm_gateway_exec sinfo "$@"
}

# Run a command in REPO on the gateway host (for submit wrappers that call sbatch locally).
slurm_gateway_run() {
  local repo="${REPO:-/common/users/ss5772/bimanual}"
  if slurm_gateway_has_local; then
    (cd "${repo}" && "$@")
  else
    slurm_gateway_remote "cd '${repo}' && $*"
  fi
}

slurm_gateway_print_status() {
  local class
  class="$(slurm_gateway_workstation_class)"
  echo "[slurm_gateway] workstation_class=${class} host=$(hostname -s 2>/dev/null || hostname)"
  echo "[slurm_gateway] gateway_host=${SLURM_GATEWAY_HOST} gateway_user=${SLURM_GATEWAY_USER}"
  if slurm_gateway_has_local; then
    echo "[slurm_gateway] local sbatch=yes (submit directly on this host)"
  else
    echo "[slurm_gateway] local sbatch=no (use slurm_gateway_run or ssh ${SLURM_GATEWAY_HOST})"
  fi
}
