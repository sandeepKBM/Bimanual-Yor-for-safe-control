#!/usr/bin/env bash
# bimanual/offload/config.sh
#
# Bimanual defaults for the Box episode uploader adapter.
#
# PORT NOTE (read this before editing):
#   The upload/archive engine itself is NOT copied into this repo. We reuse
#   it in place from the source project via PYTHONPATH:
#     /common/users/ss5772/VLAism/ur5e_benchmark/src
#       -> ur5e_benchmark.cli.box_episode_uploader
#       -> ur5e_benchmark.archive.box_episode_archive
#   Those modules were checked and found import-light: stdlib only, plus one
#   tiny local helper (ur5e_benchmark.utils.episode_markers, itself stdlib
#   only). Importing the top-level `ur5e_benchmark` package additionally
#   pulls in PyYAML (for ur5e_benchmark.config), which is present in the
#   `safe_env` and `fiper` conda envs. This file and the scripts in
#   bimanual/scripts/offload/ are a thin adapter layer: bimanual-specific
#   paths/names in, `python -m ur5e_benchmark.cli.box_episode_uploader` out.
#   The source project is treated as READ-ONLY from here.
#
# KNOWN CONSTRAINT (do not "fix" by editing the source CLI):
#   --collector-name in box_episode_uploader.py is a closed argparse enum:
#     openvla | openpi | ball | t5 | auto | unknown
#   "yor_bimanual" is therefore NOT a legal --collector-name value. We use
#   RUN_NAME=yor_bimanual (a free-form string) to carry the bimanual identity
#   in archive/remote filenames, and default COLLECTOR_NAME to "unknown".
#   Override COLLECTOR_NAME only with one of the allowed enum values above.
#
# AUTH:
#   Never put real rclone tokens/passwords/config in this repo. This file
#   only references secret material by PATH, matching the existing
#   ur5e_benchmark convention:
#     RCLONE_CONFIG            -> $HOME/.config/rclone/rclone.conf (if present)
#     RCLONE_PASSWORD_COMMAND  -> $HOME/.secrets/rclone-pass.sh (if present)
#   If those files don't exist yet, see docs/offload.md for the (manual,
#   one-time, outside-of-this-repo) rclone setup steps. This adapter does not
#   create or print secret contents.

set -euo pipefail

# --- Reused engine location (read-only source project) ---
UR5E_BENCHMARK_SRC="${UR5E_BENCHMARK_SRC:-/common/users/ss5772/VLAism/ur5e_benchmark/src}"

# --- Bimanual-specific defaults ---
SOURCE_ROOT="${SOURCE_ROOT:-/common/users/ss5772/bimanual/episodes}"
SPOOL_ROOT="${SPOOL_ROOT:-/common/users/ss5772/bimanual/box_spool}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rutgers_box:UR5E_ARCHIVE/bimanual}"
RUN_NAME="${RUN_NAME:-yor_bimanual}"
COLLECTOR_NAME="${COLLECTOR_NAME:-unknown}"

# --- Auth: reference existing secrets by path only; never recreate/print them ---
if [[ -z "${RCLONE_CONFIG:-}" && -f "${HOME}/.config/rclone/rclone.conf" ]]; then
  RCLONE_CONFIG="${HOME}/.config/rclone/rclone.conf"
fi
if [[ -z "${RCLONE_PASSWORD_COMMAND:-}" && -x "${HOME}/.secrets/rclone-pass.sh" ]]; then
  RCLONE_PASSWORD_COMMAND="${HOME}/.secrets/rclone-pass.sh"
fi
RCLONE_CONFIG="${RCLONE_CONFIG:-}"
RCLONE_PASSWORD_COMMAND="${RCLONE_PASSWORD_COMMAND:-}"

export UR5E_BENCHMARK_SRC
export SOURCE_ROOT SPOOL_ROOT RCLONE_REMOTE RUN_NAME COLLECTOR_NAME
export RCLONE_CONFIG RCLONE_PASSWORD_COMMAND
