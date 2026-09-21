#!/usr/bin/env bash
# bimanual/scripts/offload/run_box_episode_uploader.sh
#
# Thin adapter that invokes the reused ur5e_benchmark Box episode uploader
# engine (python -m ur5e_benchmark.cli.box_episode_uploader) with bimanual
# paths/names. No engine code is copied here; PYTHONPATH points at the
# read-only source project (see bimanual/offload/config.sh for the port
# rationale and the collector-name enum constraint).
#
# Adapted from:
#   /common/users/ss5772/VLAism/ur5e_benchmark/scripts/run_box_episode_uploader.sh
#
# One-pass run. For a poll loop see watch_box_episode_uploader.sh (NOT started
# automatically -- see docs/offload.md).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=bimanual/offload/config.sh
source "${REPO_ROOT}/offload/config.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"

EPISODE_ALLOWLIST_PATH="${EPISODE_ALLOWLIST_PATH:-}"
LOCAL_REMOTE_ROOT="${LOCAL_REMOTE_ROOT:-}"
BW_LIMIT="${BW_LIMIT:-}"
PRUNE_AFTER_UPLOAD="${PRUNE_AFTER_UPLOAD:-}"
DELETE_AFTER_UPLOAD="${DELETE_AFTER_UPLOAD:-}"
I_UNDERSTAND_DELETE_SOURCE="${I_UNDERSTAND_DELETE_SOURCE:-}"
KEEP_LOCAL_ARCHIVES="${KEEP_LOCAL_ARCHIVES:-}"
RECURSIVE_DEPTH="${RECURSIVE_DEPTH:-1}"
EPISODES_PER_ARCHIVE="${EPISODES_PER_ARCHIVE:-200}"
TARGET_ARCHIVE_GB="${TARGET_ARCHIVE_GB:-5}"
MAX_ARCHIVE_GB="${MAX_ARCHIVE_GB:-10}"
ABSOLUTE_BOX_FILE_LIMIT_GB="${ABSOLUTE_BOX_FILE_LIMIT_GB:-50}"
HARD_SAFETY_FILE_LIMIT_GB="${HARD_SAFETY_FILE_LIMIT_GB:-45}"
MAX_BATCHES_PER_PASS="${MAX_BATCHES_PER_PASS:-2}"
MAX_EPISODES_PER_PASS="${MAX_EPISODES_PER_PASS:-}"
MIN_AGE_SECONDS="${MIN_AGE_SECONDS:-60}"
TRANSFERS="${TRANSFERS:-2}"
CHECKERS="${CHECKERS:-4}"
UPLOAD_ATTEMPTS="${UPLOAD_ATTEMPTS:-5}"
DRY_RUN="${DRY_RUN:-}"

if [[ -z "${SOURCE_ROOT}" ]]; then
  echo "SOURCE_ROOT is required." >&2
  exit 2
fi
if [[ -z "${SPOOL_ROOT}" ]]; then
  echo "SPOOL_ROOT is required." >&2
  exit 2
fi
if [[ -z "${RUN_NAME}" ]]; then
  echo "RUN_NAME is required." >&2
  exit 2
fi
if [[ -z "${RCLONE_REMOTE}" && -z "${LOCAL_REMOTE_ROOT}" ]]; then
  echo "Either RCLONE_REMOTE or LOCAL_REMOTE_ROOT is required." >&2
  exit 2
fi

# Reuse the ur5e_benchmark uploader/archive engine in place (read-only source).
export PYTHONPATH="${UR5E_BENCHMARK_SRC}:${PYTHONPATH:-}"

cmd=(
  "${PYTHON_BIN}"
  -m ur5e_benchmark.cli.box_episode_uploader
  --source-root "${SOURCE_ROOT}"
  --spool-root "${SPOOL_ROOT}"
  --run-name "${RUN_NAME}"
  --collector-name "${COLLECTOR_NAME}"
  --episodes-per-archive "${EPISODES_PER_ARCHIVE}"
  --target-archive-gb "${TARGET_ARCHIVE_GB}"
  --max-archive-gb "${MAX_ARCHIVE_GB}"
  --absolute-box-file-limit-gb "${ABSOLUTE_BOX_FILE_LIMIT_GB}"
  --hard-safety-file-limit-gb "${HARD_SAFETY_FILE_LIMIT_GB}"
  --max-spool-gb 15
  --pause-spool-threshold-gb 12
  --min-free-spool-gb 2
  --min-age-seconds "${MIN_AGE_SECONDS}"
  --upload-attempts "${UPLOAD_ATTEMPTS}"
  --require-rclone-auth-check
  --no-delete-after-upload
  --no-i-understand-delete-source
)
if [[ -n "${EPISODE_ALLOWLIST_PATH}" ]]; then
  cmd+=(--episode-allowlist-path "${EPISODE_ALLOWLIST_PATH}")
fi
cmd+=(--recursive-depth "${RECURSIVE_DEPTH}")
if [[ -n "${MAX_BATCHES_PER_PASS}" ]]; then
  cmd+=(--max-batches-per-pass "${MAX_BATCHES_PER_PASS}")
fi
if [[ -n "${MAX_EPISODES_PER_PASS}" ]]; then
  cmd+=(--max-episodes-per-pass "${MAX_EPISODES_PER_PASS}")
fi

if [[ -n "${RCLONE_REMOTE}" ]]; then
  cmd+=(--rclone-remote "${RCLONE_REMOTE}")
fi
if [[ -n "${LOCAL_REMOTE_ROOT}" ]]; then
  cmd+=(--local-remote-root "${LOCAL_REMOTE_ROOT}")
fi
if [[ -n "${RCLONE_CONFIG}" ]]; then
  cmd+=(--rclone-config "${RCLONE_CONFIG}")
fi
if [[ -n "${RCLONE_PASSWORD_COMMAND}" ]]; then
  cmd+=(--rclone-password-command "${RCLONE_PASSWORD_COMMAND}")
fi
if [[ -n "${BW_LIMIT}" ]]; then
  cmd+=(--bwlimit "${BW_LIMIT}")
fi
if [[ -n "${PRUNE_AFTER_UPLOAD}" ]]; then
  case "${PRUNE_AFTER_UPLOAD}" in
    1|true|TRUE|yes|YES|on|ON)
      cmd+=(--prune-after-upload)
      ;;
    0|false|FALSE|no|NO|off|OFF)
      ;;
    *)
      echo "PRUNE_AFTER_UPLOAD must be a boolean-like value if set." >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${DELETE_AFTER_UPLOAD}" ]]; then
  case "${DELETE_AFTER_UPLOAD}" in
    1|true|TRUE|yes|YES|on|ON)
      cmd+=(--delete-after-upload)
      ;;
    0|false|FALSE|no|NO|off|OFF)
      ;;
    *)
      echo "DELETE_AFTER_UPLOAD must be a boolean-like value if set." >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${I_UNDERSTAND_DELETE_SOURCE}" ]]; then
  case "${I_UNDERSTAND_DELETE_SOURCE}" in
    1|true|TRUE|yes|YES|on|ON)
      cmd+=(--i-understand-delete-source)
      ;;
    0|false|FALSE|no|NO|off|OFF)
      ;;
    *)
      echo "I_UNDERSTAND_DELETE_SOURCE must be a boolean-like value if set." >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${KEEP_LOCAL_ARCHIVES}" ]]; then
  case "${KEEP_LOCAL_ARCHIVES}" in
    1|true|TRUE|yes|YES|on|ON)
      cmd+=(--keep-local-archives)
      ;;
    0|false|FALSE|no|NO|off|OFF)
      ;;
    *)
      echo "KEEP_LOCAL_ARCHIVES must be a boolean-like value if set." >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${DRY_RUN}" ]]; then
  case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON)
      cmd+=(--dry-run)
      ;;
    0|false|FALSE|no|NO|off|OFF)
      ;;
    *)
      echo "DRY_RUN must be a boolean-like value if set." >&2
      exit 2
      ;;
  esac
fi

"${cmd[@]}"
