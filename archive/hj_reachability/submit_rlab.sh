#!/usr/bin/env bash
# bimanual/hj_reachability/submit_rlab.sh
#
# SCAFFOLD ONLY -- do NOT run this yet (see README.md "what's missing before
# real training"). This is the rlab-gateway submission entry point for the
# bimanual arm-arm-avoidance DeepReach training job: it reuses
# scripts/offload/submit_via_slurm_gateway.sh (which sources
# scripts/offload/slurm_gateway.sh) exactly the way
# scripts/offload/run_box_episode_uploader.sh submits
# scripts/offload/slurm_box_episode_uploader.sbatch -- only the target sbatch
# script differs (hj_reachability/train.sbatch instead of the box uploader).
#
# Usage (from the bimanual repo root, on arrakis/westeros or rlab1):
#   bash hj_reachability/submit_rlab.sh
#
# Env overrides: SLURM_GATEWAY_HOST (default rlab1), SLURM_GATEWAY_USER.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[submit_rlab] SCAFFOLD/DATA ONLY -- refusing to submit." >&2
echo "[submit_rlab] BimanualArmArmAvoidance now EXISTS (branch bimanual-armarm, uncommitted," >&2
echo "[submit_rlab] working-tree-only, in /common/users/ss5772/DeepReach/deepreach) and has been" >&2
echo "[submit_rlab] import/forward/backward smoke-tested against the FAITHFUL-proxy 200k" >&2
echo "[submit_rlab] datasets/bimanual_yor_v2 dataset, but no real training run has happened, and" >&2
echo "[submit_rlab] the fork branch is not merged to main. See ${REPO_ROOT}/hj_reachability/README.md" >&2
echo "[submit_rlab] 'What's missing before real training'. Exiting without sbatch." >&2
exit 1

# --- real launch (left in place, unreachable, for when the above is resolved) ---
# bash "${REPO_ROOT}/scripts/offload/submit_via_slurm_gateway.sh" hj_reachability/train.sbatch
