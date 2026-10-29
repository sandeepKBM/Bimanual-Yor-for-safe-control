#!/usr/bin/env bash
# agentchord_probe/e0_grid.sh -- E0 reproduction: 3 tasks x plan source {paper, terra} x p in {0, 0.05, 0.10} x N episodes.
#   paper = the plans shipped in their repo (embodichain/database/examples/<task>, copied to runs/plans/e0_<task>_paper)
#   terra = plans written once by gpt-5.6-terra via Codex (runs/plans/e0_<task>_terra; terra2 = regenerated with textures)
#   paperfix = paper plans with stale hold_lost thresholds 0.05 -> 0.01 (see plans/e0_<task>_paperfix/PATCH.json)
#   env SRCS picks sources (default "paperfix terra2")
# All runs replay existing plans (no model calls). Holds the westeros GPU lock; the 6 (task, source) chains run in parallel.
# Usage: e0_grid.sh GPU [N=20] [MODE=upright] [TAG=e0b] [DEVICE=cpu]   (cpu = their CLI default; their atom_actions mix torch and numpy and crash on cuda)
set -uo pipefail
GPU=$1; N=${2:-20}; MODE=${3:-upright}; TAG=${4:-e0b}; DEV=${5:-cpu}
ROOT=/data/local/ss5772/agentchord; L=$(dirname "$(readlink -f "$0")")/launch.sh
exec 9>/data/local/ss5772/gpu_locks/gpu$GPU.lock
flock -n 9 || { echo "gpu$GPU locked by someone else"; exit 3; }
echo "$(date -Is) bimanual agentchord_probe $TAG grid pid=$$" > /data/local/ss5772/gpu_locks/gpu$GPU.owner

chain() {  # task source idx
  local T=$1 SRC=$2 K=$3 J=0
  for P in 0 0.05 0.10; do
    PLAN_NAME=e0_${T}_${SRC} AC_PORT=$((8800 + 10*K + J)) nice -n 10 "$L" ${TAG}_${T}_${SRC}_p${P} replay "$GPU" -- \
      --task "$T" --episodes "$N" --recovery --seed $((100*K + J)) --p "$P" --drop-mode "$MODE" --device "$DEV" 9>&-
    J=$((J+1))
  done
}
K=0
for T in SinglePourWater DualPourWater Rearrangement; do
  for SRC in ${SRCS:-paperfix terra2}; do chain "$T" "$SRC" "$K" & K=$((K+1)); done
done
wait
rm -f /data/local/ss5772/gpu_locks/gpu$GPU.owner
touch "$ROOT/runs/_${TAG}_DONE"
echo "$TAG grid done $(date -Is)"
