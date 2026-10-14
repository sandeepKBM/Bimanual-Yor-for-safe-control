#!/usr/bin/env bash
#SBATCH --job-name=wm2_rssm
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/wm_v2/runs/slurm_%x_%j.out
# OPTION A: DreamerV3 RSSM (vendored, read-only) + object-weighted recon, ONE member per job.
# Finite: --steps; resumable (latest.pt every 1000 steps) -> resubmitting the same file continues.
set -euo pipefail
MEMBER=0                     # <- make_members.sh rewrites this line per job
STEPS=100000                 # 50000 for the cheaper variant
source /common/users/ss5772/bimanual/world_model/v2/model/slurm/common_env.sh
OUT=$RUN_ROOT/rssm/m$MEMBER; mkdir -p "$OUT"
$PY -m world_model.v2.model.wm_rssm train --prep "$PREP" --out "$OUT" --steps $STEPS --img 64 --batch 16 --seq 64 \
    --seed $((2000 + MEMBER)) --action-key action_cup --device cuda --log-every 500 --ckpt-every 1000
date -u +%FT%TZ > "$OUT/_DONE"
