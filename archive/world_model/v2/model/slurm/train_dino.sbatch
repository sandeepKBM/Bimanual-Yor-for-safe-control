#!/usr/bin/env bash
#SBATCH --job-name=wm2_dino
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/wm_v2/runs/slurm_%x_%j.out
# OPTION B: frozen DINOv2-small + latent predictor, ONE ensemble member per job.
# Ensemble = N independent single-GPU jobs (see make_members.sh), merged afterwards.
set -euo pipefail
MEMBER=0                     # <- make_members.sh rewrites this line per job
STEPS=30000
source /common/users/ss5772/bimanual/world_model/v2/model/slurm/common_env.sh
OUT=$RUN_ROOT/dino/m$MEMBER; mkdir -p "$OUT"
# token cache (idempotent: skips episodes already cached; ~10 min for ~800k frames)
if [ ! -f "$FEATS/cache_meta.json" ]; then
  $PY -m world_model.v2.model.wm_dino cache --prep "$PREP" --out "$FEATS" --device cuda --img-px 126 --pool 3
fi
$PY -m world_model.v2.model.wm_dino train --prep "$PREP" --feats "$FEATS" --out "$OUT" --steps $STEPS \
    --members 1 --seed $((1000 + MEMBER)) --cam-weight zed=2 --action-key action_cup --device cuda --log-every 500
date -u +%FT%TZ > "$OUT/_DONE"
