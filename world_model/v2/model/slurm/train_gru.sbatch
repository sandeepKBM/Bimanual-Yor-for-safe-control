#!/usr/bin/env bash
#SBATCH --job-name=wm2_gru
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/wm_v2/runs/slurm_%x_%j.out
# Proprio-history GRU baseline (5-member ensemble in ONE job: minutes). Finite: --steps.
set -euo pipefail
source /common/users/ss5772/bimanual/world_model/v2/model/slurm/common_env.sh
OUT=$RUN_ROOT/gru; mkdir -p "$OUT"
$PY -m world_model.v2.model.proprio_gru train --prep "$PREP" --out "$OUT" --steps 5000 --members 5 \
    --action-key action_cup --device cuda
$PY -m world_model.v2.eval.run_eval --prep "$PREP" --model gru:$OUT/model.pt --model slosh:b3 --model clock \
    --ood ood=$PREP:ood --device cuda --out "$OUT/eval.json" | tee "$OUT/eval.txt"
date -u +%FT%TZ > "$OUT/_DONE"
