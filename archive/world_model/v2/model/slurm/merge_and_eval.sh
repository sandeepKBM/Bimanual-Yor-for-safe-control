#!/usr/bin/env bash
# After all member jobs wrote _DONE: merge DINO members, evaluate every option on identical splits.
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
$PY -m world_model.v2.model.wm_dino merge --out $RUN_ROOT/dino/ensemble.pt $RUN_ROOT/dino/m*/model.pt
RSSM=$(ls $RUN_ROOT/rssm/m*/model.pt | paste -sd, -)
$PY -m world_model.v2.eval.run_eval --prep "$PREP" --device cuda --ood ood=$PREP:ood \
    --model clock --model slosh:b3 --model gru:$RUN_ROOT/gru/model.pt \
    --model dino:$RUN_ROOT/dino/ensemble.pt --model rssm:$RSSM --out $RUN_ROOT/eval_all.json | tee $RUN_ROOT/eval_all.txt
$PY -m world_model.v2.model.latency --prep "$PREP" --gru $RUN_ROOT/gru/model.pt \
    --dino $RUN_ROOT/dino/ensemble.pt --rssm "$RSSM" --plans 16 --K 8 | tee $RUN_ROOT/latency.txt
