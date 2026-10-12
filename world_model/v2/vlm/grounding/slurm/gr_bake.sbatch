#!/bin/bash
#SBATCH --job-name=gr_bake
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ilab3
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vlm/grounding/logs/gr_%j.out
# Grounding bake-off: per model a smoke (8 req) then the full request file. usage: sbatch [--export=ALL,MODELS="a b"] gr_bake.sbatch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME=/freespace/local/ss5772/vlm/hf HUGGINGFACE_HUB_CACHE=/freespace/local/ss5772/vlm/hf/hub TRANSFORMERS_CACHE=/freespace/local/ss5772/vlm/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/freespace/local/ss5772/vlm/tmp; mkdir -p $TMPDIR
D=/common/users/ss5772/bimanual_data/vlm/grounding
MODELS=${MODELS:-"qwen3-vl-8b robobrain2.5-4b qwen3.5-9b molmo2-er"}
cd /common/users/ss5772/bimanual; export PYTHONPATH=/common/users/ss5772/bimanual
PY=/freespace/local/ss5772/vlm/env/bin/python
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
if [ -n "$DOWNLOAD" ]; then   # approved: anonymous HF download of OPEN weights (no token); gated -> skipped
  for r in $DOWNLOAD; do HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download as s;print('OK',s('$r'))" 2>&1 | tail -2 | cut -c1-300; done
fi
for m in $MODELS; do
  echo "=== $m smoke $(date +%T)"
  rm -f $D/out/smoke_$m.jsonl.*
  $PY -m world_model.v2.vlm.grounding.gr_runner --req $D/req/gr_smoke_$m.jsonl --out $D/out/smoke_$m.jsonl --model $m 2>&1 | tail -12
  if [ -f $D/out/smoke_$m.jsonl.DONE ]; then
    echo "=== $m full $(date +%T)"
    rm -f $D/out/gr_$m.jsonl.DONE $D/out/gr_$m.jsonl.FAIL
    $PY -m world_model.v2.vlm.grounding.gr_runner --req $D/req/gr_$m.jsonl --out $D/out/gr_$m.jsonl --model $m 2>&1 | grep -v "^\[" | tail -5
    echo "=== $m end $(date +%T) $(ls $D/out/gr_$m.jsonl.DONE $D/out/gr_$m.jsonl.FAIL 2>/dev/null)"
  else
    echo "=== $m SMOKE FAILED (skipping full)"
  fi
done
echo ALLDONE
