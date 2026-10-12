#!/bin/bash
#SBATCH --job-name=vlm_bake
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ilab3
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vlm/logs/bake_%j.out
# Sequential bake-off on ONE GPU: for each model, an 8-request smoke, then (if it passed) the full request file.
# usage: sbatch [--export=ALL,MODELS="qwen3.5-9b cosmos-reason1-7b"] bakeoff.sbatch      (REQ defaults to req/bake.jsonl)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME=/freespace/local/ss5772/vlm/hf HUGGINGFACE_HUB_CACHE=/freespace/local/ss5772/vlm/hf/hub TRANSFORMERS_CACHE=/freespace/local/ss5772/vlm/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/freespace/local/ss5772/vlm/tmp; mkdir -p $TMPDIR
D=/common/users/ss5772/bimanual_data/vlm
REQ=${REQ:-$D/req/bake.jsonl}
MODELS=${MODELS:-"qwen3.5-9b cosmos-reason1-7b robobrain2.5-4b molmo2-er qwen3-vl-8b robobrain2.0-7b cosmos-reason2-8b"}
[ -n "$PIP_PKGS" ] && /freespace/local/ss5772/vlm/env/bin/python -m pip install $PIP_PKGS 2>&1 | tail -2   # e.g. PIP_PKGS=einops (Molmo2 remote code)
cd /common/users/ss5772/bimanual; export PYTHONPATH=/common/users/ss5772/bimanual
PY=/freespace/local/ss5772/vlm/env/bin/python
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
for m in $MODELS; do
  echo "=== $m smoke $(date +%T)"
  rm -f $D/out/smoke_$m.jsonl.DONE $D/out/smoke_$m.jsonl.FAIL
  $PY -m world_model.v2.vlm.runner --req $D/req/smoke.jsonl --out $D/out/smoke_$m.jsonl --model $m 2>&1 | tail -15
  if [ -f $D/out/smoke_$m.jsonl.DONE ]; then
    echo "=== $m full $(date +%T)"
    rm -f $D/out/${OUTTAG:-bake}_$m.jsonl.DONE $D/out/${OUTTAG:-bake}_$m.jsonl.FAIL
    $PY -m world_model.v2.vlm.runner --req $REQ --out $D/out/${OUTTAG:-bake}_$m.jsonl --model $m 2>&1 | grep -v "^\[" | tail -5
    echo "=== $m end $(date +%T) $(ls $D/out/${OUTTAG:-bake}_$m.jsonl.DONE $D/out/${OUTTAG:-bake}_$m.jsonl.FAIL 2>/dev/null)"
  else
    echo "=== $m SMOKE FAILED (skipping full)"
  fi
done
echo ALLDONE
