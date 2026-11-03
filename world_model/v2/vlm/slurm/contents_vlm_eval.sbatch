#!/bin/bash
#SBATCH -J bim_vlm_contents
#SBATCH -w ilab1
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -o /common/users/ss5772/bimanual_data/vlm_contents/slurm_%j.out
# Hidden-contents VLM test (user-approved 11-03): serve each model with vLLM (TP=4) on node-local weights, run
# contents_eval ask on the same records, stop the server, next model.  No robot, no cloud API; HF downloads only.
#   MODELS="cosmos3-nano qwen3.6-27b-fp8 qwen3.5-9b"   REC=<records.jsonl on NFS>   OUT=<dir on NFS>   THINK=on
set -u
REPO=/common/users/ss5772/bimanual/.claude/worktrees/contents-armside2
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
SIF=/filer/tmp1/ss5772/containers/vllm-openai_cosmos3.sif
NL=${NL:-/freespace/local/ss5772/vlm_contents}      # node-local stage; ilab1 /freespace was 100% full on 11-03
HUB=${HUB:-$NL/hf/hub}                              # HF hub cache to serve from (e.g. /filer/tmp1/ss5772/huggingface/hub, NFS)
MODELS=${MODELS:-"cosmos3-nano qwen3.6-27b-fp8 qwen3.5-9b"}
REC=${REC:-/common/users/ss5772/bimanual_data/vlm_contents/v1/records.jsonl}
OUT=${OUT:-/common/users/ss5772/bimanual_data/vlm_contents/v1/answers}
THINK=${THINK:-on}; PORT=${PORT:-8791}; TP=${TP:-4}; GPU_UTIL=${GPU_UTIL:-0.70}; MAXSEQ=${MAXSEQ:-8}
LOGS=$OUT/logs; TMPD=${TMPD:-/tmp/ss5772_vlm_$SLURM_JOB_ID}
mkdir -p $LOGS $TMPD $OUT; mkdir -p $HUB 2>/dev/null
export HF_HUB_CACHE=$HUB HUGGINGFACE_HUB_CACHE=$HUB TMPDIR=$TMPD PYTHONNOUSERSITE=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_USE_DEEP_GEMM=0
RUN=(singularity exec --nv --bind /filer/tmp1,/common/users,/freespace/local $SIF)
hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader; df -h /freespace/local | tail -1
VPID=""
cleanup() { [ -n "$VPID" ] && kill $VPID 2>/dev/null; sleep 5; [ -n "$VPID" ] && kill -9 $VPID 2>/dev/null; }
trap cleanup EXIT

for NAME in $MODELS; do
  case $NAME in
    cosmos3-nano)    MID=nvidia/Cosmos3-Nano;   EXTRA=(--mm-encoder-tp-mode data) ;;
    qwen3.6-27b-fp8) MID=Qwen/Qwen3.6-27B-FP8;  EXTRA=() ;;
    qwen3.5-9b)      MID=Qwen/Qwen3.5-9B;       EXTRA=() ;;
    *) echo "unknown $NAME"; continue ;;
  esac
  echo "=== $NAME ($MID) $(date -Is)"
  if [ ! -d $HUB/models--${MID//\//--}/snapshots ]; then
    "${RUN[@]}" hf download $MID --exclude "assets/*" --exclude "images/*" > $LOGS/download_$NAME.log 2>&1 \
      || { echo "download failed: $(tail -3 $LOGS/download_$NAME.log)"; continue; }
  fi
  HF_HUB_OFFLINE=1 "${RUN[@]}" vllm serve $MID "${EXTRA[@]}" --host 127.0.0.1 --port $PORT --tensor-parallel-size $TP \
    --max-model-len 16384 --gpu-memory-utilization $GPU_UTIL --max-num-seqs $MAXSEQ --limit-mm-per-prompt {\"image\":1} --allowed-local-media-path / ${VLLM_EXTRA:-} \
    > $LOGS/vllm_$NAME.log 2>&1 &
  VPID=$!
  ok=""
  for i in $(seq 1 180); do
    curl -s -m 5 http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && { ok=1; break; }
    kill -0 $VPID 2>/dev/null || break
    sleep 10
  done
  if [ -z "$ok" ]; then echo "server for $NAME did not come up:"; tail -20 $LOGS/vllm_$NAME.log; cleanup; VPID=""; continue; fi
  echo "ready $(date -Is)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  cd $REPO && PYTHONPATH=$REPO $PY -m world_model.v2.vlm.contents_eval ask --endpoint http://127.0.0.1:$PORT/v1 --model $MID \
    --records $REC --out $OUT/answers_$NAME.jsonl --thinking $THINK --max-tokens 4096
  echo "ask exit $? $(date -Is)"
  cleanup; VPID=""
done
echo "all done $(date -Is)"
