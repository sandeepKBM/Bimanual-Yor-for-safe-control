#!/bin/bash
# ilabu (user-approved 11-03): serve one VLM with vLLM on 4 A4000s (one NUMA group), run contents_eval ask, stop.
# Weights are read from an NFS HF cache (no local disk on ilabu); TMPDIR/caches on /var/tmp (10 GB quota).
# Memory: all ss5772 processes on ilabu share one 43.7 GB cgroup (systemd-oomd kills the whole slice at 60% pressure),
# so the launcher logs memory.current every 30 s and stops the server above MEM_STOP_GB.
#   MID=Qwen/Qwen3.6-27B-FP8 NAME=qwen3.6-27b-fp8 GPUS=4,5,6,7 HUB=/common/users/ss5772/bimanual_data/hf/hub bash ilabu_serve_ask.sh
set -u
MID=${MID:?}; NAME=${NAME:?}; GPUS=${GPUS:-4,5,6,7}; PORT=${PORT:-8793}
HUB=${HUB:-/common/users/ss5772/bimanual_data/hf/hub}
REPO=/common/users/ss5772/bimanual/.claude/worktrees/contents-armside2
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
SIF=/filer/tmp1/ss5772/containers/vllm-openai_cosmos3.sif
REC=${REC:-/common/users/ss5772/bimanual_data/vlm_contents/v1/records.jsonl}
OUT=${OUT:-/common/users/ss5772/bimanual_data/vlm_contents/v1/answers}
PROMPTS=${PROMPTS:-"v1 v2"}; MAXTOK=${MAXTOK:-8192}; THINK=${THINK:-on}; GPU_UTIL=${GPU_UTIL:-0.80}; MAXSEQ=${MAXSEQ:-8}; MEM_STOP_GB=${MEM_STOP_GB:-38}
LOGS=$OUT/logs; mkdir -p $LOGS
T=/var/tmp/ss5772/bim_vlm; mkdir -p $T/tmp $T/cache
export HF_HUB_CACHE=$HUB HUGGINGFACE_HUB_CACHE=$HUB HF_HUB_OFFLINE=1 TMPDIR=$T/tmp VLLM_CACHE_ROOT=$T/cache XDG_CACHE_HOME=$T/cache
export PYTHONNOUSERSITE=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_USE_DEEP_GEMM=0
LOCKS=/var/tmp/ss5772/gpu_locks; mkdir -p $LOCKS
for g in ${GPUS//,/ }; do
  exec {fd}>$LOCKS/gpu$g.lock; flock -n $fd || { echo "gpu$g locked: $(cat $LOCKS/gpu$g.owner 2>/dev/null)"; exit 3; }
  echo "$(date -Is) bimanual vlm $NAME pid=$$ port=$PORT until=done" > $LOCKS/gpu$g.owner
done
export CUDA_VISIBLE_DEVICES=$GPUS
CG=/sys/fs/cgroup/user.slice/user-$(id -u).slice
VPID=""; MPID=""
cleanup() { [ -n "$MPID" ] && kill $MPID 2>/dev/null; [ -n "$VPID" ] && kill $VPID 2>/dev/null; sleep 5; [ -n "$VPID" ] && kill -9 $VPID 2>/dev/null
            for g in ${GPUS//,/ }; do echo "$(date -Is) released by bimanual vlm $NAME" > $LOCKS/gpu$g.owner; done; }
trap cleanup EXIT
hostname; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
# ilabu: singularity.conf points ldconfig at /usr/sbin/ldconfig.real (missing), so --nv finds no driver libs (11-03);
# bind the host driver libs into /.singularity.d/libs ourselves.
NVB=$(ldconfig -p | grep -oE "/usr/lib/x86_64-linux-gnu/lib(cuda|nvidia-[a-z-]+|nvoptix|nvcuvid)\.so\.1$" | sort -u | awk '{printf "%s%s:/.singularity.d/libs/%s", (NR>1?",":""), $0, gensub(".*/","","g",$0)}')
[ -n "$NVB" ] || { echo "no NVIDIA libs found"; exit 5; }
singularity exec --bind /filer/tmp1,/common/users,/var/tmp,$NVB $SIF vllm serve $MID --host 127.0.0.1 --port $PORT \
  --tensor-parallel-size 4 --max-model-len 20480 --gpu-memory-utilization $GPU_UTIL --max-num-seqs $MAXSEQ \
  --limit-mm-per-prompt '{"image":1}' --allowed-local-media-path / ${VLLM_EXTRA:-} > $LOGS/vllm_$NAME.log 2>&1 &
VPID=$!
( while kill -0 $VPID 2>/dev/null; do
    cur=$(cat $CG/memory.current); anon=$(awk '/^anon /{print $2}' $CG/memory.stat)
    echo "$(date +%T) mem_current_GB=$((cur/1000000000)) anon_GB=$((anon/1000000000))" >> $LOGS/mem_$NAME.log
    [ $((anon/1000000000)) -ge $MEM_STOP_GB ] && { echo "anon over $MEM_STOP_GB GB: stopping server" >> $LOGS/mem_$NAME.log; kill $VPID; }
    sleep 30; done ) &
MPID=$!
ok=""
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && { ok=1; break; }
  kill -0 $VPID 2>/dev/null || break
  sleep 10
done
[ -z "$ok" ] && { echo "server for $NAME did not come up"; tail -25 $LOGS/vllm_$NAME.log; exit 4; }
echo "ready $(date -Is)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cd $REPO
for PV in $PROMPTS; do
  SUF=$([ $PV = v1 ] && echo "" || echo "_$PV")
  PYTHONPATH=$REPO $PY -m world_model.v2.vlm.contents_eval ask --endpoint http://127.0.0.1:$PORT/v1 --model $MID \
    --records $REC --out $OUT/answers_$NAME$SUF.jsonl --thinking $THINK --max-tokens $MAXTOK --prompt $PV
  echo "ask $PV exit $? $(date -Is)"
done
