#!/bin/bash
# Serve one VLM with vLLM (TP over the given GPUs), run a command file against it, stop.  User-approved 11-03 (ilabu GPUs,
# ilab1 Slurm).  No robot, no cloud API.  Works on ilabu directly (GPU flock + cgroup memory guard) and inside a Slurm job.
#   MID=Qwen/Qwen3.6-27B-FP8 NAME=qwen3.6-27b-fp8 GPUS=4,5,6,7 CMDS=<bash file using $ENDPOINT $MID $NAME> bash vlm_serve_run.sh
# The launcher copies itself to local tmp and execs the copy (never edit an NFS launcher while a job reads it).
set -u
if [ -z "${_VSR_COPY:-}" ]; then
  T0=${TMPROOT:-/var/tmp/ss5772}/bim_vlm; [ -n "${SLURM_JOB_ID:-}" ] && T0=/tmp/ss5772_vlm_$SLURM_JOB_ID
  mkdir -p $T0; cp "$0" $T0/vsr_$$.sh; [ -n "${CMDS:-}" ] && cp "$CMDS" $T0/cmds_$$.sh && export CMDS=$T0/cmds_$$.sh
  _VSR_COPY=1 exec bash $T0/vsr_$$.sh
fi
MID=${MID:?}; NAME=${NAME:?}; CMDS=${CMDS:?}; PORT=${PORT:-8794}
HUB=${HUB:-/common/users/ss5772/bimanual_data/hf/hub}
REPO=/common/users/ss5772/bimanual/.claude/worktrees/contents-armside2
SIF=/filer/tmp1/ss5772/containers/vllm-openai_cosmos3.sif
OUT=${OUT:-/common/users/ss5772/bimanual_data/vlm_contents/v1/answers}
GPU_UTIL=${GPU_UTIL:-0.80}; MAXSEQ=${MAXSEQ:-8}; MAXLEN=${MAXLEN:-20480}; MEM_STOP_GB=${MEM_STOP_GB:-38}
LOGS=$OUT/logs; mkdir -p $LOGS
if [ -n "${SLURM_JOB_ID:-}" ]; then T=/tmp/ss5772_vlm_$SLURM_JOB_ID; else T=/var/tmp/ss5772/bim_vlm; fi
mkdir -p $T/tmp $T/cache
export HF_HUB_CACHE=$HUB HUGGINGFACE_HUB_CACHE=$HUB HF_HUB_OFFLINE=1 TMPDIR=$T/tmp VLLM_CACHE_ROOT=$T/cache XDG_CACHE_HOME=$T/cache
export PYTHONNOUSERSITE=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_USE_DEEP_GEMM=0
if [ -z "${SLURM_JOB_ID:-}" ]; then
  GPUS=${GPUS:?}; LOCKS=/var/tmp/ss5772/gpu_locks; mkdir -p $LOCKS
  for g in ${GPUS//,/ }; do
    exec {fd}>$LOCKS/gpu$g.lock; flock -n $fd || { echo "gpu$g locked: $(cat $LOCKS/gpu$g.owner 2>/dev/null)"; exit 3; }
    echo "$(date -Is) bimanual vlm $NAME pid=$$ port=$PORT until=done" > $LOCKS/gpu$g.owner
  done
  export CUDA_VISIBLE_DEVICES=$GPUS
fi
NG=$(echo ${CUDA_VISIBLE_DEVICES} | tr , '\n' | wc -l); TP=${TP:-$NG}
# GPU libs: ilabu's singularity.conf points ldconfig at a missing ldconfig.real, so --nv finds nothing there (11-03).
if [ -x /usr/sbin/ldconfig.real ]; then NV=(--nv); else
  NVB=$(ldconfig -p | grep -oE "/usr/lib/x86_64-linux-gnu/lib(cuda|nvidia-[a-z-]+|nvoptix|nvcuvid)\.so\.1$" | sort -u | awk '{printf "%s%s:/.singularity.d/libs/%s", (NR>1?",":""), $0, gensub(".*/","","g",$0)}')
  [ -n "$NVB" ] || { echo "no NVIDIA libs found"; exit 5; }; NV=(--bind $NVB); fi
VPID=""; MPID=""
cleanup() { [ -n "$MPID" ] && kill $MPID 2>/dev/null; [ -n "$VPID" ] && kill $VPID 2>/dev/null; sleep 5; [ -n "$VPID" ] && kill -9 $VPID 2>/dev/null
            [ -z "${SLURM_JOB_ID:-}" ] && for g in ${GPUS//,/ }; do echo "$(date -Is) released by bimanual vlm $NAME" > $LOCKS/gpu$g.owner; done; }
trap cleanup EXIT
hostname; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
case $NAME in cosmos3-nano) EXTRA=(--mm-encoder-tp-mode data) ;; *) EXTRA=() ;; esac
singularity exec "${NV[@]}" --bind /filer/tmp1,/common/users,/var/tmp${BIND_EXTRA:+,$BIND_EXTRA} $SIF vllm serve $MID "${EXTRA[@]}" --host 127.0.0.1 --port $PORT \
  --tensor-parallel-size $TP --max-model-len $MAXLEN --gpu-memory-utilization $GPU_UTIL --max-num-seqs $MAXSEQ \
  --limit-mm-per-prompt '{"image":1}' --allowed-local-media-path / ${VLLM_EXTRA:-} > $LOGS/vllm_${NAME}_run.log 2>&1 &
VPID=$!
CG=/sys/fs/cgroup/user.slice/user-$(id -u).slice
if [ -z "${SLURM_JOB_ID:-}" ] && [ -f $CG/memory.stat ]; then
  ( while kill -0 $VPID 2>/dev/null; do
      anon=$(awk '/^anon /{print $2}' $CG/memory.stat)
      echo "$(date +%T) anon_GB=$((anon/1000000000))" >> $LOGS/mem_${NAME}_run.log
      [ $((anon/1000000000)) -ge $MEM_STOP_GB ] && { echo "anon over $MEM_STOP_GB GB: stopping server" >> $LOGS/mem_${NAME}_run.log; kill $VPID; }
      sleep 30; done ) &
  MPID=$!
fi
ok=""
for i in $(seq 1 150); do
  curl -s -m 5 http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && { ok=1; break; }
  kill -0 $VPID 2>/dev/null || break
  sleep 10
done
[ -z "$ok" ] && { echo "server for $NAME did not come up"; tail -25 $LOGS/vllm_${NAME}_run.log; exit 4; }
echo "ready $(date -Is)"
export ENDPOINT=http://127.0.0.1:$PORT/v1 MID NAME OUT REPO
cd $REPO && bash $CMDS
echo "cmds exit $? $(date -Is)"
