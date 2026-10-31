#!/bin/bash
# LoRA fine-tune launcher for an `srun --overlap` step on a SHARED GPU hold.
#   GPU=2 RUN=smoke50 ETA=15min bash train_lora.sh [train_lora.py args...]
# Stage 1 snapshots this script + train_lora.py to node-local disk and execs the copy (editing NFS scripts mid-run kills steps).
# Stage 2 takes the per-GPU flock, writes the .owner file, runs python as a child, removes .owner on exit.
set -u
if [ -z "${QWENFT_SNAP:-}" ]; then
  SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  SNAP=/freespace/local/ss5772/qwenft_run/$(date +%Y%m%d_%H%M%S)_$$
  mkdir -p "$SNAP" && cp "$SRC/train_lora.sh" "$SRC/train_lora.py" "$SNAP/" || { echo "snapshot failed"; exit 2; }
  export QWENFT_SNAP=$SNAP
  exec bash "$SNAP/train_lora.sh" "$@"
fi
SNAP=$QWENFT_SNAP
GPU=${GPU:?set GPU=<index of the GPU in the shared hold>}
RUN=${RUN:-lora_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-/filer/tmp1/ss5772/ft_runs/$RUN}
mkdir -p "$OUT"
LOCKS=/freespace/local/ss5772/gpu_locks
mkdir -p "$LOCKS"
exec {fd}>"$LOCKS/gpu$GPU.lock"
flock -n $fd || { echo "gpu$GPU locked by: $(cat $LOCKS/gpu$GPU.owner 2>/dev/null)"; exit 3; }
echo "bimanual qwenft-lora $RUN step=${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-} host=$(hostname) pid=$$ since=$(date -Is) eta=${ETA:-?}" > "$LOCKS/gpu$GPU.owner"
cleanup() { rm -f "$LOCKS/gpu$GPU.owner"; }
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES=$GPU
export TMPDIR=/freespace/local/ss5772/tmp_qwenft_$$; mkdir -p "$TMPDIR"
export PYTHONNOUSERSITE=1 PYTHONPATH=/common/users/ss5772/bimanual
export HF_HOME=/filer/tmp1/ss5772/huggingface HF_HUB_CACHE=/filer/tmp1/ss5772/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYBIN=${PYBIN:-/filer/tmp1/ss5772/envs/qwenft/bin/python}
exec > >(tee -a "$OUT/train.log") 2>&1
echo "== $(date -Is) host=$(hostname) gpu=$GPU run=$RUN snapshot=$SNAP"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
PY=""
trap 'kill -TERM $PY 2>/dev/null; wait $PY 2>/dev/null; exit 143' TERM INT   # python saves a checkpoint on SIGTERM
"$PYBIN" "$SNAP/train_lora.py" --run-name "$RUN" --out "$OUT" "$@" &
PY=$!
wait $PY; rc=$?
echo "== $(date -Is) exit rc=$rc"
exit $rc
