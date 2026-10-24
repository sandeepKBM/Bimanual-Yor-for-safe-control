#!/bin/bash
# Cosmos 3 REASONER (vLLM, OpenAI API on 127.0.0.1) + the vlm file-queue daemon in front of it, on one GPU of one node.
# Runs as an srun step (e.g. inside a held allocation) or under sbatch. First run builds the env + downloads the weights
# into node-local /freespace (needs internet on the node); later runs are offline.
#   MODEL=cosmos3-edge|cosmos3-nano  GPU=<index or list to flock, e.g. 2,3; empty = use what Slurm gave>  TP=1  PORT=8771
#   SETUP_ONLY=1: build env + download weights, then exit (no GPU needed)   MAXLEN=16384  GPU_UTIL=0.88
#   RUNTIME=sif (default if the image exists): NVIDIA's vllm/vllm-openai:cosmos3 image (vLLM 0.23.1, registers Cosmos3-Edge AND Nano)
#   RUNTIME=venv: pip vLLM 0.21 + vllm-cosmos3 plugin (Nano/Super only; Edge's cosmos3_edge type is NOT registered there)
#   clients: BIMANUAL_VLM_SVC=/common/users/ss5772/bimanual_data/vlm/svc_$MODEL  (world_model.v2.vlm.service)
#   stop:    BIMANUAL_VLM_SVC=... python -m world_model.v2.vlm.service stop   (daemon exits, then vLLM is killed)
set -u
MODEL=${MODEL:-cosmos3-edge}; PORT=${PORT:-8771}; TP=${TP:-1}; MAXLEN=${MAXLEN:-16384}; GPU_UTIL=${GPU_UTIL:-0.88}
case $MODEL in
  cosmos3-edge) MID=nvidia/Cosmos3-Edge; EXTRA=() ;;
  cosmos3-nano) MID=nvidia/Cosmos3-Nano; EXTRA=(--hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' --mm-encoder-tp-mode data) ;;
  *) echo "unknown MODEL $MODEL"; exit 2 ;;
esac
ROOT=/freespace/local/ss5772/cosmos3
HF=/filer/tmp1/ss5772/huggingface               # the user's HF cache (~/.bashrc; no quota); weights live here, the env is node-local
export UV_CACHE_DIR=$ROOT/uvcache UV_PYTHON_INSTALL_DIR=$ROOT/python TMPDIR=$ROOT/tmp
export HF_HOME=$HF HF_HUB_CACHE=$HF/hub HUGGINGFACE_HUB_CACHE=$HF/hub TRANSFORMERS_CACHE=$HF/hub
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_USE_DEEP_GEMM=0
mkdir -p $ROOT/tmp $ROOT/logs
LOCKS=/freespace/local/ss5772/gpu_locks
if [ -n "${GPU:-}" ]; then                      # shared hold: Slurm stacks --overlap steps on one GPU, so lock each one (GPU=2 or GPU=2,3)
  mkdir -p $LOCKS
  for g in ${GPU//,/ }; do
    exec {fd}>$LOCKS/gpu$g.lock; flock -n $fd || { echo "gpu$g locked by: $(cat $LOCKS/gpu$g.owner 2>/dev/null)"; exit 3; }
    echo "bimanual cosmos3 $MODEL step=${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-} host=$(hostname) pid=$$ since=$(date -Is) eta=${ETA:-?}" > $LOCKS/gpu$g.owner
  done
  export CUDA_VISIBLE_DEVICES=$GPU
fi
VPID=""
cleanup() { [ -n "$VPID" ] && kill $VPID 2>/dev/null; for g in ${GPU//,/ }; do rm -f $LOCKS/gpu$g.owner; done; }
trap cleanup EXIT
hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

SIF=/filer/tmp1/ss5772/containers/vllm-openai_cosmos3.sif   # apptainer pull docker://vllm/vllm-openai:cosmos3
RUNTIME=${RUNTIME:-$([ -f $SIF ] && echo sif || echo venv)}
if [ $RUNTIME = sif ]; then
  export PYTHONNOUSERSITE=1      # else the image's python 3.12 imports ~/.local/lib/python3.12 (a stray cv2 broke vLLM engine start)
  CT=$(command -v apptainer || command -v singularity) || { echo 'no apptainer/singularity on this node'; exit 8; }   # rlab1: singularity only
  RUN=($CT exec --nv --bind /filer/tmp1,/common/users,/freespace/local $SIF); VLLM=("${RUN[@]}" vllm); PY=("${RUN[@]}" python3); HFCLI=("${RUN[@]}" hf)
else
  VLLM=($ROOT/env/bin/vllm); PY=($ROOT/env/bin/python); HFCLI=($ROOT/env/bin/hf)
fi
[ $RUNTIME = sif ] && [ $MODEL = cosmos3-nano ] && EXTRA=(--mm-encoder-tp-mode data)   # image registers Cosmos3ForConditionalGeneration itself
[ -n "${VLLM_EXTRA:-}" ] && EXTRA+=($VLLM_EXTRA)       # e.g. "--disable-custom-all-reduce" (TP on PCIe without working P2P hung after graph capture)
echo "runtime=$RUNTIME model=$MID extra=${EXTRA[*]}"
if [ $RUNTIME = venv ] && [ ! -x $ROOT/env/bin/vllm ]; then           # pins from the Cosmos3 model card (CUDA 13 driver)
  [ -d $ROOT/cosmos-framework ] || git clone --depth 1 https://github.com/NVIDIA/cosmos-framework.git $ROOT/cosmos-framework || exit 4
  uv venv --python 3.13 --seed --managed-python $ROOT/env || exit 4
  VIRTUAL_ENV=$ROOT/env uv pip install --torch-backend=cu130 "vllm==0.21.0" -e $ROOT/cosmos-framework/packages/vllm-cosmos3 \
    "huggingface_hub[cli]" || exit 4
fi
"${HFCLI[@]}" download $MID --exclude "assets/*" --exclude "images/*" > $ROOT/logs/download_$MODEL.log 2>&1 || { tail -5 $ROOT/logs/download_$MODEL.log; exit 5; }
[ -n "${SETUP_ONLY:-}" ] && { echo "setup done: $MID"; du -sh $ROOT/env; exit 0; }

HF_HUB_OFFLINE=1 "${VLLM[@]}" serve $MID "${EXTRA[@]}" --host 127.0.0.1 --port $PORT --tensor-parallel-size $TP --max-model-len $MAXLEN \
  --gpu-memory-utilization $GPU_UTIL --allowed-local-media-path / > $ROOT/logs/vllm_$MODEL.log 2>&1 &
VPID=$!
for i in $(seq 360); do                         # up to 30 min (first start compiles)
  curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && break
  kill -0 $VPID 2>/dev/null || { echo "vllm died:"; tail -30 $ROOT/logs/vllm_$MODEL.log; exit 6; }
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/v1/models || { echo "vllm not up after 30 min"; tail -30 $ROOT/logs/vllm_$MODEL.log; exit 7; }
echo; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

cd /common/users/ss5772/bimanual
VLM_OPENAI_URL=http://127.0.0.1:$PORT/v1 PYTHONPATH=/common/users/ss5772/bimanual "${PY[@]}" -u -m world_model.v2.vlm.daemon \
  --model $MODEL --svc /common/users/ss5772/bimanual_data/vlm/svc_$MODEL
