#!/bin/bash
# HF-transformers vlm daemon (vlm/daemon.py) for Qwen3.5-9B base or + LoRA adapter, inside a GPU step that already set CUDA_VISIBLE_DEVICES
# (e.g. `cdx-agent --gpu-lock 526576 --gpu 2 --eta 40 --mem 48G -- bash qwen_daemon_step.sh`).
#   MODEL=qwen3.5-9b | qwen3.5-9b-lora   QWEN_LORA_PATH=<adapter dir>   SVC=<bimanual_data/vlm/svc_*>
# Stop: BIMANUAL_VLM_SVC=$SVC python -m world_model.v2.vlm.service stop
set -u
HF=/filer/tmp1/ss5772/huggingface
export HF_HOME=$HF HF_HUB_CACHE=$HF/hub HUGGINGFACE_HUB_CACHE=$HF/hub HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/common/users/ss5772/bimanual
mkdir -p /freespace/local/ss5772/z && export TMPDIR=/freespace/local/ss5772/z
hostname; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} MODEL=$MODEL LORA=${QWEN_LORA_PATH:-} SVC=$SVC"
cd /common/users/ss5772/bimanual
exec /filer/tmp1/ss5772/envs/qwenft/bin/python -u -m world_model.v2.vlm.daemon --model "$MODEL" --svc "$SVC"
