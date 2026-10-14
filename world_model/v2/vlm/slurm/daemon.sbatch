#!/bin/bash
#SBATCH --job-name=vlm_daemon
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ilab3
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vlm/logs/daemon_%j.out
# Long-lived VLM service (vlm/daemon.py). usage: sbatch [--export=ALL,MODEL=qwen3.5-9b] daemon.sbatch ; stop: python -m world_model.v2.vlm.service stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME=/freespace/local/ss5772/vlm/hf HUGGINGFACE_HUB_CACHE=/freespace/local/ss5772/vlm/hf/hub TRANSFORMERS_CACHE=/freespace/local/ss5772/vlm/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/freespace/local/ss5772/vlm/tmp; mkdir -p $TMPDIR
cd /common/users/ss5772/bimanual; export PYTHONPATH=/common/users/ss5772/bimanual
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
/freespace/local/ss5772/vlm/env/bin/python -u -m world_model.v2.vlm.daemon --model "${MODEL:-qwen3.5-9b}"
