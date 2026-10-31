#!/bin/bash
#SBATCH --job-name=bim_hold
#SBATCH --gres=gpu:4500_ada:2
#SBATCH --nodelist=rlab1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vlm/logs/hold_%j.out
# 2-GPU hold for bimanual VLM services (user-requested 10-16). Work runs as --overlap steps, one GPU each via flock:
#   ssh rlab1 "srun --jobid=<id> --overlap --gres=gpu:4500_ada:2 --mem=40G -t <limit> bash -lc 'MODEL=cosmos3-nano GPU=0 bash <repo>/world_model/v2/vlm/slurm/cosmos3_serve.sh'"
# release: scancel <id>
mkdir -p /freespace/local/ss5772/gpu_locks
hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
exec sleep infinity
