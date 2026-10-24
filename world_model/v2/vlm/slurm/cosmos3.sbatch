#!/bin/bash
#SBATCH --job-name=cosmos3
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ilab3
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vlm/logs/cosmos3_%j.out
# Cosmos 3 reasoner + vlm daemon (see cosmos3_serve.sh). usage: sbatch --export=ALL,MODEL=cosmos3-edge cosmos3.sbatch
# clients: BIMANUAL_VLM_SVC=/common/users/ss5772/bimanual_data/vlm/svc_$MODEL ; stop: same env + python -m world_model.v2.vlm.service stop
bash /common/users/ss5772/bimanual/world_model/v2/vlm/slurm/cosmos3_serve.sh
