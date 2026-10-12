#!/bin/bash
#SBATCH --job-name=gr_diag
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ilab3
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vlm/grounding/logs/diag_%j.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME=/freespace/local/ss5772/vlm/hf HUGGINGFACE_HUB_CACHE=/freespace/local/ss5772/vlm/hf/hub HF_HUB_OFFLINE=1
cd /common/users/ss5772/bimanual; export PYTHONPATH=/common/users/ss5772/bimanual
/freespace/local/ss5772/vlm/env/bin/python - <<'PY'
import traceback, torch
import world_model.v2.vlm.grounding.gr_runner as g
from world_model.v2.vlm import runner
proc, model, mid = runner.load("molmo2-er")
tc = model.config.text_config
print("rope_theta", getattr(tc, "rope_theta", None), "rope_parameters", getattr(tc, "rope_parameters", None), "rope_scaling", getattr(tc, "rope_scaling", None))
re_ = model.model.transformer.rotary_emb
print("rope_type", re_.rope_type, "inv_freq[:4]", re_.inv_freq[:4], "inv_freq[-2:]", re_.inv_freq[-2:], "scaling", re_.attention_scaling)
print("dtype", next(model.parameters()).dtype, "vit dtype", model.model.vision_backbone.image_vit.patch_embedding.weight.dtype if hasattr(model.model,'vision_backbone') else None)
PY
