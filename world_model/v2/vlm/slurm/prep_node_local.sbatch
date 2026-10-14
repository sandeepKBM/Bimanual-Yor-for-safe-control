#!/bin/bash
#SBATCH --job-name=vlm_prep
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ilab3
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vlm/logs/prep_%j.out
# One-off prep ON ilab3 NODE-LOCAL disk (/freespace/local/ss5772/vlm): conda env + HF weights. Only code/logs/jsonl live on NFS.
# NB: the login profile points HUGGINGFACE_HUB_CACHE at NFS /filer/tmp1 -> override all three explicitly.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
L=/freespace/local/ss5772/vlm
mkdir -p $L/hf/hub $L/tmp $L/pip
export HF_HOME=$L/hf HUGGINGFACE_HUB_CACHE=$L/hf/hub TRANSFORMERS_CACHE=$L/hf/hub TMPDIR=$L/tmp PIP_CACHE_DIR=$L/pip
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h $L | tail -1
ENV=$L/env
if [ ! -x $ENV/bin/python ]; then
  /common/users/ss5772/miniforge3/bin/conda create -y -p $ENV python=3.11 || exit 1
  PY=$ENV/bin/python
  $PY -m pip install -U pip
  $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  $PY -m pip install -U transformers accelerate huggingface_hub pillow numpy qwen-vl-utils
fi
PY=$ENV/bin/python
$PY -c "import torch,transformers;print(torch.__version__,torch.cuda.is_available(),transformers.__version__)"
# move the two models that the first (mis-configured) run cached on NFS /filer/tmp1 -> node-local
$PY - <<'PYEOF'
import os, shutil
src, dst = "/filer/tmp1/ss5772/huggingface/hub", "/freespace/local/ss5772/vlm/hf/hub"
def size(p): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs if not os.path.islink(os.path.join(r, f)))
for d in ("models--Qwen--Qwen3.5-9B", "models--Qwen--Qwen3-VL-8B-Instruct"):
    s, t = f"{src}/{d}", f"{dst}/{d}"
    if os.path.isdir(s) and not os.path.isdir(t):
        shutil.copytree(s, t, symlinks=True)
    if os.path.isdir(s) and os.path.isdir(t):
        a, b = size(s), size(t); print(d, a, b, flush=True)
        if a == b and a > 1e9:
            shutil.rmtree(s); print("removed NFS copy", d, flush=True)
PYEOF
# bake-off candidates, ANONYMOUS download (no HF token). Gated repos will 401 -> reported, skipped.
for M in Qwen/Qwen3.5-9B Qwen/Qwen3-VL-8B-Instruct nvidia/Cosmos-Reason1-7B allenai/Molmo2-ER BAAI/RoboBrain2.5-4B BAAI/RoboBrain2.0-7B nvidia/Cosmos-Reason2-8B; do
  echo "== $M"
  $PY -c "from huggingface_hub import snapshot_download as s;print('OK',s('$M'))" 2>&1 | tail -2 | cut -c1-300
done
du -sh $L/hf $L/env; df -h $L | tail -1
