# world_model/v2/model/slurm/common_env.sh -- sourced by every v2 WM sbatch.
# Everything lives on /common/users (NFS, visible from rlab1: the 2026-09-06
# dr_retrain job on rlab1 wrote world_model/logdir/wm_cup_dr/ here; ilab3/rlab6
# visibility is ASSUMED, verify with the one-line probe in README_TRAINING below).
export REPO=/common/users/ss5772/bimanual
export PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
export DATA_ROOT=${DATA_ROOT:-/common/users/ss5772/bimanual_data/wm_v2}      # staged shards + prep + caches
export RUN_ROOT=${RUN_ROOT:-/common/users/ss5772/bimanual_data/wm_v2/runs}
export PREP=${PREP:-$DATA_ROOT/prep/spill_v2}                                # canonical prepared dir (index.json)
export FEATS=${FEATS:-$DATA_ROOT/prep/spill_v2_dino}                         # frozen DINOv2 token cache
export TORCH_HOME=${TORCH_HOME:-/common/home/ss5772/.cache/torch}             # dinov2_vits14 already downloaded here
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export MUJOCO_GL=disable PYTHONUNBUFFERED=1
cd "$REPO"
echo "[v2-wm] host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
