#!/usr/bin/env bash
#SBATCH --job-name=wmhj_full
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/wm_audit/runs/slurm_%x_%j.out
# WRITTEN BY THE WM AUDIT, NOT SUBMITTED. Owner runs it (submit via scripts/offload/submit_via_slurm_gateway.sh
# after checking live free GPUs + node AllocMem per docs/runbooks/slurm.md). SIMULATION / SAVED DATA ONLY.
# Full-length version of the audit's passing recipe: kin_wm + privileged slosh supervision + counterfactual
# (evasive-action) branches, 3 seeds, then the Fisac/LatentSafe value (max over the 85-action fallback set)
# and the gate vs the physics-latent value. Expected ~1.5 h/seed on an A4000/A5000-class GPU.
set -euo pipefail
SEEDS="${SEEDS:-0 1 2}"
STEPS=${STEPS:-60000}
ITERS=${ITERS:-50000}
export AUD=/common/users/ss5772/bimanual_data/wm_audit
export LATENT_HJ_SCR=/common/users/ss5772/latent_hj_scratch
export TMPDIR=$AUD/tmp XDG_CACHE_HOME=$AUD/tmp/cache TORCH_HOME=$AUD/tmp/cache/torch CUDA_CACHE_PATH=$AUD/tmp/nv
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
echo "[wmhj] host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
for s in $SEEDS; do                                   # finite loop, _DONE sentinel per seed
  OUT=$AUD/runs/full_b100_s$s
  [ -f "$OUT/_DONE" ] && continue
  $PY -m world_model.v2.wm_audit.train_kinwm --out $OUT --steps $STEPS --aux-w 1.0 --branch-frac 0.5 \
      --n-branch 60000 --seed $s --log-every 2000
  $PY -m world_model.v2.wm_audit.value_gate --wm $OUT/wm.pt --out $OUT/value --iters $ITERS
  date -u +%FT%TZ > "$OUT/_DONE"
done
