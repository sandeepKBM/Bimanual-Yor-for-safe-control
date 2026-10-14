#!/usr/bin/env bash
#SBATCH --job-name=bimanual_il_policy
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# experiments/imitation/train_policy.sbatch -- SINGLE-GPU policy training
# template (BC-MLP today; ACT / Diffusion Policy once their trainers exist).
# NOT submitted by the author. Owner submits from westeros via:
#   scripts/offload/submit_via_slurm_gateway.sh experiments/imitation/train_policy.sbatch
# (pin a free 24 GB node with e.g.  #SBATCH --nodelist=ilab3  after checking
#  free GPUs via slurm_gateway_exec scontrol show node; rules:
#  docs/runbooks/slurm.md + slurm_rules_vendor_skill.md -- no --cpus-per-task/
#  --ntasks, generous --time, finite step count, _DONE sentinel, 1 GPU here.)
#
# Training only -- no MuJoCo, no robot, no network. Data = gen_demos.py npz
# shards on shared /common storage (CPU-prepped on westeros); the whole shard
# set is loaded into host RAM / GPU memory once (state-only: ~50 MB per 50
# 3-cube demos; with 3x96x96 images @20 Hz: ~10 MB per demo).
#
# Env overrides:
#   POLICY      bc | act | dp              (default bc)
#   SHARDS      shard dir                  (default below -- CHANGE to a /common path, not a job tmp dir)
#   OUT         run dir                    (default $REPO_ROOT/experiments/imitation/runs/$POLICY_$SLURM_JOB_ID)
#   STEPS       finite optimizer steps     (default 200000)
#   TRAIN_CMD   full command override (e.g. an ACT/DP trainer); must terminate.

set -euo pipefail

REPO_ROOT="${BIMANUAL_ROOT:-/common/users/ss5772/bimanual}"
PY="${PYTHON_BIN:-/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python}"
POLICY="${POLICY:-bc}"
SHARDS="${SHARDS:-/common/users/ss5772/bimanual/experiments/imitation/data/demos_n3/shards}"
OUT="${OUT:-${REPO_ROOT}/experiments/imitation/runs/${POLICY}_${SLURM_JOB_ID:-local}}"
STEPS="${STEPS:-200000}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${REPO_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "${OUT}/gpu.txt"

case "${POLICY}" in
  bc)
    CMD=("${PY}" -u experiments/imitation/train_bc.py --shards "${SHARDS}" --out "${OUT}"
         --steps "${STEPS}" --bs 4096 --hidden 1024 --depth 4 --include-failures)
    ;;
  act|dp)
    # No ACT / Diffusion-Policy trainer exists in this repo yet and the
    # mujoco_ur5e env has no lerobot/diffusers/torchvision. Provide one via
    # TRAIN_CMD (e.g. an in-repo pure-torch ACT/DP reading the same shards:
    # obs = proprio slice + img_<cam> keys, action = the (K+1,7) chunk from
    # bc_model.build_targets with K raised to 20-50).
    : "${TRAIN_CMD:?POLICY=${POLICY} needs TRAIN_CMD (no ACT/DP trainer in-repo yet)}"
    ;;
  *) echo "unknown POLICY=${POLICY}" >&2; exit 2 ;;
esac

if [[ -n "${TRAIN_CMD:-}" ]]; then
  bash -c "${TRAIN_CMD}" 2>&1 | tee "${OUT}/train.log"
else
  "${CMD[@]}" 2>&1 | tee "${OUT}/train.log"
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/_DONE"
echo "[train_policy.sbatch] done -- ${OUT}"
