#!/usr/bin/env bash
#SBATCH --job-name=bimanual_wm_dr_retrain
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4500_ada:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# world_model/dr_retrain.sbatch
#
# DR-retrain pipeline for the cup world model, addressing the OOD spill-
# monitor brittleness found in world_model/RESULTS_WM_DR.md (in-dist spill
# ROC-AUC 0.999 -> 0.746 under domain randomization, no retraining). This
# script retrains the WM + monitor head ON domain-randomized data and
# re-evaluates OOD generalization, all on the allocated node's GPU:
#   S1) collect 200 DR cup episodes (randomization/collect_cup_dr.py,
#       safe_env, CPU+EGL render)
#   S2) train WM on that DR data (world_model/train_wm.py, mujoco_ur5e, GPU)
#   S3) train monitor head on the DR-trained WM (world_model/monitor_head.py,
#       mujoco_ur5e, GPU)
#   S4) OOD eval: DR-trained head on the held-out DR test set that scored
#       0.746 before, PLUS a regression check against the original clean
#       (cup_yor_full) eval set (world_model/eval_monitor_ood.py, mujoco_ur5e)
#
# All 4 stages are finite (fixed episode/step/epoch counts) so this script
# exits on its own and Slurm releases the node immediately -- see
# docs/runbooks/slurm.md. Submit via a Slurm login node directly with
# sbatch, or from a workstation without local sbatch via
#   bash scripts/offload/submit_via_slurm_gateway.sh world_model/dr_retrain.sbatch

set -euo pipefail

REPO_ROOT="${BIMANUAL_ROOT:-/common/users/ss5772/bimanual}"
SAFE_ENV_PY="${SAFE_ENV_PY:-/common/users/ss5772/miniforge3/bin/conda run -n safe_env python}"
MUJOCO_ENV_PY="${MUJOCO_ENV_PY:-/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python}"

JOBID="${SLURM_JOB_ID:-local}"
LOG_DIR="${REPO_ROOT}/world_model/logs"
LOG="${LOG_DIR}/dr_retrain_${JOBID}.log"
mkdir -p "${LOG_DIR}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${REPO_ROOT}"

{
echo "[dr_retrain] job=${JOBID} host=$(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -u +%FT%TZ)"

echo "[dr_retrain] S1: collect 200 DR cup episodes -> collection/data/cup_dr_train"
${SAFE_ENV_PY} -m randomization.collect_cup_dr \
  --num-episodes 200 --seed-start 6000 \
  --out-dir collection/data/cup_dr_train

echo "[dr_retrain] S2: train WM on DR data -> world_model/logdir/wm_cup_dr"
"${MUJOCO_ENV_PY}" -m world_model.train_wm \
  --collection-dir collection/data/cup_dr_train \
  --dreamer-fmt-dir world_model/data/dreamer_fmt/cup_dr_train \
  --logdir world_model/logdir/wm_cup_dr \
  --train-steps 2000 --batch-size 16 --batch-length 64 \
  --eval-every 50 --ckpt-every 100 --eval-holdout 20

echo "[dr_retrain] S3: train monitor head on DR-trained WM"
"${MUJOCO_ENV_PY}" -m world_model.monitor_head \
  --dreamer-fmt-dir world_model/data/dreamer_fmt/cup_dr_train \
  --collection-dir collection/data/cup_dr_train \
  --checkpoint world_model/logdir/wm_cup_dr/best.pt \
  --epochs 300 --eval-holdout 20 --tag cup_dr --skip-convert \
  --labels spill,collision --spill-signal per_step \
  --head-ckpt-out world_model/logdir/wm_cup_dr/monitor_head_cup_dr.pt

echo "[dr_retrain] S4a: OOD eval on held-out DR test set (dr_on_drtest)"
"${MUJOCO_ENV_PY}" -m world_model.eval_monitor_ood \
  --collection-dir collection/data/cup_dr_test \
  --dreamer-fmt-dir world_model/data/dreamer_fmt/cup_dr_test_v2 \
  --wm-checkpoint world_model/logdir/wm_cup_dr/best.pt \
  --head-checkpoint world_model/logdir/wm_cup_dr/monitor_head_cup_dr.pt \
  --tag dr_on_drtest --spill-signal per_step

echo "[dr_retrain] S4b: regression check on clean set (dr_on_clean)"
"${MUJOCO_ENV_PY}" -m world_model.eval_monitor_ood \
  --collection-dir collection/data/cup_yor_full \
  --dreamer-fmt-dir world_model/data/dreamer_fmt/cup_yor_full \
  --wm-checkpoint world_model/logdir/wm_cup_dr/best.pt \
  --head-checkpoint world_model/logdir/wm_cup_dr/monitor_head_cup_dr.pt \
  --skip-convert --tag dr_on_clean --spill-signal per_step

echo "[dr_retrain] all stages done end=$(date -u +%FT%TZ)"
} > "${LOG}" 2>&1

# Belt-and-suspenders sentinel (docs/runbooks/slurm.md): all 4 stages above
# are finite, so this script reaches here and Slurm frees the node on its
# own -- this sentinel is only for scripts/offload/slurm_autorelease.sh in
# case a future change makes a stage hang.
mkdir -p "${REPO_ROOT}/world_model/logdir/wm_cup_dr"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${REPO_ROOT}/world_model/logdir/wm_cup_dr/_DONE"
echo "[dr_retrain.sbatch] done, wrote world_model/logdir/wm_cup_dr/_DONE -- log at ${LOG}"
