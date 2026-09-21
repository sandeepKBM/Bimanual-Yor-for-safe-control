#!/usr/bin/env bash
#SBATCH --job-name=bimanual_wm_train
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#
# world_model/wm_train.sbatch
#
# Scaffold, modeled on collection/collect.sbatch -- single-pass Slurm job
# that runs an OFFLINE DreamerV3 world-model-only training loop on our
# bimanual collection/ episodes (converted via world_model/dataset_adapter.py
# into dreamerv3-torch's flat-npz replay format). Submit via sbatch directly
# on a Slurm login node, or from a workstation without local sbatch via
# world_model/submit_wm_rlab.sh. NOT executed by this task (scaffold only --
# not submitted; see world_model/README.md's smoke section for what WAS run).
#
# This does NOT call dreamerv3-torch's dreamer.py (that always constructs a
# gym env for obs/action-space discovery + does online rollouts). It calls
# world_model/train_wm.py instead: convert episodes once (idempotent), then
# loop models.WorldModel._train(batch) for config.train_steps steps with
# checkpointing + tensorboard logging, no env, no actor/critic, world model
# only. train_wm.py infers obs/action shapes from the converted data itself
# (never hardcoded), so this script does not need updating if the WM image
# view or proprio width changes upstream in dataset_adapter.py.

set -euo pipefail

REPO_ROOT="${BIMANUAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONDA_ENV="${CONDA_ENV:-mujoco_ur5e}"
COLLECTION_DIR="${COLLECTION_DIR:-${REPO_ROOT}/collection/data/run_001}"
DREAMER_FMT_DIR="${DREAMER_FMT_DIR:-${REPO_ROOT}/world_model/data/dreamer_fmt/train_001}"
CONFIG="${CONFIG:-${REPO_ROOT}/world_model/train_wm_config.yaml}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/world_model/logdir/wm_run_001}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
EVAL_EVERY="${EVAL_EVERY:-50}"
CKPT_EVERY="${CKPT_EVERY:-50}"
EVAL_HOLDOUT="${EVAL_HOLDOUT:-5}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"

# Real offline WM training: converts COLLECTION_DIR -> DREAMER_FMT_DIR (skips
# reconversion if already populated is NOT assumed -- convert_dir always
# (re)writes; point DREAMER_FMT_DIR at a fresh dir per run, or pass
# --skip-convert yourself if re-running against an already-converted dir),
# then trains for TRAIN_STEPS steps with resume-from-latest.pt, periodic
# eval, and best.pt/latest.pt checkpointing.
python -m world_model.train_wm \
  --config "${CONFIG}" \
  --collection-dir "${COLLECTION_DIR}" \
  --dreamer-fmt-dir "${DREAMER_FMT_DIR}" \
  --logdir "${LOGDIR}" \
  --train-steps "${TRAIN_STEPS}" \
  --eval-every "${EVAL_EVERY}" \
  --ckpt-every "${CKPT_EVERY}" \
  --eval-holdout "${EVAL_HOLDOUT}"

echo "[wm_train.sbatch] done -- logdir=${LOGDIR}"
