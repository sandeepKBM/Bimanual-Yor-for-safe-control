#!/usr/bin/env bash
#SBATCH --job-name=vd_nom
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/vdirect_ilab/logs/slurm_%x_%j.out
# WRITTEN BY THE vdirect KILL TEST, NOT SUBMITTED. Owner submits (scripts/offload/submit_via_slurm_gateway.sh) after
# checking live free GPUs + node AllocMem (docs/runbooks/slurm.md). SIMULATION / SAVED DATA ONLY: no hardware, no
# network. Stage 1 of 2: shared setup (normalization, closed-loop state pick, phys/hold references) + the 9 nominal
# pre-trains (3 seeds x 3 ensemble members; member 0 also trains the round-0 probe value). ~1 h on one GPU.
# Stage 2 = vd_grid_array.sbatch with --dependency=afterok:<this job id>.
set -uo pipefail
export VD_ROOT=/common/users/ss5772/bimanual_data/vdirect_ilab      # small outputs only (~1 GB total)
export TMPDIR=/tmp/ss5772_vd_${SLURM_JOB_ID:-local}; mkdir -p $TMPDIR $VD_ROOT/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache TORCH_HOME=$TMPDIR/cache/torch CUDA_CACHE_PATH=$TMPDIR/nv
export PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
echo "[vd_nom] host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -u +%FT%TZ)"
fail=0
[ -f $VD_ROOT/shared/_DONE_SETUP ] || { $PY -m world_model.v2.vdirect.run_arm --arm SETUP; echo "EXIT SETUP $?"; }
for s in 0 1 2; do for m in 0 1 2; do                         # finite loop; each cell writes its own _DONE
  [ -f $VD_ROOT/runs/nom_s${s}_m${m}/_DONE ] && continue
  $PY -m world_model.v2.vdirect.run_arm --arm NOM --seed $s --member $m; rc=$?
  echo "EXIT NOM_s${s}_m${m} $rc"; [ $rc -ne 0 ] && fail=1
done; done
rm -rf $TMPDIR
[ $fail -eq 0 ] && date -u +%FT%TZ > $VD_ROOT/_DONE_STAGE1
exit $fail
