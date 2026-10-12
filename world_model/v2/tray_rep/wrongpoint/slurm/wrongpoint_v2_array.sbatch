#!/usr/bin/env bash
#SBATCH --job-name=wpv2
#SBATCH --partition=unlimited
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --exclude=rlab1,ilab1
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/wrongpoint_v2/logs/slurm_%x_%A_%a.out
# wrongpoint_v2: TEST re-run of the key wrongpoint cells (lowI lowJ lowE lowK lowA fullA) with the CLOSED-LOOP planner (boed/run_cl via wrongpoint/run_cl_wp.py),
# fixed-physics data (data_piper_trk_full_v2) and v2 calibration (calib_v2 mu_eff map, eps_a=0, online lam 0.1 sa 0.04). One run per line of wrongpoint_v2_tasks.txt.
# Submit:  sbatch --array=0-335 world_model/v2/tray_rep/wrongpoint/slurm/wrongpoint_v2_array.sbatch     (N = lines - 1; 336 lines)
# Slurm gives 2 CPUs per job (no --cpus-per-task). Temp on the shared data disk, emptied with find -delete.
# SIMULATION / SAVED DATA ONLY: no hardware, no network, no GPU. Resumable: rerun = resume.
set -uo pipefail
export TMPDIR=/common/users/ss5772/bimanual_data/slurm_tmp/wpv2_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $TMPDIR /common/users/ss5772/bimanual_data/tray_rep/wrongpoint_v2/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
export BIMANUAL_MU_EFF_FILE=/common/users/ss5772/bimanual_data/tray_rep/calib_v2/mu_eff_calib.json BIMANUAL_F5_EPS_A=0.0
unset BIMANUAL_LEGACY_PIN
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
TASKS=${TASKS_FILE:-world_model/v2/tray_rep/wrongpoint/slurm/wrongpoint_v2_tasks.txt}
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" $TASKS)
echo "[wpv2] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} args=$ARGS $(date -u +%FT%TZ)"
$PY -m world_model.v2.tray_rep.wrongpoint.run_cl_wp $ARGS --workers 2
rc=$?
find "$TMPDIR" -delete 2>/dev/null
echo "EXIT $rc"
exit $rc
