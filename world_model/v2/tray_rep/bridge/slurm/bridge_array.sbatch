#!/usr/bin/env bash
#SBATCH --job-name=bridge
#SBATCH --partition=unlimited
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/common/users/ss5772/bimanual_data/tray_rep/bridge/logs/slurm_%x_%A_%a.out
# Bridge experiment: one array task = one run.py call = one line of bridge_tasks_<stage>.txt (CPU only, no GPU).
#   python -m world_model.v2.tray_rep.bridge.launch_local --stage tune --list-tasks     # writes bridge_tasks_tune.txt
#   sbatch --array=0-<N-1>%12 bridge_array.sbatch tune
#   (wait)  python -m world_model.v2.tray_rep.bridge.launch_local --stage select
#   python -m world_model.v2.tray_rep.bridge.launch_local --stage test --list-tasks     # writes bridge_tasks_test.txt
#   sbatch --array=0-<N-1>%12 bridge_array.sbatch test
# $1 = the stage name (tune | test). SIMULATION / SAVED DATA ONLY: no hardware, no network. Resumable: rerun = resume.
set -uo pipefail
STAGE=${1:?stage name: tune | test}
R=/common/users/ss5772/bimanual_data/tray_rep
export TMPDIR=/tmp/ss5772_bridge_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR $R/bridge/logs
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export PYTHONPATH=/common/users/ss5772/bimanual CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
ARGS=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" world_model/v2/tray_rep/bridge/slurm/bridge_tasks_${STAGE}.txt)
echo "[bridge] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} args=$ARGS $(date -u +%FT%TZ)"
$PY -m world_model.v2.tray_rep.bridge.run $ARGS --workers 2
rc=$?
echo "EXIT $rc"
rm -rf $TMPDIR
exit $rc
