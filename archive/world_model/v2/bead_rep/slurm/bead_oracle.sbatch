#!/usr/bin/env bash
#SBATCH --job-name=bead_orc
#SBATCH --partition=unlimited
#SBATCH --exclude=rlab1,ilab1
#SBATCH --mem=3G
#SBATCH --time=08:00:00
#SBATCH --array=0-49
#SBATCH --output=/common/users/ss5772/bimanual_data/bead_replication/logs/slurm_%x_%A_%a.out
# bead_rep Phase 2 oracle (search-certified savable set, DESIGN Q5).  WRITTEN, NOT SUBMITTED.  CPU ONLY.
# SIMULATION / SAVED DATA ONLY: no hardware, no network.
# Needs the generated + merged dataset (gen_nominal) and a states CSV (oracle.py candidates ...).  Required env:
#   STATES=/path/states.csv   TAG=val_v0   (NTASK defaults to the array size, 50; NPROC=2 worker processes per task)
#   sbatch --export=ALL,STATES=$STATES,TAG=$TAG world_model/v2/bead_rep/slurm/bead_oracle.sbatch
# Sub-shards = NTASK*NPROC; state s goes to sub-shard (s mod NTASK*NPROC).  Per-STATE resumable: results are appended to
# oracle/<TAG>/part_<k>.jsonl, a re-run skips the rows already there; finished parts carry part_<k>._DONE.
# After all tasks end:  python -m world_model.v2.bead_rep.oracle merge --tag $TAG
set -uo pipefail
: "${STATES:?set STATES=/path/to/states.csv}" "${TAG:?set TAG=<name>}"
NTASK=${NTASK:-50}; NPROC=${NPROC:-2}
export BEAD_ROOT=/common/users/ss5772/bimanual_data/bead_replication
export TMPDIR=/tmp/ss5772_bead_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR" "$BEAD_ROOT/logs" "$BEAD_ROOT/oracle/$TAG"
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
i=${SLURM_ARRAY_TASK_ID:-0}
echo "[bead_orc] host=$(hostname) job=${SLURM_JOB_ID:-local} task=$i/$NTASK nproc=$NPROC tag=$TAG states=$STATES $(date -u +%FT%TZ)"
pids=()
for p in $(seq 0 $((NPROC - 1))); do
  sub=$(( i * NPROC + p ))
  $PY -m world_model.v2.bead_rep.oracle run --states "$STATES" --tag "$TAG" --shard ${sub}/$(( NTASK * NPROC )) \
      > $BEAD_ROOT/logs/oracle_${TAG}_sub${sub}.log 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
echo "EXIT bead_orc task $i rc=$rc $(date -u +%FT%TZ)"
rm -rf "$TMPDIR"
exit $rc
