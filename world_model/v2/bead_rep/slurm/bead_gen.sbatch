#!/usr/bin/env bash
#SBATCH --job-name=bead_gen
#SBATCH --partition=unlimited
#SBATCH --exclude=rlab1,ilab1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --array=0-29
#SBATCH --output=/common/users/ss5772/bimanual_data/bead_replication/logs/slurm_%x_%A_%a.out
# bead_rep Phase 1 generator: replay the 180 spill_v2 source shards through the 8-bead rig.  WRITTEN, NOT SUBMITTED.
# CPU ONLY (no --gres, no GPU).  SIMULATION / SAVED DATA ONLY: no hardware, no network.
# Layout: NTASK array tasks x NPROC worker processes each (a job gets 2 CPUs) = NTASK*NPROC sub-shards of the 180 source
# shards (every n-th).  Defaults: 30 x 2 = 60 sub-shards -> 3 source shards each, ~3-5 min per sub-shard (smoke: ~85 s / source
# shard on a loaded westeros core, 23 eps).  Resumable: a finished source shard has data/<split>/shard_<g>._DONE and is skipped.
#   mkdir -p /common/users/ss5772/bimanual_data/bead_replication/logs          (the --output dir must exist at submit time)
#   sbatch world_model/v2/bead_rep/slurm/bead_gen.sbatch                       (via the rlab1 gateway from westeros)
#   then, when all are done:  python -m world_model.v2.bead_rep.gen_nominal --merge   and   --verify
# If you change --array, set NTASK to its size (NTASK=$((max+1))).
set -uo pipefail
NTASK=${NTASK:-30}; NPROC=${NPROC:-2}
export BEAD_ROOT=/common/users/ss5772/bimanual_data/bead_replication
export TMPDIR=/tmp/ss5772_bead_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR" "$BEAD_ROOT/logs" "$BEAD_ROOT/data/train" "$BEAD_ROOT/data/val" "$BEAD_ROOT/data/test"
export MPLCONFIGDIR=$TMPDIR/mpl XDG_CACHE_HOME=$TMPDIR/cache PYTHONPYCACHEPREFIX=$TMPDIR/pyc NUMBA_CACHE_DIR=$TMPDIR/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
i=${SLURM_ARRAY_TASK_ID:-0}
echo "[bead_gen] host=$(hostname) job=${SLURM_JOB_ID:-local} task=$i/$NTASK nproc=$NPROC $(date -u +%FT%TZ)"
pids=()
for p in $(seq 0 $((NPROC - 1))); do
  sub=$(( i * NPROC + p ))
  $PY -m world_model.v2.bead_rep.gen_nominal --shard ${sub}/$(( NTASK * NPROC )) --out $BEAD_ROOT/data \
      > $BEAD_ROOT/logs/gen_sub${sub}.log 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
echo "EXIT bead_gen task $i rc=$rc $(date -u +%FT%TZ)"
rm -rf "$TMPDIR"
exit $rc
