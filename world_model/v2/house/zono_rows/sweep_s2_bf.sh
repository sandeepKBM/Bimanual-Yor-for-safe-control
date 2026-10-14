#!/usr/bin/env bash
# 24-run S2 sweep: {m8,m64} x {relax, relax+level, relax+level+track} x {base filter off,on} x {uniform,selective}.  CPU only, ~1-2 min per run.
#   cd /common/users/ss5772/bimanual && setsid nohup bash world_model/v2/house/zono_rows/sweep_s2_bf.sh > /data/local/ss5772/molmospaces/zono_rows/sweep_bf.log 2>&1 &
# results: /data/local/ss5772/molmospaces/zono_rows/s2_results_sw_<tag>.json (tags never collide with the older s2_results_* files)
set -u
cd /common/users/ss5772/bimanual
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 BIMANUAL_FLMPC_IMPL=rt
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
OUT=/data/local/ss5772/molmospaces/zono_rows; mkdir -p $OUT/logs
JOBS=${JOBS:-12}
run() {  # run <mr> <lvl:0|1> <trk:0|1> <bf:0|1> <relax:uniform|selective>
  mr=$1; lvl=$2; trk=$3; bf=$4; rx=$5
  tag="sw_m${mr}_l${lvl}t${trk}_bf${bf}_${rx:0:3}"
  [ -f $OUT/s2_results_${tag}.json ] && return 0
  args="--margin 0.02 --alpha 10 --d-act 0.15 --max-rows $mr --refine 0 --relax $rx --only-rows --bottle-every 2 --tag _${tag}"
  [ "$lvl" = 1 ] && args="$args --level"
  [ "$trk" = 1 ] && args="$args --track"
  [ "$bf" = 1 ] && args="$args --base-filter"
  $PY -m world_model.v2.house.zono_rows.smoke_s2 run $args > $OUT/logs/${tag}.log 2>&1
  echo "done $tag rc=$?"
}
export -f run; export PY OUT
for mr in 8 64; do for lt in "0 0" "1 0" "1 1"; do for bf in 0 1; do for rx in uniform selective; do echo "$mr $lt $bf $rx"; done; done; done; done \
  | xargs -P $JOBS -L 1 bash -c 'run $0 $1 $2 $3 $4'
echo ALL_DONE
