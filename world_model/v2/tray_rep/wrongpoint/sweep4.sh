#!/bin/bash
# sweep 4: stress combos (larger confident error x reduced authority x low mu), ALL 600 VAL carries. SIMULATION ONLY. Starts after sweep 3.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/common/users/ss5772/bimanual
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
B=/common/users/ss5772/bimanual_data/tray_rep/wrongpoint
OUT=$B/runs4
until [ -f $B/runs3/_SWEEP3_DONE ]; do sleep 10; done
run() { name=$1; shift
  [ -f $OUT/$name/_DONE ] && return
  $PY -m world_model.v2.tray_rep.wrongpoint.run_cell --cell $name --n-eps 600 --workers 12 --out $OUT "$@" && touch $OUT/$name/_DONE; }
LOW="--mu-range 5e-4,1.5e-3"
run lowJ_e0.25_wr3-4        --I 0.99 --eps-scale 0.25 --wrong 3,4 $LOW
run lowK_e0.25_wr2-3_a0.5   --I 0.99 --eps-scale 0.25 --wrong 2,3 --auth 0.5 $LOW
run fullE_e0.25_wr2-3       --I 0.99 --eps-scale 0.25 --wrong 2,3
touch $OUT/_SWEEP4_DONE
