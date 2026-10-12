#!/bin/bash
# sweep 3: follow-ups on the best regime (low mu, eps_scale 0.25), ALL 600 VAL carries. SIMULATION ONLY. Starts after sweep 2.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/common/users/ss5772/bimanual
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
B=/common/users/ss5772/bimanual_data/tray_rep/wrongpoint
OUT=$B/runs3
until [ -f $B/runs2/_SWEEP2_DONE ]; do sleep 10; done
run() { name=$1; shift
  [ -f $OUT/$name/_DONE ] && return
  $PY -m world_model.v2.tray_rep.wrongpoint.run_cell --cell $name --n-eps 600 --workers 12 --out $OUT "$@" && touch $OUT/$name/_DONE; }
LOW="--mu-range 5e-4,1.5e-3"
run fullA_e0.25       --I 0.99 --eps-scale 0.25
run lowH_I1.5_e0.25   --I 1.5 --eps-scale 0.25 $LOW
run vlow_e0.25_mu2-6e-4 --I 0.99 --eps-scale 0.25 --mu-range 2e-4,6e-4
run lowI_e0.25_a0.25  --I 0.99 --eps-scale 0.25 --auth 0.25 $LOW
for PS in 1 2; do
  mkdir -p $OUT/lowA_ps$PS; cp $B/runs2/lowA_e0.25/oracle.jsonl $B/runs2/lowA_e0.25/F0.jsonl $OUT/lowA_ps$PS/
  run lowA_ps$PS --I 0.99 --eps-scale 0.25 --pseed $PS --arms point@0.2,conf@0.2,point@0.4,conf@0.4,oracle,F0 $LOW
done
touch $OUT/_SWEEP3_DONE
