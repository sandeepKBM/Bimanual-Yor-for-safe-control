#!/bin/bash
# sweep 2: low-mu focus, ALL 600 VAL carries/cell. SIMULATION ONLY.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/common/users/ss5772/bimanual
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
OUT=/common/users/ss5772/bimanual_data/tray_rep/wrongpoint/runs2
run() { name=$1; shift
  [ -f $OUT/$name/_DONE ] && return
  $PY -m world_model.v2.tray_rep.wrongpoint.run_cell --cell $name --n-eps 600 --workers 12 --out $OUT "$@" && touch $OUT/$name/_DONE; }
LOW="--mu-range 5e-4,1.5e-3"
run lowA_e0.25      --I 0.99 --eps-scale 0.25 $LOW
run lowB_e0.1       --I 0.99 --eps-scale 0.1 $LOW
run lowC_e0.02      --I 0.99 --eps-scale 0.02 $LOW
run lowD_e0.25_mu1e-3 --I 0.99 --eps-scale 0.25 --mu-range 5e-4,1e-3
run lowE_e0.25_wr2-3  --I 0.99 --eps-scale 0.25 --wrong 2,3 $LOW
run lowF_e0.25_a0.5   --I 0.99 --eps-scale 0.25 --auth 0.5 $LOW
run lowG_e1_a1        --I 0.99 --eps-scale 1 $LOW
touch $OUT/_SWEEP2_DONE
