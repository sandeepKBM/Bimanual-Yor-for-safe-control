#!/bin/bash
# screening sweep: VAL, 100 carries/cell, piper_trk, seed 0. SIMULATION ONLY.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/common/users/ss5772/bimanual
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
OUT=/common/users/ss5772/bimanual_data/tray_rep/wrongpoint/runs
for AUTH in 1 0.5; do for MU in full low; do for E in 1 0.5 0.25; do for I in 0.99 0.75 1.0; do
  name=I${I}_e${E}_mu${MU}_a${AUTH}
  [ -f $OUT/$name/_DONE ] && continue
  MR=""; [ $MU = low ] && MR="--mu-range 5e-4,1.5e-3"
  $PY -m world_model.v2.tray_rep.wrongpoint.run_cell --cell $name --I $I --eps-scale $E --auth $AUTH $MR --n-eps 100 --workers 12 --out $OUT && touch $OUT/$name/_DONE
done; done; done; done
touch $OUT/_SWEEP1_DONE
