#!/bin/bash
cd /common/users/ss5772/bimanual
export PYTHONPATH=$PWD MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
D=world_model/v2/nero/results/pan_baseline_scripted_2026-10-29
for set in full iid overload; do
  nice $PY -m world_model.v2.nero.pan.baseline_scripted --out $D --set $set --procs 30 >> $D/logs/run.log 2>&1
done
echo ALLDONE >> $D/logs/run.log
