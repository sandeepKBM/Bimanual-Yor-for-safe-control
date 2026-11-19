#!/bin/bash
# owner run 11-17: box, 9 cells, n=20, a=0.5 (prewarm n=1 first so workers do not race on the geometry cache)
cd /common/users/ss5772/bimanual/.claude/worktrees/basestop-1116
export PYTHONPATH=$PWD BIMANUAL_NO_NETWORK=1 BIMANUAL_PLANNER=closed_chain OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
D=world_model/v2/nero/results/accel_k_2026-11-17
C=0.5-1.0,1.0-1.5,1.5-2.0,2.0-2.5,2.5-3.0,3.0-3.5,3.5-4.0,4.0-4.5,4.5-5.0
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
$PY -u $D/measure_k.py --task box --cells $C --n 1 --procs 9 --out $D/box_a0.5 && \
$PY -u $D/measure_k.py --task box --cells $C --n 20 --procs 30 --out $D/box_a0.5
echo EXIT $?
