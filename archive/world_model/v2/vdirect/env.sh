#!/bin/bash
# vdirect env. SIMULATION / SAVED DATA ONLY. Outputs on local NVMe (the /common/users quota filled twice).
export VD_ROOT=/data/local/ss5772/bimanual_runs/vdirect
export TMPDIR=$VD_ROOT/tmp MPLCONFIGDIR=$VD_ROOT/tmp/mpl XDG_CACHE_HOME=$VD_ROOT/tmp/cache TORCH_HOME=$VD_ROOT/tmp/cache/torch
export CUDA_CACHE_PATH=$VD_ROOT/tmp/nv PYTHONPYCACHEPREFIX=$VD_ROOT/tmp/pyc NUMBA_CACHE_DIR=$VD_ROOT/tmp/cache/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
