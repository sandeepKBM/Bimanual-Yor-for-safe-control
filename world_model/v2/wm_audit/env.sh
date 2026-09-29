#!/bin/bash
# WM audit env: everything on /common/users (home has a 50 GB quota). SIM / SAVED DATA ONLY.
export AUD=/common/users/ss5772/bimanual_data/wm_audit
export LATENT_HJ_SCR=/common/users/ss5772/latent_hj_scratch      # read-only use: wmdata, v0 value, kin_wm ckpt
export TMPDIR=$AUD/tmp MPLCONFIGDIR=$AUD/tmp/mpl XDG_CACHE_HOME=$AUD/tmp/cache TORCH_HOME=$AUD/tmp/cache/torch
export CUDA_CACHE_PATH=$AUD/tmp/nv PYTHONPYCACHEPREFIX=$AUD/tmp/pyc NUMBA_CACHE_DIR=$AUD/tmp/cache/numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
cd /common/users/ss5772/bimanual
