#!/bin/bash
cd /common/users/ss5772/bimanual
export PYTHONPATH=$PWD MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
hm=""; [ "$1" != none ] && hm="--hidden-mass $2"
nice -n 19 timeout 1200 /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.v2.nero.pan.handle_grasp --placement $1 $hm --grip force --grip-force 40 --claw-N 40 --nsub 5 --payload $3 --out /data/local/ss5772/pan_cert/payload_obs_full_2026-11-01/$1_$2_$3.json > /data/local/ss5772/pan_cert/payload_obs_full_2026-11-01/log_$1_$2_$3.txt 2>&1
