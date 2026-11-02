#!/bin/bash
cd /common/users/ss5772/bimanual
export PYTHONPATH=$PWD MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
if [ "$1" = pot ]; then
  nice -n 19 $PY /common/home/ss5772/.tmp/claude-1905239669/-common-users-ss5772-bimanual/0fdf97c2-ab12-45ef-8d3a-31b6454afd0a/scratchpad/coupled/pot_claw.py $2 /data/local/ss5772/pan_cert/pot_claw40_2026-11-01 40 40 > /data/local/ss5772/pan_cert/pot_claw40_2026-11-01/log_$2.txt 2>&1
else
  hm=""; [ "$2" != none ] && hm="--hidden-mass $3"
  nice -n 19 $PY -m world_model.v2.nero.pan.handle_grasp --placement $2 $hm --grip force --grip-force $4 --claw-N $4 --out /data/local/ss5772/pan_cert/onecaw_sizing_2026-11-01/$2_$3_F$4.json > /data/local/ss5772/pan_cert/onecaw_sizing_2026-11-01/log_$2_$3_F$4.txt 2>&1
fi
