#!/bin/bash
cd /common/users/ss5772/bimanual
export PYTHONPATH=$PWD MUJOCO_GL=osmesa
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
for set in full iid overload; do
  nice $PY -m world_model.v2.nero.pan.segcert library --out world_model/v2/nero/results/pan_segments_2026-10-29/lib3 --set $set --procs 28 >> world_model/v2/nero/results/pan_segments_2026-10-29/logs/library.log 2>&1
done
echo ALLDONE >> world_model/v2/nero/results/pan_segments_2026-10-29/logs/library.log
