#!/bin/bash
cd /common/users/ss5772/bimanual
export PYTHONPATH=$PWD MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for S in full iid overload; do
  nice -n 10 /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.v2.nero.pan.cert.contact_falsify --out world_model/v2/nero/results/contact_model_2026-10-30/run158 --set $S --procs 32 > world_model/v2/nero/results/contact_model_2026-10-30/run158/log_$S.txt 2>&1
done
/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.v2.nero.pan.cert.contact_falsify --out world_model/v2/nero/results/contact_model_2026-10-30/run158 --report > world_model/v2/nero/results/contact_model_2026-10-30/run158/report_stdout.txt 2>&1
echo done > world_model/v2/nero/results/contact_model_2026-10-30/run158/_ALL_DONE
