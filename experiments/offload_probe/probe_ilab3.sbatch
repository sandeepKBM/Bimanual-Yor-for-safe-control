#!/bin/bash
# Tiny capability probe for ilab3 (runbook-compliant: 1 GPU, short --time, finite, _DONE sentinel,
# no --cpus-per-task). Answers: how many CPU cores does a job get? Is /common shared? Do our
# python env + MuJoCo EGL rendering work on the node?
#SBATCH --job-name=bim_probe
#SBATCH --nodelist=ilab3
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/common/users/ss5772/bimanual/experiments/offload_probe/probe_%j.out
set -u
OUT=/common/users/ss5772/bimanual/experiments/offload_probe
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
echo "host=$(hostname) nproc=$(nproc)"
grep -E "Cpus_allowed_list" /proc/self/status
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-} SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
ls -d /common/users/ss5772/bimanual /common/home/ss5772 >/dev/null && echo "FS_OK"
$PY -c "import mujoco, numpy, numba; print('PY_OK mujoco', mujoco.__version__, 'numba', numba.__version__)"
MUJOCO_GL=egl $PY -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>'); r=mujoco.Renderer(m,64,64); d=mujoco.MjData(m); r.update_scene(d); print('EGL_OK', r.render().shape)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/_DONE"
