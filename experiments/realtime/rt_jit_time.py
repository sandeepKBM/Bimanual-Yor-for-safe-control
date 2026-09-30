"""JIT compile vs cache-load time of the rt_flmpc / rt_pipe Numba kernels (reported separately
from all per-step timings). SIMULATION ONLY. Run twice with the same NUMBA_CACHE_DIR: the first
run compiles (empty cache), the second loads from cache.
  NUMBA_CACHE_DIR=/tmp/x python rt_jit_time.py ; NUMBA_CACHE_DIR=/tmp/x python rt_jit_time.py"""
import json
import os
import sys
import time

t_imp0 = time.perf_counter()
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_REPO, os.path.join(_REPO, "scripts"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import cup_collect_task_utils as cct  # noqa: E402
import rt_flmpc  # noqa: E402
import rt_pipe  # noqa: E402
import sim_common as sc  # noqa: E402
from rt_dynamics import LeanRobotDynamics  # noqa: E402
t_imp = time.perf_counter() - t_imp0
m, d = cct.load_yor_cup_collect_model(keep_zone_marker=False)
_, lq, ld = sc.joint_addrs(m, sc.LEFT_ARM_JOINTS)
_, rq, rd = sc.joint_addrs(m, sc.RIGHT_ARM_JOINTS)
lee = m.body("left_arm_gripper_base").id
ree = m.body("right_arm_gripper_base").id
kw = dict(NC=4, L=8, W_POS_EGO=800.0, ee_speed_cap=0.28, ee_speed_body_int=ree, cone_mode="true")
lk = sc.barrier_keypoints(m, sc.EGO_PREFIX)
rk = sc.barrier_keypoints(m, sc.INT_PREFIX)
cfg = dict(speed_body=ree, left_table_kps=lk, right_table_kps=rk, table_z=0.7, table_margin=0.006, table_kp_radius=True)
t0 = time.perf_counter()
fl = rt_flmpc.RTFLMPC(m, **kw)
t_ctor = time.perf_counter() - t0
fl.configure_lean(LeanRobotDynamics(m), m, lq, ld, rq, rd, lee, **cfg)
t0 = time.perf_counter()
pp = rt_pipe.PipelinedRTFLMPC(m, delay_steps=1, **kw)
pp.configure_lean(LeanRobotDynamics(m), m, lq, ld, rq, rd, lee, **cfg)
t_pipe = time.perf_counter() - t0
pp.close()
print(json.dumps(dict(numba_cache_dir=os.environ.get("NUMBA_CACHE_DIR"), import_s=t_imp,
                      rt_ctor_s=t_ctor, rt_warmup_rt_full_s=fl.jit_warmup_s,
                      rt_pipe_ctor_plus_warmup_s=t_pipe, rt_pipe_warmup_s=pp.jit_warmup_s)))
