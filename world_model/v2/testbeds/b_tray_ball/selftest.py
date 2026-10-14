"""selftest.py -- quick checks (CPU, ~10 s):  python -m world_model.v2.testbeds.b_tray_ball.selftest"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from world_model.v2.testbeds.b_tray_ball.tray_env import TrayEnv, B, kin_step_fast, DT, STRIDE
from world_model.v2.testbeds.b_tray_ball.carry import make_carry, carry_u
from world_model.v2.testbeds.b_tray_ball import probe as P

res = {}
# 1. fast kin_step == bead_env.kin_step (the reference, imported unedited)
rng = np.random.default_rng(0)
pos, v, q = np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]); pos2, v2, q2 = pos.copy(), v.copy(), q.copy()
err = 0.0
U = B.fallback_U()
for i in range(4000):
    u = U[rng.integers(85)] * rng.uniform(0.2, 1)
    pos, v, q, _ = B.kin_step(pos, v, q, u[:3], u[3:])
    pos2, v2, q2 = kin_step_fast(pos2, v2, q2, u[:3], u[3:])
    err = max(err, np.abs(pos - pos2).max(), np.abs(v - v2).max(), np.abs(q - q2).max())
res["kin_step_fast_vs_bead_env_max_abs_err"] = float(err)
# 2. brake
v_ = np.array([0.3, -0.2, 0.0]); c_ = np.array([0.2, 0.1, np.sqrt(0.95)])
res["brake_nb_vs_bead_max_abs_err"] = float(np.abs(P._brake(v_, c_, 5.0, 2.0, 0.02) - B.brake_action(v_, c_)).max())
# 3. determinism + snapshot/restore (bitwise)
def run(seed, stop=None, snap_at=None):
    rng = np.random.default_rng(seed); env = TrayEnv(); env.reset(env.home, np.array([1.0, 0, 0, 0]), ball_xy=rng.normal(0, 0.005, 2))
    cr = make_carry(rng, 1.7); sn = None; tr = []
    for k in range(120):
        if k == snap_at: sn = env.snapshot()
        env.tick(carry_u(cr, k * 0.02, env.axis())); tr.append(env.d.qpos[env.bq:env.bq + 7].copy())
    return env, np.array(tr), sn, cr
e1, t1, sn, cr = run(5, snap_at=60); e2, t2, _, _ = run(5)
res["determinism_bitwise"] = bool(np.array_equal(t1, t2))
# replay from the snapshot with the same remaining actions: re-generate the actions from a fresh carry controller state
rng = np.random.default_rng(5); env = TrayEnv(); env.reset(env.home, np.array([1.0, 0, 0, 0]), ball_xy=rng.normal(0, 0.005, 2))
cr = make_carry(rng, 1.7); acts = []
for k in range(120):
    u = carry_u(cr, k * 0.02, env.axis()); acts.append(u); env.tick(u)
env2 = TrayEnv(); env2.restore(sn); tr = []
for k in range(60, 120):
    env2.tick(acts[k]); tr.append(env2.d.qpos[env2.bq:env2.bq + 7].copy())
res["snapshot_restore_bitwise"] = bool(np.array_equal(np.array(tr), t1[60:120]))
# 4. speed
env = TrayEnv(); env.reset(env.home, np.array([1.0, 0, 0, 0])); t0 = time.perf_counter(); n = 0
for k in range(1500):
    env.tick(U[k % 85] * 0.2); n += 1
    if env.status: env.reset(env.home, np.array([1.0, 0, 0, 0]))
res["mujoco_ticks_per_s_1core"] = n / (time.perf_counter() - t0); res["mujoco_physics_steps_per_s_1core"] = 4 * n / (time.perf_counter() - t0)
res["rollouts_2s_per_s_1core"] = res["mujoco_ticks_per_s_1core"] / 100
print(json.dumps(res, indent=1)); P.merge_json("selftest", res)
