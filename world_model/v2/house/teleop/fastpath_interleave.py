import os, sys, time, json
os.environ["BIMANUAL_FLMPC_IMPL"] = "rt"
import numpy as np, mujoco
import world_model.v2.house.teleop.sim as S
import world_model.v2.house.teleop.fastpath as FP
FP.install_split_hook()
cfgs = [("before", 0, 0, None), ("A", 1, 0, None), ("A+B", 1, 1, None), ("A+B+ns1", 1, 1, 1), ("A+B+ns0", 1, 1, 0)]
sims = {}
for name, fast, split, ns in cfgs:
    os.environ["BIMANUAL_TELEOP_FAST"] = str(fast); os.environ["BIMANUAL_TELEOP_SPLIT"] = str(split)
    s = S.TeleopSim("h5", grasped=True)
    if ns is not None: s.m.opt.noslip_iterations = ns
    sims[name] = (s, split)
    for _ in range(5): s.tick(S.Cmd())
def cmd(k):
    ph = k / 25.0
    return S.Cmd(twist=np.array([0.4 * min(1, k / 20), 0.0, 0.3 * np.sin(ph)]), jog=np.array([0.04 * np.sin(ph), 0, 0.02 * np.cos(ph), 0, 0.05 * np.sin(ph), 0]))
res = {n: [] for n, *_ in cfgs}; cpu = {n: [] for n, *_ in cfgs}
k = 0
for rnd in range(14):
    for name, fast, split, ns in cfgs:
        s, sp = sims[name]
        os.environ["BIMANUAL_TELEOP_SPLIT"] = str(sp)
        t0 = time.perf_counter(); c0 = time.process_time()
        for j in range(10): s.tick(cmd(k + j))
        res[name].append(1e3 * (time.perf_counter() - t0) / 10); cpu[name].append(1e3 * (time.process_time() - c0) / 10)
    k += 10
out = {}
for n in res:
    w = np.array(res[n][2:]); c = np.array(cpu[n][2:])
    out[n] = dict(wall_median=float(np.median(w)), wall_min=float(w.min()), cpu_median=float(np.median(c)))
    print(f"{n:10s} tick ms wall median {np.median(w):6.1f} min {w.min():6.1f}  cpu median {np.median(c):6.1f}  rtf(median) {0.02/(np.median(w)/1e3):.2f}")
json.dump(out, open("data/molmospaces/teleop/perf/interleave.json", "w"))
