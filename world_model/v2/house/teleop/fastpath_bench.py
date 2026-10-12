"""bench.py: run N ticks of the h5 carry with given env flags; record torques & per-component timing; save npz/json."""
import argparse, os, sys, time, json
ap = argparse.ArgumentParser()
ap.add_argument("--fast", type=int, default=0); ap.add_argument("--split", type=int, default=0)
ap.add_argument("--impl", default="rt"); ap.add_argument("--ticks", type=int, default=100)
ap.add_argument("--noslip", type=int, default=None); ap.add_argument("--tol", type=float, default=None)
ap.add_argument("--noise", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None); ap.add_argument("--prof", type=int, default=1)
ap.add_argument("--hook", type=int, default=0); ap.add_argument("--warm", type=int, default=5)
a = ap.parse_args()
os.environ["BIMANUAL_TELEOP_FAST"] = str(a.fast); os.environ["BIMANUAL_TELEOP_SPLIT"] = str(a.split)
os.environ["BIMANUAL_FLMPC_IMPL"] = a.impl
import numpy as np, mujoco
import world_model.v2.house.teleop.sim as S
import world_model.v2.house.teleop.arm as A
if not a.noise:
    S.ENC_Q = S.ENC_V = 0.0
import world_model.v2.house.teleop.fastpath as FP
if a.split or a.hook: print('hook installed:', FP.install_split_hook())
sim = S.TeleopSim("h5", grasped=True, seed=a.seed)
m = sim.m
if a.noslip is not None: m.opt.noslip_iterations = a.noslip
if a.tol is not None: m.opt.tolerance = a.tol
# record torques
rec = []
orig_tau = A.ArmCtl.tau
def tau(self, d):
    te, ti = orig_tau(self, d); rec.append(np.concatenate([te, ti])); return te, ti
A.ArmCtl.tau = tau
acc = {}
if a.prof:
    def wrap(obj, name, key):
        f = getattr(obj, name)
        def g(*x, **k):
            t0 = time.perf_counter(); r = f(*x, **k); acc[key] = acc.get(key, 0) + time.perf_counter() - t0; return r
        setattr(obj, name, g)
    wrap(mujoco, "mj_step", "mj_step"); wrap(mujoco, "mj_step1", "mj_step1"); wrap(mujoco, "mj_step2", "mj_step2")
    wrap(mujoco, "mj_forward", "mj_forward"); wrap(mujoco, "mj_makeM", "makeM"); wrap(mujoco, "mj_copyData", "copyData")
    wrap(sim.ctl, "tau", "ctl.tau")
    wrap(sim.ctl.fl, "step", "fl.step")
    wrap(sim.ctl.sc, "all_pair_barriers", "barriers_sc")
    import fast_barriers as FB
    wrap(FB, "all_pair_barriers", "barriers_fast")
    wrap(sim, "_count_bumps", "bumps"); wrap(sim, "bead_check", "bead_check"); wrap(sim, "set_people", "people")
    wrap(sim.of, "jog", "jog")
cmd = S.Cmd()
for _ in range(a.warm):
    sim.tick(cmd)
rec.clear(); acc.clear()
outs = []
t0 = time.perf_counter()
for k in range(a.ticks):
    ph = k / 25.0
    cmd = S.Cmd(twist=np.array([0.4 * min(1, k / 20), 0.0, 0.3 * np.sin(ph)]), jog=np.array([0.04 * np.sin(ph), 0, 0.02 * np.cos(ph), 0, 0.05 * np.sin(ph), 0]))
    outs.append(sim.tick(cmd))
wall = time.perf_counter() - t0
o = outs[-1]
res = dict(args=vars(a), tick_ms=1e3 * wall / a.ticks, rtf=(a.ticks * S.TICK) / wall, step_ms=1e3 * wall / (4 * a.ticks),
           beads_out=int(o["beads_out"]), cup_lost=sim.cup_lost_t, max_cup_tilt=max(x["cup_tilt"] for x in outs),
           max_tray_tilt=max(x["tray_tilt"] for x in outs), infeasible=int(sum(x["infeasible"] for x in outs)),
           final_rel=[float(v) for v in o["rel"]], final_pose=[float(v) for v in o["pose"]], bumps=int(o["bump_contacts"]),
           max_tray_acc=float(max(np.linalg.norm(x["tray_acc"]) for x in outs)),
           ncon=int(sim.d.ncon), nefc=int(sim.d.nefc),
           per_tick_ms={k: 1e3 * v / a.ticks for k, v in acc.items()})
print(json.dumps(res))
if a.out:
    np.savez(a.out, tau=np.array(rec), rel=np.array([x["rel"] for x in outs]), pose=np.array([x["pose"] for x in outs]),
             cup_tilt=np.array([x["cup_tilt"] for x in outs]), qpos=sim.d.qpos.copy())
