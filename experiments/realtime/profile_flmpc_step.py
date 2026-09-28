"""Per-component wall-time profile of the flagship seed-0 `flmpc` episode
(no video), pinned threads. NEW file; monkeypatches timing wrappers onto the
module attributes run_episode()/FLMPC.step() resolve at call time -- no edits to
any existing file. SIMULATION ONLY.

Usage:
  python experiments/realtime/profile_flmpc_step.py [--fast] [--max-steps N]
    --fast : install experiments/realtime fast path first (fast_barriers.install)

Prints a ranked table of EXCLUSIVE per-step time (nested wrappers subtract
their children), separately for the FL-MPC phase (steps where FLMPC.step ran)
and the lift phase, plus the episode summary metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

import _demo_env as env

demo, sc, mpc_fl = env.demo, env.sc, env.mpc_fl
import mujoco  # noqa: E402

_stack: list = []          # [name, t0, child_time]
_cur = defaultdict(float)  # exclusive seconds, current step
_cnt = defaultdict(int)
_steps: list = []
_flag = {"fl": False}
# --cpu: use per-thread CPU time (CLOCK_THREAD_CPUTIME_ID) instead of wall time,
# so a saturated shared machine (other jobs preempting us) doesn't inflate the
# numbers; the default is wall time (perf_counter).
_clock = [time.perf_counter]


def _now():
    return _clock[0]()


def _wrap(obj, attr, name):
    fn = getattr(obj, attr)

    def w(*a, **k):
        _stack.append([name, _now(), 0.0])
        try:
            return fn(*a, **k)
        finally:
            nm, t0, child = _stack.pop()
            dt = _now() - t0
            _cur[nm] += dt - child
            _cnt[nm] += 1
            if _stack:
                _stack[-1][2] += dt
    w.__wrapped__ = fn
    setattr(obj, attr, w)


def install_timers():
    for attr in ("full_mass_matrix", "payload_reflected_terms", "all_pair_barriers",
                 "table_plane_barrier_rows", "solve_constrained_box_qp", "min_pair_distance",
                 "body_jac_full"):
        _wrap(sc, attr, "sc." + attr)
    _wrap(mpc_fl, "_assemble_fl", "mpc_fl._assemble_fl")
    _wrap(mpc_fl.FLMPC, "_relaxed_fallback", "FLMPC._relaxed_fallback(excl QP)")
    step_orig = mpc_fl.FLMPC.step

    def step_w(self, *a, **k):
        _flag["fl"] = True
        return step_orig(self, *a, **k)
    mpc_fl.FLMPC.step = step_w
    _wrap(mpc_fl.FLMPC, "step", "FLMPC.step python overhead (excl children)")
    _wrap(demo, "right_arm_to_cup_clearance", "demo.right_arm_to_cup_clearance (log)")
    for attr in ("count_spilled", "count_deposited_cubes", "count_placed_cubes"):
        _wrap(demo.cct, attr, "cct." + attr + " (log)")
    for attr in ("reattach_cup_rigidly", "carry_balls_with_cup", "cup_velocity_from_motion"):
        _wrap(demo.scene_setup, attr, "scene_setup." + attr)
    _wrap(mujoco, "mj_forward", "mujoco.mj_forward")
    # mj_step marks the step boundary
    mj_step_orig = mujoco.mj_step
    last = {"t": None}

    def mj_step_w(m, d, *a):
        t0 = _now()
        mj_step_orig(m, d, *a)
        t1 = _now()
        _cur["mujoco.mj_step"] += t1 - t0
        if last["t"] is not None:
            rec = dict(_cur)
            rec["__wall"] = t1 - last["t"]
            rec["__fl"] = _flag["fl"]
            _steps.append(rec)
        last["t"] = t1
        _cur.clear()
        _flag["fl"] = False
    mujoco.mj_step = mj_step_w


def report(steps, label):
    if not steps:
        return
    n = len(steps)
    keys = sorted({k for s in steps for k in s if not k.startswith("__")})
    wall = np.array([s["__wall"] for s in steps])
    rows = []
    for k in keys:
        v = np.array([s.get(k, 0.0) for s in steps])
        rows.append((k, v.mean() * 1e3, np.percentile(v, 99) * 1e3))
    tracked = sum(r[1] for r in rows)
    rows.append(("(untracked: demo loop python, logging, etc.)", wall.mean() * 1e3 - tracked, float("nan")))
    rows.sort(key=lambda r: -r[1])
    print(f"\n=== {label} [{'THREAD CPU time' if _clock[0] is time.thread_time else 'wall time'}]: {n} steps, per-step mean {wall.mean()*1e3:.3f} ms  p50 {np.median(wall)*1e3:.3f}  "
          f"p99 {np.percentile(wall, 99)*1e3:.3f}  max {wall.max()*1e3:.3f}  (RTF {sc.DT/wall.mean():.3f}x)")
    print(f"{'component':58s} {'mean ms':>9s} {'p99 ms':>9s} {'share':>7s}")
    for k, m, p in rows:
        print(f"{k:58s} {m:9.3f} {p:9.3f} {100*m/(wall.mean()*1e3):6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--cpu", action="store_true", help="per-thread CPU time instead of wall time")
    args = ap.parse_args()
    if args.cpu:
        _clock[0] = time.thread_time
    if args.fast:
        import fast_barriers
        import fast_qp
        fast_barriers.install(demo)
        fast_qp.install()
    install_timers()
    info = demo.build_episode(seed=0)
    t0 = _now()
    log, _ = demo.run_episode(info, controller="flmpc", capture=False, max_steps=args.max_steps)
    wall = _now() - t0
    summ = demo.summarize(log)
    fl_steps = [s for s in _steps if s["__fl"]]
    lift_steps = [s for s in _steps if not s["__fl"]]
    report(fl_steps, "FL-MPC phase (FLMPC.step called)")
    report(lift_steps, "lift phase (v4c ramp + QP)")
    print(f"\nepisode wall {wall:.1f} s for {len(log['t'])} steps; "
          f"collision={summ['collision']} placed_final={summ['placed_final']} h_min={summ['h_min']!r}")
    if args.json_out:
        json.dump(dict(summary=summ, wall_s=wall), open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
