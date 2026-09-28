"""Record the REAL QP problems FL-MPC solves in the seed-0 flmpc episode and
benchmark / bit-compare controller_core's solve_constrained_box_qp vs
fast_qp.solve_constrained_box_qp on them. NEW file. SIMULATION ONLY.

  python bench_qp.py --record --max-steps 1400   # run episode, pickle QPs (FL steps only)
  python bench_qp.py                              # compare on the pickle
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

import _demo_env as env
import fast_qp as fq

demo, sc, mpc_fl = env.demo, env.sc, env.mpc_fl
PKL = env.SCRATCH / "flmpc_qps.pkl"


def record(max_steps, every, fast=False):
    if fast:   # drive the episode with the (bit-identical) fast path to record quicker
        import fast_barriers as fb
        fb.install(demo)
        fq.install()
    probs = []
    orig = sc.solve_constrained_box_qp
    k = {"n": 0}

    def rec(H, f, lo, hi, a=None, b=None, **kw):
        if H.shape[0] >= 48:     # FL-MPC joint QP (48) / relaxed fallback (48+m_coll)
            if k["n"] % every == 0:
                probs.append(tuple(np.array(x, copy=True) for x in (H, f, lo, hi, a, b)))
            k["n"] += 1
        return orig(H, f, lo, hi, a, b, **kw)
    sc.solve_constrained_box_qp = rec
    info = demo.build_episode(seed=0)
    demo.run_episode(info, controller="flmpc", capture=False, max_steps=max_steps)
    sc.solve_constrained_box_qp = orig
    pickle.dump(probs, open(PKL, "wb"))
    print(f"recorded {len(probs)} QPs (of {k['n']} FL-sized solves) -> {PKL}")


def compare():
    probs = pickle.load(open(PKL, "rb"))
    fq.warmup()
    from controller_core import constrained_box_qp as cbq
    cbq.numba_warmup()
    orig = cbq.solve_constrained_box_qp
    to, tf, same, ms = [], [], 0, []
    for (H, f, lo, hi, a, b) in probs:
        t0 = time.perf_counter(); xo, lo_, fo = orig(H, f, lo, hi, a, b); t1 = time.perf_counter()
        xf, lf, ff = fq.solve_constrained_box_qp(H, f, lo, hi, a, b); t2 = time.perf_counter()
        to.append(t1 - t0); tf.append(t2 - t1); ms.append(a.shape[0])
        same += int(np.array_equal(xo, xf) and np.array_equal(lo_, lf) and fo == ff)
    to, tf = np.array(to) * 1e3, np.array(tf) * 1e3
    print(f"{len(probs)} real FL-MPC QPs (n={probs[0][0].shape[0]}, m rows median {int(np.median(ms))}): "
          f"bit-identical (x, lambda, feasible) {same}/{len(probs)}")
    print(f"  original: mean {to.mean():.3f} ms  p50 {np.median(to):.3f}  p99 {np.percentile(to, 99):.3f}  max {to.max():.3f}")
    print(f"  fast_qp : mean {tf.mean():.3f} ms  p50 {np.median(tf):.3f}  p99 {np.percentile(tf, 99):.3f}  max {tf.max():.3f}")
    print(f"  speedup (mean) {to.mean()/tf.mean():.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--max-steps", type=int, default=1400)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--fast-record", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()
    if args.record:
        record(args.max_steps, args.every, args.fast_record)
    if not args.no_compare:
        compare()
