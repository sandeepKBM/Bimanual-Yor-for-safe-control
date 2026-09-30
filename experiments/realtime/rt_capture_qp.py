"""Capture the FL-MPC QPs solved by the CURRENT fast path (fast_barriers + fast_qp)
during one episode, for offline solver development / equivalence checks.
SIMULATION ONLY.

Saves <out>/qp_<profile>_s<seed>.npz with, for EVERY FLMPC.step call:
  per-step stats (n, m, feasible, fallback, solve wall time, #QP calls)
and, for every `--stride`-th step plus EVERY step whose hard QP was infeasible,
the full hard QP (H, f, lo, hi, A, b, x, lam, feasible) and the relaxed QP if any.
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

import rt_harness as H
import fast_barriers as fb
import fast_qp as fq

demo, sc, mpc_fl = H.demo, H.sc, H.mpc_fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="flagship")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=H.RUN_DIR + "/capture")
    args = ap.parse_args()
    import os
    os.makedirs(args.out, exist_ok=True)

    fb.install(demo)
    fq.install()
    qp_inner = sc.solve_constrained_box_qp
    cur = {"calls": []}

    def qp_rec(Hm, f, lo, hi, A=None, b=None, **kw):
        t0 = time.perf_counter()
        x, lam, feas = qp_inner(Hm, f, lo, hi, A, b, **kw)
        dt = time.perf_counter() - t0
        cur["calls"].append(dict(H=np.array(Hm), f=np.array(f), lo=np.array(lo), hi=np.array(hi),
                                 A=np.array(A), b=np.array(b), x=x.copy(), lam=lam.copy(),
                                 feas=bool(feas), dt=dt))
        return x, lam, feas

    stats = []
    qps = []
    base_step = mpc_fl.FLMPC.step       # FastFLMPC.step after fb.install

    def step_rec(self, *a, **k):
        cur["calls"] = []
        sc.solve_constrained_box_qp = qp_rec
        t0 = time.perf_counter()
        try:
            out = base_step(self, *a, **k)
        finally:
            sc.solve_constrained_box_qp = qp_inner
        dt = time.perf_counter() - t0
        calls = cur["calls"]
        i = len(stats)
        hard = calls[0]
        stats.append((i, hard["H"].shape[0], hard["A"].shape[0], hard["feas"], self.last_fallback,
                      hard["dt"], sum(c["dt"] for c in calls), dt, len(calls)))
        if (i % args.stride == 0) or (not hard["feas"]):
            qps.append(dict(i=i, calls=calls, fallback=self.last_fallback,
                            tau=(out[0].copy(), out[1].copy())))
        return out

    mpc_fl.FLMPC.step = step_rec
    info = H.build(args.profile, args.seed)
    t0 = time.perf_counter()
    log, summ = H.run(info, args.profile, max_steps=args.max_steps)
    wall = time.perf_counter() - t0
    st = np.array(stats, dtype=float)
    print(f"[capture] profile={args.profile} seed={args.seed}: {len(log['t'])} steps, {len(stats)} FL steps, "
          f"wall {wall:.1f}s; collision={summ['collision']} placed_final={summ['placed_final']} "
          f"h_min={summ['h_min']!r} fl_relaxed={summ.get('fl_relaxed_steps')}", flush=True)
    if len(st):
        print(f"  n={int(st[0,1])}  m: min {st[:,2].min():.0f} max {st[:,2].max():.0f}  infeasible {int((st[:,3]==0).sum())}"
              f"  fallback1 {int((st[:,4]==1).sum())} fallback2 {int((st[:,4]==2).sum())}")
        for lab, col in (("hard QP", 5), ("all QPs", 6), ("FLMPC.step", 7)):
            v = st[:, col] * 1e3
            print(f"  {lab:10s} ms: mean {v.mean():.3f} p50 {np.median(v):.3f} p99 {np.percentile(v,99):.3f} max {v.max():.3f}")
    fn = f"{args.out}/qp_{args.profile}_s{args.seed}.pkl"
    with open(fn, "wb") as fh:
        pickle.dump(dict(stats=st, qps=qps, summary=summ), fh, protocol=4)
    print(f"  saved {len(qps)} QP records -> {fn}")


if __name__ == "__main__":
    main()
