"""G3 driver: comprehensive_eval.run_one jobs (identical job dicts / metrics / JSON layout) with the
realtime FAST PATH (fast_barriers + fast_qp, bit-identical to the original for py) installed in each
worker, so the simulator's own per-step Python is light -- the in-process stand-in for a real robot
loop, which matters for --controller-impl rt_pipe (its solver thread shares the GIL with the sim).
SIMULATION ONLY.
  python rt_g3_driver.py --seeds 0-9 --controller-impl rt_pipe [--rt-pipe-inject 0.05:10,20] --workers 4 --out X.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scripts"), os.path.join(_REPO, "experiments", "ab_smooth_vs_bangbang"),
           os.path.join(_REPO, "eval"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _job(job):
    import comprehensive_eval as ce
    import _demo_env  # noqa: F401
    import fast_barriers as fb
    import fast_qp as fq
    import make_cup_collect_safety_demo as demo
    if not getattr(demo, "_RT_FAST_INSTALLED", False):
        fb.install(demo)
        fq.install()
        demo._RT_FAST_INSTALLED = True
    return ce.run_one(job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--controller-impl", default="rt_pipe")
    ap.add_argument("--rt-pipe-inject", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import comprehensive_eval as ce
    seeds = ce.parse_seed_range(args.seeds)
    jobs = [dict(seed=s, controller="flmpc", v4c={}, max_steps=None, v_ee_max=None, grasp_mode="faked",
                 bugfixes="on", weld_gate_m=None, ctrlfixes="on", ctrlfix_parts="all", ctrlfix_delay_steps=0,
                 ctrlfix_cone_facets=None, ctrlfix_tau_max_vec=None,
                 **({} if args.controller_impl == "py" else
                    dict(controller_impl=args.controller_impl, rt_pipe_inject=args.rt_pipe_inject)))
            for s in seeds]
    t0 = time.perf_counter()
    rows = []
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            print(f"[{i + 1}/{len(jobs)}] seed={r['seed']} ok={r.get('ok')} "
                  + (f"collision={r['collision']} h_min_ref={r['h_min_ref']:.4f} "
                     f"miss={r.get('fl_rt_pipe_miss_steps')} brake_age={r.get('fl_rt_pipe_brake_age')} "
                     f"brake_recheck={r.get('fl_rt_pipe_brake_recheck')}" if r.get("ok") else str(r.get("err"))),
                  flush=True)
    json.dump(dict(seeds=seeds, controllers=["flmpc"], controller_impl=args.controller_impl,
                   rt_pipe_inject=args.rt_pipe_inject, fast_path_installed=True, wall_s=time.perf_counter() - t0,
                   rows=rows, aggregate={"flmpc": ce.aggregate(rows, "flmpc")}), open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
