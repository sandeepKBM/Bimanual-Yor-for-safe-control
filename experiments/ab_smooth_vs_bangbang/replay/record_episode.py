"""record_episode.py -- record FLMPC.step calls from a SHORT flagship-demo episode (CPU, no video).

Runs the same code path as `scripts/make_cup_collect_safety_demo.py --controller flmpc --no-video`
(via experiments/realtime/rt_harness.py, the profiles the rt gates use), with FLMPC.step / __init__
wrapped at run time (monkeypatch in THIS process only; no existing file is edited).

  python record_episode.py --profile flagship --seed 0 --out <dir> [--max-wall-s 420] [--stride 25]

SIMULATION ONLY: no hardware / network / ROS.
"""
from __future__ import annotations

import argparse
import json
import time

import replay_core as rc


class _Stop(Exception):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="flagship", choices=("flagship", "ce", "gate"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default=None, help="shared model store (default <out>/models)")
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--max-records", type=int, default=250)
    ap.add_argument("--max-special", type=int, default=100)
    ap.add_argument("--max-wall-s", type=float, default=420.0, help="stop the episode after this much RUN time")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)

    import rt_harness as H  # experiments/realtime (pure computation; imports the demo module)
    demo, sc, mpc_fl = H.demo, H.sc, H.mpc_fl

    rec = rc.Recorder(args.out, sc, source=f"episode:{args.profile}:s{args.seed}", stride=args.stride,
                      max_records=args.max_records, max_special=args.max_special, models_dir=args.models_dir)
    counter = [0]
    orig_step, orig_init = rec.install_class_patch(mpc_fl, counter)
    patched_step = mpc_fl.FLMPC.step
    t_run = [None]

    def guarded(self, model, data, *a, **k):
        if t_run[0] is None:
            t_run[0] = time.perf_counter()
        if time.perf_counter() - t_run[0] > args.max_wall_s:
            raise _Stop()
        return patched_step(self, model, data, *a, **k)

    mpc_fl.FLMPC.step = guarded
    t0 = time.perf_counter()
    print(f"[rec-ep] building episode profile={args.profile} seed={args.seed} ...", flush=True)
    info = H.build(args.profile, args.seed)
    print(f"[rec-ep] built in {time.perf_counter() - t0:.0f}s; running ...", flush=True)
    t1 = time.perf_counter()
    status = "completed"
    summ = None
    try:
        log, summ = H.run(info, args.profile, max_steps=args.max_steps)
    except _Stop:
        status = f"stopped at wall limit {args.max_wall_s}s"
    finally:
        mpc_fl.FLMPC.step, mpc_fl.FLMPC.__init__ = orig_step, orig_init
    meta = dict(profile=args.profile, seed=args.seed, status=status, step_calls=counter[0],
                records_saved=rec.n_saved, special=rec.n_special, fallback_hist=rec.stats["fallback"],
                run_wall_s=time.perf_counter() - t1,
                summary={k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in (summ or {}).items()
                         if k in ("collision", "placed_final", "h_min")})
    print("[rec-ep]", json.dumps(meta), flush=True)
    with open(f"{args.out}/meta_{args.profile}_s{args.seed}.json", "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
