"""replay_cli.py -- one entry point for the FL-MPC bitwise replay harness.

  python replay_cli.py record --out DIR --seeds 0 1 [--profiles flagship ce] [--synthetic] [--jobs 3]
  python replay_cli.py check  --records DIR [--coverage] [--atol 1e-9]
  python replay_cli.py selftest --records DIR

See README.md in this directory. SIMULATION ONLY (CPU, no video, no hardware / network / ROS).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cmd_record(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES="")
    jobs = []
    if not a.no_episodes:
        for prof in a.profiles:
            for s in a.seeds:
                jobs.append((f"ep_{prof}_s{s}", [sys.executable, "-u", str(HERE / "record_episode.py"), "--profile", prof,
                                                  "--seed", str(s), "--out", str(out / f"ep_{prof}_s{s}"),
                                                  "--max-wall-s", str(a.max_wall_s),
                                                  "--models-dir", str(out / "models")]))
    if a.synthetic:
        jobs.append(("synthetic", [sys.executable, "-u", str(HERE / "record_synthetic.py"), "--out",
                                   str(out / "synthetic_s7"), "--seed", "7",
                                   "--models-dir", str(out / "models")]))
    running = []
    for name, cmd in jobs:
        while len(running) >= a.jobs:
            running[0][1].wait()
            running.pop(0)
        log = open(out / "logs" / f"rec_{name}.log", "w")
        print(f"[record] start {name} -> {log.name}", flush=True)
        running.append((name, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
    for name, p in running:
        rc_ = p.wait()
        print(f"[record] {name} exit code {rc_}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0])
    r.add_argument("--profiles", nargs="+", default=["flagship"], choices=["flagship", "ce", "gate"])
    r.add_argument("--synthetic", action="store_true", help="also record the random-state branch-coverage set")
    r.add_argument("--no-episodes", action="store_true")
    r.add_argument("--jobs", type=int, default=2, help="parallel recorder processes (1 core each; keep <= 4)")
    r.add_argument("--max-wall-s", type=float, default=420.0)
    c = sub.add_parser("check")
    c.add_argument("--records", required=True)
    c.add_argument("--coverage", action="store_true")
    c.add_argument("--atol", type=float, default=1e-9)
    c.add_argument("--limit-per-class", type=int, default=None)
    c.add_argument("--cls", default=None)
    c.add_argument("--json", default=None)
    t = sub.add_parser("selftest")
    t.add_argument("--records", required=True)
    t.add_argument("--limit-per-class", type=int, default=25)
    a = ap.parse_args()
    if a.cmd == "record":
        return cmd_record(a)
    sys.path.insert(0, str(HERE))
    if a.cmd == "check":
        import check
        argv = ["--records", a.records, "--atol", str(a.atol)]
        if a.coverage: argv.append("--coverage")
        if a.limit_per_class: argv += ["--limit-per-class", str(a.limit_per_class)]
        if a.cls: argv += ["--cls", a.cls]
        if a.json: argv += ["--json", a.json]
        return check.main(argv)
    import selftest
    return selftest.main(["--records", a.records, "--limit-per-class", str(a.limit_per_class)])


if __name__ == "__main__":
    sys.exit(main())
