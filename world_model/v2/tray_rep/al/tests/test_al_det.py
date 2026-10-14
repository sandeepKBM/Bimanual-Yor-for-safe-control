"""al/tests/test_al_det.py -- determinism of the AL runner: (a) 1 worker vs 2 workers give identical records, (b) resume (run half, rerun to full) = fresh run. SIMULATION ONLY. CPU.
  python -m world_model.v2.tray_rep.al.tests.test_al_det --policy AL_probe --regime normal --n 6 --tmp <dir>
Compared: every field except wall time / ms timings. Uses subprocesses of al.run_al (so the module-level state of a worker pool is exercised for real).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(policy, regime, out, n, workers, extra=()):
    cmd = [sys.executable, "-m", "world_model.v2.tray_rep.al.run_al", "--policy", policy, "--regime", regime, "--split", "val", "--n-eps", str(n), "--workers", str(workers),
           "--out", str(out), *extra]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=os.environ.copy())


def recs(out, regime, policy):
    from world_model.v2.tray_rep.al.run_al import policy_tag
    rs = {}
    for f in (Path(out) / regime / policy_tag(policy)).glob("AL__*.jsonl"):
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            rs[r["ep"]] = r
    return rs


def strip(r):
    return {k: v for k, v in r.items() if "wall" not in k and not k.endswith("_ms") and "_ms_" not in k and k not in ("x_f5_ms", "x_f5_ms_mean", "x_f5_ms_max", "x_p_ms")}


def same(a, b):
    bad = []
    if set(a) != set(b):
        return [("eps", sorted(a), sorted(b))]
    for ep in a:
        x, y = strip(a[ep]), strip(b[ep])
        for k in x:
            if json.dumps(x[k], sort_keys=True, default=float) != json.dumps(y.get(k), sort_keys=True, default=float):
                bad.append((ep, k, str(x[k])[:60], str(y.get(k))[:60]))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="AL_probe")
    ap.add_argument("--regime", default="normal")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--voi", default="")
    a = ap.parse_args()
    t = Path(a.tmp)
    a.extra = ["--voi", a.voi] if a.voi else []
    run(a.policy, a.regime, t / "w1", a.n, 1, a.extra)
    run(a.policy, a.regime, t / "w2", a.n, 2, a.extra)
    run(a.policy, a.regime, t / "res", a.n // 2, 1, a.extra)          # first half ...
    run(a.policy, a.regime, t / "res", a.n, 2, a.extra)               # ... resumed to the full set (different worker count too)
    r1, r2, rr = (recs(t / d, a.regime, a.policy) for d in ("w1", "w2", "res"))
    b12, b1r = same(r1, r2), same(r1, rr)
    print(f"{a.policy} {a.regime}: 1 vs 2 workers {'IDENTICAL' if not b12 else 'DIFF ' + str(b12[:3])}; resume vs fresh {'IDENTICAL' if not b1r else 'DIFF ' + str(b1r[:3])} (n={len(r1)})")
    sys.exit(0 if not (b12 or b1r) else 1)


if __name__ == "__main__":
    main()
