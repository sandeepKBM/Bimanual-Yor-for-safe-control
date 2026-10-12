"""bridge/smoke_checks.py -- end-to-end checks of the bridge harness on a handful of carries. CPU ONLY. SIMULATION / SAVED DATA ONLY.

  python -m world_model.v2.tray_rep.bridge.smoke_checks [--n 8] [--workers 4]

Uses the smoke1b models (tiny, untrained: the numbers are meaningless; only the plumbing is tested). Checks:
  1. F0 reproduces the stored nominal carry: drop tick == stored onset for every carry (both setups).
  2. determinism: the same (filter, setup, seed) twice, one with 1 worker and one with several, gives identical records
     (every field except wall-clock timings), for F1 and F3 in both setups.
  3. resume: a run of n/2 carries followed by a run of n carries equals one run of n carries.
  4. seed effect: in the tracker setup run seed 1 changes the tracker noise stream (F1 differs from seed 0); in the noise-free
     setup it does not.
  5. no hardware code was imported (module list scan).
Writes <out>/smoke/checks.json (run dirs: <out>/smoke/smoke_chk_*). Exit code 0 = all pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.bridge import run as R

TIMING = ("wall_s", "filt_s", "sim_s")
S = K.ROOT / "smoke1b" / "runs"


def recs(out, tag, split, setup, spec, seed):
    _, pj, _ = R.paths(out, tag, split, setup, spec, seed)
    rows = sorted(R.read_jsonl(pj), key=lambda r: r["ep"])
    return [{k: v for k, v in r.items() if k not in TIMING} for r in rows]


def go(out, tag, spec, setup, seed, n, workers, split="test"):
    return R.run(spec, setup, seed, split, n, workers, tag, out, runs_root=str(S), model_setup="piper")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(R.OUT))
    a = ap.parse_args()
    out = Path(a.out) / "smoke"
    n, W = a.n, a.workers
    res, ok_all = {}, True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        res[name] = dict(ok=bool(ok), detail=detail)
        print(("PASS " if ok else "FAIL ") + name, detail, flush=True)

    # 1. F0 reproduces the stored carry
    for setup in ("piper", "piper_trk"):
        go(out, "smoke_chk_f0", "F0", setup, 0, n, W)
        r = recs(out, "smoke_chk_f0", "test", setup, "F0", 0)
        bad = [x["ep"] for x in r if (x["drop_tick"] if x["dropped"] else -1) != x["onset_stored"]]
        check(f"f0_matches_stored_{setup}", len(r) == n and not bad,
              f"{len(r)} carries, {sum(x['dropped'] for x in r)} nominal drops, mismatches {bad}")

    # 2. determinism (1 worker vs several workers)
    specs = ["F1:l_align=4,l_prox=40", "F3:arm=N0,tau=0.0"]
    for setup in ("piper", "piper_trk"):
        for sp in specs:
            go(out, "smoke_chk_a", sp, setup, 0, n, 1)
            go(out, "smoke_chk_b", sp, setup, 0, n, W)
            ra, rb = (recs(out, t, "test", setup, sp, 0) for t in ("smoke_chk_a", "smoke_chk_b"))
            check(f"determinism_{setup}_{sp.split(':')[0]}", ra == rb and len(ra) == n,
                  f"{len(ra)} vs {len(rb)} records identical={ra == rb}")

    # 3. resume
    sp = "F3:arm=N0,tau=0.0"
    go(out, "smoke_chk_r", sp, "piper_trk", 0, n // 2, W)
    go(out, "smoke_chk_r", sp, "piper_trk", 0, n, W)
    rr, rf = recs(out, "smoke_chk_r", "test", "piper_trk", sp, 0), recs(out, "smoke_chk_b", "test", "piper_trk", sp, 0)
    check("resume_equals_fresh", rr == rf and len(rr) == n, f"{len(rr)} records, identical={rr == rf}")

    # 4. seed effect
    sp = "F1:l_align=4,l_prox=40"
    for setup, expect_diff in (("piper_trk", True), ("piper", False)):
        go(out, "smoke_chk_s1", sp, setup, 1, n, W)
        r0, r1 = recs(out, "smoke_chk_a", "test", setup, sp, 0), recs(out, "smoke_chk_s1", "test", setup, sp, 1)
        strip = lambda rs: [{k: v for k, v in x.items() if k != "seed"} for x in rs]
        diff = strip(r0) != strip(r1)
        check(f"seed_changes_tracker_stream_{setup}", diff == expect_diff, f"differs={diff} expected={expect_diff}")

    # 4b. path-progress fields (added with the boed work): present, in range, and the old fields still reproduce the stored carry
    for setup in ("piper", "piper_trk"):
        r = recs(out, "smoke_chk_f0", "test", setup, "F0", 0)
        newf = ("path_len_nom", "path_len_done", "progress", "goal_tick", "goal_reached")
        ok = all(all(k in x for k in newf) and 0.0 <= x["progress"] <= 1.0 and x["path_len_done"] >= 0 for x in r)
        check(f"path_progress_fields_{setup}", ok and len(r) == n,
              f"progress {[round(x['progress'], 2) for x in r[:6]]} (stored nominal carries cover about 30% of their path)")

    # 5. no hardware imports
    bad = [m for m in sys.modules if m.startswith("YOR.robot") or m.split(".")[0] in ("piper_sdk", "can", "xarm", "rclpy", "serial")]
    check("no_hardware_modules_imported", not bad, str(bad))

    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").write_text(json.dumps(res, indent=1))
    print("ALL PASS" if ok_all else "SOME FAILED", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
