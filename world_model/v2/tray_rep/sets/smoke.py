"""sets/smoke.py -- smoke of F5 in the bridge harness. It only has to RUN; it is not an evaluation. SIMULATION ONLY. CPU.

  python -m world_model.v2.tray_rep.sets.smoke [--n 20] [--workers 2] [--coverage]

Runs F0 and F5 on the first n VAL carries in both setups (piper = noise-free, piper_trk = realistic tracker), F5 twice (1 worker and
`workers` workers) to check determinism, and writes <out>/smoke_summary.json with: drop / intervention / progress per run, F5 status
fractions and fallback counts, per-tick filter cost, and (with --coverage) the reachable-set coverage on the stored VAL branch files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.bridge import run as R

OUT = K.ROOT / "sets" / "smoke"
TIMING = ("wall_s", "filt_s", "sim_s", "x_f5_ms", "x_f5_ms_max", "x_f5_ms_mean")


def recs_of(out, tag, setup, spec):
    _, pj, _ = R.paths(out, tag, "val", setup, spec, 0)
    rows = sorted(R.read_jsonl(pj), key=lambda r: r["ep"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {}
    for setup in ("piper", "piper_trk"):
        s = {}
        for name, spec, tag, w in (("F0", "F0", "f0", 1), ("F5", "F5", "f5_w1", 1), ("F5_par", "F5", "f5_wN", a.workers)):
            t0 = time.perf_counter()
            r = R.run(spec, setup, 0, "val", a.n, w, tag, out)
            rows = recs_of(out, tag, setup, spec)
            s[name] = dict(drop=r["drop"]["k"], n=r["drop"]["n"], interv_rate=r["interv_rate"], dev_norm=r["mean_dev_norm"],
                           saved=r["saved"], caused=r["caused"], progress=float(np.mean([x["progress"] for x in rows])),
                           wall_s=time.perf_counter() - t0)
            if name.startswith("F5"):
                ticks = sum(x["n_ticks"] for x in rows)
                ms_ep = [x["filt_s"] / x["n_ticks"] * 1e3 for x in rows]
                s[name].update(ms_per_tick_mean=sum(x["filt_s"] for x in rows) / ticks * 1e3, ms_per_tick_median_ep=float(np.median(ms_ep)),
                               ms_tick_max=max(x["x_f5_ms_max"] for x in rows),
                               status_true=sum(x["x_f5_true"] for x in rows) / ticks, status_unknown=sum(x["x_f5_unknown"] for x in rows) / ticks,
                               status_false=sum(x["x_f5_false"] for x in rows) / ticks,
                               qp_solved=sum(x["x_f5_qp"] for x in rows), backup=sum(x["x_f5_backup"] for x in rows),
                               qp_infeasible=sum(x["x_f5_infeas"] for x in rows), relin=sum(x["x_f5_relin"] for x in rows),
                               hw_end_mm=float(np.mean([x["x_f5_hw_end_mm"] for x in rows])))
        a1, a2 = recs_of(out, "f5_w1", setup, "F5"), recs_of(out, "f5_wN", setup, "F5")
        strip = lambda rs: [{k: v for k, v in r.items() if k not in TIMING} for r in rs]
        s["deterministic_1_vs_N_workers"] = bool(strip(a1) == strip(a2) and len(a1) == a.n)
        summ[setup] = s
        print(setup, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in s.items()}, indent=1), flush=True)
    if a.coverage:
        from world_model.v2.tray_rep.sets import validate_reach as V
        from world_model.v2.tray_rep.sets import reach as RC
        tmp = K.ROOT / "sets" / "tmp"
        cov = {}
        for setup in ("piper", "piper_trk"):
            for tag, start in (("tune_ep0-79", 0), ("heldout_ep80-159", 80)):
                p = tmp / f"branches_{setup}_val_kd3_kp2_H60_n80_s{start}.npz"
                if p.exists():
                    meta = json.loads((K.ROOT / f"data_{setup}_full" / "meta.json").read_text())
                    cov[f"{setup}/{tag}"] = V.evaluate(V.load_records(p), RC.ReachParams(), 5.0 * meta["a_scale"], 2.0 * meta["w_scale"], meta["alpha"])
        summ["coverage"] = cov
        print(json.dumps(cov, indent=1), flush=True)
    (out / "smoke_summary.json").write_text(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
