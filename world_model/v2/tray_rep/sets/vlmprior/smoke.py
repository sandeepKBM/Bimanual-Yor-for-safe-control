"""sets/vlmprior/smoke.py -- plumbing smoke of the five Experiment-B arms: n VAL carries x arms. NOT evidence. SIMULATION ONLY. CPU.

  python -m world_model.v2.tray_rep.sets.vlmprior.smoke [--n 20] [--rho 0.2] [--setup piper_trk] [--workers 10] [--pseed 0]
Writes <out>/<tag>/... via bridge.run (jsonl per arm) and <out>/<tag>/smoke_table.{json,md}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.bridge import run as R

OUT = K.ROOT / "vlmprior"
ARMS = ("point", "band", "conf", "unif", "oracle")
CK = ("t0", "t1s", "t2s", "t4s", "end")


def spec_of(arm, rho, pseed):
    return f"F5P:arm={arm}" + ("" if arm == "oracle" else f",rho={rho:g},pseed={pseed}")


def table(out, tag, setup, split, specs, seed=0):
    rows = []
    for name, spec in specs:
        _, pj, _ = R.paths(out, tag, split, setup, spec, seed)
        recs = sorted(R.read_jsonl(pj), key=lambda r: r["ep"])
        if not recs:
            continue
        n = len(recs)
        g = lambda k: np.array([r[k] for r in recs], float)
        ticks = max(g("n_ticks").sum(), 1.0)
        cov = {c: [r[f"x_p_cov_{c}"] for r in recs if r.get(f"x_p_cov_{c}") is not None] for c in CK}
        rows.append(dict(arm=name, spec=spec, n=n, drops=int(g("dropped").sum()), progress=float(g("progress").mean()),
                         interv=float(g("n_interv").sum() / ticks), dev_norm=float(g("dev_norm_sum").sum() / ticks),
                         cov={c: (float(np.mean(v)) if v else None) for c, v in cov.items()},
                         cov_pooled=float(np.mean([x for v in cov.values() for x in v])) if any(cov.values()) else None,
                         logw=float(np.mean([r["x_p_logw_mean"] for r in recs])),
                         conf_share=float(np.mean([bool(r["x_p_conf"]) for r in recs])),
                         backup=int(g("x_f5_backup").sum()), ms_f5=float(np.mean([r["x_f5_ms_mean"] for r in recs])),
                         ms_p=float(np.mean([r["x_p_ms_mean"] for r in recs]))))
    return rows


def md(rows, rho, setup):
    L = [f"smoke (plumbing only, NOT evidence): {setup}, rho={rho:g}, VAL, seed 0", "",
         "| arm | n | drops | progress | interv | mean ln(hi/lo) | cov t1s | cov t2s | cov end | cov pooled | backup ticks | F5 ms/tick | F5P extra ms/tick |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "n/a" if x is None else f"{x:.2f}"
    for r in rows:
        L.append(f"| {r['arm']} | {r['n']} | {r['drops']} | {r['progress']:.3f} | {r['interv']:.3f} | {r['logw']:.2f} | {f(r['cov']['t1s'])} | "
                 f"{f(r['cov']['t2s'])} | {f(r['cov']['end'])} | {f(r['cov_pooled'])} | {r['backup']} | {r['ms_f5']:.2f} | {r['ms_p']:.3f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--pseed", type=int, default=0)
    ap.add_argument("--setup", default="piper_trk")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-refs", action="store_true", help="skip the F0 / plain-F5 reference runs")
    a = ap.parse_args()
    specs = [(arm, spec_of(arm, a.rho, a.pseed)) for arm in ARMS]
    refs = [] if a.no_refs else [("F0", "F0"), ("F5", "F5")]
    for name, spec in refs + specs:
        R.run(spec, a.setup, 0, "val", a.n, a.workers, a.tag, a.out)
    rows = table(a.out, a.tag, a.setup, "val", specs)
    d = Path(a.out) / a.tag
    (d / "smoke_table.json").write_text(json.dumps(rows, indent=1))
    (d / "smoke_table.md").write_text(md(rows, a.rho, a.setup) + "\n")
    print(md(rows, a.rho, a.setup))


if __name__ == "__main__":
    main()
