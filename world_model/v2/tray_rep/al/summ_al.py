"""al/summ_al.py -- one-table summary of al.run_al outputs (observations only). numpy only. No hardware.
  python -m world_model.v2.tray_rep.al.summ_al --runs <dir> [--regime normal] [--md out.md]
Per (regime, policy): n carries, drops, time (mean cl_t_arrive_s of non-dropped arrived carries; includes probe time), interventions (mean fraction of carry ticks with u != u_nom),
probes: run / certified (SetCertificate passed) / skipped (considered but not run: cert_failed | no_value), probe drops, set coverage (true mu inside the applied set at the
end / at every tick = 1 - ever_miss), mean applied log-width ln(hi/lo), mean carry speed I, empty-intersection events.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(root):
    D = defaultdict(dict)
    for f in sorted(glob.glob(str(Path(root) / "*" / "*" / "AL__*.jsonl"))):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                D[(r["regime"], r["alp_spec"], r.get("pseed", 0))][(r["ep"])] = r
    return D


def row(recs):
    n = len(recs)
    dr = np.array([bool(r["dropped"]) for r in recs])
    ta = np.array([r["cl_t_arrive_s"] if r.get("cl_arrived") else np.nan for r in recs], float)
    ok = ~dr & np.isfinite(ta)
    g = lambda k: np.array([np.nan if r.get(k) is None else float(r[k]) for r in recs], float)
    cons = [r for r in recs if r.get("alp_probe_considered")]
    run = sum(bool(r.get("alp_probe_run")) for r in recs)
    cert = sum(any(r.get("alp_cert_ok", {}).values()) for r in cons) if cons else 0
    skip = collections_count([r.get("alp_probe_skip") for r in cons])
    return dict(n=n, drops=int(dr.sum()), t=float(np.mean(ta[ok])) if ok.any() else float("nan"), interv=float(np.nanmean(g("interv_rate")[~dr])) if (~dr).any() else float("nan"),
                probes_run=run, certified=cert, considered=len(cons), skipped=skip, probe_drops=sum(bool(r.get("alp_probe_drop")) for r in recs),
                cov_end=float(np.nanmean(g("x_p_cov_end"))), cov_always=float(1 - np.mean([bool(r.get("alp_set_ever_miss")) for r in recs])),
                logw=float(np.nanmean(g("x_alp_logw_mean"))), I=float(np.nanmean(g("alp_I_rel"))), n_empty=int(np.nansum(g("alp_n_empty"))),
                probe_s=float(np.nanmean([r.get("probe_ticks", 0) * 0.02 for r in recs])))


def collections_count(xs):
    d = defaultdict(int)
    for x in xs:
        if x:
            d[x] += 1
    return dict(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    D = load(a.runs)
    L = ["| regime | policy | pseed | n | drops | time [s] | interv | probes run / certified / considered | skipped | probe drops | probe s | set cov end | set cov always | log-width | I | empty |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (reg, spec, ps) in sorted(D):
        if a.regime and reg != a.regime:
            continue
        s = row(list(D[(reg, spec, ps)].values()))
        L.append(f"| {reg} | {spec} | {ps} | {s['n']} | {s['drops']} | {s['t']:.2f} | {s['interv']:.3f} | {s['probes_run']} / {s['certified']} / {s['considered']} | {s['skipped'] or ''} | "
                 f"{s['probe_drops']} | {s['probe_s']:.2f} | {s['cov_end']:.2f} | {s['cov_always']:.2f} | {s['logw']:.2f} | {s['I']:.2f} | {s['n_empty']} |")
    print("\n".join(L))
    if a.md:
        Path(a.md).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
