"""boed/p7_analyze.py -- verdict of Experiment C (bimanual_data/tray_rep/boed_p7/PREREG_C_FROZEN.md). SAVED DATA ONLY. CPU.

P7 (probe at pickup, worst-case speed rule at the posterior's 2.5% quantile, set-based probe certificate) and P7s (same rule, the
original shadow certificate) against the FROZEN BOED arms P6 / P5 / P3 (bimanual_data/tray_rep/boed/main/test, read only).
Statistics = boed/report.boot unchanged (paired over carries, value seeds kept together, 2000 draws, seed 20261005).

  PRIMARY (piper_trk, P7 vs P6):
    PASS          iff upper 95% bound of drop(P7) - drop(P6) <= +2 pp AND upper 95% bound of the time ratio P7/P6 <= 0.95 AND no probe
                  drops the ball (probe guard);
    KILL          if the lower 95% bound of drop(P7) - drop(P6) > +2 pp, or the point time ratio > 0.95 (saves under 5% time);
    INCONCLUSIVE  otherwise.
  Secondary: P7s vs P6 (rule change alone), P7 vs P7s (certificate change), P7 vs P5 (oracle), P7 vs P3; noise-free setup; certificate pass
  rate, chosen intensities, fallback share, mu_q, drops per mu bin, oracle-gain capture = (t_P6 - t_P7) / (t_P6 - t_P5).

  python -m world_model.v2.tray_rep.boed.p7_analyze
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.boed import report as RP

FROZEN = Path("/common/users/ss5772/bimanual_data/tray_rep/boed")
OUT = Path("/common/users/ss5772/bimanual_data/tray_rep/boed_p7")
MU_EDGES = np.exp(np.linspace(np.log(5e-4), np.log(6e-3), 6))


def main():
    fz, p7 = RP.load(FROZEN, "main", "test"), RP.load(OUT, "main", "test")
    res, L = {}, []
    for setup in ("piper_trk", "piper"):
        d = {k: v for k, v in fz[setup].items() if k in ("P3", "P5", "P6")}
        d.update({k: v for k, v in p7[setup].items() if k in ("P7", "P7s")})
        n = {k: sum(len(x) for x in v.values()) for k, v in d.items()}
        out = dict(n=n)
        L += [f"## {setup} ({'PRIMARY' if setup == 'piper_trk' else 'secondary'}); records per arm {n}", "",
              "| A vs B | drop A | drop B | Delta (95% CI) | t A / t B (s, both kept) | ratio (95% CI) | paired kept |", "|---|---|---|---|---|---|---|"]
        cons = {}
        for a, b in (("P7", "P6"), ("P7s", "P6"), ("P7", "P7s"), ("P7", "P5"), ("P7", "P3"), ("P5", "P6")):
            if a in d and b in d:
                r = RP.boot(d, a, b)
                cons[f"{a}-{b}"] = r
                L.append(f"| {a} vs {b} | {100*r['drop_a']:.1f}% | {100*r['drop_b']:.1f}% | {100*r['delta']:+.2f} pp "
                         f"[{100*r['delta_ci'][0]:+.2f}, {100*r['delta_ci'][1]:+.2f}] | {r['t_a']:.2f} / {r['t_b']:.2f} | {r['ratio']:.3f} "
                         f"[{r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}] | {r['n_paired_ok']} |")
        out["contrasts"] = cons
        for arm in ("P7", "P7s"):
            if arm not in d:
                continue
            recs = [r for v in d[arm].values() for r in v.values()]
            I = Counter(round(float(r["I_chosen"]), 2) for r in recs if r.get("I_chosen") is not None)
            rule = Counter(r.get("rule") for r in recs)
            mq = np.array([r["mu_q"] for r in recs if r.get("mu_q") is not None])
            mu = np.array([r["mu"] for r in recs])
            dr = np.array([r["dropped"] for r in recs], float)
            idx = np.clip(np.digitize(mu, MU_EDGES[1:-1]), 0, 4)
            bins = [(float(dr[idx == k].mean()), int((idx == k).sum())) for k in range(5)]
            st = dict(cert_pass=float(np.mean([bool(r.get("cert_pass")) for r in recs])), probe_run=int(sum(bool(r.get("probe_run")) for r in recs)),
                      probe_drop=int(sum(bool(r.get("probe_drop")) for r in recs)), I_chosen=dict(I), rule=dict(rule),
                      mu_q_pct=[float(x) for x in np.percentile(mq, [5, 50, 95])] if len(mq) else None, mu_bins=bins)
            out[arm] = st
            L += ["", f"{arm}: certificate pass {100*st['cert_pass']:.1f}%, probes run {st['probe_run']} / dropped {st['probe_drop']}, "
                      f"rule {dict(rule)}, chosen I {dict(I)}, mu_q 5/50/95% {st['mu_q_pct']}",
                  f"{arm} drops per mu bin: " + ", ".join(f"{100*a:.1f}% ({b})" for a, b in bins)]
        v = "INCOMPLETE"
        r = cons.get("P7-P6")
        if r is not None and "P7" in out:
            guard = out["P7"]["probe_drop"] == 0
            if r["delta_ci"][0] > 0.02 or r["ratio"] > 0.95:
                v = "KILL"
            elif r["delta_ci"][1] <= 0.02 and r["ratio_ci"][1] <= 0.95 and guard:
                v = "PASS"
            else:
                v = "INCONCLUSIVE"
            if all(k in cons for k in ("P7-P6", "P5-P6")):
                t6, t7, t5 = cons["P7-P6"]["t_b"], cons["P7-P6"]["t_a"], cons["P5-P6"]["t_a"]
                out["oracle_gain_capture"] = (t6 - t7) / (t6 - t5) if t6 != t5 else float("nan")
        out["verdict"] = v
        L += ["", f"**{'PRIMARY' if setup == 'piper_trk' else 'Secondary'} verdict ({setup}, P7 vs P6): {v}**; "
                  f"oracle-gain capture {out.get('oracle_gain_capture')}", ""]
        res[setup] = out
    md = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "tables_C.md").write_text(md)
    (OUT / "analysis" / "summary_C.json").write_text(json.dumps(res, indent=1, default=float))
    print(md, flush=True)


if __name__ == "__main__":
    main()
