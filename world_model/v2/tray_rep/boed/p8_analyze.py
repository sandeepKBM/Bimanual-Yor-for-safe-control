"""boed/p8_analyze.py -- verdict of the Experiment C addendum, arm P8 (bimanual_data/tray_rep/boed_p8/PREREG_CP8_FROZEN.md). SAVED DATA ONLY.

P8 (no probe; online belief from the carry's own motion; worst-case 2.5%-quantile re-pick at transport boundaries; F3 on) against the
FROZEN P6 / P5 (boed/main/test) and P7 (boed_p7/main/test). Statistics = boed/report.boot unchanged. Same decision rule as C's primary:
  PASS iff upper 95% bound of drop(P8) - drop(P6) <= +2 pp AND upper bound of the time ratio P8/P6 <= 0.95;
  KILL if the lower bound of the drop difference > +2 pp or the point time ratio > 0.95; INCONCLUSIVE otherwise.
  python -m world_model.v2.tray_rep.boed.p8_analyze
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.boed import report as RP

FROZEN = Path("/common/users/ss5772/bimanual_data/tray_rep/boed")
P7DIR = Path("/common/users/ss5772/bimanual_data/tray_rep/boed_p7")
OUT = Path("/common/users/ss5772/bimanual_data/tray_rep/boed_p8")


def main():
    fz, p7, p8 = RP.load(FROZEN, "main", "test"), RP.load(P7DIR, "main", "test"), RP.load(OUT, "main", "test")
    res, L = {}, []
    for setup in ("piper_trk", "piper"):
        d = {k: v for k, v in fz[setup].items() if k in ("P5", "P6")}
        d.update({k: v for k, v in p7[setup].items() if k == "P7"})
        d.update({k: v for k, v in p8[setup].items() if k == "P8"})
        n = {k: sum(len(x) for x in v.values()) for k, v in d.items()}
        L += [f"## {setup} ({'PRIMARY' if setup == 'piper_trk' else 'secondary'}); records {n}", "",
              "| A vs B | drop A | drop B | Delta (95% CI) | t A / t B | ratio (95% CI) | paired kept |", "|---|---|---|---|---|---|---|"]
        cons = {}
        for a, b in (("P8", "P6"), ("P8", "P7"), ("P8", "P5")):
            if a in d and b in d:
                r = RP.boot(d, a, b)
                cons[f"{a}-{b}"] = r
                L.append(f"| {a} vs {b} | {100*r['drop_a']:.1f}% | {100*r['drop_b']:.1f}% | {100*r['delta']:+.2f} pp [{100*r['delta_ci'][0]:+.2f}, "
                         f"{100*r['delta_ci'][1]:+.2f}] | {r['t_a']:.2f} / {r['t_b']:.2f} | {r['ratio']:.3f} [{r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}] | {r['n_paired_ok']} |")
        out = dict(n=n, contrasts=cons)
        if "P8" in d:
            recs = [r for v in d["P8"].values() for r in v.values()]
            ih = Counter(tuple(round(x, 2) for x in (r.get("I_hist") or [])) for r in recs)
            cov = [r.get("bel_cover90") for r in recs if r.get("bel_cover90") is not None]
            out["I_hist_top"] = [(list(k), v) for k, v in ih.most_common(6)]
            out["bel_cover90"] = float(np.mean(cov)) if cov else None
            sped = [r for r in recs if any(x > 0.41 for x in (r.get("I_hist") or []))]
            out["sped_up"] = dict(n=len(sped), drop=float(np.mean([r["dropped"] for r in sped])) if sped else None)
            L += ["", f"P8 intensity histories (top): {out['I_hist_top']}; belief 90% coverage {out['bel_cover90']}; "
                      f"carries that sped up: {out['sped_up']}"]
        v = "INCOMPLETE"
        r = cons.get("P8-P6")
        if r is not None:
            if r["delta_ci"][0] > 0.02 or r["ratio"] > 0.95:
                v = "KILL"
            elif r["delta_ci"][1] <= 0.02 and r["ratio_ci"][1] <= 0.95:
                v = "PASS"
            else:
                v = "INCONCLUSIVE"
        out["verdict"] = v
        L += ["", f"**{'PRIMARY' if setup == 'piper_trk' else 'Secondary'} verdict ({setup}, P8 vs P6): {v}**", ""]
        res[setup] = out
    md = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "tables_CP8.md").write_text(md)
    (OUT / "analysis" / "summary_CP8.json").write_text(json.dumps(res, indent=1, default=float))
    print(md, flush=True)


if __name__ == "__main__":
    main()
