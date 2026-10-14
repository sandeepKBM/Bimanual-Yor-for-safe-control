"""sets/vlmprior/calibrate_online.py -- choose the tempering exponent lam (and the node filters' sigma_a) of the online belief on VAL, report coverage.

  step 1 (simulation, per setup; resumable):
      python -m world_model.v2.tray_rep.bridge.run --filter "F5P:arm=calib,sas=0.02x0.025x0.03x0.04x0.05" --setup piper_trk --split val --tag online_cal \
             --out <bimanual_data/tray_rep/vlmprior> --workers 12
      (arm=calib = plain F5 with the population range; logs the raw cumulative log-likelihood of a bank of online estimators at t = 1, 2, 4 s and the end)
  step 2 (no simulation):  python -m world_model.v2.tray_rep.sets.vlmprior.calibrate_online [--out DIR]
      posterior(lam) = softmax(log prior + lam * LL) is evaluated offline for a grid of lam, so one run serves every lam.

Pairs = (carry, checkpoint) with the checkpoint reached (t = 4 s only exists for carries of >= 200 ticks; "end" = the last tick run: a drop or the end of the plan).
Coverage = the true l is inside the central 90% credible interval. Selection on VAL carries 0-299, held-out check on 300-599:
    admissible = coverage >= 0.88 at every checkpoint with >= 100 pairs AND pooled coverage >= 0.89;  chosen = the admissible (sigma_a, lam) with the
    narrowest mean 90% width (most informative belief that is still calibrated). (Minimising |coverage - 0.9| alone is degenerate: lam -> 0, the prior, is
    perfectly calibrated and says nothing.) If nothing is admissible: the largest-coverage point.
Writes online_calib.json (chosen, full tables) and online_calib.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.bridge import run as R
from world_model.v2.tray_rep.sets.vlmprior import online_mu as OM

OUT = K.ROOT / "vlmprior"
LAMS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1.0]
CKS = ["t1s", "t2s", "t4s", "end"]
SEL_N = 300


def load(out, setup, sas_spec):
    _, pj, _ = R.paths(out, "online_cal", "val", setup, f"F5P:arm=calib,sas={sas_spec}", 0)
    return sorted(R.read_jsonl(pj), key=lambda r: r["ep"])


def evaluate(recs, k_sa, lam, level=0.9):
    """-> {ck: dict(n, cov, width_med, width_mean)} for the bank member k_sa at tempering lam."""
    a = 0.5 * (1 - level)
    res = {}
    for ck in CKS:
        cov, wid = [], []
        for r in recs:
            ll = r["x_p_ll"].get(ck)
            if ll is None:
                continue
            w = OM.posterior(np.array(ll[k_sa]), lam)
            lo, hi = OM.cell_quantile(w, a), OM.cell_quantile(w, 1 - a)
            lt = np.log(r["mu"])
            cov.append(lo <= lt <= hi)
            wid.append(hi - lo)
        res[ck] = dict(n=len(cov), cov=float(np.mean(cov)) if cov else None, width_med=float(np.median(wid)) if wid else None,
                       width_mean=float(np.mean(wid)) if wid else None)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sas", default="0.02x0.025x0.03x0.04x0.05")
    a = ap.parse_args()
    out = Path(a.out)
    sas = [float(x) for x in a.sas.split("x")]
    result = dict(grid_lam=LAMS, grid_sigma_a=sas, sel_n=SEL_N, chosen={}, tables={})
    md = ["# Online mu belief: tempering calibration on VAL", "",
          "Method: 40-node bank of Kalman filters on the analytic ball model; posterior ∝ prior * exp(lam * LL). Selection on VAL carries 0-299, held-out check on 300-599.",
          "Coverage = true ln(mu) inside the central 90% credible interval, over (carry, checkpoint) pairs. width = 90% interval width in l (population range = 2.485).", ""]
    for setup in ("piper_trk", "piper"):
        recs = load(out, setup, a.sas)
        if not recs:
            print("no calib records for", setup); continue
        sel = [r for r in recs if r["ep"] < SEL_N]
        held = [r for r in recs if r["ep"] >= SEL_N]
        best, fallback = None, None
        grid = {}
        for k, sa in enumerate(sas):
            for lam in LAMS:
                tab = evaluate(sel, k, lam)
                npair = sum(tab[c]["n"] for c in CKS)
                pooled = sum(tab[c]["cov"] * tab[c]["n"] for c in CKS if tab[c]["n"]) / npair
                wm = float(np.mean([tab[c]["width_mean"] for c in CKS if tab[c]["n"]]))
                ok = all(tab[c]["cov"] >= 0.88 for c in CKS if tab[c]["n"] >= 100) and pooled >= 0.89
                grid[f"{sa:g}/{lam:g}"] = dict(admissible=bool(ok), pooled_cov=float(pooled), width_mean=wm, tab=tab)
                if ok and (best is None or wm < best[0]):
                    best = (wm, 0, k, sa, lam)
                if fallback is None or pooled > fallback[0]:
                    fallback = (pooled, 0, k, sa, lam)
        best = best or fallback
        _, _, kb, sab, lamb = best
        tab_sel = evaluate(sel, kb, lamb)
        tab_held = evaluate(held, kb, lamb) if held else None
        tab_all = evaluate(recs, kb, lamb)
        tab_lam1 = evaluate(recs, kb, 1.0)
        prior_only = evaluate(recs, kb, 0.0)
        result["chosen"][setup] = dict(lam=lamb, sigma_a=sab, n_sel=len(sel), n_held=len(held))
        result["tables"][setup] = dict(selection=tab_sel, heldout=tab_held, all=tab_all, lam1=tab_lam1, lam0=prior_only,
                                       grid={k: dict(admissible=v["admissible"], pooled_cov=v["pooled_cov"], width_mean=v["width_mean"]) for k, v in grid.items()})
        md += [f"## {setup}: chosen sigma_a = {sab:g}, lam = {lamb:g}   (n_sel = {len(sel)}, n_held = {len(held)})", "",
               "| checkpoint | pairs (all) | coverage sel | coverage held-out | coverage all | median 90% width (l) all | mean width all | coverage at lam=1 | width at lam=1 | width at lam=0 (prior) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for c in CKS:
            f = lambda t, k: ("n/a" if t is None or t[c][k] is None else f"{t[c][k]:.3f}")
            md.append(f"| {c} | {tab_all[c]['n']} | {f(tab_sel, 'cov')} | {f(tab_held, 'cov')} | {f(tab_all, 'cov')} | {f(tab_all, 'width_med')} | "
                      f"{f(tab_all, 'width_mean')} | {f(tab_lam1, 'cov')} | {f(tab_lam1, 'width_mean')} | {f(prior_only, 'width_mean')} |")
        md.append("")
        md.append("Other sigma_a at the same lam, coverage (all pairs):")
        for k, sa in enumerate(sas):
            t = evaluate(recs, k, lamb)
            md.append(f"- sigma_a = {sa:g}: " + ", ".join(f"{c} {t[c]['cov']:.2f}/{t[c]['width_med']:.2f}" for c in CKS if t[c]["n"]))
        md.append("")
    (out / "online_calib.json").write_text(json.dumps(result, indent=1))
    (out / "online_calib.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
