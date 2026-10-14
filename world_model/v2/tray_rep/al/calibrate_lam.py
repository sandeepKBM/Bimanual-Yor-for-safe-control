"""al/calibrate_lam.py -- recalibrate the tempering exponent lam (and sigma_a) of the online mu belief on the CLOSED-LOOP carries of VAL. SIMULATION ONLY. CPU. numpy only.

  step 1 (simulation):  python -m world_model.v2.tray_rep.al.run_al --policy AL_online --regime {normal,stressed} --split val --n-eps 300 --log-ll --workers 5 --out <dir>
         (logs the RAW cumulative online log-likelihood of a bank sigma_a in {0.04, 0.06, 0.08} every 25 carry ticks and at the end: one run serves any lam)
  step 2 (no simulation): python -m world_model.v2.tray_rep.al.calibrate_lam --runs <dir> --out <online_calib_cl.json>
posterior(lam) = softmax(log prior + lam LL). Pairs = (carry, checkpoint with t >= 25). Coverage = true ln mu inside the interval, for (a) the central 90% interval (the quantity
that undercovered, 0.77, on long BOED carries) and (b) the 95% HPD hull used by the AL policies. Also the RUNNING INTERSECTION of the 95% HPD hulls over the checkpoints
(0.99 HPD hulls, what AL_online with cum=1 applies): fraction of carries where the truth was ever outside it.
Selection on VAL carries ep < SEL (default 150) of each regime, held-out check on the rest. admissible = pooled HPD95 coverage >= 0.95, every checkpoint bin with >= 60 pairs
>= 0.93, running-intersection (0.99 hulls) ever-miss <= 0.05; chosen = the admissible (sigma_a, lam) with the narrowest mean HPD95 width. If none: the largest pooled coverage minus ever-miss.
'all' = chosen over both regimes pooled (the default of the policies). Writes the json + a markdown table next to it.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.al import belief_v2 as BV

LAMS = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
SAS = [0.04, 0.06, 0.08]
BINS = [("t25-50", 25, 50), ("t75-150", 75, 150), ("t175-300", 175, 300), ("t325+", 325, 10 ** 6), ("end", None, None)]


def load(root):
    out = {}
    for f in sorted(glob.glob(str(Path(root) / "*" / "*" / "AL__*.jsonl"))):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                if r.get("alp_ll"):
                    out.setdefault(r["regime"], {})[int(r["ep"])] = r
    return out


CUM_LEVEL = 0.99          # level of the hulls that are intersected cumulatively (AL_online cum=1): union-bound style, stricter than the 0.95 of a single interval
MAX_EVER_MISS = 0.05


def evaluate(recs, k, lam):
    """-> dict(bin -> n, cov90c, covh95, w95) plus ever-miss of the running intersection and its final coverage."""
    acc = {b[0]: [] for b in BINS}
    ever, fin, n_c = 0, 0, 0
    for r in recs:
        l = float(np.log(r["mu"]))
        ll = r["alp_ll"]
        ts = sorted(int(t) for t in ll if t.isdigit() and int(t) >= 25)
        cur, miss = BV.L_POP, False
        for t in ts + ["end"]:
            LL = np.array(ll["end" if t == "end" else str(t)][k])
            w = BV.posterior([(LL, lam)])
            c90 = BV.interval(w, 0.90, "central")
            h95 = BV.interval(w, 0.95, "hpd")
            row = (c90[0] <= l <= c90[1], h95[0] <= l <= h95[1], h95[1] - h95[0])
            if t == "end":
                acc["end"].append(row)
            else:
                for name, lo, hi in BINS[:-1]:
                    if lo <= t <= hi:
                        acc[name].append(row)
            hc = BV.interval(w, CUM_LEVEL, "hpd")
            cur = BV.intersect(cur, hc) or hc
            miss |= not (cur[0] - 1e-9 <= l <= cur[1] + 1e-9)
        ever += miss
        fin += (cur[0] - 1e-9 <= l <= cur[1] + 1e-9)
        n_c += 1
    res = {}
    for name, rows in acc.items():
        a = np.array(rows, float) if rows else np.zeros((0, 3))
        res[name] = dict(n=len(a), cov90c=float(a[:, 0].mean()) if len(a) else None, covh95=float(a[:, 1].mean()) if len(a) else None, w95=float(a[:, 2].mean()) if len(a) else None)
    res["_ever_miss"] = ever / max(n_c, 1)
    res["_final_cov"] = fin / max(n_c, 1)
    res["_n"] = n_c
    return res


def pooled(res):
    pairs = [(v["n"], v["covh95"], v["cov90c"], v["w95"]) for k, v in res.items() if not k.startswith("_") and v["n"]]
    n = sum(p[0] for p in pairs)
    return dict(n=n, covh95=sum(p[0] * p[1] for p in pairs) / n, cov90c=sum(p[0] * p[2] for p in pairs) / n, w95=sum(p[0] * p[3] for p in pairs) / n)


def admissible(res):
    p = pooled(res)
    ok = p["covh95"] >= 0.95 and res["_ever_miss"] <= MAX_EVER_MISS and all(v["covh95"] >= 0.93 for k, v in res.items() if not k.startswith("_") and v["n"] >= 60)
    return ok, p


def choose(recs):
    best, fb = None, None
    for k, sa in enumerate(SAS):
        for lam in LAMS:
            res = evaluate(recs, k, lam)
            ok, p = admissible(res)
            if ok and (best is None or p["w95"] < best[0]):
                best = (p["w95"], k, lam)
            if fb is None or p["covh95"] - res["_ever_miss"] > fb[0]:
                fb = (p["covh95"] - res["_ever_miss"], k, lam)
    b = best or fb
    return b[1], b[2], best is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sel", type=int, default=150)
    a = ap.parse_args()
    D = load(a.runs)
    chosen, tables = {}, {}
    md = ["# Online mu belief: closed-loop recalibration on VAL (AL-1)", "",
          "Interval coverage of ln mu at carry-tick checkpoints (every 25 ticks, t>=25, and the end of the carry). c90 = central 90% (as in the old calibration), "
          f"h95 = 95% HPD hull (what the AL policies apply), run-int = running intersection of the h95 hulls (cum=1). Selection ep<{a.sel}, held-out the rest.", ""]
    pool = {}
    for regime, R in D.items():
        recs = [R[e] for e in sorted(R)]
        pool[regime] = recs
        sel, held = [r for r in recs if r["ep"] < a.sel], [r for r in recs if r["ep"] >= a.sel]
        k, lam, adm = choose(sel)
        chosen[regime] = dict(lam=lam, sigma_a=SAS[k], admissible=adm, n_sel=len(sel), n_held=len(held))
        tables[regime] = {}
        md += [f"## {regime}: chosen sigma_a = {SAS[k]:g}, lam = {lam:g} (admissible: {adm}); n_sel={len(sel)}, n_held={len(held)}", "",
               "| config | split | " + " | ".join(b[0] for b in BINS) + " | pooled h95 | pooled c90 | w95 | run-int ever-miss | run-int final cov |", "|---|---|" + "---|" * (len(BINS) + 5)]
        for label, kk, ll_ in (("chosen", k, lam), ("old lam=0.1 sa=0.04", 0, 0.1), ("lam=1 sa=0.04", 0, 1.0)):
            for split, rr in (("sel", sel), ("held", held), ("all", recs)):
                if not rr:
                    continue
                res = evaluate(rr, kk, ll_)
                p = pooled(res)
                tables[regime][f"{label}|{split}"] = dict(res=res, pooled=p)
                cells = [f"{res[b[0]]['covh95']:.2f}/{res[b[0]]['cov90c']:.2f} (n={res[b[0]]['n']})" if res[b[0]]["n"] else "-" for b in BINS]
                md.append(f"| {label} | {split} | " + " | ".join(cells) + f" | {p['covh95']:.3f} | {p['cov90c']:.3f} | {p['w95']:.2f} | {res['_ever_miss']:.3f} | {res['_final_cov']:.3f} |")
        md.append("(bin cells: h95 coverage / c90 coverage)\n")
    allr = [r for rs in pool.values() for r in rs]
    allr_sel = [r for r in allr if r["ep"] < a.sel]
    if allr_sel:
        k, lam, adm = choose(allr_sel)
        chosen["all"] = dict(lam=lam, sigma_a=SAS[k], admissible=adm, n_sel=len(allr_sel))
        res = evaluate([r for r in allr if r["ep"] >= a.sel] or allr, k, lam)
        md += [f"## both regimes pooled: chosen sigma_a = {SAS[k]:g}, lam = {lam:g} (admissible: {adm}); held-out pooled h95 cov {pooled(res)['covh95']:.3f}, c90 {pooled(res)['cov90c']:.3f}, "
               f"ever-miss {res['_ever_miss']:.3f}", ""]
    Path(a.out).write_text(json.dumps(dict(grid_lam=LAMS, grid_sa=SAS, chosen=chosen, tables=tables), indent=1, default=float))
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
