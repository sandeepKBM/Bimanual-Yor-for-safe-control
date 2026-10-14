"""vdirect/analyze_g.py -- verdict for PREREG_ADDENDUM_G (F vs Rc on seeds 0-8). SAVED DATA ONLY.
Imports the pre-registered analyze.py (load, boot, ci) UNCHANGED.

KG1: over seeds 0-8, F significantly better than Rc (95% CI of Rc - F > 0) at >= 1 of {300, 1000, 3000} AND Rc
     significantly better at none -> PASS, else KILL.
Diagnostics: the same contrast per seed set (0-2, 3-5, 6-8); R vs Rc where both exist.
"""
from __future__ import annotations

import json
from pathlib import Path

from world_model.v2.vdirect import analyze as A

D = Path("/common/users/ss5772/bimanual_data")
ROOTS = [D / "vdirect_f", D / "vdirect_ext", D / "vdirect_g", D / "vdirect_rb", D / "vdirect_ilab"]
BUDGETS = (300, 1000, 3000)
SETS = {"s0-8": set(range(9)), "s0-2": {0, 1, 2}, "s3-5": {3, 4, 5}, "s6-8": {6, 7, 8}}


def main():
    runs = {}
    for r in ROOTS:
        for k, bys in A.load(r).items():
            if k[0] in ("F", "Rc", "R") and k[1] in BUDGETS:
                for s, v in bys.items():
                    runs.setdefault(k, {}).setdefault(s, v)
    out, L = {}, ["| seed set | budget | F mean (n) | Rc mean (n) | Rc - F [95% CI] | verdict |", "|---|---|---|---|---|---|"]
    for name, seeds in SETS.items():
        sub = {k: {s: v for s, v in bys.items() if s in seeds} for k, bys in runs.items()}
        sub = {k: v for k, v in sub.items() if v}
        if not sub:
            continue
        B, ns = A.boot(sub)
        out[name] = {}
        for b in BUDGETS:
            if ("F", b) in B and ("Rc", b) in B:
                lo, hi = A.ci(B[("Rc", b)][1] - B[("F", b)][1])
                v = "F better" if lo > 0 else ("Rc better" if hi < 0 else "n.s.")
                out[name][b] = dict(F=B[("F", b)][0], Rc=B[("Rc", b)][0], nF=B[("F", b)][2], nRc=B[("Rc", b)][2],
                                    ci=(lo, hi), verdict=v)
                L.append(f"| {name} | {b} | {100*B[('F', b)][0]:.2f}% ({B[('F', b)][2]}) | {100*B[('Rc', b)][0]:.2f}% "
                         f"({B[('Rc', b)][2]}) | {100*(B[('Rc', b)][0]-B[('F', b)][0]):+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {v} |")
            if ("R", b) in B and ("Rc", b) in B:
                lo, hi = A.ci(B[("R", b)][1] - B[("Rc", b)][1])
                out[name][f"R_minus_Rc_{b}"] = dict(R=B[("R", b)][0], Rc=B[("Rc", b)][0], ci=(lo, hi),
                                                    nR=B[("R", b)][2], nRc=B[("Rc", b)][2])
    p = out["s0-8"]
    f_better = any(p[b]["verdict"] == "F better" for b in BUDGETS if b in p)
    rc_better = any(p[b]["verdict"] == "Rc better" for b in BUDGETS if b in p)
    complete = all(b in p and p[b]["nF"] == 9 and p[b]["nRc"] == 9 for b in BUDGETS)
    kg1 = "INCOMPLETE" if not complete else ("PASS" if f_better and not rc_better else "KILL")
    L += ["", f"**KG1 (seeds 0-8): {kg1}** (F significantly better somewhere: {f_better}; Rc significantly better "
              f"somewhere: {rc_better}; complete 9 x 9 seeds at every budget: {complete})", "",
          "R vs Rc where both exist (R - Rc, pp):"]
    for name in SETS:
        for b in BUDGETS:
            x = out.get(name, {}).get(f"R_minus_Rc_{b}")
            if x:
                L.append(f"- {name} b{b}: R {100*x['R']:.2f}% ({x['nR']}) vs Rc {100*x['Rc']:.2f}% ({x['nRc']}): "
                         f"[{100*x['ci'][0]:+.2f}, {100*x['ci'][1]:+.2f}]")
    md = "\n".join(L) + "\n"
    o = D / "vdirect_g" / "analysis"
    o.mkdir(parents=True, exist_ok=True)
    (o / "tables_g.md").write_text(md)
    (o / "summary_g.json").write_text(json.dumps(dict(KG1=kg1, contrasts=out), indent=1, default=float))
    print(md, flush=True)


if __name__ == "__main__":
    main()
