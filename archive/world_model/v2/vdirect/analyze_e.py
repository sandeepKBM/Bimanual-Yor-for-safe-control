"""vdirect/analyze_e.py -- the secondary 6-seed robustness report of PREREG_ADDENDUM_E (sha 4ff274c0). SAVED DATA
ONLY. Written 2026-10-05, after the runs finished, to implement exactly the report list of that addendum; it imports
analyze.py (load, boot, ci) and analyze_f.py (verdicts_f) UNCHANGED. It changes no verdict: K1-K6 and KF1-KF2 stay
decided on seeds 0-2.

Reports, on seed sets 0-2 (pre-registered), 3-5 (independent replication) and 0-5 (6-seed):
  1. the addendum-F contrasts: R - F at 300/1k (KF1), U - F at 1k/3k (KF2), Rr - F at 300/1k/3k, and the
     caution-vs-coverage ladder (N0, M, P, R, U per budget);
  2. agreement of each 6-seed contrast with the 3-seed one in sign and significance;
  3. seeds 3-5 alone.
"""
from __future__ import annotations

import json
from pathlib import Path

from world_model.v2.vdirect import analyze as A
from world_model.v2.vdirect import analyze_f as AF

D = Path("/common/users/ss5772/bimanual_data")
ROOTS = [D / "vdirect_ilab", D / "vdirect_f", D / "vdirect_ext"]
CELLS = {("F", b) for b in (300, 1000, 3000)} | {("R", b) for b in (300, 1000, 3000)} | \
        {("Rr", b) for b in (300, 1000, 3000)} | {("U", b) for b in (1000, 3000)} | {(a, 0) for a in ("N0", "M", "P")}
CONTRASTS = [("R", "F", 300), ("R", "F", 1000), ("U", "F", 1000), ("U", "F", 3000),
             ("Rr", "F", 300), ("Rr", "F", 1000), ("Rr", "F", 3000)]
SETS = {"s0-2": {0, 1, 2}, "s3-5": {3, 4, 5}, "s0-5": {0, 1, 2, 3, 4, 5}}


def subset(runs, seeds):
    out = {}
    for k, bys in runs.items():
        if k in CELLS:
            sel = {s: v for s, v in bys.items() if s in seeds}
            if sel:
                out[k] = sel
    return out


def contrast(B, x, y, b):
    if (x, b) not in B or (y, b) not in B:
        return None
    lo, hi = A.ci(B[(x, b)][1] - B[(y, b)][1])
    pt = B[(x, b)][0] - B[(y, b)][0]
    sig = "F better" if lo > 0 else (f"{x} better" if hi < 0 else "n.s.")
    return dict(diff=pt, ci=(lo, hi), sig=sig, n_seeds=(B[(x, b)][2], B[(y, b)][2]))


def main():
    runs = {}
    for r in ROOTS:
        for k, bys in A.load(r).items():
            for s, v in bys.items():
                runs.setdefault(k, {}).setdefault(s, v)        # first root wins (no seed is in two roots)
    res = {}
    for name, seeds in SETS.items():
        sub = subset(runs, seeds)
        B, ns = A.boot(sub)
        res[name] = dict(B=B, ns=ns, verdicts=AF.verdicts_f(B, ns),
                         contrasts={f"{x}-{y}_b{b}": contrast(B, x, y, b) for x, y, b in CONTRASTS})
    L = ["## 1. Caution-vs-coverage ladder: spill on savable TEST states, mean [95% CI] (n seeds)", "",
         "| arm | budget | " + " | ".join(SETS) + " |", "|---|---|" + "---|" * len(SETS)]
    for k in sorted(CELLS, key=lambda k: (k[1], k[0])):
        cells = []
        for name in SETS:
            B = res[name]["B"]
            if k in B:
                lo, hi = A.ci(B[k][1]); cells.append(f"{100*B[k][0]:.2f}% [{100*lo:.2f}, {100*hi:.2f}] ({B[k][2]})")
            else:
                cells.append("-")
        L.append(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + " |")
    L += ["", "## 2. Contrasts (X - F, pp) and agreement of 6 seeds with the pre-registered 3 seeds", "",
          "| contrast | s0-2 | s3-5 | s0-5 | agree (sign, significance) |", "|---|---|---|---|---|"]
    agree = {}
    for x, y, b in CONTRASTS:
        key = f"{x}-{y}_b{b}"
        c = {n: res[n]["contrasts"][key] for n in SETS}
        fmt = lambda v: "-" if v is None else f"{100*v['diff']:+.2f} [{100*v['ci'][0]:+.2f}, {100*v['ci'][1]:+.2f}] {v['sig']}"
        if c["s0-2"] and c["s0-5"]:
            same_sign = (c["s0-2"]["diff"] > 0) == (c["s0-5"]["diff"] > 0)
            same_sig = c["s0-2"]["sig"] == c["s0-5"]["sig"]
            agree[key] = dict(same_sign=same_sign, same_sig=same_sig)
            ag = ("yes" if same_sign and same_sig else "**NO: " + ("sign" if not same_sign else "") +
                  (" significance" if not same_sig else "") + "**")
        else:
            ag = "n/a"
        L.append(f"| {key} | {fmt(c['s0-2'])} | {fmt(c['s3-5'])} | {fmt(c['s0-5'])} | {ag} |")
    L += ["", "## 3. KF1 / KF2 recomputed per seed set (secondary; the verdict stays on s0-2)", ""]
    for name in SETS:
        v = res[name]["verdicts"]
        L.append(f"- {name}: KF1 {v['KF1']['verdict']}, KF2 {v['KF2']['verdict']}, headline {v['headline']}")
    md = "\n".join(L) + "\n"
    out = D / "vdirect_ext" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables_e.md").write_text(md)
    summ = {n: dict(contrasts=res[n]["contrasts"], verdicts=res[n]["verdicts"],
                    arms={f"{k[0]}_b{k[1]}": dict(mean=res[n]["B"][k][0], ci=A.ci(res[n]["B"][k][1]),
                                                  seeds=res[n]["B"][k][2]) for k in res[n]["B"]}) for n in SETS}
    summ["agreement_s0-2_vs_s0-5"] = agree
    (out / "summary_e.json").write_text(json.dumps(summ, indent=1, default=float))
    print(md, flush=True)


if __name__ == "__main__":
    main()
