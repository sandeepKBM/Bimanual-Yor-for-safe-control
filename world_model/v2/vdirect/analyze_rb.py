"""vdirect/analyze_rb.py -- verdicts for PREREG_ADDENDUM_RB. SAVED DATA ONLY. Imports the pre-registered
analyze.py (load, boot, ci) and analyze_cb.py (cons_arrays, cluster_boot) UNCHANGED.

KB1 (headline, the KF1 bar vs R): at b in {300, 1000}: mean_R / mean_Rb >= 2 (floor 0.5/n) AND CI(R - Rb) > 0;
     PASS only if both budgets pass.
KB2 (does WHERE matter; the clean contrast): Rb significantly better than Rc at >= 1 of {300, 1k, 3k}, and Rc
     significantly better at none.
Diagnostics: CI(Rc - R) and CI(Rb - R) at every budget. Guard: FLAG if cons_Rb - cons_R > 0.5 pp at any budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.vdirect import analyze as A
from world_model.v2.vdirect import analyze_cb as ACB

ROOT = Path("/common/users/ss5772/bimanual_data/vdirect_rb")
BUDGETS = (300, 1000, 3000)
GUARD_PP = 0.005


def verdicts(B, ns):
    ci, floor = A.ci, 0.5 / ns
    v, kb1 = {}, {}
    for b in (300, 1000):
        if ("R", b) not in B or ("Rb", b) not in B:
            kb1[b] = dict(status="MISSING"); continue
        pr, sr, _ = B[("R", b)]; pb, sb, _ = B[("Rb", b)]
        ratio = pr / max(pb, floor)
        kb1[b] = dict(mean_R=pr, mean_Rb=pb, ratio=ratio, ratio_ci=ci(sr / np.maximum(sb, floor)),
                      diff_R_minus_Rb_ci=ci(sr - sb), pass_=bool(ratio >= 2 and ci(sr - sb)[0] > 0))
    miss = any(x.get("status") == "MISSING" for x in kb1.values())
    v["KB1"] = dict(per_budget=kb1, verdict="INCOMPLETE" if miss else
                    ("PASS" if all(x["pass_"] for x in kb1.values()) else "KILL"))
    kb2 = {}
    for b in BUDGETS:
        if ("Rc", b) not in B or ("Rb", b) not in B:
            kb2[b] = dict(status="MISSING"); continue
        lo, hi = ci(B[("Rc", b)][1] - B[("Rb", b)][1])
        kb2[b] = dict(mean_Rc=B[("Rc", b)][0], mean_Rb=B[("Rb", b)][0], ci_Rc_minus_Rb=(lo, hi),
                      Rb_sig_better=bool(lo > 0), Rc_sig_better=bool(hi < 0))
    miss = any(x.get("status") == "MISSING" for x in kb2.values())
    ok = any(x.get("Rb_sig_better") for x in kb2.values()) and not any(x.get("Rc_sig_better") for x in kb2.values())
    v["KB2"] = dict(per_budget=kb2, verdict="INCOMPLETE" if miss else ("PASS" if ok else "KILL"))
    diag = {}
    for X in ("Rc", "Rb"):
        for b in BUDGETS:
            if (X, b) in B and ("R", b) in B:
                lo, hi = ci(B[("R", b)][1] - B[(X, b)][1])
                diag[f"R_minus_{X}_b{b}"] = dict(mean_R=B[("R", b)][0], mean_X=B[(X, b)][0], ci=(lo, hi),
                                                 X_sig_better=bool(lo > 0), R_sig_better=bool(hi < 0))
    v["diagnostics"] = diag
    return v


def guard(CBs):
    out = {}
    for b in BUDGETS:
        for ref in ("R", "Rc"):
            if ("Rb", b) in CBs and (ref, b) in CBs:
                pt = CBs[("Rb", b)]["cons"][0] - CBs[(ref, b)]["cons"][0]
                d = CBs[("Rb", b)]["cons"][1] - CBs[(ref, b)]["cons"][1]
                out[f"Rb_vs_{ref}_b{b}"] = dict(cons_Rb=CBs[("Rb", b)]["cons"][0], cons_ref=CBs[(ref, b)]["cons"][0],
                                                diff=pt, ci=A.ci(d), flag=bool(ref == "R" and pt > GUARD_PP))
    return out


def table(runs, B, CBs):
    L = ["| arm | budget | seeds | spill (savable) [95% CI] | per-seed | conservatism [95% CI] | root V mean | pool |",
         "|---|---|---|---|---|---|---|---|"]
    for (arm, b) in sorted(B, key=lambda k: (k[1], ["R", "Rc", "Rb"].index(k[0]) if k[0] in ("R", "Rc", "Rb") else 9)):
        pt, s, n = B[(arm, b)]
        lo, hi = A.ci(s)
        bys = runs[(arm, b)]
        per = ", ".join(f"{100*bys[k]['spill'].mean():.2f}%" for k in sorted(bys))
        cons = CBs.get((arm, b), {}).get("cons")
        cs = f"{100*cons[0]:.2f}% [{100*A.ci(cons[1])[0]:.2f}, {100*A.ci(cons[1])[1]:.2f}]" if cons else "n/a"
        r0 = [bys[k]["res"]["rounds"][0] for k in sorted(bys)]
        rv = "/".join(f"{m['root_V_mean']:.3f}" for m in r0) if "root_V_mean" in r0[0] else "-"
        pool = "/".join(str(m.get("pool_size", "-")) for m in r0)
        L.append(f"| {arm} | {b} | {n} | {100*pt:.2f}% [{100*lo:.2f}, {100*hi:.2f}] | {per} | {cs} | {rv} | {pool} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = {k: v for k, v in A.load(root, smoke=a.smoke).items() if k[0] in ("R", "Rc", "Rb")}
    if not runs:
        print("no finished runs"); return
    B, ns = A.boot(runs)
    V = verdicts(B, ns)
    CBs = {}
    if (root / "cb" / "cons").exists():
        res, common = ACB.cons_arrays(root / "cb")
        if common is not None:
            CBs = ACB.cluster_boot(res, common, "now")[0]
    G = guard(CBs)
    md = table(runs, B, CBs)
    md += (f"\n\nKB1 (Rb beats R by 2x at 300 and 1k): {V['KB1']['verdict']}; KB2 (Rb beats Rc somewhere, never worse): "
           f"{V['KB2']['verdict']}\nGuard (flag if cons_Rb - cons_R > 0.5 pp): {json.dumps(G)}\n")
    (out / "tables_rb.md").write_text(md)
    summ = dict(n_savable_states=ns, verdicts=V, guard=G,
                arms={f"{k[0]}_b{k[1]}": dict(mean=B[k][0], ci=A.ci(B[k][1]), seeds=B[k][2]) for k in B})
    (out / "summary_rb.json").write_text(json.dumps(summ, indent=1))
    print(md, flush=True)


if __name__ == "__main__":
    main()
