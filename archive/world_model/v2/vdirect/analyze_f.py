"""vdirect/analyze_f.py -- confirming-falsifier arms F / F0 / Rr (+ R at 300), criteria KF1-KF2, diagnostics, the
conservatism guard and the outcome-conformal audit (PREREG_ADDENDUM_F). Imports the pre-registered analyze.py and
analyze_cb.py unchanged. SAVED DATA ONLY.

  python -m world_model.v2.vdirect.analyze_f [--grid ILAB_ROOT] [--froot F_ROOT] [--cb-grid DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.vdirect import analyze as A
from world_model.v2.vdirect import analyze_cb as ACB

GRID = Path("/common/users/ss5772/bimanual_data/vdirect_ilab")
FROOT = Path("/common/users/ss5772/bimanual_data/vdirect_f")
CB_GRID = Path("/data/local/ss5772/bimanual_runs/vdirect/cb")
GUARD_PP = 0.005
ORDER = ["F", "F0", "Rr", "R", "V", "U", "Vn", "D", "N0", "M", "P", "CB", "CBr", "CBo"]


def verdicts_f(B, ns):
    ci, floor = A.ci, 0.5 / ns
    v = {}
    kf1 = {}
    for b in (300, 1000):
        if ("R", b) not in B or ("F", b) not in B:
            kf1[b] = dict(status="MISSING"); continue
        pr, sr, _ = B[("R", b)]; pf, sf, _ = B[("F", b)]
        ratio = pr / max(pf, floor)
        d = sr - sf
        ok = ratio >= 2 and ci(d)[0] > 0
        kf1[b] = dict(mean_R=pr, mean_F=pf, ratio=ratio, ratio_ci=ci(sr / np.maximum(sf, floor)),
                      diff_R_minus_F_ci=ci(d), pass_=bool(ok))
    v["KF1"] = dict(per_budget=kf1, verdict="INCOMPLETE" if any(x.get("status") == "MISSING" for x in kf1.values())
                    else ("PASS" if all(x["pass_"] for x in kf1.values()) else "KILL"))
    kf2 = {}
    for b in (1000, 3000):
        if ("U", b) not in B or ("F", b) not in B:
            kf2[b] = dict(status="MISSING"); continue
        lo, hi = ci(B[("U", b)][1] - B[("F", b)][1])
        kf2[b] = dict(mean_U=B[("U", b)][0], mean_F=B[("F", b)][0], ci_U_minus_F=(lo, hi), F_sig_better=bool(lo > 0),
                      U_sig_better=bool(hi < 0))
    miss = any(x.get("status") == "MISSING" for x in kf2.values())
    ok = any(x.get("F_sig_better") for x in kf2.values()) and not any(x.get("U_sig_better") for x in kf2.values())
    v["KF2"] = dict(per_budget=kf2, verdict="INCOMPLETE" if miss else ("PASS" if ok else "KILL"))
    v["headline"] = ("PASS" if v["KF1"]["verdict"] == v["KF2"]["verdict"] == "PASS" else
                     ("INCOMPLETE" if "INCOMPLETE" in (v["KF1"]["verdict"], v["KF2"]["verdict"]) else "KILL"))
    diag = {}
    for X, budgets in (("Rr", (300, 1000, 3000)), ("F0", (300, 1000, 3000)), ("V", (1000, 3000, 10000)),
                       ("D", (1000, 3000, 10000)), ("R", (300, 1000, 3000, 10000)), ("U", (1000, 3000, 10000))):
        for b in budgets:
            if (X, b) in B and ("F", b) in B:
                lo, hi = ci(B[(X, b)][1] - B[("F", b)][1])
                diag[f"{X}_minus_F_b{b}"] = dict(mean_X=B[(X, b)][0], mean_F=B[("F", b)][0], ci=(lo, hi),
                                                 F_sig_better=bool(lo > 0), X_sig_better=bool(hi < 0))
    v["diagnostics"] = diag
    return v


def merge_cons(*roots):
    res, common = {}, None
    for r in roots:
        if not (Path(r) / "cons").exists():
            continue
        rr, cc = ACB.cons_arrays(Path(r))
        for k, bys in rr.items():
            res.setdefault(k, {}).update(bys)
        common = common or cc
    return res, common


def guard(CBs):
    ci = A.ci
    out = {}
    for (arm, b) in sorted(CBs):
        if arm != "F" or ("R", b) not in CBs:
            continue
        d = CBs[("F", b)]["cons"][1] - CBs[("R", b)]["cons"][1]
        pt = CBs[("F", b)]["cons"][0] - CBs[("R", b)]["cons"][0]
        out[b] = dict(cons_F=CBs[("F", b)]["cons"][0], cons_R=CBs[("R", b)]["cons"][0], diff=pt, ci=ci(d),
                      flag_bought_safety_with_caution=bool(pt > GUARD_PP))
    return out


def oc_table(froot, runs):
    """per (arm, b): mean/sd over seeds of the outcome-conformal test metrics + paired-state bootstrap CI of the
    certified-savable fraction (state resample shared with the spill bootstrap is not needed here: own seed 1)."""
    rows = {}
    for f in sorted((froot / "oc").glob("*.json")):
        if not Path(str(f) + "._DONE").exists():
            continue
        d = json.loads(f.read_text())
        arm, b, s = f.stem.replace("smoke_", "").split("_")
        b, s = int(b[1:]), int(s[1:])
        for kind, r in d["kinds"].items():
            name = arm if arm != "B0" else dict(std="N0", M="M", P="P")[kind]
            rows.setdefault((name, b), {})[s] = dict(tau=r["tau_star"], test=r["test"], test0=r["test_at_tau0"],
                                                     cal=r["cal"], refit=bool(r.get("heads")))
    out = {}
    rng = np.random.default_rng(1)
    for key, bys in rows.items():
        seeds = sorted(bys)
        g = lambda k, src="test": [bys[s][src][k] for s in seeds if bys[s][src][k] is not None]
        ms = lambda x: (float(np.mean(x)), float(np.std(x))) if len(x) else (None, None)
        boot = None
        if key in runs and all(s in runs[key] for s in seeds):
            V = [runs[key][s]["per"]["Vw0"] for s in seeds]                # savable states only
            ns = len(V[0])
            J = rng.integers(0, ns, (2000, ns))
            per = np.stack([(Vi[J] >= bys[s]["tau"]).mean(1) for Vi, s in zip(V, seeds)], 1)
            pick = rng.integers(0, len(seeds), per.shape)
            boot = A.ci(np.take_along_axis(per, pick, 1).mean(1))
        out[key] = dict(n_seeds=len(seeds), tau=[bys[s]["tau"] for s in seeds],
                        cert_savable=ms(g("cert_frac_savable")), cert_savable_ci=boot,
                        fail_certified=ms(g("fail_rate_certified")), cert_clearly_safe=ms(g("cert_frac_clearly_safe")),
                        cert_savable_at_tau0=ms(g("cert_frac_savable", "test0")),
                        fail_certified_at_tau0=ms(g("fail_rate_certified", "test0")),
                        cal_fail_given_Vpos=ms([bys[s]["cal"]["fail_rate_given_V_pos"] for s in seeds
                                                if bys[s]["cal"]["fail_rate_given_V_pos"] is not None]),
                        P_heads_refit=any(bys[s]["refit"] for s in seeds))
    return out


def hit_rates(runs):
    out = {}
    for key, bys in runs.items():
        if key[0] not in ("F", "F0"):
            continue
        per = {}
        for s, r in bys.items():
            per[s] = [dict(round=m.get("round"), hit_rate=m.get("hit_rate"), n_fail=m.get("n_fail"),
                           n_probe=m.get("n_probe"), root_kind=m.get("root_kind"), n_roots=m.get("n_roots"))
                      for m in (r["res"].get("rounds") or [])]
        out[f"{key[0]}_b{key[1]}"] = per
    return out


def plot(B, out, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    col = dict(F="#7a3fd1", F0="#b08ce8", Rr="#9c5a2c", R="#eb6834", V="#2a78d6", U="#1baf7a", D="#e87ba4")
    mk = dict(F="P", F0="X", Rr="s", R="s", V="o", U="^", D="v")
    lab = dict(F="F falsify+branch", F0="F0 falsify only", Rr="Rr random, round schedule", R="R random",
               V="V value-directed", U="U ensemble", D="D DAgger")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=160)
    fig.patch.set_facecolor(surf)
    for ax, ymax, title in ((axes[0], 100.0, "full scale"), (axes[1], 12.0, "zoom, 0-12%")):
        ax.set_facecolor(surf)
        for i, arm in enumerate(col):
            bs = sorted(b for (a, b) in B if a == arm and b > 0)
            if not bs:
                continue
            m = np.array([B[(arm, b)][0] for b in bs]); c = np.array([A.ci(B[(arm, b)][1]) for b in bs])
            x = np.array(bs) * (0.94 + 0.02 * i)
            ax.errorbar(x, 100 * m, yerr=[100 * (m - c[:, 0]), 100 * (c[:, 1] - m)], color=col[arm], marker=mk[arm],
                        ms=6, lw=1.8 if arm in ("F", "R") else 1.2, capsize=3, elinewidth=1, label=lab[arm],
                        mec=surf, mew=1.2, zorder=3 if arm == "F" else 2)
        for key, nm, ls in (("N0", "N0 no data", (0, (6, 2))), ("hold", "hold still", "--")):
            val = ref.get(key)
            if val is not None and 100 * val <= ymax:
                ax.axhline(100 * val, color=ink2, lw=1, ls=ls, zorder=1)
                ax.text(1.01, 100 * val, f"{nm} {100*val:.1f}%", transform=ax.get_yaxis_transform(), color=ink2,
                        fontsize=6.5, va="center")
        ax.set_xscale("log"); ax.set_ylim(0, ymax); ax.set_xlim(220, 45000)
        ax.set_xticks([300, 1000, 3000, 10000, 30000]); ax.set_xticklabels(["300", "1k", "3k", "10k", "30k"])
        ax.minorticks_off()
        ax.set_xlabel("true-system branches incl. probes (2 s each)", color=ink2, fontsize=8)
        ax.set_title(title, color=ink2, fontsize=8, loc="left")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        for sp_ in ("left", "bottom"):
            ax.spines[sp_].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("closed-loop spill on physics-savable states (%)", color=ink2, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.suptitle("Confirming falsifier vs random / directed data (mean of 3 seeds, 95% paired bootstrap CI)",
                 color=ink, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(w_pad=6, rect=(0, 0.1, 1, 0.97))
    fig.savefig(out, facecolor=surf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--froot", default=str(FROOT))
    ap.add_argument("--cb-grid", default=str(CB_GRID))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    grid, froot = Path(a.grid), Path(a.froot)
    out = Path(a.out) if a.out else froot / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = A.load(grid, smoke=a.smoke)
    for k, bys in A.load(froot, smoke=a.smoke).items():
        runs.setdefault(k, {}).update(bys)
    if not runs:
        print("no finished runs"); return
    B, ns = A.boot(runs)
    V = verdicts_f(B, ns)
    res, common = merge_cons(a.cb_grid, froot / "cb")
    CBs = ACB.cluster_boot(res, common, "now")[0] if common is not None else {}
    G = guard(CBs)
    OC = oc_table(froot, runs)
    HR = hit_rates(runs)
    st = np.load(grid / "shared" / "cl_states.npz"); rf = np.load(grid / "shared" / "cl_ref.npz")
    sav = st["Vp0"] > 0
    ref = dict(hold=float((rf["hold"][sav] < 0).mean()), phys=float((rf["phys_greedy"][sav] < 0).mean()))
    if ("N0", 0) in B:
        ref["N0"] = B[("N0", 0)][0]
    plot(B, out / "falsifier_sample_efficiency.png", ref)
    ci = A.ci
    pc = lambda t: f"{100*t[0]:.2f}% [{100*ci(t[1])[0]:.2f}, {100*ci(t[1])[1]:.2f}]"
    keys = sorted(set(B) | set(CBs) | set(OC), key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1]))
    L = [f"Spill on {ns} physics-savable TEST states (hold-still {100*ref['hold']:.2f}%, physics {100*ref['phys']:.2f}%); "
         f"conservatism = false interventions on clearly safe TEST states; OC = outcome-conformal certified set "
         f"(eps 5%, beta 5%, LTT fixed sequence on 2000 VAL states).", "",
         "| arm | budget | seeds | spill (savable) [95% CI] | per-seed | conservatism [95% CI] | OC tau* (per seed) | "
         "OC certified savable [CI] | OC fail among certified | OC certified clearly-safe | certified savable at tau=0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda t: "n/a" if t is None or t[0] is None else f"{100*t[0]:.1f}±{100*t[1]:.1f}%"
    for k in keys:
        sp = A.fmt_pct(B[k][0], ci(B[k][1])) if k in B else "n/a"
        per_seed = ", ".join(A.fmt_pct(float(runs[k][s]["spill"].mean())) for s in sorted(runs[k])) if k in runs else ""
        cs = pc(CBs[k]["cons"]) if k in CBs else "n/a"
        o = OC.get(k)
        taus = ", ".join("inf" if t == float("inf") else f"{t:.3f}" for t in o["tau"]) if o else ""
        oci = (f" [{100*o['cert_savable_ci'][0]:.1f}, {100*o['cert_savable_ci'][1]:.1f}]"
               if o and o["cert_savable_ci"] else "")
        L.append(f"| {k[0]} | {k[1]} | {B[k][2] if k in B else ''} | {sp} | {per_seed} | {cs} | {taus} | "
                 f"{f2(o['cert_savable']) + oci if o else 'n/a'} | {f2(o['fail_certified']) if o else 'n/a'} | "
                 f"{f2(o['cert_clearly_safe']) if o else 'n/a'} | {f2(o['cert_savable_at_tau0']) if o else 'n/a'} |")
    L += ["", f"KF1 (F beats R at 300 and 1k): {V['KF1']['verdict']}; KF2 (F beats U): {V['KF2']['verdict']}; "
          f"HEADLINE (KF1 and KF2): {V['headline']}",
          f"Conservatism guard (flag if cons_F - cons_R > {100*GUARD_PP:.1f} pp): " +
          json.dumps({b: dict(diff_pp=round(100 * g["diff"], 3), flag=g["flag_bought_safety_with_caution"])
                      for b, g in G.items()})]
    (out / "tables_f.md").write_text("\n".join(L) + "\n")
    summ = dict(n_savable_states=ns, ref=ref, verdicts=V, guard=G, hit_rates=HR,
                oc={f"{k[0]}_b{k[1]}": v for k, v in OC.items()},
                arms={f"{k[0]}_b{k[1]}": dict(mean=B[k][0], ci=ci(B[k][1]), n_seeds=B[k][2],
                                              per_seed={int(s): float(runs[k][s]["spill"].mean()) for s in runs[k]})
                      for k in B})
    (out / "summary_f.json").write_text(json.dumps(summ, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print("\n".join(L))
    print(json.dumps(V, indent=1, default=str))


if __name__ == "__main__":
    main()
