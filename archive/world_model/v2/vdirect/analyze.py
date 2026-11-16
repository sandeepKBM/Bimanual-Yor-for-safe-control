"""vdirect/analyze.py -- aggregate the kill-test grid, hierarchical paired bootstrap, verdicts, plot.
SAVED DATA ONLY.  python -m world_model.v2.vdirect.analyze [--root R] [--out DIR]

Primary metric: closed-loop spill rate on physics-savable held-out states (wm_greedy, cl_eval == closed_loop.py).
Bootstrap (pre-registered): 10,000 reps; each rep resamples the savable STATES with replacement (one draw shared by
every run -> paired across arms, since every run is evaluated on the same states) and, per arm x budget, resamples
its SEEDS with replacement; the arm statistic is the mean over drawn seeds of the run's spill rate on the drawn states.
Percentile 95% CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/local/ss5772/bimanual_runs/vdirect")
ARMS = ["V", "R", "U", "Vn", "D"]
NAMES = dict(V="V value-directed (ours)", R="R random", U="U ensemble-disagreement", Vn="V-noiter one-shot",
             D="D DAgger on-policy", N0="N0 nominal, full max", M="M in-data max", P="P uncertainty-pessimistic")
COLORS = dict(V="#2a78d6", R="#eb6834", U="#1baf7a", Vn="#eda100", D="#e87ba4")
MARKERS = dict(V="o", R="s", U="^", Vn="D", D="v")
NB = 10000


def load(root, smoke=False):
    """-> dict[(arm, budget)] -> dict[seed] -> dict(result, spill_savable_vector)"""
    runs = {}
    for d in sorted((root / "runs").glob("*_b*_s*")):
        if d.name.startswith("smoke") != smoke or not (d / "_DONE").exists():
            continue
        arm, b, s = d.name.replace("smoke_", "").split("_")
        b, s = int(b[1:]), int(s[1:])
        res = json.loads((d / "result.json").read_text())
        kinds = ["std"] if arm != "B0" else ["std", "M", "P"]
        for k in kinds:
            name = arm if arm != "B0" else dict(std="N0", M="M", P="P")[k]
            per = np.load(d / f"value_{k}" / "cl_states.npz")
            sav = per["Vp0"] > 0
            runs.setdefault((name, b), {})[s] = dict(res=res, val=res["values"][k], spill=per["spill"][sav].astype(float),
                                                    per={kk: per[kk][sav] for kk in per.files})
    return runs


def boot(runs, nb=NB, seed=0):
    """-> dict[(arm,b)] -> (point, samples (nb,))"""
    rng = np.random.default_rng(seed)
    n_states = {len(next(iter(v.values()))["spill"]) for v in runs.values()}
    assert len(n_states) == 1, n_states
    ns = n_states.pop()
    J = rng.integers(0, ns, (nb, ns))
    out = {}
    for key, bys in runs.items():
        seeds = sorted(bys)
        X = np.stack([bys[s]["spill"] for s in seeds])           # (S, ns)
        per_run = np.stack([X[i][J].mean(1) for i in range(len(seeds))], 1)   # (nb, S) paired state resample
        pick = rng.integers(0, len(seeds), (nb, len(seeds)))
        samp = np.take_along_axis(per_run, pick, 1).mean(1)
        out[key] = (float(X.mean()), samp, len(seeds))
    return out, ns


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def verdicts(B, ns, runs):
    v = {}
    floor = 0.5 / ns
    # K1: at 1k and 3k, mean_R / mean_V >= 2 AND 95% CI of (R - V) excludes 0 (> 0)
    k1 = {}
    for b in (1000, 3000):
        if ("R", b) not in B or ("V", b) not in B:
            k1[b] = dict(status="MISSING"); continue
        pr, sr, _ = B[("R", b)]; pv, sv, _ = B[("V", b)]
        ratio = pr / max(pv, floor); d = sr - sv
        rs = sr / np.maximum(sv, floor)
        ok = (ratio >= 2) and (ci(d)[0] > 0)
        k1[b] = dict(mean_R=pr, mean_V=pv, ratio=ratio, ratio_ci=ci(rs), diff_R_minus_V_ci=ci(d), pass_=bool(ok),
                     strong_pass=bool(ci(rs)[0] >= 2))
    v["K1"] = dict(per_budget=k1, verdict="PASS" if all(x.get("pass_") for x in k1.values()) else
                   ("INCOMPLETE" if any(x.get("status") == "MISSING" for x in k1.values()) else "KILL"))
    # K2: B_R* = smallest R budget with mean <= 2% (else 60k); V needs mean <= 2% at some budget <= B_R*/5
    rb = sorted(b for (a, b) in B if a == "R")
    bstar = next((b for b in rb if B[("R", b)][0] <= 0.02), 60000)
    vb = sorted(b for (a, b) in B if a == "V")
    hits = [b for b in vb if b <= bstar / 5 and B[("V", b)][0] <= 0.02]
    v["K2"] = dict(B_R_star=bstar, B_R_star_source="grid" if bstar in rb else "audit full-length 60k runs (1.16-1.56%)",
                   V_budgets_le_limit=[b for b in vb if b <= bstar / 5],
                   V_means={b: B[("V", b)][0] for b in vb}, V_hits=hits,
                   verdict="PASS" if hits else ("KILL" if vb else "INCOMPLETE"))
    # K3 (and K4 for D): at 1k/3k, 95% CI of (X - V); KILL if V is never significantly better, or X significantly better
    for kname, X in (("K3", "U"), ("K4", "D")):
        per = {}
        for b in (1000, 3000):
            if (X, b) not in B or ("V", b) not in B:
                per[b] = dict(status="MISSING"); continue
            d = B[(X, b)][1] - B[("V", b)][1]
            lo, hi = ci(d)
            per[b] = dict(mean_X=B[(X, b)][0], mean_V=B[("V", b)][0], diff_X_minus_V=B[(X, b)][0] - B[("V", b)][0],
                          ci=(lo, hi), V_sig_better=bool(lo > 0), X_sig_better=bool(hi < 0))
        miss = any(x.get("status") == "MISSING" for x in per.values())
        ok = (any(x.get("V_sig_better") for x in per.values()) and not any(x.get("X_sig_better") for x in per.values()))
        v[kname] = dict(arm=X, per_budget=per, verdict="INCOMPLETE" if miss else ("PASS" if ok else "KILL"))
    # K5 premise: V at 3k vs the better of the no-data fixes M / P (0 branches)
    best = None
    for X in ("M", "P"):
        if (X, 0) in B and (best is None or B[(X, 0)][0] < B[best][0]):
            best = (X, 0)
    if best is not None and ("V", 3000) in B:
        d = B[best][1] - B[("V", 3000)][1]
        v["K5"] = dict(best_nodata=best[0], mean_best=B[best][0], mean_V3k=B[("V", 3000)][0], ci=ci(d),
                       verdict="PASS" if ci(d)[0] > 0 else "KILL")
    else:
        v["K5"] = dict(verdict="INCOMPLETE")
    # loop contribution (secondary): Vn - V at each budget
    v["loop"] = {b: dict(mean_Vn=B[("Vn", b)][0], mean_V=B[("V", b)][0], ci_Vn_minus_V=ci(B[("Vn", b)][1] - B[("V", b)][1]))
                 for b in (1000, 3000, 10000) if ("Vn", b) in B and ("V", b) in B}
    return v


def secondary_table(runs):
    rows = {}
    for key, bys in runs.items():
        vals = [r["val"] for r in bys.values()]
        wc = [r["val"]["cl"]["winners_curse_savable"] for r in bys.values()]
        g = lambda k: (float(np.mean([x[k] for x in vals])), float(np.std([x[k] for x in vals])))
        h = lambda k: (float(np.mean([x[k] for x in wc])), float(np.std([x[k] for x in wc])))
        rows[key] = dict(n_seeds=len(vals), recall5=g("doomed_recall_at_5pct_fu"), false_safe=g("false_safe"),
                         iou=g("iou"), gradcos=g("grad_cos_K10_median"),
                         spearman=(float(np.mean([r["res"]["spearman_H25"] for r in bys.values()])),
                                   float(np.std([r["res"]["spearman_H25"] for r in bys.values()]))),
                         wc_gap=h("gap_argmax_mean"), wc_gap_logged=h("gap_logged_mean"), regret=h("regret_true_mean"),
                         err_mar_log=h("err_mar_log_mean"), err_mar_arg=h("err_mar_arg_mean"),
                         err_mar10_log=h("err_mar10_log_mean"), err_mar10_arg=h("err_mar10_arg_mean"),
                         err_w_log=h("err_angvel_log_mean"), err_w_arg=h("err_angvel_arg_mean"))
    return rows


def fmt_pct(p, c=None):
    return f"{100*p:.2f}%" + (f" [{100*c[0]:.2f}, {100*c[1]:.2f}]" if c else "")


def plot(B, out, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=160)
    fig.patch.set_facecolor(surf)
    offs = dict(V=0.96, R=1.04, U=0.98, Vn=1.02, D=1.06)
    for ax, ymax, title in ((axes[0], 100.0, "full scale"), (axes[1], 12.0, "zoom, 0-12%")):
        ax.set_facecolor(surf)
        for arm in ARMS:
            bs = sorted(b for (a, b) in B if a == arm and b > 0)
            if not bs:
                continue
            m = np.array([B[(arm, b)][0] for b in bs]); c = np.array([ci(B[(arm, b)][1]) for b in bs])
            x = np.array(bs) * offs[arm]
            ax.errorbar(x, 100 * m, yerr=[100 * (m - c[:, 0]), 100 * (c[:, 1] - m)], color=COLORS[arm],
                        marker=MARKERS[arm], ms=7, lw=2, capsize=3, elinewidth=1.2, label=NAMES[arm],
                        mec=surf, mew=1.5, zorder=3 if arm == "V" else 2)
            if ymax < 100:
                for xi, mi in zip(x, m):
                    if 100 * mi > ymax:
                        ax.annotate(f"{100*mi:.0f}%", (xi, ymax), xytext=(0, -2), textcoords="offset points",
                                    ha="center", va="top", fontsize=6.5, color=ink2)
        for key, lab, ls in (("N0", "N0 no branches, full max", (0, (6, 2))), ("M", "M in-data max", "-."),
                             ("P", "P pessimistic", ":"), ("hold", "hold still", "--")):
            val = ref.get(key)
            if val is None:
                continue
            if 100 * val <= ymax:
                ax.axhline(100 * val, color=ink2, lw=1, ls=ls, zorder=1)
                ax.text(1.01, 100 * val, f"{lab} {100*val:.1f}%", transform=ax.get_yaxis_transform(), color=ink2,
                        fontsize=6.5, va="center")
        if 2.0 <= ymax:
            ax.axhline(2.0, color=ink, lw=0.8, ls=(0, (1, 2)), zorder=1)
            ax.text(1.01, 2.0, "2% target", transform=ax.get_yaxis_transform(), color=ink, fontsize=6.5, va="center")
        ax.set_xscale("log"); ax.set_ylim(0, ymax); ax.set_xlim(700, 45000)
        ax.set_xticks([1000, 3000, 10000, 30000]); ax.set_xticklabels(["1k", "3k", "10k", "30k"])
        ax.minorticks_off()
        ax.set_xlabel("counterfactual branches (2 s each)", color=ink2, fontsize=8)
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
        fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.5)
    fig.suptitle("Closed-loop spill vs counterfactual-branch budget (mean of 3 seeds, 95% paired bootstrap CI)",
                 color=ink, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(w_pad=6, rect=(0, 0.07, 1, 0.97))
    fig.savefig(out, facecolor=surf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="analyse the smoke_* runs (pipeline test only)")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = load(root, smoke=a.smoke)
    if not runs:
        print("no finished runs"); return
    B, ns = boot(runs)
    V = verdicts(B, ns, runs)
    S = secondary_table(runs)
    shared = root / "shared"
    ref = {}
    if (shared / "cl_ref.npz").exists():
        st = np.load(shared / "cl_states.npz"); rf = np.load(shared / "cl_ref.npz")
        sav = st["Vp0"] > 0
        ref["hold"] = float((rf["hold"][sav] < 0).mean()); ref["phys"] = float((rf["phys_greedy"][sav] < 0).mean())
    for k in ("M", "P", "N0"):
        if (k, 0) in B:
            ref[k] = B[(k, 0)][0]
    plot(B, out / "sample_efficiency.png", ref)
    # tables
    L = ["| arm | budget | seeds | spill (savable), mean [95% CI] | per-seed | doomed recall @5% FU | false-safe | IoU | "
         "Spearman rho (H25) | row-grad cos (K10) | winner's curse V_mod-V_true @argmax (logged) | 1-step margin err logged / argmax | 10-tick margin err logged / argmax |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["V", "R", "U", "Vn", "D", "N0", "M", "P"]
    for key in sorted(B, key=lambda k: (order.index(k[0]), k[1])):
        p, samp, nseed = B[key]
        s = S[key]
        seeds = runs[key]
        per_seed = ", ".join(fmt_pct(float(seeds[x]["spill"].mean())) for x in sorted(seeds))
        f2 = lambda t: f"{t[0]:.3f}±{t[1]:.3f}"
        L.append(f"| {key[0]} | {key[1]} | {nseed} | {fmt_pct(p, ci(samp))} | {per_seed} | {f2(s['recall5'])} | "
                 f"{f2(s['false_safe'])} | {f2(s['iou'])} | {f2(s['spearman'])} | {f2(s['gradcos'])} | "
                 f"{s['wc_gap'][0]:+.3f} ({s['wc_gap_logged'][0]:+.3f}) | {s['err_mar_log'][0]:.3f} / {s['err_mar_arg'][0]:.3f} | "
                 f"{s['err_mar10_log'][0]:.3f} / {s['err_mar10_arg'][0]:.3f} |")
    (out / "tables.md").write_text("\n".join(L) + f"\n\nReferences on the same {ns} savable states: hold-still "
                                   f"{fmt_pct(ref.get('hold', float('nan')))}, physics greedy {fmt_pct(ref.get('phys', float('nan')))}.\n")
    summ = dict(n_savable_states=ns, ref=ref,
                arms={f"{k[0]}_b{k[1]}": dict(mean=B[k][0], ci=ci(B[k][1]), n_seeds=B[k][2],
                                              per_seed={int(s): float(runs[k][s]["spill"].mean()) for s in runs[k]},
                                              secondary=S[k]) for k in B},
                verdicts=V)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    print("\n".join(L))
    print(json.dumps(V, indent=1, default=str))


if __name__ == "__main__":
    main()
