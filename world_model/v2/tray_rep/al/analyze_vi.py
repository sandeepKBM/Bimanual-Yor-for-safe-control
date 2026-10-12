"""al/analyze_vi.py -- value-of-information map tables + plots from the al.run_vi records. numpy/matplotlib only (no pandas). No hardware.

  python -m world_model.v2.tray_rep.al.analyze_vi --root <vimap dir> [--out <dir>]
Reads <root>/auth<a>_pop<p>/<set>/CL__I-<I>__s0*.jsonl. Writes into <out> (default <root>/analysis):
  cells.csv         one row per (pop, auth, I, set): n, drops, arrived, T_arrive/T_complete (mean over non-dropped, arrived), interv rate, dev, backup/infeas tick share, set coverage, log-width
  map_tables.md     per (pop, auth): T, drops, interv, dev, backup tables with I rows x set columns (+ coverage line)
  voi.md / voi.json value of information per regime: best zero-drop speed per set, time and intervention saving of oracle/narrow/medium(/popr) vs the full set
                    (a) each set at its OWN best zero-drop speed, (b) at the FULL set's best zero-drop speed (paired over carries, bootstrap 95% CI), (c) at every speed (paired)
  vimap_T.png       mean time-to-goal vs speed I, per set, panels pop x auth (marker x = cell with drops, number = drops)
  vimap_voi.png     time saving (%) and intervention saving (pp) vs the full set at the full set's best zero-drop speed, per regime
Conventions: time = cl_t_arrive_s (stricter: ref played out, <=1.5 cm, <=3 cm/s) of non-dropped carries that arrived; t_complete_s also tabulated. Paired deltas use carries
(same VAL episode, same hidden mu) where both cells are non-dropped and arrived. 'zero-drop speed' = cell with 0 drops of its n carries. No claims, observations only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

SETS = ["oracle", "narrow", "medium", "full", "popr", "mis"]
CMP = ["oracle", "narrow", "medium", "popr"]          # sets compared against the 'full' set
AUTHS = [1.0, 0.5, 0.25]


def load(root):
    D = {}
    for f in sorted(glob.glob(str(Path(root) / "auth*_pop*" / "*" / "CL__I-*.jsonl"))):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            D.setdefault((r["pop"], float(r["auth"]), float(r["I"]), r["set"]), {})[int(r["ep"])] = r
    return D


def fin(x):
    return x is not None and np.isfinite(x)


def cell_stats(R):
    rs = list(R.values())
    n = len(rs)
    drop = np.array([bool(r["dropped"]) for r in rs])
    ta = np.array([r["cl_t_arrive_s"] for r in rs], float)
    tc = np.array([r["t_complete_s"] for r in rs], float)
    ok = ~drop
    ar = ok & np.isfinite(ta)
    nt = np.array([max(r.get("x_f5_n", 1), 1) for r in rs], float)
    g = lambda k: np.array([r.get(k, np.nan) for r in rs], float)
    m = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return dict(n=n, drops=int(drop.sum()), arrived=int(ar.sum()), T_arrive=m(ta[ar]), T_complete=m(tc[ok & np.isfinite(tc)]), interv=m(g("interv_rate")[ok]),
                interv_all=m(g("interv_rate")), dev=m(g("dev_norm_mean")[ok]), backup=m((g("x_f5_backup") / nt)[ok]), infeas=m((g("x_f5_infeas") / nt)[ok]),
                cov=m(g("x_al_cov")), cov_raw=m(g("x_al_cov_raw")), lw=m(g("x_al_lw")), lw_raw=m(g("x_al_lw_raw")), pause=m(g("cl_pause_ticks")[ok]),
                mu_med=float(np.median(g("mu"))))


def paired(Ra, Rb, key, rng, nb=2000):
    """mean of (b - a) over episodes where both non-dropped and (for time) arrived. -> (mean, lo, hi, n)"""
    d = []
    for ep, a in Ra.items():
        b = Rb.get(ep)
        if b is None or a["dropped"] or b["dropped"]:
            continue
        if key == "T":
            x, y = a["cl_t_arrive_s"], b["cl_t_arrive_s"]
        else:
            x, y = a["interv_rate"], b["interv_rate"]
        if fin(x) and fin(y):
            d.append(y - x)
    d = np.array(d)
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), len(d)
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(nb)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(d)


def best_speed(C, pop, auth, s):
    """(I*, stats) of the zero-drop cell with the lowest T_arrive among the speeds of this regime/set; None if no zero-drop speed."""
    best = None
    for (p, a, I, ss), st in C.items():
        if (p, a, ss) == (pop, auth, s) and st["drops"] == 0 and st["arrived"] > 0 and np.isfinite(st["T_arrive"]):
            if best is None or st["T_arrive"] < best[1]["T_arrive"]:
                best = (I, st)
    return best


def fmt(x, d=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.root) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    D = load(a.root)
    C = {k: cell_stats(R) for k, R in D.items()}
    pops = sorted({k[0] for k in C}, key=lambda p: ["full", "low"].index(p))
    auths = sorted({k[1] for k in C}, reverse=True)
    speeds = sorted({k[2] for k in C})
    sets = [s for s in SETS if any(k[3] == s for k in C)]

    # ---- cells.csv
    with open(out / "cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["pop", "auth", "I", "set", "n", "drops", "arrived", "T_arrive", "T_complete", "interv", "dev", "backup", "infeas", "cov", "cov_raw", "lw", "lw_raw", "pause", "mu_med"]
        w.writerow(cols)
        for k in sorted(C):
            w.writerow([k[0], k[1], k[2], k[3]] + [C[k][c] for c in cols[4:]])

    # ---- map_tables.md
    L = ["# Value-of-information map: T(I, auth, pop, set), D, interventions (VAL, piper_trk, seed 0, closed loop, F5 eps_a=0, calib_v2)", "",
         "Cell = n carries (same VAL episodes in every cell). T = mean cl_t_arrive_s [s] of non-dropped arrived carries; drops = k/n; interv = fraction of carry ticks with u != u_nom (non-dropped); "
         "dev = mean |D(u-u_nom)|; bk = F5 backup-law share of ticks (= QP infeasible share when equal).", ""]
    for pop in pops:
        for auth in auths:
            L += [f"## pop={pop}, auth={auth:g}", ""]
            for title, key, d in (("T_arrive [s]", "T_arrive", 2), ("drops k/n", "drops", 0), ("interv rate", "interv", 3), ("dev", "dev", 3), ("backup share", "backup", 3), ("arrived/n", "arrived", 0)):
                L += [f"**{title}**", "", "| I | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
                for I in speeds:
                    row = []
                    for s in sets:
                        st = C.get((pop, auth, I, s))
                        if st is None:
                            row.append("")
                        elif key == "drops":
                            row.append(f"{st['drops']}/{st['n']}")
                        elif key == "arrived":
                            row.append(f"{st['arrived']}/{st['n']}")
                        else:
                            row.append(fmt(st[key], d))
                    L.append(f"| {I:g} | " + " | ".join(row) + " |")
                L.append("")
            cov = {s: [C[k]["cov"] for k in C if k[:2] == (pop, auth) and k[3] == s] for s in sets}
            lw = {s: [C[k]["lw"] for k in C if k[:2] == (pop, auth) and k[3] == s] for s in sets}
            L.append("set coverage of the true mu (mean over cells) / mean clipped log-width: " + "; ".join(f"{s} {fmt(np.mean(cov[s]))}/{fmt(np.mean(lw[s]))}" for s in sets if cov[s]) + "\n")
    (out / "map_tables.md").write_text("\n".join(L))

    # ---- VOI
    V = {}
    M = ["# Value of information per regime (observations)", "",
         "best zero-drop speed I* per set (lowest mean T_arrive among speeds with 0 drops of n); saving = full-set value - set value (positive = the narrower set is better).", "",
         "## (a) each set at its own best zero-drop speed", "",
         "| pop | auth | set | I* | T [s] | interv | dT vs full [s] | dT % | d interv vs full [pp] |", "|---|---|---|---|---|---|---|---|---|"]
    for pop in pops:
        for auth in auths:
            bf = best_speed(C, pop, auth, "full")
            for s in CMP + ["mis"]:
                if not any(k[:2] == (pop, auth) and k[3] == s for k in C):
                    continue
                bs = best_speed(C, pop, auth, s)
                rec = dict(pop=pop, auth=auth, set=s, I_star=None if bs is None else bs[0], T=None if bs is None else bs[1]["T_arrive"], interv=None if bs is None else bs[1]["interv"],
                           I_star_full=None if bf is None else bf[0], T_full=None if bf is None else bf[1]["T_arrive"], interv_full=None if bf is None else bf[1]["interv"])
                if bs and bf:
                    rec.update(dT=bf[1]["T_arrive"] - bs[1]["T_arrive"], dT_pct=100 * (bf[1]["T_arrive"] - bs[1]["T_arrive"]) / bf[1]["T_arrive"],
                               dI_pp=100 * (bf[1]["interv"] - bs[1]["interv"]))
                V[f"a|{pop}|{auth:g}|{s}"] = rec
                M.append(f"| {pop} | {auth:g} | {s} | {fmt(rec['I_star'])} | {fmt(rec['T'])} | {fmt(rec['interv'], 3)} | {fmt(rec.get('dT'))} | {fmt(rec.get('dT_pct'), 1)} | {fmt(rec.get('dI_pp'), 1)} |")
            if bf is None and any(k[:2] == (pop, auth) for k in C):
                M.append(f"| {pop} | {auth:g} | (full set has no zero-drop speed) | | | | | | |")
    M += ["", "## (b) at the FULL set's best zero-drop speed (paired over carries, both non-dropped and arrived; bootstrap 95% CI)", "",
          "| pop | auth | I (full*) | set | drops set / full | dT = T_full - T_set [s] (CI) | n | d interv = interv_full - interv_set [pp] (CI) |", "|---|---|---|---|---|---|---|---|"]
    for pop in pops:
        for auth in auths:
            bf = best_speed(C, pop, auth, "full")
            if bf is None:
                continue
            I = bf[0]
            Rf = D[(pop, auth, I, "full")]
            for s in CMP + ["mis"]:
                if (pop, auth, I, s) not in D:
                    continue
                Rs = D[(pop, auth, I, s)]
                t = paired(Rs, Rf, "T", rng)
                iv = paired(Rs, Rf, "I", rng)
                V[f"b|{pop}|{auth:g}|{s}"] = dict(I=I, drops_set=C[(pop, auth, I, s)]["drops"], drops_full=C[(pop, auth, I, "full")]["drops"], dT=t[0], dT_ci=[t[1], t[2]], n=t[3], dI_pp=100 * iv[0], dI_ci=[100 * iv[1], 100 * iv[2]])
                M.append(f"| {pop} | {auth:g} | {I:g} | {s} | {C[(pop, auth, I, s)]['drops']}/{C[(pop, auth, I, s)]['n']} / {C[(pop, auth, I, 'full')]['drops']}/{C[(pop, auth, I, 'full')]['n']} | "
                         f"{fmt(t[0])} ({fmt(t[1])}, {fmt(t[2])}) | {t[3]} | {fmt(100 * iv[0], 1)} ({fmt(100 * iv[1], 1)}, {fmt(100 * iv[2], 1)}) |")
    M += ["", "## (c) paired vs the full set at every speed", "", "| pop | auth | I | set | drops set / full | dT [s] (CI) | n | d interv [pp] (CI) |", "|---|---|---|---|---|---|---|---|"]
    for pop in pops:
        for auth in auths:
            for I in speeds:
                if (pop, auth, I, "full") not in D:
                    continue
                for s in CMP + ["mis"]:
                    if (pop, auth, I, s) not in D:
                        continue
                    t = paired(D[(pop, auth, I, s)], D[(pop, auth, I, "full")], "T", rng, 500)
                    iv = paired(D[(pop, auth, I, s)], D[(pop, auth, I, "full")], "I", rng, 500)
                    V[f"c|{pop}|{auth:g}|{I:g}|{s}"] = dict(dT=t[0], dT_ci=[t[1], t[2]], n=t[3], dI_pp=100 * iv[0], dI_ci=[100 * iv[1], 100 * iv[2]])
                    M.append(f"| {pop} | {auth:g} | {I:g} | {s} | {C[(pop, auth, I, s)]['drops']}/{C[(pop, auth, I, s)]['n']} / {C[(pop, auth, I, 'full')]['drops']}/{C[(pop, auth, I, 'full')]['n']} | "
                             f"{fmt(t[0])} ({fmt(t[1])}, {fmt(t[2])}) | {t[3]} | {fmt(100 * iv[0], 1)} ({fmt(100 * iv[1], 1)}, {fmt(100 * iv[2], 1)}) |")
    M += ["", "## (d) danger side: mis-centred narrow set (centre mu e^1.5) vs full, drops per speed", "", "| pop | auth | I | mis drops | full drops | oracle drops | mis cov | mis interv | full interv |", "|---|---|---|---|---|---|---|---|---|"]
    for pop in pops:
        for auth in auths:
            for I in speeds:
                g = lambda s: C.get((pop, auth, I, s))
                if g("mis") and g("full"):
                    M.append(f"| {pop} | {auth:g} | {I:g} | {g('mis')['drops']}/{g('mis')['n']} | {g('full')['drops']}/{g('full')['n']} | "
                             f"{'-' if not g('oracle') else str(g('oracle')['drops']) + '/' + str(g('oracle')['n'])} | {fmt(g('mis')['cov'])} | {fmt(g('mis')['interv'], 3)} | {fmt(g('full')['interv'], 3)} |")
    (out / "voi.md").write_text("\n".join(M))
    (out / "voi.json").write_text(json.dumps(V, indent=1, default=float))

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"oracle": "#0072B2", "narrow": "#009E73", "medium": "#E69F00", "popr": "#56B4E9", "full": "#444444", "mis": "#D55E00"}
    fig, ax = plt.subplots(len(pops), len(auths), figsize=(4.2 * len(auths), 3.3 * len(pops)), squeeze=False, sharex=True)
    for i, pop in enumerate(pops):
        for j, auth in enumerate(auths):
            A = ax[i][j]
            for s in sets:
                xs = [I for I in speeds if (pop, auth, I, s) in C]
                ys = [C[(pop, auth, I, s)]["T_arrive"] for I in xs]
                A.plot(xs, ys, "-o", color=col[s], ms=4, lw=1.3, label=s)
                for I, y in zip(xs, ys):
                    k = C[(pop, auth, I, s)]["drops"]
                    if k and np.isfinite(y):
                        A.annotate(f"x{k}", (I, y), color=col[s], fontsize=7, xytext=(2, 4), textcoords="offset points")
            A.set_title(f"pop={pop}, auth={auth:g}", fontsize=9)
            A.set_xlabel("speed I (rel. to feasible)", fontsize=8)
            A.set_ylabel("mean time to arrive [s]", fontsize=8)
            A.tick_params(labelsize=7)
            A.grid(alpha=0.25)
    ax[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle("T(I) by mu-set handed to F5 (xk = k drops in that cell)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "vimap_T.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    reg = [(p, au) for p in pops for au in auths]
    wd = 0.2
    for q, (key, yl) in enumerate((("dT_pct", "time saving vs full set [% of T_full]"), ("dI_pp", "intervention saving vs full set [pp of ticks]"))):
        for m, s in enumerate(["oracle", "narrow", "medium", "popr"]):
            ys = [V.get(f"a|{p}|{au:g}|{s}", {}).get(key, np.nan) for (p, au) in reg]
            ax[q].bar(np.arange(len(reg)) + (m - 1.5) * wd, np.nan_to_num(ys), wd, color=col[s], label=s)
        ax[q].set_xticks(np.arange(len(reg)))
        ax[q].set_xticklabels([f"{p}\nauth {au:g}" for p, au in reg], fontsize=7)
        ax[q].set_ylabel(yl, fontsize=8)
        ax[q].axhline(0, color="k", lw=0.6)
        ax[q].grid(alpha=0.25, axis="y")
    ax[0].legend(fontsize=7)
    fig.suptitle("each set at its own best zero-drop speed, vs the full set at its own best zero-drop speed", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "vimap_voi.png", dpi=130)
    plt.close(fig)
    print("wrote", out, "cells", len(C), "carries", sum(len(R) for R in D.values()))


if __name__ == "__main__":
    main()
