"""robust/report.py -- tables (md + json) and plots (png) for Tests 1-3. SIMULATION / SAVED DATA ONLY (numpy, matplotlib Agg).

  python -m world_model.v2.tray_rep.robust.report test1 --tag main
  python -m world_model.v2.tray_rep.robust.report test2 --tag main
  python -m world_model.v2.tray_rep.robust.report test3 --tag main
Output: <OUT>/report/<tag>/{test1,test2,test3}.{md,json} and PNGs. Nothing is read except the runners' jsonl and the budget summary.
Pairing (Test 2): records are paired by (carry, model seed); the corruption stream depends on (carry, seed) only, so every filter saw
the same corrupted stream on the same carry. Confidence intervals: Wilson for a rate, a bootstrap over carries (2000 draws) for a
paired difference.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.robust import corrupt as CR
from world_model.v2.tray_rep.robust import test1 as T1
from world_model.v2.tray_rep.robust import util as RU

PAL = {"N0": "#0072B2", "Rc3k": "#D55E00", "off": "#009E73", "trk": "#CC79A7", "plain": "#0072B2", "conf": "#999999",
       "epi": "#E69F00", "aci": "#56B4E9", "rob": "#D55E00", "f0": "#000000"}
KIND_ORDER = ("f0", "plain", "conf", "epi", "aci", "rob")
KIND_LABEL = dict(f0="no filter", plain="(i) plain F3", conf="(ii) one conformal tau", epi="(iii) epistemic",
                  aci="(iv) ACI residual set", rob="(v) tracker set (ours)")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fmt(r):
    if r["n"] == 0:
        return "n/a"
    return f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}; {100 * r['ci95'][0]:.0f}-{100 * r['ci95'][1]:.0f})"


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


# ===================================================================================== Test 1
def load_test1(tag, split="test", out_root=RU.OUT):
    g = defaultdict(list)
    base = Path(out_root) / "test1" / tag / split
    for f in sorted(base.glob("*/*.jsonl")):
        m = re.match(r"(?P<arm>[A-Za-z0-9]+)_s(?P<seed>\d+)__(?P<cond>[^_.][^.]*?)(__part.*)?$", f.stem)
        if not m:
            continue
        for r in RU.read_jsonl(f):
            r["seed"] = int(m["seed"])
            g[(f.parent.name, m["arm"], m["cond"])].append(r)
    return g


def report1(tag="main", split="test", out_root=RU.OUT, taus=None):
    G = load_test1(tag, split, out_root)
    outd = Path(out_root) / "report" / tag
    outd.mkdir(parents=True, exist_ok=True)
    res, md = {}, [f"# Test 1: false-safe map ({tag}, {split})\n",
                   "False-safe = V(z) >= tau AND the monitor's own fallback fails (T = 50), among certified-savable states. "
                   "Bins = distance of the ball to the nearest tray edge. Seeds pooled. Cells: rate (k/n; Wilson 95%).\n"]
    for (setup, arm, cond), recs in sorted(G.items()):
        tl = taus or (0.0, RU.chosen_tau(setup, arm))
        res[f"{setup}/{arm}/{cond}"] = T1.summarize_file(recs, tl)
        ref = G.get((setup, arm, "off"))
        if ref and cond != "off":
            for tau in tl:
                res[f"{setup}/{arm}/{cond}"][f"v0@{tau:g}"]["excess_vs_off"] = T1.excess_inner_outer(recs, ref, tau)
    for setup in RU.SETUPS:
        for tau_i, tau_name in ((0, "tau = 0 (HJ sign)"), (1, "tau = bridge-chosen")):
            conds = sorted({c for (s, a, c) in G if s == setup})
            if not conds:
                continue
            md.append(f"\n## {setup}, {tau_name}\n")
            for cond in conds:
                rows = []
                for arm in RU.ARMS:
                    key = f"{setup}/{arm}/{cond}"
                    if key not in res:
                        continue
                    tau = (taus or (0.0, RU.chosen_tau(setup, arm)))[tau_i]
                    S = res[key][f"v0@{tau:g}"]
                    for b in S["bins"]:
                        rows.append((arm, f"{tau:g}", b["bin"], b["n"], b["fail"], fmt(b["fs_rate"]), fmt(b["p_fail_given_safe"]),
                                     fmt(b["false_interv"])))
                md.append(f"\n### cond = {cond}\n")
                md.append(md_table(("arm", "tau", "d to edge [m]", "n savable", "fallback fails", "false-safe", "P(fail given safe)",
                                    "false interv. (hold-ok, V<tau)"), rows))
                for arm in RU.ARMS:
                    key = f"{setup}/{arm}/{cond}"
                    if key in res:
                        tau = (taus or (0.0, RU.chosen_tau(setup, arm)))[tau_i]
                        io = res[key][f"v0@{tau:g}"]["inner_outer"]
                        md.append(f"\n- R1 stat {arm}: inner (d<0.02) minus outer (d>=0.06) false-safe = "
                                  f"{io.get('diff', float('nan')):+.3f}, 95% CI ({io['ci95'][0]:+.3f}, {io['ci95'][1]:+.3f}), "
                                  f"n_inner={io['n_inner']} n_outer={io['n_outer']}")
                        ex = res[key][f"v0@{tau:g}"].get("excess_vs_off")
                        if ex:
                            md.append(f"- R1b stat {arm}: excess over the privileged reference (cond {cond} minus off), inner minus outer = "
                                      f"{ex['diff']:+.3f}, 95% CI ({ex['ci95'][0]:+.3f}, {ex['ci95'][1]:+.3f}), "
                                      f"inner excess {ex.get('excess_inner', float('nan')):+.3f}, outer excess {ex.get('excess_outer', float('nan')):+.3f}")
                ph = []
                for arm in RU.ARMS:
                    key = f"{setup}/{arm}/{cond}"
                    if key in res:
                        tau = (taus or (0.0, RU.chosen_tau(setup, arm)))[tau_i]
                        for nm, v in res[key][f"v0@{tau:g}"]["phase"].items():
                            ph.append((arm, nm, v["n"], fmt(v["fail"]), fmt(v["fs"])))
                md.append("\nBy phase at the decision tick:\n")
                md.append(md_table(("arm", "phase", "n savable", "fallback fails", "false-safe"), ph))
    (outd / "test1.md").write_text("\n".join(md) + "\n")
    RU.atomic_json(res, outd / "test1.json")
    # --- plot: false-safe rate against distance, one panel per setup, tau = 0
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    mids = [f"{(DB0 + DB1) / 2 * 100:.0f}" for DB0, DB1 in zip(T1.DBINS[:-1], T1.DBINS[1:])]
    for ax, setup in zip(axs, RU.SETUPS):
        for (s, arm, cond), recs in sorted(G.items()):
            if s != setup:
                continue
            S = res[f"{s}/{arm}/{cond}"]["v0@0"]["bins"]
            y = [b["fs_rate"]["rate"] for b in S]
            lo = [b["fs_rate"]["ci95"][0] for b in S]; hi = [b["fs_rate"]["ci95"][1] for b in S]
            ls = "--" if cond == "off" else "-"
            ax.errorbar(range(len(y)), y, yerr=[np.array(y) - lo, np.array(hi) - y], ls=ls, marker="o", ms=3, capsize=2, lw=1.4,
                        color=PAL.get(arm, "k"), alpha=0.9 if cond in ("off", "trk") else 0.45, label=f"{arm} / {cond}")
        ax.set_xticks(range(len(mids)), mids)
        ax.set_xlabel("ball distance to nearest edge [cm] (bin centre)")
        ax.set_title(f"models: {setup}  (V >= 0 and fallback fails)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("false-safe rate among savable states")
    axs[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outd / "test1_false_safe.png", dpi=130)
    plt.close(fig)
    return res


# ===================================================================================== Test 2
def load_test2(tag, split="test", out_root=RU.OUT):
    """-> dict[(setup, arm, cond, kind)] -> list of records (each with 'seed' = model seed)."""
    g = defaultdict(list)
    base = Path(out_root) / "test2" / tag / split
    newest = {}
    for f in sorted(base.glob("*/*/*.jsonl")):          # one spec per (cell, filter, seed): the newest file wins (stale specs ignored)
        kind = f.stem.split("__")[0]
        ma, ms = re.search(r"arm-([A-Za-z0-9]+)", f.stem), re.search(r"__s(\d+)$", f.stem)
        k2 = (f.parent.parent.name, f.parent.name, kind + ("f0" if "thr--1e+09" in f.stem else ""), ma.group(1) if ma else "", ms.group(1) if ms else "")
        if k2 not in newest or f.stat().st_mtime > newest[k2].stat().st_mtime:
            newest[k2] = f
    for f in sorted(newest.values()):
        kind = f.stem.split("__")[0]
        ma, ms = re.search(r"arm-([A-Za-z0-9]+)", f.stem), re.search(r"__s(\d+)$", f.stem)
        if kind not in KIND_LABEL and kind != "f0" or not ma or not ms:
            continue
        m = dict(arm=ma.group(1), seed=ms.group(1))
        if kind == "plain" and "thr--1e+09" in f.stem:
            kind = "f0"
        for r in RU.read_jsonl(f):
            r["seed"] = int(m["seed"])
            g[(f.parent.parent.name, m["arm"], f.parent.name, kind)].append(r)
    return g


def _cell(recs, subset=None):
    rs = [r for r in recs if subset is None or subset(r)]
    n = len(rs)
    k = sum(r["dropped"] for r in rs)
    return RU.rate(k, n)


def paired_diff(A, B, subset, n_boot=2000, seed=0):
    """mean(drop_A - drop_B) over carries (paired by (ep, seed)) in `subset` (a function of record A); bootstrap over carries."""
    da = {(r["ep"], r["seed"]): r for r in A}
    db = {(r["ep"], r["seed"]): r for r in B}
    keys = [k for k in da if k in db and subset(da[k])]
    if not keys:
        return dict(n=0, diff=float("nan"), ci95=(float("nan"), float("nan")))
    d = {}
    for k in keys:
        d.setdefault(k[0], []).append(float(da[k]["dropped"]) - float(db[k]["dropped"]))
    eps = list(d)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(eps), len(eps))
        diffs.append(np.mean([x for i in pick for x in d[eps[i]]]))
    mean = float(np.mean([x for e in eps for x in d[e]]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(n=len(keys), diff=mean, ci95=(float(lo), float(hi)))


def report2(tag="main", split="test", out_root=RU.OUT):
    G = load_test2(tag, split, out_root)
    outd = Path(out_root) / "report" / tag
    outd.mkdir(parents=True, exist_ok=True)
    cells = sorted({(s, a, c) for (s, a, c, k) in G})
    res, md = {}, [f"# Test 2: five filters at matched conservatism ({tag}, {split})\n",
                   "All filters use the same trained value (arm), seeds pooled, same corrupted stream per (carry, seed). "
                   "Class of a carry = worst phase inside [onset-25, onset] of a nominally failing carry. "
                   "'drop given nominal fails' = the carries the nominal law drops (can the filter save them). "
                   "Matched = thresholds tuned on VAL to the same intervention rate (tune.py).\n"]
    for (setup, arm, cond) in cells:
        kinds = [k for k in KIND_ORDER if (setup, arm, cond, k) in G]
        md.append(f"\n## {setup} / {arm} / {cond}\n")
        rows, R = [], {}
        for k in kinds:
            recs = G[(setup, arm, cond, k)]
            nt = sum(sum(r["ticks_ph"]) for r in recs)
            ni = sum(r["n_interv"] for r in recs)
            R[k] = dict(n=len(recs), drop=_cell(recs), interv=ni / max(nt, 1),
                        drop_nomfail=_cell(recs, lambda r: r["nom_drop"]), caused=_cell([r for r in recs if not r["nom_drop"]]),
                        cls={nm: dict(all=_cell(recs, lambda r, i=i: r["cls"] == i),
                                      nomfail=_cell(recs, lambda r, i=i: r["cls"] == i and r["nom_drop"]))
                             for i, nm in enumerate(CR.PHASES)},
                        interv_phase={nm: float(sum(r["interv_ph"][i] for r in recs) / max(sum(r["ticks_ph"][i] for r in recs), 1))
                                      for i, nm in enumerate(CR.PHASES)},
                        ticks_phase={nm: int(sum(r["ticks_ph"][i] for r in recs)) for i, nm in enumerate(CR.PHASES)})
            rows.append((KIND_LABEL[k], R[k]["n"], fmt(R[k]["drop"]), f"{100 * R[k]['interv']:.2f}%", fmt(R[k]["drop_nomfail"]),
                         fmt(R[k]["caused"])))
        md.append(md_table(("filter", "carries", "drop rate", "interv. rate (ticks)", "drop given nominal fails", "drop given nominal ok (caused)"), rows))
        rows = []
        for k in kinds:
            for nm in CR.PHASES:
                c = R[k]["cls"][nm]
                rows.append((KIND_LABEL[k], nm, R[k]["ticks_phase"][nm], f"{100 * R[k]['interv_phase'][nm]:.2f}%", fmt(c["all"]),
                             fmt(c["nomfail"])))
        md.append("\nSplit by phase (interventions per tick of that phase; drops by carry class):\n")
        md.append(md_table(("filter", "phase", "ticks", "interv. rate", "drop given class", "drop given class, nominal fails"), rows))
        P = {}
        if "rob" in kinds:
            sub_drop = lambda r: r["cls"] == CR.PH_DROP and r["nom_drop"]
            sub_all = lambda r: True
            sub_nf = lambda r: r["nom_drop"]
            if "aci" in kinds:
                P["R2_rob_minus_aci_dropout_nomfail"] = paired_diff(G[(setup, arm, cond, "rob")], G[(setup, arm, cond, "aci")], sub_drop)
            for k in ("plain", "conf", "epi", "aci"):
                if k in kinds:
                    P[f"R3_rob_minus_{k}_all"] = paired_diff(G[(setup, arm, cond, "rob")], G[(setup, arm, cond, k)], sub_all)
                    P[f"R3_rob_minus_{k}_nomfail"] = paired_diff(G[(setup, arm, cond, "rob")], G[(setup, arm, cond, k)], sub_nf)
            md.append("\nPaired differences, drop(rob) - drop(other); negative = ours drops fewer; bootstrap 95% CI over carries:\n")
            md.append(md_table(("comparison", "n pairs", "diff", "95% CI"),
                               [(k, v["n"], f"{v['diff']:+.3f}", f"({v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f})") for k, v in P.items()]))
            ints = {k: R[k]["interv"] for k in kinds if k != "f0"}
            md.append(f"\nMatching check (intervention rate per tick): " + ", ".join(f"{k} {100 * v:.2f}%" for k, v in ints.items()))
        res[f"{setup}/{arm}/{cond}"] = dict(filters=R, paired=P)
    (outd / "test2.md").write_text("\n".join(md) + "\n")
    RU.atomic_json(res, outd / "test2.json")
    # --- plot: drop rate among nominally failing carries whose critical window has a dropout, per filter
    plt = _plt()
    keys = [c for c in cells if any(cls_n > 0 for cls_n in
                                    [res[f"{c[0]}/{c[1]}/{c[2]}"]["filters"][k]["cls"]["dropout"]["nomfail"]["n"]
                                     for k in res[f"{c[0]}/{c[1]}/{c[2]}"]["filters"]])]
    if keys:
        fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys) + 1, 3.8), squeeze=False, sharey=True)
        for ax, c in zip(axs[0], keys):
            F = res[f"{c[0]}/{c[1]}/{c[2]}"]["filters"]
            ks = [k for k in KIND_ORDER if k in F]
            y = [F[k]["cls"]["dropout"]["nomfail"]["rate"] for k in ks]
            lo = [F[k]["cls"]["dropout"]["nomfail"]["ci95"][0] for k in ks]
            hi = [F[k]["cls"]["dropout"]["nomfail"]["ci95"][1] for k in ks]
            ax.bar(range(len(ks)), y, yerr=[np.array(y) - lo, np.array(hi) - y], color=[PAL[k] for k in ks], capsize=2)
            ax.set_xticks(range(len(ks)), [k for k in ks], rotation=30, fontsize=8)
            ax.set_title("/".join(c), fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        axs[0][0].set_ylabel("drop rate | dropout window, nominal fails")
        fig.tight_layout()
        fig.savefig(outd / "test2_dropout_drops.png", dpi=130)
        plt.close(fig)
    return res


# ===================================================================================== Test 3
def report3(tag="main", split="test", out_root=RU.OUT, budget_tag="main"):
    G = load_test2(f"{tag}_lat", split, out_root)
    outd = Path(out_root) / "report" / tag
    outd.mkdir(parents=True, exist_ok=True)
    res, md = {}, [f"# Test 3: drop rate against injected latency ({tag}, {split})\n",
                   "Thresholds fixed (plain: bridge tau; rob: matched on VAL at the base condition). Latency in ticks of 20 ms.\n"]
    rows = defaultdict(dict)
    for (setup, arm, cond, kind), recs in G.items():
        m = re.fullmatch(r"(?:trk)?lat(\d+)", cond)
        if not m:
            continue
        rows[(setup, arm, kind)][int(m.group(1))] = (_cell(recs), sum(r["n_interv"] for r in recs) / max(sum(sum(r["ticks_ph"]) for r in recs), 1))
    bud = {}
    for s in RU.SETUPS:
        p = Path(out_root) / "budget" / budget_tag / f"summary_{s}.json"
        if p.exists():
            bud[s] = json.loads(p.read_text())
    for (setup, arm, kind), d in sorted(rows.items()):
        md.append(f"\n## {setup} / {arm} / {kind}\n")
        md.append(md_table(("latency [ticks]", "latency [ms]", "drop rate", "interv. rate"),
                           [(k, 20 * k, fmt(d[k][0]), f"{100 * d[k][1]:.2f}%") for k in sorted(d)]))
        res[f"{setup}/{arm}/{kind}"] = {str(k): dict(drop=d[k][0], interv=d[k][1]) for k in sorted(d)}
    if bud:
        md.append("\n## Derived budget\n```\n" + json.dumps({s: b.get("derived_budget_ticks") for s, b in bud.items()}, indent=1) + "\n```\n"
                  + "\n".join(f"{s}: {b['how']}" for s, b in bud.items() if "how" in b) + "\n")
    (outd / "test3.md").write_text("\n".join(md) + "\n")
    RU.atomic_json(res, outd / "test3.json")
    plt = _plt()
    sets = sorted({s for (s, a, k) in rows})
    if sets:
        fig, axs = plt.subplots(1, len(sets), figsize=(5 * len(sets), 3.8), squeeze=False, sharey=True)
        for ax, setup in zip(axs[0], sets):
            for (s, arm, kind), d in sorted(rows.items()):
                if s != setup:
                    continue
                x = sorted(d)
                y = [d[k][0]["rate"] for k in x]
                ax.errorbar(x, y, yerr=[[d[k][0]["rate"] - d[k][0]["ci95"][0] for k in x], [d[k][0]["ci95"][1] - d[k][0]["rate"] for k in x]],
                            marker="o", ms=3, capsize=2, color=PAL[kind], label=f"{arm} {kind}", lw=1.4)
            b = bud.get(setup, {}).get("derived_budget_ticks")
            if b:
                ax.axvline(b["measured_D_median"], color="#555", ls=":", lw=1)
                ax.text(b["measured_D_median"], ax.get_ylim()[1] * 0.95, " oracle median\n tolerable delay", fontsize=7, va="top")
                ax.axvline(max(b["from_turn"], 0), color="#999", ls="--", lw=1)
                ax.text(max(b["from_turn"], 0), ax.get_ylim()[1] * 0.75, " derived budget\n (3.5 - turnaround)", fontsize=7, va="top")
            ax.set_xlabel("injected latency [ticks, 20 ms]")
            ax.set_title(setup, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7, frameon=False)
        axs[0][0].set_ylabel("drop rate")
        fig.tight_layout()
        fig.savefig(outd / "test3_latency.png", dpi=130)
        plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["test1", "test2", "test3", "all"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(RU.OUT))
    ap.add_argument("--budget-tag", default="main")
    a = ap.parse_args()
    if a.cmd in ("test1", "all"):
        report1(a.tag, a.split, a.out)
    if a.cmd in ("test2", "all"):
        report2(a.tag, a.split, a.out)
    if a.cmd in ("test3", "all"):
        report3(a.tag, a.split, a.out, a.budget_tag)
    RU.log(f"[report] wrote {Path(a.out) / 'report' / a.tag}")


if __name__ == "__main__":
    main()
