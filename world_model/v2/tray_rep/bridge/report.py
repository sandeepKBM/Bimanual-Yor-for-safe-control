"""bridge/report.py -- tables, VAL parameter selection and Pareto plots for the bridge experiment.
SIMULATION / SAVED DATA ONLY. Reads <out>/<tag>/<split>/<setup>/*.jsonl written by bridge.run.

  python -m world_model.v2.tray_rep.bridge.report select --tag main     # VAL sweeps -> <out>/<tag>/chosen_params.json
  python -m world_model.v2.tray_rep.bridge.report table  --tag main --split test
  python -m world_model.v2.tray_rep.bridge.report plots  --tag main     # PNGs under <out>/<tag>/plots/
  python -m world_model.v2.tray_rep.bridge.report all    --tag main

select (all on VAL, never on TEST):
  F3 tau   per (setup, arm): the tau with the lowest drop rate; among taus within `--tol` (default 0.01) of that rate, the lowest
           mean deviation. Seeds are pooled (mean). A tau at the edge of the swept grid is flagged.
  F2       the (theta_max, a_cap) point chosen by the same rule.
  F1       key F1@<arm>: of the F1 grid points whose mean deviation is within --dev-tol (default 15%) of F3-<arm>'s chosen point,
           the lowest drop rate (nearest point if none is that close). Key F1@best: the lowest drop rate on the grid.
Keys in chosen_params.json are "<setup>/<name>", for example "piper_trk/F3@Rc3k"; bridge.run takes --filter @F3@Rc3k.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.bridge import metrics as MT
from world_model.v2.tray_rep.bridge.run import read_jsonl, OUT

COL = {"F0": "#666666", "F1": "#D55E00", "F2": "#009E73", "F3N0": "#56B4E9", "F3Rc": "#0072B2", "F4": "#CC79A7"}


def label(spec):
    n, p = F.parse_spec(spec)
    if n == "F3":
        return f"F3 {p.get('arm', '?')} tau={p.get('tau', 0):g}"
    if n == "F1":
        return f"F1 l_align={p.get('l_align', 1):g} l_prox={p.get('l_prox', 10):g}" + (f" vlook={p['vlook']:g}" if p.get("vlook") else "")
    if n == "F2":
        return f"F2 theta={p.get('theta_max', 0.15):g} acap={p.get('a_cap', 1):g}"
    return spec


# ===================================================================================== loading / aggregation
def load_runs(root, tag, split):
    """-> list of dict(setup, spec, seed, recs). Skips runs without a summary."""
    out = []
    for pj in sorted(glob.glob(str(Path(root) / tag / split / "*" / "*.jsonl"))):
        pj = Path(pj)
        ps = pj.with_suffix(".json")
        if not ps.exists():
            continue
        sm = json.loads(ps.read_text())
        out.append(dict(setup=pj.parent.name, spec=sm["spec"], seed=int(sm["seed"]), recs=sorted(read_jsonl(pj), key=lambda r: r["ep"]),
                        n_target=sm["n"]))
    return out


def nominal_drops(runs):
    """{setup: {ep: nominal drops}} from the F0 run (seed 0) of each setup; episodes without one fall back to the stored onset."""
    nd = {}
    for r in runs:
        if r["spec"] == "F0" and r["seed"] == 0:
            nd[r["setup"]] = {x["ep"]: bool(x["dropped"]) for x in r["recs"]}
    return nd


def run_summary(r, nd):
    m = nd.get(r["setup"], {})
    nom = np.array([m.get(x["ep"], x["onset_stored"] >= 0) for x in r["recs"]], bool)
    return MT.summarize(r["recs"], nom)


def aggregate(runs):
    """group by (setup, spec) over seeds -> dict(setup, spec, seeds, n, metrics...)."""
    nd = nominal_drops(runs)
    g = defaultdict(list)
    for r in runs:
        g[(r["setup"], r["spec"])].append((r, run_summary(r, nd)))
    out = []
    for (setup, spec), lst in sorted(g.items()):
        S = [s for _, s in lst]
        rates = np.array([s["drop"]["rate"] for s in S])
        k, n = sum(s["drop"]["k"] for s in S), sum(s["drop"]["n"] for s in S)
        mean = lambda f: float(np.mean([f(s) for s in S]))
        out.append(dict(
            setup=setup, spec=spec, label=label(spec), seeds=sorted(r["seed"] for r, _ in lst), n=int(np.mean([s["n"] for s in S])),
            drop=float(rates.mean()), drop_sd=float(rates.std(ddof=1)) if len(rates) > 1 else float("nan"),
            drop_ci=MT.wilson(k, n), interv=mean(lambda s: s["interv_rate"]), dev_raw=mean(lambda s: s["mean_dev_raw"]),
            dev=mean(lambda s: s["mean_dev_norm"]), compl=mean(lambda s: s["completion"]),
            drop_nomdrop=mean(lambda s: s["drop_given_nom_drop"]["rate"]), drop_nomok=mean(lambda s: s["drop_given_nom_ok"]["rate"]),
            n_nomdrop=int(np.mean([s["drop_given_nom_drop"]["n"] for s in S])), n_nomok=int(np.mean([s["drop_given_nom_ok"]["n"] for s in S])),
            saved=mean(lambda s: s["saved"]), caused=mean(lambda s: s["caused"]), tunnel=sum(s["n_tunnel"] for s in S),
            filt_s=mean(lambda s: s["filt_s_per_ep"]), sim_s=mean(lambda s: s["sim_s_per_ep"]),
            # path-progress fields: NaN for old result files that were written before the metric existed
            progress=mean(lambda s: s.get("progress", float("nan"))),
            goal=mean(lambda s: s["goal"]["rate"] if "goal" in s else float("nan")),
            goal_time_s=float(np.nanmean([s.get("goal_time_s", float("nan")) for s in S])) if any(
                np.isfinite(s.get("goal_time_s", float("nan"))) for s in S) else float("nan")))
    return out


# ===================================================================================== table
def _fmt_prog(a):
    """'path progress | goal reached (mean time)' cells; n/a for result files from before the metric existed."""
    if not np.isfinite(a.get("progress", float("nan"))):
        return "n/a | n/a"
    gt = a.get("goal_time_s", float("nan"))
    return f"{a['progress']:.3f} | {a['goal']:.3f}" + (f" ({gt:.1f})" if np.isfinite(gt) else "")


def fmt_table(agg):
    lines = []
    for setup in sorted({a["setup"] for a in agg}):
        lines.append(f"\n### setup: {setup}\n")
        lines.append("| method | seeds | n | drop rate (mean over seeds, sd) | pooled 95% CI | intervention | mean abs(u-u_nom) raw / norm | "
                     "completion (time) | path progress | goal reached (time s) | drop given nominal drops (n) | drop given nominal ok (n) | saved / caused |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in [x for x in agg if x["setup"] == setup]:
            sd = "" if np.isnan(a["drop_sd"]) else f" (sd {a['drop_sd']:.3f})"
            lines.append(f"| {a['label']} | {len(a['seeds'])} | {a['n']} | {a['drop']:.3f}{sd} | ({a['drop_ci'][0]:.3f}, {a['drop_ci'][1]:.3f}) | "
                         f"{a['interv']:.3f} | {a['dev_raw']:.3f} / {a['dev']:.3f} | {a['compl']:.3f} | {_fmt_prog(a)} | "
                         f"{a['drop_nomdrop']:.3f} ({a['n_nomdrop']}) | {a['drop_nomok']:.3f} ({a['n_nomok']}) | "
                         f"{a['saved']:.1f} / {a['caused']:.1f} |")
    return "\n".join(lines)


# ===================================================================================== selection (VAL)
def pick(points, tol=0.01):
    """lowest drop; among those within tol of it, the lowest deviation."""
    best = min(p["drop"] for p in points)
    return min((p for p in points if p["drop"] <= best + tol), key=lambda p: (p["dev"], p["drop"]))


def select(root, tag, tol=0.01, dev_tol=0.15):
    runs = load_runs(root, tag, "val")
    agg = aggregate(runs)
    chosen, notes = {}, []
    for setup in sorted({a["setup"] for a in agg}):
        A = [a for a in agg if a["setup"] == setup]
        by = defaultdict(list)
        for a in A:
            by[F.parse_spec(a["spec"])[0] + (":" + str(F.parse_spec(a["spec"])[1].get("arm")) if a["spec"].startswith("F3") else "")].append(a)
        rec = lambda a, **kw: dict(spec=a["spec"], val=dict(drop=a["drop"], drop_ci=a["drop_ci"], dev=a["dev"], interv=a["interv"], n=a["n"],
                                                             seeds=a["seeds"]), **kw)
        for key, pts in sorted(by.items()):
            if key == "F0":
                chosen[f"{setup}/F0"] = rec(pts[0])
            elif key == "F2":
                chosen[f"{setup}/F2"] = rec(pick(pts, tol))
            elif key.startswith("F3:"):
                arm = key.split(":")[1]
                c = pick(pts, tol)
                taus = sorted(F.parse_spec(p["spec"])[1].get("tau", 0.0) for p in pts)
                edge = F.parse_spec(c["spec"])[1].get("tau", 0.0) in (taus[0], taus[-1]) and len(taus) > 2
                chosen[f"{setup}/F3@{arm}"] = rec(c, edge_of_grid=bool(edge))
                if edge:
                    notes.append(f"{setup} F3 {arm}: chosen tau is at the edge of the swept grid {taus}; widen the sweep")
        f1 = by.get("F1", [])
        if f1:
            chosen[f"{setup}/F1@best"] = rec(min(f1, key=lambda p: (p["drop"], p["dev"])))
            for k in [k for k in chosen if k.startswith(f"{setup}/F3@")]:
                arm = k.split("@")[1]
                tgt = chosen[k]["val"]["dev"]
                near = [p for p in f1 if abs(p["dev"] - tgt) <= dev_tol * max(tgt, 1e-6)]
                c = min(near, key=lambda p: (p["drop"], abs(p["dev"] - tgt))) if near else min(f1, key=lambda p: abs(p["dev"] - tgt))
                chosen[f"{setup}/F1@{arm}"] = rec(c, matched_to=k, target_dev=tgt, within_tol=bool(near))
                if not near:
                    notes.append(f"{setup} F1@{arm}: no F1 grid point within {dev_tol:.0%} of the F3 deviation {tgt:.4f}; nearest used")
    p = Path(root) / tag / "chosen_params.json"
    p.write_text(json.dumps(dict(selected_on="val", tol=tol, dev_tol=dev_tol, notes=notes, **chosen), indent=1, default=float))
    return chosen, notes, p


# ===================================================================================== plots
def pareto_front(pts):
    """lower-left front of (dev, drop) pairs."""
    pts = sorted(pts)
    front, best = [], 1e9
    for d, y in pts:
        if y < best - 1e-12:
            front.append((d, y)); best = y
    return front


def plots(root, tag, outdir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = Path(outdir or Path(root) / tag / "plots")
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    chosen_p = Path(root) / tag / "chosen_params.json"
    chosen = json.loads(chosen_p.read_text()) if chosen_p.exists() else {}
    for split in ("val", "test"):
        agg = aggregate(load_runs(root, tag, split))
        for setup in sorted({a["setup"] for a in agg}):
            A = [a for a in agg if a["setup"] == setup]
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
            # --- Pareto: drop rate vs mean deviation
            a0 = ax[0]
            f1 = [a for a in A if a["spec"].startswith("F1")]
            f2 = [a for a in A if a["spec"].startswith("F2")]
            f4 = [a for a in A if a["spec"].startswith("F4")]
            for a in A:
                if a["spec"] == "F0":
                    a0.scatter([a["dev"]], [a["drop"]], c=COL["F0"], s=60, marker="s", zorder=5, label="F0 nominal")
            for pts, c, nm in ((f1, COL["F1"], "F1 keypoint-cost grid"), (f2, COL["F2"], "F2 hand rule grid")):
                if pts:
                    a0.scatter([p["dev"] for p in pts], [p["drop"] for p in pts], c=c, s=22, alpha=0.55, label=nm)
                    fr = pareto_front([(p["dev"], p["drop"]) for p in pts])
                    a0.step([x for x, _ in fr], [y for _, y in fr], where="post", c=c, lw=1.4)
            arms = sorted({F.parse_spec(a["spec"])[1].get("arm") for a in A if a["spec"].startswith("F3")})
            for arm in arms:
                pts = sorted((a for a in A if a["spec"].startswith("F3") and F.parse_spec(a["spec"])[1].get("arm") == arm and
                              np.isfinite(F.parse_spec(a["spec"])[1].get("tau", 0))), key=lambda a: F.parse_spec(a["spec"])[1].get("tau", 0))
                c = COL["F3Rc"] if arm.startswith("R") else COL["F3N0"]
                if pts:
                    a0.plot([p["dev"] for p in pts], [p["drop"] for p in pts], "-o", c=c, ms=5, lw=1.6, label=f"F3 {arm} (tau sweep)")
                    for p in pts:
                        a0.annotate(f"{F.parse_spec(p['spec'])[1].get('tau', 0):g}", (p["dev"], p["drop"]), fontsize=6, xytext=(3, 3),
                                    textcoords="offset points", color=c)
            for a in f4:
                a0.scatter([a["dev"]], [a["drop"]], c=COL["F4"], marker="*", s=130, zorder=6, label="F4 oracle look-ahead")
            for k, v in chosen.items():
                if k.startswith(setup + "/") and split == "val":
                    ex = [a for a in A if a["spec"] == v["spec"]]
                    if ex:
                        a0.scatter([ex[0]["dev"]], [ex[0]["drop"]], facecolors="none", edgecolors="k", s=140, lw=1.2, zorder=7)
            a0.set_xlabel("mean deviation from the nominal command (normalised)")
            a0.set_ylabel(f"ball-drop rate ({split})")
            a0.set_title(f"{setup}: failure vs deviation ({split}); rings = VAL-chosen points", fontsize=9)
            a0.grid(alpha=0.25)
            a0.legend(fontsize=7)
            # --- tau curve
            a1 = ax[1]
            for arm in arms:
                pts = sorted((a for a in A if a["spec"].startswith("F3") and F.parse_spec(a["spec"])[1].get("arm") == arm and
                              np.isfinite(F.parse_spec(a["spec"])[1].get("tau", 0))), key=lambda a: F.parse_spec(a["spec"])[1].get("tau", 0))
                if not pts:
                    continue
                c = COL["F3Rc"] if arm.startswith("R") else COL["F3N0"]
                tx = [F.parse_spec(p["spec"])[1].get("tau", 0) for p in pts]
                a1.plot(tx, [p["drop"] for p in pts], "-o", c=c, ms=5, label=f"{arm} drop rate")
                a1.plot(tx, [p["interv"] for p in pts], "--s", c=c, ms=4, alpha=0.7, label=f"{arm} intervention rate")
            for a in A:
                if a["spec"] == "F0":
                    a1.axhline(a["drop"], c=COL["F0"], lw=1, ls=":", label="F0 drop rate")
            a1.set_xlabel("tau (apply u_nom while V(next latent under u_nom) >= tau)")
            a1.set_ylabel("rate")
            a1.set_title(f"{setup}: F3 threshold curve ({split})", fontsize=9)
            a1.grid(alpha=0.25)
            a1.legend(fontsize=7)
            fig.tight_layout()
            p = outdir / f"pareto_{split}_{setup}.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            made.append(str(p))
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "table", "plots", "all"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--dev-tol", type=float, default=0.15)
    a = ap.parse_args()
    if a.cmd in ("select", "all"):
        ch, notes, p = select(a.out, a.tag, a.tol, a.dev_tol)
        for k, v in ch.items():
            print(f"{k:24s} {v['spec']:44s} VAL drop {v['val']['drop']:.3f} dev {v['val']['dev']:.4f} interv {v['val']['interv']:.3f}", flush=True)
        for n in notes:
            print("NOTE", n, flush=True)
        print("wrote", p, flush=True)
    if a.cmd in ("table", "all"):
        agg = aggregate(load_runs(a.out, a.tag, a.split))
        txt = fmt_table(agg)
        print(txt, flush=True)
        (Path(a.out) / a.tag / f"table_{a.split}.md").write_text(txt + "\n")
        (Path(a.out) / a.tag / f"table_{a.split}.json").write_text(json.dumps(agg, indent=1, default=float))
    if a.cmd in ("plots", "all"):
        for p in plots(a.out, a.tag):
            print("plot", p, flush=True)


if __name__ == "__main__":
    main()
