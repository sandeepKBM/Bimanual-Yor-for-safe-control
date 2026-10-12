"""report.py -- aggregate trials_single.jsonl / trials_mix.jsonl into markdown tables. SIMULATION ONLY."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KINDS = ["ball", "item", "box", "block", "bottle", "mug", "L", "cup", "dish", "tray"]
METH = ["pca_box", "ubox", "hz2", "hz3", "hz4", "split_pa", "split_q", "split_pa_fs", "split_q_fs", "ubox+p", "hz2+p", "split_pa_fs+p"]


def _load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def _m(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(x.mean()) if len(x) else float("nan")


def _f(v, w=6, d=2):
    return f"{v:{w}.{d}f}" if np.isfinite(v) else " " * (w - 3) + "nan"


def table(rows, cols, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def collect(trials, view, cond, key="kind"):
    d = defaultdict(lambda: defaultdict(list))
    for t in trials:
        if t["view"] != view or t["cond"] != cond:
            continue
        for m, v in t.get("methods", {}).items():
            d[(t[key], m)]["_n"].append(1)
            for k, x in v.items():
                d[(t[key], m)][k].append(x)
        if t.get("hull_ref"):
            for k, x in t["hull_ref"].items():
                d[(t[key], "hull(P)")][k].append(x)
            d[(t[key], "hull(P)")]["_n"].append(1)
    return d


def make_report(out_dir, tag=""):
    out_dir = Path(out_dir)
    S = _load(out_dir / f"trials_single{tag}.jsonl")
    M = _load(out_dir / f"trials_mix{tag}.jsonl")
    L = []
    if S:
        views = sorted({t["view"] for t in S})
        conds = sorted({t["cond"] for t in S})
        ns = len({t["seed"] for t in S})
        L.append(f"# perception_zono study ({len(S)} single-object trials, {ns} seeds per cell)\n")
        # coverage / points
        L.append("## view coverage (fraction of the true surface within 6 mm of an observed point) and #points, GT segments\n")
        rows = []
        for k in KINDS:
            r = [k]
            for v in views:
                for c in ["nominal"]:
                    tt = [t for t in S if t["kind"] == k and t["view"] == v and t["cond"] == c]
                    r.append(f"{_m([t['coverage'] for t in tt]):.2f} / {_m([t['n_pts'] for t in tt]):.0f}")
            rows.append(r)
        L.append("cond=nominal; cell = coverage / n_pts after voxel+outlier filter\n")
        L.append(table(rows, None, ["kind"] + views))
        L.append("")
        # fail counts
        nf = defaultdict(int)
        for t in S:
            if t.get("fail"):
                nf[(t["kind"], t["view"], t["cond"])] += 1
        if nf:
            L.append("trials with < 8 points (no fit possible): " + "; ".join(f"{k[0]}/{k[1]}/{k[2]}:{v}" for k, v in sorted(nf.items())) + "\n")
        # headline tables
        for view in views:
            for cond in conds:
                d = collect(S, view, cond)
                L.append(f"## per kind x method  |  view={view}  cond={cond}\n")
                L.append("contain = true-surface fraction inside; vis/hid = visible / hidden part; oc_hull = vol(fit)/vol(convex hull of truth); "
                         "clear = signed margin error d_fit - d_true (mm; >0 unsafe), unsafe = share of queries with error > 2 mm; "
                         "inside = Inside(tray footprint) margin error (mm; >0 unsafe); ms = fit time\n")
                rows = []
                for k in KINDS:
                    for m in ["hull(P)"] + METH:
                        e = d.get((k, m))
                        if not e:
                            continue
                        n = len(e["_n"])
                        rows.append([k, m, str(n), _f(_m(e["contain"]), 5, 3), _f(_m(e.get("contain_vis", [])), 5, 3), _f(_m(e.get("contain_hid", [])), 5, 3),
                                     _f(_m(e.get("leak_p95_mm", [])), 5, 1), _f(_m(e["oc_hull"]), 5, 2), _f(_m(e["oc_true"]), 6, 2),
                                     _f(_m(e.get("clear_mean_mm", [])), 6, 1), _f(_m(e.get("clear_mae_mm", [])), 5, 1), _f(_m(e.get("clear_unsafe", [])), 4, 2),
                                     _f(_m(e.get("inside_err_mm", [])), 6, 1), _f(_m(e.get("n_parts", [])), 3, 1), _f(_m(e.get("ms", [])), 6, 1)])
                L.append(table(rows, None, ["kind", "method", "n", "contain", "vis", "hid", "leak95mm", "oc_hull", "oc_true", "clear_mean", "clear_mae",
                                            "unsafe", "inside", "parts", "ms"]))
                L.append("")
        # averaged across kinds, by view / cond
        L.append("## mean over kinds (ball box block bottle mug L cup dish; item and tray excluded) per view, cond, method\n")
        kk = ["ball", "box", "block", "bottle", "mug", "L", "cup", "dish"]
        rows = []
        for view in views:
            for cond in conds:
                d = collect(S, view, cond)
                for m in ["hull(P)"] + METH:
                    vals = defaultdict(list)
                    for k in kk:
                        e = d.get((k, m))
                        if e:
                            for q in ["contain", "contain_vis", "oc_hull", "clear_mean_mm", "clear_unsafe", "inside_err_mm", "ms"]:
                                if q in e:
                                    vals[q].append(_m(e[q]))
                    if vals:
                        rows.append([view, cond, m] + [_f(_m(vals[q]), 6, 2) for q in ["contain", "contain_vis", "oc_hull", "clear_mean_mm", "clear_unsafe", "inside_err_mm", "ms"]])
        L.append(table(rows, None, ["view", "cond", "method", "contain", "vis", "oc_hull", "clear_mean", "unsafe", "inside", "ms"]))
        L.append("")
        # timing of the pipeline
        L.append("## pipeline cost (ms, mean over trials): ray cast, voxel+outlier filter\n")
        rows = []
        for view in views:
            tt = [t for t in S if t["view"] == view]
            rows.append([view, _f(_m([t["t_cap"] for t in tt]), 6, 0), _f(_m([t["t_prep"] for t in tt]), 6, 0), _f(_m([t["n_raw"] for t in tt]), 8, 0)])
        L.append(table(rows, None, ["view", "cast ms", "prep ms", "raw pts on object"]))
        L.append("")
    if M:
        L.append(f"# mixed scenes ({len(M)} trials)\n")
        L.append("## segmentation (object-level: recovered = best cluster IoU >= 0.5)\n")
        rows = []
        for sc in sorted({t["scene"] for t in M}):
            for view in sorted({t["view"] for t in M}):
                for cond in ["ideal", "nominal", "n3mm"]:
                    for seg in ["dbscan_xyz", "dbscan_xyzn", "dbscan_xyz_kin"]:
                        tt = [t["segs"][seg] for t in M if t["scene"] == sc and t["view"] == view and t["cond"] == cond]
                        if not tt:
                            continue
                        sm = [x["seg"] for x in tt if x["seg"]]
                        rows.append([sc, view, cond, seg, _f(_m([s["recovered"] / max(s["n_gt"], 1) for s in sm]), 5, 2),
                                     _f(_m([s["n_clusters"] - s["n_gt"] for s in sm]), 5, 1), _f(_m([s["mean_iou"] for s in sm]), 5, 2),
                                     _f(_m([s["contamination"] for s in sm]), 5, 2), _f(_m([s["tray_leak"] for s in sm]), 5, 3), _f(_m([x["t_seg_ms"] for x in tt]), 6, 0)])
        L.append(table(rows, None, ["scene", "view", "cond", "seg", "recovered", "n_cl - n_gt", "mean IoU", "contam", "tray leak", "ms"]))
        L.append("")
        L.append("## end-to-end per-object fit from GT vs DBSCAN segments (mean over objects with >= 8 points); pair margin d_fit - d_true (mm)\n")
        rows = []
        for sc in sorted({t["scene"] for t in M}):
            for view in ["head_aim", "head_aim+wrist"]:
                for cond in ["nominal", "n3mm"]:
                    for seg in ["gt", "dbscan_xyz", "dbscan_xyz_kin"]:
                        for mname in ["ubox", "hz2", "split_pa_fs"]:
                            cont, oc, nfail, nobj, pe, pcol = [], [], 0, 0, [], []
                            for t in M:
                                if t["scene"] != sc or t["view"] != view or t["cond"] != cond:
                                    continue
                                r = t["segs"][seg]
                                for o, v in r["objs"].items():
                                    nobj += 1
                                    if "fail" in v:
                                        nfail += 1
                                    else:
                                        cont.append(v["methods"][mname]["contain"])
                                        oc.append(v["methods"][mname]["oc_hull"])
                                for p in r["pairs"].values():
                                    if mname in p["d_fit"]:
                                        pe.append((p["d_fit"][mname] - p["d_true"]) * 1e3)
                                        pcol.append(float(p["d_fit"][mname] <= 1e-9 and p["d_true"] > 0.002))
                            rows.append([sc, view, cond, seg, mname, f"{nfail}/{nobj}", _f(_m(cont), 5, 3), _f(_m(oc), 5, 2), _f(_m(pe), 6, 1), _f(_m(pcol), 5, 2)])
        L.append(table(rows, None, ["scene", "view", "cond", "seg", "method", "no-pts", "contain", "oc_hull", "pair err mm", "false-collision"]))
        L.append("")
    p = out_dir / f"report{tag}.md"
    p.write_text("\n".join(L))
    print("wrote", p, len("\n".join(L)), "chars")
