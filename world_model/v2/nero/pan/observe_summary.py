"""Tables for the observe_pose evaluation (E1 accuracy, E2 closed loop): python -m world_model.v2.nero.pan.observe_summary DIR"""
import collections
import json
import sys
from pathlib import Path

import numpy as np


def main(d):
    d = Path(d)
    lines = []
    f = d / "accuracy.jsonl"
    if f.exists():
        rows = [json.loads(x) for x in open(f)]
        err = [r for r in rows if "error" in r]
        lines += [f"E1 accuracy (n per level; errors: {len(err)})", "| noise | n | fit ok | pos p50/p95/max mm | yaw |res| p50/p95/max deg | "
                  "within 5 mm & 2 deg | within 5 mm | within 2 deg |", "|---|---|---|---|---|---|---|---|"]
        by = collections.defaultdict(list)
        for r in rows:
            if "noise" in r:
                by[r["noise"]].append(r)
        for nz, rs in by.items():
            ok = [r for r in rs if r["fit_ok"]]
            pm = np.array([r["residual_mm"] for r in ok])
            py = np.abs([r["residual_yaw_deg"] for r in ok])
            q = lambda a: "/".join(f"{v:.2f}" for v in (np.median(a), np.quantile(a, 0.95), a.max())) if len(a) else "-"
            lines.append(f"| {nz} | {len(rs)} | {len(ok)} | {q(pm)} | {q(py)} | {int(((pm <= 5) & (py <= 2)).sum())}/{len(ok)} | "
                         f"{int((pm <= 5).sum())} | {int((py <= 2).sum())} |")
    raw = d / "raw"
    if raw.exists():
        rs = [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))]
        lines += ["", "E2 closed loop, plan pinch 0.65 / push 0.07, FULL prereg box (V = 32 vertices, I = random interior)",
                  "| noise | vertices held | interior held | refused (no fit) | failure categories | failed runs: residual mm / deg |",
                  "|---|---|---|---|---|---|"]
        by = collections.defaultdict(list)
        for r in rs:
            by[r["noise"]].append(r)
        for nz, g in by.items():
            V = [r for r in g if r["tag"].startswith("V")]
            I = [r for r in g if r["tag"].startswith("I")]
            ref = sum("observe_pose" in (r.get("error") or "") for r in g)
            cats = collections.Counter(r["category"] for r in g if not r["held"])
            fails = [f"{r['tag'].split('_')[0]}:{(r.get('observe') or {}).get('residual_mm', float('nan')):.1f}/"
                     f"{(r.get('observe') or {}).get('residual_yaw_deg', float('nan')):+.1f}" for r in g if not r["held"]]
            lines.append(f"| {nz} | {sum(r['held'] for r in V)}/{len(V)} | {sum(r['held'] for r in I)}/{len(I)} | {ref} | "
                         f"{dict(cats)} | {' '.join(fails[:12])} |")
    txt = "\n".join(lines)
    (d / "SUMMARY.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1])
