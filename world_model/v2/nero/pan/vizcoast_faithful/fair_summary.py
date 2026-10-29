"""Same-worlds table: every executed plan x condition (exec_raw/ dirs of run_baseline), plus our plan's runs.
    python -m world_model.v2.nero.pan.vizcoast_faithful.fair_summary OUT.md DIR[:label] ...
Each DIR is a run_baseline --out (exec_raw/<plan>__<world>.json) or observe_eval closed --out (raw/<world>_<noise>.json).
"""
import collections
import json
import sys
from pathlib import Path


def rows(d):
    d = Path(d)
    out = collections.defaultdict(list)
    for p in sorted((d / "exec_raw").glob("*.json")):
        plan, world = p.stem.split("__")
        out[plan].append((world, json.loads(p.read_text())))
    for p in sorted((d / "raw").glob("*.json")):                      # observe_eval closed: our plan, one row per noise level
        world, noise = p.stem.split("_", 1)
        out[f"ours_f0.650_p0.070 (observe {noise})"].append((world, json.loads(p.read_text())))
    return out


def main(dst, *dirs):
    lines = ["| condition | plan | vertices held | interior held | total | failure categories |", "|---|---|---|---|---|---|"]
    for spec in dirs:
        d, _, label = spec.partition(":")
        from ..vizcoast.evidence import category
        for plan, rs in rows(d).items():
            V = [r for w, r in rs if w.startswith("V")]
            I = [r for w, r in rs if w.startswith("I")]
            fails = collections.Counter(category(r) for _, r in rs if not r.get("held"))
            n = sum(bool(r.get("held")) for _, r in rs)
            lines.append(f"| {label or Path(d).name} | {plan} | {sum(bool(r.get('held')) for r in V)}/{len(V)} | "
                         f"{sum(bool(r.get('held')) for r in I)}/{len(I)} | {n}/{len(rs)} | {dict(fails)} |")
    Path(dst).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
