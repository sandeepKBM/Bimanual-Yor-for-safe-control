"""baseline_scripted vs lib3 (segment pipeline) on the same tags.  Output: markdown table to stdout."""
import json, glob, collections, numpy as np, sys
from pathlib import Path
B = Path("world_model/v2/nero/results/pan_baseline_scripted_2026-10-29/raw")
L = Path("world_model/v2/nero/results/pan_segments_2026-10-29/lib3/raw")
SETS = dict(full=("I", "V"), iid=("R",), overload=("O",))     # O* covers O and OI
def load(d, pre):
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if p.stem.startswith(pre)}
print("| set | n | method | refused at start | task success | entered U (where) | errors |")
print("|---|---|---|---|---|---|---|")
mass = collections.defaultdict(list)
for name, pre in SETS.items():
    b = load(B, pre); l = load(L, pre); tags = sorted(set(b) & set(l))
    if not tags: continue
    bs = [b[t] for t in tags]; ls = [l[t] for t in tags]
    bU = collections.Counter(r["U_entered"]["kind"] for r in bs if r.get("U_entered"))
    print(f"| {name} | {len(tags)} | scripted baseline | {sum(bool(r.get('refused')) for r in bs)} | "
          f"{sum(bool(r.get('task_success')) for r in bs)} | {sum(bU.values())} {dict(bU)} | {sum('error' in r for r in bs)} |")
    lU = collections.Counter(s["nominal"]["viol"]["kind"] + "@" + s["seg"] for r in ls for s in r.get("segments", []) if s["nominal"]["viol"])
    lsucc = sum(bool(len(r.get("segments", [])) == 4 and r["segments"][-1]["nominal"].get("success")) for r in ls)
    print(f"| {name} | {len(tags)} | ours (segment pipeline, lib3) | {sum(bool(r.get('refused')) for r in ls)} | {lsucc} | "
          f"{sum(lU.values())} {dict(lU)} | {sum('error' in r for r in ls)} |")
    st = collections.Counter(r.get("stage") for r in bs)
    print(f"|  |  | baseline stages | {dict(st)} |  |  |  |")
    for r in bs:
        p = r.get("probe") or {}
        if p.get("contents_kg_est") is not None:
            mass[name].append((p["contents_kg_est"], p.get("contents_kg_true") or 0.0))
print()
print("| set | n probes | mass est error p50 / p95 / max [kg] (max over n is not a bound) |")
print("|---|---|---|")
for k, v in mass.items():
    e = np.abs([a - t for a, t in v])
    print(f"| {k} | {len(v)} | {np.median(e):.3f} / {np.percentile(e, 95):.3f} / {e.max():.3f} |")
