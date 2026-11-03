"""max object/mug tilt and spill over INTERIOR sampled worlds (cert run + ungated baseline, a = 0.5) vs the CORNER max (tilt sweep, a = 0.5)."""
import json
from collections import defaultdict

D = "/data/local/ss5772/contents/"
inner = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
for l in open(D + "carry_cert_fc_stiff_tray33_2026-11-02/trials.jsonl"):
    r = json.loads(l)
    if "error" in r:
        continue
    k = (r["key"].split("|")[0], tuple(r["cell"]))
    g = inner[k]
    for s in r["segments"]:
        n = s["nominal"]
        g[0] = max(g[0], n["obj_tilt_max"]); g[1] = max(g[1], n["cup_tilt_max"]); g[2] = max(g[2], n["spill_ratio"])
    g[3] += 1
for l in open(D + "baseline_ungated_fc_stiff_2026-11-02/trials.jsonl"):
    r = json.loads(l)
    if "error" in r:
        continue
    k = (r["task"], tuple(r["cell"]))
    g = inner[k]
    for s in r["segments"]:
        g[0] = max(g[0], s["obj_tilt_max"]); g[1] = max(g[1], s["cup_tilt_max"]); g[2] = max(g[2], s["spill_ratio"])
    g[3] += 1
corner = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
for l in open(D + "tilt_sweep_2026-11-02/runs.jsonl"):
    r = json.loads(l)
    if "error" in r or r["a"] != 0.5:
        continue
    g = corner[(r["task"], tuple(r["cell"]))]
    for s in r["segments"].values():
        g[0] = max(g[0], s["obj_tilt_max"]); g[1] = max(g[1], s["cup_tilt_max"]); g[2] = max(g[2], s["spill_ratio"])
    g[3] += 1
bad = 0
for k in sorted(corner):
    i, c = inner.get(k), corner[k]
    if not i:
        continue
    flag = "" if (i[0] <= c[0] + 1e-9 and i[2] <= c[2] + 1e-9) else "  <-- interior exceeds corners"
    bad += bool(flag)
    print(f"{k[0]:6s} {str(k[1]):12s} interior n={i[3]:2d} tilt {i[0]:5.2f} spill {i[2]:4.2f} | corners n={c[3]:2d} tilt {c[0]:5.2f} spill {c[2]:4.2f}{flag}")
print("cells where interior exceeds corners:", bad)
