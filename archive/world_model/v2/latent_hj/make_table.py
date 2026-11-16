"""make_table.py -- aggregate scenario_*.json rows into a markdown table.
  python -m world_model.v2.latent_hj.make_table results/scenario_smoke.json [more.json ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

COLS = [("spill", "spill", lambda v: f"{int(np.sum(v))}/{len(v)}"),
        ("phi_max_deg", "phi max [deg] mean", lambda v: f"{np.mean(v):.0f}"),
        ("collision", "collide", lambda v: f"{int(np.sum(v))}/{len(v)}"),
        ("h_min", "h_min worst [m]", lambda v: f"{np.min(v):+.3f}"),
        ("track_rms_cm", "track RMS [cm]", lambda v: f"{np.mean(v):.1f}"),
        ("final_err_cm", "final err [cm]", lambda v: f"{np.mean(v):.1f}"),
        ("jerk_rms", "jerk_rms", lambda v: f"{np.mean(v):.0f}"),
        ("sparc", "SPARC", lambda v: f"{np.nanmean(v):.2f}"),
        ("switches", "switches", lambda v: f"{np.mean(v):.1f}"),
        ("engaged_frac", "engaged %", lambda v: f"{100*np.mean(v):.0f}"),
        ("da_p99", "p99 |da| [rad/s^2/step]", lambda v: f"{np.mean(v):.1f}"),
        ("qp_infeas_steps", "QP infeas steps", lambda v: f"{np.mean(v):.1f}"),
        ("value_relaxed_steps", "value-relaxed steps", lambda v: f"{np.mean(v):.1f}"),
        ("ms_mean", "ctrl ms mean", lambda v: f"{np.mean(v):.0f}"),
        ("value_ms_max", "value ms max", lambda v: f"{np.max(v):.0f}")]


def main(paths):
    rows = [r for p in paths for r in json.load(open(p)) if r.get("ok")]
    bad = [r for p in paths for r in json.load(open(p)) if not r.get("ok")]
    modes = []
    for r in rows:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    print("| mode | n | " + " | ".join(c[1] for c in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for m in modes:
        g = [r for r in rows if r["mode"] == m]
        print(f"| {m} | {len(g)} | " + " | ".join(f(np.array([r[k] for r in g], float)) for k, _, f in COLS) + " |")
    print()
    for m in modes:
        g = sorted([r for r in rows if r["mode"] == m], key=lambda r: r["seed"])
        print(f"{m:12s} " + "  ".join(f"s{r['seed']}:{'S' if r['spill'] else '-'}{'C' if r['collision'] else '-'}"
                                        f"/{r['phi_max_deg']:.0f}/{r['jerk_rms']:.0f}" for r in g))
    if bad:
        print(f"\nFAILED jobs: {[(r['seed'], r['mode'], r['err'][:80]) for r in bad]}")


if __name__ == "__main__":
    main(sys.argv[1:])
