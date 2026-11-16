#!/usr/bin/env python
"""Aggregate f2_harness result JSONs into one comparison table (markdown)."""
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("/common/home/ss5772/.claude/jobs/f1580deb/tmp/f2_eval")
ORDER = ["flmpc", "nominal", "analytic_hard", "analytic_blend", "analytic_qp",
         "learned_hard", "learned_blend", "learned_qp", "learned_qp_logit",
         "learned_hard@tight", "learned_blend@tight", "learned_qp@tight"]


def main(tags):
    rows = []
    for tag in tags:
        p = OUT_DIR / f"results_{tag}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if not r.get("ok"):
                print("FAILED", tag, r.get("mode"), r.get("seed"), r.get("err"))
                continue
            if tag.endswith("tight"):
                r["mode"] = r["mode"] + "@tight"
            elif tag not in ("base", "valid", "deploy"):
                r["mode"] = r["mode"] + "@" + tag
            rows.append(r)
    hdr = ("| mode | n | jerk_rms | SPARC | collide | h_min worst / mean [m] | grip tilt max [deg] "
           "| cup tilt max (start) [deg] | engaged % | switches | p99 |dv_cmd| [rad/s/step] | filter ms |")
    print(hdr)
    print("|" + "---|" * 12)
    extra = sorted({r["mode"] for r in rows} - set(ORDER))
    modes = [m for m in ORDER + extra if any(r["mode"] == m for r in rows)]
    for m in modes:
        g = [r for r in rows if r["mode"] == m]
        f = lambda k: np.nanmean([r[k] for r in g])  # noqa: E731
        print(f"| {m} | {len(g)} | {f('jerk_rms'):.0f} | {f('sparc'):.2f} | "
              f"{sum(r['collision'] for r in g)}/{len(g)} | "
              f"{min(r['h_min'] for r in g):+.3f} / {f('h_min'):+.3f} | {f('grip_tilt_max_deg'):.1f} | "
              f"{f('cup_tilt_max_deg'):.1f} ({f('cup_tilt0_deg'):.1f}) | "
              f"{100 * f('engaged_frac'):.0f} | {f('switches'):.0f} | {f('dvcmd_p99'):.3f} | {f('filter_ms'):.1f} |")
    print("\nper-seed jerk / collide / h_min:")
    for m in modes:
        g = sorted([r for r in rows if r["mode"] == m], key=lambda r: r["seed"])
        print(f"  {m:22s} " + "  ".join(f"s{r['seed']}:{r['jerk_rms']:.0f}/{'C' if r['collision'] else '-'}/{r['h_min']:+.3f}" for r in g))


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "valid", "deploy", "deploy_tight", "aqp_e08a5", "aqp_e08a2"])
