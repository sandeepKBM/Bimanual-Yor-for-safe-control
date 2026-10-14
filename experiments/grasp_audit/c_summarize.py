"""Summarize Part C stress-test JSONs into onset tables. usage: python c_summarize.py C_*.json"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def onset(tests, key, thr, meas):
    """first ladder level (ascending) whose metric > thr; returns 'lvl(meas)' or '>max' (+ '*' if rig not ok)."""
    for t in sorted(tests, key=lambda r: r["level"]):
        if t[key] > thr or t["dropped"]:
            tag = "" if t["rig_ok"] else "*"
            return f"{t['level']}({t[meas]:.0f}){tag}"
    mx = max(tests, key=lambda r: r["level"])
    return f">{mx['level']}"


def main(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        g = d["grasp"]
        name = p.split("C_")[-1].replace(".json", "")
        fam = defaultdict(list)
        hold = None
        for t in d["tests"]:
            if t["test"] == "hold":
                hold = t
            else:
                fam[t["test"]].append(t)
        print(f"\n### {name}: Fn jaw7/8 after lift = {g['after_lift_fn'][0]:.2f}/{g['after_lift_fn'][1]:.2f} N, "
              f"finger act = {g['act_force']}, finger q = {[round(x*1e3,1) for x in g['finger_q']]} mm, "
              f"grasp pen = {g['pen_mm']:.2f} mm, lift(6cm) slip = {g['lift_slip_mm']:.2f} mm / {g['lift_rot_deg']:.2f} deg, "
              f"obj mass = {g['obj_mass_kg']*1e3:.1f} g, weight/(mu*Fn_total) = "
              f"{g['obj_mass_kg']*9.81/(d['variant']['pad_mu']*sum(g['after_lift_fn'])+1e-9):.3f}")
        if hold:
            print(f"   hold 2s: slip {hold['slip_mm']:.2f} mm, rot {hold['rot_deg']:.2f} deg, cone util max {hold['util_max']:.2f}, "
                  f"pen {hold['pen_mm']:.2f} mm")
        print("   family      | slip>2mm @lvl(meas) | slip>5mm | rot>2deg | drop | max util@lvl<=onset | chatter toggles (max)")
        for f, ts in fam.items():
            meas = "a_peak" if f.startswith("lin") else ("al_peak" if f.startswith("rot") else "level")
            print(f"   {f:12s}| {onset(ts, 'slip_mm', 2.0, meas):19s} | {onset(ts, 'slip_mm', 5.0, meas):8s} | "
                  f"{onset(ts, 'rot_deg', 2.0, meas):8s} | {onset([dict(t, slip_mm=0, dropped=t['dropped']) for t in ts], 'slip_mm', 1e9, meas):6s} | "
                  f"{max(t['util_max'] for t in ts):.2f} | {max(t['toggles'] for t in ts)}")
            rows.append((name, f, ts))
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
