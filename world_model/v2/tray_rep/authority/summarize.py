"""summarize.py -- compact table of robot_authority.json for the report (usable poses = joint margin >= 0.1 rad)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OUT = Path("/common/users/ss5772/bimanual_data/tray_rep/authority")


def main():
    r = json.loads((OUT / "robot_authority.json").read_text())
    P = r["poses"]
    print("pose              use  margin  gravJ(arm-only,max)  static_tau_max(ls) | a_lim ls: x+  y+  -z  +z | a_lim lp: horiz_min | alpha_xy_min ls/lp | v_lim_h  w_lim_xy  w_lim_z")
    agg = {}
    for k, v in P.items():
        if not v.get("reachable"):
            continue
        g = np.abs(np.array(v["arm_only_gravity_tau"])).max()
        ls, lp = v["real8_ls"], v["real8_lp"]
        vh = min(x for kk, x in v["v_lim"].items() if kk != "z")
        wxy = min(v["w_lim"]["x"], v["w_lim"]["y"])
        print(f"{k:16s} {'Y' if v['usable'] else 'n'}  {v['joint_margin_rad']:5.2f}   {g:5.1f}   {np.abs(ls['static_tau']).max():5.1f}   |   "
              f"{ls['a_lim']['h0']:5.1f} {ls['a_lim']['h90']:5.1f} {ls['a_lim']['-z']:5.1f} {ls['a_lim']['+z']:5.1f} |  {lp['a_horiz_min']:5.1f} ({ls['a_horiz_min']:4.1f}) | {ls['alpha_xy_min']:5.0f}/{lp['alpha_xy_min']:5.0f} | {vh:4.2f}  {wxy:4.2f}  {v['w_lim']['z']:4.2f}")
        if v["usable"]:
            for st in (k.split("/")[0], "all"):
                d = agg.setdefault(st, dict(g=[], s=[], ah_ls=[], ah_lp=[], al_ls=[], al_lp=[], vh=[], wxy=[], a100=[]))
                d["g"].append(g); d["s"].append(ls["static_s"]); d["ah_ls"].append(ls["a_horiz_min"]); d["ah_lp"].append(lp["a_horiz_min"])
                d["al_ls"].append(ls["alpha_xy_min"]); d["al_lp"].append(lp["alpha_xy_min"]); d["vh"].append(vh); d["wxy"].append(wxy)
                d["a100"].append(v["mjcf100_ls"]["a_horiz_min"])
    print("\nusable poses: min / median / max")
    for st, d in agg.items():
        print(st, "n=", len(d["g"]))
        for kk, vals in d.items():
            print(f"   {kk:6s} {np.min(vals):7.2f} {np.median(vals):7.2f} {np.max(vals):7.2f}")
    n_reach = sum(1 for v in P.values() if v.get("reachable")); n_use = sum(1 for v in P.values() if v.get("usable"))
    print("\nreachable", n_reach, "usable", n_use, "of", len(P))
    print("gravity refs (arm-only, j1..j6, Nm):")
    for k, v in r["gravity_refs"].items():
        print("  ", k, np.round(v, 1).tolist())


if __name__ == "__main__":
    main()
