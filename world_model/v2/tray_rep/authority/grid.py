"""grid.py -- (accel scale la, tilt slew alpha) feasibility grid over the reachable poses, plus what-ifs.  SIMULATION ONLY.
Feasible = every full-magnitude element of the 85-set (accel la*5 m/s^2, tilt rate 2 rad/s) is inside the torque cap in both
phases (see recommend.py).  Output: fraction of reachable poses that are feasible, and the largest feasible la per pose.
What-ifs (all with mode 'ls' = plain load sharing):  payload mass x0.5 / x0 ;  cap x1.5 / x2 (e.g. torque from the PD term).
  python -m world_model.v2.tray_rep.authority.grid
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from world_model.v2.tray_rep.authority import robot_authority as RA  # noqa: E402
from world_model.v2.tray_rep.authority import recommend as R  # noqa: E402

LAS = [0.3, 0.5, 0.65, 1.0]
ALS = [5.0, 10.0, 20.0]
CAPSC = {"real8": RA.CAPS["real8"], "real8x1.5": 1.5 * RA.CAPS["real8"], "real8x2": 2.0 * RA.CAPS["real8"]}


def work(args):
    key, q, o, style = args
    rig = RA.Rig()
    cx = RA.Ctx(rig, np.array(q), o, style)
    out = {}
    for mode in ("lp", "ls"):
        for al in ALS:
            for la in LAS:
                out[f"{mode}|a{al}|la{la}"] = bool(R.feasible(cx, RA.CAPS["real8"], mode, la, 1.0, al))
    # what-ifs (ls, alpha 10): largest la
    m0, I0 = RA.MASS, RA.I_COM.copy()
    for pm in (0.5, 0.0):
        RA.MASS = m0 * pm; RA.I_COM = I0 * pm
        out[f"whatif_mass{pm}|ls|a10"] = R.best_scale(cx, RA.CAPS["real8"], "ls", 1.0, 10.0)
    RA.MASS, RA.I_COM = m0, I0
    out["base|ls|a10"] = R.best_scale(cx, RA.CAPS["real8"], "ls", 1.0, 10.0)
    for cn in ("real8x1.5", "real8x2"):
        out[f"whatif_{cn}|ls|a10"] = R.best_scale(cx, CAPSC[cn], "ls", 1.0, 10.0)
    return key, out


def main():
    res = json.loads((RA.OUT / "robot_authority.json").read_text())
    tasks = [(k, v["q"], v["tray_o"], k.split("/")[0]) for k, v in res["poses"].items() if v.get("reachable") and v.get("usable")]
    with Pool(8) as pool:
        out = dict(pool.map(work, tasks, chunksize=1))
    (RA.OUT / "grid.json").write_text(json.dumps(out, indent=1))
    print("fraction of reachable poses where ALL 85 elements are feasible at real caps (la = accel scale)")
    for style in ("front", "side", "all"):
        ks = [k for k in out if style == "all" or k.startswith(style)]
        print(style, f"(n={len(ks)})")
        for mode in ("lp", "ls"):
            for al in ALS:
                print(f"  {mode} alpha={al:>4}: " + "  ".join(f"la={la}: {np.mean([out[k][f'{mode}|a{al}|la{la}'] for k in ks]):.2f}" for la in LAS))
        for w in ("base|ls|a10", "whatif_mass0.5|ls|a10", "whatif_mass0.0|ls|a10", "whatif_real8x1.5|ls|a10", "whatif_real8x2|ls|a10"):
            v = np.array([out[k][w] for k in ks])
            print(f"  {w:26s} largest la: min={v.min():.2f} p25={np.percentile(v, 25):.2f} median={np.median(v):.2f}")


if __name__ == "__main__":
    main()
