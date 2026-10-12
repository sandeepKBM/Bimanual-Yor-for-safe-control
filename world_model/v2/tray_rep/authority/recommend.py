"""recommend.py -- turn the per-pose torque limits into a scale for the 85-element fallback set.  SIMULATION ONLY.

For every reachable pose in robot_authority.json and each tilt-rate slew alpha, find the largest isotropic accel scale la
(accel = la * 5 m/s^2 on the 8 horizontal directions, tilt rate w = sw * 2 rad/s about x/y) such that EVERY element of
the full-magnitude set is jointly feasible (torque inside the cap, gravity included) in both phases of the action:
  phase 1: the tilt rate ramps up: a = la*a_i, alpha_ang = alpha along w   (omega = 0)
  phase 2: the tilt rate is held:   a = la*a_i, omega = sw*w_i              (alpha_ang = 0, gyro/centripetal terms on)
(half-magnitude accels are convex combinations of 0 and full, so full-magnitude elements are the binding ones.)
Also checks joint-rate limits (URDF 5 rad/s, wrist j6 3 rad/s) at omega = sw*2 and at the speed v = la*5*0.2 reached in a 0.2 s hold.
  python -m world_model.v2.tray_rep.authority.recommend
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
from world_model.v2.testbeds.b_tray_ball import probe as P  # noqa: E402

U85 = P.U85
FULL = [u for u in U85 if np.hypot(u[0], u[1]) > 4.9 or (np.hypot(u[0], u[1]) < 1e-9 and np.hypot(u[3], u[4]) > 0)]
ALPHAS = [10.0, 20.0, 40.0]
SW = [1.0]


def feasible(cx, cap, mode, la, sw, alpha):
    for u in FULL:
        a = la * u[:3]
        w = sw * u[3:]
        nw = np.linalg.norm(w)
        if nw > 0:
            if cx.solve(cap, a=a, al=alpha * w / nw, mode=mode)[0] > 1.0:      # phase 1
                return False
            if cx.solve(cap, a=a, w=w, mode=mode)[0] > 1.0:                    # phase 2
                return False
        elif cx.solve(cap, a=a, mode=mode)[0] > 1.0:
            return False
    return True


def best_scale(cx, cap, mode, sw, alpha):
    if not feasible(cx, cap, mode, 0.0, sw, alpha):
        return 0.0
    if feasible(cx, cap, mode, 1.0, sw, alpha):
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > 0.01:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(cx, cap, mode, mid, sw, alpha) else (lo, mid)
    return lo


def work(args):
    key, q, o, style = args
    rig = RA.Rig()
    cx = RA.Ctx(rig, np.array(q), o, style)
    out = {}
    for mode in ("lp", "ls"):
        for sw in SW:
            for al in ALPHAS:
                out[f"{mode}|sw{sw}|a{al}"] = best_scale(cx, RA.CAPS["real8"], mode, sw, al)
    # also: la for the 100 Nm cap at sw=1, alpha=40 (expect 1.0)
    out["mjcf100|sw1.0|a40.0"] = best_scale(cx, RA.CAPS["mjcf100"], "ls", 1.0, 40.0)
    return key, out


def main():
    res = json.loads((RA.OUT / "robot_authority.json").read_text())
    tasks = [(k, v["q"], v["tray_o"], k.split("/")[0]) for k, v in res["poses"].items() if v.get("reachable") and v.get("usable")]
    with Pool(8) as pool:
        out = dict(pool.map(work, tasks, chunksize=1))
    (RA.OUT / "recommend.json").write_text(json.dumps(out, indent=1))
    keys = sorted(next(iter(out.values())).keys())
    print("scale la (accel x 5 m/s^2): min / p25 / median over reachable poses, by style")
    for style in ("side", "front", "all"):
        for k in keys:
            vals = np.array([v[k] for kk, v in out.items() if style == "all" or kk.startswith(style)])
            print(f"{style:5s} {k:22s} n={len(vals):2d} min={vals.min():.2f} p25={np.percentile(vals, 25):.2f} med={np.median(vals):.2f}")


if __name__ == "__main__":
    main()
