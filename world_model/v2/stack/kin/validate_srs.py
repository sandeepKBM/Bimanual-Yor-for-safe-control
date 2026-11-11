"""validate_srs.py -- FK -> IK round trip for SRSArm.  SIMULATION ONLY, CPU.
   PYTHONPATH=. python -m world_model.v2.stack.kin.validate_srs --n 2000 --out srs_val.json
Per side: n random q inside the MJCF limits -> MuJoCo FK pose -> ik(pose, arm_angle(q)) -> errors of the best-matching branch,
recovery of q (some branch within 1e-6 rad), time per ik call.  Near-singular cases are reported as separate rows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .srs_ik import SRSArm

SCENE = "nero.democ.scene.compose() (NYU gripper, built in memory)"   # 11-07: was scenes/yor_nero_democ_nyu.mjcf, a generated file with
# worktree-absolute mesh paths that was never committed; same model (ee_grasp pose diff 1.0e-6 over 400 random q, XML float rounding).


def load_model():
    """the DemoC NERO scene with the NYU gripper (the only build), compiled in memory."""
    from world_model.v2.nero.democ import scene as DS
    return DS.compose()


def stats(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    return dict(n=int(x.size), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)), max=float(x.max()))


def run_side(m, side, n, seed):
    arm = SRSArm(m, side)
    rng = np.random.default_rng(seed)
    rec = []
    for _ in range(n):
        q = rng.uniform(arm.lo, arm.hi)
        T = arm.fk(q)
        psi = arm.arm_angle(q)
        _, _, R123, W = arm._fk_poe(q)
        e = R123 @ arm.u0
        v = (W - arm.S) / np.linalg.norm(W - arm.S)
        straight = float(np.linalg.norm(e - (e @ v) * v) / np.linalg.norm(arm.u0))      # 0 = elbow straight
        vref = float(np.degrees(np.arccos(abs(v @ arm.ref))))                             # S-W vs reference direction [deg]
        t0 = time.perf_counter()
        res = arm.ik_full(T, psi)
        dt = time.perf_counter() - t0
        if not res or not np.isfinite(psi):
            rec.append(dict(ok=False, straight=straight, vref=vref, dt=dt, nan=not np.isfinite(psi), nres=len(res)))
            continue
        best = min(res, key=lambda r: r["pos_err"] + 0.1 * r["rot_err"])
        dq = min(float(np.abs((r["q"] - q + np.pi) % (2 * np.pi) - np.pi).max()) for r in res)    # closest branch to the original q
        rec.append(dict(ok=True, pos=best["pos_err"], rot=best["rot_err"], dq=float(dq), straight=straight, vref=vref, dt=dt,
                        nres=len(res), q=q.tolist()))
    return rec


def summarize(rec):
    ok = [r for r in rec if r["ok"]]
    out = dict(n=len(rec), n_no_solution=len(rec) - len(ok),
               pos_err_m=stats([r["pos"] for r in ok]), rot_err_rad=stats([r["rot"] for r in ok]),
               recovery_rate_1e6=float(np.mean([r["dq"] < 1e-6 for r in ok])) if ok else None,
               dq_rad=stats([r["dq"] for r in ok]), ik_time_s=stats([r["dt"] for r in rec]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="srs_val.json")
    a = ap.parse_args()
    m = load_model()
    out = dict(scene=SCENE, n=a.n, seed=a.seed, sides={})
    for side in ("left", "right"):
        arm = SRSArm(m, side)
        rec = run_side(m, side, a.n, a.seed)
        sing = [r for r in rec if r["straight"] < 0.1]
        near_ref = [r for r in rec if r["vref"] < 5.0]
        rest = [r for r in rec if r["straight"] >= 0.1 and r["vref"] >= 5.0]
        worst = sorted([r for r in rec if r["ok"]], key=lambda r: -r["dq"])[:3]
        out["sides"][side] = dict(
            geometry=dict(shoulder_gap_m=arm.shoulder_gap, wrist_gap_m=arm.wrist_gap, upper_arm_m=float(np.linalg.norm(arm.u0)),
                          forearm_m=float(np.linalg.norm(arm.w0))),
            all=summarize(rec), regular=summarize(rest),
            elbow_near_straight_lt0p1=summarize(sing), SW_within_5deg_of_ref=summarize(near_ref),
            worst_dq=[dict(dq=r["dq"], pos=r["pos"], rot=r["rot"], q=r["q"]) for r in worst])
        print(side, json.dumps({k: out["sides"][side][k] for k in ("all", "regular", "elbow_near_straight_lt0p1")}, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
