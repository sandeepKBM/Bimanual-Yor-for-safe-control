#!/usr/bin/env python
"""fit_physics.py -- models (a0) static tilt, (a1) effective gravity, (b1) world
pendulum, (b2) cup-frame oscillator; (b*) parameters FIT on TRAIN trajectories
(grid search maximizing train causal AUROC at H). Saves per-step scores for all
episodes + fitted params to data/scores_phys.npz. SIM DATA ONLY.

  python experiments/slosh/fit_physics.py --data experiments/slosh/data/rig_v1.pkl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slosh_models as sm  # noqa: E402
from data_prep import load_rig, labels  # noqa: E402
from evaluate import evaluate, fmt_row, HEADER  # noqa: E402

W_GRID = [1, 2, 4, 6, 9, 13, 18, 25, 35, 50, 70]
Z_GRID = [0.05, 0.15, 0.3, 0.7, 1.0, 1.5, 2.5, 4.0]
K_GRID = [0.0, 0.01, 0.03]


def train_auc(score, D, H, sub=4):
    """fit objective = mean of train per-step causal AUROC and train EPISODE
    AUROC (the constraint semantics: max risk along a trajectory)."""
    tr = ~D["test"]
    valid_full = labels(D["onset"], D["k0"], D["T"], H)[1]
    es = np.where(valid_full, score, -np.inf).max(-1)
    ep = sm.auroc(es[tr], (D["onset"] >= 0)[tr])
    y, v = labels(D["onset"], D["k0"], D["T"], H)
    s = score[tr][:, ::sub]; yy = y[tr][:, ::sub]; vv = v[tr][:, ::sub]
    return 0.5 * (sm.auroc(s[vv], yy[vv]) + ep)


def fit(fn, K, D, H):
    best = (-1, None)
    for w, z in itertools.product(W_GRID, Z_GRID):
        phi = fn(K, w, z)
        for kap in K_GRID:
            a = train_auc(sm.risk(phi, kap), D, H)
            if a > best[0]:
                best = (a, (w, z, kap))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "rig_v1.pkl"))
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--out", default=str(HERE / "data" / "scores_phys.npz"))
    a = ap.parse_args()
    D = load_rig(a.data)
    K = sm.kinematics(D["pos"], D["n"])
    scores, params = {}, {}
    scores["a0_static_tilt"] = K["theta0"]
    scores["a1_eff_gravity"] = K["theta_eff"]
    # ladder rung2 default physics omega (Abramson) for reference, zeta 0.15, kappa 0
    scores["b2_cup_osc_physics_w24.6"] = sm.pendulum_cup(K, 24.6, 0.15)
    for name, fn in (("b1_world_pendulum", sm.pendulum_world), ("b2_cup_oscillator", sm.pendulum_cup),
                     ("b3_world_lti", sm.lti_world)):
        t0 = time.time()
        auc, (w, z, kap) = fit(fn, K, D, a.H)
        params[name] = dict(w=w, zeta=z, kappa=kap, train_auroc=auc)
        scores[name] = sm.risk(fn(K, w, z), kap)
        print(f"fit {name}: w={w} zeta={z} kappa={kap} train AUROC={auc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(HEADER)
    res = {}
    for k, s in scores.items():
        m = evaluate(s, D, H=a.H); res[k] = m
        print(fmt_row(k, m), flush=True)
    # HARD band: test episodes whose peak static tilt is in [60,110] deg (where a
    # static-tilt rule is ambiguous) and the dynamic families
    pk = np.degrees(np.max(np.where(labels(D["onset"], D["k0"], D["T"], a.H)[1], K["theta0"], 0), -1))
    subsets = {"HARD tilt 60-110deg": (pk > 60) & (pk < 110)}
    for f in ("mixed", "shake", "tilt"):
        subsets[f"family={f}"] = D["family"] == f
    for sname, msk in subsets.items():
        te = D["test"] & msk
        print(f"\n[{sname}] n_test={te.sum()} spilled={(te & (D['onset']>=0)).sum()}")
        print(HEADER)
        for k, s in scores.items():
            print(fmt_row(k, evaluate(s, D, H=a.H, mask=msk)), flush=True)
    for k in ("b1_world_pendulum", "b2_cup_oscillator", "b3_world_lti", "a1_eff_gravity", "a0_static_tilt"):
        print(f"  {k}: spill-boundary threshold @10% ep-FA = {np.degrees(res[k]['thr']):.1f} deg-equiv")
    np.savez_compressed(a.out, **{k: v.astype(np.float32) for k, v in scores.items()},
                        params=json.dumps(params), metrics=json.dumps(res))
    print("saved", a.out)


if __name__ == "__main__":
    main()
