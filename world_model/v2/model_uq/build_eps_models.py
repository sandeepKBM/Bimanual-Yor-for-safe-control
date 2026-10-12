"""model_uq/build_eps_models.py -- fit a state-dependent bound on a budget-n draw of the residual table and pickle it for F5UQ. SIMULATION ONLY.
  python -m world_model.v2.model_uq.build_eps_models --table <residuals.npz> --k 15 --method tree --n 5000 --name tree_k15_n5000
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from world_model.v2.model_uq import protocol as P
from world_model.v2.model_uq.experiment import make_methods
from world_model.v2.model_uq.f5_uq import EpsModel, EPS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    t = P.load(a.table, a.k)
    pool, test = P.split_pool_test(t)
    rng = np.random.default_rng(a.seed)
    fit, cal = P.draw_budget(pool, a.n, rng)
    fac, cols = make_methods()[a.method]
    X = P.design(t, cols)
    m = fac().fit_calibrate(X[fit], t["y"][fit], X[cal], t["y"][cal], P.ALPHA, np.random.default_rng(a.seed + 1))
    g = P.GlobalConformal().fit_calibrate(None, None, None, t["y"][cal])
    ev = P.evaluate(m, t, cols, test, cap=g.eps)
    em = EpsModel(m, cols, a.k, cap=g.eps)
    print(a.name, "global", round(g.eps, 3), "test cov", round(ev["cov"], 3), "mean width", round(ev["width"], 3),
          "regime cov", np.round(ev["cov_by_regime"], 2).tolist(), "regime width", np.round(ev["width_by_regime"], 3).tolist())
    EPS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EPS_DIR / f"{a.name}.pkl", "wb") as f:
        pickle.dump(em, f)


if __name__ == "__main__":
    main()
