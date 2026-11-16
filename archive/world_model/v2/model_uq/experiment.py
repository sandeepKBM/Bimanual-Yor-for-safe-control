"""model_uq/experiment.py -- methods x budget n x repetitions on the residual table (protocol.py). SIMULATION / SAVED DATA ONLY. CPU.

  python -m world_model.v2.model_uq.experiment --table <residuals.npz> --k 15 --out <dir> --workers 6 [--reps 10,10,10,5]

Budgets n = 50, 200, 1000, 5000 labelled windows (half FIT, half CAL), `reps` independent draws from the POOL episodes, evaluated on the TEST episodes.
Methods (all split-conformal at 95%, scalar target y = L-inf equivalent accel error):
  today      the fixed box eps_a = 0.08 (not calibrated; width 0.08)                                  [no fit]
  global     one conformal box (CAL 95% quantile)                                                       [task 2]
  c1         SimPriorEnsemble (heteroscedastic NN ensemble, analytic-model prior + function-space regulariser), normalised conformal   [C1]
  pa1/pa2/pa3  PAPartition: polynomial-annihilation edges on lrho_m / (lrho_m, speed) / (lrho_m, speed, acc), per-cell conformal    [C2]
  pa1_true   PA on the TRUE lrho (uses the hidden mu; analysis only)
  tree       CART partition (log y) + Mondrian conformal; cqr  conformalised quantile regression (NN)       [baselines]
Features given to c1 / tree / cqr: FILT_COLS = lrho_m, speed, edge, tilt, acc (all computable by a filter without knowing mu).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.model_uq import protocol as P
from world_model.v2.model_uq.c1_ensemble import SimPriorEnsemble
from world_model.v2.model_uq.c2_partition import PAPartition, TreePartition, CQRBaseline

FILT_COLS = ("lrho_m", "speed", "edge", "tilt", "acc")
NS = (50, 200, 1000, 5000)


class Today:
    name = "today"

    def fit_calibrate(self, *a, **k):
        return self

    def bound(self, X):
        return np.full(len(X), P.EPS_TODAY)


def make_methods():
    """name -> (factory, cols)"""
    return {
        "today": (lambda: Today(), ("lrho_m",)),
        "global": (lambda: P.GlobalConformal(), ("lrho_m",)),
        "c1": (lambda: SimPriorEnsemble(), FILT_COLS),
        "c1_const": (lambda: SimPriorEnsemble(mean_only=True), FILT_COLS),
        "pa1": (lambda: PAPartition(cols_edge=(0,)), ("lrho_m",)),
        "pa2": (lambda: PAPartition(cols_edge=(0, 1)), ("lrho_m", "speed")),
        "pa3": (lambda: PAPartition(cols_edge=(0, 1, 2)), ("lrho_m", "speed", "acc")),
        "pa1_true": (lambda: PAPartition(cols_edge=(0,)), ("lrho",)),
        "tree": (lambda: TreePartition(), FILT_COLS),
        "cqr": (lambda: CQRBaseline(), FILT_COLS),
    }


G = {}


def _init(path, k):
    import torch
    torch.set_num_threads(1)
    G["t"] = P.load(path, k)
    G["pool"], G["test"] = P.split_pool_test(G["t"])
    G["M"] = make_methods()


def _task(args):
    n, rep, only = args
    t, M = G["t"], G["M"]
    rng = np.random.default_rng(1000 * n + rep)
    fit, cal = P.draw_budget(G["pool"], n, rng)
    res = []
    for name, (fac, cols) in M.items():
        if only and name not in only:
            continue
        X = P.design(t, cols)
        t0 = time.perf_counter()
        m = fac()
        m.fit_calibrate(X[fit], t["y"][fit], X[cal], t["y"][cal], P.ALPHA, np.random.default_rng(rng.integers(1 << 31)))
        ft = time.perf_counter() - t0
        ev = P.evaluate(m, t, cols, G["test"], cap=float(np.quantile(t["y"][cal], 0.999)))
        ev.update(method=name, n=n, rep=rep, fit_s=ft)
        if hasattr(m, "edges"):
            ev["edges"] = {str(k): [float(e) for e in v] for k, v in m.edges.items()}
        res.append(ev)
    return res


def run(path, k, out, workers=6, reps=(10, 10, 10, 5), ns=NS, only=None):
    tasks = [(n, r, only) for n, nr in zip(ns, reps) for r in range(nr)]
    tasks.sort(key=lambda a: -a[0])
    rows = []
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init, initargs=(path, k), maxtasksperchild=4) as Pl:
        for i, r in enumerate(Pl.imap_unordered(_task, tasks)):
            rows += r
            print(f"  {i + 1}/{len(tasks)} {time.perf_counter() - t0:.0f}s", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"grid_k{k}.json").write_text(json.dumps(rows))
    return rows


def summarize(rows):
    """-> {method: {n: dict(cov, cov_sd, width, width_sd, cov_by_regime, width_by_regime, frac_inf, fit_s)}}"""
    out = {}
    for r in rows:
        out.setdefault(r["method"], {}).setdefault(r["n"], []).append(r)
    S = {}
    for m, d in out.items():
        S[m] = {}
        for n, L in sorted(d.items()):
            f = lambda key: np.array([x[key] for x in L], np.float64)
            cb = np.array([x["cov_by_regime"] for x in L], np.float64)
            wb = np.array([x["width_by_regime"] for x in L], np.float64)
            S[m][n] = dict(reps=len(L), cov=float(f("cov").mean()), cov_sd=float(f("cov").std()), width=float(f("width").mean()),
                           width_sd=float(f("width").std()), cov_by_regime=np.nanmean(cb, 0).tolist(), cov_min_by_regime=np.nanmin(cb, 0).tolist(),
                           width_by_regime=np.nanmean(wb, 0).tolist(), frac_inf=float(f("frac_inf").mean()), fit_s=float(f("fit_s").mean()))
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reps", default="10,10,10,5")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rows = run(a.table, a.k, a.out, a.workers, tuple(int(x) for x in a.reps.split(",")), only=set(filter(None, a.only.split(","))))
    S = summarize(rows)
    (Path(a.out) / f"summary_k{a.k}.json").write_text(json.dumps(S, indent=1))
    for m, d in S.items():
        for n, s in d.items():
            print(f"{m:9s} n={n:5d} cov {s['cov']:.3f}+-{s['cov_sd']:.3f} width {s['width']:.3f}+-{s['width_sd']:.3f} "
                  f"regime cov {np.round(s['cov_by_regime'], 2).tolist()}", flush=True)


if __name__ == "__main__":
    main()
