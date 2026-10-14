"""tests for model_uq/c1_ensemble.py (SIMULATION / SAVED DATA ONLY). Run: python world_model/v2/model_uq/tests/test_c1.py"""
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch  # noqa: E402

torch.set_num_threads(1)

from world_model.v2.model_uq import protocol as P  # noqa: E402
from world_model.v2.model_uq.c1_ensemble import SimPriorEnsemble  # noqa: E402

DATA = "/common/users/ss5772/bimanual_data/model_uq/residuals_val_v2.npz"
COLS = ("lrho_m", "speed", "edge", "tilt", "acc")


def _synth(n, rng):
    x = rng.normal(size=(n, 1))
    y = np.abs(rng.normal(size=n)) * (0.05 + 0.4 / (1 + np.exp(-4 * x[:, 0])))
    return x, y


def _cov(n, reps, seed=0):
    covs, ub = [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        X, y = _synth(n, rng)
        h = n // 2
        m = SimPriorEnsemble().fit_calibrate(X[:h], y[:h], X[h:], y[h:], 0.05, rng)
        Xt, yt = _synth(20000, rng)
        covs.append(float((yt <= m.bound(Xt)).mean()))
        ub.append(m)
    return float(np.mean(covs)), covs, ub[-1]


def test_synthetic_coverage_and_heteroscedasticity():
    for n, reps in ((200, 24), (2000, 3)):
        c, cs, m = _cov(n, reps, seed=1000)
        print(f"synthetic n={n}: mean coverage {c:.3f} per-rep min/max {min(cs):.3f}/{max(cs):.3f}", flush=True)
        assert 0.93 <= c <= 0.97, (n, c)
    b = m.bound(np.array([[-2.0], [2.0]]))
    print(f"synthetic n=2000 bound at x=-2: {b[0]:.3f}, x=+2: {b[1]:.3f}", flush=True)
    assert b[1] > 1.8 * b[0]


def test_tiny_n_inf_and_graceful():
    rng = np.random.default_rng(0)
    X, y = _synth(25, rng)
    m = SimPriorEnsemble().fit_calibrate(X[:12], y[:12], X[12:], y[12:], 0.05, rng)
    # n_cal=13 -> k=ceil(14*.95)=14>13 -> q=inf -> +inf bound
    assert np.isinf(m.bound(X)).all()


def test_real_table():
    t = P.load(DATA, k=5)
    pool, test = P.split_pool_test(t)
    for mo in (False, True):
        for n in (200, 1000):
            rows = []
            for rep in range(3):
                rng = np.random.default_rng(100 * n + rep)
                fi, ci = P.draw_budget(pool, n, rng)
                X = P.design(t, COLS)
                m = SimPriorEnsemble(mean_only=mo)
                t0 = time.time()
                m.fit_calibrate(X[fi], t["y"][fi], X[ci], t["y"][ci], P.ALPHA, rng)
                dt = time.time() - t0
                r = P.evaluate(m, t, COLS, test, cap=10.0)
                rows.append((r, dt))
            cov = np.mean([r["cov"] for r, _ in rows])
            wid = np.mean([r["width"] for r, _ in rows])
            cr = np.nanmean([r["cov_by_regime"] for r, _ in rows], 0)
            wr = np.nanmean([r["width_by_regime"] for r, _ in rows], 0)
            print(f"[{m.name}] n={n}: cov {cov:.3f} (reps {[round(r['cov'], 3) for r, _ in rows]}) width {wid:.3f} "
                  f"fit {np.mean([d for _, d in rows]):.1f}s\n   cov/regime {np.round(cr, 3).tolist()}\n   wid/regime {np.round(wr, 3).tolist()}",
                  flush=True)
            assert 0.88 <= cov <= 0.99


def test_runtime():
    t = P.load(DATA, k=5)
    pool, _ = P.split_pool_test(t)
    X = P.design(t, COLS)
    for n in (200, 1000, 5000):
        rng = np.random.default_rng(1)
        fi, ci = P.draw_budget(pool, n, rng)
        t0 = time.time()
        SimPriorEnsemble().fit_calibrate(X[fi], t["y"][fi], X[ci], t["y"][ci], P.ALPHA, rng)
        print(f"runtime n={n}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    for f in (test_synthetic_coverage_and_heteroscedasticity, test_tiny_n_inf_and_graceful, test_real_table, test_runtime):
        f()
    print("ALL OK")
