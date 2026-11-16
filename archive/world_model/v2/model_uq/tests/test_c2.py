"""tests/test_c2.py -- C2 (PA partition + baselines). Run: python world_model/v2/model_uq/tests/test_c2.py [--no-real]. SIMULATION / SAVED DATA ONLY."""
from __future__ import annotations

import sys
import time

import numpy as np

from world_model.v2.model_uq import protocol as P
from world_model.v2.model_uq.c2_partition import CQRBaseline, PAPartition, TreePartition, pa_edges_1d

DATA = "/common/users/ss5772/bimanual_data/model_uq/residuals_val_v2.npz"
COLS = ("lrho_m", "speed", "edge", "tilt", "acc")


def test_pa_synthetic():
    x = np.linspace(0, 1, 128)
    h = 1 / 127
    for seed in range(5):
        r = np.random.default_rng(seed)
        f = np.sin(2 * np.pi * x) + 0.5 * (x > 0.3) + 1.0 * (x > 0.7) + 0.01 * r.normal(size=128)
        e = pa_edges_1d(x, f)
        assert len(e) == 2, e
        assert abs(e[0] - 0.3) <= 2 * h and abs(e[1] - 0.7) <= 2 * h, e
        g = np.sin(2 * np.pi * x) + 0.01 * r.normal(size=128)
        assert len(pa_edges_1d(x, g)) == 0
    # nonuniform grid
    r = np.random.default_rng(9)
    xn = np.sort(r.uniform(0, 1, 160))
    fn = np.sin(2 * np.pi * xn) + 0.5 * (xn > 0.3) + 1.0 * (xn > 0.7) + 0.01 * r.normal(size=160)
    en = pa_edges_1d(xn, fn)
    assert len(en) == 2 and abs(en[0] - 0.3) < 0.03 and abs(en[1] - 0.7) < 0.03, en
    print("PA synthetic OK (uniform x5 seeds, nonuniform)")


def _synth(n, rng):
    x = rng.uniform(0, 1, n)
    y = np.abs(rng.normal(size=n)) * np.where(x < 0.5, 0.05, 0.3)
    return x[:, None], y


def test_synth_coverage():
    for n in (400, 4000):
        covs, hits = {}, []
        for rep in range(40 if n == 400 else 20):
            rng = np.random.default_rng(100 + rep)
            X, y = _synth(n, rng)
            h = n // 2
            Xt, yt = _synth(20000, rng)
            for mk in (lambda: PAPartition((0,)), lambda: TreePartition(), lambda: CQRBaseline()):
                m = mk().fit_calibrate(X[:h], y[:h], X[h:], y[h:], 0.05, rng)
                covs.setdefault(m.name, []).append(float((yt <= m.bound(Xt)).mean()))
                if n == 4000 and m.name.startswith("pa"):
                    hits.append(any(abs(v - 0.5) < 0.05 for v in m.edges[0]))
        if n == 4000:
            print(f"  PA edge within 0.05 of 0.5 in {np.mean(hits):.0%} of {len(hits)} reps")
            assert np.mean(hits) >= 0.7, hits
        for k, v in covs.items():
            mc = float(np.mean(v))
            print(f"  synth n={n} {k:8s} marginal cov = {mc:.3f}")
            assert 0.92 <= mc <= 0.98, (n, k, mc)
    print("synthetic coverage + PA edge near 0.5 OK")


def _run_real(t, methods, ns=(200, 1000), reps=3):
    pool, test = P.split_pool_test(t)
    res = {}
    for n in ns:
        for rep in range(reps):
            for name, mk, cols in methods:
                rng = np.random.default_rng(1000 * n + rep)
                fi, ci = P.draw_budget(pool, n, rng)
                X = P.design(t, cols)
                m = mk().fit_calibrate(X[fi], t["y"][fi], X[ci], t["y"][ci], P.ALPHA, rng)
                ev = P.evaluate(m, t, cols, test, cap=10.0)
                res.setdefault((n, name), []).append((ev, m))
    return res


def real_report():
    t = P.load(DATA, k=5)
    full = COLS
    methods = [
        ("global", P.GlobalConformal, full),
        ("pa[0]", lambda: PAPartition((0,), name="pa[0]"), full),
        ("pa[0,1]", lambda: PAPartition((0, 1), name="pa[0,1]"), full),
        ("pa[0]rel.15", lambda: PAPartition((0,), rel_jump=0.15, name="pa[0]rel.15"), full),
        ("tree", TreePartition, full),
        ("cqr", CQRBaseline, full),
    ]
    res = _run_real(t, methods)
    print("\nREAL TABLE (test episodes; 3 reps/budget).  regimes:", ", ".join(P.RHO_NAMES))
    for (n, name), lst in res.items():
        cov = np.mean([e["cov"] for e, _ in lst])
        w = np.mean([e["width"] for e, _ in lst])
        fi = np.mean([e["frac_inf"] for e, _ in lst])
        cb = np.nanmean([e["cov_by_regime"] for e, _ in lst], 0)
        wb = np.nanmean([e["width_by_regime"] for e, _ in lst], 0)
        print(f"n={n:5d} {name:12s} cov={cov:.3f} width={w:.3f} inf={fi:.2f} | cov/regime {np.round(cb, 2).tolist()} | width/regime {np.round(wb, 2).tolist()}")
        if name.startswith("pa"):
            for e, m in lst:
                print(f"      lrho_m edges: {np.round(m.edges[0], 2).tolist()}  (raw {np.round(m.edges_raw[0], 2).tolist()}) cells={m.cells['n_cells']} fallback={m.cells['n_fallback']}")
    # runtime
    print("\nfit_calibrate runtime [s]:")
    pool, test = P.split_pool_test(t)
    X = P.design(t, full)
    for n in (200, 1000, 5000):
        rng = np.random.default_rng(5)
        fi, ci = P.draw_budget(pool, n, rng)
        out = []
        for name, mk, _ in methods:
            t0 = time.time()
            mk().fit_calibrate(X[fi], t["y"][fi], X[ci], t["y"][ci], P.ALPHA, rng)
            out.append(f"{name}={time.time() - t0:.3f}")
        print(f"  n={n}: " + "  ".join(out))


if __name__ == "__main__":
    test_pa_synthetic()
    test_synth_coverage()
    if "--no-real" not in sys.argv:
        real_report()
    print("ALL OK")
