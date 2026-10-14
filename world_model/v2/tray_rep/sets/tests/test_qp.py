"""qp.py (Hildreth) against OSQP on random projection problems, feasible and infeasible."""
import numpy as np
import pytest

from world_model.v2.tray_rep.sets import qp

osqp = pytest.importorskip("osqp")
import scipy.sparse as sp


def osqp_solve(A, b, u0, w):
    m = osqp.OSQP()
    m.setup(sp.csc_matrix(np.diag(w ** 2)), -(w ** 2) * u0, sp.csc_matrix(A), -np.inf * np.ones(len(b)), b, verbose=False,
            eps_abs=1e-9, eps_rel=1e-9, max_iter=40000)
    r = m.solve()
    return r.x, r.info.status


def test_matches_osqp_on_random_problems():
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(60):
        n, m = 6, rng.integers(20, 120)
        A = rng.normal(size=(m, n))
        u0 = rng.normal(size=n) * 0.2
        w = rng.uniform(0.5, 3.0, n)
        x_feas = rng.normal(size=n) * 0.3
        b = A @ x_feas + rng.uniform(0.0, 0.3, m)                    # x_feas is feasible by construction
        u, ok, sw, viol = qp.project(A, b, u0, w)
        ref, st = osqp_solve(A, b, u0, w)
        assert ok and st == "solved"
        assert np.all(A @ u <= b + 1e-3)
        c = lambda x: 0.5 * np.sum((w * (x - u0)) ** 2)
        assert c(u) <= c(ref) * (1 + 2e-3) + 1e-5
        assert np.abs(u - ref).max() < 2e-3 + 1e-2 * np.abs(ref - u0).max()
        n_ok += 1
    assert n_ok == 60


def test_unconstrained_and_infeasible():
    w = np.ones(3)
    u0 = np.array([1.0, 2.0, 3.0])
    u, ok, _, _ = qp.project(np.array([[1.0, 0, 0]]), np.array([10.0]), u0, w)
    assert ok and np.allclose(u, u0)
    A = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    u, ok, _, viol = qp.project(A, np.array([-1.0, -1.0]), u0, w)           # x <= -1 and x >= 1
    assert not ok and viol > 0.5
    u, ok, _, _ = qp.project(np.zeros((0, 3)), np.zeros(0), u0, w)
    assert ok
