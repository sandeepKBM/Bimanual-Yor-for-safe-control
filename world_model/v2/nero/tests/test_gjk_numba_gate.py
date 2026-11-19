"""Owner gate (11-17): the Numba GJK (nero/gjk_numba.py, default) = the distance between two convex hulls, checked against an INDEPENDENT
minimiser (SLSQP over the two simplices, tight tolerance) and against a scipy-nnls Gilbert implementation kept here as a test oracle.  Pure, no physics.
PYTHONPATH=. python -m pytest -q world_model/v2/nero/tests/test_gjk_numba_gate.py"""
import numpy as np
import pytest
from scipy.optimize import minimize

from world_model.v2.nero import clearance as CL
from world_model.v2.nero.gjk_numba import gjk


def _box(rng, c):
    h = rng.uniform(0.01, 0.15, 3)
    P = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return P @ R.T + c


def _cloud(rng, c):
    return rng.normal(size=(int(rng.integers(4, 30)), 3)) * rng.uniform(0.01, 0.1) + c


def _scipy_oracle(Wa, Wb, tol=1e-10, maxit=60):
    """Gilbert distance between the convex hulls of two point sets (0.0 when they touch / overlap)."""
    from scipy.optimize import nnls
    sup = lambda dr: Wa[np.argmax(Wa @ dr)] - Wb[np.argmin(Wb @ dr)]
    v = Wa[0] - Wb[0]
    S = [v]
    for _ in range(maxit):
        nv = float(v @ v)
        if nv < 1e-12:
            return 0.0
        w = sup(-v)
        if nv - float(v @ w) < tol + 1e-9 * nv:
            break
        S.append(w)
        P = np.array(S)
        M = 1e3
        lam, _ = nnls(np.vstack([P.T, M * np.ones((1, len(P)))]), np.array([0.0, 0.0, 0.0, M]))
        keep = lam > 1e-12
        S = [S[i] for i in range(len(S)) if keep[i]]
        v = lam[keep] @ P[keep]
    return float(np.linalg.norm(v))


def _exact(A, B):
    """min |A^T l - B^T u| over the simplices (independent of GJK), best of several starts."""
    na, nb = len(A), len(B)

    def f(x):
        v = x[:na] @ A - x[na:] @ B
        return float(v @ v)

    def g(x):
        v = x[:na] @ A - x[na:] @ B
        return np.r_[2 * A @ v, -2 * B @ v]
    cons = [dict(type="eq", fun=lambda x: x[:na].sum() - 1.0), dict(type="eq", fun=lambda x: x[na:].sum() - 1.0)]
    best = np.inf
    rng = np.random.default_rng(0)
    for k in range(4):
        x0 = np.r_[rng.dirichlet(np.ones(na)), rng.dirichlet(np.ones(nb))] if k else np.r_[np.ones(na) / na, np.ones(nb) / nb]
        r = minimize(f, x0, jac=g, bounds=[(0, 1)] * (na + nb), constraints=cons, method="SLSQP", options=dict(ftol=1e-16, maxiter=2000))
        best = min(best, r.fun)
    return float(np.sqrt(max(best, 0.0)))


def _pairs(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        mk = _box if i % 2 == 0 else _cloud
        A = mk(rng, np.zeros(3))
        B = mk(rng, rng.normal(size=3) * rng.uniform(0.0, 0.4))
        out.append((A, B))
    return out


def test_numba_matches_an_independent_minimiser():
    worst = 0.0
    for A, B in _pairs(120, 1117):
        dn = float(gjk(np.ascontiguousarray(A), np.ascontiguousarray(B), 1e-10, 60))
        de = _exact(A, B)
        if de < 1e-6:
            assert dn <= 1e-6, (dn, de)                                  # touching / overlapping -> 0 (or ~0)
            continue
        worst = max(worst, abs(dn - de))
        assert dn == pytest.approx(de, abs=1e-7), (dn, de)
        assert dn >= de - 1e-7                                           # never reports MORE clearance than there is
    assert worst < 1e-7, worst


def test_numba_vs_scipy_oracle_same_verdicts():
    for A, B in _pairs(400, 20261117):
        dn = CL._gjk(A, B)
        do = _scipy_oracle(A, B)
        assert (dn == 0.0) == (do == 0.0), (dn, do)
        for dmin in (2e-5, 0.005, 0.01, 0.02):
            assert (dn >= dmin) == (do >= dmin) or abs(dn - dmin) < 1e-6, (dmin, dn, do)
        if do < CL._DMAX:
            assert abs(dn - do) <= 1e-6, (dn, do)                        # the oracle carries the nnls penalty bias (agent: <= ~1e-6)

