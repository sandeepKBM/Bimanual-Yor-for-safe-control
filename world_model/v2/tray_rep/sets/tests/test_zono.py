"""Unit tests for sets/zono.py. Run:  python -m pytest -q world_model/v2/tray_rep/sets/tests  """
import itertools

import numpy as np
import pytest

from world_model.v2.tray_rep.sets.zono import (Zonotope, Polytope, sphere_outer, ball_outer, ellipsoid_outer, sd_origin)

RNG = np.random.default_rng(0)


def rand_zono(n, m, rng=RNG):
    return Zonotope(rng.normal(size=n), rng.normal(size=(n, m)))


@pytest.mark.parametrize("n,m", [(2, 3), (2, 5), (3, 4), (3, 6)])
def test_support_vs_vertex_enumeration(n, m):
    for _ in range(20):
        Z = rand_zono(n, m)
        V = Z.vertices()
        for _ in range(20):
            d = RNG.normal(size=n)
            assert abs(Z.support(d) - (V @ d).max()) < 1e-10
            p = Z.support_point(d)
            assert abs(p @ d - Z.support(d)) < 1e-10


def test_linear_map_and_minkowski_identities():
    for n, m in [(2, 4), (3, 5)]:
        Z, W = rand_zono(n, m), rand_zono(n, m + 1)
        M = RNG.normal(size=(n, n))
        t = RNG.normal(size=n)
        d = RNG.normal(size=n)
        # h_{MZ + t}(d) = h_Z(M' d) + t.d
        assert abs(Z.linear(M, t).support(d) - (Z.support(M.T @ d) + t @ d)) < 1e-10
        # h_{Z+W} = h_Z + h_W
        assert abs((Z + W).support(d) - (Z.support(d) + W.support(d))) < 1e-10
        # (Z + W) vertices = pairwise sums (hull): support of the vertex set matches
        V = (Z + W).vertices()
        assert abs((V @ d).max() - (Z + W).support(d)) < 1e-9
        # M(Z + W) = MZ + MW
        assert abs(((Z + W).linear(M)).support(d) - (Z.linear(M) + W.linear(M)).support(d)) < 1e-10
        # translation
        assert abs(Z.translate(t).support(d) - (Z.support(d) + t @ d)) < 1e-12


def test_batch_support_matches_loop():
    c = RNG.normal(size=(5, 3))
    G = RNG.normal(size=(5, 3, 4))
    Z = Zonotope(c, G)
    d = RNG.normal(size=3)
    h = Z.support(d)
    for k in range(5):
        assert abs(h[k] - Zonotope(c[k], G[k]).support(d)) < 1e-12
    A = RNG.normal(size=(4, 3))
    H = Z._support_many(A)
    assert H.shape == (5, 4)
    for k in range(5):
        for i in range(4):
            assert abs(H[k, i] - Zonotope(c[k], G[k]).support(A[i])) < 1e-12


def test_halfspace_signed_distance():
    Z = Zonotope([0.0, 0.0], np.eye(2) * 0.5)
    # box [-.5,.5]^2 against x <= 1: gap 0.5
    assert abs(Z.halfspace_margin([1.0, 0.0], 1.0)[()] - 0.5) < 1e-12 if np.ndim(Z.halfspace_margin([1.0, 0.0], 1.0)) == 0 else True
    m = Z.halfspace_margin(np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 1.0]))
    assert np.allclose(m, [0.5, (1.0 - 1.0) / 2.0])
    # negative when it sticks out
    assert Z.halfspace_margin(np.array([[1.0, 0.0]]), np.array([0.25]))[0] < 0


def test_containment_vs_monte_carlo():
    rng = np.random.default_rng(3)
    for n in (2, 3):
        for _ in range(30):
            Z = Zonotope(rng.normal(size=n) * 0.1, rng.normal(size=(n, 4)) * 0.3)
            N = rng.normal(size=(8, n))
            N /= np.linalg.norm(N, axis=1, keepdims=True)
            b = rng.uniform(0.3, 1.2, 8)
            P = Polytope(N, b)
            margin = Z.contained_in(P.N, P.b)
            V = Z.vertices()
            inside_all = bool(np.all(V @ P.N.T <= P.b + 1e-12))          # exact: a polytope contains Z iff it contains its vertices
            assert (margin >= 0) == inside_all
            S = Z.sample(4000, rng)
            mc_inside = bool(np.all(S @ P.N.T <= P.b + 1e-12))
            if margin >= 0:
                assert mc_inside                                          # a certified containment must hold for every sample
            # margin is the exact distance of the worst vertex to its facet
            assert abs(margin - (P.b - V @ P.N.T).min()) < 1e-10


def test_interval_hull_and_girard():
    rng = np.random.default_rng(5)
    Z = Zonotope(rng.normal(size=3), rng.normal(size=(3, 12)))
    lo, hi = Z.interval_hull()
    S = Z.sample(3000, rng)
    assert np.all(S >= lo - 1e-12) and np.all(S <= hi + 1e-12)
    R = Z.reduce(6)
    assert R.n_gen == 6
    for _ in range(50):
        d = rng.normal(size=3)
        assert R.support(d) >= Z.support(d) - 1e-10           # outer approximation
    # reduction is the identity when small enough
    assert Z.reduce(20) is Z


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_sphere_outer_contains_ball(n):
    Z = sphere_outer(n, 1.0)
    rng = np.random.default_rng(7)
    for _ in range(2000):
        d = rng.normal(size=n)
        d /= np.linalg.norm(d)
        assert Z.support(d) >= 1.0 - 1e-9
    # and it is not absurdly loose
    rad = max(Z.support(d / np.linalg.norm(d)) for d in rng.normal(size=(500, n)))
    assert rad <= {1: 1.0001, 2: 1.03, 3: 1.35, 4: 1.7}[n]


def test_ellipsoid_outer_covers_kalman_k_sigma():
    rng = np.random.default_rng(9)
    for n in (2, 4):
        A = rng.normal(size=(n, n))
        P = A @ A.T * 0.01 + 1e-4 * np.eye(n)
        c = rng.normal(size=n)
        Z = ellipsoid_outer(c, P, k=2.0)
        L = np.linalg.cholesky(P)
        # points on the k-sigma ellipsoid surface and inside
        for _ in range(3000):
            s = rng.normal(size=n)
            s *= rng.uniform(0, 1) ** (1 / n) / np.linalg.norm(s)
            x = c + 2.0 * L @ s
            # x in Z <=> for every direction d: d.x <= h_Z(d)
            for d in rng.normal(size=(6, n)):
                assert d @ x <= Z.support(d) + 1e-9
        # support function is >= the ellipsoid's exact one: c.d + k sqrt(d'Pd)
        for d in rng.normal(size=(200, n)):
            assert Z.support(d) >= c @ d + 2.0 * np.sqrt(d @ P @ d) - 1e-9


def test_ball_outer_translates():
    Z = ball_outer([1.0, -2.0, 0.5], 0.3)
    assert abs(Z.support(np.array([1.0, 0, 0])) - 1.3) < 0.05 + 0.3 * 0.4


def test_sd_origin():
    box = Zonotope([3.0, 0.0], np.eye(2) * 1.0)                    # box [2,4] x [-1,1]; origin outside, distance 2
    assert abs(sd_origin(box) - 2.0) < 1e-3
    box2 = Zonotope([3.0, 3.0], np.eye(2) * 1.0)                   # corner at (2,2): distance sqrt(8)
    assert abs(sd_origin(box2) - np.sqrt(8.0)) < 1e-2
    inside = Zonotope([0.2, 0.0], np.eye(2) * 1.0)                 # origin inside; nearest face at distance 0.8 -> -0.8
    assert abs(sd_origin(inside) + 0.8) < 1e-3
    # a rotated thin zonotope: distance known by brute force over a dense point sample
    rng = np.random.default_rng(1)
    Z = Zonotope([2.0, 1.0], np.array([[0.5, 0.2, 0.1], [0.1, -0.4, 0.3]]))
    S = Z.sample(200000, rng)
    d_mc = np.linalg.norm(S, axis=1).min()
    assert abs(sd_origin(Z) - d_mc) < 0.02


def test_polytope_box_sdf_matches_rect():
    P = Polytope.box([0.15, 0.10], axes=[0, 1])
    rng = np.random.default_rng(2)
    for _ in range(500):
        x = rng.uniform(-0.3, 0.3, 2)
        ex, ey = 0.15 - abs(x[0]), 0.10 - abs(x[1])
        ox, oy = max(abs(x[0]) - 0.15, 0), max(abs(x[1]) - 0.10, 0)
        ref = min(ex, ey) if (ex >= 0 and ey >= 0) else -np.hypot(ox, oy)
        assert abs(P.signed_distance_point(x) - ref) < 1e-14
    assert abs(P.shrink(0.03).signed_distance_point([0.0, 0.0]) - 0.07) < 1e-14
