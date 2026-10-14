"""sets/zono.py -- zonotopes and H-rep polytopes. Pure numpy. SIMULATION / SAVED DATA ONLY (no hardware, no network).

A zonotope is  Z = { c + G s : |s_i| <= 1 }  with centre c (n,) and generators G (n, m).
Every array may carry leading batch axes: c (..., n), G (..., n, m).  A batch is a time sequence in the filter
(one zonotope per tick of the horizon).  All the formulas below broadcast over those axes.

What the library gives
    linear map, translation, Minkowski sum        exact (a zonotope again)
    support function h(d) = c.d + sum_i |g_i.d|   exact; also the support point
    signed distance to a half-space {x : a.x <= b}      (b - h(a)) / |a|     > 0  <=>  Z lies inside
    containment in a polytope {A x <= b}          Z in P  <=>  h_Z(A_i) <= b_i for all i   (exact)
    interval hull, Girard order reduction         outer approximations
    sphere / ball / ellipsoid  ->  zonotope       outer approximations (2-D: circumscribed regular polygon; 3-D and up:
                                                  a fixed generator set scaled until its support covers a direction grid)
    signed distance of the origin to a zonotope   sd(Z) = max_{|n|=1} ( -h_Z(n) ), positive outside, negative inside

Not here: constrained zonotopes (intersections), non-convex sets. For the tray and block testbeds a polytope in H-rep is
enough, because the only intersection we need is "A inside B", and that is a support-function test.
"""
from __future__ import annotations

import itertools

import numpy as np


# ====================================================================================== zonotope
class Zonotope:
    __slots__ = ("c", "G")

    def __init__(self, c, G=None):
        c = np.asarray(c, np.float64)
        if G is None:
            G = np.zeros(c.shape + (0,))
        G = np.asarray(G, np.float64)
        if G.ndim == 1:
            G = G[:, None]
        self.c, self.G = c, G

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def point(c):
        return Zonotope(c)

    @staticmethod
    def from_box(lo, hi):
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        return Zonotope((lo + hi) / 2.0, np.diag((hi - lo) / 2.0))

    @staticmethod
    def segment(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return Zonotope((a + b) / 2.0, ((b - a) / 2.0)[:, None])

    # ------------------------------------------------------------------ basic
    @property
    def dim(self):
        return self.c.shape[-1]

    @property
    def n_gen(self):
        return self.G.shape[-1]

    @property
    def batch(self):
        return self.c.shape[:-1]

    def __repr__(self):
        return f"Zonotope(dim={self.dim}, n_gen={self.n_gen}, batch={self.batch})"

    # ------------------------------------------------------------------ operations
    def translate(self, v):
        return Zonotope(self.c + np.asarray(v, np.float64), self.G)

    def linear(self, M, t=None):
        """image under x -> M x + t. M (m, n) or batched (..., m, n)."""
        M = np.asarray(M, np.float64)
        c = np.einsum("...ij,...j->...i", M, self.c)
        G = np.matmul(M, self.G)
        if t is not None:
            c = c + np.asarray(t, np.float64)
        return Zonotope(c, G)

    def minkowski(self, other):
        """Minkowski sum Z (+) W  (exact)."""
        if not isinstance(other, Zonotope):
            return self.translate(other)
        sh = np.broadcast_shapes(self.c.shape, other.c.shape)
        gb = sh[:-1]
        a = np.broadcast_to(self.G, gb + self.G.shape[-2:])
        b = np.broadcast_to(other.G, gb + other.G.shape[-2:])
        return Zonotope(self.c + other.c, np.concatenate([a, b], axis=-1))

    __add__ = minkowski

    def neg(self):
        return Zonotope(-self.c, self.G)

    def scale(self, s):
        return Zonotope(self.c * s, self.G * s)

    # ------------------------------------------------------------------ support
    def support(self, d):
        """h_Z(d) = max_{x in Z} d.x.   d (n,) or (..., n); broadcast with the batch axes of Z."""
        d = np.asarray(d, np.float64)
        gd = np.einsum("...nm,...n->...m", self.G, d) if self.G.shape[-1] else np.zeros(d.shape[:-1] + (0,))
        return (self.c * d).sum(-1) + np.abs(gd).sum(-1)

    def support_point(self, d):
        d = np.asarray(d, np.float64)
        gd = np.einsum("...nm,...n->...m", self.G, d)
        return self.c + np.einsum("...nm,...m->...n", self.G, np.where(gd >= 0, 1.0, -1.0))

    def halfspace_margin(self, a, b):
        """signed distance of Z to the boundary of {x: a.x <= b}, inside positive:  (b - h(a)) / |a|.
        a (n,) or (K, n) -> margin shape (batch..., K)."""
        a = np.asarray(a, np.float64)
        na = np.linalg.norm(a, axis=-1)
        return (np.asarray(b) - self._support_many(a)) / na

    def _support_many(self, A):
        """h_Z(A_i) for the rows of A (K, n) -> (batch..., K)."""
        A = np.atleast_2d(A)
        gd = np.einsum("...nm,kn->...km", self.G, A)
        return np.einsum("...n,kn->...k", self.c, A) + np.abs(gd).sum(-1)

    def contained_in(self, A, b):
        """margin of Z inside the polytope {A x <= b}: min_i (b_i - h_Z(A_i)) / |A_i|  (>= 0 <=> contained, exact)."""
        A = np.atleast_2d(np.asarray(A, np.float64))
        m = (np.asarray(b) - self._support_many(A)) / np.linalg.norm(A, axis=-1)
        return m.min(-1)

    # ------------------------------------------------------------------ outer approximations
    def interval_hull(self):
        r = np.abs(self.G).sum(-1)
        return self.c - r, self.c + r

    def to_box(self):
        lo, hi = self.interval_hull()
        return Zonotope.from_box(lo, hi)

    def reduce(self, m_max):
        """Girard order reduction to at most m_max generators (m_max >= dim): keep the m_max - n generators with the largest
        (|g|_1 - |g|_inf) and replace the rest by their interval hull (n box generators). Outer approximation."""
        if self.batch:
            raise NotImplementedError("reduce: unbatched zonotopes only")
        n, m = self.G.shape
        if m <= m_max:
            return self
        assert m_max >= n
        score = np.abs(self.G).sum(0) - np.abs(self.G).max(0)
        order = np.argsort(-score)
        keep, rest = order[:m_max - n], order[m_max - n:]
        box = np.diag(np.abs(self.G[:, rest]).sum(1))
        return Zonotope(self.c, np.concatenate([self.G[:, keep], box], axis=1))

    def to_polytope_2d(self):
        """exact H-rep of a 2-D zonotope: one facet pair per generator (normal = perpendicular of the generator). Returns Polytope."""
        assert self.dim == 2 and not self.batch
        G = self.G[:, np.abs(self.G).sum(0) > 1e-15]
        N = np.stack([-G[1], G[0]], 1)
        N = N / np.linalg.norm(N, axis=1, keepdims=True)
        N = np.concatenate([N, -N], 0)
        b = np.array([self.support(n) for n in N])
        return Polytope(N, b, orthogonal=False)

    # ------------------------------------------------------------------ testing helpers
    def vertices(self):
        """all 2^m sign combinations (candidate vertices; the convex hull of them is Z). m <= 16."""
        m = self.n_gen
        assert not self.batch and m <= 16
        S = np.array(list(itertools.product([-1.0, 1.0], repeat=m))).reshape(-1, m)
        return self.c + S @ self.G.T

    def sample(self, k, rng):
        """k random points of Z: s uniform in [-1, 1]^m; the first half are pushed to the corners (s = +-1) to stress the extremes."""
        m = self.n_gen
        s = rng.uniform(-1, 1, (k, m))
        s[: k // 2] = np.sign(s[: k // 2])
        return self.c + s @ self.G.T


def sd_origin(Z: Zonotope, n_dirs=None, iters=8):
    """signed distance of the origin to the zonotope Z (unbatched): > 0 outside (the distance), < 0 inside (minus the
    penetration depth). Exact as the direction set becomes dense; here: a direction grid, then Gilbert-style refinement.
    sd = max over unit n of (-h_Z(n))."""
    n = Z.dim
    dirs = _unit_grid(n, n_dirs)
    h = Z.support(dirs)
    i = int(np.argmax(-h))
    best, bn = float(-h[i]), dirs[i]
    for _ in range(iters):                         # refine: point of Z support-extreme along bn, push n toward it
        z = Z.support_point(bn)
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        cand = -z / nz if best > 0 else bn
        v = float(-Z.support(cand))
        if v > best + 1e-12:
            best, bn = v, cand
        else:
            break
    return best


_GRID = {}


def _unit_grid(n, k=None):
    key = (n, k)
    if key not in _GRID:
        if n == 1:
            g = np.array([[1.0], [-1.0]])
        elif n == 2:
            kk = k or 180
            a = np.linspace(0, 2 * np.pi, kk, endpoint=False)
            g = np.stack([np.cos(a), np.sin(a)], 1)
        else:
            kk = k or (400 if n == 3 else 2000)
            rng = np.random.default_rng(1234 + n)
            g = rng.normal(size=(kk, n))
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            g = np.concatenate([g, np.eye(n), -np.eye(n)], 0)
        _GRID[key] = g
    return _GRID[key]


# ====================================================================================== sphere / ellipsoid outer zonotopes
def sphere_outer(n, r=1.0, n_gen=None):
    """zonotope that contains the Euclidean ball of radius r centred at 0 in R^n.
    n = 1: the interval. n = 2: circumscribed regular polygon with 2*m sides (m generators; default m = 8, so a 16-gon,
    radius excess 1/cos(pi/16) = 1.9%). n >= 3: the n axes, the +-1 diagonals of every pair of axes (and, for n = 3, the 4 body diagonals),
    scaled so that the support function covers r in every direction (minimum found on a dense grid plus local refinement, lowered by 0.2%;
    validated by a Monte-Carlo test)."""
    if n == 1:
        return Zonotope(np.zeros(1), np.array([[r]]))
    if n == 2:
        m = n_gen or 8
        # a regular 2m-gon with inradius r is the sum of m segments of half-length  r * tan(pi/(2m)) ... computed numerically below
        ang = np.arange(m) * np.pi / m
        d = np.stack([np.cos(ang), np.sin(ang)], 1)           # (m, 2) directions of the segments
        # support of the zonotope sum_i s_i d_i (|s_i|<=1, scaled by L) along n_hat = L * sum_i |d_i . n_hat|; the minimum over n_hat
        # of that is the inradius. Scale so the inradius is r.
        L = r / _min_support(d, 2)
        return Zonotope(np.zeros(2), (d * L).T)
    base = [np.eye(n)[i] for i in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        for s in (1.0, -1.0):
            v = np.zeros(n)
            v[i], v[j] = 1.0, s
            base.append(v)
    if n == 3:
        for s in itertools.product([1.0], [1.0, -1.0], [1.0, -1.0]):
            base.append(np.array(s))
    D = np.array(base)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    L = r / _min_support(D, n)
    return Zonotope(np.zeros(n), (D * L).T)


def _min_support(D, n):
    """min over unit n-vectors u of sum_i |D_i . u|  (the inradius of the unit-generator zonotope). Dense grid, then local
    refinement from the 12 lowest grid points; the result is lowered by 0.2% so the scaled zonotope still covers the ball."""
    from scipy.optimize import minimize
    grid = _unit_grid(n, 720 if n == 2 else 4000)
    f = lambda u: np.abs(D @ (u / np.linalg.norm(u))).sum()
    h = np.abs(grid @ D.T).sum(1)
    best = h.min()
    for i in np.argsort(h)[:12]:
        res = minimize(f, grid[i], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=2000))
        best = min(best, res.fun)
    return best * 0.998


def ball_outer(centre, r, n_gen=None):
    """zonotope containing the ball {x : |x - centre| <= r}."""
    centre = np.asarray(centre, np.float64)
    return sphere_outer(len(centre), r, n_gen).translate(centre)


def ellipsoid_outer(centre, P, k=1.0, n_gen=None):
    """zonotope containing the ellipsoid {x : (x - c)' P^-1 (x - c) <= k^2}, P = covariance (n, n) (the Kalman covariance at k sigma).
    x = c + k L s with P = L L', |s| <= 1; the unit-ball outer zonotope is mapped linearly (exact image)."""
    centre = np.asarray(centre, np.float64)
    P = np.asarray(P, np.float64)
    w, V = np.linalg.eigh(0.5 * (P + P.T))
    L = V * np.sqrt(np.maximum(w, 0.0))[None, :]
    return sphere_outer(len(centre), 1.0, n_gen).linear(k * L).translate(centre)


# ====================================================================================== polytopes (H-rep)
class Polytope:
    """{x : N x <= b}, unit rows. `orthogonal=True` marks box-like sets (all facet normals axis-parallel in some frame): then the
    outside distance of a point is the exact Euclidean one, sqrt(sum of the squared violated slacks). Otherwise it is the largest
    violation (a lower bound on the Euclidean distance; the sign is always right)."""

    def __init__(self, N, b, orthogonal=False, name=""):
        N = np.atleast_2d(np.asarray(N, np.float64))
        nn = np.linalg.norm(N, axis=1)
        self.N = N / nn[:, None]
        self.b = np.asarray(b, np.float64) / nn
        self.orthogonal = bool(orthogonal)
        self.name = name

    @staticmethod
    def box(half, axes=None):
        """axis-parallel box |x_i| <= half_i on the listed axes (default all)."""
        half = np.asarray(half, np.float64)
        n = len(half)
        axes = list(range(n)) if axes is None else list(axes)
        rows, b = [], []
        for i in axes:
            e = np.zeros(n)
            e[i] = 1.0
            rows += [e, -e]
            b += [half[i], half[i]]
        return Polytope(np.array(rows), np.array(b), orthogonal=True)

    def shrink(self, r):
        """erosion by a ball of radius r (exact for H-rep with unit normals): b <- b - r."""
        return Polytope(self.N, self.b - r, self.orthogonal, self.name)

    def slacks(self, x):
        return self.b - self.N @ np.asarray(x, np.float64)

    def contains(self, x):
        return bool(np.all(self.slacks(x) >= 0))

    def margin_from_slacks(self, s):
        """s (..., K) slacks -> signed distance (inside > 0). Orthogonal sets: exact Euclidean outside."""
        s = np.asarray(s)
        inside = s.min(-1)
        if self.orthogonal:
            viol = np.where(s < 0, -s, 0.0)
            out = -np.sqrt((viol ** 2).sum(-1))
            return np.where(inside >= 0, inside, out)
        return inside

    def signed_distance_point(self, x):
        return self.margin_from_slacks(self.slacks(x))

    def contains_zonotope_margin(self, Z: Zonotope):
        s = self.b - Z._support_many(self.N)
        return self.margin_from_slacks(s)
