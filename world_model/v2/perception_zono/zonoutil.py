"""zonoutil.py -- 3-D zonotope helpers on top of tray_rep.sets.zono.Zonotope: exact H-rep, containment, exact volume, signed distance of
points, and a union-of-zonotopes class. Pure numpy/scipy. SIMULATION ONLY."""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from world_model.v2.tray_rep.sets.zono import Zonotope


def halfspaces(Z: Zonotope, tol=1e-9):
    """exact H-rep (A, b), A x <= b, unit rows, of a 3-D zonotope: one facet pair per generator pair (normal = g_i x g_j).
    Rank-deficient zonotopes (flat / line / point) get a +-eps slab on the missing directions so containment is still defined."""
    G = Z.G
    G = G[:, np.linalg.norm(G, axis=0) > 1e-12]
    m = G.shape[1]
    rows = []
    if m >= 2:
        i, j = np.triu_indices(m, 1)
        Nn = np.cross(G[:, i].T, G[:, j].T)
        nn = np.linalg.norm(Nn, axis=1)
        Nn = Nn[nn > 1e-12 * max(1.0, np.linalg.norm(G) ** 2)] / nn[nn > 1e-12 * max(1.0, np.linalg.norm(G) ** 2)][:, None]
        # de-duplicate parallel normals
        if len(Nn):
            key = np.round(np.where(Nn[:, [np.argmax(np.abs(Nn).max(0))]] < 0, -Nn, Nn), 6)
            _, u = np.unique(key, axis=0, return_index=True)
            Nn = Nn[np.sort(u)]
        rows.append(Nn)
    A = np.concatenate(rows) if rows else np.zeros((0, 3))
    rank = np.linalg.matrix_rank(G, tol=1e-9) if m else 0
    if rank < 3:                                           # add slabs along directions orthogonal to the span
        U, S, _ = np.linalg.svd(G if m else np.zeros((3, 1)), full_matrices=True)
        extra = U[:, rank:].T
        A = np.concatenate([A, extra]) if len(A) else extra
    A = np.concatenate([A, -A])
    b = A @ Z.c + np.abs(A @ G).sum(1) + tol
    return A, b


def volume(Z: Zonotope):
    """exact volume of a 3-D zonotope: 8 sum_{i<j<k} |det(g_i, g_j, g_k)|."""
    G = Z.G
    m = G.shape[1]
    if m < 3:
        return 0.0
    import itertools
    idx = np.array(list(itertools.combinations(range(m), 3)))
    D = np.linalg.det(np.stack([G[:, idx[:, 0]].T, G[:, idx[:, 1]].T, G[:, idx[:, 2]].T], 2))
    return float(8.0 * np.abs(D).sum())


class ZUnion:
    """a union of zonotopes (the output of a split fit; a single zonotope is a union of one)."""

    def __init__(self, zs):
        self.zs = list(zs)
        self._hr = [halfspaces(z) for z in self.zs]

    @property
    def n_gen(self):
        return int(sum(z.G.shape[1] for z in self.zs))

    def contains(self, X):
        X = np.asarray(X)
        out = np.zeros(len(X), bool)
        for A, b in self._hr:
            out |= ((X @ A.T - b) <= 0).all(1)
        return out

    def outside_depth(self, X):
        """per point, a LOWER bound of the distance to the union (0 if inside): min over members of the largest facet violation."""
        X = np.asarray(X)
        d = np.full(len(X), np.inf)
        for A, b in self._hr:
            d = np.minimum(d, np.maximum((X @ A.T - b).max(1), 0.0))
        return d

    def sd_points(self, X):
        """signed distance of points to the union: negative = inside (penetration depth into the best member), positive = exact
        Euclidean distance to the nearest member (bounded least squares)."""
        X = np.atleast_2d(X)
        best = np.full(len(X), np.inf)
        for z, (A, b) in zip(self.zs, self._hr):
            lb = (X @ A.T - b).max(1)
            sd = lb.copy()
            for k in np.flatnonzero(lb > 0):
                sd[k] = _dist_bvls(z, X[k], lb[k])
            best = np.minimum(best, sd)
        return best

    def volume_mc(self, lo, hi, rng, n=200000):
        X = rng.uniform(lo, hi, (n, 3))
        return float(self.contains(X).mean() * np.prod(hi - lo))

    def volume_sum(self):
        return float(sum(volume(z) for z in self.zs))

    def aabb(self):
        lo = np.min([z.interval_hull()[0] for z in self.zs], 0)
        hi = np.max([z.interval_hull()[1] for z in self.zs], 0)
        return lo, hi

    def inflate(self, r):
        """Minkowski sum with a cube of half-side r (outer approximation of the r-ball dilation): add 3 box generators."""
        from world_model.v2.tray_rep.sets.zono import Zonotope as Z
        return ZUnion([Z(z.c, np.concatenate([z.G, np.eye(3) * r], 1)) for z in self.zs])


def _dist_bvls(z: Zonotope, q, lb):
    G = z.G[:, np.linalg.norm(z.G, axis=0) > 1e-12]
    if G.shape[1] == 0:
        return float(np.linalg.norm(q - z.c))
    res = lsq_linear(G, q - z.c, bounds=(-1.0, 1.0), method="bvls", tol=1e-10, max_iter=200)
    return float(np.linalg.norm(G @ res.x - (q - z.c)))


def dist_union_union(U1: ZUnion, U2: ZUnion):
    """exact Euclidean distance between two unions of zonotopes (0 if any pair intersects)."""
    best = np.inf
    for z1 in U1.zs:
        for z2 in U2.zs:
            G = np.concatenate([z1.G, -z2.G], 1)
            keep = np.linalg.norm(G, axis=0) > 1e-12
            G = G[:, keep]
            d0 = z2.c - z1.c
            if G.shape[1] == 0:
                best = min(best, float(np.linalg.norm(d0)))
                continue
            # min_s |c1 + G1 s1 - c2 - G2 s2| = min_s |G s - d0| with G = [G1, -G2]... sign: c1+G1 s1 - (c2+G2 s2) = -d0 + G1 s1 - G2 s2
            res = lsq_linear(G, d0, bounds=(-1.0, 1.0), method="bvls", tol=1e-10, max_iter=300)
            best = min(best, float(np.linalg.norm(G @ res.x - d0)))
    return best
