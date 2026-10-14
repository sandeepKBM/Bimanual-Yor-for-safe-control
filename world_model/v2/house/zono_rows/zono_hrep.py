"""zono_hrep.py -- H-representation of a 3-D zonotope + signed-distance bounds.  Pure numpy/scipy.  SIMULATION ONLY.

A zonotope  Z = { c + G s : |s_i| <= 1 }  (G in R^{3 x m}) is a convex polytope.  Its facets come in parallel pairs, one pair per
(non-parallel) generator pair (i, j), with outward normal  n = +-(g_i x g_j)/|g_i x g_j|  and offset  d = n.c + sum_k |n.g_k|
(the support function of Z in direction n), so   Z = { x : n_f . x <= d_f  for all facets f }.

Signed-distance lower bound (the quantity the HOCBF rows use):

    sd_lb(p) = max_f ( n_f . p - d_f )

  * inside Z : exact (= minus the distance to the nearest facet plane = minus the penetration depth, the inradius-style depth);
  * outside Z: a LOWER BOUND of the Euclidean distance dist(p, Z) -- every facet half-space contains Z, so
                dist(p, Z) >= dist(p, {n_f.x <= d_f}) = n_f.p - d_f for every f.  It is exact when the nearest point of Z lies in the
                relative interior of the active facet and under-estimates (conservative: reports LESS clearance) in the Voronoi
                regions of edges and vertices (largest relative gap just off a vertex of a box: |p - v| vs max_f component).
  * Staying in the active facet's outer half-space {n_a.x >= d_a + margin} keeps the point outside Z, which is why one linear
    barrier per (point, zonotope) is a sound sufficient condition for non-penetration (a union of half-space escapes).  The active
    facet can SWITCH when a point moves around an edge/vertex (the barrier is then only piecewise smooth); the rows are recomputed
    every control step so the switch is followed one step later.

Degenerate zonotopes (rank < 3: flat / segment / point) are made full-dimensional by adding +-eps generators along the missing
directions (outer approximation, eps = DEG_EPS m), so a 2-D patch is a very thin slab.  Zero generators are dropped, parallel
generators are merged (exact: g and k g sum to (1+|k|) g direction-wise), parallel facet normals are de-duplicated.
Cost: m generators -> at most m(m-1) facets (m = 10 -> <= 90).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEG_EPS = 1e-3          # m   thickness added along missing directions of a rank-deficient zonotope
GEN_TOL = 1e-9          # m   generators shorter than this are dropped
PAR_TOL = 1e-9          #     |sin angle| below which two generators are treated as parallel (merged)


@dataclass
class HRep:
    c: np.ndarray       # (3,)    centre
    G: np.ndarray       # (3, m') merged / augmented generators actually used
    N: np.ndarray       # (F, 3)  unit outward facet normals
    d: np.ndarray       # (F,)    offsets, Z = {x : N x <= d}
    lo: np.ndarray      # (3,)    axis-aligned hull
    hi: np.ndarray      # (3,)

    @property
    def n_facets(self) -> int:
        return int(self.N.shape[0])


def merge_parallel(G, tol=PAR_TOL):
    """drop zero generators, merge parallel/antiparallel ones (exact for a zonotope).  G (3, m) -> (3, m')."""
    G = np.asarray(G, float).reshape(3, -1)
    G = G[:, np.linalg.norm(G, axis=0) > GEN_TOL]
    out = []
    for k in range(G.shape[1]):
        g = G[:, k]
        for j, h in enumerate(out):
            if np.linalg.norm(np.cross(g, h)) <= tol * np.linalg.norm(g) * np.linalg.norm(h):
                s = 1.0 if float(g @ h) >= 0 else -1.0
                out[j] = h + s * g
                break
        else:
            out.append(g.copy())
    return np.array(out).T.reshape(3, -1)


def zono_hrep(c, G, eps=DEG_EPS) -> HRep:
    """(c, G) -> HRep.  See the module docstring for the degenerate handling."""
    c = np.asarray(c, float).reshape(3)
    G = merge_parallel(G)
    if G.shape[1] == 0:
        G = np.zeros((3, 0))
    rank = int(np.linalg.matrix_rank(G, tol=1e-9)) if G.shape[1] else 0
    if rank < 3:
        U = np.linalg.svd(G if G.shape[1] else np.zeros((3, 1)), full_matrices=True)[0]
        G = np.concatenate([G, eps * U[:, rank:]], 1)          # missing directions get a +-eps slab
        G = merge_parallel(G)
    m = G.shape[1]
    i, j = np.triu_indices(m, 1)
    X = np.cross(G[:, i].T, G[:, j].T)
    nn = np.linalg.norm(X, axis=1)
    X = X[nn > 1e-12] / nn[nn > 1e-12, None]
    # de-duplicate parallel normals (sign-canonical rounding)
    s = np.where(X[np.arange(len(X)), np.abs(X).argmax(1)] < 0, -1.0, 1.0)[:, None]
    Xc = X * s
    _, u = np.unique(np.round(Xc, 7), axis=0, return_index=True)
    Xc = Xc[np.sort(u)]
    N = np.concatenate([Xc, -Xc])
    d = N @ c + np.abs(N @ G).sum(1)
    ext = np.abs(G).sum(1)
    return HRep(c=c, G=G, N=N, d=d, lo=c - ext, hi=c + ext)


def hrep_of(z) -> HRep:
    """accepts an HRep, a (c, G) tuple, or anything with .c and .G (tray_rep Zonotope)."""
    if isinstance(z, HRep):
        return z
    if isinstance(z, tuple):
        return zono_hrep(*z)
    return zono_hrep(z.c, z.G)


def facet_slack(h: HRep, P):
    """(K, F) matrix  N p - d  (<= 0 inside facet f)."""
    return np.atleast_2d(P) @ h.N.T - h.d


def sd_lb(h: HRep, P, return_active=False):
    """conservative signed distance (see module docstring) of points P (K,3): max_f (n_f.p - d_f)."""
    S = facet_slack(h, P)
    k = S.argmax(1)
    v = S[np.arange(len(S)), k]
    return (v, k) if return_active else v


def vertices(h: HRep, max_gen=14):
    """all 2^m vertex candidates c + G s, s in {-1,1}^m (superset of the true vertices; m <= max_gen)."""
    G = h.G
    m = G.shape[1]
    if m > max_gen:
        raise ValueError(f"{m} generators: vertex enumeration disabled above {max_gen}")
    S = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1) * 2.0 - 1.0
    return h.c + S @ G.T


def dist_exact(h: HRep, p) -> float:
    """exact Euclidean distance from p to Z (0 if inside) via the bounded least-squares QP  min_{|s|<=1} |c + G s - p|.
    For validation (not used in the control loop)."""
    from scipy.optimize import lsq_linear
    p = np.asarray(p, float)
    r = lsq_linear(h.G, p - h.c, bounds=(-1.0, 1.0), method="bvls", tol=1e-12, max_iter=500)
    return float(np.linalg.norm(h.G @ r.x - (p - h.c)))


def sd_exact(h: HRep, p) -> float:
    """exact signed distance: +dist outside, minus the penetration depth (min over facets of d_f - n_f.p) inside."""
    p = np.asarray(p, float)
    v = float((h.N @ p - h.d).max())
    return v if v <= 0 else dist_exact(h, p)
