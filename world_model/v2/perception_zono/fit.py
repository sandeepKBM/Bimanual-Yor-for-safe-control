"""fit.py -- point set -> zonotope / union of zonotopes. SIMULATION ONLY, numpy / scipy.

  fit_box(P, frame)            order-1 zonotope = oriented box. frame 'pca' (principal axes of the points) or 'up' (gravity z fixed, yaw by
                               min-area rectangle of the xy projection: the robot knows gravity from the IMU)
  fit_hull_zono(P, order, ..)  outer zonotope of conv(P): choose a rich generator-direction set (17 directions in the object frame), solve the
                               exact LP   min  sum_i w_i lam_i   s.t.  h_Z(n) >= h_P(n)  for every facet normal n = +-(u_i x u_j)   (c free)
                               (for a fixed direction set the facet normals of Z do not depend on lam, so the LP is EXACT containment of conv(P)),
                               re-weighting by the volume gradient a few times (sequential LP), then Girard order reduction (done in the object
                               frame so the box generators that replace the dropped ones are object-aligned) to `order` (generators = 3*order).
  fit_split(P, ..)            union of zonotopes by recursive splitting where one zonotope wastes volume: a plane is accepted when
                               vol(hull(S1)) + vol(hull(S2)) < (1 - gain) vol(hull(S)); planes proposed by quantiles of the coordinates and/or by a
                               polynomial-annihilation (PA) edge detector on the cross-section-area profile along each frame axis.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from world_model.v2.tray_rep.sets.zono import Zonotope
from world_model.v2.perception_zono.zonoutil import volume


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def object_frame(P, frame="up"):
    """(origin, R) with columns of R the frame axes in world coordinates; coordinates Q = (P - origin) @ R."""
    o = P.mean(0)
    if frame == "pca":
        w, V = np.linalg.eigh(np.cov((P - o).T))
        R = V[:, ::-1].copy()
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        return o, R
    xy = P[:, :2] - o[:2]
    best, bt = np.inf, 0.0
    for t in np.deg2rad(np.arange(0, 90, 1.0)):
        c, s = np.cos(t), np.sin(t)
        u, v = xy @ [c, s], xy @ [-s, c]
        a = (u.max() - u.min()) * (v.max() - v.min())
        if a < best:
            best, bt = a, t
    return o, _Rz(bt)


def _box_in_frame(Q):
    lo, hi = Q.min(0), Q.max(0)
    return (lo + hi) / 2, np.diag(np.maximum((hi - lo) / 2, 5e-4))


def fit_box(P, frame="up"):
    if len(P) < 4:
        return Zonotope(P.mean(0) if len(P) else np.zeros(3), np.eye(3) * 1e-3)
    o, R = object_frame(P, frame)
    c, G = _box_in_frame((P - o) @ R)
    return Zonotope(o + R @ c, R @ G)


# ------------------------------------------------------------------------------------------------ hull zonotope
def _dirs_rich():
    D = [np.eye(3)[i] for i in range(3)]
    for i, j in itertools.combinations(range(3), 2):
        for s in (1.0, -1.0):
            v = np.zeros(3)
            v[i], v[j] = 1.0, s
            D.append(v)
    for s in itertools.product([1.0], [1.0, -1.0], [1.0, -1.0]):
        D.append(np.array(s))
    for t in np.pi * np.array([1, 3, 5, 7]) / 8:
        D.append(np.array([np.cos(t), np.sin(t), 0.0]))
    D = np.array(D)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


_D = _dirs_rich()
_PAIR = np.array(list(itertools.combinations(range(len(_D)), 2)))
_N = np.cross(_D[_PAIR[:, 0]], _D[_PAIR[:, 1]])
_N = _N / np.linalg.norm(_N, axis=1, keepdims=True)
_key = np.round(np.where(_N[:, [0]] < -1e-9, -_N, _N), 5)
_N = _N[np.sort(np.unique(_key, axis=0, return_index=True)[1])]
_NN = np.concatenate([_N, -_N])
_ABS = np.abs(_NN @ _D.T)                                    # (K, m): |u_i . n_k|
_TRI = np.array(list(itertools.combinations(range(len(_D)), 3)))
_DET = np.abs(np.linalg.det(np.stack([_D[_TRI[:, 0]], _D[_TRI[:, 1]], _D[_TRI[:, 2]]], 2)))


def _vol_grad(lam):
    g = np.zeros(len(lam))
    t = lam[_TRI[:, 0]] * lam[_TRI[:, 1]] * lam[_TRI[:, 2]] * 0 + 0
    for a, b, c in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        np.add.at(g, _TRI[:, a], 8 * lam[_TRI[:, b]] * lam[_TRI[:, c]] * _DET)
    return g


def _vol(lam):
    return float(8 * (lam[_TRI[:, 0]] * lam[_TRI[:, 1]] * lam[_TRI[:, 2]] * _DET).sum())


def hull_lp(Q, n_it=4, hull_verts=True):
    """Q (n,3) points in the object frame -> (c (3,), lam (m,)) for the zonotope c + sum lam_i u_i [-1,1] containing conv(Q). Returns None on failure."""
    V = Q
    if hull_verts and len(Q) > 12:
        try:
            V = Q[ConvexHull(Q).vertices]
        except Exception:
            V = Q
    hP = (V @ _NN.T).max(0)                                  # (K,)
    m = len(_D)
    A = np.concatenate([-_NN, -_ABS], 1)                     # -n.c - sum lam |u.n| <= -h_P(n)
    b = -hP
    bounds = [(None, None)] * 3 + [(0, None)] * m
    w = np.ones(m)
    best = None
    for it in range(n_it):
        res = linprog(np.concatenate([np.zeros(3), w]), A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if not res.success:
            break
        c, lam = res.x[:3], res.x[3:]
        v = _vol(lam)
        if best is None or v < best[2] - 1e-15:
            best = (c, lam, v)
        else:
            break
        g = _vol_grad(lam)
        w = g / max(g.mean(), 1e-30) * 0.9 + 0.1 * np.ones(m)
        w = np.maximum(w, 1e-3)
    return None if best is None else (best[0], best[1])


def fit_hull_zono(P, order=2, frame="up", n_it=4, lam_tol=1e-6):
    """order = None: no reduction (up to 17 generators). Girard reduction to 3*order generators otherwise."""
    if len(P) < 8:
        return fit_box(P, frame)
    o, R = object_frame(P, frame)
    Q = (P - o) @ R
    r = hull_lp(Q, n_it)
    if r is None:
        return fit_box(P, frame)
    c, lam = r
    keep = lam > lam_tol * max(lam.max(), 1e-12)
    Z = Zonotope(c, (_D[keep] * lam[keep][:, None]).T)
    if order is not None and Z.n_gen > 3 * order:
        Z = Z.reduce(3 * order)
    return Zonotope(o + R @ Z.c, R @ Z.G)


# ------------------------------------------------------------------------------------------------ PA edge detector
def pa_jump(f, m=2):
    """polynomial-annihilation jump function L_m f on a uniform grid (Archibald-Chen-Gelb-Renaut), minmod across the stencils that
    contain a point. f (B,) -> L (B,). Large |L| = edge (jump) of f; ~0 on smooth parts."""
    B = len(f)
    x = np.arange(B, dtype=float)
    fact = float(np.prod(np.arange(1, m + 1)))
    out = np.zeros(B)
    stenc = []
    for k in range(B - m):
        xs = x[k:k + m + 1]
        cvec = np.array([fact / np.prod([xs[i] - xs[l] for l in range(m + 1) if l != i]) for i in range(m + 1)])
        stenc.append((k, xs, cvec))
    for j in range(B):
        vals = []
        for k, xs, cvec in stenc:
            if k <= j <= k + m:
                qm = cvec[xs >= j].sum()
                if abs(qm) < 1e-12:
                    continue
                vals.append((cvec @ f[k:k + m + 1]) / qm)
        if not vals:
            continue
        vals = np.array(vals)
        out[j] = vals[np.argmin(np.abs(vals))] if (vals > 0).all() or (vals < 0).all() else 0.0
    return out


def profile_area(Q, axis, nb=32):
    """cross-section area proxy a(t) of the points along a frame axis: bounding-rectangle area of the other two coordinates per bin."""
    t = Q[:, axis]
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-4:
        return None, None
    edges = np.linspace(lo, hi, nb + 1)
    ib = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nb - 1)
    o = [i for i in range(3) if i != axis]
    a = np.zeros(nb)
    for k in range(nb):
        s = Q[ib == k][:, o]
        if len(s) >= 3:
            a[k] = np.ptp(s[:, 0]) * np.ptp(s[:, 1])
    return a, edges


def pa_candidates(Q, thr=0.25, nb=32, m=2):
    """candidate split planes {(axis, t)} from PA edges of the area profile along each frame axis."""
    out = []
    for ax in range(3):
        a, edges = profile_area(Q, ax, nb)
        if a is None or a.max() <= 0:
            continue
        L = pa_jump(a / a.max(), m)
        flag = np.abs(L) > thr
        j = 0
        while j < nb:
            if flag[j]:
                k = j
                while k + 1 < nb and flag[k + 1]:
                    k += 1
                jj = j + int(np.argmax(np.abs(L[j:k + 1])))
                out.append((ax, float(edges[jj])))
                j = k + 1
            else:
                j += 1
    return out


# ------------------------------------------------------------------------------------------------ split
def _hull_vol(Q):
    if len(Q) < 6:
        return 0.0
    try:
        return float(ConvexHull(Q).volume)
    except Exception:
        return 0.0


def _hull_eq(Q):
    if len(Q) < 6:
        return None
    try:
        return ConvexHull(Q).equations
    except Exception:
        return None


def _in_hull(X, eq):
    if eq is None:
        return np.zeros(len(X), bool)
    return ((X @ eq[:, :3].T + eq[:, 3]) <= 1e-9).all(1)


def fit_split(P, cand="quantile", base_order=2, frame="up", gain=0.15, max_depth=3, min_pts=30, max_parts=6, overlap=0.0,
              max_pts=2500, rng=None, criterion="hull", views=None, free_margin=0.010, n_mc=3000):
    """returns (list of Zonotope, info). Splits in the object frame; leaf fit = fit_hull_zono(order=base_order) on the leaf points.
    criterion 'hull' : accept a plane when vol(hull(S1)) + vol(hull(S2)) < (1 - gain) vol(hull(S)).  Points only. For a PARTIAL view of a convex
                       object this is fooled (the visible surface sheets are 'thin'): see the report.
    criterion 'free' : accept when the part of hull(S) that the cameras have ray-carved as FREE space (a ray passed through it, margin `free_margin`)
                       and that the two child hulls no longer cover is more than `gain` of vol(hull(S)). Needs `views` (sensing.CamView list)."""
    from world_model.v2.perception_zono.sensing import known_free
    info = dict(n_eval=0)
    if len(P) < 2 * min_pts:
        return [fit_hull_zono(P, base_order, frame)], info
    o, R = object_frame(P, frame)
    Q = (P - o) @ R
    rng = rng or np.random.default_rng(0)
    parts = [(Q, 0)]
    done = []

    def propose(S):
        c = []
        if cand in ("quantile", "both"):
            for ax in range(3):
                for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                    c.append((ax, float(np.quantile(S[:, ax], q))))
        if cand in ("pa", "both"):
            c += pa_candidates(S)
        return c

    while parts:
        S, depth = parts.pop()
        if depth >= max_depth or len(S) < 2 * min_pts or len(done) + len(parts) + 1 >= max_parts:
            done.append(S)
            continue
        Sd = S if len(S) <= max_pts else S[rng.choice(len(S), max_pts, replace=False)]
        best = None
        if criterion == "hull":
            v0 = _hull_vol(Sd)
            score0 = (1 - gain) * v0
            sc = lambda s1, s2: _hull_vol(s1) + _hull_vol(s2)          # lower is better
        else:
            eq0 = _hull_eq(Sd)
            if eq0 is None:
                done.append(S)
                continue
            lo, hi = Sd.min(0), Sd.max(0)
            X = rng.uniform(lo, hi, (n_mc, 3))
            ins0 = _in_hull(X, eq0)
            if ins0.sum() < 200:
                done.append(S)
                continue
            free = known_free(views, o + X @ R.T, free_margin) & ins0
            n0 = ins0.sum()
            score0 = -gain                                              # score = - removed free fraction; accept if < -gain
            sc = lambda s1, s2: -float((free & ~(_in_hull(X, _hull_eq(s1)) | _in_hull(X, _hull_eq(s2)))).sum() / n0)
        for ax, t in propose(Sd):
            s1, s2 = Sd[Sd[:, ax] <= t + overlap], Sd[Sd[:, ax] >= t - overlap]
            if len(s1) < min_pts or len(s2) < min_pts:
                continue
            v = sc(s1, s2)
            info["n_eval"] += 1
            if best is None or v < best[0]:
                best = (v, ax, t)
        if best is not None and best[0] < score0:
            _, ax, t = best
            parts.append((S[S[:, ax] <= t + overlap], depth + 1))
            parts.append((S[S[:, ax] >= t - overlap], depth + 1))
        else:
            done.append(S)
    zs = []
    for S in done:
        Z = fit_hull_zono(S @ R.T + o, base_order, frame)         # leaf fit in the leaf's own frame
        zs.append(Z)
    info["n_parts"] = len(zs)
    return zs, info


def inflate(Z: Zonotope, r):
    """outer approximation of Z (+) ball(r): add three world-axis box generators of half-length r."""
    return Zonotope(Z.c, np.concatenate([Z.G, np.eye(3) * r], 1))
