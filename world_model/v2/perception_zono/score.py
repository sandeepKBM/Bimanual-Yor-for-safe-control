"""score.py -- truth bundle for an object and the metrics of a fitted ZUnion against it. SIMULATION ONLY.

metrics:
  contain      fraction of the TRUE surface points (6000 uniform samples of the union boundary) inside the fitted set
  leak_p95     95th percentile (mm) of the outside distance of the true surface points (a lower bound: largest facet violation of the best part)
  oc_true      vol(fitted) / vol(true object)    (true = union of primitives; for a thin shell this is huge by construction)
  oc_hull      vol(fitted) / vol(convex hull of the true object)   (what a convex fit can hope for; 1.0 = as tight as a convex set)
  clear_*      signed-distance error of Clear(object, query point): e = d_fit - d_true over 200 query points with 0 < d_true <= 10 cm.
               e > 0 means the fitted set claims MORE clearance than the truth (unsafe). mean / mae in mm; unsafe = fraction with e > 2 mm.
  inside_err   Inside(object, tray footprint) margin error (fit - true), mm; positive = unsafe. Facets: the four tray edges.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from world_model.v2.perception_zono.geometry import TRAY_HX, TRAY_HY
from world_model.v2.perception_zono.zonoutil import ZUnion

_NI = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])
_BI = np.array([TRAY_HX, TRAY_HX, TRAY_HY, TRAY_HY])


class Truth:
    def __init__(self, obj, rng, n_surf=6000, n_q=200, dmax=0.10, n_mc=150000):
        self.obj = obj
        self.surf = obj.surface(n_surf, rng)
        self.vol = obj.volume(rng, n_mc)
        try:
            self.vol_hull = float(ConvexHull(self.surf).volume)
        except Exception:
            self.vol_hull = float("nan")
        base = self.surf[rng.choice(len(self.surf), 4 * n_q)]
        dirs = rng.normal(size=base.shape)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        Q = base + dirs * rng.uniform(0.0, dmax, (len(base), 1))
        d = obj.sdf(Q)
        ok = (d > 5e-4) & (d <= dmax)
        self.Q, self.d = Q[ok][:n_q], d[ok][:n_q]
        self.m_inside = float(np.min(_BI - (self.surf @ _NI.T).max(0)))
        self.tree = cKDTree(self.surf)

    def set_visible(self, P, tol=0.008):
        """mask of the true surface points that the view actually observed (within tol of an observed point)."""
        if len(P) == 0:
            self.vis = np.zeros(len(self.surf), bool)
        else:
            d, _ = cKDTree(P).query(self.surf)
            self.vis = d < tol

    def view_coverage(self, P, tol=0.006):
        """fraction of the true surface within tol of an observed point (how much of the object the view actually saw)."""
        if len(P) == 0:
            return 0.0
        d, _ = cKDTree(P).query(self.surf)
        return float((d < tol).mean())


def score_fit(U: ZUnion, T: Truth, rng):
    s = {}
    s["n_parts"] = len(U.zs)
    s["n_gen"] = U.n_gen
    inn = U.contains(T.surf)
    s["contain"] = float(inn.mean())
    if getattr(T, "vis", None) is not None:
        s["contain_vis"] = float(inn[T.vis].mean()) if T.vis.any() else float("nan")
        s["contain_hid"] = float(inn[~T.vis].mean()) if (~T.vis).any() else float("nan")
    od = U.outside_depth(T.surf)
    s["leak_p95_mm"] = float(np.percentile(od, 95) * 1e3)
    s["leak_max_mm"] = float(od.max() * 1e3)
    if len(U.zs) == 1:
        from world_model.v2.perception_zono.zonoutil import volume
        vf = volume(U.zs[0])
    else:
        lo, hi = U.aabb()
        vf = U.volume_mc(lo, hi, rng, 100000)
    s["vol_fit"] = vf
    s["oc_true"] = vf / T.vol
    s["oc_hull"] = vf / T.vol_hull
    if len(T.Q):
        e = U.sd_points(T.Q) - T.d
        s["clear_mean_mm"] = float(e.mean() * 1e3)
        s["clear_mae_mm"] = float(np.abs(e).mean() * 1e3)
        s["clear_unsafe"] = float((e > 0.002).mean())
        s["clear_p95_mm"] = float(np.percentile(e, 95) * 1e3)
        near = T.d < 0.03
        s["clear_near_mean_mm"] = float(e[near].mean() * 1e3) if near.any() else float("nan")
        s["clear_near_unsafe"] = float((e[near] > 0.002).mean()) if near.any() else float("nan")
    hf = np.max([[z.support(n) for n in _NI] for z in U.zs], axis=0)
    s["inside_err_mm"] = float((np.min(_BI - hf) - T.m_inside) * 1e3)
    return s
