"""checks/collision.py -- swept collision row (checker design 3.6 / K5).  PURE numpy: the pair distances come from an adapter (adapters/collision_rows.py).

Input per pair class ("arm_arm", "arm_world", "obj_world"):
  d_samples[c]   (n,)  min distance [m] over the pairs of the class at each sample k (negative = penetration)
  speed_bound[c] (n,)  [m/s] bound on the relative speed of the closest points of the pairs that can be binding, over the interval k -> k+1 (entry n-1 unused)
  d_min[c]       min allowed clearance [m]
  t              (n,)  sample times [s]
Swept lower bound on the clearance of class c anywhere in [t_k, t_{k+1}] (MODEL):

    lb_k = min(d_k, d_{k+1}) - v_k * (t_{k+1} - t_k) / 2

Why: the clearance is Lipschitz in time with constant v_k, so on an interval of length h the minimum over the interval is at least the smaller endpoint
value minus v_k * h/2 (the two endpoint cones of slope v_k meet at the mid-point; the bound is tight at a point where d_k = d_{k+1}).  v_k is a MODEL bound
(adapter): for every geom, |Delta x| + theta * r_bound over the interval, summed over the two geoms of the pair; it assumes the geoms follow the interpolation
between the two samples (linear in q, curvature error is second order in the 5 ms step).
Row margin [m] = min over classes and over samples AND intervals of (lower bound - d_min[c]).  proof when ok (MODEL above), refuted with the counterexample
(class, pair names, index, t, distance, bound) when not.  The samples themselves enter with lb = d_k (no sweep term).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.stack.checks.row import Row

CLASSES = ("arm_arm", "arm_world", "obj_world")


def swept_lower_bound(d, v, t):
    """-> (lb_samples (n,), lb_intervals (n-1,)): the samples' own distance and the swept lower bound on each interval."""
    d, t = np.asarray(d, float), np.asarray(t, float)
    v = np.asarray(v, float)[: len(d) - 1]
    return d, np.minimum(d[:-1], d[1:]) - v * np.diff(t) / 2.0


def swept_clearance_row(d_samples, speed_bound, t, d_min, name="collision", pair_names=None):
    """see the module docstring.  pair_names: optional {class: (n,) sequence of 'geomA|geomB'} naming the closest pair at each sample (for the counterexample)."""
    t = np.asarray(t, float)
    best, by_class = None, {}
    for c, d in d_samples.items():
        d = np.asarray(d, float)
        ls, li = swept_lower_bound(d, speed_bound[c], t)
        ms, mi = ls - d_min[c], li - d_min[c]
        ks, ki = int(np.argmin(ms)), (int(np.argmin(mi)) if len(mi) else -1)
        if len(mi) and mi[ki] < ms[ks]:
            k, m, lb, kind = ki, float(mi[ki]), float(li[ki]), "interval"
            d_here = float(min(d[ki], d[ki + 1]))
        else:
            k, m, lb, kind = ks, float(ms[ks]), float(ls[ks]), "sample"
            d_here = float(d[ks])
        names = None
        if pair_names is not None and c in pair_names:
            pn = pair_names[c]
            names = pn[k] if kind == "sample" or d[k] <= d[min(k + 1, len(d) - 1)] else pn[k + 1]
        w = dict(cls=c, pair=names, index=k, kind=kind, t=float(t[k]), distance=d_here, bound=lb, d_min=float(d_min[c]), margin=m)
        by_class[c] = w
        if best is None or m < best["margin"]:
            best = w
    if best is None:
        return Row(name, ok=False, margin=None, guarantee="missing", why="no pair class given")
    ok = best["margin"] >= 0.0
    return Row(name, ok=ok, margin=best["margin"], unit="m", guarantee="proof" if ok else "refuted", worst=dict(best, by_class=by_class), n=int(len(t)),
               assumptions=dict(d_min={c: float(v) for c, v in d_min.items()}, between_samples="MODEL lb = min(d_k, d_k+1) - v_k dt/2, v_k from geom |dx| + theta r_bound"),
               why="" if ok else f"{best['cls']} {best['pair']} distance {best['distance']:.4f} m, swept bound {best['bound']:.4f} m < d_min {best['d_min']:.4f} m "
                                 f"at t = {best['t']:.3f} s ({best['kind']} {best['index']})")
