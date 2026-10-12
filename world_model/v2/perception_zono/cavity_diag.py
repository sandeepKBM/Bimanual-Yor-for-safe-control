"""cavity_diag.py -- how much of a fitted zonotope's wasted volume do the cameras PROVE empty (ray carving)? SIMULATION ONLY.
For kinds with a cavity or a hole (cup, mug, dish, L): fit hz2 on the GT-segmented points, draw uniform samples inside the fit and compare
  true_empty : share of the fit that is truly outside the object (including the cavity)
  carved     : share of the fit that is ray-carved free (a ray crossed it with 10 mm margin)
  carved_correct : share of the carved points that are truly empty (should be ~1; < 1 means carving cuts real material)
  carved/true_empty : how much of the true waste is provable from this view
"""
from __future__ import annotations

import json
import sys
import zlib
from itertools import product

import numpy as np

from world_model.v2.perception_zono import geometry as G, sensing as S, segment as SG, fit as F, run_study as R
from world_model.v2.perception_zono.zonoutil import ZUnion


def one(args):
    kind, view, cond, s = args
    seed = R._seed("cav" + kind, view, cond, s)
    rng = np.random.default_rng(seed)
    x, y, yaw = R._pose(rng, kind)
    obj = G.proto(kind).moved(x, y, yaw)
    sc = G.Scene([obj])
    cl = S.capture(sc, view, cond, seed)
    pp = SG.preprocess(cl)
    P = pp["pts"][pp["label"] == 1]
    if len(P) < 8:
        return None
    U = ZUnion([F.fit_hull_zono(P, 2)])
    lo, hi = U.aabb()
    X = rng.uniform(lo, hi, (60000, 3))
    X = X[U.contains(X)][:6000]
    empty = ~obj.contains(X)
    free = S.known_free(cl.views, X, 0.010)
    return dict(kind=kind, view=view, cond=cond, true_empty=float(empty.mean()), carved=float(free.mean()),
                carved_correct=float(empty[free].mean()) if free.any() else float("nan"),
                recall=float((free & empty).sum() / max(empty.sum(), 1)))


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = list(product(["cup", "mug", "dish", "L", "box", "bottle"], ["head_aim", "head_aim+wrist", "head_hi"], ["nominal", "n3mm"], range(4)))
    with Pool(8) as p:
        res = [r for r in p.map(one, jobs) if r]
    out = R.OUT / "cavity_diag.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in res))
    agg = {}
    for r in res:
        agg.setdefault((r["kind"], r["view"]), []).append(r)
    print("kind view | true_empty carved carved_correct recall(share of true waste proven empty)")
    for (k, v), rr in sorted(agg.items()):
        m = lambda q: np.nanmean([x[q] for x in rr])
        print(f"{k:7s} {v:15s} | {m('true_empty'):.2f} {m('carved'):.2f} {m('carved_correct'):.2f} {m('recall'):.2f}")
