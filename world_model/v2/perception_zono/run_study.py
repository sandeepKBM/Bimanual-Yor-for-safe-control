"""run_study.py -- the driver. SIMULATION ONLY, CPU only, <= 8 worker processes.

  python -m world_model.v2.perception_zono.run_study single --workers 8 --seeds 5      # one object per scene, GT segments  -> trials_single.jsonl
  python -m world_model.v2.perception_zono.run_study mix --workers 8 --seeds 5          # several objects, GT vs DBSCAN segments -> trials_mix.jsonl
  python -m world_model.v2.perception_zono.run_study report                              # tables (markdown) from the jsonl files
Output dir: /common/users/ss5772/bimanual_data/zono_pc/results
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.perception_zono import geometry as G, sensing as S, segment as SG, methods as MT
from world_model.v2.perception_zono.score import Truth, score_fit
from world_model.v2.perception_zono.zonoutil import ZUnion, dist_union_union
from world_model.v2.perception_zono import fit as F

OUT = Path("/common/users/ss5772/bimanual_data/zono_pc/results")
KINDS = ["ball", "item", "box", "block", "bottle", "mug", "L", "cup", "dish", "tray"]
VIEWS = ["head_aim", "head_aim+wrist", "wrist_both", "head_hi"]
CONDS = ["ideal", "nominal", "n3mm"]


def _seed(kind, view, cond, s):
    return zlib.crc32(f"{kind}|{view}|{cond}|{s}".encode()) & 0x7FFFFFFF


def _pose(rng, kind):
    if kind == "tray":
        return 0.0, 0.0, 0.0
    return float(rng.uniform(0.0, 0.07)), float(rng.uniform(-0.04, 0.04)), float(rng.uniform(0, 2 * np.pi))


def _fit_all(P, views, with_pad=True):
    """-> {name: (ZUnion, ms)} for every method (pad variants derived from the base fits)."""
    out = {}
    for name, fn in MT.BASE.items():
        t = time.perf_counter()
        U = fn(P, views)
        out[name] = (U, (time.perf_counter() - t) * 1e3)
    if with_pad:
        for k in MT.PADDED:
            t = time.perf_counter()
            Up = ZUnion([F.inflate(z, MT.PAD) for z in out[k][0].zs])
            out[k + "+p"] = (Up, out[k][1] + (time.perf_counter() - t) * 1e3)
    return out


# ------------------------------------------------------------------------------------------------ single-object trials
def single_trial(args):
    kind, view, cond, s = args
    seed = _seed(kind, view, cond, s)
    rng = np.random.default_rng(seed)
    x, y, yaw = _pose(rng, kind)
    obj = G.proto(kind).moved(x, y, yaw)
    if kind == "tray":
        sc = G.Scene([], with_tray=True)
        tgt = 0
    else:
        sc = G.Scene([obj])
        tgt = 1
    t0 = time.perf_counter()
    cl = S.capture(sc, view, cond, seed)
    t_cap = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    pp = SG.preprocess(cl)
    t_prep = (time.perf_counter() - t0) * 1e3
    P = pp["pts"][pp["label"] == tgt]
    truth_obj = sc.tray if kind == "tray" else obj
    T = Truth(truth_obj, rng)
    T.set_visible(P)
    rec = dict(kind=kind, view=view, cond=cond, seed=s, n_raw=int((cl.label == tgt).sum()), n_pts=int(len(P)), t_cap=t_cap, t_prep=t_prep,
               coverage=T.view_coverage(P), vol=T.vol, vol_hull=T.vol_hull, methods={})
    if len(P) < 8:
        rec["fail"] = "no_points"
        return rec
    # reference: the convex hull of the observed points itself (not a zonotope)
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(P)
        A, b = h.equations[:, :3], -h.equations[:, 3]
        ins = ((T.surf @ A.T - b) <= 1e-9).all(1).mean()
        rec["hull_ref"] = dict(contain=float(ins), oc_true=float(h.volume / T.vol), oc_hull=float(h.volume / T.vol_hull))
    except Exception:
        rec["hull_ref"] = None
    for name, (U, ms) in _fit_all(P, cl.views).items():
        m = score_fit(U, T, rng)
        m["ms"] = ms
        rec["methods"][name] = m
    return rec


# ------------------------------------------------------------------------------------------------ mixed scenes
def _stack_blocks(x, y, yaw):
    objs = []
    for i in range(3):
        p = G.Prim("box", (0.025 - 3e-4 * i, 0.025 - 2.5e-4 * i, 0.02), np.array([0.003 * ((-1) ** i), 0.002 * i, 0.02 + 0.04 * i + 1e-4 * (i + 1)]))
        objs.append(G.Obj(f"blk{i}", "block", [p]).moved(x, y, yaw))
    return objs


def mix_scene(name, rng):
    """-> list of placed Obj (positions jittered a little with rng)."""
    j = lambda a=0.004: float(rng.uniform(-a, a))
    if name == "mix4":
        return [G.proto("box").moved(0.02 + j(), 0.062 + j(), 0.3), G.proto("bottle").moved(0.10 + j(), -0.035 + j(), 0.0),
                G.proto("mug").moved(0.02 + j(), -0.055 + j(), np.pi), G.proto("ball").moved(0.115, 0.06 + j(), 0.0)]
    if name == "tower3":
        return _stack_blocks(0.04 + j(), j(), 0.4)
    if name == "cupbeads":
        cx, cy = 0.04 + j(), j()
        objs = [G.proto("cup").moved(cx, cy, 0.0)] + [G.bead(i).moved(cx, cy, 0.0) for i in range(8)]
        return objs
    if name == "dishitems":
        cx, cy = 0.05 + j(), j()
        return [G.proto("dish").moved(cx, cy, 0.0)] + [G.dish_item(i).moved(cx, cy, 0.0) for i in range(6)]
    if name == "Lblock":
        x, y, yaw = 0.04, 0.0, float(rng.uniform(0, 2 * np.pi))
        blk = G.Obj("blk", "block", [G.Prim("box", (0.025, 0.025, 0.02), np.array([0.017, 0.017, 0.02]))])
        return [G.proto("L").moved(x, y, yaw), blk.moved(x, y, yaw)]
    raise KeyError(name)


MIX = ["mix4", "tower3", "cupbeads", "dishitems", "Lblock"]
SEGS = ["gt", "dbscan_xyz", "dbscan_xyzn", "dbscan_xyz_kin"]
MIX_METHODS = ["ubox", "hz2", "split_pa", "split_pa_fs"]


def mix_trial(args):
    name, view, cond, s = args
    seed = _seed(name, view, cond, s)
    rng = np.random.default_rng(seed)
    objs = mix_scene(name, rng)
    pd = G.pair_distances(objs, rng)
    sc = G.Scene(objs)
    cl = S.capture(sc, view, cond, seed)
    pp = SG.preprocess(cl)
    gt = pp["label"]
    oids = [o.oid for o in sc.objs]
    Ts = {o.oid: Truth(o, rng, n_surf=3000, n_q=0, n_mc=60000) for o in sc.objs}
    rec = dict(scene=name, view=view, cond=cond, seed=s, n_obj=len(objs), n_pts=int(len(pp["pts"])), segs={})
    for seg in SEGS:
        t0 = time.perf_counter()
        if seg == "gt":
            lab = gt.copy()
            lab[lab == 0] = -2
            lab = np.where(lab > 0, lab - 1, lab)               # cluster id = oid - 1
            sm = None
        else:
            kin = (np.array([0.0, 0.0, 1.0]), 1.0 + float(rng.normal(0, 0.002)), 0.003) if seg.endswith("_kin") else None   # tray plane from FK, 2 mm error
            lab, info = SG.unsup_segments(pp, cl.views, rng, use_normals=(seg == "dbscan_xyzn"), plane=kin)
            sm = SG.seg_metrics(gt, lab)
        t_seg = (time.perf_counter() - t0) * 1e3
        # per-object point sets: GT cluster = object id; unsupervised = best-IoU cluster
        if seg == "gt":
            sets = {o: np.flatnonzero(gt == o) for o in oids}
        else:
            sets = SG.match_clusters(gt, lab, oids)
        r = dict(t_seg_ms=t_seg, seg=sm, objs={}, pairs={})
        fits = {}
        for o in oids:
            idx = sets[o]
            if idx is None or len(idx) < 8:
                r["objs"][o] = dict(fail="no_points", n=0 if idx is None else int(len(idx)))
                continue
            P = pp["pts"][idx]
            T = Ts[o]
            ro = dict(n=int(len(P)), purity=float((gt[idx] == o).mean()), methods={})
            for mname in MIX_METHODS:
                t = time.perf_counter()
                U = MT.BASE[mname](P, cl.views)
                ms = (time.perf_counter() - t) * 1e3
                fits[(o, mname)] = U
                T_ = T
                ro["methods"][mname] = dict(contain=float(U.contains(T_.surf).mean()), oc_hull=float(U.volume_sum() / T_.vol_hull), ms=ms)
            r["objs"][o] = ro
        for (i, j), d_true in pd.items():
            oi, oj = objs[i].oid, objs[j].oid
            pr = {}
            for mname in MIX_METHODS:
                if (oi, mname) in fits and (oj, mname) in fits:
                    d = dist_union_union(fits[(oi, mname)], fits[(oj, mname)])
                    pr[mname] = d
            r["pairs"][f"{oi}-{oj}"] = dict(d_true=d_true, d_fit=pr)
        rec["segs"][seg] = r
    return rec


# ------------------------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["single", "mix", "report"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--scenes", default=",".join(MIX))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert a.workers <= 8
    if a.what == "report":
        from world_model.v2.perception_zono.report import make_report
        make_report(OUT, a.tag)
        return
    if a.what == "single":
        jobs = list(product(a.kinds.split(","), a.views.split(","), a.conds.split(","), range(a.seeds)))
        fn, path = single_trial, OUT / f"trials_single{a.tag}.jsonl"
    else:
        jobs = list(product(a.scenes.split(","), a.views.split(","), a.conds.split(","), range(a.seeds)))
        fn, path = mix_trial, OUT / f"trials_mix{a.tag}.jsonl"
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=20) as pool, open(path, "w") as f:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            if i % 20 == 0:
                f.flush()
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", len(jobs), f"{time.time() - t0:.0f}s ->", path, flush=True)


if __name__ == "__main__":
    main()
