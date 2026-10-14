"""grounding/downstream.py -- does a VLM box segment the depth point cloud? (SIMULATION, CPU)
Tabletop: clouds from perception_zono.sensing (n3mm / nominal, same ray-cast rig as the DBSCAN study) for the same scene+view as the VLM image.
House: cloud from the MuJoCo depth render (stride 2, 3 mm iid noise, 2 % dropout), voxel 1 cm.
Methods (all score the best-IoU segment against the GT object's points after the same preprocess; recovered = IoU >= 0.5):
  dbscan_ransac  perception_zono segment.unsup_segments default (RANSAC tray plane + DBSCAN)          [baseline 1]
  dbscan_kin     same with the tray plane given by kinematics (z = 1.0, thr 8 mm)                       [baseline 2, tabletop only]
  gtbox          GT visible-mask bbox -> points projecting into it -> cleanup                            [pipeline upper bound]
  box:<model>    the model's first predicted box -> same pipeline
cleanup: tabletop = drop tray-plane points (kin), DBSCAN (eps 9 mm, min 6) on the in-box points, keep the largest cluster;
         house = depth-gap split of the in-box points (gap 4 cm), keep the NEAREST group holding >= 10 % of the box points (>= 15 pts).
Optional: fit_hull_zono on each segment, containment of the true object surface (tabletop; scene regenerated from its seed)."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

from world_model.v2.perception_zono import segment as SG
from world_model.v2.vlm.grounding import common as C
from world_model.v2.vlm.grounding.make_tabletop import F_PX, build_scene, project
from world_model.v2.vlm.grounding.make_requests import MODELS
from world_model.v2.vlm.grounding.score import load_meta

KIN_PLANE = (np.array([0.0, 0.0, 1.0]), 1.0, 0.003)


def cleanup_tab(P):
    if len(P) < 6:
        return np.zeros(0, int)
    lab = SG.dbscan(P, 0.009, 6)
    if (lab >= 0).sum() == 0:
        return np.zeros(0, int)
    return np.flatnonzero(lab == np.bincount(lab[lab >= 0]).argmax())


def cleanup_house(z):
    """z: z-depths of the in-box points -> indices of the nearest substantial depth group."""
    if len(z) < 15:
        return np.arange(len(z))
    o = np.argsort(z)
    zs = z[o]
    cuts = np.flatnonzero(np.diff(zs) > 0.04) + 1
    groups = np.split(o, cuts)
    for g in groups:
        if len(g) >= max(15, 0.10 * len(z)):
            return g
    return max(groups, key=len)


def in_box(pts, origin, R, f_px, box, W=C.IMG_W, H=C.IMG_H):
    uv, z = project(pts, origin, R, f_px, W, H)
    return (z > 0.05) & (uv[:, 0] >= box[0]) & (uv[:, 0] <= box[2]) & (uv[:, 1] >= box[1]) & (uv[:, 1] <= box[3])


def iou_sets(S, G):
    S, G = np.asarray(S, int), np.asarray(G, int)
    if len(S) == 0 or len(G) == 0:
        return 0.0
    i = np.intersect1d(S, G).size
    return i / (len(S) + len(G) - i)


def best_cluster_iou(lab, gt_idx):
    ncl = int(lab.max() + 1) if (lab >= 0).any() else 0
    if ncl == 0:
        return 0.0, None
    ov = np.bincount(lab[gt_idx][lab[gt_idx] >= 0], minlength=ncl)
    csz = np.bincount(lab[lab >= 0], minlength=ncl)
    iou = ov / (len(gt_idx) + csz - ov)
    c = int(iou.argmax())
    return float(iou[c]), np.flatnonzero(lab == c)


def model_boxes(models):
    """{model: {(image id, qkey): [x1,y1,x2,y2] px}} from the scored records."""
    out = {}
    for m in models:
        p = C.DATA / "results" / f"records_{m}.jsonl"
        if not p.exists():
            continue
        d = {}
        for r in C.read_jsonl(p):
            if r["task"] == "box" and r.get("box_px") is not None:
                iid, _, qk = r["id"].split("|")
                d[(iid, qk)] = r["box_px"]
        out[m] = d
    return out


def zono_contain(P, T):
    from world_model.v2.perception_zono import fit as F
    from world_model.v2.perception_zono.zonoutil import ZUnion
    if len(P) < 8:
        return None
    try:
        return float(ZUnion([F.fit_hull_zono(P, order=2)]).contains(T.surf).mean())
    except Exception:
        return None


def run_tab(meta, boxes, conds=("n3mm", "nominal"), zono=True):
    from world_model.v2.perception_zono.score import Truth
    rows = []
    for iid, im in meta.items():
        if im["domain"] != "tab":
            continue
        c = np.load(im["cloud"])
        objs, _, _ = build_scene(im["scene"], im["seed"])
        for cond in conds:
            cl = SimpleNamespace(pts=c[f"{cond}_pts"], label=c[f"{cond}_label"], cam=c[f"{cond}_cam"])
            pp = SG.preprocess(cl)
            P, L = pp["pts"], pp["label"]
            rng = np.random.default_rng(0)
            lab_r, _ = SG.unsup_segments(pp, [], rng)
            lab_k, _ = SG.unsup_segments(pp, [], rng, plane=KIN_PLANE, plane_thr=0.008)
            above = P[:, 2] >= 1.0 + 0.008
            for o in im["objects"]:
                oid = o["oid"]
                gt = np.flatnonzero(L == oid)
                if len(gt) < 5:
                    continue
                T = Truth((objs[oid - 1] if oid > 0 else None), np.random.default_rng(1), n_surf=2000, n_q=0, n_mc=20000) if (zono and oid > 0) else None
                base = dict(id=iid, scene=im["scene"], view=im["view"], cond=cond, oid=oid, phrase=o["phrase"], queryable=o["queryable"], n_gt=len(gt),
                            touch=im["scene"].startswith("touch") or im["scene"] == "tower3",
                            ref_inst=o["ref_instance"] or sum(x["phrase"] == o["phrase"] and x["queryable"] for x in im["objects"]) > 1)
                def seg_of_box(box):
                    m = in_box(P, c["origin"], c["R"], F_PX, box)
                    if oid == 0:
                        idx = np.flatnonzero(m & ~above)
                        sub = cleanup_tab(P[idx])
                    else:
                        idx = np.flatnonzero(m & above)
                        sub = cleanup_tab(P[idx])
                    return idx[sub]
                meths = {}
                if oid > 0:
                    for nm, lab in (("dbscan_ransac", lab_r), ("dbscan_kin", lab_k)):
                        iou, S = best_cluster_iou(lab, gt)
                        meths[nm] = (iou, S)
                else:
                    meths["dbscan_ransac"] = (0.0, None)      # tray removed as the plane by construction
                    meths["dbscan_kin"] = (0.0, None)
                if o["bbox"]:
                    S = seg_of_box(o["bbox"])
                    meths["gtbox"] = (iou_sets(S, gt), S)
                for mname, d in boxes.items():
                    b = d.get((iid, f"o{oid}"))
                    if b is not None and o["queryable"]:
                        S = seg_of_box(b)
                        meths[f"box:{mname}"] = (iou_sets(S, gt), S)
                    elif o["queryable"]:
                        meths[f"box:{mname}"] = (0.0, None)    # no parsable box -> not recovered
                for nm, (iou, S) in meths.items():
                    r = dict(base, method=nm, iou=iou)
                    if T is not None and S is not None and len(S) >= 8:
                        r["contain"] = zono_contain(P[S], T)
                    rows.append(r)
    return rows


def run_house(meta, boxes):
    rows = []
    for iid, im in meta.items():
        if im["domain"] != "house":
            continue
        z = np.load(im["depth"])
        dep = z["depth"].astype(np.float32) / 1000.0
        origin, R = z["origin"], z["R"]
        seg = np.load(im["seg"])["seg"]
        Hh, Ww = dep.shape
        vv, uu = np.mgrid[0:Hh:2, 0:Ww:2]
        d = dep[vv, uu]
        ok = (d > 0.3) & (d < 6.0)
        rng = np.random.default_rng(zlib_seed(iid))
        d = d + rng.normal(0, 3e-3, d.shape)
        ok &= rng.random(d.shape) > 0.02
        u, v, d = uu[ok], vv[ok], d[ok]
        lab_img = seg[vv, uu][ok].astype(np.int64) + 1             # 0 = unlabeled geom
        x = (u + 0.5 - Ww / 2) / F_PX * d
        y = -(v + 0.5 - Hh / 2) / F_PX * d
        pts = origin + (np.stack([x, y, -d], 1)) @ R.T
        uniq, inv = np.unique(lab_img, return_inverse=True)
        cl = SimpleNamespace(pts=pts, label=inv.astype(np.int32), cam=np.zeros(len(pts), np.int32))
        pp = SG.preprocess(cl, vox=0.01, out_r=0.03, out_k=4)
        P, Lc = pp["pts"], pp["label"]
        lab_r, info = SG.unsup_segments(pp, [], np.random.default_rng(0), eps=0.02, min_samples=6, min_cluster=15)
        # camera-frame depth of the preprocessed points
        zc = -((P - origin) @ R)[:, 2]
        uq = {int(o): i for i, o in enumerate(uniq)}
        for k, q in enumerate(im["objq"]):
            if not q["unique"]:
                continue
            oid = q["targets"][0]
            if oid + 1 not in uq:
                continue
            gt = np.flatnonzero(Lc == uq[oid + 1])
            if len(gt) < 5:
                continue
            gb = [x_ for x_ in im["objects"] if x_["oid"] == oid][0]["bbox"]
            base = dict(id=iid, scene=im["scene"], view=im["view"], cond="depth3mm", oid=oid, phrase=q["phrase"], queryable=True, n_gt=len(gt), touch=False,
                        ref_inst=False)
            def seg_of_box(box):
                m = np.flatnonzero(in_box(P, origin, R, F_PX, box))
                if len(m) == 0:
                    return m
                return m[cleanup_house(zc[m])]
            meths = {}
            iou, S = best_cluster_iou(lab_r, gt)
            meths["dbscan_ransac"] = (iou, S)
            S = seg_of_box(gb)
            meths["gtbox"] = (iou_sets(S, gt), S)
            for mname, dct in boxes.items():
                b = dct.get((iid, f"h{k}"))
                meths[f"box:{mname}"] = (iou_sets(seg_of_box(b), gt), None) if b is not None else (0.0, None)
            for nm, (iou, S) in meths.items():
                rows.append(dict(base, method=nm, iou=iou, n_gt_house=len(gt)))
    return rows


def zlib_seed(s):
    import zlib
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def table(rows, sel, title):
    from collections import defaultdict
    G = defaultdict(list)
    for r in rows:
        if sel(r):
            G[r["method"]].append(r)
    lines = [f"\n#### {title}", "| method | n objs | recovered (IoU>=.5) | mean IoU | zono contain (mean) |", "|---|---|---|---|---|"]
    for m in sorted(G, key=lambda k: (not k.startswith("dbscan"), k)):
        L = G[m]
        ct = [r["contain"] for r in L if r.get("contain") is not None]
        lines.append(f"| {m} | {len(L)} | {np.mean([r['iou'] >= 0.5 for r in L]):.2f} | {np.mean([r['iou'] for r in L]):.2f} | "
                     f"{('%.3f' % np.mean(ct)) if ct else '-'} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--no-zono", action="store_true")
    ap.add_argument("--skip-house", action="store_true")
    a = ap.parse_args()
    meta = load_meta()
    boxes = model_boxes(a.models.split(","))
    print("models with boxes:", {k: len(v) for k, v in boxes.items()}, flush=True)
    rows = run_tab(meta, boxes, zono=not a.no_zono)
    hrows = [] if a.skip_house else run_house(meta, boxes)
    C.write_jsonl(C.DATA / "results" / "downstream_rows.jsonl", rows + hrows)
    T = []
    qy = lambda r: r["queryable"] and r["oid"] > 0 and not r["ref_inst"]
    for cond in ("n3mm", "nominal"):
        T.append(table(rows, lambda r: r["cond"] == cond and qy(r), f"tabletop {cond}: all queryable objects (excl. tray)"))
        T.append(table(rows, lambda r: r["cond"] == cond and qy(r) and r["touch"], f"tabletop {cond}: touching-object scenes (tower3, touch_*)"))
        T.append(table(rows, lambda r: r["cond"] == cond and qy(r) and not r["touch"], f"tabletop {cond}: non-touching scenes"))
    for cond in ("n3mm", "nominal"):
        T.append(table(rows, lambda r: r["cond"] == cond and r["oid"] > 0 and r["method"] in ("dbscan_ransac", "dbscan_kin", "gtbox"),
                       f"tabletop {cond}: ALL GT objects incl. tiny beads/items and ref-scene instances (baseline-comparable set; no VLM)"))
    T.append(table(rows, lambda r: r["cond"] == "n3mm" and r["oid"] == 0 and r["queryable"], "tabletop n3mm: tray"))
    T.append(table(rows, lambda r: r["cond"] == "n3mm" and r["scene"] in ("cupbeads", "dishitems", "Lblock") and qy(r), "tabletop n3mm: container/composite scenes"))
    if hrows:
        T.append(table(hrows, lambda r: True, "house (unique-category queries, depth cloud 3 mm)"))
    (C.DATA / "results" / "downstream_table.md").write_text("\n".join(T))
    print("\n".join(T))


if __name__ == "__main__":
    main()
