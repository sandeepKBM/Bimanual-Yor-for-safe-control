"""grounding/score.py -- 2D grounding metrics from raw model outputs (offline, CPU).
  python -m world_model.v2.vlm.grounding.score [--models ...]  -> $DATA/results/records_<model>.jsonl and results/grounding_table.md
Per request: parse first point/box (as the model wrote it), convert with the model's native convention (relative 0-1000) and, as a sanity flag,
with absolute pixels. point_in_mask: point falls in the visible GT mask of the target (house multi-instance category: in ANY instance).
box IoU: vs the GT visible-mask bbox (house multi: best instance). Referring: point in target mask (and not distractor); box IoU(target) >= 0.5 and
> IoU(distractor). Only the FIRST answer is scored (no oracle choice among several)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from world_model.v2.vlm.grounding import common as C
from world_model.v2.vlm.grounding.make_requests import MODELS

TOUCH = ("tower3", "touch_boxblock", "touch_bottlemug", "touch_ballbox")


def load_meta():
    meta = {}
    for d in ("tab", "house"):
        p = C.DATA / d / "meta.json"
        if p.exists():
            for im in json.load(open(p)):
                meta[im["id"]] = im
    return meta


_seg_cache = {}


def seg_of(im):
    if im["id"] not in _seg_cache:
        _seg_cache[im["id"]] = np.load(im["seg"])["seg"]
    return _seg_cache[im["id"]]


def target_info(im, qk):
    """-> (kind, target oids, wrong oids, group) for a query key."""
    if qk[0] == "o":
        oid = int(qk[1:])
        o = [x for x in im["objects"] if x["oid"] == oid][0]
        same = [x["oid"] for x in im["objects"] if x["phrase"] == o["phrase"] and x["queryable"]]
        if len(same) > 1:                      # identical phrase (beads, dish items): ambiguous -> any-instance match
            return "obj", same, [], "tab_ambig"
        grp = "tab_tray" if oid == 0 else ("tab_touch" if im["scene"] in TOUCH or im["scene"].startswith("touch") else "tab_obj")
        return "obj", [oid], [], grp
    if qk[0] == "r":
        t, side = qk[1:].split("_")
        r = [x for x in im["refs"] if x["target"] == int(t) and x["side"] == side][0]
        return "ref", [r["target"]], [r["wrong"]], "tab_ref"
    k = int(qk[1:])
    q = im["objq"][k]
    return "obj", q["targets"], [], "house_uniq" if q["unique"] else "house_multi"


def score_model(model, meta):
    p = C.DATA / "out" / f"gr_{model}.jsonl"
    if not p.exists():
        return []
    recs = []
    for r in C.read_jsonl(p):
        iid, task, qk = r["id"].split("|")
        im = meta[iid]
        seg = seg_of(im)
        kind, tg, wr, grp = target_info(im, qk)
        text = r["texts"][0]
        W, H = im["W"], im["H"]
        tm = np.isin(seg, tg)
        wm = np.isin(seg, wr) if wr else None
        rec = dict(id=r["id"], model=model, domain=im["domain"], scene=im["scene"], view=im["view"], group=grp, kind=kind, task=task,
                   seconds=r["seconds"], vram=r["peak_vram_gb"], ntok=r["n_new_tokens"][0], text=text[:300])
        if task == "point":
            pts = C.parse_points(model, text)
            rec["parsed"] = bool(pts)
            rec["n_ans"] = len(pts)
            for conv in ("native", "abs"):
                ok = None
                if pts:
                    pp = C.to_px(model, pts[0], W, H, conv)
                    ok = C.point_in_mask(tm, pp)
                    if kind == "ref":
                        rec[f"wrong_{conv}"] = C.point_in_mask(wm, pp)
                rec[f"hit_{conv}"] = ok
        else:
            bxs = C.parse_boxes(model, text)
            rec["parsed"] = bool(bxs)
            rec["n_ans"] = len(bxs)
            for conv in ("native", "abs"):
                iou = iouw = None
                if bxs:
                    b = C.to_px(model, bxs[0], W, H, conv)
                    gts = [C.mask_bbox(seg == o) for o in tg]
                    gts = [g for g in gts if g]
                    iou = max([C.box_iou(b, g) for g in gts], default=0.0)
                    if kind == "ref":
                        gw = C.mask_bbox(seg == wr[0])
                        iouw = C.box_iou(b, gw) if gw else 0.0
                rec[f"iou_{conv}"] = iou
                rec[f"iouw_{conv}"] = iouw
            if bxs:
                rec["box_px"] = [float(x) for x in C.to_px(model, bxs[0], W, H, "native")]
        recs.append(rec)
    return recs


def _mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else float("nan")


def summarize(recs, conv="native"):
    """-> dict group -> metrics."""
    out = {}
    G = defaultdict(list)
    for r in recs:
        G[r["group"]].append(r)
        if r["group"] in ("tab_obj", "tab_touch"):
            G["tab_all_obj"].append(r)
        if r["group"] in ("house_uniq", "house_multi"):
            G["house_all"].append(r)
    for g, L in G.items():
        P = [r for r in L if r["task"] == "point"]
        B = [r for r in L if r["task"] == "box"]
        m = dict(n_point=len(P), n_box=len(B))
        if g == "tab_ref":
            m["ref_acc_point"] = _mean([bool(r[f"hit_{conv}"]) and not r.get(f"wrong_{conv}") if r["parsed"] else False for r in P])
            m["ref_wrong_point"] = _mean([bool(r.get(f"wrong_{conv}")) if r["parsed"] else False for r in P])
            m["ref_pick_box"] = _mean([(r[f"iou_{conv}"] or 0) > max(0.1, r[f"iouw_{conv}"] or 0) for r in B])
            m["ref_acc_box"] = _mean([(r[f"iou_{conv}"] or 0) >= 0.5 and (r[f"iou_{conv}"] or 0) > (r[f"iouw_{conv}"] or 0) for r in B])
        else:
            m["point_in_mask"] = _mean([bool(r[f"hit_{conv}"]) if r["parsed"] else False for r in P])
            ious = [(r[f"iou_{conv}"] or 0.0) if r["parsed"] else 0.0 for r in B]
            m["box_iou"] = _mean(ious)
            m["iou50"] = _mean([v >= 0.5 for v in ious])
        m["parse_pt"] = _mean([r["parsed"] for r in P])
        m["parse_box"] = _mean([r["parsed"] for r in B])
        m["lat_s"] = _mean([r["seconds"] for r in L])
        m["vram_gb"] = max(r["vram"] for r in L)
        out[g] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    meta = load_meta()
    res = {}
    for model in a.models.split(","):
        recs = score_model(model, meta)
        if not recs:
            continue
        C.write_jsonl(C.DATA / "results" / f"records_{model}.jsonl", recs)
        res[model] = dict(native=summarize(recs, "native"), abs=summarize(recs, "abs"))
    (C.DATA / "results" / "grounding_summary.json").write_text(json.dumps(res, indent=1))
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"
    lines = []
    for g in ("tab_obj", "tab_touch", "tab_tray", "tab_all_obj", "tab_ambig", "tab_ref", "house_uniq", "house_multi", "house_all"):
        lines.append(f"\n### {g}\n| model | n(pt/box) | point-in-mask | box IoU | IoU>=.5 | ref acc pt / box>=.5 / box-pick | parse pt/box | lat s | VRAM GB | [abs-px conv: pt / IoU>=.5] |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for model, s in res.items():
            if g not in s["native"]:
                continue
            m, ma = s["native"][g], s["abs"][g]
            ref = f"{f(m.get('ref_acc_point'))} / {f(m.get('ref_acc_box'))} / pick {f(m.get('ref_pick_box'))}" if g == "tab_ref" else "-"
            lines.append(f"| {model} | {m['n_point']}/{m['n_box']} | {f(m.get('point_in_mask'))} | {f(m.get('box_iou'))} | {f(m.get('iou50'))} | {ref} | "
                         f"{f(m['parse_pt'])}/{f(m['parse_box'])} | {m['lat_s']:.2f} | {m['vram_gb']:.1f} | {f(ma.get('point_in_mask'))} / {f(ma.get('iou50'))} |")
    (C.DATA / "results" / "grounding_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
