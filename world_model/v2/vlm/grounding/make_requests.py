"""grounding/make_requests.py -- build per-model request JSONL (native prompts) from tab/house meta.json.
  python -m world_model.v2.vlm.grounding.make_requests [--models a,b,..]  -> $DATA/req/gr_<model>.jsonl and gr_smoke_<model>.jsonl
Request id = <image id>|<task>|<qkey>;  qkey o<oid> (single object) | r<target>_<side> (referring) | h<k> (house category query)."""
from __future__ import annotations

import argparse
import json

from world_model.v2.vlm.grounding import common as C

MODELS = ["qwen3-vl-8b", "robobrain2.5-4b", "qwen3.5-9b", "molmo2-er", "robobrain2.0-7b"]


def queries(meta):
    """-> list of (image_meta, qkey, phrase)."""
    out = []
    for im in meta:
        if im["domain"] == "tab":
            for o in im["objects"]:
                if o["queryable"]:
                    out.append((im, f"o{o['oid']}", o["phrase"]))
            for r in im["refs"]:
                if r["valid"]:
                    out.append((im, f"r{r['target']}_{r['side']}", r["phrase"]))
        else:
            for k, q in enumerate(im["objq"]):
                out.append((im, f"h{k}", q["phrase"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    meta = []
    for d in ("tab", "house"):
        p = C.DATA / d / "meta.json"
        if p.exists():
            meta += json.load(open(p))
    qs = queries(meta)
    print(len(meta), "images", len(qs), "queries")
    for model in a.models.split(","):
        rows = []
        for im, qk, phrase in qs:
            for task in ("point", "box"):
                rows.append(dict(id=f"{im['id']}|{task}|{qk}", images=[im["image"]], system=None, text=C.prompt(model, task, phrase),
                                 max_new_tokens=200, temperature=0.0))
        C.write_jsonl(C.DATA / "req" / f"gr_{model}.jsonl", rows)
        sm = rows[:: max(1, len(rows) // 8)][:8]
        C.write_jsonl(C.DATA / "req" / f"gr_smoke_{model}.jsonl", sm)
        print(model, len(rows), "requests", len(sm), "smoke")


if __name__ == "__main__":
    main()
