"""split.py -- train / val / test_cat / test_house / test_asset JSONL + manifest.json from the generator's shards.  Pure python.

  python -m world_model.v2.vlm.ftdata.split --data DIR [--n-heldout-cats 9] [--n-heldout-houses 4] [--asset-frac 0.10] [--val-frac 0.05] [--seed 0]

Held-out categories: terciles of the median max object dimension (small / medium / large), an equal number drawn from each (eligible: >= --min-cat-instances instances).
Held-out houses: n random houses of those present.  Held-out assets: ~asset-frac of all assets, at most 1 per category, only categories with >= 2 assets (the category keeps >= 1 asset),
never inside a held-out category.  Priority for a sample: category > house > asset (each sample lands in exactly ONE file; train/val contain NO held-out category, house or asset).
val = val-frac of the remaining (house, instance) groups (all views of an instance stay together); val shares categories / houses / assets with train (in-distribution).
Both tasks ('arms' and 'name') of an instance follow the same file.  Every instance (all its views) stays in one file."""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

import numpy as np

SPLITS = ("train", "val", "test_cat", "test_house", "test_asset")


def read_records(data):
    recs = []
    for f in sorted(glob.glob(os.path.join(data, "shards", "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except Exception:      # noqa: BLE001
                    pass
    return recs


def make_spec(metas, n_cats=9, n_houses=4, asset_frac=0.10, val_frac=0.05, seed=0, min_cat_instances=5):
    """metas: list of meta dicts (one per record).  -> spec dict (JSON-serialisable)."""
    rng = np.random.default_rng(seed)
    inst = {}
    for m in metas:
        inst[(m["house"], m["instance"])] = m
    by_cat = defaultdict(list)
    for m in inst.values():
        by_cat[m["category"]].append(m)
    size = {c: statistics.median(max(m["true_extent_obb"]) for m in v) for c, v in by_cat.items()}
    elig = sorted((c for c, v in by_cat.items() if len(v) >= min_cat_instances), key=lambda c: size[c])
    held_cats = []
    if elig and n_cats > 0:
        k = min(n_cats, len(elig))
        thirds = [elig[i * len(elig) // 3:(i + 1) * len(elig) // 3] for i in range(3)]
        per = [k // 3 + (1 if i < k % 3 else 0) for i in range(3)]
        for t, n in zip(thirds, per):
            held_cats += [str(c) for c in rng.choice(t, size=min(n, len(t)), replace=False)] if t else []
    houses = sorted({m["house"] for m in inst.values()})
    held_houses = sorted(int(h) for h in rng.choice(houses, size=min(n_houses, max(len(houses) - 1, 0)), replace=False)) if n_houses > 0 and len(houses) > 1 else []
    assets_by_cat = defaultdict(set)
    for m in inst.values():
        if m["category"] not in held_cats:
            assets_by_cat[m["category"]].add(m["asset_id"])
    n_assets_all = len({m["asset_id"] for m in inst.values()})
    want = int(round(asset_frac * n_assets_all))
    cands = sorted(c for c, a in assets_by_cat.items() if len(a) >= 2)
    rng.shuffle(cands)
    held_assets = []
    for c in cands[:want]:
        held_assets.append(str(rng.choice(sorted(assets_by_cat[c]))))
    return dict(seed=seed, held_cats=sorted(held_cats), held_houses=held_houses, held_assets=sorted(held_assets), val_frac=val_frac, cat_size_m={c: round(size[c], 3) for c in held_cats},
                n_instances=len(inst), houses=houses)


def assign_split(meta, spec):
    if meta["category"] in spec["held_cats"]:
        return "test_cat"
    if meta["house"] in spec["held_houses"]:
        return "test_house"
    if meta["asset_id"] in spec["held_assets"]:
        return "test_asset"
    # val by instance group (deterministic hash in [0,1))
    import zlib
    u = (zlib.crc32(f"{spec['seed']}|val|{meta['house']}|{meta['instance']}".encode()) & 0xFFFFFF) / float(0x1000000)
    return "val" if u < spec["val_frac"] else "train"


def split_records(recs, spec):
    out = {s: [] for s in SPLITS}
    for r in recs:
        out[assign_split(r["meta"], spec)].append(r)
    return out


def manifest(out, spec):
    man = dict(spec=spec, splits={})
    for s, rs in out.items():
        c = Counter(r["meta"]["task"] for r in rs)
        man["splits"][s] = dict(n_records=len(rs), n_arms=c.get("arms", 0), n_name=c.get("name", 0), n_instances=len({(r["meta"]["house"], r["meta"]["instance"]) for r in rs}),
                                categories=sorted({r["meta"]["category"] for r in rs}), houses=sorted({r["meta"]["house"] for r in rs}),
                                n_assets=len({r["meta"]["asset_id"] for r in rs}),
                                rule_mode=dict(Counter(r["meta"]["rule_mode"] for r in rs if r["meta"]["task"] == "arms")),
                                arms=dict(Counter(json.loads(r["messages"][2]["content"])["arms"] for r in rs if r["meta"]["task"] == "arms")))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="default: --data")
    ap.add_argument("--n-heldout-cats", type=int, default=9)
    ap.add_argument("--n-heldout-houses", type=int, default=4)
    ap.add_argument("--asset-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--min-cat-instances", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out_dir = a.out or a.data
    recs = read_records(a.data)
    assert recs, f"no records under {a.data}/shards"
    spec = make_spec([r["meta"] for r in recs], a.n_heldout_cats, a.n_heldout_houses, a.asset_frac, a.val_frac, a.seed, a.min_cat_instances)
    out = split_records(recs, spec)
    for s, rs in out.items():
        with open(os.path.join(out_dir, f"{s}.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    man = manifest(out, spec)
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps({s: v["n_records"] for s, v in man["splits"].items()}), "held cats", spec["held_cats"], "held houses", spec["held_houses"], "held assets", len(spec["held_assets"]))


if __name__ == "__main__":
    main()
