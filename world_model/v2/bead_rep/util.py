"""bead_rep/util.py -- small I/O helpers (NFS-race-tolerant mkdir, atomic npz writes, shard bookkeeping).
SIMULATION / SAVED DATA ONLY."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/common/users/ss5772/bimanual_data/bead_replication")
DATA = ROOT / "data"
SPLITS = ("train", "val", "test")


def safe_mkdir(p, tries: int = 8):
    """mkdir -p that survives NFS races: a previous array hit FileExistsError from
    pathlib.mkdir(parents=True, exist_ok=True) when many tasks created the same dir at once.
    Catch OSError/FileExistsError, sleep briefly, and accept the outcome iff is_dir()."""
    p = Path(p)
    for i in range(tries):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError):
            pass
        if p.is_dir():
            return p
        time.sleep(0.2 * (i + 1))
    if not p.is_dir():
        raise RuntimeError(f"could not create directory {p}")
    return p


def atomic_savez(path, **arrays):
    """np.savez to `<path>.tmp.<pid>.npz` then os.replace -> never a half-written shard."""
    path = Path(path)
    safe_mkdir(path.parent)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_sentinel(path, text=None):
    path = Path(path)
    safe_mkdir(path.parent)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    os.replace(tmp, path)


def parse_shard(s: str):
    """'i/n' -> (i, n)."""
    i, n = s.split("/")
    i, n = int(i), int(n)
    assert 0 <= i < n
    return i, n


def source_shards():
    """[(split, global_idx, path)] over the SAME shard split rule as the pendulum wmdata
    (spill_data.shard_split: sorted shard names, idx%5==4 -> test, ==3 -> val, else train).
    Order within a split = sorted-name order = wmdata episode order."""
    from world_model.v2.latent_hj import spill_data as sd
    ps = sorted(sd.list_shards())
    out = []
    for i, p in enumerate(ps):
        out.append(("test" if i % 5 == 4 else ("val" if i % 5 == 3 else "train"), i, p))
    return out


def split_index(root, split):
    """Episode index of a generated split: list of dict(gidx, n_eps, lens(array), seeds(array), file).
    Reads only the small arrays of each shard file (np.load is lazy)."""
    out = []
    d = Path(root) / split
    for f in sorted(d.glob("shard_*.npz")):
        if ".tmp" in f.name:
            continue
        with np.load(f, allow_pickle=False) as z:
            out.append(dict(gidx=int(f.stem.split("_")[1]), file=str(f), lens=z["lens"].copy(), seeds=z["seed"].copy(),
                            n_eps=int(len(z["lens"]))))
    return out


def locate(index, ep):
    """split-level episode number -> (shard record, local episode)."""
    for rec in index:
        if ep < rec["n_eps"]:
            return rec, ep
        ep -= rec["n_eps"]
    raise IndexError("episode out of range")


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
