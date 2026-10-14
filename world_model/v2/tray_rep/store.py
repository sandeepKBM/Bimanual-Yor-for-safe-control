"""store.py -- read access to a generated tray_rep dataset: logged arrays, branch-point snapshots, `load_split`.
SIMULATION / SAVED DATA ONLY.

Layout of a dataset directory `<root>/<name>/` (default name `data_full` or `data_tray_only`):
    meta.json                       settings of the generator
    <split>/shard_<g>.npz           per-shard arrays (see gen_nominal)
    <split>/snap_<g>.npz            snapshots at grid ticks (every `snap_stride` ticks from `snap_min_tick`)
    <split>/shard_<g>._DONE         sentinel
    <split>.npz                     merged split (after `gen_nominal --merge`); read by load_split
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.env import RepSnap

EP_KEYS = ("lens", "onset", "seed", "hid", "tunnel")
TICK_KEYS = ("obs", "act", "l", "n", "nd", "priv")


def load_split(name, data):
    """Superset of wm_audit.common.load_split: also returns start, hid, priv, tunnel. Reads the merged file."""
    f = np.load(Path(data) / f"{name}.npz")
    lens = f["lens"]
    st = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return dict(obs=f["obs"], act=f["act"], l=f["l"], n=f["n"], nd=f["nd"], lens=lens, start=st, onset=f["onset"],
                seed=f["seed"], hid=f["hid"], priv=f["priv"], tunnel=f["tunnel"])


def read_meta(root):
    return json.loads((Path(root) / "meta.json").read_text())


class Store:
    """Episode-level access to one split. Episode numbers are split-level (the same numbering as load_split)."""

    def __init__(self, root, split):
        self.root, self.split = Path(root), split
        self.meta = read_meta(self.root)
        files = sorted(f for f in (self.root / split).glob("shard_*.npz") if ".tmp" not in f.name
                       and f.with_suffix("._DONE").exists())
        self.files = files
        self.gidx, self.n_eps_shard, self.lens_shard = [], [], []
        for f in files:
            with np.load(f) as z:
                self.gidx.append(int(f.stem.split("_")[1]))
                self.lens_shard.append(z["lens"].copy())
                self.n_eps_shard.append(len(z["lens"]))
        self.offsets = np.concatenate([[0], np.cumsum(self.n_eps_shard)]).astype(np.int64)
        self.n_eps = int(self.offsets[-1])
        self.lens = np.concatenate(self.lens_shard).astype(np.int64)      # (n_eps,) ticks per episode
        self._sh, self._sh_i = None, -1
        self._sn, self._sn_i = None, -1

    # ------------------------------------------------------------------ indexing
    def locate(self, ep):
        i = int(np.searchsorted(self.offsets, ep, side="right") - 1)
        return i, int(ep - self.offsets[i])

    def shard(self, i):
        if i != self._sh_i:
            with np.load(self.files[i]) as z:
                d = {k: z[k] for k in z.files}
            d["start"] = np.concatenate([[0], np.cumsum(d["lens"])[:-1]])
            self._sh, self._sh_i = d, i
        return self._sh

    def snaps(self, i):
        if i != self._sn_i:
            f = self.files[i].with_name(f"snap_{self.gidx[i]:04d}.npz")
            with np.load(f) as z:
                d = {k: z[k] for k in z.files}
            d["_idx"] = {(int(e), int(t)): j for j, (e, t) in enumerate(zip(d["snap_ep"], d["snap_tick"]))}
            self._sn, self._sn_i = d, i
        return self._sn

    # ------------------------------------------------------------------ accessors
    def rows(self, key, ep, t0, t1):
        """logged rows [t0, t1) of episode `ep` for a per-tick key (obs, act, l, n, nd, priv)."""
        i, e = self.locate(ep)
        S = self.shard(i)
        s = S["start"][e]
        return S[key][s + t0:s + t1]

    def ep_info(self, ep):
        i, e = self.locate(ep)
        S = self.shard(i)
        return dict(lens=int(S["lens"][e]), onset=int(S["onset"][e]), seed=int(S["seed"][e]), hid=float(S["hid"][e]),
                    tunnel=int(S["tunnel"][e]))

    def snapshot_at(self, ep, t):
        """(RepSnap, hid) at the grid tick t of episode ep. Raises KeyError if t is not on the snapshot grid."""
        i, e = self.locate(ep)
        S = self.snaps(i)
        j = S["_idx"][(e, int(t))]
        snap = RepSnap.from_arrays(S["snap_state"][j], S["snap_prev"][j], S["snap_has_prev"][j], S["snap_kin"][j],
                                   S["snap_status"][j], S["snap_k"][j],
                                   S["snap_wslew"][j] if "snap_wslew" in S else None,   # absent in phase-1a data -> 0
                                   S["snap_trk"][j] if "snap_trk" in S and S["snap_trk"].ndim == 2 and S["snap_trk"].shape[1] else None)
        return snap, float(self.shard(i)["hid"][e])

    def grid_ticks(self, ep, lo=0, hi=None):
        """grid ticks of episode ep inside [lo, hi]."""
        L = int(self.lens[ep])
        stride, tmin = self.meta["snap_stride"], self.meta["snap_min_tick"]
        hi = L - 1 if hi is None else min(hi, L - 1)
        first = max(lo, tmin)
        first += (-(first - tmin)) % stride
        return np.arange(first, hi + 1, stride)
