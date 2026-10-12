"""robust/oracle_val.py -- certified-savable labels on the VAL split, written OUTSIDE the dataset directory.
SIMULATION ONLY (MuJoCo CPU; no hardware, no network). Calls tray_rep.oracle.run unchanged (same candidate law as test_v0:
75% near an onset, 25% random; a_scale from the dataset meta). The labels do not depend on the observation setup (the TEST oracle is
identical in data_piper_full and data_piper_trk_full), so one file serves both setups.

  python -m world_model.v2.tray_rep.robust.oracle_val --n 1500 --workers 4            # about 14 CPU-min at 0.55 s per state
  python -m world_model.v2.tray_rep.robust.oracle_val --n 1500 --shard 3/20           # array style, then --merge --n-parts 20
Output: <OUT>/oracle/val_v0.npz  (+ .json). With --shard i/n each task writes val_v0_part<i>of<n>.npz; then --merge --n-parts n.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep import oracle as OR
from world_model.v2.bead_rep import util as U
from world_model.v2.tray_rep.robust import util as RU


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--tag", default="val_v0")
    ap.add_argument("--out", default=str(RU.OUT))
    a = ap.parse_args()
    odir = Path(a.out) / "oracle"
    odir.mkdir(parents=True, exist_ok=True)
    data = RU.data_dir("piper")
    if a.merge:
        parts = [OR.load_oracle(odir / f"{a.tag}_part{i}of{a.n_parts}.npz") for i in range(a.n_parts)]
        out = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
        o = np.argsort(out["idx"]); out = {k: v[o] for k, v in out.items()}
        U.atomic_savez(odir / f"{a.tag}.npz", **out)
        print("[merge]", OR.summarize(out), flush=True)
        return
    out, meta = OR.run(data, "val", a.n, 0, a.workers, a.shard)
    si, sn = U.parse_shard(a.shard)
    name = f"{a.tag}.npz" if sn == 1 else f"{a.tag}_part{si}of{sn}.npz"
    U.atomic_savez(odir / name, **out)
    U.jdump(meta, odir / name.replace(".npz", ".json"))
    print("[oracle_val]", meta, flush=True)


if __name__ == "__main__":
    main()
