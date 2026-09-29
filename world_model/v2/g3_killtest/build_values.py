"""build_values.py -- exact grid HJ V(s; theta) for every particle theta of a class.

    python -m world_model.v2.g3_killtest.build_values --cls T [--n 201] [--subset 0]
Uses NUMBA threads (set NUMBA_NUM_THREADS). SIMULATION ONLY.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.g3_killtest import core as C  # noqa: E402
from world_model.v2.g3_killtest import config as K  # noqa: E402


def build(cls, n, subset=0, iters=K.HJ_ITERS):
    TH = K.particle_grid(cls)
    idx = np.arange(len(TH))
    if subset:
        idx = np.unique(np.linspace(0, len(TH) - 1, subset).round().astype(int))
    TH = TH[idx]
    cid = K.CLASSES[cls]["cid"]
    AX = C.make_axes(cid, TH, n, n)
    t0 = time.time()
    V, its, res = C.solve_all(cid, TH, AX, n, n, C.U_FB, iters, K.HJ_TOL)
    return dict(V=V, TH=TH, AX=AX, idx=idx, its=its, res=res, n=n, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True)
    ap.add_argument("--n", type=int, default=K.N_GRID)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--iters", type=int, default=K.HJ_ITERS)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = build(a.cls, a.n, a.subset, a.iters)
    out = Path(a.out) if a.out else K.value_path(a.cls, a.n)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **r)
    print(f"[hj] {a.cls} n={a.n} N_theta={len(r['TH'])} {r['secs']:.1f}s iters max {r['its'].max()} "
          f"resid max {r['res'].max():.2e} median {np.median(r['res']):.2e}  frac V>0 {np.mean(r['V'] > 0):.3f} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
