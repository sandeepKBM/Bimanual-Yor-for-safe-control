"""model_uq/run_f5uq.py -- bridge.run with the extra filter spec F5UQ (state-dependent model-error bound). SIMULATION ONLY. CPU.
Does what `python -m world_model.v2.tray_rep.bridge.run` does, but registers F5UQ first (no tray_rep file is edited).

  python -m world_model.v2.model_uq.run_f5uq --filter "F5UQ:eps=pa_k15_n1000" --setup piper_trk --split val --n-eps 40 --out <dir> [--data <dir>]
  --filter F5 (or any other bridge spec) also works, so the baseline runs through the same code path.
"""
from __future__ import annotations

import argparse

from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.bridge import run as BR

_orig = F.make_filter


def make_filter(spec, renv, **kw):
    name, p = F.parse_spec(spec)
    if name == "F5UQ":
        from world_model.v2.model_uq.f5_uq import F5UQFilter
        return F5UQFilter(renv, **p)
    return _orig(spec, renv, **kw)


F.make_filter = make_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--setup", default="piper_trk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-eps", type=int, default=40)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="")
    a = ap.parse_args()
    BR.run(a.filter, a.setup, a.seed, a.split, n_eps=a.n_eps, workers=a.workers, tag=a.tag, out_root=a.out,
           data=a.data or None, ep_start=a.ep_start)


if __name__ == "__main__":
    main()
