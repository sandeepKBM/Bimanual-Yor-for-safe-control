"""world_model/v2/model/latency.py -- per-TICK inference latency of each option in
the streaming shield setting (one vision tick = 1/15 s = 66.7 ms budget).

  python -m world_model.v2.model.latency --prep $PREP/ds --gru G.pt --dino D.pt --rssm R.pt \
      --device cuda --plans 16 --K 15

Per tick measured (CUDA-synchronised wall clock, median / p95 over the episode):
  gru   : GRUPredictor.step (proprio + cup kin + action row)
  dino  : DINOv2-small on every camera frame (live, not cached) + observe_step
          + imagine(plans x K) through all ensemble members
  rssm  : observe_step (conv encoder + RSSM posterior) + imagine(plans x K)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.model import common as C  # noqa: E402
from world_model.v2.model import data as D  # noqa: E402


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def _stats(ts):
    ts = np.asarray(ts[3:]) * 1e3        # drop warm-up ticks
    return dict(med_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)), max_ms=float(ts.max()))


def bench_gru(ckpt, ep, dev):
    from world_model.v2.model.proprio_gru import GRUPredictor
    p = GRUPredictor(ckpt, dev); p.reset()
    X = C.step_features(ep, p.ck["use_action"], p.ck["use_cup_kin"]); ts = []
    for t in range(len(X)):
        _sync(dev); t0 = time.perf_counter(); p.step(X[t]); _sync(dev); ts.append(time.perf_counter() - t0)
    return dict(tick=_stats(ts), members=len(p.nets))


def bench_dino(ckpt, ep, dev, plans, K):
    from world_model.v2.model.wm_dino import DinoWMPredictor, DinoFeaturizer
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    fz = DinoFeaturizer(dev, ck["feats_meta"]["img_px"], ck["feats_meta"]["pool"])
    p = DinoWMPredictor(ckpt, dev, featurizer=fz); p.reset()
    low = C.step_features(ep, use_action=False); act = C.get_action(ep, ck.get("action_key", "action"))
    A = np.repeat(act[None, :K], plans, 0) + 0.01 * np.random.randn(plans, min(K, len(act)), act.shape[1])
    t_enc, t_obs, t_img = [], [], []
    for t in range(ep["T"]):
        _sync(dev); t0 = time.perf_counter()
        tok = fz(np.concatenate([ep["images"][c][t:t + 1] for c in p.cams], 0))   # all cams, ONE forward
        _sync(dev); t1 = time.perf_counter()
        p.observe_step(tok, low[t], act[t - 1] if t else None)
        _sync(dev); t2 = time.perf_counter()
        p.imagine(A)
        _sync(dev); t3 = time.perf_counter()
        t_enc.append(t1 - t0); t_obs.append(t2 - t1); t_img.append(t3 - t2)
    tot = np.array(t_enc) + np.array(t_obs) + np.array(t_img)
    return dict(dino_encode=_stats(t_enc), observe=_stats(t_obs), imagine=_stats(t_img), tick=_stats(tot),
                members=len(p.wms), cams=len(p.cams))


def bench_rssm(ckpt, ep, dev, plans, K):
    from world_model.v2.model.wm_rssm import RSSMPredictor
    p = RSSMPredictor(ckpt, dev); p.reset()
    low = C.step_features(ep, use_action=False); act = C.get_action(ep, p.ck.get("action_key", "action"))
    A = np.repeat(act[None, :K], plans, 0)
    t_obs, t_img = [], []
    for t in range(ep["T"]):
        _sync(dev); t0 = time.perf_counter()
        p.observe_step({c: ep["images"][c][t] for c in p.cams}, low[t], act[t - 1] if t else None)
        _sync(dev); t1 = time.perf_counter(); p.imagine(A); _sync(dev); t2 = time.perf_counter()
        t_obs.append(t1 - t0); t_img.append(t2 - t1)
    return dict(observe=_stats(t_obs), imagine=_stats(t_img), tick=_stats(np.array(t_obs) + np.array(t_img)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--gru"); ap.add_argument("--dino"); ap.add_argument("--rssm")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--plans", type=int, default=16)
    ap.add_argument("--K", type=int, default=15)
    a = ap.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    ep = max(D.load_prepared(a.prep, a.split, limit=4), key=lambda e: e["T"])
    budget = 1e3 / ep["hz"]
    res = {}
    if a.gru:
        res["gru(cpu)"] = bench_gru(a.gru, ep, "cpu")
        if a.device != "cpu":
            res[f"gru({a.device})"] = bench_gru(a.gru, ep, a.device)
    with torch.no_grad():
        if a.dino:
            res["dino"] = bench_dino(a.dino, ep, a.device, a.plans, a.K)
        if a.rssm:
            res["rssm"] = bench_rssm(a.rssm, ep, a.device, a.plans, a.K)
    print(f"[latency] episode T={ep['T']} hz={ep['hz']:.2f} (tick budget {budget:.1f} ms); plans={a.plans} K={a.K}")
    for k, v in res.items():
        parts = " ".join(f"{n}: med {s['med_ms']:.2f} / p95 {s['p95_ms']:.2f} ms" for n, s in v.items() if isinstance(s, dict))
        print(f"  {k:10s} {parts}  {'| members=' + str(v['members']) if 'members' in v else ''}")


if __name__ == "__main__":
    main()
