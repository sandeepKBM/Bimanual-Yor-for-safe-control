"""world_model/v2/dinowm/smoke_api.py -- API + harness smoke for a dinowm checkpoint.

SIMULATION DATA ONLY. Checks: encode from raw images (live DINOv2) == encode from the
token cache; imagine() under planned actions runs, batch-broadcasts, and is
differentiable w.r.t. actions (finite-difference check); spill_margin/hazard shapes;
harness evaluation on the test split (numbers meaningful only for trained ckpts).

  CUDA_VISIBLE_DEVICES=7 python -m world_model.v2.dinowm.smoke_api --ckpt $RUN/model.pt --cache $CACHE [--n-eval 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from world_model.v2.dinowm import shards as S  # noqa: E402
from world_model.v2.dinowm.api import DinoWMOfficialPredictor, SpillWM  # noqa: E402
from world_model.v2.dinowm.train import attach_tokens, load_eps  # noqa: E402
from world_model.v2.eval import harness as Hn  # noqa: E402
from world_model.v2.model import common as C  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--K", type=int, default=15); ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--limit-shards", type=int, default=None)
    a = ap.parse_args(argv)
    wm = SpillWM(a.ckpt, "cuda"); ck = wm.ck
    eps = load_eps(ck["data"], ck["label"], tuple(ck["kinds"].split(",")), a.limit_shards, a.cache, ck["cache_tag"])
    mm = attach_tokens(eps, a.cache, ck["cache_tag"])
    test = S.by_split(eps)["test"]
    # ---- one episode with images: live encoder vs cache
    e = next(x for x in test if x["onset"] > 20)
    img_ep = next(x for x in S.load_shard(Path(ck["data"]) / (e["id"].split("#")[0] + ".npz"), True, ("zed",), ck["label"])
                  if x["id"] == e["id"])
    H = wm.H; t = e["onset"] - 10
    low = C.step_features(e, use_action=False)[t - H + 1:t + 1][None]
    act = C.get_action(e, wm.action_key)[t - H + 1:t + 1][None]
    z_img = wm.encode(dict(images=img_ep["images"]["zed"][t - H + 1:t + 1][None], low=low, actions=act))
    sh, st = e["_tok"]
    z_tok = wm.encode(dict(tokens=np.asarray(mm[sh][st + t - H + 1:st + t + 1], np.float32)[None], low=low, actions=act))
    print(f"encode: z {tuple(z_img.shape)}  |z_img - z_cache|max = {float((z_img - z_tok).detach().abs().max()):.4f} "
          f"(cache is fp16)")
    # ---- imagine under the logged future actions (== the plan) and N perturbed plans
    plan = torch.as_tensor(C.get_action(e, wm.action_key)[t:t + a.K], device=wm.device)[None]
    N = 4
    plans = plan.repeat(N, 1, 1) + 0.3 * plan.std() * torch.randn(N, *plan.shape[1:], device=wm.device)
    plans.requires_grad_(True)
    zs = wm.imagine(z_tok, plans)
    l = wm.spill_margin(zs); p = wm.hazard(zs)
    print(f"imagine: z_seq {tuple(zs.shape)}  margin {tuple(l.shape)}  hazard {tuple(p.shape)}")
    obj = l.min(-1).values.sum(); obj.backward()
    g = plans.grad
    print(f"d(min margin)/d(actions): shape {tuple(g.shape)} |g|mean={float(g.abs().mean()):.3e} finite={bool(torch.isfinite(g).all())}")
    # finite-difference check on one coordinate
    with torch.no_grad():
        eps_fd = 1e-2; d = torch.zeros_like(plans); d[0, 2, 0] = eps_fd
        f = lambda P: float(wm.spill_margin(wm.imagine(z_tok, P)).min(-1).values[0])
        fd = (f(plans + d) - f(plans - d)) / (2 * eps_fd)
    print(f"finite-diff check a[0,2,0]: autograd {float(g[0, 2, 0]):.4e} vs FD {fd:.4e}")
    # ---- open-loop latent error vs horizon under the true actions (1 plan)
    with torch.no_grad():
        zt = wm.imagine(z_tok, plan)
        T0 = st + t + 1
        tgt = torch.as_tensor(np.asarray(mm[sh][T0:T0 + a.K], np.float32), device=wm.device)
        err = ((zt[0, H:H + len(tgt), :, :384] - tgt) ** 2).mean((-1, -2)).cpu().numpy()
        copy = ((tgt - torch.as_tensor(np.asarray(mm[sh][T0 - 1], np.float32), device=wm.device)) ** 2).mean((-1, -2)).cpu().numpy()
    print("open-loop visual MSE k=1..K:  model " + " ".join(f"{x:.2f}" for x in err))
    print("                     copy-last-frame " + " ".join(f"{x:.2f}" for x in copy))
    # ---- harness
    pr = DinoWMOfficialPredictor(a.ckpt, "cuda", mm=mm)
    va = S.by_split(eps)["val"][:a.n_eval]; te = test[:a.n_eval]
    rep = Hn.evaluate(pr, va, {"test": te})
    print(Hn.format_report(rep))


if __name__ == "__main__":
    main()
