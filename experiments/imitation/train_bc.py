#!/usr/bin/env python
"""experiments/imitation/train_bc.py -- brief state-based BC (MLP + action
chunking) on gen_demos.py shards. GPU if available (CUDA_VISIBLE_DEVICES=<n>).

  CUDA_VISIBLE_DEVICES=2 python experiments/imitation/train_bc.py \\
      --shards <demos>/shards --out <run_dir> --steps 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bc_model import ChunkMLP, build_targets  # noqa: E402

# spec constants duplicated from il_common (importing il_common pulls MuJoCo/the demo;
# kept in sync by the assert in load()).
OBS_DIM, K, STRIDE = 77, 10, 10
RQ = slice(0, 6)


def load(shard_dir, only_success=True):
    obs, qref, grip, starts, lens, seeds = [], [], [], [], [], []
    off = 0
    for f in sorted(Path(shard_dir).glob("shard_*.npz")):
        z = np.load(f)
        for s, L, sd, ok in zip(z["ep_start"], z["ep_len"], z["ep_seed"], z["ep_success"]):
            if only_success and not ok:
                continue
            obs.append(z["obs"][s:s + L]); qref.append(z["q_ref"][s:s + L]); grip.append(z["grip_cmd"][s:s + L])
            starts.append(off); lens.append(L); seeds.append(int(sd)); off += L
    obs = np.concatenate(obs); assert obs.shape[1] == OBS_DIM, obs.shape
    return obs, np.concatenate(qref), np.concatenate(grip), np.array(starts), np.array(lens), np.array(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--obs-noise", type=float, default=0.02, help="gaussian noise, normalized-obs units")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    obs, qref, grip, starts, lens, seeds = load(a.shards, only_success=not a.include_failures)
    act = build_targets(obs, qref, grip, starts, lens, K, STRIDE, RQ)
    n_ep = len(starts)
    perm = rng.permutation(n_ep)
    n_val = max(1, int(round(a.val_frac * n_ep)))
    val_eps, tr_eps = perm[:n_val], perm[n_val:]
    idx_of = lambda eps: np.concatenate([np.arange(starts[e], starts[e] + lens[e]) for e in eps])
    tr_idx, va_idx = idx_of(tr_eps), idx_of(val_eps)

    om, os_ = obs[tr_idx].mean(0), obs[tr_idx].std(0) + 1e-3
    am, as_ = act[tr_idx].mean(0), act[tr_idx].std(0) + 1e-3
    to = lambda x: torch.as_tensor(x, device=dev, dtype=torch.float32)
    O, A = to((obs - om) / os_), to((act - am) / as_)
    trI, vaI = torch.as_tensor(tr_idx, device=dev), torch.as_tensor(va_idx, device=dev)
    As = to(as_)

    model = ChunkMLP(OBS_DIM, K + 1, 7, a.hidden, a.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    w = torch.ones(7, device=dev); w[6] = 2.0

    def val_metrics():
        model.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, len(vaI), 8192):
                b = vaI[i:i + 8192]
                errs.append(((model(O[b]) - A[b]).abs() * As).cpu())   # physical units
            e = torch.cat(errs)
        model.train()
        return dict(val_l1_norm=float((e / As.cpu()).mean()),
                    val_joint_mae_rad_j1=float(e[:, 1, :6].mean()), val_joint_mae_rad_jK=float(e[:, K, :6].mean()),
                    val_grip_mae_j0=float(e[:, 0, 6].mean()))

    curve, t0 = [], time.time()
    print(f"[bc] dev={dev} episodes train={len(tr_eps)} val={len(val_eps)} samples train={len(tr_idx)} "
          f"val={len(va_idx)} obs={OBS_DIM} act=({K + 1},7)", flush=True)
    run = 0.0
    for it in range(1, a.steps + 1):
        b = trI[torch.randint(len(trI), (a.bs,), device=dev)]
        x = O[b]
        if a.obs_noise > 0:
            x = x + a.obs_noise * torch.randn_like(x)
        loss = (torch.nn.functional.smooth_l1_loss(model(x), A[b], reduction="none", beta=0.05) * w).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * float(loss.detach()) if it > 1 else float(loss.detach())
        if it % 1000 == 0 or it == a.steps:
            m = dict(step=it, train_loss=run, wall_s=time.time() - t0, **val_metrics())
            curve.append(m); print(json.dumps(m), flush=True)
    cfg = dict(obs_dim=OBS_DIM, K=K, stride=STRIDE, hidden=a.hidden, depth=a.depth,
               train_seeds=seeds[tr_eps].tolist(), val_seeds=seeds[val_eps].tolist(), args=vars(a))
    torch.save(dict(model=model.state_dict(), cfg=cfg, obs_mean=om, obs_std=os_, act_mean=am, act_std=as_),
               out / "bc.pt")
    (out / "curve.json").write_text(json.dumps(curve, indent=1))
    print(f"[bc] saved {out / 'bc.pt'}", flush=True)


if __name__ == "__main__":
    main()
