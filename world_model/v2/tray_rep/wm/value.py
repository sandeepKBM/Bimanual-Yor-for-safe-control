"""wm/value.py -- latent-HJ value on the tray WM (port of vdirect.run_arm.train_value + value_gate.build_bank).
SIMULATION / SAVED DATA ONLY. reach_value.train is used unchanged with the vdirect hyper-parameters:
iters (VAL_IT = 10000), batch 2048, gamma 0.8 -> 0.98, K = 4, lam = 0.5, state bank = filtered TRAIN latents (stride 2) +
200k imagined states from random fallback actions (10 steps).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.wm import common as C
from world_model.v2.tray_rep.wm.dyn import TrayDyn


def build_bank(wm, data, dev, U, alpha=None, stride=2, n_imag=200000, imag_k=10, seed=0, project=True):
    import torch
    D = C.load_data(data, "train")
    Z = C.filtered_latents(wm, D, dev)
    Z = torch.tensor(np.concatenate([z[::stride] for z in Z], 0), device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    dyn = TrayDyn(wm, Z, dev, U, project, alpha)
    zi = Z[torch.randint(len(Z), (n_imag // imag_k,), device=dev, generator=g)]
    out = [Z]
    with torch.no_grad():
        for _ in range(imag_k):
            u = dyn.U[torch.randint(len(dyn.U), (len(zi),), device=dev, generator=g)]
            zi = dyn.step(zi, u)
            out.append(zi)
    return torch.cat(out, 0)


def train_value(wm, data, dev, out, iters, seed, U, alpha=None, batch=2048, n_imag=200000, log=print, extra_meta=None):
    """-> (net frozen eval on dev, dyn). Writes out/value.pt and out/curve.json."""
    from world_model.v2.latent_hj import reach_value as rv
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bank = build_bank(wm, data, dev, U, alpha, seed=seed, n_imag=n_imag)
    dyn = TrayDyn(wm, bank, dev, U, True, alpha)
    log(f"[value] bank {tuple(bank.shape)} in {time.time()-t0:.0f}s")
    net, curve = rv.train(dyn, out, iters=iters, batch=batch, gamma0=0.8, gamma_final=0.98, K=4, lam=0.5, device=dev,
                          log_every=max(iters // 5, 1), seed=seed)
    meta = dict(dx=int(bank.shape[1]), hidden=256, layers=3, gamma=0.98, K=4, kind="tray_rep", P=C.TrayP.as_dict(),
                project=True, iters=iters, seed=seed, d_obs=int(wm.d_obs), alpha=alpha, n_bank=int(len(bank))) | (extra_meta or {})
    rv.save(net.cpu(), out / "value.pt", meta)
    (out / "curve.json").write_text(json.dumps(curve))
    net = net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    log(f"[value] {iters} iters in {time.time()-t0:.0f}s")
    return net, dyn


def load_value(path, dev):
    from world_model.v2.latent_hj import reach_value as rv
    net, meta = rv.load(str(path), dev)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, meta
