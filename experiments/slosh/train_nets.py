#!/usr/bin/env python
"""train_nets.py -- learned spill predictors (torch, CPU). SIM DATA ONLY.

 (c)  PENDULUM + RESIDUAL: the (b1) world-frame spherical pendulum, with its
      (w, zeta) LEARNABLE (init from fit_physics) and a small MLP residual:
        c-readout : logit = alpha*(phi - phi0) + beta*relu(dphi) + MLP(x_t)
        c-dyn     : additionally u'' += MLP_dyn(u, u', g_eff)  (dynamics residual)
      x_t = [phi, dphi, theta0, theta_eff, |f|/g, f_c/g (3), j_c/100 (3), ndot_c (3)]
      Trained end-to-end with BPTT through the pendulum (T=800 steps).
 (d)  GRU reference: 1-layer GRU(hidden 32) on raw kinematics
        [n (3), f_c/g (3), v_c (3), ndot_c (3), j_c/100 (3), theta0, theta_eff]
All trained with BCE on y_H (spill within H steps) over valid (pre-onset,
post-motion-start) steps of TRAIN trajectories only; scores for all episodes are
saved to data/scores_nets.npz.

  python experiments/slosh/train_nets.py --model all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slosh_models as sm  # noqa: E402
from data_prep import load_rig, labels  # noqa: E402
from evaluate import evaluate, fmt_row, HEADER  # noqa: E402

G, DT = sm.G, sm.DT


def feats_common(K):
    fc = K["f_c"] / G
    jc = np.clip(K["j_c"] / 100.0, -20, 20)
    return fc, jc


class PendulumResidual(nn.Module):
    def __init__(self, w0, z0, dyn=False, hid=32):
        super().__init__()
        self.logw = nn.Parameter(torch.tensor(float(np.log(w0))))
        self.logz = nn.Parameter(torch.tensor(float(np.log(z0))))
        self.alpha = nn.Parameter(torch.tensor(8.0)); self.phi0 = nn.Parameter(torch.tensor(1.45))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.head = nn.Sequential(nn.Linear(14, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.dyn = dyn
        if dyn:
            self.mdyn = nn.Sequential(nn.Linear(9, 16), nn.Tanh(), nn.Linear(16, 3))
            nn.init.zeros_(self.mdyn[-1].weight); nn.init.zeros_(self.mdyn[-1].bias)

    def pendulum(self, f, n):
        """f, n: (B,T,3) tensors. Returns phi (B,T) = angle(-u, n). Same
        discretization as slosh_models.pendulum_world."""
        w = torch.exp(self.logw); c = 2 * torch.exp(self.logz) * w
        B, T, _ = f.shape
        u = -f[:, 0] / f[:, 0].norm(dim=-1, keepdim=True); ud = torch.zeros_like(u)
        phis = []
        for k in range(T):
            ge = -f[:, k] / G
            acc = w * w * (ge - (ge * u).sum(-1, keepdim=True) * u) - c * ud - (ud * ud).sum(-1, keepdim=True) * u
            if self.dyn:
                acc = acc + self.mdyn(torch.cat([u, ud / 10.0, ge], -1)) * 100.0
            ud = ud + DT * acc
            ud = ud - (ud * u).sum(-1, keepdim=True) * u
            u = u + DT * ud; u = u / u.norm(dim=-1, keepdim=True)
            cosang = (-(u * n[:, k]).sum(-1)).clamp(-1 + 1e-6, 1 - 1e-6)
            phis.append(torch.acos(cosang))
        return torch.stack(phis, 1)

    def forward(self, f, n, x):
        phi = self.pendulum(f, n)
        dphi = torch.zeros_like(phi); dphi[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / DT
        xin = torch.cat([phi[..., None], (dphi / 10.0)[..., None], x], -1)
        return self.alpha * (phi - self.phi0) + self.beta * torch.relu(dphi) + self.head(xin).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, nin=17, hid=32):
        super().__init__()
        self.gru = nn.GRU(nin, hid, batch_first=True); self.out = nn.Linear(hid, 1)

    def forward(self, x):
        return self.out(self.gru(x)[0]).squeeze(-1)


def train(model, make_batch, tr_idx, y, valid, epochs, bs, lr, log_every=5, name=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = y[tr_idx][valid[tr_idx]].mean(); pw = torch.tensor((1 - pos) / pos).clamp(max=50.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="sum")
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        perm = rng.permutation(tr_idx); tot = 0.0; t0 = time.time()
        for b in range(0, len(perm), bs):
            idx = perm[b:b + bs]
            logit = model(*make_batch(idx))
            m = torch.from_numpy(valid[idx]); yy = torch.from_numpy(y[idx].astype(np.float32))
            loss = lossf(logit[m], yy[m]) / m.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(idx)
        if ep % log_every == 0 or ep == epochs - 1:
            extra = ""
            if hasattr(model, "logw"):
                extra = f" w={float(torch.exp(model.logw)):.1f} zeta={float(torch.exp(model.logz)):.2f}"
            print(f"  [{name}] ep {ep:3d} loss {tot/len(perm):.4f} ({time.time()-t0:.1f}s/ep){extra}", flush=True)
    return model


@torch.no_grad()
def predict(model, make_batch, E, bs=512):
    out = []
    for b in range(0, E, bs):
        out.append(torch.sigmoid(model(*make_batch(np.arange(b, min(b + bs, E))))).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "rig_v1.pkl"))
    ap.add_argument("--model", default="all", choices=["all", "c", "cdyn", "gru"])
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--epochs-c", type=int, default=40)
    ap.add_argument("--epochs-gru", type=int, default=60)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=str(HERE / "data" / "scores_nets.npz"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    D = load_rig(a.data)
    K = sm.kinematics(D["pos"], D["n"])
    y, valid = labels(D["onset"], D["k0"], D["T"], a.H)
    tr_idx = np.where(~D["test"])[0]; E = len(D["onset"])
    fc, jc = feats_common(K)
    phys = json.loads(str(np.load(HERE / "data" / "scores_phys.npz")["params"]))["b1_world_pendulum"]
    t_f = torch.from_numpy(K["f"].astype(np.float32)); t_n = torch.from_numpy(K["n"].astype(np.float32))
    xc = np.concatenate([K["theta0"][..., None], K["theta_eff"][..., None], (K["fnorm"] / G)[..., None],
                         fc, jc, K["ndot_c"] / 10.0], -1).astype(np.float32)
    t_xc = torch.from_numpy(xc)
    xg = np.concatenate([K["n"], fc, K["v_c"], K["ndot_c"] / 10.0, jc, K["theta0"][..., None],
                         K["theta_eff"][..., None]], -1).astype(np.float32)
    mu, sd = xg[tr_idx].reshape(-1, xg.shape[-1]).mean(0), xg[tr_idx].reshape(-1, xg.shape[-1]).std(0) + 1e-6
    t_xg = torch.from_numpy((xg - mu) / sd)
    prev = dict(np.load(a.out)) if Path(a.out).exists() else {}
    scores = {k: v for k, v in prev.items() if not k.startswith("_")}
    todo = ["c", "cdyn", "gru"] if a.model == "all" else [a.model]
    for name in todo:
        t0 = time.time()
        if name in ("c", "cdyn"):
            mdl = PendulumResidual(phys["w"], phys["zeta"], dyn=(name == "cdyn"))
            mb = lambda idx: (t_f[idx], t_n[idx], t_xc[idx])
            train(mdl, mb, tr_idx, y, valid, a.epochs_c, 256, 3e-3, name=name)
            key = "c_pendulum_residual" if name == "c" else "c_pendulum_dyn_residual"
        else:
            mdl = GRUModel()
            mb = lambda idx: (t_xg[idx],)
            train(mdl, mb, tr_idx, y, valid, a.epochs_gru, 128, 3e-3, name=name)
            key = "d_gru"
        mdl.eval()
        scores[key] = predict(mdl, mb, E).astype(np.float32)
        torch.save(mdl.state_dict(), HERE / "data" / f"{key}.pt")
        print(f"trained {key} in {time.time()-t0:.0f}s; params={sum(p.numel() for p in mdl.parameters())}", flush=True)
        np.savez_compressed(a.out, **scores)
    print(HEADER)
    for k, s in scores.items():
        print(fmt_row(k, evaluate(s, D, H=a.H)), flush=True)


if __name__ == "__main__":
    main()
