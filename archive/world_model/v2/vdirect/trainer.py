"""vdirect/trainer.py -- kin_wm trainer for the value-directed-branch kill test. SIMULATION / SAVED DATA ONLY.

The loss, batching, optimizer, LR schedule and aux head MIRROR world_model/v2/wm_audit/train_kinwm.py
(aux-w 1.0, branch-frac 0.5, batch 64, window 160, burn 60, K 10, n_os 1024, w_mar 2.0, AdamW 1e-3,
cosine to 5% over the TOTAL step budget). Differences, identical for every arm:
  * FIXED normalization (computed once from the nominal TRAIN ticks + an equal number of ticks from a
    reference random-branch sample, i.e. what train_kinwm computes at >= ~6.6k branches), so a model can
    be warm-started and have branches added between DAgger rounds without re-normalising;
  * an extendable branch pool (random, value-directed or uncertainty-directed branches, same tensor
    format as train_kinwm.make_branches);
  * checkpoint / resume of the WM + aux + AdamW + cosine schedule + RNG streams.
The saved wm.pt is kin_wm.load_wm-compatible.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C
from world_model.v2.wm_audit.train_kinwm import feats_t

_DATA = {}


def train_data(dev):
    """TRAIN / VAL splits (+ privileged aux features) on `dev`; cached per process."""
    import torch
    if dev in _DATA:
        return _DATA[dev]
    tr, va = C.load_split("train"), C.load_split("val")
    for D in (tr, va):
        n = torch.tensor(D["n"]); nd = torch.tensor(D["nd"]); c = torch.tensor(D["obs"][:, :3])
        c = c / c.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            D["feat"] = feats_t(n, nd, c).numpy().astype(np.float32)
    gtr = {k: torch.tensor(tr[k], device=dev) for k in ("obs", "act", "l", "feat")}
    gva = {k: torch.tensor(va[k], device=dev) for k in ("obs", "act", "l", "feat")}
    _DATA[dev] = (tr, va, gtr, gva)
    return _DATA[dev]


def compute_norm(dev, n_ref=7000, seed=12345):
    """Reference normalization: nominal TRAIN ticks + an equal number of ticks from `n_ref` random
    branches (train_kinwm.make_branches, fixed seed). Statistics only -- these branches are never trained on."""
    from world_model.v2.wm_audit.train_kinwm import make_branches
    tr, _, _, _ = train_data(dev)
    BR = make_branches(tr, n_ref, 100, 160, dev, seed=seed)
    rng = np.random.default_rng(seed)
    bo = BR["obs"].reshape(-1, 9).cpu().numpy(); ba = BR["act"].reshape(-1, 6).cpu().numpy()
    nbt = min(len(tr["obs"]), len(bo))
    j = rng.choice(len(bo), nbt, replace=False)
    O = np.concatenate([tr["obs"], bo[j]], 0); A = np.concatenate([tr["act"], ba[j]], 0)
    return dict(o_mu=O.mean(0).tolist(), o_sd=(O.std(0) + 1e-3).tolist(), a_mu=A.mean(0).tolist(),
                a_sd=(A.std(0) + 1e-3).tolist(), n_ref=n_ref, seed=seed, n_nominal_ticks=int(len(tr["obs"])),
                n_branch_ticks=int(nbt))


class Trainer:
    def __init__(self, seed, norm, total_steps=15000, dev="cuda", batch=64, lr=1e-3, window=160, burn=60, K=10,
                 n_os=1024, w_mar=2.0, aux_w=1.0, branch_frac=0.5):
        import torch
        import torch.nn as nn
        from world_model.v2.latent_hj import kin_wm
        self.cfg = dict(seed=seed, total_steps=total_steps, batch=batch, lr=lr, window=window, burn=burn, K=K,
                        n_os=n_os, w_mar=w_mar, aux_w=aux_w, branch_frac=branch_frac)
        self.dev = dev
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.tr, self.va, self.gtr, self.gva = train_data(dev)
        n_tr = len(self.tr["lens"])
        self.rng.choice(n_tr, n_tr, replace=False)      # same RNG consumption as train_kinwm's --frac 1.0 subset
        self.wm = kin_wm.build_model().to(dev)
        self.aux = nn.Sequential(nn.Linear(self.wm.d_h + 9, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 10)).to(dev)
        self.params = list(self.wm.parameters()) + list(self.aux.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, total_steps, eta_min=lr * 0.05)
        for k in ("o_mu", "o_sd", "a_mu", "a_sd"):
            getattr(self.wm, k).copy_(torch.tensor(norm[k], dtype=torch.float32))
        self.step = 0
        self.BR = None
        self.curve = []

    # ------------------------------------------------------------------ branch pool
    def add_branches(self, BR):
        import torch
        BR = {k: BR[k].to(self.dev) for k in ("obs", "act", "l", "feat")}
        self.BR = BR if self.BR is None else {k: torch.cat([self.BR[k], BR[k]], 0) for k in BR}

    def n_branches(self):
        return 0 if self.BR is None else int(self.BR["obs"].shape[0])

    # ------------------------------------------------------------------ mirror of train_kinwm
    def _batch(self, D, G, B, from_branch=0):
        import torch
        W, rng, dev = self.cfg["window"], self.rng, self.dev
        ok = np.flatnonzero(D["lens"] > W + 1)
        wts = np.where(D["onset"][ok] >= 0, 2.0, 1.0); wts /= wts.sum()
        e = rng.choice(ok, B, p=wts)
        s0 = np.array([0 if rng.random() < 0.3 else rng.integers(0, D["lens"][i] - W) for i in e])
        ix = torch.tensor((D["start"][e] + s0)[:, None] + np.arange(W)[None], device=dev)
        out = [G[k][ix] for k in ("obs", "act", "l", "feat")] + [torch.tensor(s0 == 0, device=dev)]
        if from_branch and self.BR is not None:
            j = torch.randint(len(self.BR["obs"]), (from_branch,), device=dev)
            add = [self.BR[k][j] for k in ("obs", "act", "l", "feat")] + [torch.zeros(from_branch, dtype=torch.bool,
                                                                                        device=dev)]
            out = [torch.cat([x, y], 0) for x, y in zip(out, add)]
        return out

    def _loss(self, obs, act, l, feat, from_start):
        import torch
        wm, aux, c, dev = self.wm, self.aux, self.cfg, self.dev
        burn, K = c["burn"], c["K"]
        Hs, on = C.filter_fast(wm, obs, act)
        B, T, _ = obs.shape
        m = torch.ones(B, T, device=dev); m[~from_start, :burn] = 0.0
        L_obs = (((wm.dec(Hs[:, 1:]) - on[:, 1:]) ** 2).mean(-1) * m[:, 1:]).sum() / m[:, 1:].sum()
        lh = wm.margin(Hs, on)
        L_mar = (((lh - l) ** 2) * m).sum() / m.sum()
        fa = aux(torch.cat([Hs, on], -1))
        L_aux = (((fa - feat) ** 2).mean(-1) * m).sum() / m.sum()
        ts = torch.randint(burn, T - K - 1, (c["n_os"],), device=dev)
        bs = torch.randint(0, B, (c["n_os"],), device=dev)
        h = Hs[bs, ts]; o_n = on[bs, ts]
        L_os_o, L_os_m, L_os_a = 0.0, 0.0, 0.0
        for k in range(K):
            h, o_hat = wm.step(h, o_n, act[bs, ts + k])
            L_os_o = L_os_o + ((o_hat - on[bs, ts + k + 1]) ** 2).mean()
            L_os_m = L_os_m + ((wm.margin(h, o_hat) - l[bs, ts + k + 1]) ** 2).mean()
            L_os_a = L_os_a + ((aux(torch.cat([h, o_hat], -1)) - feat[bs, ts + k + 1]) ** 2).mean()
            o_n = o_hat
        tot = (L_obs + L_os_o / K + c["w_mar"] * (L_mar + L_os_m / K) + c["aux_w"] * (L_aux + L_os_a / K))
        return tot, dict(obs=float(L_obs), mar=float(L_mar), os_o=float(L_os_o / K), os_m=float(L_os_m / K),
                         aux=float(L_aux))

    def train(self, n_steps, log_every=1000, tag=""):
        import torch
        self.wm.train()
        for p in self.params:
            p.requires_grad_(True)
        B = self.cfg["batch"]
        nb = int(round(B * self.cfg["branch_frac"])) if self.BR is not None else 0
        t0 = time.time()
        for _ in range(n_steps):
            self.step += 1
            obs, act, l, feat, fs = self._batch(self.tr, self.gtr, B - nb, from_branch=nb)
            loss, lg = self._loss(obs, act, l, feat, fs)
            self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.params, 10.0)
            self.opt.step(); self.sched.step()
            if self.step % log_every == 0 or self.step == 1:
                with torch.no_grad():
                    vl = self._loss(*self._batch(self.va, self.gva, 256))[1]
                self.curve.append(dict(it=self.step, n_branch=self.n_branches(), train=lg, val=vl,
                                       sec=time.time() - t0, tag=tag))
                print(f"[wm{tag}] it {self.step} nb={self.n_branches()} ({time.time()-t0:.0f}s) "
                      + " ".join(f"{k}={v:.4f}" for k, v in lg.items())
                      + " | val " + " ".join(f"{k}={v:.4f}" for k, v in vl.items()), flush=True)
        return self

    # ------------------------------------------------------------------ frozen copy / io
    def frozen_wm(self):
        """kin_wm.load_wm-equivalent frozen copy (eval mode, no grads)."""
        from world_model.v2.latent_hj import kin_wm
        wm = kin_wm.build_model().to(self.dev)
        wm.load_state_dict(self.wm.state_dict()); wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)
        return wm

    def save_wm(self, path, extra=None):
        import torch
        torch.save(dict(state={k: v.cpu() for k, v in self.wm.state_dict().items()}, args=self.cfg,
                        step=self.step, n_branch=self.n_branches(),
                        aux_state={k: v.cpu() for k, v in self.aux.state_dict().items()}) | (extra or {}), path)

    def save_ckpt(self, path):
        import torch
        torch.save(dict(wm=self.wm.state_dict(), aux=self.aux.state_dict(), opt=self.opt.state_dict(),
                        sched=self.sched.state_dict(), step=self.step, cfg=self.cfg, curve=self.curve,
                        np_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        n_branch=self.n_branches()), path)

    def load_ckpt(self, path):
        """resume (the branch pool is NOT stored in the ckpt; the caller re-adds it)."""
        import torch
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.wm.load_state_dict(ck["wm"]); self.aux.load_state_dict(ck["aux"])
        self.opt.load_state_dict(ck["opt"]); self.sched.load_state_dict(ck["sched"])
        self.step = ck["step"]; self.curve = list(ck["curve"])
        self.rng.bit_generator.state = ck["np_rng"]
        # map_location=self.dev moves the saved ByteTensor RNG states onto the GPU, but both setters require a CPU
        # ByteTensor (owner fix 2026-09-29; CPU-only smokes never hit this). Pure bug fix: same states restored.
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        return self


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
