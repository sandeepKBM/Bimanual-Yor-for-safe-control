"""wm/trainer.py -- the vdirect Trainer (vdirect/trainer.py) with the pendulum wiring replaced. SIMULATION / SAVED DATA ONLY.

Same recipe as vdirect: aux-w 1.0, branch-frac 0.5, batch 64, window 160, burn 60, K 10, n_os 1024, w_mar 2.0, AdamW 1e-3 +
cosine to 5% over TOTAL steps, 30% of nominal windows start at tick 0, failing episodes weighted 2x. The step count is a
parameter (vdirect: S_TOTAL = 15000).
Changes, all forced by the new data:
  * obs width `d_obs` (13 or 9) in the WM, the aux head input (d_h + d_obs) and the normalisation buffers
  * data from a tray dataset dir; aux target = env.feats_tray(priv)
  * the branch pool comes from tray_rep.branches (npz); no branches = the N0 arm
  * the checkpoint stores d_obs, so wm.pt reloads (wm.model.load_wm)
  * the window needs lens > 161; shorter episodes are never sampled (a few % at the piper intensity; reported)
The loss function and `train` loop are copied unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.wm import common as C
from world_model.v2.tray_rep.wm import model as M

BATCH, LR, WINDOW, BURN, KSTEP, N_OS, W_MAR, AUX_W, BRANCH_FRAC = 64, 1e-3, 160, 60, 10, 1024, 2.0, 1.0, 0.5


class TrayTrainer:
    def __init__(self, seed, norm, data, total_steps=15000, dev="cuda", batch=BATCH, lr=LR, window=WINDOW, burn=BURN,
                 K=KSTEP, n_os=N_OS, w_mar=W_MAR, aux_w=AUX_W, branch_frac=BRANCH_FRAC):
        import torch
        self.cfg = dict(seed=seed, total_steps=total_steps, batch=batch, lr=lr, window=window, burn=burn, K=K, n_os=n_os,
                        w_mar=w_mar, aux_w=aux_w, branch_frac=branch_frac, data=str(data))
        self.dev = dev
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.tr, self.va, self.gtr, self.gva = C.torch_data(data, dev)
        self.d_obs = int(self.tr["obs"].shape[1])
        assert len(norm["o_mu"]) == self.d_obs, "normalisation width != data obs width"
        self.wm = M.build_wm(self.d_obs, dev)
        self.aux = M.build_aux(self.wm).to(dev)
        self.params = list(self.wm.parameters()) + list(self.aux.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, total_steps, eta_min=lr * 0.05)
        for k in ("o_mu", "o_sd", "a_mu", "a_sd"):
            getattr(self.wm, k).copy_(torch.tensor(norm[k], dtype=torch.float32))
        self.step = 0
        self.BR = None
        self.curve = []
        n_ok = int((self.tr["lens"] > window + 1).sum())
        self.window_eps = dict(n_eps=int(len(self.tr["lens"])), n_long_enough=n_ok)

    # ------------------------------------------------------------------ branch pool
    def add_branches(self, BR):
        import torch
        BR = {k: BR[k].to(self.dev) for k in ("obs", "act", "l", "feat")}
        self.BR = BR if self.BR is None else {k: torch.cat([self.BR[k], BR[k]], 0) for k in BR}

    def n_branches(self):
        return 0 if self.BR is None else int(self.BR["obs"].shape[0])

    # ------------------------------------------------------------------ batches and loss (vdirect.trainer, unchanged)
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
            if self.step % log_every == 0 or self.step == 1 or self.step == self.cfg["total_steps"]:
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
        wm = M.build_wm(self.d_obs, self.dev)
        wm.load_state_dict(self.wm.state_dict()); wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)
        return wm

    def save_wm(self, path, extra=None):
        import torch
        torch.save(dict(state={k: v.cpu() for k, v in self.wm.state_dict().items()}, args=self.cfg, d_obs=self.d_obs,
                        step=self.step, n_branch=self.n_branches(),
                        aux_state={k: v.cpu() for k, v in self.aux.state_dict().items()}) | (extra or {}), path)

    def save_ckpt(self, path):
        """atomic; the branch pool is not stored (the caller re-adds it from its npz)."""
        import torch
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(dict(wm=self.wm.state_dict(), aux=self.aux.state_dict(), opt=self.opt.state_dict(),
                        sched=self.sched.state_dict(), step=self.step, cfg=self.cfg, curve=self.curve,
                        np_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        n_branch=self.n_branches(), d_obs=self.d_obs), tmp)
        tmp.replace(path)

    def load_ckpt(self, path):
        import torch
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.wm.load_state_dict(ck["wm"]); self.aux.load_state_dict(ck["aux"])
        self.opt.load_state_dict(ck["opt"]); self.sched.load_state_dict(ck["sched"])
        self.step = ck["step"]; self.curve = list(ck["curve"])
        self.rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())                 # setters need CPU ByteTensors (vdirect owner fix)
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        return self


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
