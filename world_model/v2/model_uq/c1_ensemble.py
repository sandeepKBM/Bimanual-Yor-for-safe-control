"""model_uq/c1_ensemble.py -- C1: sim-prior residual model with calibrated uncertainty. SIMULATION / SAVED DATA ONLY.

Spirit of SIM-FSVGD (Rothfuss et al. 2403.16644): the analytic model is the prior MEAN, the learned part is the residual; here the prior over
the (log) residual-magnitude function is "a global constant" m0 with std s0 (function-space prior), imposed as a regulariser at random context
points in the FIT feature box. Heteroscedastic deep ensemble on z = log(y + 1e-3), normalised split-conformal on CAL:
    score_i = (z_i - m(x_i)) / max(s(x_i), S_MIN),  q = conformal quantile,  bound(x) = exp(m(x) + q s(x)) - 1e-3.
Method API (protocol.py): fit_calibrate(Xf, yf, Xc, yc, alpha, rng) -> self ; bound(X) -> (m,) ; name.
mean_only=True: constant s (scale-free ablation) -> bound = exp(m(x) + q*s_const) - 1e-3.
"""
from __future__ import annotations

import numpy as np
import torch

from world_model.v2.model_uq import protocol as P

EPS_Y = 1e-3
S_MIN = 0.05
LV_MIN, LV_MAX = -7.0, 3.0


class _Net(torch.nn.Module):
    def __init__(self, d, h=32, m0=0.0, lv0=-1.0):
        super().__init__()
        self.l1 = torch.nn.Linear(d, h)
        self.l2 = torch.nn.Linear(h, h)
        self.out = torch.nn.Linear(h, 2)
        with torch.no_grad():
            self.out.weight.mul_(0.3)
            self.out.bias.copy_(torch.tensor([m0, lv0]))      # mean head bias = prior mean m0

    def forward(self, x):
        h = torch.tanh(self.l2(torch.tanh(self.l1(x))))
        o = self.out(h)
        return o[:, 0], o[:, 1].clamp(LV_MIN, LV_MAX)


class SimPriorEnsemble:
    def __init__(self, M=5, hidden=32, epochs=300, lr=1e-2, wd=1e-4, lam=0.3, lam_v=0.1, n_ctx=128, mean_only=False, name=None):
        self.M, self.hidden, self.epochs, self.lr, self.wd, self.lam, self.n_ctx = M, hidden, epochs, lr, wd, lam, n_ctx
        self.lam_v = lam_v
        self.mean_only = mean_only
        self.name = name or ("c1_simprior_meanonly" if mean_only else "c1_simprior")

    # ---- prediction ---------------------------------------------------------------------------------------------------------------
    def _predict(self, X):
        xs = np.clip((np.asarray(X, np.float64) - self.mu) / self.sd, self.lo, self.hi)   # no extrapolation beyond the FIT box (exp() blow-up)
        x = torch.as_tensor(xs, dtype=torch.float32)
        ms, vs = [], []
        with torch.no_grad():
            for net in self.nets:
                m, lv = net(x)
                ms.append(m.double().numpy())
                vs.append(np.exp(lv.double().numpy()))
        ms, vs = np.stack(ms), np.stack(vs)
        mean = ms.mean(0)
        s = np.sqrt(vs.mean(0) + ms.var(0))
        return mean, s

    def _mean_std(self, X):
        m, s = self._predict(X)
        if self.mean_only:
            s = np.full_like(m, self.s_const)
        # prior-based safeguards (deterministic given FIT, applied before calibration -> validity kept): the mean stays within the FIT range
        # +-0.5 and the std within [S_MIN, s0] (s0 = prior = marginal std; conditional std <= marginal) so sparse-data extrapolation cannot explode exp()
        m = np.clip(m, self.zlo, self.zhi)
        return m, np.clip(s, S_MIN, self.s_cap)

    # ---- fit + calibrate ----------------------------------------------------------------------------------------------------------
    def fit_calibrate(self, Xf, yf, Xc, yc, alpha=P.ALPHA, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        Xf = np.asarray(Xf, np.float64)
        zf = np.log(np.asarray(yf, np.float64) + EPS_Y)
        d = Xf.shape[1]
        self.mu = Xf.mean(0)
        self.sd = np.maximum(Xf.std(0), 1e-6)
        m0 = float(zf.mean())
        s0 = max(float(zf.std()), 0.1)
        Xs = (Xf - self.mu) / self.sd
        lo, hi = Xs.min(0), Xs.max(0)
        self.lo, self.hi = lo, hi
        self.zlo, self.zhi, self.s_cap = float(zf.min()) - 0.5, float(zf.max()) + 0.5, max(s0, S_MIN)
        xt = torch.as_tensor(Xs, dtype=torch.float32)
        zt = torch.as_tensor(zf, dtype=torch.float32)
        lo_t, hi_t = torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
        n = len(Xf)
        self.nets = []
        for i in range(self.M):
            seed = int(rng.integers(0, 2**31 - 1))
            torch.manual_seed(seed)
            net = _Net(d, self.hidden, m0, 2.0 * float(np.log(s0)))
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            g = torch.Generator().manual_seed(seed + 1)
            bs = n if n <= 512 else 512
            for ep in range(self.epochs):
                perm = torch.randperm(n, generator=g)[:bs] if bs < n else slice(None)
                m, lv = net(xt[perm])
                nll = 0.5 * (lv + (zt[perm] - m) ** 2 * torch.exp(-lv)).mean()
                xc = lo_t + (hi_t - lo_t) * torch.rand(self.n_ctx, d, generator=g)
                mc, lvc = net(xc)
                reg = self.lam * ((mc - m0) ** 2 / s0**2).mean() + self.lam_v * ((lvc - 2.0 * np.log(s0)) ** 2).mean()
                loss = nll + reg
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            self.nets.append(net.eval())
        # constant scale for the ablation: rms predicted std over FIT
        _, s_fit = self._predict(Xf)
        self.s_const = float(np.sqrt(np.mean(s_fit**2)))
        # normalised split-conformal on CAL
        mc_, sc_ = self._mean_std(np.asarray(Xc, np.float64))
        zc = np.log(np.asarray(yc, np.float64) + EPS_Y)
        self.q = P.conformal_quantile((zc - mc_) / sc_, alpha)
        return self

    def bound(self, X):
        if not np.isfinite(self.q):
            return np.full(len(X), np.inf)
        m, s = self._mean_std(X)
        return np.maximum(np.exp(m + self.q * s) - EPS_Y, 0.0)
