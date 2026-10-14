"""wm/policy.py -- WM-greedy closed-loop policy for tray_rep.cl_eval_mj.evaluate. SIMULATION / SAVED DATA ONLY.

Each tick: u = argmax_{u in U} V(f_WM(z, u)), z = [h, o_n] the filtered latent, U = the env's scaled 85-element set.
The latent is filtered on the REAL MuJoCo observation stream (nothing is synthesised):
  first call of a state: filter the whole logged history  obs_hist[:t+1], act_hist[:t]  from tick 0 (as cl_eval_mj hands it)
  later calls:           h <- GRU(h, o_n_prev, act_hist[-1]);  o_n <- no(obs_hist[-1])
  act_hist[-1] is the realised action of the last tick: [a, realised axis rate] (what the WM was trained on).
`needs_acts = True`, `begin(ctx)` resets the state. The candidate actions go through TrayDyn.step, which applies the
slew-limit model (alpha) and the tilt projection, exactly as in value training.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep.wm import common as C
from world_model.v2.tray_rep.wm.dyn import TrayDyn


class WMGreedy:
    needs_acts = True

    def __init__(self, wm, net, dev, U, alpha=None):
        import torch
        self.torch, self.wm, self.net, self.dev = torch, wm, net, dev
        self.dyn = TrayDyn(wm, None, dev, U, True, alpha)
        self.U_np = np.asarray(U, np.float64)
        self.h = self.o_n = None
        self.n = -1
        self.last_q = None

    def begin(self, ctx):
        self.h = self.o_n = None
        self.n = -1

    @property
    def z(self):
        return self.torch.cat([self.h, self.o_n], -1)

    def _sync(self, obs_hist, act_hist):
        torch, wm, dev = self.torch, self.wm, self.dev
        n = len(obs_hist)
        with torch.no_grad():
            if self.h is None or n != self.n + 1:                      # first call (or out of sync): filter the history
                o = torch.tensor(np.asarray(obs_hist), dtype=torch.float32, device=dev)[None]
                a = torch.tensor(np.asarray(act_hist), dtype=torch.float32, device=dev)
                a = torch.cat([a, torch.zeros(1, 6, device=dev)], 0)[None]           # act[t] is not used for H_t
                H, on = C.filter_fast(wm, o, a)
                self.h, self.o_n = H[:, -1], on[:, -1]
            else:
                a = torch.tensor(np.asarray(act_hist[-1]), dtype=torch.float32, device=dev)[None]
                self.h, _ = wm.step(self.h, self.o_n, a)
                self.o_n = wm.no(torch.tensor(np.asarray(obs_hist[-1]), dtype=torch.float32, device=dev)[None])
        self.n = n

    def q(self):
        """V(f(z, u)) for every u in U -> (nu,) tensor."""
        torch = self.torch
        nu = len(self.dyn.U)
        with torch.no_grad():
            nxt = self.dyn.step(self.z.expand(nu, -1), self.dyn.U)
            return self.net(nxt, self.dyn.margin(nxt))

    def __call__(self, obs_hist, act_hist):
        self._sync(obs_hist, act_hist)
        q = self.q()
        self.last_q = q
        return self.U_np[int(q.argmax())]
