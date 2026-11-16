"""robust/lat.py -- a running WM latent on a (possibly corrupted) observation stream, with batched value queries.
SIMULATION / SAVED DATA ONLY (torch CPU, one thread). No hardware, no network.

LatentTrack mirrors wm/policy.WMGreedy (same filter, same incremental update) and adds
  * resid        ||dec(h_t) - no(obs_t)||  over all 13 normalised obs numbers (and `resid_ball` over the 4 ball ones):
                 the gap between the latent the WM PREDICTED for this tick and the one the encoder gets from the observation
                 (the Cao & Bansal residual). NaN on the first call of a carry.
  * obs variants: V and Q = V(f(z, u)) evaluated at K variants of the latest ball observation, in one batch.
Conventions: z = [h, o_n]; the value is  V(z) = net(z, margin(z));  Q(z, u) = V(f(z, u)); the fallback policy is
argmax_u Q(z, u) over the env's scaled 85-element set (the F3 fallback).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.wm import common as C
from world_model.v2.tray_rep.wm.dyn import TrayDyn

BALL = slice(9, 13)


class LatentTrack:
    def __init__(self, wm, net, U, alpha=None, dev="cpu", name=""):
        import torch
        torch.set_num_threads(1)
        self.torch, self.wm, self.net, self.dev, self.name = torch, wm, net, dev, name
        self.dyn = TrayDyn(wm, None, dev, U, True, alpha)
        self.Ut = self.dyn.U
        self.U_np = np.asarray(U, np.float64)
        self.begin()

    def begin(self):
        self.h = self.o_n = None
        self.n = -1
        self.resid = self.resid_ball = float("nan")

    # ------------------------------------------------------------------ filtering
    def sync(self, obs_hist, act_hist):
        torch, wm, dev = self.torch, self.wm, self.dev
        n = len(obs_hist)
        with torch.no_grad():
            if self.h is None or n != self.n + 1:
                o = torch.tensor(np.asarray(obs_hist), dtype=torch.float32, device=dev)[None]
                a = torch.tensor(np.asarray(act_hist), dtype=torch.float32, device=dev)
                a = torch.cat([a, torch.zeros(1, 6, device=dev)], 0)[None]
                H, on = C.filter_fast(wm, o, a)
                self.h, self.o_n = H[:, -1], on[:, -1]
                self.resid = self.resid_ball = float("nan")
            else:
                a = torch.tensor(np.asarray(act_hist[-1]), dtype=torch.float32, device=dev)[None]
                self.h, o_pred = wm.step(self.h, self.o_n, a)
                self.o_n = wm.no(torch.tensor(np.asarray(obs_hist[-1]), dtype=torch.float32, device=dev)[None])
                d = (o_pred - self.o_n)[0]
                self.resid = float(d.norm())
                self.resid_ball = float(d[BALL].norm())
        self.n = n

    # ------------------------------------------------------------------ variants of the latest observation
    def ball_variants(self, balls):
        """balls (K, 4) RAW [x, y, vx, vy] -> o_n variants (K, 13) (only the ball columns change)."""
        torch, wm = self.torch, self.wm
        b = torch.as_tensor(np.asarray(balls), dtype=torch.float32, device=self.dev)
        o = self.o_n.expand(len(b), -1).clone()
        o[:, BALL] = (b - wm.o_mu[BALL]) / wm.o_sd[BALL]
        return o

    def radial_variants(self, r):
        """8 points at distance r (normalised-obs units) from the latest o_n along +-e_k of the 4 ball columns, plus the
        centre -> (9, 13)."""
        torch = self.torch
        o = self.o_n.expand(9, -1).clone()
        for j in range(4):
            o[1 + 2 * j, 9 + j] += r
            o[2 + 2 * j, 9 + j] -= r
        return o

    # ------------------------------------------------------------------ values
    def _z(self, o_var):
        torch = self.torch
        if o_var is None:
            o_var = self.o_n
        o_var = o_var.reshape(-1, o_var.shape[-1])
        return torch.cat([self.h.expand(len(o_var), -1), o_var], -1)

    def v_state(self, o_var=None):
        """V(z) for each variant -> (K,) numpy."""
        with self.torch.no_grad():
            z = self._z(o_var)
            return self.net(z, self.dyn.margin(z)).cpu().numpy()

    def q(self, o_var=None, u_extra=None):
        """Q = V(f(z_k, u_j)) -> (K, J) numpy. J = len(u_extra) + 85 (u_extra rows first)."""
        torch = self.torch
        U = self.Ut
        if u_extra is not None:
            ue = torch.as_tensor(np.atleast_2d(u_extra), dtype=torch.float32, device=self.dev)
            U = torch.cat([ue, U], 0)
        with torch.no_grad():
            z = self._z(o_var)                                             # (K, dz)
            Kn, J = len(z), len(U)
            nxt = self.dyn.step(z[:, None, :].expand(Kn, J, -1), U[None].expand(Kn, J, -1))
            return self.net(nxt, self.dyn.margin(nxt)).cpu().numpy()


def load_tracks(runs_root, model_setup, arm, seeds, U, alpha, dev="cpu"):
    from world_model.v2.tray_rep.bridge import filters as F
    out = []
    for s in seeds:
        wm, net, d = F.load_arm(runs_root, model_setup, arm, int(s), "full", dev)
        out.append(LatentTrack(wm, net, U, alpha, dev, name=f"{arm}_s{s}"))
    return out
