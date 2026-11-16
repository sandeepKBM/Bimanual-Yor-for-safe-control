"""wm/dyn.py -- the `Dyn` object reach_value.train calls, for the tray (port of wm_audit.value_gate.ProjWMDyn).
SIMULATION / SAVED DATA ONLY.

Interface (torch, batched): U, sample_states(B), step(z, u), margin(z), features(z), P (P.m_lo is the value floor).
z = [h, o_n]  (GRU state, normalised obs). Differences against ProjWMDyn:
  * U is the SCALED fallback set of the env (TrayRepEnv.U = 85 elements: accel x a_scale, tilt rate x w_scale)
  * P.m_lo = -0.5 (the tray margin floor), P.tilt_lim = 1.5
  * `alpha` (tilt-rate slew limit, default None): the tilt rate fed to the WM is the one that would REACH the tray,
    w_prev + clip(w_cmd - w_prev, +-alpha*dt), with w_prev = the realised rate in the latent's obs (columns 6:9). The WM was
    trained on realised rates (act[3:6] = the realised axis rate), so this keeps training and imagination consistent.
  * the tilt projection (zero the rate when the WM's own axis nears 1.5 rad) is kept; the tray tilt stays < 0.4 rad (p99)
    so it almost never fires. It reads obs columns 0..2 (the axis), the same in `full` and `tray_only`.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep.wm import common as C


class TrayDyn:
    def __init__(self, wm, bank, device, U, project=True, alpha=None):
        import torch
        self.wm, self.dev, self.bank, self.project, self.alpha = wm, device, bank, project, alpha
        self.U = torch.as_tensor(np.asarray(U), dtype=torch.float32, device=device)
        self.dh = wm.d_h
        self.P = C.TrayP

    def sample_states(self, B):
        import torch
        return self.bank[torch.randint(len(self.bank), (B,), device=self.dev)]

    def effective_u(self, o_n, u):
        """the action the WM should see: slew-limited tilt rate (if alpha), then the tilt projection (if project)."""
        import torch
        u = u.clone()
        if self.alpha is not None:
            w_prev = o_n[:, 6:9] * self.wm.o_sd[6:9] + self.wm.o_mu[6:9]
            lim = self.alpha * C.DT_WM
            u[:, 3:] = w_prev + torch.clamp(u[:, 3:] - w_prev, -lim, lim)
        if self.project:
            c = o_n[:, :3] * self.wm.o_sd[:3] + self.wm.o_mu[:3]
            c = c / (c.norm(dim=-1, keepdim=True) + 1e-9)
            tilt = torch.arccos(c[:, 2].clamp(-1, 1))
            dcz = u[:, 3] * c[:, 1] - u[:, 4] * c[:, 0]           # (w x c)_z; < 0 increases tilt
            block = (tilt >= self.P.tilt_lim - 0.02) & (dcz < 0)
            u[block, 3:] = 0.0
        return u

    def step(self, z, u):
        import torch
        shp = torch.broadcast_shapes(z.shape[:-1], u.shape[:-1])
        z = z.expand(*shp, z.shape[-1]).reshape(-1, z.shape[-1])
        u = u.expand(*shp, u.shape[-1]).reshape(-1, u.shape[-1])
        h, o_n = z[:, :self.dh], z[:, self.dh:]
        u = self.effective_u(o_n, u)
        h2, o2 = self.wm.step(h, o_n, u)
        return torch.cat([h2, o2], -1).view(*shp, -1)

    def margin(self, z):
        return self.wm.margin(z[..., :self.dh], z[..., self.dh:])

    def features(self, z):
        return z
