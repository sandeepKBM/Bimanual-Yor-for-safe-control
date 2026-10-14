"""experiments/imitation/bc_model.py -- minimal state-based BC policy with
action chunking (MLP obs -> (K+1, 7) chunk; see il_common's ACTION SPACE)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ChunkMLP(nn.Module):
    def __init__(self, obs_dim: int, k1: int, act_dim: int = 7, hidden: int = 512, depth: int = 3):
        super().__init__()
        layers, d = [], obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, k1 * act_dim))
        self.net = nn.Sequential(*layers)
        self.k1, self.act_dim = k1, act_dim

    def forward(self, x):
        return self.net(x).view(-1, self.k1, self.act_dim)


def build_targets(obs, q_ref, grip, ep_start, ep_len, K, stride, rq_slice):
    """Chunk targets for EVERY step: a[n, j] = [q_ref[n+j*stride] - q_meas[n], grip[n+j*stride]],
    index clamped at the episode end (hold last)."""
    N = obs.shape[0]
    act = np.zeros((N, K + 1, 7), np.float32)
    for s, L in zip(ep_start, ep_len):
        n = np.arange(L)
        for j in range(K + 1):
            idx = s + np.minimum(n + j * stride, L - 1)
            act[s + n, j, :6] = q_ref[idx] - obs[s + n, rq_slice]
            act[s + n, j, 6] = grip[idx]
    return act


class BCPolicy:
    """CPU inference wrapper used in the closed loop: __call__(obs f32 (OBS_DIM,)) -> (K+1, 7)."""

    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.cfg = ck["cfg"]
        self.model = ChunkMLP(self.cfg["obs_dim"], self.cfg["K"] + 1, 7,
                              self.cfg["hidden"], self.cfg["depth"]).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.om, self.os = [torch.as_tensor(ck[k], device=device) for k in ("obs_mean", "obs_std")]
        self.am, self.as_ = [torch.as_tensor(ck[k], device=device) for k in ("act_mean", "act_std")]
        self.device = device
        torch.set_num_threads(1)

    @torch.no_grad()
    def __call__(self, obs):
        x = (torch.as_tensor(obs, device=self.device, dtype=torch.float32)[None] - self.om) / self.os
        y = self.model(x)[0] * self.as_ + self.am
        return y.cpu().numpy().astype(np.float64)
