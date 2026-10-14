"""dense_octree.py -- drop-in replacement for FoundationPose's kaolin-based `Utils.OctreeManager` (foundation env only).

Why: BundleSDF's Neural Object Field (the model-free onboarding of FoundationPose) hard-requires kaolin (sparse-octree ray tracing).  kaolin is not on PyPI
(NVIDIA S3 / GitHub wheels only) and the conda-forge build needs conda-pytorch, so it is NOT installed.  This class provides the three methods the NOF uses
(ray_trace, get_center_ids, draw) on a dense boolean occupancy grid at the octree's finest level (<=128^3), with ONE in/out interval per ray (first / last
occupied sample along the ray).  Same occupied-space culling idea; slightly looser than the real octree for non-convex objects (cup interior), no change to the
network / losses.  Numerics of the NOF therefore differ slightly from upstream: stated in the report.
"""
from __future__ import annotations

import numpy as np
import torch
import trimesh


class DenseOctreeManager:
    def __init__(self, pts=None, max_level=None, octree=None):
        assert pts is not None
        self.max_level = int(max_level)
        n = 2 ** self.max_level
        self.finest_vox_size = 2.0 / n
        idx = torch.floor((pts + 1.0) / self.finest_vox_size).long().clamp(0, n - 1)
        g = torch.zeros((n, n, n), dtype=torch.bool, device=pts.device)
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self.grid = g
        self.octree = g                      # nerf_runner stores it in a dict when saving checkpoints
        self.n_level = self.max_level + 1
        self._lv = {self.max_level: g}

    def _grid_at(self, level):
        level = min(int(level), self.max_level)
        if level not in self._lv:
            f = 2 ** (self.max_level - level)
            g = self.grid.float()[None, None]
            self._lv[level] = (torch.nn.functional.max_pool3d(g, f, f)[0, 0] > 0)
        return self._lv[level], level

    def get_vox_size_at_level(self, level):
        return 2.0 / (2 ** level)

    def get_center_ids(self, x, level):
        g, lv = self._grid_at(level)
        n = g.shape[0]
        i = torch.floor((x + 1.0) / (2.0 / n)).long()
        inb = ((i >= 0) & (i < n)).all(-1)
        i = i.clamp(0, n - 1)
        occ = g[i[:, 0], i[:, 1], i[:, 2]] & inb
        return torch.where(occ, torch.arange(len(x), device=x.device), -torch.ones(len(x), dtype=torch.long, device=x.device))

    def draw(self, level, method="point"):
        g, lv = self._grid_at(level)
        n = g.shape[0]
        ii = torch.nonzero(g).float()
        return trimesh.points.PointCloud(((ii + 0.5) * (2.0 / n) - 1).cpu().numpy().reshape(-1, 3))

    @torch.no_grad()
    def ray_trace(self, rays_o, rays_d, level, debug=False):
        g, lv = self._grid_at(level)
        n = g.shape[0]
        vox = 2.0 / n
        dev = rays_o.device
        N = rays_o.shape[0]
        d = rays_d / rays_d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        # slab test against the [-1,1]^3 cube
        inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t0 = (-1.0 - rays_o) * inv
        t1 = (1.0 - rays_o) * inv
        tmin = torch.minimum(t0, t1).max(-1)[0].clamp(min=0.0)
        tmax = torch.maximum(t0, t1).min(-1)[0]
        hit = tmax > tmin
        step = vox * 0.5
        K = int(np.ceil(2 * np.sqrt(3.0) / step)) + 1
        ts = torch.arange(K, device=dev, dtype=torch.float32)[None] * step + tmin[:, None]     # (N,K)
        pts = rays_o[:, None, :] + d[:, None, :] * ts[..., None]
        i = torch.floor((pts + 1.0) / vox).long()
        inb = ((i >= 0) & (i < n)).all(-1) & (ts <= tmax[:, None]) & hit[:, None]
        i = i.clamp(0, n - 1)
        occ = g[i[..., 0], i[..., 1], i[..., 2]] & inb                                          # (N,K)
        any_ = occ.any(-1)
        first = torch.argmax(occ.int(), dim=-1)
        last = K - 1 - torch.argmax(occ.flip(-1).int(), dim=-1)
        t_in = torch.gather(ts, 1, first[:, None])[:, 0] - 0.5 * vox
        t_out = torch.gather(ts, 1, last[:, None])[:, 0] + 0.5 * vox + step
        t_in = torch.where(any_, t_in.clamp(min=0), torch.zeros_like(t_in))
        t_out = torch.where(any_, t_out, torch.zeros_like(t_out))
        depths_in_out = torch.stack([t_in, t_out], -1)[:, None, :].contiguous()                  # (N,1,2)
        rays_near = t_in.reshape(-1, 1)
        rays_far = t_out.reshape(-1, 1)
        rays_pid = -torch.ones_like(rays_o[:, :1])
        return rays_near, rays_far, rays_pid, depths_in_out
