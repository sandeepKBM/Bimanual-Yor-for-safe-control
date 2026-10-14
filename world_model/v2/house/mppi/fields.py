"""fields.py -- clearance fields for the batched rollouts.  SIMULATION ONLY.

Zono3DField : min over perceived zonotopes of sd_lb(p) = max_f (n_f.p - d_f)  (zono_hrep.sd_lb, the quantity the HOCBF / base-filter rows use),
              tabulated once per perception update on a regular (x, y, z) grid ON THE GPU and read with trilinear interpolation
              (F.grid_sample).  The union is NON-CONVEX (min over zonotopes): the planner may go around either side of any of them.
              The interpolation of a piecewise-linear convex function lies above the function: error <~ res/2 at facet creases -> `res`
              is 1.5 cm; the hard filter keeps its own exact rows.
Grid2DField : the shield's 1-cm EDT of the occupancy grid (OccGrid.edt), bilinear lookup.  z is ignored.
Both: query(pw (...,P,2) world xy, z (P,)) -> (...,P) clearance of a point (before subtracting its radius / margin).
people_clearance: distance of footprint points to predicted person discs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pack_hreps(hreps, device):
    """list of HRep -> (Fall,3) normals, (Fall,) offsets, (Z,Fmax) gather index into [0..Fall] (Fall = -inf pad column)."""
    N = np.concatenate([h.N for h in hreps]); d = np.concatenate([h.d for h in hreps])
    Fmax = max(h.n_facets for h in hreps)
    idx = np.full((len(hreps), Fmax), len(d), np.int64)
    o = 0
    for i, h in enumerate(hreps):
        idx[i, :h.n_facets] = np.arange(o, o + h.n_facets); o += h.n_facets
    return (torch.as_tensor(N, dtype=torch.float32, device=device), torch.as_tensor(d, dtype=torch.float32, device=device),
            torch.as_tensor(idx, device=device))


def sd_union(P, N, d, idx):
    """P (n,3) -> (n,) min_z max_{f in z} (n_f.p - d_f)."""
    S = P @ N.T - d                                                   # (n,F)
    S = torch.cat([S, torch.full((S.shape[0], 1), -1e9, device=S.device, dtype=S.dtype)], 1)
    return S[:, idx].amax(2).amin(1)                                  # (n,Z,Fmax) -> (n,)


class Zono3DField:
    def __init__(self, hreps, lo, hi, res=0.015, dz=0.04, cap=1.0, device="cuda", chunk=12000):
        self.dev = device
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        self.nx = int(np.ceil((hi[0] - lo[0]) / res)) + 1
        self.ny = int(np.ceil((hi[1] - lo[1]) / res)) + 1
        self.nz = int(np.ceil((hi[2] - lo[2]) / dz)) + 1
        self.lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
        self.span = torch.as_tensor([(self.nx - 1) * res, (self.ny - 1) * res, (self.nz - 1) * dz], dtype=torch.float32, device=device)
        self.lo_np, self.res, self.dz, self.cap = lo, res, dz, cap
        N, d, idx = pack_hreps(hreps, device)
        xs = lo[0] + res * np.arange(self.nx); ys = lo[1] + res * np.arange(self.ny); zs = lo[2] + dz * np.arange(self.nz)
        gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")           # (nz,ny,nx)
        pts = torch.as_tensor(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1), dtype=torch.float32, device=device)
        out = torch.empty(len(pts), device=device)
        for i in range(0, len(pts), chunk):
            out[i:i + chunk] = sd_union(pts[i:i + chunk], N, d, idx)
        self.vol = out.clamp(max=cap).reshape(1, 1, self.nz, self.ny, self.nx).contiguous()

    def query(self, pw, z):
        """pw (...,P,2) world xy; z (P,) heights -> (...,P) interpolated sd_lb (capped)."""
        sh = pw.shape[:-1]
        zz = z.expand(*sh)
        g = torch.stack([pw[..., 0], pw[..., 1], zz], -1).reshape(1, 1, 1, -1, 3)
        gn = 2.0 * (g - self.lo) / self.span - 1.0
        v = F.grid_sample(self.vol, gn, mode="bilinear", padding_mode="border", align_corners=True)
        return v.reshape(sh)

    def exact_np(self, P, hreps):
        from ..zono_rows.zono_hrep import sd_lb
        return np.min([sd_lb(h, P) for h in hreps], axis=0)


class Grid2DField:
    """EDT lookup (bilinear, border padding).  `grid` = teleop.scene.OccGrid (res, x0, y0, edt[ny,nx]); cell centres at x0 + (i+0.5) res."""

    def __init__(self, grid, device="cuda", cap=3.0):
        self.res, self.x0, self.y0 = float(grid.res), float(grid.x0), float(grid.y0)
        e = torch.as_tensor(np.minimum(grid.edt, cap), dtype=torch.float32, device=device)
        self.vol = e[None, None].contiguous()
        ny, nx = e.shape
        self.size = torch.as_tensor([nx, ny], dtype=torch.float32, device=device)
        self.org = torch.as_tensor([self.x0, self.y0], dtype=torch.float32, device=device)

    def query(self, pw, z=None):
        sh = pw.shape[:-1]
        u = (pw.reshape(1, 1, -1, 2) - self.org) / self.res                 # cell coordinate (0 = left edge of cell 0)
        gn = 2.0 * u / self.size - 1.0                                     # align_corners=False: -1/+1 = outer cell edges
        v = F.grid_sample(self.vol, gn, mode="bilinear", padding_mode="border", align_corners=False).reshape(-1)
        # outside the grid = obstacle (OccGrid.lookup returns 0 there); inside, never more clearance than the distance to the grid edge
        uu = u.reshape(-1, 2)
        edge = torch.minimum(uu, self.size - uu).amin(-1) * self.res
        v = torch.where(edge < 0, torch.zeros_like(v), torch.minimum(v, edge))
        return v.reshape(sh)


def people_clearance(pw, t, people, radius):
    """pw (B,H,P,2) footprint points; t (H,) time of each step; people (Np,4) [x,y,vx,vy] (zero-radius sentinel rows far away) ->
    (B,H,P) min over persons of |p - (x + v t)| - radius."""
    c = people[:, :2][None, :] + people[:, 2:][None, :] * t[:, None, None]         # (H,Np,2)
    dd = torch.linalg.norm(pw[:, :, :, None, :] - c[None, :, None, :, :], dim=-1)  # (B,H,P,Np)
    return dd.amin(-1) - radius
