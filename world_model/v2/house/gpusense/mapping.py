"""gpusense/mapping.py -- GPU log-odds occupancy mapping + exact 2-D Euclidean distance transform + gradient (SIMULATION ONLY).

  * `integrate_gpu`  : torch port of loc.mapper._integrate (same ray stepping, same cell indexing incl. int() truncation, same consecutive-duplicate
                       skip, same z-band, same truncation point).  Differences: all free-space updates of a frame are applied, then all occupied
                       updates (the numba loop interleaves them per ray) and the clamp is applied after each phase -- identical on unsaturated cells.
  * `occ_mask_gpu`   : PerceivedMap.occ_mask (max over height bins > thresh, 3x3 despeckle, 3x3 binary closing with border value 0) in torch.
  * `edt_gpu`        : EXACT squared Euclidean distance transform (Felzenszwalb-Huttenlocher, two separable passes, numba.cuda, float64) --
                       identical to scipy.ndimage.distance_transform_edt up to float rounding.
  * `grad_gpu`       : teleop.geom.edt_grad (np.gradient, edge_order=1, unit-normalised) in torch.
  * `GpuOccGrid`     : holds occ / edt / gx / gy on the GPU; `.lookup_grad(x, y)` (the GM.grid_lookup_grad API) on the GPU, and `.to_host()` -> a
                       teleop.scene.OccGrid with the gradient cache pre-filled, so `GM.grid_lookup_grad(grid, ...)` in the shield works unchanged.
  * `GpuMapper`      : camera capture (GPU depth + noise) -> place with the ESTIMATED pose -> integrate; build_map_gpu mirrors loc.mapper.build_map.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from numba import cuda

BIG = 1e10


# ------------------------------------------------------------------------------------------------ integrate
KMAX = 1 << 20                                         # per-ray step slots in the sequence key (> any step count)
INF = 1e30


def _compose(prev, cur):
    """clamped-add monoid: element = (a, lo, hi) meaning f(x) = min(hi, max(lo, x + a)); returns cur o prev."""
    a1, lo1, hi1 = prev
    a2, lo2, hi2 = cur
    return a1 + a2, torch.clamp(lo1 + a2, min=lo2, max=hi2) if False else torch.minimum(torch.maximum(lo1 + a2, lo2), hi2), \
        torch.minimum(torch.maximum(hi1 + a2, lo2), hi2)


def _exact_cells(cell, val, seq, is_occ, clamp):
    """final clamped value composition per cell for updates sorted-by-(cell, seq).  -> (unique cells, a, lo, hi) of the whole sequence."""
    o1 = torch.argsort(seq, stable=True)
    o2 = torch.argsort(cell[o1], stable=True)
    o = o1[o2]
    cell, val, is_occ = cell[o], val[o], is_occ[o]
    a = val.double()
    lo = torch.where(is_occ, torch.full_like(a, -INF), torch.full_like(a, -clamp))
    hi = torch.where(is_occ, torch.full_like(a, clamp), torch.full_like(a, INF))
    n = cell.shape[0]
    s = 1
    while s < n:
        same = torch.zeros(n, dtype=torch.bool, device=cell.device)
        same[s:] = cell[s:] == cell[:-s]
        if not bool(same.any()):
            break
        pa, plo, phi = torch.roll(a, s), torch.roll(lo, s), torch.roll(hi, s)
        na, nlo, nhi = _compose((pa, plo, phi), (a, lo, hi))
        a, lo, hi = torch.where(same, na, a), torch.where(same, nlo, lo), torch.where(same, nhi, hi)
        s *= 2
    last = torch.ones(n, dtype=torch.bool, device=cell.device)
    last[:-1] = cell[1:] != cell[:-1]
    return cell[last], a[last], lo[last], hi[last]


def integrate_gpu(L, res, x0, y0, o, E, F, hit, zlo, zhi, l_occ, l_free, clamp, dz, chunk=4096, exact=True):
    """L (nz,ny,nx) float32 GPU, updated in place.  o (3,) origin; E (N,3) hit / no-return endpoints; F (N,3) free-trace ends; hit (N,) bool.
    exact=True : the per-cell update sequence of the numba loop (ray order, free steps then the hit) is reproduced for every cell that can
                 saturate (clamped-add composition scan); cells that cannot saturate use the plain sum -> equals loc.mapper._integrate up to
                 float32 rounding.   exact=False: free phase then occupied phase (cheaper; differs on saturating cells only)."""
    nz, ny, nx = L.shape
    Lf = L.view(-1)
    ox, oy, oz = (float(v) for v in o)
    dev = L.device
    cells, vals, seqs, occs = [], [], [], []
    N = E.shape[0]
    for c0 in range(0, N, chunk):
        Fc = F[c0:c0 + chunk]
        ddx, ddy, ddz = Fc[:, 0] - ox, Fc[:, 1] - oy, Fc[:, 2] - oz
        n = (torch.hypot(ddx, ddy) / (0.5 * res)).long() + 1
        nmax = int(n.max())
        k = torch.arange(nmax, device=dev, dtype=torch.float64)[None]
        nn = n[:, None].double()
        f = (k + 0.5) / nn
        z = oz + f * ddz[:, None]
        valid = (k < nn) & (z >= zlo) & (z <= zhi)
        ix = ((ox + f * ddx[:, None] - x0) / res).long()
        iy = ((oy + f * ddy[:, None] - y0) / res).long()
        valid &= (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
        iz = ((z - zlo) / dz).long().clamp(max=nz - 1)
        cell = (iz * ny + iy) * nx + ix
        same = torch.zeros_like(valid)
        same[:, 1:] = valid[:, 1:] & valid[:, :-1] & (cell[:, 1:] == cell[:, :-1])
        keep = valid & ~same
        ri = (torch.arange(Fc.shape[0], device=dev)[:, None] + c0).expand_as(cell)
        cells.append(cell[keep])
        seqs.append(ri[keep] * KMAX + k.long().expand_as(cell)[keep])
    cell_f = torch.cat(cells) if cells else torch.zeros(0, dtype=torch.long, device=dev)
    seq_f = torch.cat(seqs) if seqs else torch.zeros(0, dtype=torch.long, device=dev)
    ex, ey, ez = E[:, 0], E[:, 1], E[:, 2]
    ok = hit & (ez >= zlo) & (ez <= zhi)
    ix = ((ex - x0) / res).long()
    iy = ((ey - y0) / res).long()
    iz = ((ez - zlo) / dz).long().clamp(max=nz - 1)
    ok &= (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cell_o = ((iz * ny + iy) * nx + ix)[ok]
    seq_o = torch.arange(N, device=dev)[ok] * KMAX + (KMAX - 1)
    if not exact:
        Lf.index_add_(0, cell_f, torch.full(cell_f.shape, l_free, dtype=L.dtype, device=dev))
        Lf.clamp_(min=-clamp)
        Lf.index_add_(0, cell_o, torch.full(cell_o.shape, l_occ, dtype=L.dtype, device=dev))
        Lf.clamp_(max=clamp)
        return
    nfree = torch.zeros_like(Lf)
    nocc = torch.zeros_like(Lf)
    nfree.index_add_(0, cell_f, torch.full(cell_f.shape, l_free, dtype=L.dtype, device=dev))
    nocc.index_add_(0, cell_o, torch.full(cell_o.shape, l_occ, dtype=L.dtype, device=dev))
    risky = ((Lf + nfree < -clamp) | (Lf + nocc > clamp)) & ((nfree != 0) | (nocc != 0))
    # non-saturating cells: the sum is order independent
    plain = ~risky
    Lf.add_(torch.where(plain, nfree + nocc, torch.zeros_like(Lf)))
    # saturating cells: reproduce the sequence
    rf = risky[cell_f]
    ro = risky[cell_o]
    if bool(risky.any()):
        cell = torch.cat([cell_f[rf], cell_o[ro]])
        val = torch.cat([torch.full((int(rf.sum()),), l_free, device=dev), torch.full((int(ro.sum()),), l_occ, device=dev)])
        seq = torch.cat([seq_f[rf], seq_o[ro]])
        isocc = torch.cat([torch.zeros(int(rf.sum()), dtype=torch.bool, device=dev), torch.ones(int(ro.sum()), dtype=torch.bool, device=dev)])
        uc, a, lo, hi = _exact_cells(cell, val, seq, isocc, clamp)
        x = Lf[uc].double()
        Lf[uc] = torch.minimum(torch.maximum(x + a, lo), hi).float()


def occ_mask_gpu(L, occ_thresh=0.5, unknown="free", despeckle=True, close_iters=1):
    import torch.nn.functional as Fn
    occ = (L > occ_thresh).any(dim=0)
    if unknown == "occ":
        occ = occ | (L == 0.0).all(dim=0)
    if despeckle:
        x = occ[None, None].float()
        nb = Fn.conv2d(x, torch.ones(1, 1, 3, 3, device=L.device), padding=1)[0, 0] - occ.float()
        occ = occ & (nb > 0.5)
    for _ in range(close_iters):                     # scipy binary_closing: dilation then erosion, border value 0 for both
        x = occ[None, None].float()
        dil = Fn.max_pool2d(x, 3, 1, 1) > 0.5
        er = Fn.conv2d(dil.float(), torch.ones(1, 1, 3, 3, device=L.device), padding=1)[0, 0] > 8.5
        occ = er
    return occ


# ------------------------------------------------------------------------------------------------ exact EDT (numba.cuda)
@cuda.jit
def _edt_cols(f0, d, v, z, ny, nx):
    x = cuda.grid(1)
    if x >= nx:
        return
    k = 0
    v[x, 0] = 0
    z[x, 0] = -1e30
    z[x, 1] = 1e30
    for q in range(1, ny):
        fq = 0.0 if f0[q, x] == 0 else BIG
        while True:
            vk = v[x, k]
            fv = 0.0 if f0[vk, x] == 0 else BIG
            s = ((fq + q * q) - (fv + vk * vk)) / (2.0 * q - 2.0 * vk)
            if s <= z[x, k]:
                k -= 1
            else:
                break
        k += 1
        v[x, k] = q
        z[x, k] = s
        z[x, k + 1] = 1e30
    k = 0
    for q in range(ny):
        while z[x, k + 1] < q:
            k += 1
        vk = v[x, k]
        fv = 0.0 if f0[vk, x] == 0 else BIG
        d[q, x] = (q - vk) * (q - vk) + fv


@cuda.jit
def _edt_rows(f1, d, v, z, ny, nx):
    y = cuda.grid(1)
    if y >= ny:
        return
    k = 0
    v[y, 0] = 0
    z[y, 0] = -1e30
    z[y, 1] = 1e30
    for q in range(1, nx):
        fq = f1[y, q]
        while True:
            vk = v[y, k]
            s = ((fq + q * q) - (f1[y, vk] + vk * vk)) / (2.0 * q - 2.0 * vk)
            if s <= z[y, k]:
                k -= 1
            else:
                break
        k += 1
        v[y, k] = q
        z[y, k] = s
        z[y, k + 1] = 1e30
    k = 0
    for q in range(nx):
        while z[y, k + 1] < q:
            k += 1
        vk = v[y, k]
        d[y, q] = (q - vk) * (q - vk) + f1[y, vk]


def edt_gpu(occ, res):
    """occ (ny,nx) bool GPU tensor -> distance [m] of every cell centre to the nearest occupied cell centre (0 on occupied), float32 GPU.
    Same quantity as scipy.ndimage.distance_transform_edt(~occ) * res."""
    ny, nx = occ.shape
    dev = occ.device
    # pass 1 along y reads "occupied" as f==0 (encoded: input 0 where occupied, nonzero elsewhere)
    f0 = (~occ).to(torch.float64)
    d1 = torch.empty((ny, nx), dtype=torch.float64, device=dev)
    d2 = torch.empty((ny, nx), dtype=torch.float64, device=dev)
    v1 = torch.empty((nx, ny + 1), dtype=torch.int32, device=dev)
    z1 = torch.empty((nx, ny + 2), dtype=torch.float64, device=dev)
    v2 = torch.empty((ny, nx + 1), dtype=torch.int32, device=dev)
    z2 = torch.empty((ny, nx + 2), dtype=torch.float64, device=dev)
    tb = 64
    _edt_cols[(nx + tb - 1) // tb, tb](cuda.as_cuda_array(f0), cuda.as_cuda_array(d1), cuda.as_cuda_array(v1), cuda.as_cuda_array(z1), ny, nx)
    _edt_rows[(ny + tb - 1) // tb, tb](cuda.as_cuda_array(d1), cuda.as_cuda_array(d2), cuda.as_cuda_array(v2), cuda.as_cuda_array(z2), ny, nx)
    return (torch.sqrt(d2.clamp(max=BIG)) * res).float()


def grad_gpu(edt, res):
    """teleop.geom.edt_grad: unit-normalised np.gradient(edt, res) (central differences, first-order edges)."""
    gy, gx = torch.gradient(edt, spacing=res, dim=(0, 1), edge_order=1)
    n = torch.hypot(gx, gy) + 1e-9
    return gx / n, gy / n


class GpuOccGrid:
    def __init__(self, res, x0, y0, occ):
        self.res, self.x0, self.y0 = float(res), float(x0), float(y0)
        self.occ = occ
        self.edt = edt_gpu(occ, self.res)
        self.gx, self.gy = grad_gpu(self.edt, self.res)

    def lookup_grad(self, wx, wy):
        """GM.grid_lookup_grad API on the GPU (torch tensors in, tensors out): (edt, gx, gy), 0 outside the grid."""
        ny, nx = self.occ.shape
        ix = torch.floor((wx - self.x0) / self.res).long()
        iy = torch.floor((wy - self.y0) / self.res).long()
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ixc, iyc = ix.clamp(0, nx - 1), iy.clamp(0, ny - 1)
        z = torch.zeros_like(wx)
        return (torch.where(ok, self.edt[iyc, ixc].to(wx.dtype), z), torch.where(ok, self.gx[iyc, ixc].to(wx.dtype), z),
                torch.where(ok, self.gy[iyc, ixc].to(wx.dtype), z))

    def to_host(self):
        """-> teleop.scene.OccGrid (numpy) with the gradient cache filled: GM.grid_lookup_grad(grid, x, y) works unchanged."""
        from ..teleop.scene import OccGrid
        g = OccGrid(self.res, self.x0, self.y0, self.occ.cpu().numpy(), self.edt.double().cpu().numpy())
        g._grad = (self.gx.double().cpu().numpy(), self.gy.double().cpu().numpy())
        return g


# ------------------------------------------------------------------------------------------------ mapper
class GpuMapper:
    """GPU twin of loc.mapper.HeadDepth + PerceivedMap.  `core` = SenseCore holding a camera config named `cam` (spec = the mapper CamSpec)."""

    def __init__(self, core, cfg, x0, y0, ny, nx, cam="map", seed=0):
        from . import core as C
        self.core, self.cfg, self.camname = core, cfg, cam
        self.c = core._cam(cam)
        self.spec = self.c.cfg.spec
        dev = core.device
        self.dev = dev
        self.x0, self.y0 = float(x0), float(y0)
        nz = int(math.ceil((cfg.obs_zhi - cfg.obs_zlo) / cfg.dz))
        self.L = torch.zeros((nz, ny, nx), dtype=torch.float32, device=dev)
        self.xn = self.c.X.reshape(-1)
        self.yn = -self.c.Y.reshape(-1)                  # image down per unit z (HeadDepth.yn)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(np.random.SeedSequence([int(seed), 4242]).generate_state(1, np.uint64)[0] & 0x7FFFFFFFFFFFFFFF))
        self.n_frames = 0

    def axes(self, pose):
        c = self.cfg
        mx, my, mz = c.mount
        cy, sy = math.cos(pose[2]), math.sin(pose[2])
        o = np.array([pose[0] + cy * mx - sy * my, pose[1] + sy * mx + cy * my, mz])
        p = math.radians(c.pitch_deg)
        fwd = np.array([math.cos(p) * cy, math.cos(p) * sy, -math.sin(p)])
        left = np.array([-sy, cy, 0.0])
        up = np.array([math.sin(p) * cy, math.sin(p) * sy, math.cos(p)])
        return o, fwd, left, up

    def capture(self, true_pose):
        """render at the TRUE pose + noise -> (ok, zn, nofree) GPU tensors (flattened HxW) and the true axes."""
        o, fwd, left, up = self.axes(true_pose)
        R = np.stack([-left, up, -fwd], 1)               # MuJoCo camera: x right (= -left), y up, z backward
        z, geom = self.core.depth(self.camname, o, R)
        ok, zn, nofree = self.core.sensor.map_capture(self.c, z, geom, self.gen)
        return ok, zn, nofree

    def place_integrate(self, ok, zn, nofree, est_pose):
        cfg, s, dev = self.cfg, self.spec, self.dev
        o, fwd, left, up = (torch.as_tensor(a, dtype=torch.float64, device=dev) for a in self.axes(est_pose))
        xn, yn = self.xn, self.yn
        pl, pu = -xn * zn, -yn * zn
        P = o[None] + zn[:, None] * fwd[None] + pl[:, None] * left[None] + pu[:, None] * up[None]
        zmax = s.z_max * 0.9
        dirv = fwd[None] + xn[:, None] * (-left)[None] + yn[:, None] * (-up)[None]
        Pn = o[None] + zmax * dirv / torch.linalg.norm(dirv, dim=1)[:, None]
        E = torch.where(ok[:, None], P, Pn)
        rng_ = torch.linalg.norm(E - o[None], dim=1) + 1e-9
        from .core import sigma_z
        pull = torch.where(ok, cfg.trunc_k * sigma_z(s, zn.clamp(min=0.0)), torch.zeros_like(zn))
        F = o[None] + (E - o[None]) * (torch.clamp(rng_ - pull, min=0.0) / rng_)[:, None]
        sel = ok | nofree
        integrate_gpu(self.L, cfg.res, self.x0, self.y0, o.cpu().numpy(), E[sel], F[sel], ok[sel], cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ, cfg.l_free,
                      cfg.l_clamp, cfg.dz)
        self.n_frames += 1

    def update(self, true_pose, est_pose):
        ok, zn, nofree = self.capture(true_pose)
        self.place_integrate(ok, zn, nofree, est_pose)

    def occ_mask(self):
        c = self.cfg
        return occ_mask_gpu(self.L, c.occ_thresh, c.unknown, c.despeckle, c.close_iters)

    def grid(self):
        return GpuOccGrid(self.cfg.res, self.x0, self.y0, self.occ_mask())


def build_map_gpu(scene, cfg=None, loc_cfg=None, verbose=True, device="cuda", tmpdir=None):
    """GPU twin of loc.mapper.build_map: same exploration script / pose estimator; sensing + integration + (returned) EDT on the GPU.
    -> (PerceivedMap, GpuMapper, timings dict)."""
    import mujoco
    import tempfile
    from world_model.v2.house import build as B
    from world_model.v2.house.loc import mapper as MP
    from world_model.v2.house.loc.estimator import LocCfg, PoseEstimator, scaled
    from world_model.v2.house.teleop import scene as SC
    from world_model.v2.tray_rep.perception import depth_sim as D
    from . import core as C
    cfg = cfg or MP.MapCfg()
    t0 = time.time()
    spec = scene.spec_loader() if scene.spec_loader else B.load_house_spec(scene.house, True)
    m = spec.compile()
    mujoco.mj_forward(m, mujoco.MjData(m))
    path = C.snapshot_model_bytes(m, (tmpdir or tempfile.gettempdir()) + f"/gpumap_{scene.name}_{id(m)}.mjb")
    cs = D.CamSpec(name="zed_map", mount="world", pos=(0, 0, 0), R=tuple(np.eye(3).ravel()), fovy=cfg.fovy, W=cfg.W, H=cfg.H, z_max=cfg.z_map_max)
    cc = C.GpuCamCfg("map", cs, (3, 4), kind="map")
    core = C.SenseCore(path, [cc], device)
    t_setup = time.time() - t0
    t1 = time.time()
    core.warmup()
    t_warm = time.time() - t1
    true = SC.load_occupancy(scene)
    ny, nx = int(round(true.occ.shape[0] * true.res / cfg.res)), int(round(true.occ.shape[1] * true.res / cfg.res))
    gm = GpuMapper(core, cfg, true.x0, true.y0, ny, nx, "map", seed=cfg.seed)
    route = np.asarray(scene.out_route, float)
    poses = MP.exploration_path(route, cfg)
    est = PoseEstimator(scaled(loc_cfg or LocCfg(), cfg.map_loc_scale), seed=cfg.mapping_loc_seed, true_pose0=poses[0])
    every = int(round(1.0 / (cfg.frame_hz * 0.02)))
    nf = 0
    t2 = time.time()
    for k, p in enumerate(poses[1:], 1):
        est.update(k * 0.02, p, 0.02)
        if k % every:
            continue
        gm.update(p, est.pose)
        nf += 1
    torch.cuda.synchronize()
    t_loop = time.time() - t2
    grid = gm.grid()
    torch.cuda.synchronize()
    meta = dict(frames=nf, duration_s=len(poses) * 0.02, wall_s=time.time() - t0, map_ate=est.errors(), route=route.tolist(), setup_s=t_setup,
                warmup_s=t_warm, loop_s=t_loop)
    if verbose:
        print(f"[gpumapper] {nf} frames, setup {t_setup:.1f}s warm {t_warm:.1f}s loop {t_loop:.1f}s", flush=True)
    pm = MP.PerceivedMap(cfg.res, true.x0, true.y0, gm.L.cpu().numpy(), cfg.asdict(), meta)
    return pm, gm, grid, core, meta
