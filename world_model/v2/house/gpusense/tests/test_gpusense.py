"""gpusense unit tests (GPU, no sim/scene needed): exact EDT vs scipy, torch log-odds integrator vs the numba loop, host-grid API."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from scipy import ndimage

from world_model.v2.house.gpusense import mapping as GMAP
from world_model.v2.house.loc import mapper as MP
from world_model.v2.house.teleop import geom as GM
from world_model.v2.house.teleop.scene import OccGrid


def test_edt_exact_and_grad():
    rng = np.random.default_rng(0)
    for shape, p in (((97, 131), 0.02), ((64, 64), 0.002), ((50, 80), 0.3)):
        occ = rng.random(shape) < p
        occ[3, 4] = True
        ref = ndimage.distance_transform_edt(~occ) * 0.02
        e = GMAP.edt_gpu(torch.as_tensor(occ, device="cuda"), 0.02).cpu().numpy()
        assert np.abs(e - ref).max() < 1e-5
        g = OccGrid(0.02, 0.0, 0.0, occ, ref)
        gx, gy = GMAP.grad_gpu(torch.as_tensor(ref, device="cuda", dtype=torch.float32), 0.02)
        rx, ry = GM.edt_grad(g)
        assert np.abs(gx.cpu().numpy() - rx).max() < 1e-3 and np.abs(gy.cpu().numpy() - ry).max() < 1e-3
        go = GMAP.GpuOccGrid(0.02, 1.0, -2.0, torch.as_tensor(occ, device="cuda"))
        h = go.to_host()
        qx, qy = rng.uniform(0.5, 4.0, 200), rng.uniform(-2.5, 0.0, 200)
        a = GM.grid_lookup_grad(h, qx, qy)
        t = go.lookup_grad(torch.as_tensor(qx, device="cuda"), torch.as_tensor(qy, device="cuda"))
        for x, y in zip(a, t):
            assert np.abs(x - y.cpu().numpy()).max() < 1e-4


def test_integrator_matches_numba():
    cfg = MP.MapCfg()
    rng = np.random.default_rng(1)
    nz, ny, nx = 20, 120, 140
    L0 = np.zeros((nz, ny, nx), np.float32)
    Lg = torch.zeros((nz, ny, nx), device="cuda")
    x0, y0 = -1.0, -1.0
    frames = []
    for frame in range(8):
        o = np.array([rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5), 1.1])
        n = 600
        ang = rng.uniform(0, 2 * np.pi, n)
        rg = rng.uniform(0.3, 1.2, n)
        E = np.stack([o[0] + rg * np.cos(ang), o[1] + rg * np.sin(ang), rng.uniform(0.0, 1.3, n)], 1)
        hit = rng.random(n) < 0.7
        F = o + (E - o) * 0.95
        frames.append((o, E, F, hit))
    for o, E, F, hit in frames * 6:             # repeated -> cells saturate, so the update ORDER matters
        MP._integrate(L0, cfg.res, x0, y0, o[0], o[1], o[2], E[:, 0].copy(), E[:, 1].copy(), E[:, 2].copy(), hit, cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ,
                      cfg.l_free, cfg.l_clamp, F[:, 0].copy(), F[:, 1].copy(), F[:, 2].copy(), cfg.dz)
        T = lambda a: torch.as_tensor(a, device="cuda")
        GMAP.integrate_gpu(Lg, cfg.res, x0, y0, o, T(E), T(F), T(hit), cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ, cfg.l_free, cfg.l_clamp, cfg.dz)
    # saturate cells: many frames over the same rays
    assert np.abs(Lg.cpu().numpy() - L0).max() < 1e-4
