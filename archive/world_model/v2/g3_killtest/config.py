"""config.py -- class configs (priors, particle grids, noise levels, paths). Fixed by PREREG."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from world_model.v2.g3_killtest import core as C

OUT = Path("/common/users/ss5772/bimanual_data/g3_killtest")
VAL_DIR = OUT / "values"

N_GRID = 201           # base state-grid points per dim (validation: 301)
HJ_ITERS = 500         # 5 s viability horizon at dt = 0.01
HJ_TOL = 1e-6

CLASSES = {
    "T": dict(
        cid=C.TIP, name="tipping block (COM height h unknown, width 5 cm known)",
        theta_names=("h_com_m",), prior=((0.04, 0.12),), grid=(201,),
        sigmas=(np.deg2rad(1.0), np.deg2rad(2.0), np.deg2rad(4.0)), sigma_labels=("1deg", "2deg", "4deg"),
        primary_sigma=1),
    "S": dict(
        cid=C.SLIDE, name="sliding item (Coulomb mu unknown)",
        theta_names=("mu",), prior=((0.15, 0.60),), grid=(201,),
        sigmas=(0.002, 0.005, 0.010), sigma_labels=("2mm", "5mm", "10mm"), primary_sigma=1),
    "P": dict(
        cid=C.PEND, name="pendulum-like hanging payload (omega, zeta unknown)",
        theta_names=("omega_rad_s", "zeta"), prior=((2 * np.pi * 0.6, 2 * np.pi * 1.6), (0.01, 0.15)),
        grid=(81, 11),
        sigmas=(np.deg2rad(1.0), np.deg2rad(2.0), np.deg2rad(4.0)), sigma_labels=("1deg", "2deg", "4deg"),
        primary_sigma=1),
}


def particle_grid(cls):
    c = CLASSES[cls]
    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(c["prior"], c["grid"])]
    if len(axes) == 1:
        TH = np.stack([axes[0], np.zeros_like(axes[0])], 1)
    else:
        A, B = np.meshgrid(axes[0], axes[1], indexing="ij")
        TH = np.stack([A.ravel(), B.ravel()], 1)
    return TH


def sample_theta(cls, rng):
    c = CLASSES[cls]
    th = [rng.uniform(lo, hi) for lo, hi in c["prior"]]
    return np.array([th[0], th[1] if len(th) > 1 else 0.0])


def value_path(cls, n=N_GRID):
    return VAL_DIR / f"V_{cls}_n{n}.npz"
