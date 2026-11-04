"""pan/theta.py -- the pot/pan hard-world parameter box THETA and world_from_theta, used by contents/envelope.py.
Moved VERBATIM (values and code unchanged, 2026-11-03) from pan/hardworlds.py (now archived).  Every bound is labelled in THETA.
SIMULATION ONLY."""
from __future__ import annotations

import numpy as np

# ---- Theta: (lo, hi, label).  Raised to the user 11-01 (demos_2026-11-01.md sec. 3); USER said "get that running" 11-01.
THETA = {
    "dx": (-0.015, 0.015, "ASSUMED (prior box, unchanged)"),
    "dy": (-0.015, 0.015, "ASSUMED (prior box, unchanged)"),
    "yaw_deg": (-8.0, 8.0, "ASSUMED (prior box, unchanged)"),
    "mu_scale": (0.4, 1.2, "ASSUMED (widened from 0.6-1.2: wet/greasy)"),
    "mass": (0.02, 1.5, "ASSUMED (widened from 0-0.4 kg; 0.02 = an almost empty pot)"),
    "c_r": (0.0, 1.0, "MODEL: contents centre radius as a fraction of C_R (uniform over the disc via sqrt)"),
    "c_phi": (0.0, 1.0, "MODEL: contents centre angle / 2 pi"),
    "lid_mu_scale": (0.5, 1.0, "ASSUMED (wet lid)"),
    "handle_len": (0.09, 0.13, "ASSUMED (stove-clearance tier; 0.09 refused 6/9 on 11-01)"),
}
C_R = 0.06            # m, MODEL: pot inner radius 0.09 - puck radius 0.025 - 5 mm


def world_from_theta(th):
    r, phi = C_R * np.sqrt(th["c_r"]), 2 * np.pi * th["c_phi"]
    return dict(placement=[float(r * np.cos(phi)), float(r * np.sin(phi))], hidden_mass=th["mass"], handle_len=th["handle_len"],
                perturb=dict(dx=th["dx"], dy=th["dy"], yaw_deg=th["yaw_deg"], mu_scale=th["mu_scale"], lid_mu_scale=th["lid_mu_scale"]))
