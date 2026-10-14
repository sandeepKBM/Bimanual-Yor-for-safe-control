"""lang/prior.py -- Phys2Real-style VLM prior on the cup-tray friction mu -> the shield's mu set.  SIMULATION / SAVED DATA ONLY.

Protocol (arXiv 2510.11689, local open VLM): N images x M repeated queries (sampling T>0, one request per image with n=M), every sample = (estimate, self-reported
sigma).  Fusion:
    center       = inverse-variance weighted mean of the estimates (w_i = 1 / sigma_i^2)
    sigma_indep  = (sum w_i)^-1/2                      what independent samples would give (reported, NOT used: samples of one model share their errors)
    sigma        = sqrt( mean(sigma_i^2) + var(estimates) )   law of total variance; does not shrink with N
    interval     = [center - k sigma, center + k sigma]  intersected with the physically sane range SANE_MU
The shield only uses mu_lo (ShieldCfg.ratio = min(kappa mu_lo, tan theta_spill)).  No parsable sample -> the hard-coded default set, source='default'.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.vlm.spec_eval import extract_json

KEY = "cup_tray_friction"
SANE_MU = (0.08, 0.90)
DEFAULT_MU = (0.18, 0.35)         # the hard-coded ShieldCfg set
SIM_TRUTH = 0.25                  # teleop.scene.MU_TRUE (EVALUATION only: never shown to the VLM)

SYSTEM = ("You estimate physical parameters of objects in simulated robot scenes from images. Give your best ESTIMATE and a self-reported uncertainty SIGMA: one standard "
          "deviation in the SAME units as the estimate (the true value should lie within estimate +/- sigma about 68% of the time). Be honest about uncertainty. "
          "Output ONLY a JSON object of the form {\"" + KEY + "\": {\"estimate\": number, \"sigma\": number}}.")
USER = ("The robot carries a cup (with beads in it) standing on a flat tray. Look at the image and estimate \"" + KEY + "\" (dimensionless): the Coulomb sliding "
        "friction coefficient between the base of the cup and the tray surface. Typical values: about 0.1-0.2 for a wet or glossy plastic surface, 0.3-0.5 for dry "
        "paper / plastic, 0.6-0.9 for rubber. Both estimate and sigma (> 0) are required. Answer in exactly this JSON format:\n"
        "{\"" + KEY + "\": {\"estimate\": <number>, \"sigma\": <number>}}")


@dataclass
class MuInterval:
    lo: float
    hi: float
    center: float | None = None
    sigma: float | None = None
    k: float = 1.0
    source: str = "default"          # 'vlm' | 'default'
    n_images: int = 0
    n_queries: int = 0
    n_parsed: int = 0
    samples: list = field(default_factory=list)
    sigma_indep: float | None = None
    sigma_vlm_mean: float | None = None
    std_est: float | None = None
    seconds: float = 0.0
    wall_s: float = 0.0
    reasons: list = field(default_factory=list)

    def covers(self, mu=SIM_TRUTH):
        return bool(self.lo <= mu <= self.hi)


def parse_sample(text):
    j = extract_json(text)
    if not isinstance(j, dict):
        return None
    v = j.get(KEY)
    try:
        e, s = float(v["estimate"]), float(v["sigma"])
    except Exception:       # noqa: BLE001
        return None
    return (e, s) if np.isfinite(e) and np.isfinite(s) and e > 0 and s > 0 else None


def fuse_samples(samples, k=1.0, sane=SANE_MU):
    """samples [(estimate, sigma)] -> dict(center, sigma, sigma_indep, sigma_vlm_mean, std_est, lo, hi)."""
    e = np.array([s[0] for s in samples])
    sg = np.array([s[1] for s in samples])
    w = 1.0 / sg ** 2
    center = float((w * e).sum() / w.sum())
    sigma_indep = float(w.sum() ** -0.5)
    var_between = float(e.var(ddof=1)) if len(e) > 1 else 0.0
    sigma = float(np.sqrt((sg ** 2).mean() + var_between))
    lo = max(sane[0], center - k * sigma)
    hi = min(sane[1], center + k * sigma)
    if hi <= lo:                     # center outside the sane range: pin to the nearest sane edge with a minimal width
        lo, hi = (sane[0], sane[0] + 0.05) if center < sane[0] else (sane[1] - 0.05, sane[1])
    return dict(center=center, sigma=sigma, sigma_indep=sigma_indep, sigma_vlm_mean=float(sg.mean()), std_est=(float(e.std(ddof=1)) if len(e) > 1 else None), lo=float(lo), hi=float(hi))


def estimate_mu(backend, images, n_queries=4, k=1.0, temperature=0.7, max_new_tokens=120, sane=SANE_MU):
    """N images x M queries -> MuInterval (falls back to DEFAULT_MU when nothing parses)."""
    samples, sec, wall = [], 0.0, 0.0
    for im in images:
        o = backend.ask(USER, images=[im], system=SYSTEM, max_new_tokens=max_new_tokens, temperature=temperature, n=n_queries)
        sec += float(o.get("seconds") or 0.0)
        wall += float(o.get("wall_s") or 0.0)
        for t in o["texts"]:
            s = parse_sample(t)
            if s is not None:
                samples.append(s)
    n_tot = len(images) * n_queries
    if not samples:
        return MuInterval(DEFAULT_MU[0], DEFAULT_MU[1], None, None, k, "default", len(images), n_tot, 0, [], seconds=sec, wall_s=wall,
                          reasons=[f"no parsable sample of {n_tot}: hard-coded set {list(DEFAULT_MU)}"])
    f = fuse_samples(samples, k, sane)
    return MuInterval(f["lo"], f["hi"], f["center"], f["sigma"], k, "vlm", len(images), n_tot, len(samples), [list(s) for s in samples], f["sigma_indep"], f["sigma_vlm_mean"],
                      f["std_est"], sec, wall, [f"{len(samples)}/{n_tot} samples parsed; center {f['center']:.3f} sigma {f['sigma']:.3f} (indep {f['sigma_indep']:.3f})"])
