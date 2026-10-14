"""stereo/study_calib.py -- STAGE 2a: calibration of the stereo depth (SGBM) vs the MuJoCo GT and vs the ray-cast noise model (depth_sim.CamSpec).  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.study_calib sample --tags speckle_s1_640x360 ...     # pooled per-pixel sample table
    python -m world_model.v2.house.stereo.study_calib report --tags ... [--tex-compare speckle_s1_640x360 objects_s1_640x360 none_s1_640x360]

Per range bin (on the GT z-depth): fill (valid fraction), signed bias, robust sigma (1.4826 MAD), inlier std, p95 |err|, gross-outlier rate.  Fits the CamSpec
parameters (sigma_disp_px, sigma_floor, p_drop, p_out, cos_min) to the SGBM result and lists where the independent-noise model cannot represent SGBM
(depth-edge band, left-border strip, grazing, low-texture, biases that are spatially correlated).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2

from .cfg import StereoCfg
from .twin import OUT

BINS = (0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 5.0)
OUTLIER_M = 0.10


def frame_table(z, gt, cls, pers, left, cfg, nsamp, rng):
    """per-pixel attributes for a random sample of GT-valid pixels of one frame."""
    H, W = gt.shape
    K = cfg.K()
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x, y = (uu - K["cx"]) / K["f"], -(vv - K["cy"]) / K["f"]
    gtc = np.where(gt > 0, gt, np.nan).astype(np.float32)
    P = np.stack([x * gtc, y * gtc, -gtc], -1)
    dx = np.gradient(P, axis=1)
    dy = np.gradient(P, axis=0)
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1) + 1e-12
    ray = np.stack([x, y, -np.ones_like(x)], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    cosinc = np.abs((n * ray).sum(-1)) / nn
    g = np.nan_to_num(gtc, nan=0.0)
    k = np.ones((9, 9), np.uint8)
    rel = (cv2.dilate(g, k) - cv2.erode(np.where(g > 0, g, 99.0).astype(np.float32), k)) / np.maximum(g, 0.05)
    edge = (rel > 0.05) & ((cv2.dilate(g, k) - cv2.erode(np.where(g > 0, g, 99.0).astype(np.float32), k)) > 0.05)
    gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.boxFilter(gray, -1, (9, 9))
    tex = np.sqrt(np.maximum(cv2.boxFilter(gray * gray, -1, (9, 9)) - mu * mu, 0))
    ok = (g > cfg.z_min) & (g < cfg.z_max)
    vi, ui = np.nonzero(ok)
    if len(vi) > nsamp:
        s = rng.choice(len(vi), nsamp, replace=False)
        vi, ui = vi[s], ui[s]
    zz = z[vi, ui]
    return dict(gt=g[vi, ui], z=np.where(zz > 0, zz, np.nan), cos=cosinc[vi, ui], edge=edge[vi, ui], tex=tex[vi, ui], col=ui.astype(np.int16), row=vi.astype(np.int16),
                cls=cls[vi, ui], pers=(pers[vi, ui] != 255))


def sample(tags, out_root=OUT, nsamp=3000, seed=0):
    rng = np.random.default_rng(seed)
    for tag in tags:
        W, H = (int(v) for v in tag.split("_")[-1].split("x"))
        cfg = StereoCfg(W=W, H=H)
        cols = {}
        for f in sorted((Path(out_root) / "cache" / tag).glob("fr_*.npz")):
            z = np.load(f)
            t = frame_table(z["depth_mm"].astype(np.float32) / 1000.0, z["gt_mm"].astype(np.float32) / 1000.0, z["cls"], z["person_g"], z["left"], cfg, nsamp, rng)
            for k, v in t.items():
                cols.setdefault(k, []).append(v)
        T = {k: np.concatenate(v) for k, v in cols.items()}
        np.savez_compressed(Path(out_root) / "cache" / f"calibsamples_{tag}.npz", **T)
        print(f"[calib] {tag}: {len(T['gt'])} samples", flush=True)


def _bin_stats(T, mask=None):
    gt, z = T["gt"], T["z"]
    m = np.ones(len(gt), bool) if mask is None else mask
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = m & (gt >= lo) & (gt < hi)
        if s.sum() < 200:
            continue
        v = s & np.isfinite(z)
        e = (z - gt)[v]
        if len(e) < 50:
            rows.append(dict(lo=lo, hi=hi, n=int(s.sum()), fill=float(v.sum() / s.sum())))
            continue
        med = np.median(e)
        mad = 1.4826 * np.median(np.abs(e - med))
        inl = np.abs(e - med) < max(5 * mad, 0.01)
        rows.append(dict(lo=lo, hi=hi, n=int(s.sum()), fill=float(v.sum() / s.sum()), bias=float(med), sigma_mad=float(mad), std_inl=float(e[inl].std()),
                         p95=float(np.percentile(np.abs(e), 95)), outlier=float((np.abs(e) > max(OUTLIER_M, 0.05 * gt[v].mean())).mean())))
    return rows


def fit_noise(rows, cfg: StereoCfg):
    """sigma_z^2 = floor^2 + (z^2 / (f B) sigma_d)^2 fitted to the per-bin robust sigma (weighted by sqrt(n)); z = bin centre."""
    r = [x for x in rows if "sigma_mad" in x]
    z = np.array([0.5 * (x["lo"] + x["hi"]) for x in r])
    s = np.array([x["std_inl"] for x in r])
    w = np.sqrt([x["n"] for x in r])
    A = np.stack([np.ones_like(z), (z ** 2 / (cfg.f * cfg.baseline)) ** 2], 1)
    from scipy.optimize import nnls
    coef, _ = nnls(A * w[:, None], s ** 2 * w)
    return dict(sigma_floor=float(np.sqrt(coef[0])), sigma_disp_px=float(np.sqrt(coef[1])), f=cfg.f, native_H=cfg.H)


def report(tags, tex_compare=(), out_root=OUT, name="calib"):
    pooled = {}
    for tag in tags:
        Z = np.load(Path(out_root) / "cache" / f"calibsamples_{tag}.npz")
        for k in Z.files:
            pooled.setdefault(k, []).append(Z[k])
    T = {k: np.concatenate(v) for k, v in pooled.items()}
    W, H = (int(v) for v in tags[0].split("_")[-1].split("x"))
    cfg = StereoCfg(W=W, H=H)
    res = dict(tags=list(tags), n=int(len(T["gt"])), res=[W, H], f=cfg.f, n_disp=cfg.n_disp())
    scene = (T["cls"] == 0) & ~T["pers"]
    easy = scene & ~T["edge"] & (T["cos"] > 0.5) & (T["tex"] > 3.0)
    res["bins_scene"] = _bin_stats(T, scene)
    res["bins_easy"] = _bin_stats(T, easy)
    res["bins_robot"] = _bin_stats(T, T["cls"] > 0)
    res["bins_person"] = _bin_stats(T, T["pers"])
    res["fit_scene"] = fit_noise(res["bins_easy"], cfg)
    fill_easy = float(np.isfinite(T["z"][easy & (T["gt"] > 0.5) & (T["gt"] < 3.0)]).mean())
    ze = (T["z"] - T["gt"])[easy & np.isfinite(T["z"]) & (T["gt"] > 0.5) & (T["gt"] < 3.0)]
    res["p_drop_easy"] = 1.0 - fill_easy
    res["p_out_easy"] = float((np.abs(ze) > OUTLIER_M).mean())
    # failure modes the independent-noise model does not have
    sel = scene & (T["gt"] > 0.5) & (T["gt"] < 3.0)
    def fe(m):
        m = sel & m
        v = m & np.isfinite(T["z"])
        e = (T["z"] - T["gt"])[v]
        return dict(n=int(m.sum()), fill=float(v.sum() / max(m.sum(), 1)), p95=float(np.percentile(np.abs(e), 95)) if len(e) > 20 else None,
                    outlier=float((np.abs(e) > OUTLIER_M).mean()) if len(e) > 20 else None)
    res["modes"] = dict(
        interior=fe(~T["edge"] & (T["cos"] > 0.5) & (T["tex"] > 3.0)), edge_band=fe(T["edge"]),
        grazing_cos_lt_0p3=fe((T["cos"] < 0.3) & ~T["edge"]), cos_0p3_0p5=fe((T["cos"] >= 0.3) & (T["cos"] < 0.5) & ~T["edge"]),
        low_texture_lt3=fe((T["tex"] <= 3.0) & ~T["edge"]), left_border_strip=fe(T["col"] < cfg.n_disp()), right_of_strip=fe(T["col"] >= cfg.n_disp()))
    cb = []
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.01)):
        m = sel & (T["cos"] >= lo) & (T["cos"] < hi) & ~T["edge"]
        cb.append(dict(cos_lo=lo, cos_hi=hi, n=int(m.sum()), fill=float(np.isfinite(T["z"][m]).mean()) if m.sum() else None))
    res["fill_vs_cos"] = cb
    tb = []
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)):
        m = sel & (T["tex"] >= lo) & (T["tex"] < hi) & ~T["edge"]
        v = m & np.isfinite(T["z"])
        e = (T["z"] - T["gt"])[v]
        tb.append(dict(tex_lo=lo, tex_hi=hi, n=int(m.sum()), fill=float(v.sum() / max(m.sum(), 1)), p95=float(np.percentile(np.abs(e), 95)) if len(e) > 20 else None))
    res["fill_vs_texture"] = tb
    # texture-mode comparison on the same frames (index-matched subset): overall fill / error by mode
    if tex_compare:
        comp = {}
        for tag in tex_compare:
            Z = np.load(Path(out_root) / "cache" / f"calibsamples_{tag}.npz")
            Tm = {k: Z[k] for k in Z.files}
            sc = (Tm["cls"] == 0) & ~Tm["pers"] & (Tm["gt"] > 0.5) & (Tm["gt"] < 4.0)
            v = sc & np.isfinite(Tm["z"])
            e = (Tm["z"] - Tm["gt"])[v]
            pe = Tm["pers"] & (Tm["gt"] > 0.5) & (Tm["gt"] < 4.0)
            pv = pe & np.isfinite(Tm["z"])
            ep = (Tm["z"] - Tm["gt"])[pv]
            rb = Tm["cls"] > 0
            comp[tag] = dict(scene_fill=float(v.sum() / max(sc.sum(), 1)), scene_p95=float(np.percentile(np.abs(e), 95)) if len(e) > 20 else None,
                             scene_outlier=float((np.abs(e) > OUTLIER_M).mean()) if len(e) > 20 else None,
                             person_fill=float(pv.sum() / max(pe.sum(), 1)) if pe.sum() else None,
                             person_p95=float(np.percentile(np.abs(ep), 95)) if len(ep) > 20 else None,
                             robot_fill=float(np.isfinite(Tm["z"][rb]).mean()) if rb.any() else None, bins=_bin_stats(Tm, sc))
        res["tex_compare"] = comp
    p = Path(out_root) / f"{name}.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"[calib] report -> {p}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sample", "report"])
    ap.add_argument("--tags", nargs="+")
    ap.add_argument("--tex-compare", nargs="*", default=[])
    ap.add_argument("--name", default="calib")
    a = ap.parse_args()
    if a.what == "sample":
        sample(a.tags)
    else:
        report(a.tags, a.tex_compare, name=a.name)


if __name__ == "__main__":
    main()
