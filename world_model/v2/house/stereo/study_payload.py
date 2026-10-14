"""stereo/study_payload.py -- STAGE 2c: the head-stereo payload estimator (tray plane + cup) on SGBM depth vs ray-cast-model depth, vs truth.  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.study_payload --tags speckle_s0_640x360 ... [--noise fitted.json]

Same RGB (stereo left image) and same FK tray prior for both depth sources:  sgbm = the SGBM depth;  raycast = the MuJoCo GT depth of the left camera + the depth_sim
CamSpec noise model (stereo-law sigma_z with native_H = stereo height, dropout, grazing mask, outliers, 1 mm quantisation), i.e. what a ray-cast head camera of the
same geometry would deliver.  Truth: tray / cup poses of the snapshot (EVALUATION ONLY).
Per frame: tray tilt error [deg], tray in-plane position error [mm] (true tray frame), cup position in the TRAY frame error [mm], cup position in the BASE frame
error [mm] (independent of the tray in-plane error), cup tilt (rel. tray) error [deg], plus the set-coverage of the ObjectState sets (the producer's k-sigma + floors).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .cfg import StereoCfg, head_axes
from .twin import OUT
from . import payload as PL


def rc_depth(gt, cfg: StereoCfg, rng, sigma_disp_px=0.25, sigma_floor=5e-4, p_drop=0.02, p_out=0.002, cos_min=math.cos(math.radians(80)), quant=1e-3):
    """ray-cast depth + depth_sim noise model on the full stereo grid (gt: z-depth image, 0 = no hit)."""
    H, W = gt.shape
    K = cfg.K()
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x, y = (uu - K["cx"]) / K["f"], -(vv - K["cy"]) / K["f"]
    g = np.where(gt > 0, gt, np.nan).astype(np.float32)
    P = np.stack([x * g, y * g, -g], -1)
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    ray = np.stack([x, y, -np.ones_like(x)], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    cosinc = np.abs((n * ray).sum(-1)) / (np.linalg.norm(n, axis=-1) + 1e-12)
    sig = np.sqrt(sigma_floor ** 2 + (g * g / (cfg.f * cfg.baseline) * sigma_disp_px) ** 2)
    zn = g + sig * rng.normal(size=g.shape).astype(np.float32)
    out = rng.random(g.shape) < p_out
    zn = np.where(out, g + rng.uniform(0.05, 0.20, g.shape) * np.where(rng.random(g.shape) < 0.5, -1, 1), zn)
    zn = np.round(zn / quant) * quant
    ok = np.isfinite(g) & (np.nan_to_num(cosinc) >= cos_min) & (rng.random(g.shape) >= p_drop) & (zn >= cfg.z_min) & (zn <= cfg.z_max)
    return np.where(ok, zn, np.nan).astype(np.float32)


def truth(snap, i, z):
    x, y, yaw = snap["base"][i]
    c, s = math.cos(yaw), math.sin(yaw)
    Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Rt = Rb.T @ snap["tray_R"][i].reshape(3, 3)
    pt = Rb.T @ (snap["tray_p"][i] - np.array([x, y, 0]))
    cp = Rb.T @ (snap["cup_p"][i] - np.array([x, y, 0]))
    Rc = Rb.T @ snap["cup_R"][i].reshape(3, 3)
    return dict(Rt=Rt, pt=pt, cp=cp, rel=Rt.T @ (cp - pt), tray_tilt=math.degrees(math.acos(np.clip(Rt[2, 2], -1, 1))),
                cup_tilt=math.degrees(math.acos(np.clip(Rt[:, 2] @ Rc[:, 2], -1, 1))))


def evaluate(tags, noise_kw=None, pcfg=None, out_root=OUT, k=3.0):
    W, H = (int(v) for v in tags[0].split("_")[-1].split("x"))
    cfg = StereoCfg(W=W, H=H)
    pcfg = pcfg or PL.PayloadCfg()
    rng = np.random.default_rng(1)
    rows = {"sgbm": [], "raycast": []}
    import time
    ms = {"sgbm": [], "raycast": []}
    for tag in tags:
        seed = int(tag.split("_s")[1].split("_")[0])
        snap = np.load(Path(out_root) / "snap" / f"carry_s{seed}.npz", allow_pickle=True)
        for f in sorted((Path(out_root) / "cache" / tag).glob("fr_*.npz")):
            z = np.load(f)
            i = int(z["i"])
            T = truth(snap, i, z)
            deps = {"sgbm": np.where(z["depth_mm"] > 0, z["depth_mm"] / 1000.0, np.nan).astype(np.float32),
                    "raycast": rc_depth(z["gt_mm"].astype(np.float32) / 1000.0, cfg, rng, **(noise_kw or {}))}
            for name, dep in deps.items():
                fr = dict(depth=dep, left=z["left"])
                t0 = time.perf_counter()
                e = PL.estimate(fr, cfg, (z["tray_fk_p"], z["tray_fk_R"]), float(z["lift"]), pcfg)
                ms[name].append(1e3 * (time.perf_counter() - t0))
                r = dict(i=i, tag=tag, tray_ok=e["tray"] is not None, cup_ok=bool(e["cup"]["ok"]), cup_px=int(e["cup"]["n_px"]), true_tray_tilt=T["tray_tilt"], true_cup_tilt=T["cup_tilt"])
                if e["tray"] is not None:
                    tr = e["tray"]
                    perr = T["Rt"].T @ (tr["p"] - T["pt"])
                    r.update(tray_tilt_err=math.degrees(tr["tilt"]) - T["tray_tilt"], tray_dz_mm=1e3 * perr[2], tray_dx_mm=1e3 * perr[0], tray_dy_mm=1e3 * perr[1],
                             tray_hw_mm=[1e3 * w for w in tr["inplane_hw"]], tray_plane_sig_mm=1e3 * tr["sig"],
                             tray_tilt_sig=[math.sqrt(max(tr["tilt_cov"][j, j], 0)) for j in (0, 1)])
                    if e["cup"]["ok"]:
                        cu = e["cup"]
                        rel_e = cu["p"][:2] - T["rel"][:2]
                        # cup position in the BASE frame: tray estimate o cup estimate vs the true cup
                        cb = tr["p"] + tr["R"] @ cu["p"]
                        r.update(cup_ex_mm=1e3 * rel_e[0], cup_ey_mm=1e3 * rel_e[1], cup_base_err_mm=1e3 * float(np.linalg.norm((cb - T["cp"])[:2])),
                                 cup_base_ez_mm=1e3 * float((cb - T["cp"])[2]), cup_tilt_err=math.degrees(cu["tilt"]) - T["cup_tilt"], cup_tilt_est=math.degrees(cu["tilt"]),
                                 cup_sxy_mm=1e3 * math.sqrt(max(cu["cov"][0, 0], cu["cov"][1, 1], 0)), cup_stilt=math.sqrt(max(cu["cov"][2, 2], cu["cov"][3, 3], 0)))
                rows[name].append(r)
    return rows, ms


def summarize(rows, pcfg: PL.PayloadCfg, k=3.0):
    def st(key, absval=True):
        v = np.array([r[key] for r in rows if key in r], float)
        if not len(v):
            return None
        a = np.abs(v) if absval else v
        return dict(n=int(len(v)), median=float(np.median(a)), p95=float(np.percentile(a, 95)), rmse=float(np.sqrt((v ** 2).mean())), bias=float(v.mean()))
    out = dict(n=len(rows), tray_found=float(np.mean([r["tray_ok"] for r in rows])), cup_found=float(np.mean([r["cup_ok"] for r in rows])),
               tray_tilt_err_deg=st("tray_tilt_err"), tray_dz_mm=st("tray_dz_mm"), tray_dx_mm=st("tray_dx_mm"), tray_dy_mm=st("tray_dy_mm"),
               cup_ex_mm=st("cup_ex_mm"), cup_ey_mm=st("cup_ey_mm"), cup_base_err_mm=st("cup_base_err_mm"), cup_tilt_err_deg=st("cup_tilt_err"))
    # set coverage with the producer's sets: cup xy box r = k*sxy + floor; tilt = k*stilt + floor; tray tilt = k*max(sx,sy) + floor
    cov = {}
    cr = [r for r in rows if "cup_ex_mm" in r]
    if cr:
        cov["cup_pos_box_no_tray_hw"] = float(np.mean([max(abs(r["cup_ex_mm"]), abs(r["cup_ey_mm"])) <= 1e3 * (k * 1e-3 * r["cup_sxy_mm"] + pcfg.sigma_floor_pos) for r in cr]))
        cov["cup_tilt"] = float(np.mean([abs(math.radians(r["cup_tilt_err"])) <= k * r["cup_stilt"] + pcfg.sigma_floor_rot for r in cr]))
        cov["cup_pos_box"] = float(np.mean([max(abs(r["cup_ex_mm"]), abs(r["cup_ey_mm"])) <= 1e3 * (k * 1e-3 * r["cup_sxy_mm"] + pcfg.sigma_floor_pos) + max(r["tray_hw_mm"]) + 1e3 * pcfg.hw_margin for r in cr]))
        cov["cup_pos_box_median_mm"] = float(np.median([1e3 * (k * 1e-3 * r["cup_sxy_mm"] + pcfg.sigma_floor_pos) + max(r["tray_hw_mm"]) + 1e3 * pcfg.hw_margin for r in cr]))
    tr = [r for r in rows if "tray_tilt_err" in r]
    if tr:
        cov["tray_tilt"] = float(np.mean([abs(math.radians(r["tray_tilt_err"])) <= k * max(r["tray_tilt_sig"]) + pcfg.sigma_floor_rot for r in tr]))
        cov["tray_inplane_hw"] = float(np.mean([abs(r["tray_dx_mm"]) <= max(r["tray_hw_mm"]) + 1e3 * pcfg.hw_margin and abs(r["tray_dy_mm"]) <= max(r["tray_hw_mm"]) + 1e3 * pcfg.hw_margin for r in tr]))
    out["coverage"] = cov
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--name", default="payload")
    a = ap.parse_args()
    nk = json.loads(Path(a.noise).read_text()) if a.noise else None
    pcfg = PL.PayloadCfg()
    rows, ms = evaluate(a.tags, nk, pcfg)
    res = {name: summarize(r, pcfg) for name, r in rows.items()}
    for name in res:
        res[name]["ms_per_estimate"] = float(np.mean(ms[name]))
    res["tags"] = a.tags
    (Path(OUT) / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    (Path(OUT) / f"{a.name}_rows.json").write_text(json.dumps(rows, default=float))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
