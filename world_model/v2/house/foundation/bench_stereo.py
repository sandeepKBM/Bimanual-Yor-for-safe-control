"""bench_stereo.py -- FoundationStereo vs CPU SGBM on rendered head-ZED pairs (capture.py stereo).  mujoco env; the model runs in the server process.

    python -m world_model.v2.house.foundation.bench_stereo --gpu 5 --engine trt --res 640x360 --iters 12 --tag trt640
Writes <OUT>/bench/stereo_<tag>.json.  Depth error is vs the MuJoCo depth buffer (z-depth of the left camera); range bins by GT depth.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("FOUNDATION_OUT", "/data/local/ss5772/molmospaces/percept/foundation"))
BINS = [(0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]


def gpu_mem_mb(pid):
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        for ln in o.strip().splitlines():
            a, b = (x.strip() for x in ln.split(","))
            if int(a) == int(pid):
                return int(b)
    except Exception:
        pass
    return None


def stats(est, gt, wall, mask_extra=None):
    """est/gt (H,W) float, NaN invalid; returns per-bin dict for all / wall / nonwall pixels."""
    out = {}
    gv = np.isfinite(gt) & (gt >= 0.3) & (gt <= 5.0)
    if mask_extra is not None:
        gv &= mask_extra
    ev = np.isfinite(est)
    for name, sel in (("all", np.ones_like(gv)), ("wall", wall), ("nonwall", ~wall)):
        for lo, hi in BINS + [(0.3, 5.0)]:
            m = gv & sel & (gt >= lo) & (gt < hi if hi < 5.0 else gt <= hi)
            n = int(m.sum())
            if n == 0:
                continue
            ok = m & ev
            err = np.abs(est[ok] - gt[ok])
            rel = err / gt[ok]
            out[f"{name}[{lo},{hi})"] = dict(n=n, invalid=float(1 - ok.sum() / n), mae=float(err.mean()) if ok.any() else None, med=float(np.median(err)) if ok.any() else None,
                                           p90=float(np.percentile(err, 90)) if ok.any() else None, bad5=float((rel > 0.05).mean()) if ok.any() else None,
                                           rmse=float(np.sqrt((err ** 2).mean())) if ok.any() else None)
    return out


def merge(lst):
    """pool per-frame stat dicts by n-weighted mean (invalid, mae, bad5) -- median-of-frames for med/p90"""
    keys = set().union(*[set(d) for d in lst])
    res = {}
    for k in sorted(keys):
        rows = [d[k] for d in lst if k in d]
        n = np.array([r["n"] for r in rows], float)
        w = n / n.sum()
        g = lambda f: float(np.nansum([w[i] * (rows[i][f] if rows[i][f] is not None else np.nan) for i in range(len(rows))]))
        res[k] = dict(n=int(n.sum()), invalid=g("invalid"), mae=g("mae"), med=float(np.median([r["med"] for r in rows if r["med"] is not None])),
                      p90=float(np.median([r["p90"] for r in rows if r["p90"] is not None])), bad5=g("bad5"), rmse=g("rmse"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--engine", default="trt")
    ap.add_argument("--res", default="640x360")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--ckpt", default="23-51-11")
    ap.add_argument("--vit", default="vitl")
    ap.add_argument("--max-disp", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tex", nargs="*", default=["speckle", "objects", "none"])
    ap.add_argument("--timing-reps", type=int, default=40)
    ap.add_argument("--no-sgbm", action="store_true")
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.split("x"))
    from world_model.v2.house.foundation.stereo_fs import Client
    from world_model.v2.house.stereo.cfg import StereoCfg
    from world_model.v2.house.stereo.backends import SGBMBackend
    t0 = time.time()
    cl = Client(W=W, H=H, gpu=a.gpu, engine=a.engine, iters=a.iters, ckpt=a.ckpt, vit=a.vit, max_disp=a.max_disp)
    warm = cl.warmup()
    t_ready = time.time() - t0
    print(f"READY after {t_ready:.1f}s wall: {json.dumps(warm)[:400]}", flush=True)
    cfg = StereoCfg(W=W, H=H, backend="sgbm")
    sg = SGBMBackend(cfg)
    sg.warmup()
    res = dict(max_disp=a.max_disp, res=a.res, engine=a.engine, iters=a.iters, ckpt=a.ckpt, ready_s=t_ready, warmup=warm, per_tex={})
    for tex in a.tex:
        z = np.load(OUT / "stereo" / f"{tex}_{W}x{H}.npz")
        K = dict(f=float(z["f"]), cx=(W - 1) / 2, cy=(H - 1) / 2, W=W, H=H, B=float(z["B"]))
        B = float(z["B"])
        fs_stats, sg_stats = [], []
        t_fs, t_sg = [], []
        for i in range(len(z["left"])):
            L, R, gt, wall = z["left"][i], z["right"][i], z["gt_depth"][i], z["wall"][i]
            d, ok, ms = cl.depth(L, R, K, B)
            d = np.where((d >= 0.3) & (d <= 5.0), d, np.nan)
            t_fs.append(ms)
            fs_stats.append(stats(d, gt, wall))
            if not a.no_sgbm:
                ds, oks, mss = sg.depth(L, R, K, B)
                t_sg.append(mss)
                sg_stats.append(stats(ds, gt, wall))
        res["per_tex"][tex] = dict(fs=merge(fs_stats), sgbm=merge(sg_stats) if sg_stats else None, fs_ms_frames=float(np.mean(t_fs)),
                                   sgbm_ms=float(np.mean(t_sg)) if t_sg else None)
        print(f"[{tex}] FS ms {np.mean(t_fs):.1f}  SGBM ms {np.mean(t_sg) if t_sg else float('nan'):.1f}", flush=True)
    # sustained timing on one frame (speckle): client wall + server breakdown
    z = np.load(OUT / "stereo" / f"speckle_{W}x{H}.npz")
    K = dict(f=float(z["f"]), cx=(W - 1) / 2, cy=(H - 1) / 2, W=W, H=H, B=float(z["B"]))
    ts, infos = [], []
    for _ in range(a.timing_reps):
        _, _, ms = cl.depth(z["left"][3], z["right"][3], K, float(z["B"]))
        ts.append(ms)
        infos.append(cl.last_info)
    ts = np.array(ts)
    res["timing"] = dict(mean=float(ts.mean()), median=float(np.median(ts)), p95=float(np.percentile(ts, 95)), max=float(ts.max()), hz=float(1e3 / np.median(ts)),
                         server={k: float(np.median([i[k] for i in infos])) for k in ("upload", "infer", "download", "total")})
    res["gpu_mem_mb"] = gpu_mem_mb(cl.server_pid)
    print("timing", json.dumps(res["timing"]), "gpu MB", res["gpu_mem_mb"], flush=True)
    (OUT / "bench").mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.engine}_{a.res}_i{a.iters}"
    (OUT / "bench" / f"stereo_{tag}.json").write_text(json.dumps(res, indent=1))
    cl.close()
    os._exit(0)


if __name__ == "__main__":
    main()
