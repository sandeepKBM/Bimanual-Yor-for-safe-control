"""world_model/v2/dinowm/bench.py -- per-tick latency of the official-style DINO-WM
(frozen DINOv2 encoder on 1 new frame + K-step imagination for N candidate plans),
vs the 15 Hz shield budget (66.7 ms/tick). Untrained weights (timing only).

SIMULATION / OFFLINE ONLY.
  CUDA_VISIBLE_DEVICES=7 python -m world_model.v2.dinowm.bench --K 15 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from world_model.v2.dinowm.model import DinoWMSpill, FrozenDino, n_params  # noqa: E402


def timeit(fn, dev, reps=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(dev); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(dev); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return dict(med_ms=ts[len(ts) // 2], p90_ms=ts[int(0.9 * (len(ts) - 1))])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--plans", default="1,16,64")
    ap.add_argument("--configs", default="p64,p256,p16,p64_small")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    res = dict(gpu=torch.cuda.get_device_name(dev), K=a.K, H=a.H, budget_ms=1000 / 15.38, rows=[])
    cfgs = dict(p64=dict(px=112, pool=None, depth=6, mlp=2048),      # native 112 px crop, 8x8 (default)
                p256=dict(px=224, pool=None, depth=6, mlp=2048),     # 224 upsample, 16x16 (~official 14x14=196)
                p16=dict(px=112, pool=4, depth=6, mlp=2048),         # 4x4 pooled tokens
                p64_small=dict(px=112, pool=None, depth=3, mlp=1024))
    for name in a.configs.split(","):
        c = cfgs[name]
        enc = FrozenDino(c["px"], c["pool"]).to(dev)
        wm = DinoWMSpill(enc.n_patches, 42, 6, a.H, depth=c["depth"], mlp_dim=c["mlp"]).to(dev).eval()
        img = torch.randint(0, 255, (1, 112, 112, 3), dtype=torch.uint8, device=dev)
        t_enc = timeit(lambda: enc(img), dev)
        for amp in (False, True):
            for N in map(int, a.plans.split(",")):
                z0 = torch.randn(1, a.H, enc.n_patches, wm.dim, device=dev)
                act = torch.randn(N, a.K, 6, device=dev)

                def imag():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        zs = wm.rollout(z0.expand(N, -1, -1, -1), act)
                        return wm.margin(wm.windows(torch.cat([z0.expand(N, -1, -1, -1), zs], 1)))

                def grad():
                    ac = act.clone().requires_grad_(True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        zs = wm.rollout(z0.expand(N, -1, -1, -1), ac)
                        m = wm.margin(wm.windows(torch.cat([z0.expand(N, -1, -1, -1), zs], 1)))
                    m.min(-1).values.sum().backward()
                    return ac.grad
                try:
                    t_im = timeit(imag, dev); t_gr = timeit(grad, dev, reps=10)
                except torch.OutOfMemoryError:
                    t_im = t_gr = dict(med_ms=float("nan"), p90_ms=float("nan")); torch.cuda.empty_cache()
                row = dict(cfg=name, P=enc.n_patches, depth=c["depth"], params_M=n_params(wm) / 1e6, amp=amp, N=N,
                           enc_ms=t_enc["med_ms"], imagine_ms=t_im["med_ms"], imagine_p90=t_im["p90_ms"],
                           imagine_grad_ms=t_gr["med_ms"], tick_ms=t_enc["med_ms"] + t_im["med_ms"])
                res["rows"].append(row)
                print(f"{name:10s} P={row['P']:3d} depth={row['depth']} amp={int(amp)} N={N:3d}  enc {row['enc_ms']:6.2f} ms"
                      f"  imagine K={a.K} {row['imagine_ms']:7.2f} ms (p90 {row['imagine_p90']:.2f})  +grad {row['imagine_grad_ms']:7.2f} ms"
                      f"  tick {row['tick_ms']:7.2f} ms  {'OK' if row['tick_ms'] < res['budget_ms'] else 'OVER'} 15Hz", flush=True)
        del enc, wm; torch.cuda.empty_cache()
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
