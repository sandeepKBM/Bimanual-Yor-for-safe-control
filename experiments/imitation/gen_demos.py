#!/usr/bin/env python
"""experiments/imitation/gen_demos.py -- expert demo generation (SIM ONLY).

Runs the flagship scripted-IK expert executed by FL-MPC
(make_cup_collect_safety_demo.run_episode(controller="flmpc"), unmodified,
driven via il_common's in-process hooks) over varied seeds (cup jitter from
the stock init + per-seed cube xy jitter), records per control step
(200 Hz, t >= RIGHT_START_T) observation + task-reference action, and packs
contiguous npz shards.

Per-episode file  <out>/episodes/ep_<seed>.npz:
  obs (T, OBS_DIM) f32         -- il_common.OBS_FIELDS
  q_ref (T, 6) f32             -- right-arm joint reference FL-MPC consumed at step t
  grip_cmd (T,) f32            -- right gripper command (0 closed .. 1 open)
  pinned (T,) bool, pin_idx (T,) i16  -- weld state (diagnostic, not a policy input)
  t_r (T,) f32
  [--images] img_<cam> (T_img, 96, 96, 3) u8 at 20 Hz, img_step (T_img,) index into T
+ ep_<seed>.json (summary metrics).

Shard  <out>/shards/shard_XXX.npz: the SAME keys concatenated over episodes +
ep_start / ep_len / ep_seed / ep_success. (CPU prep; load whole shard to GPU.)

Usage (CPU-only unless --images):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
  python experiments/imitation/gen_demos.py --out <dir> --seeds 0:40 --workers 40 --n-cubes 3
  (--images needs MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<n>)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _one(args):
    seed, out, n_cubes, cup_jitter, images = args
    import il_common as C
    f_npz = Path(out) / "episodes" / f"ep_{seed:05d}.npz"
    if f_npz.exists():
        return json.loads(f_npz.with_suffix(".json").read_text())
    t0 = time.time()
    try:
        s, ep, log, info = C.run(seed, mode="expert", n_cubes=n_cubes, cup_jitter_m=cup_jitter,
                                 images=images, override_meter=True)
    except Exception as e:  # noqa: BLE001
        return dict(seed=seed, error=repr(e))
    r = ep.rec
    arrs = dict(obs=np.stack(r["obs"]).astype(np.float32),
                q_ref=np.stack(r["q_ref"]).astype(np.float32),
                grip_cmd=np.asarray(r["grip_cmd"], np.float32),
                pinned=np.asarray(r["pinned"], bool),
                pin_idx=np.asarray(r["pin_idx"], np.int16),
                t_r=np.asarray(r["t_r"], np.float32))
    if images:
        for c in C.IMG_CAMS:
            arrs[f"img_{c}"] = np.stack(ep.img[c]).astype(np.uint8)
        arrs["img_step"] = np.asarray(ep.img_step, np.int32)
    np.savez(f_npz, **arrs)
    s = {k: v for k, v in s.items() if isinstance(v, (int, float, bool, str, list))}
    s["success"] = bool(s["placed_sel_final"] == s["n_cubes_sel"] and not s["collision"])
    s["wall_s"] = time.time() - t0
    s["T"] = int(arrs["obs"].shape[0])
    f_npz.with_suffix(".json").write_text(json.dumps(s))
    return s


def pack(out: Path, shard_eps: int = 64, only_success: bool = False):
    eps = sorted((out / "episodes").glob("ep_*.npz"))
    (out / "shards").mkdir(exist_ok=True)
    for f in (out / "shards").glob("shard_*.npz"):
        f.unlink()
    metas = [json.loads(f.with_suffix(".json").read_text()) for f in eps]
    keep = [(f, m) for f, m in zip(eps, metas) if (m.get("success") or not only_success)]
    for si in range(0, len(keep), shard_eps):
        chunk = keep[si:si + shard_eps]
        cat, starts, lens, seeds, succ = {}, [], [], [], []
        img_off, img_starts, img_lens = 0, [], []
        n = 0
        for f, m in chunk:
            z = dict(np.load(f))
            T = z["obs"].shape[0]
            if z["obs"].shape[1] == 71:   # pre-task_mask episodes: append the mask from the episode json
                mask = np.array([1.0 if f"cube_{i}" in m["cube_names"] else 0.0 for i in range(6)], np.float32)
                z["obs"] = np.concatenate([z["obs"], np.tile(mask, (T, 1))], axis=1)
            starts.append(n); lens.append(T); seeds.append(m["seed"]); succ.append(m["success"])
            if "img_step" in z:
                z["img_step"] = z["img_step"] + n
                img_starts.append(img_off); img_lens.append(len(z["img_step"]))
                img_off += len(z["img_step"])
            for k, v in z.items():
                cat.setdefault(k, []).append(v)
            n += T
        arr = {k: np.concatenate(v) for k, v in cat.items()}
        arr.update(ep_start=np.array(starts), ep_len=np.array(lens), ep_seed=np.array(seeds),
                   ep_success=np.array(succ))
        if img_starts:
            arr.update(img_ep_start=np.array(img_starts), img_ep_len=np.array(img_lens))
        np.savez(out / "shards" / f"shard_{si // shard_eps:03d}.npz", **arr)
    return len(keep), len(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0:8", help="a:b range")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-cubes", type=int, default=3)
    ap.add_argument("--cup-jitter", type=float, default=0.02)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--pack-only", action="store_true")
    ap.add_argument("--only-success", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    if not a.pack_only:
        lo, hi = map(int, a.seeds.split(":"))
        jobs = [(s, str(out), a.n_cubes, a.cup_jitter, a.images) for s in range(lo, hi)]
        t0 = time.time()
        with Pool(a.workers, maxtasksperchild=3) as pool:
            for s in pool.imap_unordered(_one, jobs):
                keys = ("seed", "success", "placed_sel_final", "n_cubes_sel", "collision",
                        "cup_tilt_max_deg", "spilled_max", "T", "wall_s", "error")
                print(json.dumps({k: s.get(k) for k in keys if k in s}), flush=True)
        print(f"[gen] {len(jobs)} episodes in {time.time() - t0:.0f}s", flush=True)
    kept, tot = pack(out, only_success=a.only_success)
    print(f"[pack] {kept}/{tot} episodes -> {out / 'shards'}", flush=True)


if __name__ == "__main__":
    main()
