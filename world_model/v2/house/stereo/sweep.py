"""stereo/sweep.py -- SGBM parameter / confidence-gate sweep on rendered pairs, per texture mode.  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.sweep --seed 1 --every 6
Renders each pair once (GT depth + self-mask classes), then runs every variant on the SAME pair: fill and error on SCENE pixels (not robot / person) in 0.5..4 m, plus the
person pixels.  Purpose: pick a conservative profile that does not trade holes for wrong depth on textureless surfaces.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from .cfg import StereoCfg
from . import twin as TW
from .sensor import StereoSensor
from .backends import SGBMBackend

VARIANTS = {
    "nopad(old)": dict(pad_left=False),
    "base": dict(),
    "uniq15": dict(uniqueness=15),
    "uniq25": dict(uniqueness=25),
    "bs9_uniq15": dict(block=9, uniqueness=15),
    "uniq15_spk400": dict(uniqueness=15, speckle_win=400, speckle_range=1),
    "tex12": dict(min_texture=12.0),
    "tex16": dict(min_texture=16.0),
    "uniq15_tex12": dict(uniqueness=15, min_texture=12.0),
    "uniq15_tex16": dict(uniqueness=15, min_texture=16.0),
    "3way_uniq15_tex12": dict(mode="3way", uniqueness=15, min_texture=12.0),
}


VARIANTS2 = {
    "hh": dict(),
    "3way": dict(mode="3way"),
    "sgbm5": dict(mode="sgbm"),
    "3way_bs3": dict(mode="3way", block=3),
    "hh_bs9": dict(block=9),
    "3way_bs9": dict(mode="3way", block=9),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants2", action="store_true")
    ap.add_argument("--tex", nargs="+", default=["speckle", "objects", "none"])
    ap.add_argument("--name", default="sweep")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--res", type=int, nargs=2, default=[640, 360])
    a = ap.parse_args()
    from world_model.v2.house.teleop.sim import TeleopSim
    snap = np.load(TW.OUT / "snap" / f"carry_s{a.seed}.npz", allow_pickle=True)
    sim = TeleopSim("h5", grasped=True, seed=a.seed, with_people=True, people_clock="time")
    idx = list(range(0, len(snap["t"]), a.every))
    acc = {}
    V = VARIANTS2 if a.variants2 else VARIANTS
    for tex in a.tex:
        cfg = StereoCfg(W=a.res[0], H=a.res[1], tex=tex, gt=True, keep_pair=True)
        m = TW.build_twin_model(sim, cfg)
        s = StereoSensor(m, cfg)
        s.load(snap["qpos"][0], snap["mpos"][0], snap["mquat"][0], snap["eq"][0])
        s.warmup()
        bes = {k: SGBMBackend(StereoCfg(W=a.res[0], H=a.res[1], **kw)) for k, kw in V.items()}
        for i in idx:
            s.load(snap["qpos"][i], snap["mpos"][i], snap["mquat"][i], snap["eq"][i])
            fr = s.capture(int(snap["k"][i]), float(snap["t"][i]))
            L, R = fr["pair"]
            gt = fr["gt_depth"]
            pers = np.isin(fr["gt_seg"], list(s.person_geoms))
            sc = (fr["cls"] == 0) & ~pers & (gt > 0.5) & (gt < 4.0)
            pe = pers & (gt > 0.5) & (gt < 4.0)
            for k, b in bes.items():
                z, ok, ms = b.depth(L, R, cfg.K(), cfg.baseline)
                e = (z - gt)
                A = acc.setdefault((tex, k), dict(sc=0, scv=0, out=0, big=0, pe=0, pev=0, ms=[], e2=0.0, ne=0, peout=0))
                v = sc & ok
                A["sc"] += int(sc.sum()); A["scv"] += int(v.sum()); A["out"] += int((np.abs(e[v]) > 0.10).sum()); A["big"] += int((np.abs(e[v]) > 0.5).sum())
                A["e2"] += float((np.clip(e[v & (np.abs(e) < 0.1)], -.1, .1) ** 2).sum()); A["ne"] += int((v & (np.abs(e) < 0.1)).sum())
                pv = pe & ok
                A["pe"] += int(pe.sum()); A["pev"] += int(pv.sum()); A["peout"] += int((np.abs(e[pv]) > 0.10).sum())
                A["ms"].append(ms)
        s.close()
    res = {}
    for (tex, k), A in acc.items():
        res.setdefault(tex, {})[k] = dict(scene_fill=A["scv"] / max(A["sc"], 1), scene_outlier_frac_of_valid=A["out"] / max(A["scv"], 1), scene_wrong_gt50cm=A["big"] / max(A["scv"], 1),
                                          scene_inlier_rmse=(A["e2"] / max(A["ne"], 1)) ** 0.5, person_fill=A["pev"] / max(A["pe"], 1), person_outlier=A["peout"] / max(A["pev"], 1),
                                          ms=float(np.mean(A["ms"])))
    (TW.OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    for tex, d in res.items():
        print(tex)
        for k, v in d.items():
            print(f"  {k:20s} fill {v['scene_fill']:.3f} out {v['scene_outlier_frac_of_valid']:.3f} wrong50 {v['scene_wrong_gt50cm']:.3f} rmse_in {v['scene_inlier_rmse']*1e3:.1f}mm | person fill {v['person_fill']:.3f} out {v['person_outlier']:.3f} | {v['ms']:.0f} ms", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
