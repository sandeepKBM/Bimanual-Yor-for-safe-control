#!/usr/bin/env python
"""smoke_report.py -- label stats + image montage for a gen_spill_data.py output dir.

SIMULATION DATA ONLY (reads npz shards + manifest.json). New file; edits nothing.
  python world_model/v2/data/smoke_report.py --data <out_dir> --png <montage.png>
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np


def phi(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--png", default=None)
    ap.add_argument("--workers", type=int, default=None, help="workers used (for per-core throughput)")
    ap.add_argument("--merge", action="store_true", help="write manifest_merged.json from per-slice manifests")
    a = ap.parse_args()
    d = Path(a.data)
    mans = [json.loads(f.read_text()) for f in sorted(d.glob("manifest*.json")) if f.name != "manifest_merged.json"]
    eps = [e for mm in mans for e in mm["episodes"]]; rej = [r for mm in mans for r in mm["rejected"]]
    print(f"episodes kept {len(eps)}  rejected {len(rej)}  ({len(mans)} manifest(s), wall sum {sum(mm['wall_s'] for mm in mans):.0f}s)")
    if a.merge:
        merged = dict(mans[0]); merged.update(episodes=eps, rejected=rej, n_episodes=len(eps), n_rejected=len(rej),
                                               shards=sorted({x for mm in mans for x in mm["shards"]}))
        (d / "manifest_merged.json").write_text(json.dumps(merged, indent=1))
        print(f"merged -> {d / 'manifest_merged.json'}")
    for r in rej:
        print("  reject:", r.get("mode"), r.get("seed"), r.get("reason"), {k: v for k, v in r.items() if k not in ("mode", "seed", "reason", "err")})
    for mode in sorted({e["mode"] for e in eps}):
        E = [e for e in eps if e["mode"] == mode]
        sp = np.array([e["spill_any"] for e in E]); col = np.array([e["collision_any"] for e in E])
        rc = np.array([e["right_cup_any"] or e["right_ball_any"] for e in E])
        on_m = np.array([e["onset_t_from_motion"] for e in E if e["spill_any"]], float)
        on_r = np.array([e["onset_t_from_rec0"] for e in E if e["spill_any"]], float)
        print(f"\n== mode {mode}: n={len(E)}  spill fraction {sp.mean():.2f}  arm-arm collision fraction {col.mean():.2f}  "
              f"right-arm->cup/ball contact {rc.mean():.2f}")
        if len(on_r):
            print(f"   onset from rec start [s]: min {on_r.min():.2f} med {np.median(on_r):.2f} max {on_r.max():.2f} std {on_r.std():.2f}")
            print(f"   onset from motion start [s]: min {on_m.min():.2f} med {np.median(on_m):.2f} max {on_m.max():.2f} std {on_m.std():.2f}")
            print(f"   onset histogram (from rec start, 0.5 s bins): {np.histogram(on_r, bins=np.arange(0, 6.5, 0.5))[0].tolist()}")
        print(f"   episode-level corr(spill, arm-arm collision) = {phi(sp, col):.3f}   "
              f"corr(spill, right->cup/ball contact) = {phi(sp, rc):.3f}")
        if mode == "rig":
            print(f"   families: " + ", ".join(f"{f}:{sum(e['family']==f for e in E)}" for f in sorted({e['family'] for e in E})))
            print(f"   n_balls: {sorted(e['n_balls'] for e in E)}")
            tr = np.array([e.get("tau_ratio_max", np.nan) for e in E])
            print(f"   ID torque / TAU_MAX: med {np.nanmedian(tr):.2f} max {np.nanmax(tr):.2f}; torque-feasible {np.mean(tr <= 1):.2f}")
            print(f"   cup tracking err med/max [m]: {np.median([e['track_err_med'] for e in E]):.4f} / "
                  f"{np.max([e['track_err_max'] for e in E]):.3f};  tilt cmd/real max med "
                  f"{np.median([e['tilt_cmd_max'] for e in E]):.0f}/{np.median([e['tilt_real_max'] for e in E]):.0f} deg")
        sim = np.array([e["sim_s"] for e in E]); obs = np.array([e["obs_s"] for e in E])
        nf = np.array([e["n_frames"] for e in E])
        print(f"   per-episode cost: sim {sim.mean():.1f}s  obs/render {obs.mean():.1f}s  frames {nf.mean():.0f}  "
              f"-> {60.0 / (sim.mean() + obs.mean()):.1f} episodes/min/core")
    # frame-level correlation within rig episodes, per-step
    shards = sorted(d.glob("shard_*.npz"))
    S = [np.load(s) for s in shards]
    spf = np.concatenate([s["spilled"] for s in S]) > 0
    cof = np.concatenate([s["collision"] for s in S]) > 0
    print(f"\nframe-level: {len(spf)} frames, spilled-frames {spf.mean():.2f}, collision-frames {cof.mean():.3f}, "
          f"corr {phi(spf, cof):.3f}")
    # causality check: spill onsets with ANY arm-arm / right->cup contact in the 0.25 s (50 steps) before onset
    if "collision_200hz" in S[0]:
        near, n_on = 0, 0
        for s in S:
            for e in range(len(s["ep_dense_start"])):
                st, ln = int(s["ep_dense_start"][e]), int(s["ep_dense_len"][e])
                spd = s["spilled_200hz"][st:st + ln]; cl = s["collision_200hz"][st:st + ln]
                on = np.flatnonzero(spd > 0)
                if len(on):
                    n_on += 1; k = on[0]
                    near += int(cl[max(0, k - 50):k + 1].any())
        print(f"spill onsets with a contact in the preceding 0.25 s: {near}/{n_on}")
    rig_eps = [e for e in eps if e["mode"] == "rig"]
    for fam in sorted({e["family"] for e in rig_eps}):
        E = [e for e in rig_eps if e["family"] == fam]
        if len(E) >= 4:
            print(f"   within family {fam:9s} n={len(E):3d}: spill {np.mean([e['spill_any'] for e in E]):.2f} "
                  f"coll {np.mean([e['collision_any'] for e in E]):.2f} corr {phi([e['spill_any'] for e in E], [e['collision_any'] for e in E]):.3f}")
    tot = sum(s.stat().st_size for s in shards)
    print(f"disk: {tot / 1e6:.1f} MB for {len(eps)} episodes ({tot / 1e6 / max(1, len(eps)):.2f} MB/episode)")
    for k in ("img_zed", "proprio", "action", "action_cup", "cup_pose", "time_to_spill"):
        if k in S[0]:
            print(f"   {k}: {S[0][k].shape} {S[0][k].dtype}")
    if a.png and "img_zed" in S[0]:
        from PIL import Image
        rows = []
        picked = 0
        for s in S:
            for e in range(len(s["ep_start"])):
                if picked >= 4:
                    break
                st, ln = int(s["ep_start"][e]), int(s["ep_len"][e])
                idx = np.linspace(st, st + ln - 1, 6).astype(int)
                row = [np.concatenate([s["img_zed"][i], s["img_lwrist"][i], s["img_rwrist"][i]], 0) for i in idx]
                rows.append(np.concatenate(row, 1)); picked += 1
        img = np.concatenate(rows, 0)
        Image.fromarray(img).save(a.png)
        print(f"montage -> {a.png}  ({img.shape[1]}x{img.shape[0]}; per episode: 6 time points x [zed crop / left wrist / right wrist])")


if __name__ == "__main__":
    main()
