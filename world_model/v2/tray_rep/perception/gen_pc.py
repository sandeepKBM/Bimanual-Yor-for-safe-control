"""gen_pc.py -- sharded, resumable generator of `pc_est` datasets: the SAME episodes as the tray_rep nominal data (same carry law,
seeds, shard split), with the ball part of obs replaced by the point-cloud tracker. SIMULATION ONLY (CPU, no GL, no GPU, no hardware).

It reuses gen_nominal by import (run_episode, merge_split, load_config, shard layout). Seeds: episode seed = seed0 + g*eps_per_shard + j,
exactly as gen_nominal, so with `--config piper` the episodes equal `data_piper_full`: the first 9 obs columns, act, l, n, nd, priv,
lens, onset, seed, hid are identical (checked in selftest_pc.py), and the existing oracle labels (data_piper_full/oracle) index the
same (split, episode, tick) rows.

Output (default <root>/data_<config>_pc_est_<camera>):
    meta.json                         settings (camera, rate, latency, noise overrides, estimator cfg)
    <split>/shard_<g>.npz, snap_<g>.npz, shard_<g>._DONE     as gen_nominal; obs is 16-D; snap_trk = sensor + estimator state
    <split>.npz                       after --merge: readable by tray_rep.store.load_split / wm_audit.common.load_split
    <split>/cloud_<g>.npz             with --save-cloud: pts (ticks,128,3) float16 [tray frame], mask (ticks,128) bool
                                      (row-aligned with shard_<g>.npz; optional, not merged)

  python -m world_model.v2.tray_rep.perception.gen_pc --config piper --camera head_aim --n-eps 3000 --workers 8
  python -m world_model.v2.tray_rep.perception.gen_pc --config piper --camera head_aim --n-eps 3000 --merge
  options: --shard i/n  --save-cloud  --rate-hz  --latency-ticks  --sigma-a  --no-snaps  --seed0
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import gen_nominal as GN
from world_model.v2.bead_rep import util as U
from world_model.v2.tray_rep.perception import pc_env as PC
from world_model.v2.tray_rep.perception import depth_sim as D

SNAP_NOTE = ("snap_trk = [seed, pipeline tick, sensor clock + queue, Kalman state + history, last quality]; restore with "
             "perception.pc_env.PCRepEnv.restore (snapshots from other datasets restore to a steady start)")


def process_shard(args):
    (g, out, camera, a_scale, lip, intensity, eps_per_shard, seed0, snap_stride, snap_min_tick, w_scale, alpha,
     carry_mode, sens, save_cloud, with_snaps) = args
    split = K.shard_split(g)
    d = Path(out) / split
    sent = d / f"shard_{g:04d}._DONE"
    if sent.exists():
        return dict(g=g, split=split, skipped=True)
    t0 = time.perf_counter()
    renv = PC.PCRepEnv(camera, a_scale, lip, w_scale=w_scale, alpha=alpha, rate_hz=sens["rate_hz"],
                       latency_ticks=sens["latency_ticks"], sigma_a=sens["sigma_a"], cam_overrides=sens["cam_overrides"],
                       want_cloud=save_cloud, pack_cloud=False)
    eps, clouds = [], []
    for j in range(eps_per_shard):
        if save_cloud:
            renv.cloud_log = []
        e = GN.run_episode(renv, seed0 + g * eps_per_shard + j, intensity, snap_stride, snap_min_tick,
                           with_snaps=with_snaps, carry_mode=carry_mode)
        eps.append(e)
        if save_cloud:
            T = len(e["obs"])
            clouds.append(renv.cloud_log[:T])
            renv.cloud_log = None
    cat = lambda k: np.concatenate([e[k] for e in eps], 0)
    S = dict(state=[], prev=[], has_prev=[], kin=[], status=[], k=[], w_slew=[], trk=[], ep=[], tick=[])
    for j, e in enumerate(eps):
        for t, sn in e["snaps"]:
            a = sn.arrays()
            for key in ("state", "prev", "has_prev", "kin", "status", "k", "w_slew", "trk"):
                S[key].append(a[key])
            S["ep"].append(j); S["tick"].append(t)
    U.safe_mkdir(d)
    if save_cloud:
        pts = np.concatenate([np.stack([c[0] for c in cl]) for cl in clouds], 0).astype(np.float16)
        msk = np.concatenate([np.stack([c[1] for c in cl]) for cl in clouds], 0)
        U.atomic_savez(d / f"cloud_{g:04d}.npz", pts=pts, mask=msk)
    U.atomic_savez(d / f"shard_{g:04d}.npz", **{k: cat(k) for k in GN.BASE_KEYS},
                   lens=np.array([len(e["obs"]) for e in eps], np.int64),
                   onset=np.array([e["onset"] for e in eps], np.int64), seed=np.array([e["seed"] for e in eps], np.int64),
                   hid=np.array([e["hid"] for e in eps], np.float64), tunnel=np.array([e["tunnel"] for e in eps], np.int8))
    U.atomic_savez(d / f"snap_{g:04d}.npz", snap_state=np.array(S["state"]), snap_prev=np.array(S["prev"]),
                   snap_has_prev=np.array(S["has_prev"], np.int8), snap_kin=np.array(S["kin"]),
                   snap_status=np.array(S["status"], np.int8), snap_k=np.array(S["k"], np.int32),
                   snap_wslew=np.array(S["w_slew"], np.float64).reshape(-1, 3), snap_trk=np.array(S["trk"], np.float64),
                   snap_ep=np.array(S["ep"], np.int32), snap_tick=np.array(S["tick"], np.int32))
    U.write_sentinel(sent)
    wall = time.perf_counter() - t0
    return dict(g=g, split=split, skipped=False, wall_s=wall, n_eps=len(eps), ticks=int(sum(len(e["obs"]) for e in eps)),
                fail_eps=int(sum(e["onset"] >= 0 for e in eps)), tunnel_eps=int(sum(e["tunnel"] for e in eps)),
                frames=renv.cost["frames"], rays=renv.cost["rays"], t_cast=renv.cost["t_cast"], t_fit=renv.cost["t_fit"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="head_aim", choices=list(D.RIGS))
    ap.add_argument("--out", default="")
    ap.add_argument("--config", default="piper", help="name or path of a JSON in tray_rep/configs/ (default piper)")
    ap.add_argument("--n-eps", type=int, default=3000)
    ap.add_argument("--eps-per-shard", type=int, default=GN.EPS_PER_SHARD)
    ap.add_argument("--intensity", type=float, default=None)
    ap.add_argument("--a-scale", type=float, default=None)
    ap.add_argument("--w-scale", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--carry-mode", default=None, choices=list(GN.CARRY_MODES))
    ap.add_argument("--lip", action="store_true")
    ap.add_argument("--rate-hz", type=float, default=None, help="frame rate (default: the camera spec, 30)")
    ap.add_argument("--latency-ticks", type=int, default=None, help="capture->delivery delay in 50 Hz ticks (default: spec, 3)")
    ap.add_argument("--sigma-a", type=float, default=1.5, help="Kalman process noise: ball accel std [m/s^2]")
    ap.add_argument("--save-cloud", action="store_true", help="also write the K=128 compact cloud per tick (cloud_<g>.npz)")
    ap.add_argument("--no-snaps", action="store_true")
    ap.add_argument("--seed0", type=int, default=GN.SEED0)
    ap.add_argument("--snap-stride", type=int, default=GN.SNAP_STRIDE)
    ap.add_argument("--snap-min-tick", type=int, default=GN.SNAP_MIN_TICK)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    cfg = GN.load_config(a.config) if a.config else {}
    pick = lambda v, key, dflt: v if v is not None else cfg.get(key, dflt)
    a.intensity, a.a_scale = pick(a.intensity, "intensity", GN.INTENSITY), pick(a.a_scale, "a_scale", 1.0)
    a.w_scale, a.alpha = pick(a.w_scale, "w_scale", None), pick(a.alpha, "alpha", None)
    a.carry_mode = pick(a.carry_mode, "carry_mode", "orig")
    cname = Path(a.config).stem if a.config else ""
    out = Path(a.out) if a.out else K.ROOT / "perception" / f"data_{cname + '_' if cname else ''}pc_est_{a.camera}"
    n_shards = -(-a.n_eps // a.eps_per_shard)
    if a.merge:
        for sp in K.SPLITS:
            exp = sum(1 for g in range(n_shards) if K.shard_split(g) == sp)
            print("[merge]", GN.merge_split(out, sp, a.allow_partial, exp), flush=True)
        return
    U.safe_mkdir(out)
    meta_path = out / "meta.json"
    sens = dict(rate_hz=a.rate_hz, latency_ticks=a.latency_ticks, sigma_a=a.sigma_a, cam_overrides=None)
    if not meta_path.exists():
        specs = D.rig_specs(a.camera)
        U.jdump(dict(kind="tray_rep pc_est (point-cloud ball tracker)", obs_mode="pc_est", base_obs_mode="full",
                     obs_dim=PC.PC_OBS_DIM, priv_dim=11, feat_dim=10, camera=a.camera, n_eps=a.n_eps,
                     eps_per_shard=a.eps_per_shard, intensity=a.intensity, a_scale=a.a_scale, w_scale=a.w_scale, alpha=a.alpha,
                     carry_mode=a.carry_mode, config=cname, lip=a.lip, seed0=a.seed0, snap_stride=a.snap_stride,
                     snap_min_tick=a.snap_min_tick, dt_wm=GN.DT_WM, rate_hz=a.rate_hz, latency_ticks=a.latency_ticks,
                     rate_hz_eff=min(s.rate_hz for s in specs) if a.rate_hz is None else a.rate_hz,
                     latency_eff=max(s.latency_ticks for s in specs) if a.latency_ticks is None else a.latency_ticks,
                     sigma_a=a.sigma_a, cam_overrides=None, save_cloud=a.save_cloud, split_rule="shard g: g%5==4 test, g%5==3 val, else train",
                     obs_cols="axis(3) v(3) w(3) | ball x y vx vy (Kalman on sphere fits) | n_points/100, rms/4mm, since_good/25",
                     priv_cols="loc(3) vrel(3) omega_tray(3) d lift", snap_note=SNAP_NOTE,
                     same_episodes_as=f"data_{cname}_full" if cname else "data_full",
                     cameras={s.name: dict(fovy=s.fovy, W=s.W, H=s.H, rate_hz=s.rate_hz, latency_ticks=s.latency_ticks,
                                           assumed_depth=s.assumed_depth, note=s.note) for s in specs}), meta_path)
    i, n = U.parse_shard(a.shard)
    mine = [g for g in range(n_shards) if g % n == i]
    print(f"[gen_pc] camera={a.camera} shard {a.shard}: {len(mine)}/{n_shards} shards, out={out}", flush=True)
    tasks = [(g, str(out), a.camera, a.a_scale, a.lip, a.intensity, a.eps_per_shard, a.seed0, a.snap_stride, a.snap_min_tick,
              a.w_scale, a.alpha, a.carry_mode, sens, a.save_cloud, not a.no_snaps) for g in mine]
    t0 = time.perf_counter()
    res = []
    with Pool(a.workers, maxtasksperchild=4) as P:
        for r in P.imap_unordered(process_shard, tasks):
            res.append(r)
            if not r["skipped"]:
                print(f"[gen_pc] shard {r['g']:04d} {r['split']}: {r['n_eps']} eps {r['ticks']} ticks fail {r['fail_eps']}/{r['n_eps']} "
                      f"{r['wall_s']:.1f}s (cast {r['t_cast']:.1f}s fit {r['t_fit']:.1f}s, {r['frames']} frames)", flush=True)
    new = [r for r in res if not r["skipped"]]
    print(f"[gen_pc] done: {len(new)} new, {len(res) - len(new)} skipped, wall {time.perf_counter() - t0:.0f}s, "
          f"cpu-s in shards {sum(r['wall_s'] for r in new):.0f}, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB",
          flush=True)


if __name__ == "__main__":
    main()
