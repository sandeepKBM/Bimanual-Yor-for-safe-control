"""bead_rep/gen_nominal.py -- Phase 1 generator: replay the spill_v2 logged cup trajectories through the bead rig
(8 nominal beads) and emit a dataset in the wm_audit.common.load_split format (+ bead extras + sparse snapshots).
SIMULATION / SAVED DATA ONLY (MuJoCo CPU; no GPU, no network, no hardware).

Unit of work = ONE spill_v2 source shard (<= 25 episodes, ~1 min). `--shard i/n` takes every n-th source shard of
the (train, val, test) list; each finished source shard writes
    <out>/<split>/shard_<gidx>.npz     obs act l n nd lens onset onset200 seed + bead extras   (see README)
    <out>/<split>/snap_<gidx>.npz      sparse MuJoCo snapshots at candidate branch points
    <out>/<split>/shard_<gidx>._DONE   sentinel (a shard with a sentinel is skipped on re-run)
`--merge` concatenates the shard files of a split into <out>/<split>.npz (exactly what
wm_audit.common.load_split(<split>, data=<out>) reads).  `--verify` checks the episode mapping and the obs/act
parity against the pendulum wmdata.

  python -m world_model.v2.bead_rep.gen_nominal --shard 0/60                 (generate)
  python -m world_model.v2.bead_rep.gen_nominal --merge                      (after all shards are _DONE)
  python -m world_model.v2.bead_rep.gen_nominal --verify
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.bead_rep import bead_env as B  # noqa: E402
from world_model.v2.bead_rep import util as U  # noqa: E402
from world_model.v2.latent_hj import spill_data as sd  # noqa: E402

STRIDE, DT = B.STRIDE, B.DT
SNAP_STRIDE_TICKS = 5          # DESIGN Q3: snapshot every 5 WM ticks (100 ms) ...
SNAP_MIN_TICK = 20             # ... from tick 20 (closed_loop.py candidate range starts at t=20; branches need >= 60 ticks of
                               # history only for the WM filter, which uses the logged prefix, not the snapshot)
WMDATA = Path("/common/users/ss5772/latent_hj_scratch/wmdata")


# ============================================================================================ obs / act (== kin_wm._episode_arrays)
def episode_obs_act(ep):
    """obs = [axis, cup-origin lin vel (np.gradient of pos), axis ang vel (backward cross)] at every 4th sim step;
    act = mean over the following 4 sim steps of [cup-origin accel (np.gradient of v), ang vel]. Identical formulas to
    kin_wm._episode_arrays (parity is checked by --verify), so a bead episode has the SAME obs/act as its pendulum twin."""
    ax = ep["axis"] / np.linalg.norm(ep["axis"], axis=1, keepdims=True)
    v = np.gradient(ep["pos"], 0.005, axis=0)
    acc = np.gradient(v, 0.005, axis=0)
    w = np.cross(ax[:-1], ax[1:]) / 0.005
    w = np.concatenate([w[:1], w], 0)
    T = (len(ax) // STRIDE) * STRIDE
    idx = np.arange(0, T, STRIDE)
    obs = np.concatenate([ax[idx], v[idx], w[idx]], 1).astype(np.float32)
    act = np.concatenate([acc[:T].reshape(-1, STRIDE, 3).mean(1), w[:T].reshape(-1, STRIDE, 3).mean(1)], 1).astype(np.float32)
    return obs, act, T


# ============================================================================================ replay one episode
def replay_episode(env: B.BeadEnv, ep, snap_stride=SNAP_STRIDE_TICKS, snap_min_tick=SNAP_MIN_TICK):
    """Replay the logged 200 Hz cup pose (pos, quat) with free beads. Returns per-WM-tick bead labels / features and the
    snapshots (taken in the state 'pose k known, not yet pinned', see bead_env)."""
    pos, quat = ep["pos"], ep["quat"]
    Tn = len(pos)
    T4 = (Tn // STRIDE) * STRIDE
    nt = T4 // STRIDE
    vb = np.zeros_like(pos)
    vb[1:] = (pos[1:] - pos[:-1]) / DT          # backward-difference velocity (consistent with the integrator's accel meaning)
    env.reset(pos[0], quat[0])
    n = env.n
    l = np.zeros(nt, np.float32)
    bpos = np.zeros((nt, n, 3), np.float32)
    cen = np.zeros((nt, 3), np.float32)
    n_in = np.zeros(nt, np.int8)
    onset200 = -1
    snaps = []
    for k in range(Tn):
        env.set_kin(pos[k], quat[k], vb[k])
        is_tick = (k % STRIDE == 0) and (k < T4)
        j = k // STRIDE
        if is_tick and j >= snap_min_tick and (j % snap_stride == 0):
            snaps.append((j, env.snapshot()))
        loc = env.phys_step()
        if is_tick:
            l[j] = env.margin(loc)
            bp, c, ni = env.bead_features(loc)
            bpos[j], cen[j], n_in[j] = bp, c, ni
        if onset200 < 0 and env.n_spilled > 0:
            onset200 = k
    nd = np.zeros_like(cen)
    nd[1:] = (cen[1:] - cen[:-1]) / B.DT_WM          # centroid rate in the cup frame (m/s), causal backward difference
    return dict(l=l, bpos=bpos, n=cen, nd=nd, n_in=n_in, onset200=onset200,
                onset=int(onset200 // STRIDE) if onset200 >= 0 else -1,
                tunnel=int(env.tunnel), n_rim=int(env.n_spilled), snaps=snaps, T4=T4)


# ============================================================================================ one source shard
def process_source(env, split, gidx, path, out, limit_eps=None, snap_stride=SNAP_STRIDE_TICKS, snap_min_tick=SNAP_MIN_TICK,
                   log=print):
    t0 = time.perf_counter()
    eps = sd.load_dense(path)
    if limit_eps:
        eps = eps[:limit_eps]
    obs_l, act_l, l_l, n_l, nd_l, bp_l, nin_l = [], [], [], [], [], [], []
    lens, onset, onset200, seed, tunnel, nrim, wall = [], [], [], [], [], [], []
    S = dict(state=[], prev=[], has_prev=[], kin=[], status=[], exit_step=[], k=[], ep=[], tick=[])
    for e, ep in enumerate(eps):
        te = time.perf_counter()
        obs, act, T4 = episode_obs_act(ep)
        r = replay_episode(env, ep, snap_stride, snap_min_tick)
        assert len(obs) == len(r["l"])
        obs_l.append(obs); act_l.append(act); l_l.append(r["l"]); n_l.append(r["n"]); nd_l.append(r["nd"])
        bp_l.append(r["bpos"]); nin_l.append(r["n_in"])
        lens.append(len(obs)); onset.append(r["onset"]); onset200.append(r["onset200"]); seed.append(ep["seed"])
        tunnel.append(r["tunnel"]); nrim.append(r["n_rim"])
        for j, s in r["snaps"]:
            a = s.arrays()
            for key in ("state", "prev", "has_prev", "kin", "status", "exit_step", "k"):
                S[key].append(a[key])
            S["ep"].append(e); S["tick"].append(j)
        wall.append(time.perf_counter() - te)
        log(f"  [{split} shard {gidx:04d}] ep {e} seed {ep['seed']} T={len(ep['pos'])} ticks={len(obs)} "
            f"onset={r['onset']} tunnel={r['tunnel']} rim={r['n_rim']} {wall[-1]:.1f}s ({1e3 * wall[-1] / len(ep['pos']):.2f} ms/step)")
    cat = lambda x: np.concatenate(x, 0)
    d = Path(out) / split
    U.safe_mkdir(d)
    U.atomic_savez(d / f"shard_{gidx:04d}.npz", obs=cat(obs_l), act=cat(act_l), l=cat(l_l), n=cat(n_l), nd=cat(nd_l),
                   lens=np.array(lens, np.int64), onset=np.array(onset, np.int64), onset200=np.array(onset200, np.int64),
                   seed=np.array(seed, np.int64), bpos=cat(bp_l), n_in=cat(nin_l), tunnel=np.array(tunnel, np.int8),
                   n_rim=np.array(nrim, np.int16), src_gidx=np.int32(gidx), partial=np.int8(bool(limit_eps)))
    if S["k"]:
        U.atomic_savez(d / f"snap_{gidx:04d}.npz", snap_state=np.array(S["state"]), snap_prev=np.array(S["prev"]),
                       snap_has_prev=np.array(S["has_prev"], np.int8), snap_kin=np.array(S["kin"]),
                       snap_status=np.array(S["status"], np.int8), snap_exit_step=np.array(S["exit_step"], np.int32),
                       snap_k=np.array(S["k"], np.int32), snap_ep=np.array(S["ep"], np.int32),
                       snap_tick=np.array(S["tick"], np.int32))
    U.write_sentinel(d / f"shard_{gidx:04d}._DONE")
    return dict(split=split, gidx=gidx, n_eps=len(eps), steps=int(sum(len(e["pos"]) for e in eps)), wall_s=time.perf_counter() - t0,
                n_snaps=len(S["k"]), tunnel_eps=int(sum(tunnel)), spill_eps=int(sum(o >= 0 for o in onset)))


# ============================================================================================ merge / verify
BASE_KEYS = ("obs", "act", "l", "n", "nd")
EPS_KEYS = ("lens", "onset", "onset200", "seed", "tunnel", "n_rim")
EXTRA_KEYS = ("bpos", "n_in")


def merge_split(root, split, expect=None, allow_partial=False):
    root = Path(root)
    files = sorted(f for f in (root / split).glob("shard_*.npz") if ".tmp" not in f.name)
    done = [f for f in files if f.with_suffix("._DONE").exists()]
    if expect is not None and len(done) != expect and not allow_partial:
        raise SystemExit(f"[merge] {split}: {len(done)} / {expect} shards done (use --allow-partial to merge anyway)")
    acc = {k: [] for k in BASE_KEYS + EPS_KEYS + EXTRA_KEYS}
    for f in done:
        with np.load(f, allow_pickle=False) as z:
            for k in acc:
                acc[k].append(z[k])
    out = {k: np.concatenate(v, 0) for k, v in acc.items()}
    U.atomic_savez(root / f"{split}.npz", **out)
    return dict(split=split, shards=len(done), eps=int(len(out["lens"])), ticks=int(out["lens"].sum()),
                spill_frac=float((out["onset"] >= 0).mean()), tunnel_frac=float((out["tunnel"] > 0).mean()))


def verify_vs_wmdata(root, split, shards=None, wm=WMDATA, log=print):
    """Episode mapping + obs/act parity against the pendulum wmdata split (matched by episode seed; the pendulum
    build iterates episodes in the same shard order, so for complete splits the order must also match)."""
    root = Path(root)
    f = np.load(Path(wm) / f"{split}.npz")
    seeds_wm, lens_wm = f["seed"], f["lens"]
    start_wm = np.concatenate([[0], np.cumsum(lens_wm)[:-1]])
    pos_of = {int(s): i for i, s in enumerate(seeds_wm)}
    files = sorted(g for g in (root / split).glob("shard_*.npz") if ".tmp" not in g.name)
    ok, n, order_prev, in_order = True, 0, -1, True
    obs_wm, act_wm = f["obs"], f["act"]
    for g in files:
        with np.load(g) as z:
            lens, seed, obs, act = z["lens"], z["seed"], z["obs"], z["act"]
        st = np.concatenate([[0], np.cumsum(lens)[:-1]])
        for e in range(len(lens)):
            i = pos_of.get(int(seed[e]))
            if i is None:
                log(f"  MISSING seed {int(seed[e])} in wmdata/{split}"); ok = False; continue
            if i < order_prev:
                in_order = False
            order_prev = i
            same = (lens[e] == lens_wm[i]
                    and np.array_equal(obs[st[e]:st[e] + lens[e]], obs_wm[start_wm[i]:start_wm[i] + lens[e]])
                    and np.array_equal(act[st[e]:st[e] + lens[e]], act_wm[start_wm[i]:start_wm[i] + lens[e]]))
            ok &= bool(same); n += 1
            if not same:
                log(f"  MISMATCH {split} seed {int(seed[e])}")
    return dict(split=split, episodes_checked=n, all_match=bool(ok), same_order=bool(in_order))


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="0/1", help="i/n: this task takes every n-th source shard")
    ap.add_argument("--out", default=str(U.DATA))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit-eps", type=int, default=0, help="smoke: only the first N episodes of each selected source shard")
    ap.add_argument("--only-gidx", default="", help="comma list of source-shard indices (overrides --shard); smoke use")
    ap.add_argument("--snap-stride", type=int, default=SNAP_STRIDE_TICKS)
    ap.add_argument("--snap-min-tick", type=int, default=SNAP_MIN_TICK)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    splits = a.splits.split(",")
    allsrc = [s for s in U.source_shards() if s[0] in splits]
    out = Path(a.out)

    if a.merge:
        for sp in splits:
            exp = sum(1 for s in allsrc if s[0] == sp)
            print("[merge]", merge_split(out, sp, exp, a.allow_partial), flush=True)
        return
    if a.verify:
        for sp in splits:
            print("[verify]", verify_vs_wmdata(out, sp), flush=True)
        return

    if a.only_gidx:
        want = {int(x) for x in a.only_gidx.split(",")}
        mine = [s for s in allsrc if s[1] in want]
    else:
        i, n = U.parse_shard(a.shard)
        mine = allsrc[i::n]
    U.safe_mkdir(out)
    meta_path = out / "meta.json"
    if not meta_path.exists():
        env0 = B.BeadEnv()
        U.jdump(dict(kind="bead_rep nominal replay", contents=env0.c.to_dict(), n_balls=env0.n, stride=STRIDE, dt_sim=DT,
                     snap_stride_ticks=a.snap_stride, snap_min_tick=a.snap_min_tick, margin=[B.MARGIN_LO, B.MARGIN_HI],
                     rim_tol=B.RIM_TOL, source=str(sd.DATA), split_rule="spill_data.shard_split (sorted name, idx%5: 4 test, 3 val)"),
                meta_path)
    print(f"[gen] shard {a.shard}: {len(mine)} source shards, out={out}, host={os.uname().nodename}", flush=True)
    env = B.BeadEnv()
    t0 = time.perf_counter()
    for split, gidx, path in mine:
        sent = out / split / f"shard_{gidx:04d}._DONE"
        if sent.exists() and not a.limit_eps:
            print(f"[gen] skip {split} shard {gidx:04d} (done)", flush=True)
            continue
        r = process_source(env, split, gidx, path, out, a.limit_eps or None, a.snap_stride, a.snap_min_tick,
                           log=lambda s: print(s, flush=True))
        print(f"[gen] DONE {r} elapsed {time.perf_counter() - t0:.0f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB",
              flush=True)
    print(f"[gen] finished {len(mine)} shards in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
