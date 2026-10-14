"""oracle.py -- search-certified "savable" labels for (split, episode, tick) states, with MuJoCo as the judge.
SIMULATION ONLY (MuJoCo CPU, no GPU, no rendering, no hardware, no network).

A state is CERTIFIED SAVABLE iff hold-still fails within the horizon (K = 100 ticks = 2 s) AND some sequence in a fixed
library keeps the ball on the tray for the whole horizon. The library is b_tray_ball.probe.lib_sequences, imported
unchanged: each of the 85 fallback-set elements held for `d_hold` ticks (default 10 = 0.2 s), then a brake controller
(stop the tray, level it); plus brake-only. `--d-holds 5,10,25` adds more hold lengths (more certificates, more CPU).
This is an existence proof, not an exhaustive search: the certified set UNDER-approximates the savable set, and an
uncertified state is AMBIGUOUS, never "doomed". Tunnel (fall-through artifact) sequences count as unsafe.

Labels:  0 = hold_ok (hold-still survives the horizon; the rest of the library is not run)
         1 = savable (certified: hold fails, some library sequence survives)
         2 = uncertified (hold fails, nothing in the library survives)
Per state also: hold_ftk (tick of the first failure under hold, -1 if none), brake_ok, n_lib_ok, best_kind (0 none, 1 brake,
2 library element), best_dh, best_i (library element index; 85 = hold-still, 86 = brake-only), best_dmin (the largest
min-over-ticks signed distance [m] among surviving sequences), n_tunnel, d0, l0, hid, wall_s.
`a_scale` scales the fallback set and the brake limits (read from the dataset meta unless --a-scale is given).

Candidate states (TEST by default). `candidates()` draws, on the snapshot grid with logged l > 0:
  near   (`--near-frac`, default 0.75): ticks in [onset-100, onset) of episodes that fail, i.e. the NOMINAL continuation
         fails within the horizon. At most `--cap` per episode (default 8, to spread over episodes).
  random (the rest): any grid state with l > 0, at most 3 per episode (hold-ok and far-from-failure contrast).

  python -m world_model.v2.tray_rep.oracle run --data <root>/data_full --split test --n 2000 --workers 16 --tag test_v0
  python -m world_model.v2.tray_rep.oracle run ... --shard 3/50            (array task; then:)
  python -m world_model.v2.tray_rep.oracle merge --data <root>/data_full --tag test_v0 --n-parts 50
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.bead_rep import util as U
from world_model.v2.testbeds.b_tray_ball.probe import lib_sequences, run_seq   # unchanged library search

K_H = 100
D_HOLD = 10
HOLD_OK, SAVABLE, UNCERT = 0, 1, 2
LABELS = {0: "hold_ok", 1: "savable", 2: "uncertified"}
N_FIELDS = ("label", "hold_ftk", "brake_ok", "n_lib_ok", "best_kind", "best_dh", "best_i", "best_dmin", "n_tunnel",
            "d0", "l0", "hid", "wall_s")


def best_sequence(pos, v, quat, U85, dh, K, a_max, w_max, w0=None, alpha=None):
    """library tensor (n+2, K, 6) for one hold length. With a slew limit `alpha` the sequences are REALISED ones (the
    tilt rate follows the command through the slew limit, starting from the snapshot's rate `w0`); they are run on the
    bare TrayEnv, exactly as authority/scaled_probe.lib_sequences_slew does. alpha None: b's library, unchanged."""
    if alpha is None:
        return lib_sequences(pos, v, quat, U85, dh, K, a_max, w_max)
    from world_model.v2.tray_rep.authority.scaled_probe import lib_sequences_slew
    return lib_sequences_slew(pos, v, quat, U85, dh, K, a_max, w_max, np.zeros(3) if w0 is None else w0, alpha)


def hold_sequence(K, w0=None, alpha=None):
    """hold-still (u = 0) as a (K, 6) sequence; with a slew limit the tilt rate decays from w0 by alpha*dt per tick."""
    z = np.zeros((K, 6))
    if alpha is not None and w0 is not None:
        w = np.asarray(w0, float).copy()
        for t in range(K):
            w = w + np.clip(-w, -alpha * E.DT_WM, alpha * E.DT_WM)
            z[t, 3:] = w
    return z


def certify(renv: E.TrayRepEnv, snap: E.RepSnap, d_holds=(D_HOLD,), K=K_H):
    """-> record dict. `renv` must already have mu_roll set. The env is left in an arbitrary state afterwards."""
    t0 = time.perf_counter()
    env, U85 = renv.env, renv.U
    n = len(U85)
    sn = snap.snap
    pos, v, quat = sn.kin[:3].copy(), sn.kin[7:10].copy(), sn.kin[3:7].copy()
    env.restore(sn)
    d0 = env.dist(); l0 = env.margin()
    rec = dict(label=UNCERT, hold_ftk=-1, brake_ok=0, n_lib_ok=0, best_kind=0, best_dh=-1, best_i=-1, best_dmin=-1e9,
               n_tunnel=0, d0=d0, l0=l0, hid=renv.mu_roll)
    alpha, w0 = renv.alpha, snap.w_slew
    zeros = hold_sequence(K, w0, alpha)
    f, ftk, tun, dmin = run_seq(env, sn, zeros)
    rec["n_tunnel"] += int(tun)
    if f:
        rec["hold_ftk"] = int(ftk)
    if (not f) and (not tun):
        rec.update(label=HOLD_OK, best_kind=0, best_i=n, best_dmin=float(dmin), wall_s=time.perf_counter() - t0)
        return rec
    best = (-1e9, 0, -1, -1)
    n_ok = 0
    for j, dh in enumerate(d_holds):
        seqs = best_sequence(pos, v, quat, U85, dh, K, renv.a_max, renv.w_max, w0, alpha)
        if j == 0:
            f, ftk, tun, dmin = run_seq(env, sn, seqs[n + 1])           # brake only
            rec["n_tunnel"] += int(tun)
            if (not f) and (not tun):
                rec["brake_ok"] = 1
                best = (float(dmin), 1, -1, n + 1)
        for i in range(n):
            f, ftk, tun, dmin = run_seq(env, sn, seqs[i])
            rec["n_tunnel"] += int(tun)
            if (not f) and (not tun):
                n_ok += 1
                if dmin > best[0]:
                    best = (float(dmin), 2, dh, i)
    rec["n_lib_ok"] = n_ok
    if best[1] != 0:
        rec.update(label=SAVABLE, best_dmin=best[0], best_kind=best[1], best_dh=best[2], best_i=best[3])
    rec["wall_s"] = time.perf_counter() - t0
    return rec


# ============================================================================================ candidates
def candidates(st: Store, n, seed=0, near_frac=0.75, cap=8, cap_rand=3):
    """-> (n, 2) int array of (episode, tick) on the snapshot grid, sorted by (episode, tick)."""
    rng = np.random.default_rng(seed)
    near, rand = [], []
    for ep in range(st.n_eps):
        info = st.ep_info(ep)
        L = info["lens"]
        g = st.grid_ticks(ep, st.meta["snap_min_tick"], L - 2)
        l = st.rows("l", ep, 0, L)
        g = g[l[g] > 0]
        if len(g) == 0:
            continue
        if info["onset"] >= 0:
            gn = g[(g >= info["onset"] - K_H) & (g < info["onset"])]
            if len(gn):
                near.append(np.stack([np.full(len(gn), ep), gn], 1))
        rand.append(np.stack([np.full(len(g), ep), g], 1))

    def draw(pool_list, m, cap_):
        if m <= 0 or not pool_list:
            return np.zeros((0, 2), int)
        pool = np.concatenate([p[rng.permutation(len(p))[:cap_]] for p in pool_list], 0)
        return pool[rng.permutation(len(pool))[:m]]
    n_near = int(round(near_frac * n))
    a = draw(near, n_near, cap)
    taken = {tuple(x) for x in a.tolist()}
    b = draw(rand, int(n - len(a)) + 3 * n, cap_rand)                  # oversample, then drop duplicates of `a`
    b = np.array([x for x in b.tolist() if tuple(x) not in taken][:n - len(a)], int).reshape(-1, 2)
    c = np.concatenate([a, b], 0)
    return c[np.lexsort((c[:, 1], c[:, 0]))]


# ============================================================================================ run / merge
_W = {}


def _work(args):
    data, split, a_scale, d_holds, rows = args
    if _W.get("key") != (data, split, a_scale):
        _W["st"] = Store(data, split)
        _W["env"] = E.make_env(_W["st"].meta, a_scale=a_scale)
        _W["key"] = (data, split, a_scale)
    st, renv = _W["st"], _W["env"]
    out = []
    for (i, e, t) in rows:
        snap, hid = st.snapshot_at(e, t)
        renv.set_mu_roll(hid)
        out.append((i, certify(renv, snap, d_holds)))
    return out


def run(data, split, n, seed=0, workers=1, shard="0/1", a_scale=None, d_holds=(D_HOLD,), near_frac=0.75, cap=8, log=print):
    st = Store(data, split)
    a_scale = st.meta.get("a_scale", 1.0) if a_scale is None else a_scale
    cand = candidates(st, n, seed, near_frac, cap)
    si, sn = U.parse_shard(shard)
    idx = np.arange(len(cand))[si::sn]
    rows = [(int(i), int(cand[i, 0]), int(cand[i, 1])) for i in idx]
    ch = 8
    groups = [(str(data), split, a_scale, tuple(d_holds), rows[j:j + ch]) for j in range(0, len(rows), ch)]
    res = {}
    t0 = time.perf_counter()
    if workers <= 1:
        for g in groups:
            for i, r in _work(g):
                res[i] = r
    else:
        with Pool(workers, maxtasksperchild=4) as P:
            for out in P.imap_unordered(_work, groups):
                for i, r in out:
                    res[i] = r
    ids = np.array(sorted(res))
    out = dict(idx=ids, ep=cand[ids, 0], t=cand[ids, 1], split=np.array([K.SPLITS.index(split)] * len(ids), np.int8))
    for f in N_FIELDS:
        out[f] = np.array([res[i][f] for i in ids])
    out["label"] = out["label"].astype(np.int8)
    meta = dict(split=split, n=len(ids), seed=seed, a_scale=a_scale, d_holds=list(d_holds), near_frac=near_frac,
                wall_s=time.perf_counter() - t0, workers=workers, cpu_s=float(out["wall_s"].sum()), **summarize(out))
    return out, meta


def summarize(o):
    lab = o["label"]
    return {f"frac_{LABELS[k]}": float((lab == k).mean()) for k in LABELS} | dict(
        n_states=int(len(lab)), n_savable=int((lab == SAVABLE).sum()), mean_s_per_state=float(o["wall_s"].mean()),
        n_tunnel_states=int((o["n_tunnel"] > 0).sum()))


def load_oracle(path):
    f = np.load(path)
    return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "merge"])
    ap.add_argument("--data", default=str(K.ROOT / "data_full"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--a-scale", type=float, default=None)
    ap.add_argument("--d-holds", default=str(D_HOLD))
    ap.add_argument("--near-frac", type=float, default=0.75)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--tag", default="test_v0")
    ap.add_argument("--n-parts", type=int, default=1)
    a = ap.parse_args()
    odir = Path(a.data) / "oracle"
    if a.cmd == "run":
        out, meta = run(a.data, a.split, a.n, a.seed, a.workers, a.shard, a.a_scale,
                        tuple(int(x) for x in a.d_holds.split(",")), a.near_frac, a.cap)
        si, sn = U.parse_shard(a.shard)
        U.atomic_savez(odir / (f"{a.tag}.npz" if sn == 1 else f"{a.tag}_part{si}of{sn}.npz"), **out)
        U.jdump(meta, odir / (f"{a.tag}.json" if sn == 1 else f"{a.tag}_part{si}of{sn}.json"))
        print("[oracle]", meta, flush=True)
    else:
        parts = [load_oracle(odir / f"{a.tag}_part{i}of{a.n_parts}.npz") for i in range(a.n_parts)]
        out = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
        o = np.argsort(out["idx"]); out = {k: v[o] for k, v in out.items()}
        U.atomic_savez(odir / f"{a.tag}.npz", **out)
        U.jdump(dict(tag=a.tag, **summarize(out)), odir / f"{a.tag}.json")
        print("[merge]", summarize(out), flush=True)


if __name__ == "__main__":
    main()
