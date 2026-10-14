"""branches.py -- counterfactual branches from logged states, run in MuJoCo (R's branch law, `train_kinwm.make_branches`).
SIMULATION ONLY (MuJoCo CPU, no GPU, no rendering, no hardware, no network).

A branch = 160-tick window: the 60-tick LOGGED prefix (from the nominal dataset) + a 100-tick (2 s) continuation that
restarts from the MuJoCo snapshot of the logged state at tick t0. Output tensors have exactly make_branches' layout:
    obs (N, 160, obs_dim)  act (N, 160, 6)  l (N, 160)  feat (N, 160, 10)
(as numpy float32 in an npz; `load_branches(path, dev)` returns the same dict of torch tensors). Extras for audits:
    priv (N, 160, 11), root_e, root_t, root_failed (the root state had already failed), failed (some l < 0 in the
    continuation), tunnel, hid (the episode's mu_roll), law.

Branch actions (the make_branches law, unchanged except a_scale):
  piecewise constant, segments of 3..15 ticks; each segment 20% hold-still, 40% a fallback-set element, 40% uniform in the
  box (|a_h| <= 5 m/s^2 in a disc, a_z ~ N(0, 0.5), |w_xy| <= 2 rad/s, w_z = 0); one gain per branch, 1.0 with prob 0.4
  else U(0.15, 1); the applied acceleration is  a = gain*u_a - 2*v  (mild velocity feedback, v = tray-origin velocity).
  With `a_scale` != 1 the fallback set and the box are multiplied by a_scale (a_z too); the tilt rates by `w_scale`
  (default a_scale). With a slew limit `alpha` the env slews the commanded tilt rate (see env.py).
  Row layout as in make_branches: obs[60] = the logged state at t0, act[60 + k] = the action of tick t0 + k,
  obs[60 + k + 1] = the state after it; the last state is dropped (100 states, 100 actions).

Root laws (branch point (e, t0); t0 on the snapshot grid, t0 >= 60 so a full 60-tick prefix exists, t0 <= L-2)
  R   episode uniform among episodes with L > 65, then t0 uniform on the grid (== make_branches, restricted to the grid).
      Includes states where the ball has ALREADY left the tray (post-failure roots). Those are wasted.
  Rc  uniform over all not-yet-failed grid states (logged l > 0 and snapshot status OK) of the split.

  python -m world_model.v2.tray_rep.branches --data <root>/data_full --split train --law R  --n 3000 --workers 16 --out <f>.npz
  python -m world_model.v2.tray_rep.branches --data <root>/data_full --split train --law Rc --n 3000 --workers 16 --out <f>.npz
"""
from __future__ import annotations

import argparse
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.bead_rep import util as U

PRE, LB = K.PRE, K.LB


# ============================================================================================ branch action law
def branch_actions(rng, Lb, Uset, a_scale=1.0, p_fallback=0.5, w_scale=None):
    """(Lb, 6) action plan; same law as train_kinwm.make_branches. Uset = the (already scaled) fallback set.
    The box scales a_h and a_z by a_scale and the tilt rates by w_scale (default: a_scale, the phase-1a behaviour)."""
    w_scale = a_scale if w_scale is None else w_scale
    A = np.zeros((Lb, 6))
    gain = 1.0 if rng.random() < 0.4 else rng.uniform(0.15, 1.0)
    k = 0
    while k < Lb:
        L = int(rng.integers(3, 16))
        r_ = rng.random()
        if r_ < 0.2:
            u = np.zeros(6)
        elif r_ < 0.2 + 0.8 * p_fallback:
            u = Uset[rng.integers(len(Uset))].copy()
        else:
            r = 5.0 * a_scale * math.sqrt(rng.random()); th = rng.uniform(0, 2 * math.pi)
            u = np.array([r * math.cos(th), r * math.sin(th), a_scale * rng.normal(0, 0.5),
                          w_scale * rng.uniform(-2.0, 2.0), w_scale * rng.uniform(-2.0, 2.0), 0.0])
        A[k:k + L] = gain * u
        k += L
    return A


# ============================================================================================ root selection
def eligible_roots(st: Store, law: str):
    """-> (E, T) arrays of all candidate roots of the split under `law` (R: every grid tick; Rc: only not-yet-failed)."""
    Es, Ts = [], []
    for ep in range(st.n_eps):
        L = int(st.lens[ep])
        if L <= PRE + 5:
            continue
        t = st.grid_ticks(ep, PRE, L - 2)
        if law == "Rc":
            l = st.rows("l", ep, 0, L)
            t = t[l[t] > 0]
        if len(t):
            Es.append(np.full(len(t), ep)); Ts.append(t)
    return np.concatenate(Es), np.concatenate(Ts)


def pick_roots(st: Store, law: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    if law == "R":
        eps = np.flatnonzero(st.lens > PRE + 5)
        e = rng.choice(eps, n)
        t = np.array([rng.choice(st.grid_ticks(int(i), PRE, int(st.lens[i]) - 2)) for i in e])
        return np.stack([e, t], 1)
    Eall, Tall = eligible_roots(st, law)
    j = rng.choice(len(Eall), n, replace=len(Eall) < n)
    return np.stack([Eall[j], Tall[j]], 1)


# ============================================================================================ run
def run_branch(renv: E.TrayRepEnv, st: Store, e, t0, seed, Lb=LB, pre=PRE):
    snap, hid = st.snapshot_at(e, t0)
    renv.restore(snap, mu_roll=hid)
    rng = np.random.default_rng(seed)
    A = branch_actions(rng, Lb, renv.U, renv.a_scale, w_scale=renv.w_scale)
    obs_b = [st.rows("obs", e, t0, t0 + 1)[0]]
    l_b = [st.rows("l", e, t0, t0 + 1)[0]]
    priv_b = [st.rows("priv", e, t0, t0 + 1)[0]]
    act_b = []
    tunnel = False
    for k in range(Lb):
        a = A[k, :3] - 2.0 * renv.env.v
        r = renv.tick(np.concatenate([a, A[k, 3:]]))
        act_b.append(r["act"]); tunnel |= bool(r["tunnel"])
        if k < Lb - 1:
            obs_b.append(r["obs"]); l_b.append(r["l"]); priv_b.append(r["priv"])
    cat = lambda pre_rows, b: np.concatenate([pre_rows, np.asarray(b, np.float32)], 0)
    obs = cat(st.rows("obs", e, t0 - pre, t0), obs_b)
    act = cat(st.rows("act", e, t0 - pre, t0), act_b)
    l = cat(st.rows("l", e, t0 - pre, t0), l_b)
    priv = cat(st.rows("priv", e, t0 - pre, t0), priv_b)
    lb = l[pre:]
    return dict(obs=obs, act=act, l=l, priv=priv, failed=bool((lb < 0).any()), tunnel=tunnel, hid=hid,
                root_failed=bool(snap.snap.status != 0 or st.rows("l", e, t0, t0 + 1)[0] < 0))


_W = {}


def _work(args):
    data, split, a_scale, tasks = args
    if "st" not in _W or _W["key"] != (data, split):
        _W["st"] = Store(data, split); _W["key"] = (data, split)
        _W["env"] = E.make_env(_W["st"].meta, a_scale=a_scale)
    st, renv = _W["st"], _W["env"]
    return [(i, run_branch(renv, st, e, t0, seed)) for (i, e, t0, seed) in tasks]


def make_branches_mj(data, split, law, n, seed=0, workers=1, a_scale=None, log=print):
    """-> (dict of numpy arrays, meta). Branch i uses the seed (seed*1_000_003 + i), so results do not depend on `workers`."""
    st = Store(data, split)
    a_scale = st.meta.get("a_scale", 1.0) if a_scale is None else a_scale
    t0 = time.perf_counter()
    picks = pick_roots(st, law, n, seed)
    order = np.lexsort((picks[:, 1], picks[:, 0]))                  # group by episode -> shard locality
    tasks = [(int(i), int(picks[i, 0]), int(picks[i, 1]), seed * 1_000_003 + int(i)) for i in order]
    ch = max(1, min(64, len(tasks) // max(1, workers * 4)))
    groups = [(str(data), split, a_scale, tasks[j:j + ch]) for j in range(0, len(tasks), ch)]
    res = [None] * n
    if workers <= 1:
        for g in groups:
            for i, r in _work(g):
                res[i] = r
    else:
        with Pool(workers, maxtasksperchild=4) as P:
            for out in P.imap_unordered(_work, groups):
                for i, r in out:
                    res[i] = r
    st_ = lambda k, dt=np.float32: np.stack([r[k] for r in res]).astype(dt)
    out = dict(obs=st_("obs"), act=st_("act"), l=st_("l"), priv=st_("priv"))
    out["feat"] = E.feats_tray(out["priv"])
    out.update(root_e=picks[:, 0].astype(np.int64), root_t=picks[:, 1].astype(np.int64),
               root_failed=np.array([r["root_failed"] for r in res]), failed=np.array([r["failed"] for r in res]),
               tunnel=np.array([r["tunnel"] for r in res]), hid=np.array([r["hid"] for r in res]))
    lb = out["l"][:, PRE:]
    first = np.array([int(np.argmax(r < 0)) if (r < 0).any() else -1 for r in lb])
    meta = dict(law=law, n=int(n), split=split, seed=seed, a_scale=a_scale, w_scale=st.meta.get("w_scale"),
                alpha=st.meta.get("alpha"), data=str(data),
                fail_tick_median=float(np.median(first[first >= 0])) if (first >= 0).any() else None,
                frac_safe_first25=float(((first < 0) | (first >= 25)).mean()),
                branch_fail_frac=float(out["failed"].mean()), root_failed_frac=float(out["root_failed"].mean()),
                tunnel_frac=float(out["tunnel"].mean()), tick_fail_frac=float((out["l"][:, PRE:] < 0).mean()),
                wall_s=time.perf_counter() - t0, workers=workers)
    return out, meta


def load_branches(path, dev="cpu"):
    """torch dict identical in layout to train_kinwm.make_branches: obs, act, l, feat (+ priv)."""
    import torch
    f = np.load(path)
    return {k: torch.tensor(f[k], dtype=torch.float32, device=dev) for k in ("obs", "act", "l", "feat", "priv")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(K.ROOT / "data_full"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--law", default="R", choices=["R", "Rc"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--a-scale", type=float, default=None, help="default: the dataset's meta value")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, meta = make_branches_mj(a.data, a.split, a.law, a.n, a.seed, a.workers, a.a_scale)
    U.atomic_savez(a.out, **out)
    U.jdump(meta, Path(a.out).with_suffix(".json"))
    print("[branches]", meta, flush=True)


if __name__ == "__main__":
    main()
