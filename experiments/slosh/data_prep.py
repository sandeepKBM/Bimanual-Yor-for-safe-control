#!/usr/bin/env python
"""data_prep.py -- rig pickle -> dense arrays + labels + trajectory-level split.

Label definitions (per step t, dt = 5 ms):
  onset      = first step >= t_motion0 with count_spilled >= 1 (-1 if never);
               episodes whose first ball exit is NOT over the rim (tunneling
               artifact of the kinematic carry) are DROPPED.
  y_H(t)     = 1 iff t < onset <= t + H   ("spill within horizon H")
  valid(t)   = t_motion0 <= t < onset (or < T if never spilled): post-onset
               steps are excluded (the event already happened).
  time-to-spill(t) = (onset - t) * dt for spilled episodes.
Split: 70/30 BY TRAJECTORY, stratified by (family, spilled), fixed seed.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slosh_models as sm  # noqa: E402


def load_rig(path, split_seed=0, test_frac=0.3):
    eps = pickle.load(open(path, "rb"))
    n_tun = sum(int(e["tunnel"]) for e in eps)
    eps = [e for e in eps if not e["tunnel"]]
    T = min(len(e["spilled"]) for e in eps)
    pos = np.stack([e["pos"][:T] for e in eps]).astype(float)
    n = sm.quat_axis(np.stack([e["quat"][:T] for e in eps]).astype(float))
    k0 = np.array([e["t_motion0"] for e in eps])
    onset = np.full(len(eps), -1)
    for i, e in enumerate(eps):
        w = np.where(e["spilled"][k0[i]:T] > 0)[0]
        if len(w):
            onset[i] = w[0] + k0[i]
    fam = np.array([e["family"] for e in eps])
    maxout = np.array([int(e["spilled"][:T].max()) for e in eps])
    rng = np.random.default_rng(split_seed)
    test = np.zeros(len(eps), bool)
    for f in np.unique(fam):
        for sp in (False, True):
            idx = np.where((fam == f) & ((onset >= 0) == sp))[0]
            rng.shuffle(idx)
            test[idx[: int(round(test_frac * len(idx)))]] = True
    centroid = np.stack([e["centroid"][:T] for e in eps]).astype(float)
    return dict(pos=pos, n=n, k0=k0, onset=onset, family=fam, test=test, maxout=maxout,
                centroid=centroid, n_tunnel_dropped=n_tun, T=T)


def load_flagship(path):
    """FL-MPC flagship episodes (experiments/wm_requirement collect_ladder format,
    traj_clean.npz: faked grasp, ZERO-VELOCITY kinematic pin (pre-inertial-fix),
    oracle-fixed). Cup axis = logged n_axis."""
    d = np.load(path)
    out = []
    for s in d["seeds"]:
        c = d[f"spilled_curve_{s}"]
        k0 = 320                         # RIGHT_START_T = 1.6 s: balls released here
        w = np.where(c[k0:] > 0)[0]
        out.append(dict(seed=int(s), pos=d[f"cup_pos_{s}"].astype(float), n=d[f"n_axis_{s}"].astype(float),
                        k0=k0, onset=int(w[0] + k0) if len(w) else -1))
    return out


def labels(onset, k0, T, H):
    """y (E,T) and valid mask (E,T) for horizon H steps."""
    E = len(onset); t = np.arange(T)[None]
    on = np.where(onset >= 0, onset, 10**9)[:, None]
    valid = (t >= k0[:, None]) & (t < np.minimum(on, T))
    y = (t < on) & (on <= t + H)
    return y & valid, valid


if __name__ == "__main__":
    D = load_rig(sys.argv[1])
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in D.items()})
    for f in np.unique(D["family"]):
        m = D["family"] == f
        print(f, m.sum(), "spilled", (D["onset"][m] >= 0).sum(), "test", D["test"][m].sum())
