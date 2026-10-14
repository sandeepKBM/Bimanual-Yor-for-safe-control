"""lib.py -- replay of spill_v2 carries, branch search, helpers for the d_opaque_fill probe. SIMULATION ONLY."""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.testbeds.d_opaque_fill import opaque_env as O   # noqa: E402
from world_model.v2.bead_rep import bead_env as B                  # noqa: E402
from world_model.v2.latent_hj import spill_data as sd              # noqa: E402

STRIDE, DT, DT_WM = O.STRIDE, O.DT, O.DT_WM
T_BR = 100                       # ticks per branch (2 s)
V_CAP = 1.6                      # kinematic-wall tunnel limit (rig.py), as in the bead rig


def load_carries(shard_idx=4, k=12):
    paths = sorted(glob.glob(str(Path(sd.DATA) / "shard_*.npz")))
    eps = sd.load_dense(paths[shard_idx])
    return eps[:k]


def replay(env: "O.OpaqueEnv", ep, snap_stride=5, snap_min_tick=20, log_wrench=False):
    """replay logged 200 Hz pose (pos, quat) with free grains. Returns dict(l (nt,), onset (tick or -1), snaps [(tick, snap)],
    tunnel, n_rim, tilt (nt,), wrench (nt,6) if asked)."""
    pos, quat = ep["pos"], ep["quat"]
    Tn = len(pos)
    T4 = (Tn // STRIDE) * STRIDE
    nt = T4 // STRIDE
    vb = np.zeros_like(pos)
    vb[1:] = (pos[1:] - pos[:-1]) / DT
    env.reset_settled(pos[0], quat[0])
    l = np.zeros(nt, np.float32)
    tilt = np.zeros(nt, np.float32)
    wr = np.zeros((nt, 6), np.float32) if log_wrench else None
    onset200, snaps = -1, []
    accF = np.zeros(6)
    for k in range(Tn):
        env.set_kin(pos[k], quat[k], vb[k])
        is_tick = (k % STRIDE == 0) and (k < T4)
        j = k // STRIDE
        if is_tick and j >= snap_min_tick and (j % snap_stride == 0):
            snaps.append((j, env.snapshot()))
        loc, fm = env.phys_step(log=log_wrench)
        if is_tick:
            l[j] = O.margin(loc, env.status == O.ST_TUNNEL)
            tilt[j] = float(np.arccos(np.clip(env.axis()[2], -1, 1)))
            if log_wrench:
                wr[j] = np.concatenate(fm)
        if onset200 < 0 and env.n_spilled > 0:
            onset200 = k
    return dict(l=l, tilt=tilt, onset=int(onset200 // STRIDE) if onset200 >= 0 else -1, snaps=snaps,
                tunnel=env.tunnel, n_rim=env.n_spilled, wrench=wr, T4=T4)


def run_branch(env, snap, U_seq, early=True):
    """restore snap, apply the held/segment actions (len T_BR array of u), return (spilled?, first-spill tick, tunnel)."""
    env.restore(snap)
    for t in range(len(U_seq)):
        r = env.tick(U_seq[t])
        if r["n_spilled"] > 0:
            return True, t, r["tunnel"]
        if r["tunnel"]:
            return None, t, True
    return False, -1, False


def held(u, T=T_BR):
    return np.tile(u, (T, 1))


def spills_now(env, snap):
    env.restore(snap)
    return env.n_spilled > 0


def savability(env, snap, U, order, T=T_BR):
    """held-constant certificate search.  returns dict(hold_spills, savable, n_tried, saver_idx, hold_t).
    The gentlest-first order is used and the search stops at the first saver (cheap for savable states)."""
    i0 = int(np.argmin(np.abs(U).sum(1)))                      # u = 0 (hold still: zero accel, zero tilt rate)
    hs, ht, tun = run_branch(env, snap, held(U[i0], T))
    out = dict(hold_spills=bool(hs), hold_t=int(ht), savable=False, saver=-1, tried=1, tunnel=bool(tun))
    if hs is not True:
        return out
    for i in order:
        if i == i0:
            continue
        out["tried"] += 1
        s, t, tun = run_branch(env, snap, held(U[i], T))
        if s is False:
            out["savable"], out["saver"] = True, int(i)
            return out
    return out


def gentle_order(U):
    return np.argsort(np.abs(U[:, :3]).sum(1) / 5.0 + np.abs(U[:, 3:]).sum(1) / 2.0, kind="stable")
