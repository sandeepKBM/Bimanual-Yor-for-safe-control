"""spill_data.py -- read the dense 200 Hz cup kinematics of the spill_v2 shards
(world_model/v2/data/gen_spill_data.py) and replay the liquid proxy on them.

Split rule (deterministic, used by Stage 0 AND Stage 1 so the gate is on the same
held-out episodes): shards sorted by name; shard index % 5 == 4 -> test,
% 5 == 3 -> val, else train.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from numba import njit

from world_model.v2.latent_hj import slosh_pend as sp

DATA = Path("/common/users/ss5772/datasets/bimanual_wm_v2/spill_v2")
DT = 0.005


def shard_split(paths):
    paths = sorted(paths)
    out = dict(train=[], val=[], test=[])
    for i, p in enumerate(paths):
        out["test" if i % 5 == 4 else ("val" if i % 5 == 3 else "train")].append(p)
    return out


def list_shards(data=DATA, complete_only=True):
    ps = sorted(glob.glob(str(Path(data) / "shard_*.npz")))
    return ps


def load_dense(path):
    """-> list of dict(seed, pos (T,3), quat (T,4), axis (T,3), acc_cmd (T,6), vel (T,6))."""
    eps = []
    with np.load(path, allow_pickle=False) as f:
        st, ln, seeds = f["ep_dense_start"], f["ep_dense_len"], f["ep_seed"]
        pos, quat, axis = f["cup_pos_200hz"], f["cup_quat_200hz"], f["cup_axis_200hz"]
        acc_cmd, vel, acc = f["cup_acc_cmd_200hz"], f["cup_vel_200hz"], f["cup_acc_200hz"]
        for s, l, sd in zip(st, ln, seeds):
            sl = slice(int(s), int(s + l))
            eps.append(dict(seed=int(sd), pos=pos[sl].astype(np.float64), quat=quat[sl].astype(np.float64),
                            axis=axis[sl].astype(np.float64), acc_cmd=acc_cmd[sl].astype(np.float64),
                            vel=vel[sl].astype(np.float64), acc=acc[sl].astype(np.float64)))
    return eps


@njit(cache=True)
def _replay_nb(piv, ax, L, c2, dt, sub, G):
    """numba twin of slosh_pend.PendulumObserver (bit-for-bit same update order)."""
    T = piv.shape[0]
    N = np.zeros((T, 3)); ND = np.zeros((T, 3)); PHI = np.zeros(T)
    n = -ax[0].copy(); nd = np.zeros(3)
    h = dt / sub
    g = np.array([0.0, 0.0, -G])
    N[0] = n
    for k in range(1, T):
        if k >= 2:
            a = (piv[k] - 2 * piv[k - 1] + piv[k - 2]) / dt ** 2
        else:
            a = np.zeros(3)
        w = np.cross(ax[k - 1], ax[k]) / dt
        for _ in range(sub):
            f = g - a
            f = f - n * (n @ f)
            ndd = f / L - (nd @ nd) * n - c2 * (nd - np.cross(w, n))
            nd = nd + h * ndd
            n = n + h * nd
            n = n / np.sqrt(n @ n)
            nd = nd - n * (n @ nd)
        N[k] = n; ND[k] = nd
        cs = -(n @ ax[k])
        PHI[k] = np.arccos(min(1.0, max(-1.0, cs)))
    return N, ND, PHI


def replay_episode(ep, P: sp.PendParams, sub=4):
    """Liquid-proxy replay on the realized cup motion (pivot = cup origin + h*axis)."""
    ax = ep["axis"] / np.linalg.norm(ep["axis"], axis=1, keepdims=True)
    piv = ep["pos"] + ax * P.h
    N, ND, PHI = _replay_nb(np.ascontiguousarray(piv), np.ascontiguousarray(ax), P.L, 2 * P.zeta * P.w1, DT, sub, sp.G)
    C = ax
    l = np.clip((P.phi_star - PHI) / P.phi_star, P.m_lo, 1)
    on = np.flatnonzero(PHI >= P.phi_star)
    return dict(n=N, nd=ND, c=C, phi=PHI, l=l, onset=int(on[0]) if len(on) else -1)
