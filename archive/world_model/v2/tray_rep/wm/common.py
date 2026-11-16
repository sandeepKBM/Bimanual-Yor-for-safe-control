"""wm/common.py -- data access, fixed normalisation and latent filtering for the tray WM port.
SIMULATION / SAVED DATA ONLY (no hardware, no network).

What changed against the pendulum code (wm_audit/common.py, vdirect/trainer.py), nothing else:
  * obs width is a parameter (13 = `full`, 9 = `tray_only`); read from the data, never hard-coded
  * the privileged aux target `feat` is env.feats_tray(priv) (10 numbers), not the pendulum's feats_t
  * the dataset directory is an argument (the pendulum code reads a global default)
  * the normalisation comes from a FIXED reference sample of tray branches (law R, fixed seed), computed once per
    dataset and cached next to it (`<data>/norm/norm_R<n_ref>_s<seed>.json`)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep import store as S
from world_model.v2.bead_rep import util as U

DT_WM = E.DT_WM
M_LO = -0.5            # the tray margin floor: l = clip(d, -0.04, 0.08) / 0.08 in [-0.5, 1]
TILT_LIM = 1.5
N_REF, REF_SEED = 2000, 12345


class TrayP:
    """what reach_value.train reads from `dyn.P` (m_lo = value floor) plus the tilt limit the projection uses."""
    m_lo = M_LO
    tilt_lim = TILT_LIM

    @staticmethod
    def as_dict():
        return dict(m_lo=M_LO, tilt_lim=TILT_LIM, dt=DT_WM)


def load_data(data, split):
    """merged split + the aux target `feat` (10). Keys: obs act l n nd lens start onset seed hid priv tunnel feat."""
    D = S.load_split(split, data)
    D["feat"] = E.feats_tray(D["priv"])
    return D


def d_obs_of(data):
    return int(S.read_meta(data)["obs_dim"])


_TDATA = {}


def torch_data(data, dev):
    """TRAIN / VAL numpy dicts and torch tensors on `dev` (obs act l feat); cached per process."""
    import torch
    key = (str(data), str(dev))
    if key not in _TDATA:
        tr, va = load_data(data, "train"), load_data(data, "val")
        g = lambda D: {k: torch.tensor(D[k], device=dev) for k in ("obs", "act", "l", "feat")}
        _TDATA[key] = (tr, va, g(tr), g(va))
    return _TDATA[key]


def atomic_json(obj, path):
    """write JSON to a temp file, then rename (two jobs may write the same cache file)."""
    import os
    path = Path(path)
    U.safe_mkdir(path.parent)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- fixed normalisation
def norm_path(data, n_ref=N_REF, seed=REF_SEED):
    return Path(data) / "norm" / f"norm_R{n_ref}_s{seed}.json"


def compute_norm(data, n_ref=N_REF, seed=REF_SEED, workers=1, log=print):
    """nominal TRAIN ticks + an equal number of ticks from `n_ref` law-R branches of the TRAIN split (fixed seed).
    Statistics only: these branches are never trained on. Deterministic, so a race between two jobs is harmless."""
    from world_model.v2.tray_rep.branches import make_branches_mj
    tr = load_data(data, "train")
    BR, meta = make_branches_mj(data, "train", "R", n_ref, seed=seed, workers=workers)
    log(f"[norm] reference branches: n={n_ref} fail {meta['branch_fail_frac']:.3f} tick-fail {meta['tick_fail_frac']:.3f} "
        f"wall {meta['wall_s']:.0f}s")
    d = tr["obs"].shape[1]
    bo = BR["obs"].reshape(-1, d); ba = BR["act"].reshape(-1, 6)
    rng = np.random.default_rng(seed)
    nbt = min(len(tr["obs"]), len(bo))
    j = rng.choice(len(bo), nbt, replace=False)
    O = np.concatenate([tr["obs"], bo[j]], 0); A = np.concatenate([tr["act"], ba[j]], 0)
    return dict(o_mu=O.mean(0).tolist(), o_sd=(O.std(0) + 1e-3).tolist(), a_mu=A.mean(0).tolist(),
                a_sd=(A.std(0) + 1e-3).tolist(), n_ref=n_ref, seed=seed, d_obs=int(d),
                n_nominal_ticks=int(len(tr["obs"])), n_branch_ticks=int(nbt), ref_branch_fail_frac=meta["branch_fail_frac"])


def get_norm(data, n_ref=N_REF, seed=REF_SEED, workers=1, log=print):
    p = norm_path(data, n_ref, seed)
    if not p.exists():
        nrm = compute_norm(data, n_ref, seed, workers, log)
        atomic_json(nrm, p)
    return json.loads(p.read_text())


# ----------------------------------------------------------------------------- latent filtering
def filter_fast(wm, obs, act):
    """Same semantics as kin_wm.KinWM.filter (H[:, t] has seen o_<t, a_<t) in one cuDNN GRU call.
    obs / act raw (B, T, .) torch -> (H (B, T, dh), on = normalised obs)."""
    import torch
    on = wm.no(obs)
    x = torch.cat([wm.enc(on), wm.na(act)], -1)
    B = obs.shape[0]
    h0 = wm.h0.expand(B, -1).contiguous()[None]
    c = wm.cell
    out, _ = torch._VF.gru(x, h0, [c.weight_ih, c.weight_hh, c.bias_ih, c.bias_hh], True, 1, 0.0,
                           wm.training, False, True)
    H = torch.cat([h0[0][:, None], out[:, :-1]], 1)
    return H, on


def pad_episodes(D, eps, T=None):
    """pad to the longest episode (last obs repeated). Width = D['obs'].shape[1] (13 or 9)."""
    lens = D["lens"][eps]
    T = int(lens.max()) if T is None else T
    d = D["obs"].shape[1]
    obs = np.zeros((len(eps), T, d), np.float32); act = np.zeros((len(eps), T, 6), np.float32)
    for j, e in enumerate(eps):
        s, L = D["start"][e], min(D["lens"][e], T)
        obs[j, :L] = D["obs"][s:s + L]; act[j, :L] = D["act"][s:s + L]
        obs[j, L:] = obs[j, L - 1]
    return obs, act, lens


def filtered_latents(wm, D, device="cpu", eps=None, chunk=64):
    """-> list per episode of z (L, dh + d_obs) numpy, z_t = [h_t, o_t normalised]."""
    import torch
    eps = np.arange(len(D["lens"])) if eps is None else np.asarray(eps)
    Z = []
    with torch.no_grad():
        for i0 in range(0, len(eps), chunk):
            ii = eps[i0:i0 + chunk]
            obs, act, lens = pad_episodes(D, ii)
            H, on = filter_fast(wm, torch.tensor(obs, device=device), torch.tensor(act, device=device))
            for j, e in enumerate(ii):
                L = D["lens"][e]
                Z.append(torch.cat([H[j, :L], on[j, :L]], -1).cpu().numpy())
    return Z
