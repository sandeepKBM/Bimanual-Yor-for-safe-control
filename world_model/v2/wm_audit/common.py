"""wm_audit/common.py -- shared helpers for the WM audit (SIMULATION / SAVED DATA ONLY).

Conventions are those of world_model/v2/latent_hj/kin_wm.py (50 Hz WM ticks):
    obs o_t = [cup axis c (3), cup-origin lin vel (3), axis ang vel (3)]
    act a_t = [cup-origin lin accel (3), axis ang vel (3)]  (mean over the next 20 ms)
    l_t     = liquid-proxy (pendulum, fill h = 4 cm) margin (phi* - phi)/phi*, clipped [-1, 1]
    n, nd   = privileged pendulum bob direction / rate (never a WM input)

Counterfactual ground truth = the SAME pendulum model the labels were made with,
integrated at 200 Hz (4 sub-steps / 5 ms) under a held cup action; the pivot sits
h above the cup origin on the axis, so pivot accel = a + h * c'' (c'' = w x (w x c)
for a held w).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.latent_hj import slosh_pend as sp  # noqa: E402

SCR = Path("/common/users/ss5772/latent_hj_scratch")
AUD = Path("/common/users/ss5772/bimanual_data/wm_audit")
WMDATA = SCR / "wmdata"
STRIDE = 4
DT_SIM = 0.005
DT_WM = DT_SIM * STRIDE
P0 = sp.PendParams()                  # h=0.04, A_max 5, W_max 2, tilt_lim 1.5 (same U as Stage 0/1)


def load_split(name, data=WMDATA):
    f = np.load(Path(data) / f"{name}.npz")
    lens = f["lens"]
    st = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return dict(obs=f["obs"], act=f["act"], l=f["l"], n=f["n"], nd=f["nd"], lens=lens, start=st,
                onset=f["onset"], seed=f["seed"])


def subset_eps(D, idx):
    """episode subset -> new split dict (contiguous)."""
    idx = np.asarray(idx)
    parts = {k: [] for k in ("obs", "act", "l", "n", "nd")}
    for e in idx:
        s, L = D["start"][e], D["lens"][e]
        for k in parts:
            parts[k].append(D[k][s:s + L])
    out = {k: np.concatenate(v, 0) for k, v in parts.items()}
    out["lens"] = D["lens"][idx].copy()
    out["start"] = np.concatenate([[0], np.cumsum(out["lens"])[:-1]])
    out["onset"] = D["onset"][idx].copy()
    out["seed"] = D["seed"][idx].copy()
    return out


def fallback_U(P=P0):
    return sp.control_set(P)          # (45, 6): 17 horizontal accels x 5 tilt rates


# ----------------------------------------------------------------------------- fast filtering
def filter_fast(wm, obs, act):
    """Same semantics as kin_wm.KinWM.filter (H[:,t] has seen o_<t, a_<t) but one cuDNN GRU
    call using the GRUCell's weights. obs/act raw (B,T,.) torch -> (H (B,T,dh), on)."""
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
    lens = D["lens"][eps]
    T = int(lens.max()) if T is None else T
    obs = np.zeros((len(eps), T, 9), np.float32)
    act = np.zeros((len(eps), T, 6), np.float32)
    for j, e in enumerate(eps):
        s, L = D["start"][e], min(D["lens"][e], T)
        obs[j, :L] = D["obs"][s:s + L]; act[j, :L] = D["act"][s:s + L]
        obs[j, L:] = obs[j, L - 1]
    return obs, act, lens


def filtered_latents(wm, D, device="cpu", eps=None, chunk=64):
    """-> list per episode of (z (L, dh+9) numpy)."""
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


# ----------------------------------------------------------------------------- pendulum GT (torch, batched)
def pend_rollout_torch(n, nd, c, a_seq, w_seq, P=P0, project=True, return_c=False, return_state=False):
    """Ground-truth liquid proxy under a WM-rate action sequence.
    n, nd, c (B,3); a_seq, w_seq (B,K,3) (cup-origin accel, commanded axis ang vel) held per
    20 ms tick. 200 Hz integration, 4 sub-steps (== replay). Returns margins l (B,K) at the
    END of each tick, realized w_eff (B,K,3) (tilt-projected), and optionally c (B,K,3)."""
    import torch
    g = torch.tensor([0.0, 0.0, -sp.G], dtype=n.dtype, device=n.device)
    c2 = 2.0 * P.zeta * P.w1
    hsub = DT_SIM / 4
    ls, weff, cs = [], [], []
    for k in range(a_seq.shape[1]):
        a, w = a_seq[:, k], w_seq[:, k]
        c_start = c
        for _ in range(STRIDE):
            # kinematic cup update for this 5 ms step (with the same tilt projection as the fallback)
            c_new = c + DT_SIM * torch.cross(w, c, dim=-1)
            c_new = c_new / c_new.norm(dim=-1, keepdim=True)
            if project:
                cz = torch.clamp(c_new[..., 2], -1 + 1e-7, 1 - 1e-7)
                over = torch.arccos(cz) > P.tilt_lim
                hxy = c_new[..., :2] / (c_new[..., :2].norm(dim=-1, keepdim=True) + 1e-12)
                cc = torch.cat([hxy * math.sin(P.tilt_lim), torch.full_like(c_new[..., 2:], math.cos(P.tilt_lim))], -1)
                c_new = torch.where(over[..., None], cc, c_new)
            w_real = torch.cross(c, c_new, dim=-1) / DT_SIM
            a_piv = a + P.h * torch.cross(w_real, torch.cross(w_real, c_new, dim=-1), dim=-1)
            for _ in range(4):
                f = g - a_piv
                f = f - n * (n * f).sum(-1, keepdim=True)
                ndd = f / P.L - (nd * nd).sum(-1, keepdim=True) * n - c2 * (nd - torch.cross(w_real, n, dim=-1))
                nd = nd + hsub * ndd
                n = n + hsub * nd
                n = n / n.norm(dim=-1, keepdim=True)
                nd = nd - n * (n * nd).sum(-1, keepdim=True)
            c = c_new
        cs_ = torch.clamp(-(n * c).sum(-1), -1 + 1e-6, 1 - 1e-6)
        phi = torch.arccos(cs_)
        ls.append(torch.clamp((P.phi_star - phi) / P.phi_star, P.m_lo, 1.0))
        weff.append(torch.cross(c_start, c, dim=-1) / DT_WM)
        cs.append(c)
    out = (torch.stack(ls, 1), torch.stack(weff, 1))
    if return_c:
        out = out + (torch.stack(cs, 1),)
    if return_state:
        out = out + (n, nd, c)
    return out


def pend_state_after(n, nd, c, a_seq, w_seq, P=P0, project=True):
    """same as pend_rollout_torch but returns the final (n, nd, c)."""
    import torch
    g = torch.tensor([0.0, 0.0, -sp.G], dtype=n.dtype, device=n.device)
    c2 = 2.0 * P.zeta * P.w1
    hsub = DT_SIM / 4
    for k in range(a_seq.shape[1]):
        a, w = a_seq[:, k], w_seq[:, k]
        for _ in range(STRIDE):
            c_new = c + DT_SIM * torch.cross(w, c, dim=-1)
            c_new = c_new / c_new.norm(dim=-1, keepdim=True)
            if project:
                cz = torch.clamp(c_new[..., 2], -1 + 1e-7, 1 - 1e-7)
                over = torch.arccos(cz) > P.tilt_lim
                hxy = c_new[..., :2] / (c_new[..., :2].norm(dim=-1, keepdim=True) + 1e-12)
                cc = torch.cat([hxy * math.sin(P.tilt_lim), torch.full_like(c_new[..., 2:], math.cos(P.tilt_lim))], -1)
                c_new = torch.where(over[..., None], cc, c_new)
            w_real = torch.cross(c, c_new, dim=-1) / DT_SIM
            a_piv = a + P.h * torch.cross(w_real, torch.cross(w_real, c_new, dim=-1), dim=-1)
            for _ in range(4):
                f = g - a_piv
                f = f - n * (n * f).sum(-1, keepdim=True)
                ndd = f / P.L - (nd * nd).sum(-1, keepdim=True) * n - c2 * (nd - torch.cross(w_real, n, dim=-1))
                nd = nd + hsub * ndd
                n = n + hsub * nd
                n = n / n.norm(dim=-1, keepdim=True)
                nd = nd - n * (n * nd).sum(-1, keepdim=True)
            c = c_new
    return n, nd, c


def wm_rollout(wm, z, a_seq, w_seq):
    """WM open-loop imagination from latent z (B, dh+9) under (a, w) (B,K,3) -> margins (B,K), latents."""
    import torch
    dh = wm.d_h
    h, o = z[:, :dh], z[:, dh:]
    ls, zs = [], []
    for k in range(a_seq.shape[1]):
        h, o = wm.step(h, o, torch.cat([a_seq[:, k], w_seq[:, k]], -1))
        ls.append(wm.margin(h, o)); zs.append(torch.cat([h, o], -1))
    return torch.stack(ls, 1), torch.stack(zs, 1)


def auroc(score, y):
    """AUROC of `score` (higher = more positive) for binary y (Mann-Whitney)."""
    score = np.asarray(score, float); y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_phys_value(path=SCR / "v0" / "value.pt", device="cpu"):
    from world_model.v2.latent_hj import reach_value as rv
    net, meta = rv.load(str(path), device)
    P = sp.PendParams.from_meta(meta["P"])
    dyn = rv.PendDyn(P, device=device)
    return net, dyn, P


def phys_V(net, dyn, n, nd, c):
    import torch
    z = torch.cat([n, nd, c], -1)
    return net(dyn.features(z), dyn.margin(z))
